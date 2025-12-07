import numpy as np
import mujoco as mu
from mujoco.viewer import launch_passive
from gymnasium import Env, spaces
from anti_collision_policy.skill_striaght import SkillStraight
from anti_collision_policy.skill_turn import SkillTurn


class TestEnv(Env):
    """
    SB3-ready Test Environment for obstacle avoidance + goal-reaching on Spot+Arm.
    Action space: Box([0, -1, 0], [1, 1, 1])
      - action[0] -> skill switch (<=0.5: go_straight, >0.5: turn)
      - action[1] -> turn command in [-1, 1]
      - action[2] -> walk speed in [0, 1]
    Observation: Dict with depth (1,H,W), obstacle_mask (1,H,W), state (7,), heading_yaw (1,)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, model=None, data=None,
                 target_position=(12.5, -3.0, 0.5),
                 render_mode=None):
        super().__init__()
        assert model is not None and data is not None, "Provide mujoco model and data"

        # core mujoco objects
        self.model = model
        self.data = data
        mu.mj_forward(self.model, self.data)

        # rendering (lazy)
        self.renderer = None
        self.viewer = None
        self.render_mode = render_mode
        if render_mode == "human":
            self.viewer = launch_passive(self.model, self.data)

        # skills (assumed to exist and control via data.ctrl)
        self.skill_turn = SkillTurn(model=self.model, data=self.data)
        self.skill_walk = SkillStraight(model=self.model, data=self.data)

        # target and bookkeeping
        self.camera_name = "main"
        self.target_position = np.array(target_position, dtype=np.float32)
        self.step_count = 0
        self.prev_dist = None
        self.reward_components = {}

        # ----- Spot Arm folding safety -----
        # Joint names from your scene_arm.xml
        self.arm_joint_names = [
            "arm_sh0",   # shoulder yaw
            "arm_sh1",   # shoulder pitch
            "arm_el0",   # elbow 0
            "arm_el1",   # elbow 1
            "arm_wr0",   # wrist 0
            "arm_wr1",   # wrist 1
            "arm_f1x",   # finger
        ]

        # Use 'home' keyframe arm config if available, else fall back to constants
        self.arm_folded = self._init_arm_folded_from_keyframe(default_angles=[
            0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0
        ])

        # cache qpos addresses for arm joints (None if not found)
        self._arm_qpos_addrs = []
        for name in self.arm_joint_names:
            jid = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                self._arm_qpos_addrs.append(None)
            else:
                self._arm_qpos_addrs.append(int(self.model.jnt_qposadr[jid]))

        # ----- observation space -----
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(0.0, 1.0, (1, 480, 640), dtype=np.float32),
            "obstacle_mask": spaces.Box(0.0, 1.0, (1, 480, 640), dtype=np.float32),
            "state": spaces.Box(-np.inf, np.inf, (7,), dtype=np.float32),
            "heading_yaw": spaces.Box(-np.pi, np.pi, (1,), dtype=np.float32),
        })

        # ----- action space (SB3-compatible) -----
        self.action_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0], dtype=np.float32),
            high=np.array([1.0,  1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # storage
        self.observation = None

    # -----------------------
    # Helpers
    # -----------------------
    def _init_arm_folded_from_keyframe(self, default_angles):
        """Try to read arm joint angles from 'home' keyframe; else use defaults."""
        key_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_KEY, "home")
        if key_id == -1:
            return np.array(default_angles, dtype=np.float32)

        home_qpos = self.model.key_qpos[key_id]  # full qpos for freejoint + legs + arm
        angles = []
        for jname in self.arm_joint_names:
            jid = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                # Fallback to provided default if missing
                angles.append(default_angles[len(angles)])
                continue
            addr = int(self.model.jnt_qposadr[jid])
            angles.append(float(home_qpos[addr]))
        return np.array(angles, dtype=np.float32)

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def get_robot_state(self):
        """
        Return [pos(3), quat(4)] as float32 np.array from body named 'body'.
        Matches your XML <body name="body"> with freejoint.
        """
        body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, "body")
        pos = np.array(self.data.xpos[body_id], dtype=np.float32)
        quat = np.array(self.data.xquat[body_id], dtype=np.float32)
        return np.concatenate([pos, quat])

    def get_robot_yaw(self, quat):
        """
        Convert quaternion (w,x,y,z) to yaw.
        Your keyframe uses [w x y z] ordering, so this is consistent.
        """
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z))
        return float(yaw)

    def safe_set_arm_folded(self):
        """
        Safely set arm joints to folded positions using the cached qpos addresses.
        Skips joints not found in the model.
        """
        for addr, val in zip(self._arm_qpos_addrs, self.arm_folded):
            if addr is None:
                continue
            self.data.qpos[addr] = float(val)
            # Optionally zero velocities for those DoFs
            # self.data.qvel[addr] = 0.0

    # -----------------------
    # Perception
    # -----------------------
    def get_depth(self):
        if self.renderer is None:
            try:
                self.renderer = mu.Renderer(self.model, 480, 640)
            except Exception as e:
                raise RuntimeError(
                    "mu.Renderer creation failed. If you run headless, either "
                    "provide an X display or modify the env to use sensors."
                ) from e

        self.renderer.update_scene(self.data, camera=self.camera_name)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        depth = np.nan_to_num(depth, nan=30.0).astype(np.float32)
        depth = np.clip(depth, 0.5, 30.0)
        depth_norm = (depth - 0.5) / (30.0 - 0.5)  # 0..1
        mask = (depth < 1.5).astype(np.float32)
        return depth_norm[np.newaxis, :, :], mask[np.newaxis, :, :]

    # -----------------------
    # Heading / observation
    # -----------------------
    def get_heading_yaw(self, robot_pos, robot_quat):
        robot_xy = robot_pos[:2]
        dir_to_target = self.target_position[:2] - robot_xy
        target_yaw = np.arctan2(dir_to_target[1], dir_to_target[0])
        robot_yaw = self.get_robot_yaw(robot_quat)
        heading_error = self.normalize_angle(target_yaw - robot_yaw)
        return np.array([heading_error], dtype=np.float32)

    def obs(self):
        depth, mask = self.get_depth()
        state = self.get_robot_state()  # 7-vector
        heading = self.get_heading_yaw(state[:3], state[3:])
        return {
            "depth": depth,
            "obstacle_mask": mask,
            "state": state,
            "heading_yaw": heading,
        }

    # -----------------------
    # Reward
    # -----------------------
    def compute_reward(self):
        obs = self.observation
        obstacle_mask = obs["obstacle_mask"]
        coverage = float(np.mean(obstacle_mask))

        obstacle_penalty = -6.0 * (coverage ** 2)

        robot_xy = obs["state"][:2]
        dist = float(np.linalg.norm(robot_xy - self.target_position[:2]))

        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            if progress > 0:
                progress_reward = 4.0 * progress * (0.8 + 0.2 * (1.0 - coverage))
            else:
                progress_reward = -0.5
            self.prev_dist = dist
        else:
            self.prev_dist = dist
            progress_reward = 0.0

        # --- NEW: heading alignment reward ---
        heading_error = float(obs["heading_yaw"][0])
        # Reward facing the goal, scaled down if many obstacles are visible
        heading_weight = 2.0
        heading_reward = heading_weight * np.cos(heading_error) * (1.0 - coverage)
        # -------------------------------------

        if dist < 0.25:
            goal_reward = 100.0
            speed_bonus = max(0, 500 - self.step_count) / 10.0
            goal_reward += speed_bonus
        else:
            goal_reward = 0.0

        time_penalty = -0.01
        clear_bonus = 0.5 if coverage < 0.3 else 0.0

        total_reward = (
            obstacle_penalty
            + progress_reward
            + goal_reward
            + time_penalty
            + clear_bonus
            + heading_reward        # include new term
        )

        self.reward_components = {
            "obstacle": obstacle_penalty,
            "progress": progress_reward,
            "goal": goal_reward,
            "time": time_penalty,
            "clear": clear_bonus,
            "heading": heading_reward,   # log it for debugging
            "total": total_reward,
            "distance": dist,
            "coverage": coverage,
        }

        return float(total_reward)

    # -----------------------
    # Step / Reset / Render
    # -----------------------
    def step(self, action):
        """
        Action: 3-dim continuous array [skill_switch, turn_cmd, walk_speed]
        - skill_switch: float in [0,1] -> <=0.5 go_straight, >0.5 turn
        - turn_cmd: float in [-1,1] -> used by turn skill
        - walk_speed: float in [0,1] -> used by straight skill
        """
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != 3:
            raise ValueError(f"Expected action shape (3,), got {action.shape}")

        skill_switch = float(np.clip(action[0], 0.0, 1.0))
        turn_cmd = float(np.clip(action[1], -1.0, 1.0))
        walk_speed = float(np.clip(action[2], 0.0, 1.0))

        skill_name = "turn" if skill_switch > 0.5 else "go_straight"

        # ensure arm is folded safely before applying controls
        self.safe_set_arm_folded()

        # Execute macro-step (multiple physics steps)
        skill_steps = 20
        for _ in range(skill_steps):
            if skill_name == "go_straight":
                self.skill_walk.run(
                    velocity=walk_speed, data=self.data, model=self.model
                )
            else:
                self.skill_turn.run(
                    velocity=turn_cmd, data=self.data, model=self.model
                )

            mu.mj_step(self.model, self.data)

            if self.viewer is not None:
                self.viewer.sync()

        # update state
        self.observation = self.obs()
        self.step_count += skill_steps

        # compute reward & termination
        reward = self.compute_reward()
        dist = float(
            self.reward_components.get(
                "distance",
                np.linalg.norm(self.observation["state"][:2] - self.target_position[:2]),
            )
        )
        terminated = bool(dist < 0.25)
        truncated = bool(self.step_count >= 50000)

        obstacle_coverage = float(
            self.reward_components.get(
                "coverage", np.mean(self.observation["obstacle_mask"])
            )
        )
        if obstacle_coverage > 0.8:
            truncated = True

        info = {
            "skill": skill_name,
            "velocity": turn_cmd if skill_name == "turn" else walk_speed,
            "distance": float(self.reward_components.get("distance", np.nan)),
            "obstacle_coverage": float(self.reward_components.get("coverage", np.nan)),
            "reward_components": self.reward_components,
        }

        # Gymnasium step API: obs, reward, terminated, truncated, info
        return self.observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mu.mj_resetData(self.model, self.data)

        # Initialize to 'home' keyframe if present
        key_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mu.mj_resetDataKeyframe(self.model, self.data, key_id)

        mu.mj_forward(self.model, self.data)
        self.step_count = 0
        self.prev_dist = None

        # Fold arm once at reset to safe pose
        self.safe_set_arm_folded()
        mu.mj_forward(self.model, self.data)

        self.observation = self.obs()
        return self.observation, {}

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = launch_passive(self.model, self.data)
            return
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mu.Renderer(self.model, 640, 480)
            self.renderer.update_scene(self.data, camera=self.camera_name)
            frame = self.renderer.render()
            return np.asarray(frame)
