import numpy as np
import mujoco as mu
from mujoco.viewer import launch_passive
from gymnasium import Env, spaces


from anti_collision_policy.skill_striaght import SkillStraight
from anti_collision_policy.skill_turn import SkillTurn


class TestEnv(Env):
    """
    SB3-ready Test Environment for obstacle avoidance + goal-reaching on Spot+Arm.
    Action space: Box([-1, -1, -1], [1, 1, 1])
      - a[0] -> skill switch (<=0: go_straight, >0: turn)
      - a[1] -> turn command in [-1, 1]
      - a[2] -> walk speed in [0, 1] (mapped from [-1,1])
    Observation: Dict with depth (1,H,W), obstacle_mask (1,H,W), state (7,),
                 heading_yaw (1,), target_position (3,)
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        model=None,
        data=None,
        target_position=(12.5, -5.0, 0.5),
        render_mode=None,
        max_episode_steps: int = 20000,  # shorter horizon
    ):
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

        # skills
        self.skill_turn = SkillTurn(model=self.model, data=self.data)
        self.skill_walk = SkillStraight(model=self.model, data=self.data)

        # target and bookkeeping
        self.camera_name = "main"
        self.target_position = np.array(target_position, dtype=np.float32)
        self.step_count = 0
        self.prev_dist = None
        self.init_dist = None
        self.reward_components = {}
        self.max_episode_steps = int(max_episode_steps)

        # ----- Spot Arm folding safety -----
        self.arm_joint_names = [
            "arm_sh0",
            "arm_sh1",
            "arm_el0",
            "arm_el1",
            "arm_wr0",
            "arm_wr1",
            "arm_f1x",
        ]

        self.arm_folded = self._init_arm_folded_from_keyframe(default_angles=[
            0.0, -3.14, 3.06, 0.0, 0.0, 0.0, 0.0
        ])
        self.near_collision_steps = 0

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
            "target_position": spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32),
        })

        # ----- symmetric action space for SB3 -----
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([1.0,  1.0,  1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation = None

    # -----------------------
    # Helpers
    # -----------------------
    def _init_arm_folded_from_keyframe(self, default_angles):
        key_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_KEY, "home")
        if key_id == -1:
            return np.array(default_angles, dtype=np.float32)

        home_qpos = self.model.key_qpos[key_id]
        angles = []
        for jname in self.arm_joint_names:
            jid = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_JOINT, jname)
            if jid == -1:
                angles.append(default_angles[len(angles)])
                continue
            addr = int(self.model.jnt_qposadr[jid])
            angles.append(float(home_qpos[addr]))
        return np.array(angles, dtype=np.float32)

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def get_robot_state(self):
        body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, "body")
        pos = np.array(self.data.xpos[body_id], dtype=np.float32)
        quat = np.array(self.data.xquat[body_id], dtype=np.float32)
        return np.concatenate([pos, quat])

    def get_robot_yaw(self, quat):
        w, x, y, z = quat
        yaw = np.arctan2(2.0 * (w * z + x * y),
                         1.0 - 2.0 * (y * y + z * z))
        return float(yaw)

    def safe_set_arm_folded(self):
        for addr, val in zip(self._arm_qpos_addrs, self.arm_folded):
            if addr is None:
                continue
            self.data.qpos[addr] = float(val)

    # -----------------------
    # Perception
    # -----------------------
    def get_depth(self):
        if self.renderer is None:
            self.renderer = mu.Renderer(self.model, 480, 640)

        self.renderer.update_scene(self.data, camera=self.camera_name)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()

        depth = np.nan_to_num(depth, nan=6.0).astype(np.float32)
        depth = np.clip(depth, 0.3, 6.0)
        depth_norm = (depth - 0.3) / (6.0 - 0.3)
        mask = (depth < 1.0).astype(np.float32)  # slightly more conservative
        return depth_norm[np.newaxis, :, :], mask[np.newaxis, :, :]
    
    def get_min_center_depth(self, obs, center_ratio=0.4):
        depth = obs["depth"][0]  # (H, W), already normalized 0..1
        H, W = depth.shape
        w_center = int(W * center_ratio)
        h_center = int(H * center_ratio)
        x0 = (W - w_center) // 2
        y0 = (H - h_center) // 2
        center_patch = depth[y0:y0 + h_center, x0:x0 + w_center]
        # convert back to meters: depth_norm = (d - 0.3) / (6 - 0.3)
        depth_m = center_patch * (6.0 - 0.3) + 0.3
        return float(center_patch.min()), float(depth_m.min())

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
    
    def get_base_velocity_xy(self):
        # body id
        body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, "body")

        # ensure velocity-related fields are up to date
        mu.mj_fwdVelocity(self.model, self.data)

        # Option A: use cvel (6D: ang(3), lin(3) in body frame)
        # cvel shape: (nbody, 6)
        cvel = self.data.cvel[body_id]  # [wx, wy, wz, vx, vy, vz] in body frame
        v_body = np.array(cvel[3:], dtype=np.float32)  # (3,)

        # transform to world frame using body orientation if you want world XY,
        # or just use body-frame x as "forward speed":
        forward_speed = float(v_body[0])  # x-axis in body frame

        return v_body, forward_speed


    def obs(self):
        depth, mask = self.get_depth()
        state = self.get_robot_state()
        heading = self.get_heading_yaw(state[:3], state[3:])
        return {
            "depth": depth,
            "obstacle_mask": mask,
            "state": state,
            "heading_yaw": heading,
            "target_position": self.target_position,
        }

    # -----------------------
    # Collision detection hook
    # -----------------------
    def check_collision(self):
        """
        Simple example: any contact between robot geom group 1 and world/obstacle group 2.
        Adapt to your XML (geom groups or names).
        """
        ncon = self.data.ncon
        for i in range(ncon):
            con = self.data.contact[i]
            g1 = self.model.geom_group[con.geom1]
            g2 = self.model.geom_group[con.geom2]
            # tweak these groups to match robot vs obstacles
            if {g1, g2} == {1, 2}:
                return True
        return False

    # -----------------------
    # Reward
    # -----------------------
    def compute_reward(self, collided: bool, near_collision: bool, depth_min_m: float, forward_speed: float):
        obs = self.observation
        obstacle_mask = obs["obstacle_mask"][0]
        H, W = obstacle_mask.shape
        w3 = W // 3
        center = obstacle_mask[:, w3:2 * w3]
        cov_center = float(np.mean(center))
        coverage = float(np.mean(obstacle_mask))

        robot_xy = obs["state"][:2]
        dist = float(np.linalg.norm(robot_xy - self.target_position[:2]))

        if self.prev_dist is not None:
            progress = self.prev_dist - dist
            progress_reward = 5.0 * progress
        else:
            self.prev_dist = dist
            self.init_dist = dist
            progress_reward = 0.0
        self.prev_dist = dist

        # obstacle coverage penalty (kept moderate)
        obstacle_penalty = -2.0 * (cov_center ** 2)

        v_max = 1.0  # adjust to your robot's nominal top speed
        speed_norm = max(0.0, min(1.0, forward_speed / v_max))

        heading_error = float(obs["heading_yaw"][0])
        base_heading = np.cos(heading_error) - 1.0
        heading_weight = 0.2  # often 0.2–0.5 is enough
        heading_reward = heading_weight * base_heading * speed_norm

        goal_radius = 0.25
        if dist < goal_radius:
            goal_reward = 100.0
            speed_bonus = max(0, self.max_episode_steps - self.step_count) / 100.0
            goal_reward += speed_bonus
        else:
            goal_reward = 0.0

        collision_penalty = -80.0 if collided else 0.0

        # NEW: near-collision shaping: strong penalty increasing as depth decreases
        if near_collision and not collided:
            self.near_collision_steps += 1
        else:
            self.near_collision_steps = 0

        terminated = bool(dist < 0.25 or collided or self.near_collision_steps > 5)

        time_penalty = -0.01
        clear_bonus = 0.00 if coverage < 0.3 else 0.0
        speed_reward = 1.0 * max(forward_speed, 0.0)

        total_reward = (
            progress_reward
            + speed_reward
            + obstacle_penalty
            + heading_reward
            + goal_reward
            + collision_penalty
            + time_penalty
            + clear_bonus
        )

        self.reward_components = {
            "progress": progress_reward,
            "obstacle": obstacle_penalty,
            "heading": heading_reward,
            "goal": goal_reward,
            "collision": collision_penalty,
            "time": time_penalty,
            "clear": clear_bonus,
            "total": total_reward,
            "distance": dist,
            "coverage": coverage,
            "depth_min_m": depth_min_m,
            "forward_speed": forward_speed
        }

        return float(total_reward)

    # -----------------------
    # Step / Reset / Render
    # -----------------------
    def step(self, action):
        """
        Action: 3-dim continuous array in [-1,1]^3
          - a[0] -> skill switch (<=0: go_straight, >0: turn)
          - a[1] -> turn_cmd in [-1,1]
          - a[2] -> walk_speed in [0,1]
        """
        action = np.asarray(action, dtype=np.float32).ravel()
        if action.shape[0] != 3:
            raise ValueError(f"Expected action shape (3,), got {action.shape}")

        # map to internal params
        switch_raw = float(np.clip(action[0], -1.0, 1.0))
        turn_cmd = float(np.clip(action[1], -1.0, 1.0))
        walk_speed = float(0.5 * (np.clip(action[2], -1.0, 1.0) + 1.0))  # [-1,1] -> [0,1]

        skill_name = "turn" if switch_raw > 0.0 else "go_straight"

        # ensure arm is folded safely before applying controls
        self.safe_set_arm_folded()

        collided = False

        # Execute macro-step (reduced)
        skill_steps = 2
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

            # collision check inside macro-step
            if self.check_collision():
                collided = True
                break

            if self.viewer is not None:
                self.viewer.sync()

        # update state
        self.observation = self.obs()
        self.step_count += skill_steps

        depth_min_norm, depth_min_m = self.get_min_center_depth(self.observation)
        near_collision = depth_min_m < 0.5  # e.g., 0.5 m safety radius
        v_body, forward_speed = self.get_base_velocity_xy()
        self.last_forward_speed = forward_speed

        # compute reward & termination
        reward = self.compute_reward(
            collided=collided,
            near_collision=near_collision,
            depth_min_m=depth_min_m,
            forward_speed=forward_speed,
        )
        dist = float(self.reward_components.get("distance", np.nan))

        terminated = bool(dist < 0.25 or collided)
        # optionally also terminate on persistent near-collision:
        if near_collision and not collided:
            terminated = True

        truncated = bool(self.step_count >= self.max_episode_steps)

        # also truncate if camera is almost fully blocked
        obstacle_coverage = float(self.reward_components.get("coverage", np.nan))
        if obstacle_coverage > 0.9:
            truncated = True

        info = {
            "skill": skill_name,
            "velocity": turn_cmd if skill_name == "turn" else walk_speed,
            "distance": dist,
            "obstacle_coverage": obstacle_coverage,
            "reward_components": self.reward_components,
            "collided": collided,
        }

        return self.observation, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mu.mj_resetData(self.model, self.data)

        key_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_KEY, "home")
        if key_id != -1:
            mu.mj_resetDataKeyframe(self.model, self.data, key_id)

        mu.mj_forward(self.model, self.data)
        self.step_count = 0
        self.prev_dist = None
        self.near_collision_steps = 0

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
