import numpy as np
import mujoco as mu
from mujoco.viewer import launch_passive
from gym import Env, spaces
from anti_collision_policy.spot_locomotion import LocomotionSkill
import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
MUJOCO_MODEL = os.path.join(ROOT_DIR, 'boston_dynamics_spot', 'scene_arm.xml')

# Load model
try:
    model = mu.MjModel.from_xml_path(MUJOCO_MODEL)
    print("MuJoCo model loaded successfully.")
except Exception as e:
    print(f"ERROR loading MuJoCo model: {e}")
    exit()


class TestEnv(Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, model=model, target_position=[-3, -3, 0.5], render_mode="human"):
        super().__init__()
        self.model = model
        self.data = mu.MjData(model)
        mu.mj_forward(self.model, self.data)
        self.renderer = None
        self.viewer = None
        self.render_mode = render_mode
        self.camera_name = "main"
        self.target_position = np.array(target_position, dtype=np.float32)
        self.step_count = 0

        self.observation_space = spaces.Dict({
            "depth": spaces.Box(0, 1, (1, 480, 640), np.float32),
            "obstacle_mask": spaces.Box(0, 1, (1, 480, 640), np.float32),
            "state": spaces.Box(-np.inf, np.inf, (7,), np.float32)
        })

        self.action_space = spaces.Box(
            low=np.array([0, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32
        )
        self.observation = None

    def get_robot_state(self):
        body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, "body")
        pos = self.data.xpos[body_id]
        quat = self.data.xquat[body_id]
        return np.concatenate([pos, quat])

    def get_depth(self):
        if self.renderer is None:
            self.renderer = mu.Renderer(self.model, 480, 640)
        self.renderer.update_scene(self.data, camera=self.camera_name)
        self.renderer.enable_depth_rendering()
        depth = self.renderer.render()
        self.renderer.disable_depth_rendering()
        depth = np.nan_to_num(depth, nan=30.0)
        depth = np.clip(depth, 0.2, 30.0)
        depth_norm = (depth - 0.2) / (30.0 - 0.2)
        mask = (depth < 1.5).astype(np.float32)
        return depth_norm[np.newaxis], mask[np.newaxis]

    def obs(self):
        depth, mask = self.get_depth()
        return {"depth": depth, "obstacle_mask": mask, "state": self.get_robot_state()}

    def step(self, action):
        skill_prob, turn_vel, linear_vel = action
        skill_name = "go_straight" if skill_prob > 0.5 else "turn"
        velocity = linear_vel if skill_name == "go_straight" else turn_vel
        locomotion = LocomotionSkill(skill_name)
        locomotion.run(velocity, duration_sec=0.02, state=None)
        self.step_count += 1
        self.observation = self.obs()

        dist = np.linalg.norm(self.get_robot_state()[:2] - self.target_position[:2])
        reward = 1.5 * np.exp(-dist) + 0.5 * np.dot(
            np.array([np.cos(0), np.sin(0)]),
            (self.target_position[:2] - self.get_robot_state()[:2]) / max(dist, 1e-5)
        ) - 0.01
        if np.mean(self.observation["obstacle_mask"]) > 0.05:
            reward -= 5.0 * np.mean(self.observation["obstacle_mask"])
        else:
            reward -= 0.0

        terminated = dist < 0.25
        truncated = self.step_count >= 1000
        if terminated:
            reward += 10.0
        return self.observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mu.mj_resetData(self.model, self.data)
        mu.mj_forward(self.model, self.data)
        self.step_count = 0
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
