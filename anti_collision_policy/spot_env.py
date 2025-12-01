from gym import Env, spaces
import numpy as np
import os
import cv2
import mujoco as mu
import mujoco.viewer
from pathlib import Path
from spot_locomotion import LocomotionSkill


ROOT_DIR = Path(__file__).resolve().parents[1]
MUJOCO_MODEL = os.path.join(ROOT_DIR, 'boston_dynamics', 'scene_arm.xml')

# Load model
try:
    model = mu.MjModel.from_xml_path(MUJOCO_MODEL)
except Exception as e:
    print(f"ERROR loading MuJoCo model: {e}")
    exit()


class TestEnv(Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, target_position, render_mode=None, model=model):
        super().__init__()

        self.model = model
        self.data = mu.MjData(model)
        self.render_mode = render_mode
        self.viewer = None
        self.render = None
        self.camera_name = "main"

        self.step_count = 0
        self.target_position = np.array(target_position, dtype=np.float32)

        # Observation format: Depth, Mask, State
        self.observation_space = spaces.Dict({
            "depth": spaces.Box(low=0, high=1, shape=(1, 480, 640), dtype=np.float32),
            "obstacle_mask": spaces.Box(low=0, high=1, shape=(1, 480, 640), dtype=np.float32),
            "state": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        })

        # Action: [choose skill, turn velocity, linear velocity]
        self.action_space = spaces.Box(
            low=np.array([0, -1, 0]),
            high=np.array([1, 1, 1]),
            dtype=np.float32
        )

        self.observation = None

    # ---------------- DEPTH PROCESSING ---------------- #

    def get_depth(self):
        width, height = 640, 480
        rgb = np.zeros((height, width, 3), dtype=np.uint8)
        depth = np.zeros((height, width), dtype=np.float32)

        mu.mjr_render(self.data, self.model, self.camera_name, rgb, depth)

        return self.process_depth(depth)

    def process_depth(self, depth_raw):

        depth = np.nan_to_num(depth_raw, nan=30.0)
        depth = np.clip(depth, 0.2, 30.0)

        # Normalize for NN input: 0 = near, 1 = far
        depth_norm = (depth - 0.2) / (30.0 - 0.2)
        depth_norm = cv2.GaussianBlur(depth_norm, (5, 5), 0)

        # Obstacle detection mask
        obstacle_threshold_meters = 1.5
        mask = (depth < obstacle_threshold_meters).astype(np.float32)

        # Cleanup mask
        kernel1 = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((7, 7), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)

        return depth_norm[np.newaxis], mask[np.newaxis]

    # ---------------- STATE ---------------- #

    def get_robot_state(self):
        try:
            body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, "body")
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]
        except Exception as e:
            print(f"ERROR getting robot state: {e}")
            pos = np.zeros(3)
            quat = np.zeros(4)

        return np.concatenate([pos, quat])

    def obs(self):
        depth, mask = self.get_depth()
        return {
            "depth": depth,
            "obstacle_mask": mask,
            "state": self.get_robot_state()
        }

    # ---------------- REWARD FUNCTIONS ---------------- #

    def distance_reward(self):
        pos = self.get_robot_state()[:2]
        dist = np.linalg.norm(pos - self.target_position[:2])
        return 1.5 * np.exp(-dist)

    def obstacle_penalty(self):
        mask = self.observation["obstacle_mask"]
        obstacle_ratio = np.mean(mask)

        if obstacle_ratio > 0.05:
            return -5.0 * obstacle_ratio

        return 0.0

    def orientation_reward(self):

        pos = self.get_robot_state()[:3]
        quat = self.get_robot_state()[3:7]

        w, x, y, z = quat
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))

        forward = np.array([np.cos(yaw), np.sin(yaw)])
        target_vec = self.target_position[:2] - pos[:2]
        target_vec /= np.linalg.norm(target_vec)

        return 0.5 * np.dot(forward, target_vec)

    def live_penalty(self):
        return -0.01

    # ------------------- RENDER ------------------ #

    def render(self):
        """Handles two modes:
           - human: Real-time interactive viewer
           - rgb_array: returns a frame for ML pipelines
        """

        if self.render_mode == "human":
            # Launch viewer once
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            return  # nothing to return

        if self.render_mode == "rgb_array":
            width, height = 640, 480

            if self.renderer is None:
                self.renderer = mu.Renderer(self.model, width, height)

            self.renderer.update_scene(self.data)
            frame = self.renderer.render()

            return np.asarray(frame)   # (480,640,3) RGB frame

    # ---------------- MAIN STEP ---------------- #

    def step(self, action):

        skill_prob, turn_vel, linear_vel = action
        skill_name = "go_straight" if skill_prob > 0.5 else "turn"
        velocity = linear_vel if skill_name == "go_straight" else turn_vel

        locomotion = LocomotionSkill(skill_name)
        locomotion.run(velocity, duration_sec=0.02, state=None)

        self.step_count += 1
        self.observation = self.obs()

        reward = (
            self.distance_reward() +
            self.orientation_reward() +
            self.obstacle_penalty() +
            self.live_penalty()
        )

        terminated = False
        truncated = False

        dist = np.linalg.norm(self.get_robot_state()[:2] - self.target_position[:2])

        if dist < 0.25:
            reward += 10.0
            terminated = True

        if self.step_count >= 1000:
            truncated = True

        return self.observation, reward, terminated, truncated, {}

    # ---------------- ENV RESET ---------------- #
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mu.mj_resetData(self.model, self.data)
        self.step_count = 0

        self.observation = self.obs()
        return self.observation, {}
