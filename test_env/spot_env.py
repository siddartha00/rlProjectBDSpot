import gym 
from gym import Env, spaces
import numpy as np
import random
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
import mujoco as mu
import mujoco.viewer as m
import numpy as np
from pathlib import Path
from spot_locomotion import LocomotionSkill


ROOT_DIR = Path(__file__).resolve().parents[1]
MUJOCO_MODEL = os.path.join(ROOT_DIR, 'boston_dynamics', 'scene_arm.xml')

try:
    model = mu.MjModlel.from_xml_path(MUJOCO_MODEL)
except Exception as e:
    print(f"Unable to create a Mujoco environment: {e}")
    exit()


class TestEnv(Env):
    def __init__(self, target_position, render_mode = None, model = model):
        ROOT =Path(__file__).resolve().parents[1]
        MODEL_PATH = os.path.join(ROOT, 'boston_dynamics_spot', 'scene_arm.xml')
        super().__init__()
        try:
            self.model = model
            self.data = mu.MjData(self.model)
            self.render_mode = render_mode
        except Exception as e:
            print(f"Unable to access Mujoco environment: {e}")
        self.camera_name = 'main'
        self.step_count = 0
        self.target_position = target_position
        self.dpth_shape = (1, 480, 640)
        self.observation_space = spaces.Dict({
            "dpth" : spaces.Box(low = 0, high = 30, shape = self.dpth_shape, dtype = np.float32),
            "state" : spaces.Box(low = -np.inf, high = np.inf, shape = (7,), dtype = np.float32)
        })
        self.action_space = spaces.Box(low = np.array([0, -1, 0]), high = np.array([1, 1, 1]))
        self.observation = None

    def get_depth(self):
        w, h = 640, 480
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        depth = np.zeros((h, w), dtype=np.float32)
        mu.mjr_render(self.data, self.model, self.camera_name, rgb, depth)
        return depth[np.newaxis, :, :]

    def get_robot_state(self):
        # Example: base position and quaternion
        try:
            body_id = mu.mj_name2id(self.model, mu.mjtObj.mjOBJ_BODY, "body")
            pos = self.data.xpos[body_id]
            quat = self.data.xquat[body_id]
        except Exception as e:
            print(f'Warning! unable to access robot pose: {e}')
            pass
        return np.concatenate([pos, quat])
    
    def obs(self):
        obs = {
            'dpth' : self.get_depth(),
            'state' : self.get_robot_state()
        }
        return obs

    def reset(self, seed = None, options = None):
        mu.mj_resetData(self.model, self.data)
        self.step_count = 0
        obs = self.obs()
        return obs, {}

    def distance_reward(self):
        """Negative Euclidean distance to target (closer = higher reward)."""
        pos = self.get_robot_state()[:3]
        dist = np.linalg.norm(pos[:2] - self.target_position[:2])
        return -dist  # negative distance for shaping

    def obstacle_penalty(self):
        """Penalty if robot is too close to obstacles, based on depth image."""
        depth = self.observation['dpth']
        min_depth = np.min(depth)
        if min_depth < 0.5:  # threshold for obstacle proximity
            return -2.0
        return 0.0

    def live_penalty(self):
        """Small negative reward to encourage efficiency."""
        return -0.01

    def orientation_reward(self):
        """Optional: reward if robot heading roughly towards target."""
        # Simple approximation: angle between robot forward vector and vector to target
        pos = self.get_robot_state()[:3]
        quat = self.get_robot_state()[3:7]  # w,x,y,z
        # convert quaternion to yaw
        w, x, y, z = quat
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
        forward_vec = np.array([np.cos(yaw), np.sin(yaw)])
        to_target = self.target_position[:2] - pos[:2]
        to_target /= np.linalg.norm(to_target)
        alignment = np.dot(forward_vec, to_target)  # cos(angle)
        return alignment  # between -1 and 1


    def step(self, action):
        [skill_prob, turn_vel, linear_vel] = action
        self.step_count += 1
        skill_name = "go_straight" if skill_prob > 0.5 else "turn"
        velocity = linear_vel if skill_name == "go_straight" else turn_vel
        skill = LocomotionSkill(skill_name)
        skill.run(velocity, 0.02, None)
        self.observation = self.obs()
        reward = 1.0