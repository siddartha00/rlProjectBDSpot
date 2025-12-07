# import time
# from spot_env_turn import SpotEnv as TurnEnv
from scipy.spatial.transform import Rotation as R
from ppo import PPO
from pathlib import Path
import os
import numpy as np

# Paths to pretrained PPO models
ROOT_PATH = Path(__file__).resolve().parents[1]
MODEL_DIR = os.path.join(ROOT_PATH, 'PPO_preTrained', 'spot_env_walking', 'updated_final_policies')

SKILL_MODEL = os.path.join(MODEL_DIR, 'PPO_spot_env_walking_0_0_turn_4_final.pth')

default_joint_angles = {
    "fl_hx": 0.0,
    "fl_hy": 0.8,
    "fl_kn": -1.5,
    "fr_hx": 0.0,
    "fr_hy": 0.8,
    "fr_kn": -1.5,
    "hl_hx": 0.0,
    "hl_hy": 1.0,
    "hl_kn": -1.5,
    "hr_hx": 0.0,
    "hr_hy": 1.0,
    "hr_kn": -1.5,
}

arm_folded = np.array([
    0.0,    # shoulder yaw
    -3.3,    # shoulder pitch up
    3.3,   # elbow folded inward
    -1.4,    # wrist pitch up
    0.0     # wrist yaw
])


class SkillTurn:
    """
    Generic low-level locomotion skill wrapper for HRL.
    Executes straight walking, turns, etc. using pretrained PPO policies.
    """

    def __init__(self, frame_delay: float = 0.0, data=None, model=None):
        """
        Args:
            env: SpotEnv instance shared with high-level agent
            skill_name: str, one of ["go_straight", "turn"]
            frame_delay: optional sleep between timesteps
        """
        # Use the environment instance (either shared or skill-specific)
        # self.env = TurnEnv(mode='command', model=model, data=data, show_viewer=True)

        self.frame_delay = frame_delay
        self.prev_action = np.zeros(12)
        self.current_cmd = [0.5, 0.0, 0.0]
        self.target_distance = 2
        self.start_pos = None
        self.default_pos = np.array(list(default_joint_angles.values()))

        # Initialize PPO agent
        state_dim = 52
        action_dim = 12
        self.ppo_agent = PPO(
            state_dim, action_dim,
            lr_actor=0.001,
            lr_critic=0.003,
            gamma=0.99,
            K_epochs=20,
            eps_clip=0.2,
            has_continuous_action_space=True,
            action_std_init=0.5
        )

        # Load pretrained weights for this skill
        self.ppo_agent.load(SKILL_MODEL)

    def get_state(self, data):
        """Build EXACTLY the same 52-dim observation as during training"""
        if self.start_pos is None:
            self.start_pos = data.qpos[0:3].copy()
        
        quat = data.qpos[3:7]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        
        # Base velocities (EXACTLY as in training)
        base_lin_vel_world = data.qvel[0:3]
        base_ang_vel_body = data.qvel[3:6]
        base_lin_vel_body = rot.apply(base_lin_vel_world, inverse=True)
        
        # Orientation
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)
        
        # Joint states (EXACTLY as in training)
        q = data.qpos[7:19]
        qdot = data.qvel[6:18]
        final_pos = q - self.default_pos  # OFFSETS from default!
        
        # Commands (EXACT SCALING as in training!)
        cmd = np.asarray(self.current_cmd)
        remaining_distance = self.target_distance - np.linalg.norm(data.qpos[0:2] - self.start_pos[:2])
        
        # Gravity in body frame
        g_world = np.array([0, 0, -9.8])
        g_body = rot.apply(g_world, inverse=True)
        
        # Build EXACT 52-dim observation
        obs = [
            np.array(base_lin_vel_body) * 2,        # (3) *2 scaling
            np.array(base_ang_vel_body) * 0.25,     # (3) *0.25 scaling
            roll,                                   # (1)
            pitch,                                  # (1)
            [cmd[0]*2, cmd[1]*2, cmd[2]*0.25],      # (3) SCALED commands!
            final_pos,                              # (12) OFFSETS!
            np.array(qdot) * 0.05,                  # (12) *0.05 scaling
            self.prev_action,                       # (12)
            remaining_distance * 0.5,               # (1) *0.5 scaling
            data.qpos[2],                           # (1) height
            g_body                                  # (3)
        ]
        
        obs_flat = np.concatenate([np.ravel(x) for x in obs])
        return obs_flat

    def run(self, velocity: float, data, model):
        """
        Execute the skill for a fixed duration.

        Args:
            velocity: float, linear or angular velocity command
            duration_sec: float, how long to run the skill (seconds)
            state: current observation from environment (matches low-level PPO input)

        Returns:
            final_state: environment observation after running the skill
        """
        if self.start_pos is None:
            self.start_pos = data.qpos[0:3].copy()
        self.current_cmd = [0.0, 0.0, velocity]
        current_state = self.get_state(data=data)

        # Get action from low-level PPO

        action_offset = self.ppo_agent.select_action(current_state)
        actual_action = 0.25 * action_offset + self.default_pos
        data.ctrl[:12] = actual_action
        self.prev_action = action_offset

        # Step environment
        # current_state, reward, done, _ = self.env.step(action)
        # print(f"Worker: velocity={velocity}, action={actual_action[:3]}...")  # Debug

        return actual_action
