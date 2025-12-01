import time
from ..spot_env import SpotEnv as StraightEnv
from ..spot_env_turn import SpotEnv as TurnEnv
from scipy.spatial.transform import Rotation as R
from ppo import PPO
from pathlib import Path
import os
import numpy as np

# Paths to pretrained PPO models
ROOT_PATH = Path(__file__).resolve().parents[1]
MODEL_DIR = os.path.join(ROOT_PATH, 'PPO_preTrained', 'spot_env_walking', 'updated_final_policies')

SKILL_MODELS = {
    "go_straight": os.path.join(MODEL_DIR, 'PPO_spot_env_walking_0_0_straight_2.pth'),
    "turn": os.path.join(MODEL_DIR, 'PPO_spot_env_walking_turn_4_final.pth')
    }

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


class LocomotionSkill:
    """
    Generic low-level locomotion skill wrapper for HRL.
    Executes straight walking, turns, etc. using pretrained PPO policies.
    """

    def __init__(self, skill_name: str, frame_delay: float = 0.0, model=None, data=None):
        """
        Args:
            env: SpotEnv instance shared with high-level agent
            skill_name: str, one of ["go_straight", "turn"]
            frame_delay: optional sleep between timesteps
        """
        if skill_name not in SKILL_MODELS:
            raise ValueError(f"Unknown skill '{skill_name}'. Available: {list(SKILL_MODELS.keys())}")

        # Use the environment instance (either shared or skill-specific)
        if skill_name == "go_straight":
            self.env = StraightEnv(mode='command', model=model, data=data)
        elif skill_name == "turn":
            self.env = TurnEnv(mode='command', model=model, data=data)
        self.frame_delay = frame_delay
        self.skill_name = skill_name

        # Initialize PPO agent
        state_dim = self.env.obs_dims()
        action_dim = self.env.action_dims()
        self.ppo_agent = PPO(
            state_dim, action_dim,
            lr_actor=0.001,
            lr_critic=0.003,
            gamma=0.99,
            K_epochs=20,
            eps_clip=0.2,
            has_continuous_action_space=True,
            action_std=0.5
        )

        # Load pretrained weights for this skill
        self.ppo_agent.load(SKILL_MODELS[skill_name])

    def get_state(self):
        """Build observation vector directly from the shared environment"""
        env = self.env
        quat = env.data.qpos[3:7]  # [w, x, y, z]
        rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])

        # Base velocities
        base_lin_vel_world = env.data.qvel[0:3]
        base_ang_vel_body = env.data.qvel[3:6]
        base_lin_vel_body = rot.apply(base_lin_vel_world, inverse=True)

        # Orientation
        roll, pitch, yaw = rot.as_euler('xyz', degrees=False)

        # Joint states
        q = env.data.qpos[7:19]
        qdot = env.data.qvel[6:18]
        default_pos = np.array(list(default_joint_angles.values()))
        final_pos = q - default_pos

        # Commands
        cmd = np.asarray(env.current_cmd)
        remaining_distance = env.target_distance - np.linalg.norm(env.data.qpos[0:2] - env.start_pos[:2])

        # Gravity in body frame
        g_world = np.array([0, 0, -9.8])
        g_body = rot.apply(g_world, inverse=True)

        # Build flattened observation
        obs = [
            np.array(base_lin_vel_body) * 2,
            np.array(base_ang_vel_body) * 0.25,
            roll,
            pitch,
            [cmd[0]*2, cmd[1]*2, cmd[2]*0.25],
            final_pos,
            np.array(qdot) * 0.05,
            env.prev_action,
            remaining_distance * 0.5,
            env.data.qpos[2],
            g_body
        ]
        obs_flat = np.concatenate([np.ravel(x) if hasattr(x, "__len__") else [x] for x in obs])
        return obs_flat

    def run(self, velocity: float, duration_sec: float, state):
        """
        Execute the skill for a fixed duration.

        Args:
            velocity: float, linear or angular velocity command
            duration_sec: float, how long to run the skill (seconds)
            state: current observation from environment (matches low-level PPO input)

        Returns:
            final_state: environment observation after running the skill
        """
        current_state = state if state is not None else self.get_state()
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            # Send velocity command to env
            self.env.give_vel_command(velocity)

            # Get action from low-level PPO
            action = self.ppo_agent.select_action(current_state)

            # Step environment
            current_state, reward, done, _ = self.env.step(action)

            # Optional delay for real-time simulation
            if self.frame_delay > 0:
                time.sleep(self.frame_delay)

            # Stop if low-level episode ends
            if done:
                break

        return current_state
