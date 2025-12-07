from anti_collision_policy.spot_env import TestEnv
from anti_collision_policy.ppo_agent import SpotEnvWrapper
from anti_collision_policy.neural_network import SpotCNN
import mujoco as mu
import torch
import numpy as np
import os
from pathlib import Path


def test_observation_structure():
    """Test that observation has correct structure with target info"""

    # Create environment
    ROOT_DIR = Path(__file__).resolve().parents[0]
    MUJOCO_MODEL = os.path.join(ROOT_DIR, 'boston_dynamics_spot', 'scene_arm.xml')

    model = mu.MjModel.from_xml_path(MUJOCO_MODEL)
    data = mu.MjData(model)

    env = TestEnv(model=model, data=data, target_position=[12.5, -1.5, 0.5])
    wrapped_env = SpotEnvWrapper(env)

    # Get observation
    obs_dict, _ = env.reset()
    obs_flat, _ = wrapped_env.reset()

    print("\nObservation structure test:")
    print(f"Original obs dict keys: {obs_dict.keys()}")
    print(f"Flattened obs shape: {obs_flat.shape}")
    print("Expected: 24587")

    # Verify components
    # expected_sizes = {
    #     "depth": 12288,
    #     "mask": 12288,
    #     "robot_state": 7,
    #     "target_info": 4
    # }

    actual_sizes = {
        "depth": 96 * 128,
        "mask": 96 * 128,
        "robot_state": 7,
        "target_info": 4
    }

    print("\nComponent sizes:")
    for component, size in actual_sizes.items():
        print(f"  {component}: {size}")

    # Test that target info is included
    robot_pos = obs_dict["state"][:2]
    target_pos = env.target_position[:2]
    distance = np.linalg.norm(target_pos - robot_pos)

    print(f"\nRobot position: {robot_pos}")
    print(f"Target position: {target_pos}")
    print(f"Actual distance: {distance:.3f}")
    print(f"Distance in obs: {obs_flat[-4]:.3f}")  # Should match

    # Test forward pass through CNN
    cnn = SpotCNN(wrapped_env.observation_space, 256)
    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)
    features = cnn(obs_tensor)

    print("\nCNN forward pass successful!")
    print(f"Input shape: {obs_tensor.shape}")
    print(f"Output features shape: {features.shape}")

    env.close()


if __name__ == "__main__":
    test_observation_structure()
