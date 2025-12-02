import time
# import numpy as np
import mujoco as mu
from anti_collision_policy.spot_env import TestEnv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[0]
MUJOCO_MODEL = os.path.join(ROOT_DIR, 'boston_dynamics_spot', 'scene_arm.xml')

# Load model
try:
    model = mu.MjModel.from_xml_path(MUJOCO_MODEL)
    data = mu.MjData(model)
    print("MuJoCo model loaded successfully.")
    print(f"Model timestep: {model.opt.timestep}")  # Usually 0.002
except Exception as e:
    print(f"ERROR loading MuJoCo model: {e}")
    exit()

# Create environment
env = TestEnv(model=model, data=data, target_position=[3, 3, 0], render_mode="human")

# Reset environment
obs, _ = env.reset()
print(f"Initial observation keys: {obs.keys()}")
print(f"Initial position: {obs['state'][:3]}")

# Test specific skill actions instead of random ones
test_actions = [
    [0.7, 0.5, 0.4],   # Turn right at 0.3 rad/s
    [0.3, 0.7, 0.4],  # Turn left at -0.2 rad/s  
    [0.6, 0.4, 0.3],   # Turn slightly right
]

for episode in range(3):  # Test 3 episodes
    print(f"\n=== Episode {episode} ===")
    obs, _ = env.reset()
    
    for step in range(50):  # 50 high-level steps
        # Use test actions instead of random
        if step < len(test_actions):
            action = test_actions[step]
        else:
            action = [0.5, 0.3, 0.0]  # Default: slight forward
        
        print(f"\nHigh-level step {step}: Action = {action}")
        print(f"  Skill: {'turn' if action[0] > 0.5 else 'go_straight'}")
        print(f"  Velocity: {action[2] if action[0] <= 0.5 else action[1]}")
        
        # Execute step
        obs, reward, done, truncated, info = env.step(action)
        
        # Print current state
        print(f"  Position: {obs['state'][:3]}")
        print(f"  Reward: {reward:.3f}")
        print(f"  Controls (first 3): {data.ctrl[:3]}")
        
        # Check termination
        if done or truncated:
            print(f"Episode ended: done={done}, truncated={truncated}")
            break
        
        # Sleep for real-time viewing (adjust as needed)
        time.sleep(0.01)  # 100Hz viewing
    
    if episode < 2:  # Pause between episodes
        print("\nPausing for 2 seconds...")
        time.sleep(2)

# Close viewer
if env.viewer is not None:
    env.viewer.close()
print("\nTest completed!")