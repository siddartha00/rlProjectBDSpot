import os
import time
from pathlib import Path

import mujoco as mu
import numpy as np

from anti_collision_policy.spot_env import TestEnv


# ============================================================
# Load MuJoCo Model
# ============================================================

# Resolve path relative to this file; fall back to CWD if __file__ is unavailable
try:
    ROOT_DIR = Path(__file__).resolve().parents[0]
except NameError:
    ROOT_DIR = Path.cwd()

MUJOCO_MODEL = ROOT_DIR / "boston_dynamics_spot" / "scene_arm.xml"

print("\n=== Loading MuJoCo Model ===")
try:
    model = mu.MjModel.from_xml_path(str(MUJOCO_MODEL))
    data = mu.MjData(model)
    print(f"Loaded model: OK, timestep={model.opt.timestep}")
except Exception as e:
    print(f"ERROR: Could not load model from {MUJOCO_MODEL}:\n{e}")
    raise SystemExit


# ============================================================
# Create Test Environment
# ============================================================

print("\n=== Creating Test Environment ===")
env = TestEnv(
    model=model,
    data=data,
    target_position=[12.5, -3.0, 0.0],
    render_mode="human",
)

obs, info = env.reset()
print(f"Initial State Position: {obs['state'][:3]}")
print(f"Observation keys: {list(obs.keys())}")


# ============================================================
# Predefined Action Pattern
# ============================================================

def pick_action(step: int, test_actions):
    """Return predefined actions for first N steps, default later."""
    if step < len(test_actions):
        return np.array(test_actions[step], dtype=np.float32)
    # default: slow straight walk
    return np.array([0.4, 0.0, 0.5], dtype=np.float32)


# [skill_switch, turn_cmd, walk_speed]
test_actions = [
    [0.8, 0.5, 0.4],   # mostly "turn", positive turn_cmd
    [0.3, -0.7, 0.8],  # "go_straight" with higher forward speed
    [0.6, -0.4, 0.8],  # "turn" with slight negative turn_cmd
]


# ============================================================
# Helper: Colored printing
# ============================================================

def color(text, c="cyan"):
    COLORS = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "magenta": "\033[95m",
        "cyan": "\033[96m",
        "reset": "\033[0m",
    }
    return COLORS.get(c, "") + str(text) + COLORS["reset"]


# ============================================================
# Run Episodes
# ============================================================

EPISODES = 3
MAX_HL_STEPS = 50

print("\n=== Starting Test Episodes ===")

for ep in range(EPISODES):
    print(color(f"\n=============== Episode {ep + 1}/{EPISODES} ===============", "yellow"))
    obs, info = env.reset()
    last_pos = obs["state"][:2].copy()

    for step in range(MAX_HL_STEPS):
        action = pick_action(step, test_actions)

        skill = "turn" if action[0] > 0.5 else "go_straight"
        vel = action[1] if skill == "turn" else action[2]

        print(color(f"\n--- HL Step {step} ---", "magenta"))
        print(f"Action = {action}")
        print(f"Skill = {skill}, Velocity = {vel:.3f}")

        # Gymnasium step API: obs, reward, terminated, truncated, info
        obs, reward, terminated, truncated, info = env.step(action)

        pos = obs["state"][:3]
        print(f"Position = [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        print(color(f"Reward = {reward:.3f}", "green"))
        print("Reward Breakdown:", info["reward_components"])

        # Check movement
        movement = np.linalg.norm(pos[:2] - last_pos)
        if movement < 1e-4:
            print(color("⚠ No movement detected (might be stuck)", "red"))
        last_pos = pos[:2].copy()

        # Exit conditions
        if terminated:
            print(color("🎉 Goal reached (terminated=True)!", "green"))
            break
        if truncated:
            print(color("⛔ Episode truncated!", "red"))
            break

        time.sleep(0.01)  # realtime pacing

    print(color("\nEpisode Summary:", "blue"))
    print(f"  Final Position: {pos[:2]}")
    print(f"  Distance to target: {info['distance']:.3f}\n")
    time.sleep(1.5)  # pause between episodes


# ============================================================
# Cleanup
# ============================================================

if hasattr(env, "viewer") and env.viewer is not None:
    env.viewer.close()

print(color("\n=== Test Completed ===\n", "green"))
