import os
from pathlib import Path

import mujoco as mu
import numpy as np
from stable_baselines3 import PPO

from anti_collision_policy.spot_env import TestEnv
from anti_collision_policy.ppo_agent import SpotCombinedExtractor  # ensures class is registered


MODEL_PATH = "./ppo_spot_nav_50k.zip"   # or your 500k checkpoint
TARGET_POSITION = [12.5, -3.0, 0.0]
NUM_EPISODES = 5
MAX_HL_STEPS = 1000    # high-level steps (each contains multiple mj_steps)


def make_env(render_mode="human"):
    """Create a single Spot+Arm env consistent with training."""
    try:
        root_dir = Path(__file__).resolve().parents[0]
    except NameError:
        root_dir = Path.cwd()

    mjcf_path = root_dir / "boston_dynamics_spot" / "scene_arm.xml"

    model = mu.MjModel.from_xml_path(str(mjcf_path))
    data = mu.MjData(model)

    env = TestEnv(
        model=model,
        data=data,
        target_position=TARGET_POSITION,
        render_mode=render_mode,
    )
    return env


def main():
    # -----------------------
    # Load environment
    # -----------------------
    env = make_env(render_mode="human")

    # -----------------------
    # Load trained model
    # -----------------------
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    # env is only needed here to infer spaces if not stored; SB3 stores them,
    # so we can pass env=None when loading. Keeping env ensures same structure.
    model = PPO.load(MODEL_PATH, env=env)
    # During inference, use deterministic=True to exploit the learned policy[web:73][web:116][web:148]

    # -----------------------
    # Run evaluation episodes
    # -----------------------
    for ep in range(NUM_EPISODES):
        print(f"\n========== Evaluation Episode {ep + 1}/{NUM_EPISODES} ==========")
        obs, info = env.reset()
        done = False
        truncated = False
        step = 0

        while not (done or truncated) and step < MAX_HL_STEPS:
            # SB3 PPO .predict handles dict observations directly
            action, _states = model.predict(obs, deterministic=True)

            obs, reward, done, truncated, info = env.step(action)

            pos = obs["state"][:3]
            dist = info.get("distance", np.nan)
            print(
                f"Step {step:03d}: "
                f"pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
                f"dist_to_target={dist:.3f}, "
                f"reward={reward:.3f}, "
                f"skill={info.get('skill')}, "
                f"obs_coverage={info.get('obstacle_coverage'):.3f}"
            )

            step += 1

        if done:
            print("✅ Episode terminated: goal reached.")
        elif truncated:
            print("⛔ Episode truncated (time/obstacle condition).")
        else:
            print("⚠ Max high-level steps reached without termination/truncation.")

    if hasattr(env, "viewer") and env.viewer is not None:
        env.viewer.close()


if __name__ == "__main__":
    main()
