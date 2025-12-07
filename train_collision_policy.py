import os
from pathlib import Path

import mujoco as mu
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from anti_collision_policy.spot_env import TestEnv
from anti_collision_policy.ppo_agent import SpotCombinedExtractor


def make_env():
    """
    Factory for a single Spot+Arm environment instance.
    This is wrapped by DummyVecEnv for SB3 compatibility.
    """
    # Resolve XML path relative to this file
    try:
        root_dir = Path(__file__).resolve().parents[0]
    except NameError:
        root_dir = Path.cwd()

    mjcf_path = root_dir / "boston_dynamics_spot" / "scene_arm.xml"

    # Load MuJoCo model + data
    model = mu.MjModel.from_xml_path(str(mjcf_path))
    data = mu.MjData(model)

    # Use no on-screen rendering for training
    env = TestEnv(
        model=model,
        data=data,
        target_position=[12.5, -3.0, 0.0],
        render_mode=None,
    )
    return env


def main():
    # ===========================
    # Create VecEnv
    # ===========================
    # Single-env DummyVecEnv; you can increase n_envs with multiple make_env copies
    vec_env = DummyVecEnv([make_env])

    # ===========================
    # PPO + custom feature extractor
    # ===========================
    log_dir = "./logs/ppo_spot"
    os.makedirs(log_dir, exist_ok=True)

    policy_kwargs = dict(
        features_extractor_class=SpotCombinedExtractor,
        features_extractor_kwargs=dict(target_hw=(96, 128)),
        # depth/mask are already normalized in the env
        normalize_images=False,
    )

    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=log_dir,  # enables TensorBoard logging[web:85][web:90]
    )

    # ===========================
    # Train
    # ===========================
    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps)

    # Save final model
    save_path = "./ppo_spot_nav_50k"
    model.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
