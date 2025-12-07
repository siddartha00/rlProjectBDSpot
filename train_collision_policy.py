import os
from pathlib import Path

import mujoco as mu
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import BaseCallback

from anti_collision_policy.spot_env import TestEnv
from anti_collision_policy.ppo_agent import SpotCombinedExtractor


class RewardComponentsCallback(BaseCallback):
    def __init__(self, log_freq=100, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq != 0:
            return True
        infos = self.locals.get("infos", [])
        if not infos:
            return True
        rc = infos[0].get("reward_components")
        if rc is None:
            return True
        for k, v in rc.items():
            self.logger.record(f"reward_components/{k}", float(v))
        return True


def make_env():
    """
    Factory for a single Spot+Arm environment instance.
    This is wrapped by DummyVecEnv for SB3 compatibility.
    """
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
        target_position=[12.5, -3.0, 0.0],
        render_mode=None,
    )
    return env


def main():
    # ===========================
    # Create VecEnv + VecMonitor
    # ===========================
    log_dir = "./logs/ppo_spot"
    os.makedirs(log_dir, exist_ok=True)

    vec_env = DummyVecEnv([make_env])
    # VecMonitor is what makes SB3 log ep_rew_mean, ep_len_mean, etc. [web:197][web:219]
    vec_env = VecMonitor(vec_env, log_dir)

    # ===========================
    # PPO + custom feature extractor
    # ===========================
    policy_kwargs = dict(
        features_extractor_class=SpotCombinedExtractor,
        features_extractor_kwargs=dict(target_hw=(96, 128)),
        normalize_images=False,
    )
    callback = RewardComponentsCallback()

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
        tensorboard_log=log_dir,   # enable TensorBoard [web:85]
    )

    # ===========================
    # Train
    # ===========================
    total_timesteps = 100_000
    model.learn(total_timesteps=total_timesteps, callback=callback)

    save_path = "./ppo_spot_nav_50k"
    model.save(save_path)
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    main()
