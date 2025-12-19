import os
import numpy as np
# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import CheckpointCallback
import time

# Import your HRL environment class
from walking_hrl_policy import HierarchicalSpotEnv  # Adjust import path as needed
# from stable_baselines3.common.utils import LinearSchedule



from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

def train_hrl_policy_dqn():

    STRAIGHT_POLICY_PATH = "/home/sankarap/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_straight_2.pth"
    TURN_POLICY_PATH = "/home/sankarap/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_turn_4_final.pth"

    os.makedirs("./hrl_checkpoints_new_1", exist_ok=True)
    os.makedirs("./hrl_tensorboard_new_1", exist_ok=True)

    def make_env():
        env = HierarchicalSpotEnv(STRAIGHT_POLICY_PATH, TURN_POLICY_PATH)
        env = Monitor(env, "./hrl_logs_1/")
        return env
    

    # model_path = "/home/prashanth/rlProjectBDSpot/hrl_final_policy_dqn_better_5.zip"

    env = DummyVecEnv([make_env])

    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=3e-5,          # LOWER LR = more stable
        buffer_size=400000,          # larger buffer helps long episodes
        learning_starts=20000,       # wait longer before learning
        batch_size=256,              # ok
        tau=1.0,                     # ok (DQN uses hard update)
        gamma=0.995,                 # slightly longer horizon
        train_freq=4,
        gradient_steps=1,
        exploration_fraction=0.4,    # MORE EXPLORATION for HRL
        exploration_final_eps=0.02,  # slightly lower final eps
        target_update_interval=1000, # slower, smoother updates
        policy_kwargs=dict(net_arch=[512, 512]),  # bigger network needed
        verbose=1,
        tensorboard_log="./hrl_tensorboard_new_1/"
    )


    checkpoint_callback = CheckpointCallback(
        save_freq=10000,                  # MORE FREQUENT: Save more checkpoints
        save_path="./hrl_checkpoints_new_1/",
        name_prefix="dqn_hrl",
        save_replay_buffer=True,         # ADDED: Save replay buffer
        save_vecnormalize=True           # ADDED: Save normalization if using
    )

    model.learn(
        total_timesteps=3000000,
        callback=checkpoint_callback,
        tb_log_name="dqn_hrl_new_1"
    )

    model.save("hrl_final_policy_dqn_better_15")
    print("Saved final DQN policy as hrl_final_policy_dqn_better_15.zip")

def test_hrl_policy(model_path=None):
    """Test the trained HRL policy with discrete actions"""
    
    STRAIGHT_POLICY_PATH = "/home/sankarap/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_straight_2.pth"
    TURN_POLICY_PATH = "/home/sankarap/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_turn_4_final.pth"
    
    env = HierarchicalSpotEnv(STRAIGHT_POLICY_PATH, TURN_POLICY_PATH)

    # Default path
    if model_path is None:
        model_path = "/home/sankarap/rlProjectBDSpot/hrl_final_policy_dqn_better_15"

    print("Loading model...")
    
    # Load DQN (or PPO)
    model = DQN.load(model_path, env=env)   # <-- change DDPG → DQN

    print("Testing HRL policy...")

    ACTION_NAMES = ["TURN_LEFT", "TURN_RIGHT", "STRAIGHT"]

    for episode in range(10):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        steps = 0
        policy_switches = 0
        last_action = None

        while not done:
            # DQN returns a SINGLE integer
            action, _ = model.predict(obs)  
            obs, reward, done, truncated, info = env.step(action)

            total_reward += reward
            steps += 1

            # Track number of discrete action switches
            if last_action is not None and action != last_action:
                policy_switches += 1
            last_action = action

            # Debug every 100 steps
            if steps % 100 == 0:
                current_pose = env.current_policy.get_current_pose()
                target_pos = env.pos_or_command[:2]
                pos_err = np.linalg.norm(np.array(current_pose[:2]) - np.array(target_pos))
                dir_err , _ = env.compute_direction_error(current_pose[0],current_pose[1],current_pose[2],target_pos[0],target_pos[1])
                signed_yaw_err = np.arctan2(np.sin(current_pose[2] - env.pos_or_command[2]), np.cos(current_pose[2] - env.pos_or_command[2]))
                yaw_err = abs(signed_yaw_err)
                #yaw_err = abs(current_pose[2] - env.pos_or_command[2])

                print(f"Step {steps}: "
                      f"Action={ACTION_NAMES[action]}, "
                      f"PosErr={pos_err:.2f}, "
                      f"DirErr={dir_err:.2f}, "
                      f"FinalYawErr={yaw_err:.2f}, "
                      f"Reward={reward:.2f}")

        # Episode summary
        current_pose = env.current_policy.get_current_pose()
        target_pos = env.pos_or_command[:2]

        pos_err = np.linalg.norm(np.array(current_pose[:2]) - np.array(target_pos))
        yaw_err = abs(current_pose[2] - env.pos_or_command[2])

        print("\nEpisode", episode + 1, "Summary:")
        print(f"  Total reward: {total_reward:.2f}")
        print(f"  Steps: {steps}")
        print(f"  Action switches: {policy_switches}")
        print(f"  Final position error: {pos_err:.2f} m")
        print(f"  Final yaw error: {np.degrees(yaw_err):.1f}°")

        success = (pos_err < 1.0) and (yaw_err < 0.4)
        print(f"  Success: {success}")

        if success:
            print("  Reason: SUCCESS (Reached goal)")
        elif steps >= 10000:
            print("  Reason: TIMEOUT (Max steps reached)")
        else:
            print("  Reason: TERMINATED (Robot fell)")
        
        print("-" * 50)

    env.close()


if __name__ == "__main__":
    # Train the policy
    #train_hrl_policy_dqn()
    
    # Test the policy
    test_hrl_policy()
