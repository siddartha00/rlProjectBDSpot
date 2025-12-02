import os
import glob
import time
from datetime import datetime
import torch
import numpy as np
from spot_env import SpotEnv
import keyboard
import time
# from spot_env_turn import SpotEnv
# from spot_env_back import SpotEnv
from ppo import PPO


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    env_name = "spot_env_walking"
    has_continuous_action_space = True
    max_ep_len = 20000           # max timesteps in one episode
    action_std = 0.5            # set same std for action distribution which was used while saving

    render = True              # render environment on screen
    frame_delay = 0             # if required; add delay b/w frames

    total_test_episodes = 10    # total num of testing episodes

    K_epochs = 20               # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    gamma = 0.99                # discount factor

    lr_actor = 0.001           # learning rate for actor
    lr_critic = 0.003           # learning rate for critic

    #####################################################

    env = SpotEnv(mode='command')

    # state space dimension
    state_dim = env.obs_dims()

    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_dims()
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

    # preTrained weights directory

    random_seed = 0             #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0      #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    checkpoint_path = "PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_straight_2.pth"
    # checkpoint_path = "/home/prashanth/rlProjectBDSpot/PPO_preTrained/spot_env_walking/updated_final_policies/PPO_spot_env_walking_0_0_turn_4_final.pth"
    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")
    state, timeout_ = env.reset()
    if timeout_:
        return


    try:
        while True:
            vel = 0.5       # change this based on requirement (this will be angular velocity for turn and linear velocity for straight)
            env.give_vel_command(vel)
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)            
            time.sleep(frame_delay)
            
    except KeyboardInterrupt:
        print("Test stopped by user")

    ppo_agent.buffer.clear()



if __name__ == '__main__':

    test()
