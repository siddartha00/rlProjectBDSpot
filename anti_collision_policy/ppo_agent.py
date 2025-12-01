from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from pathlib import Path
from neural_network import FeatureExtractor
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class nav_agent(PPO):
    def __init__(self, env):
        super().__init__()
        self.extractor = FeatureExtractor()
        self.
        self.env = env
        self.model = 