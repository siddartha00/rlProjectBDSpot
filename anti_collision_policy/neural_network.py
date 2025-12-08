import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpotCNN(BaseFeaturesExtractor):
    """CNN updated for observation WITH target info"""

    def __init__(self, observation_space: gym.spaces.Dict, target_hw=(96, 128)):
        super().__init__(observation_space, features_dim=1)

        assert isinstance(observation_space, gym.spaces.Dict)

        self.target_h, self.target_w = target_hw

        depth_space = observation_space["depth"]
        mask_space = observation_space["obstacle_mask"]
        state_space = observation_space["state"]
        heading_space = observation_space["heading_yaw"]
        target_space = observation_space["target_position"]  # NEW

        assert isinstance(depth_space, gym.spaces.Box) and depth_space.shape[0] == 1
        assert isinstance(mask_space, gym.spaces.Box) and mask_space.shape[0] == 1
        assert isinstance(state_space, gym.spaces.Box) and state_space.shape[0] == 7
        assert isinstance(heading_space, gym.spaces.Box) and heading_space.shape[0] == 1
        assert isinstance(target_space, gym.spaces.Box) and target_space.shape[0] == 3

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 2, self.target_h, self.target_w)
            n_cnn = self.cnn(dummy).shape[1]

        n_state = state_space.shape[0]          # 7
        n_heading = heading_space.shape[0]      # 1
        n_target = target_space.shape[0]        # 3

        total_features = n_cnn + n_state + n_heading + n_target  # UPDATED

        self.linear = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
        )

        self._features_dim = 256

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, observations: dict) -> torch.Tensor:
        depth = observations["depth"]
        mask = observations["obstacle_mask"]

        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        x = torch.cat([depth, mask], dim=1)
        x = F.interpolate(x, size=(self.target_h, self.target_w),
                          mode="bilinear", align_corners=False)
        cnn_out = self.cnn(x)

        state = observations["state"]
        heading = observations["heading_yaw"]
        target_position = observations["target_position"]

        flat = torch.cat([cnn_out, state, heading, target_position], dim=1)
        return self.linear(flat)
