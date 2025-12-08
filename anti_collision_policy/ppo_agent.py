import gymnasium as gym
import torch as th
from torch import nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpotCombinedExtractor(BaseFeaturesExtractor):
    """
    Custom extractor for Dict obs:
      - ['depth'] and ['obstacle_mask'] -> stacked, resized to 96x128, CNN
      - ['state'] (7,) and ['heading_yaw'] (1,) -> concatenated with CNN features
    Output is a single feature vector per obs.
    """

    def __init__(self, observation_space: gym.spaces.Dict,
                 target_hw=(96, 128)):
        super().__init__(observation_space, features_dim=1)  # dummy, will overwrite

        assert isinstance(observation_space, gym.spaces.Dict)

        self.target_h, self.target_w = target_hw

        depth_space = observation_space["depth"]
        mask_space = observation_space["obstacle_mask"]
        state_space = observation_space["state"]
        heading_space = observation_space["heading_yaw"]
        target_space = observation_space["target_position"]

        # Sanity checks
        assert isinstance(depth_space, gym.spaces.Box) and depth_space.shape[0] == 1
        assert isinstance(mask_space, gym.spaces.Box) and mask_space.shape[0] == 1
        assert isinstance(state_space, gym.spaces.Box) and state_space.shape[0] == 7
        assert isinstance(heading_space, gym.spaces.Box) and heading_space.shape[0] == 1
        assert isinstance(target_space, gym.spaces.Box) and target_space.shape[0] == 3

        # CNN on stacked (depth, mask) of shape (B, 2, 96, 128)
        # Tune architecture as needed
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output dim with dummy forward
        with th.no_grad():
            dummy = th.zeros(1, 2, self.target_h, self.target_w)
            n_cnn = self.cnn(dummy).shape[1]

        n_state = state_space.shape[0]           # 7
        n_heading = heading_space.shape[0]       # 1
        n_target = target_space.shape[0]
        total_features = n_cnn + n_state + n_heading + n_target

        # Optional linear layer to control final features_dim
        self.linear = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(),
        )

        self._features_dim = 256  # what PPO will see

    def forward(self, observations: dict) -> th.Tensor:
        # depth, mask: assume (B, 1, H, W) or (B, H, W)
        depth = observations["depth"]
        mask = observations["obstacle_mask"]

        # Ensure 4D (B, C, H, W)
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)

        # Stack depth + mask along channel dimension -> (B, 2, H, W)
        x = th.cat([depth, mask], dim=1)

        # Resize to target (96, 128)
        x = F.interpolate(
            x,
            size=(self.target_h, self.target_w),
            mode="bilinear",
            align_corners=False,
        )

        cnn_out = self.cnn(x)  # (B, n_cnn)

        # Low-dim inputs
        state = observations["state"]          # (B, 7)
        heading = observations["heading_yaw"]  # (B, 1)
        target_position = observations["target_position"]

        flat = th.cat([cnn_out, state, heading, target_position], dim=1)
        return self.linear(flat)  # (B, 256)

