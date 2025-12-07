import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class SpotCNN(BaseFeaturesExtractor):
    """CNN updated for observation WITH target info"""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        # New observation size: 12288(depth) + 12288(mask) + 7(state) + 4(target) = 24587
        super().__init__(observation_space, features_dim)

        # CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 8)),
            nn.Flatten(),
        )

        # Calculate CNN output size
        with torch.no_grad():
            test_input = torch.zeros(1, 2, 96, 128)
            n_flatten = self.cnn(test_input).shape[1]

        print(f"CNN output features: {n_flatten}")

        # Non-visual features: robot state (7) + target info (4)
        self.non_visual_features = 11

        # MLP for combining features
        self.mlp = nn.Sequential(
            nn.Linear(n_flatten + self.non_visual_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights for better training"""
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Split observation into components
        # Total: 24587 = 12288(depth) + 12288(mask) + 7(state) + 4(target)
        depth = observations[:, :12288].reshape(batch_size, 1, 96, 128)
        mask = observations[:, 12288:24576].reshape(batch_size, 1, 96, 128)
        robot_state = observations[:, 24576:24583]  # 7 values
        target_info = observations[:, 24583:]       # 4 values [distance, heading_error, rel_x, rel_y]

        # Combine depth and mask as 2-channel input
        visual_input = torch.cat([depth, mask], dim=1)

        # Extract visual features
        visual_features = self.cnn(visual_input)

        # Combine all features
        combined = torch.cat([visual_features, robot_state, target_info], dim=1)

        # Final feature representation
        return self.mlp(combined)
