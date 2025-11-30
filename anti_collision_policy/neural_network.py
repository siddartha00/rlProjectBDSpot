import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, input_channels=1):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU()
        )

    def preprocess(self, x):
        """Normalize depth input: clamp noise, scale to [0,1]"""
        x = torch.clamp(x, 0, 30) / 30  # stays differentiable
        return x

    def forward(self, x):
        x = self.preprocess(x)
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)  # ensures fixed output size
        x = self.mlp(x)
        return x
