import numpy as np
import gym
import torch
import torch.nn as nn
import torch.autograd as autograd

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        self.observation_space = observation_space

        self.features = nn.Sequential(
            nn.Conv2d(
                self.observation_space.shape[0], 32, kernel_size=8, stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.features(observations)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self) -> int:
        return self.features(
            autograd.Variable(torch.zeros(1, *self.observation_space.shape))
        ).view(1, -1).size(1)