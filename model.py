import torch as th
import torch.nn as nn
import gym

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNNPolicy(BaseFeaturesExtractor):
    """
    Policy object that implements actor critic, using a CNN as feature extractor.

    :param observation_space: (gym.spaces.Box)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super(CustomCNNPolicy, self).__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))