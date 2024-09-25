import torch
import torch.nn as nn
import numpy as np

class RegressionNetwork(torch.nn.Module):
    def __init__(self):
        """
        Implementation of the network layers. The image size of the input
        observations is 96x96 pixels.
        """
        super().__init__()
        self.gpu = torch.device('cuda')
        self.model = torch.nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=5, stride=2),
                nn.BatchNorm2d(32),
                nn.Sigmoid(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.Dropout(p=0.25),
                nn.Sigmoid(),
                nn.Conv2d(32, 32, kernel_size=5, stride=2),
                nn.Sigmoid(),
                nn.BatchNorm2d(32),
                nn.Flatten(),
                nn.Linear(31968, 2048),
                nn.Dropout(p=0.25),
                nn.Sigmoid(),
                nn.Linear(2048, 2048),
                nn.Dropout(p=0.25),
                nn.Sigmoid(),
                nn.Linear(2048, 3)
                ).to(self.gpu)


    def forward(self, observation):
        """
        The forward pass of the network. Returns the prediction for the given
        input observation.
        observation:   torch.Tensor of size (batch_size, height, width, channel)
        return         torch.Tensor of size (batch_size, C)
        """
        observation = self.model(observation)
        return observation



