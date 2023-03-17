import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CONFIG


class Model(torch.nn.Module):
    def __init__(self, num_channels: int, num_classes: int) -> None:
        """
        Function to initialize the model

        Arguments:
        num_channels(int):  number of channels
        num_classes(int):

        Returns:
        None

        """
        super(Model, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_channels, 16, kernel_size=3, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = torch.nn.ReLU()
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = torch.nn.Linear(64 * 4 * 4, 128)
        self.fc2 = torch.nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward function of the CNN

        Arguments:
        x (int): Input to the network

        Output:
        (torch.Tensor): Output of the network
        """

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x
