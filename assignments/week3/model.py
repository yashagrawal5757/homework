import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    """
    Implementation of multi-layer perceptron.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = torch.nn.ReLU,
        initializer: Callable = torch.nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.
        Arguments:
            input_size (int): The dimension D of the input data.
            hidden_size (int): The number of neurons H in the hidden layer.
            num_classes (int): The number of classes C.
            activation (Callable): The activation function to use in the hidden layer.
            initializer (Callable): The initializer to use for the weights.
        """
        super().__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size))
        self.layers.append(activation())
        for _ in range(hidden_count - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(activation())
        self.layers.append(nn.Linear(hidden_size, num_classes))

        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                initializer(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        Arguments:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the network.
        """
        for layer in self.layers:
            x = layer(x)

        return x
