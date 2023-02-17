from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import (
    Compose,
    Normalize,
    ToTensor,
    RandomCrop,
    RandomHorizontalFlip,
    RandomRotation,
    ColorJitter,
)


class CONFIG:
    batch_size = 64
    num_epochs = 4
    initial_learning_rate = 0.0005
    initial_weight_decay = 0

    lrs_kwargs = {
        # You can pass arguments to the learning rate scheduler
        # constructor here.
        "step_size": 2,
        "gamma": 0.1,
    }

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(
        model.parameters(),
        lr=CONFIG.initial_learning_rate,
        weight_decay=CONFIG.initial_weight_decay,
    )

    transforms = Compose(
        [
            ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            RandomRotation(15),
            ToTensor(),
            Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
