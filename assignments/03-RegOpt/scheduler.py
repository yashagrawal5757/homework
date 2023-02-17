import math
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom LR Scheduler class
    """

    def __init__(self, optimizer, last_epoch=-1, **kwargs):
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        Initializing the CustomLRScheduler

        Arguments:
          optimizer (torch.optim): To set the type of optimizer
          last_epoch (int): Default value -1

        """
        self.step_size = kwargs["step_size"]
        self.gamma = kwargs["gamma"]
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns the learning rates generated using the chainable form of the
        scheduler.
        """
        return [group["lr"] for group in self.optimizer.param_groups]
