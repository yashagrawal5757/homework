import math
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Custom LR Scheduler class
    """

    def __init__(self, optimizer, last_epoch=-1, **kwargs) -> None:
        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        Initializing the CustomLRScheduler

        Arguments:
          optimizer (torch.optim): To set the type of optimizer
          last_epoch (int): Default value -1

        Returns : None

        """
        self.max_epochs = kwargs["max_epochs"]
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        """
        Returns the list of learning rates generated using the chainable form of the
        scheduler.
        """
        # return [group["lr"] for group in self.optimizer.param_groups]

        # Set the learning rate for each parameter group
        curr_epoch = self.last_epoch + 1
        return [
            0.25 * (1.0 + math.cos(math.pi * curr_epoch / self.max_epochs)) * base_lr
            for base_lr in self.base_lrs
        ]
