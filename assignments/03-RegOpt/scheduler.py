import math
from typing import List

from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):

        """
        Create a new scheduler.
        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        Initializing the CustomLRScheduler 

        Arguments:
          optimizer (torch.optim): To set the type of optimizer 
          last_epoch (int): Default value -1

        """
        self.optimizer = optimizer
        self.last_epoch = last_epoch
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> List[float]:
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        # Here's our dumb baseline implementation:
        """
        Required method of a learning scheduler class

        Arguments: 
          None
        
        Output:
          List[float]: List of new learning rates for all optimizer parameter groups
        """
        return [base_lr * math.exp(-0.1 * self.last_epoch) for base_lr in self.base_lrs]
