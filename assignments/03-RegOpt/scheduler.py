from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """
    Implementation of a custom learning rate scheduler
    """

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
    ) -> None:
        # Dummy text to make it commit
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float]:
        """
        Returns the learning rates generated using the chainable form of the
        scheduler.
        """
        return [group["lr"] for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self) -> list[float]:
        """
        Returns the learning rates generated using the closed form approach.
        """
        return self.base_lrs
