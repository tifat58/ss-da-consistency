from torch.optim.lr_scheduler import _LRScheduler

class ConstantLR(_LRScheduler):
    def __init__(self, optimizer, last_epoch=-1):
        super(ConstantLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, gamma=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.gamma = gamma
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 - self.last_epoch / float(self.max_iter)) ** self.gamma
        factor = max(factor, 0.0)
        return [base_lr * factor for base_lr in self.base_lrs]

class InvLR(_LRScheduler):
    def __init__(self, optimizer,  power=0.75, gamma=0.001, max_iter=1, last_epoch=-1):
        self.max_iter = max_iter
        self.power = power
        self.gamma = gamma
        super(InvLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1 + self.gamma * self.last_epoch / float(self.max_iter)) ** (- self.power)
        return [base_lr * factor for base_lr in self.base_lrs]

class WarmUpLR(_LRScheduler):
    def __init__(
        self, optimizer, scheduler, mode="linear", warmup_iters=100, gamma=0.2, last_epoch=-1
    ):
        self.mode = mode
        self.scheduler = scheduler
        self.warmup_iters = warmup_iters
        self.gamma = gamma
        super(WarmUpLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        cold_lrs = self.scheduler.get_lr()

        if self.last_epoch < self.warmup_iters:
            if self.mode == "linear":
                alpha = self.last_epoch / float(self.warmup_iters)
                factor = self.gamma * (1 - alpha) + alpha

            elif self.mode == "constant":
                factor = self.gamma
            else:
                raise KeyError("WarmUp type {} not implemented".format(self.mode))

            return [factor * base_lr for base_lr in cold_lrs]

        return cold_lrs
