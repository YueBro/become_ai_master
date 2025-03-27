from dataclasses import dataclass


@dataclass
class Cfg:
    total_train_epoch: int  # epoch counts as `range(total_train_epoch)`
