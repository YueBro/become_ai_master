from enum import StrEnum


class Milestones(StrEnum):
    TR_ST       = "train_start"
    TR_EP_ST    = "train_epoch_start"
    TR_IT_ST    = "train_iter_start"
    TR_BEF_STEP = "train_before_step"
    TR_IT_EN    = "train_iter_end"
    TR_EP_EN    = "train_epoch_end"
    TR_EN       = "train_end"

    EV_ST       = "eval_start"
    EV_IT_ST    = "eval_iter_start"
    EV_IT_EN    = "eval_iter_end"
    EV_EN       = "eval_end"
