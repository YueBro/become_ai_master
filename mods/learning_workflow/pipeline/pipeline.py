import time

from .milestones import Milestones
from ..data_cls import Cfg, PipelineStatus
from ..executor import Executor
from ..data_list import DataList
from ..recaller import Recaller
from ..widgets import WidgetsBase


class Pipeline:
    def __init__(self, cfg: Cfg, executor: Executor):
        self.cfg = cfg
        self.executor = executor
        self.recaller = Recaller(list(Milestones))

        self.status = PipelineStatus(
            is_train=False,
            start_ep_idx=-1,
            end_ep_idx=-1,
            current_ep_idx=-1,
            current_it_idx=-1,
            total_its=-1,
            start_time=-1.0,
        )
    
    def register(self, widget: WidgetsBase):
        if widget.on_train_start is not WidgetsBase.on_train_start:
            self.recaller.register(Milestones.TR_ST, widget.on_train_start)
        if widget.on_train_epoch_start is not WidgetsBase.on_train_epoch_start:
            self.recaller.register(Milestones.TR_EP_ST, widget.on_train_epoch_start)
        if widget.on_train_iter_start is not WidgetsBase.on_train_iter_start:
            self.recaller.register(Milestones.TR_IT_ST, widget.on_train_iter_start)
        if widget.on_train_before_step is not WidgetsBase.on_train_before_step:
            self.recaller.register(Milestones.TR_BEF_STEP, widget.on_train_before_step)
        if widget.on_train_iter_end is not WidgetsBase.on_train_iter_end:
            self.recaller.register(Milestones.TR_IT_EN, widget.on_train_iter_end)
        if widget.on_train_epoch_end is not WidgetsBase.on_train_epoch_end:
            self.recaller.register(Milestones.TR_EP_EN, widget.on_train_epoch_end)
        if widget.on_train_end is not WidgetsBase.on_train_end:
            self.recaller.register(Milestones.TR_EN, widget.on_train_end)
        if widget.on_eval_start is not WidgetsBase.on_eval_start:
            self.recaller.register(Milestones.EV_ST, widget.on_eval_start)
        if widget.on_eval_iter_start is not WidgetsBase.on_eval_iter_start:
            self.recaller.register(Milestones.EV_IT_ST, widget.on_eval_iter_start)
        if widget.on_eval_iter_end is not WidgetsBase.on_eval_iter_end:
            self.recaller.register(Milestones.EV_IT_EN, widget.on_eval_iter_end)
        if widget.on_eval_end is not WidgetsBase.on_eval_end:
            self.recaller.register(Milestones.EV_EN, widget.on_eval_end)

    def train(self, start_ep_idx: int, end_ep_idx: int):
        """
        start_ep: first epoch is 0
        end_ep: inclusive
        """

        if (start_ep_idx < 0) or (end_ep_idx > (self.cfg.total_train_epoch - 1)):
            raise ValueError(f"{start_ep_idx=}, {end_ep_idx=}, total_epoch={self.cfg.total_train_epoch}")

        self._update_status_on_train_start(start_ep_idx, end_ep_idx)
        self.recaller.trigger(Milestones.TR_ST, self.cfg, self.status)
        for ep_idx in range(start_ep_idx, end_ep_idx + 1):
            self._update_status_on_train_epoch_start(ep_idx)
            self.recaller.trigger(Milestones.TR_EP_ST, self.cfg, self.status)
            data_list: DataList = self.executor.get_train_datas()
            for it_idx, data_batch in enumerate(data_list):
                self._update_status_on_iter_start(it_idx, len(data_list))
                self.recaller.trigger(Milestones.TR_IT_ST, self.cfg, self.status, data_batch)
                ret = self.executor.train_forward(data_batch)
                self.recaller.trigger(Milestones.TR_BEF_STEP, self.cfg, self.status, data_batch, ret)
                self.executor.step(data_batch, ret)
                self.recaller.trigger(Milestones.TR_IT_EN, self.cfg, self.status, data_batch, ret)
            self.recaller.trigger(Milestones.TR_EP_EN, self.cfg, self.status)
        self.recaller.trigger(Milestones.TR_EN, self.cfg, self.status)

    def eval(self):
        self._update_status_on_eval_start()
        self.recaller.trigger(Milestones.EV_ST, self.cfg, self.status)
        data_list: DataList = self.executor.get_train_datas()
        for it_idx, data_batch in enumerate(data_list):
            self._update_status_on_iter_start(it_idx, len(data_list))
            self.recaller.trigger(Milestones.EV_IT_ST, self.cfg, self.status, data_batch)
            ret = self.executor.eval_forward(data_batch)
            self.recaller.trigger(Milestones.EV_IT_EN, self.cfg, self.status, data_batch, ret)
        self.recaller.trigger(Milestones.EV_EN, self.cfg, self.status)

    def _update_status_on_train_start(self, start_ep_idx: int, end_ep_idx: int):
        self.status.is_train = True
        self.status.start_ep_idx = start_ep_idx
        self.status.end_ep_idx = end_ep_idx
        self.status.start_time = time.time()

    def _update_status_on_train_epoch_start(self, curr_ep_idx: int):
        self.status.current_ep_idx = curr_ep_idx

    def _update_status_on_iter_start(self, curr_it_idx: int, total_its: int):
        self.status.current_it_idx = curr_it_idx
        self.status.total_its = total_its

    def _update_status_on_eval_start(self):
        self.status.is_train = False
        self.status.start_time = time.time()
