from ..data_cls import Cfg, PipelineStatus

from typing import Any


class WidgetsBase:
    def on_train_start(self, cfg: Cfg, status: PipelineStatus):
        pass

    def on_train_epoch_start(self, cfg: Cfg, status: PipelineStatus):
        pass

    def on_train_iter_start(self, cfg: Cfg, status: PipelineStatus, data_batch: Any):
        pass

    def on_train_before_step(self, cfg: Cfg, status: PipelineStatus, data_batch: Any, forward_ret: Any):
        pass

    def on_train_iter_end(self, cfg: Cfg, status: PipelineStatus, data_batch: Any, forward_ret: Any):
        pass

    def on_train_epoch_end(self, cfg: Cfg, status: PipelineStatus):
        pass

    def on_train_end(self, cfg: Cfg, status: PipelineStatus):
        pass

    def on_eval_start(self, cfg: Cfg, status: PipelineStatus):
        pass

    def on_eval_iter_start(self, cfg: Cfg, status: PipelineStatus, data_batch: Any):
        pass

    def on_eval_iter_end(self, cfg: Cfg, status: PipelineStatus, data_batch: Any, forward_ret: Any):
        pass

    def on_eval_end(self, cfg: Cfg, status: PipelineStatus):
        pass
