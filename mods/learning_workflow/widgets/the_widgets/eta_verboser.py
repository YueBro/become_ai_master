import time

from tqdm import tqdm

from ..widgets_base import WidgetsBase
from ...data_cls import Cfg, PipelineStatus

from typing import Any, Optional, Callable


class EtaVerboser(WidgetsBase):
    def __init__(self, extra_verbose_fn: Optional[Callable[[Cfg, PipelineStatus], str]] = None):
        self.iter_tqdm: Optional[tqdm] = None
        self.extra_verbosing_fn = extra_verbose_fn
    
    def on_train_epoch_start(self, cfg: Cfg, status: PipelineStatus):
        avg_epoch_dur = (time.time() - status.start_time) / (status.current_ep_idx - status.start_ep_idx + 1)
        eta = avg_epoch_dur * (status.end_ep_idx - status.current_ep_idx)
        eta_str = self._format_dur(eta)
        vbs_str = f"eta={eta_str}"
        if self.extra_verbosing_fn is not None:
            vbs_str += " | " + self.extra_verbosing_fn(cfg, status)
        print(f"[Epoch {status.current_ep_idx+1}/{cfg.total_train_epoch}] {vbs_str}")
    
    def on_train_iter_start(self, cfg: Cfg, status: PipelineStatus, data_batch: Any):
        if status.current_it_idx == 0:
            self.iter_tqdm = tqdm(total=status.total_its)
    
    def on_train_iter_end(self, cfg: Cfg, status: PipelineStatus, data_batch: Any, forward_ret: Any):
        if self.iter_tqdm is None:
            raise RuntimeError
        self.iter_tqdm.update()

    def on_train_epoch_end(self, cfg: Cfg, status: PipelineStatus):
        if self.iter_tqdm is None:
            raise RuntimeError
        self.iter_tqdm.close()
    
    @staticmethod
    def _format_dur(dur: float):
        h = int(dur) // 3600
        dur -= h * 3600.0

        m = int(dur) // 60
        dur -= m * 60.0

        s = int(dur)

        return f"{h:0>2}:{m:0>2}:{s:0>2}"
