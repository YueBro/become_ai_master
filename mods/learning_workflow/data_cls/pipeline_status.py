from dataclasses import dataclass


@dataclass
class PipelineStatus:
    is_train: bool

    start_ep_idx: int
    end_ep_idx: int  # inclusive
    current_ep_idx: int

    current_it_idx: int
    total_its: int

    start_time: float
