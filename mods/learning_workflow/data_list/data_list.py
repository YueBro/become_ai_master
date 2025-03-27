import math

from typing import Sequence, List, Optional, Iterable


def _regroup_by_group_size(itr: Sequence, group_size: int):
    cache = []
    for i, v in enumerate(itr):
        cache.append(v)
        if (i + 1) % group_size == 0:
            yield cache
            cache = []
    
    if len(cache) > 0:
        yield cache


class DataList:
    def __init__(self, data_list: Sequence, batch_size: int, idx_order: Optional[List[int]] = None):
        self.data_list = data_list
        self.batch_size = batch_size
        self.idx_order = idx_order if (idx_order is not None) else list(range(len(data_list)))
    
    def __iter__(self):
        idxs_grp: Iterable[List[int]] = _regroup_by_group_size(self.idx_order, self.batch_size)
        return (
            [self.data_list[idx] for idx in idxs]
            for idxs in idxs_grp
        )

    def __len__(self):
        return math.ceil(len(self.data_list) / self.batch_size)
