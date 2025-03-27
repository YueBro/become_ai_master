"""
This is an executor only for you to debug other stuffs!
"""

import random

from ..data_list import DataList
from .executor import Executor

from typing import List, Any


class DummyExecutor(Executor):
    def get_train_datas(self) -> DataList:
        vals = [(val ** 2) for val in list(range(8))]
        idxs = list(range(8))
        random.shuffle(idxs)
        return DataList(vals, batch_size=1, idx_order=idxs)

    def train_forward(self, data: List[int]) -> Any:
        return ...

    def step(self, data: List[int], forward_ret: Any):
        pass

    def get_eval_datas(self) -> DataList:
        return self.get_train_datas()

    def eval_forward(self, data: Any) -> Any:
        return self.train_forward(data)
