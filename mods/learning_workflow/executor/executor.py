from abc import ABC, abstractmethod

from ..data_list import DataList

from typing import Any


class Executor(ABC):
    @abstractmethod
    def get_train_datas(self) -> DataList:
        pass

    @abstractmethod
    def train_forward(self, data: Any) -> Any:
        pass

    @abstractmethod
    def step(self, data: Any, forward_ret: Any):
        pass

    @abstractmethod
    def get_eval_datas(self) -> DataList:
        pass

    @abstractmethod
    def eval_forward(self, data: Any) -> Any:
        pass
