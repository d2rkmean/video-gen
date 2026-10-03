from abc import ABC, abstractmethod
from typing import Any


class BaseWorker(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        pass
