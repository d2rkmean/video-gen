from typing import Any

from video_gen.config import AppConfig


class BaseTask:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def run(self, *args, **kwargs) -> Any:
        pass
