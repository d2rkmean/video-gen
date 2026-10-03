from dataclasses import dataclass


@dataclass
class BaseComposer:
    out_stream: str
    script_string: str | None = None

    def build(self) -> str:
        raise NotImplementedError()
