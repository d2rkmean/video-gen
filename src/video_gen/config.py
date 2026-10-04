from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class VideoSource(BaseModel):
    type: Literal["video", "pygame"]
    source: str


class VideosConfig(BaseModel):
    top_videos: list[VideoSource] = Field(
        default_factory=lambda: [VideoSource(type="video", source="input/")]
    )
    buttom_videos: list[VideoSource] = Field(
        default_factory=lambda: [VideoSource(type="pygame", source="bouncing_balls")]
    )


class PipelineParams(BaseModel):
    output_dir: Path = Field(default=Path("output/"))
    gemini_key: str
    gemini_model: str = "gemini-3.5-flash-lite"
    parts_count: int = Field(default=5, ge=1, le=20)
    part_duration: int = Field(default=60, ge=10, le=180)
    whisper_model: str = "large-v3"

    @field_validator("output_dir", mode="after")
    @classmethod
    def ensure_output_exists(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


class AppConfig(BaseModel):
    videos: VideosConfig
    params: PipelineParams

    @classmethod
    def from_toml(cls, path: str | Path) -> "AppConfig":
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)
