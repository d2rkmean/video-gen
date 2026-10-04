"""Конфигурация фермы: загрузка и валидация config.toml."""

import os
import re
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ASS_COLOR = re.compile(r"^&H[0-9A-Fa-f]{8}$")


class Strict(BaseModel):
    """База: опечатка в ключе toml даёт ошибку, а не молча игнорируется."""

    model_config = ConfigDict(extra="forbid")


class VideoSource(Strict):
    type: Literal["video", "pygame"]
    source: str


class VideosConfig(Strict):
    top_videos: list[VideoSource] = Field(
        default_factory=lambda: [VideoSource(type="video", source="input/")]
    )
    buttom_videos: list[VideoSource] = Field(
        default_factory=lambda: [VideoSource(type="video", source="bottom/")]
    )

    @field_validator("top_videos")
    @classmethod
    def top_only_video(cls, v: list[VideoSource]) -> list[VideoSource]:
        if not v:
            raise ValueError("top_videos не может быть пустым")
        if any(s.type != "video" for s in v):
            raise ValueError('в top_videos поддерживается только type = "video"')
        return v

    @field_validator("buttom_videos")
    @classmethod
    def bottom_not_empty(cls, v: list[VideoSource]) -> list[VideoSource]:
        if not v:
            raise ValueError("buttom_videos не может быть пустым")
        return v


class PipelineParams(Strict):
    output_dir: Path = Field(default=Path("output/"))

    # Gemini
    gemini_key: str  # можно не писать в toml, а задать через GEMINI_API_KEY
    gemini_model: str = "gemini-3.5-flash-lite"
    parts_count: int = Field(default=5, ge=1, le=20)
    min_part_duration: int = Field(default=20, ge=5, le=170)
    part_duration: int = Field(default=60, ge=10, le=180)

    # Whisper
    whisper_model: str = "large-v3"
    whisper_device: Literal["cuda", "cpu", "auto"] = "cuda"
    whisper_compute_type: Literal["float16", "int8_float16", "int8", "float32"] = "float16"
    beam_size: int = Field(default=5, ge=1, le=10)
    language: str = Field(default="ru", min_length=2, max_length=5)

    @model_validator(mode="before")
    @classmethod
    def key_from_env(cls, data):
        """Если ключа нет в toml, берём из переменной окружения."""
        if isinstance(data, dict) and not data.get("gemini_key"):
            env = os.environ.get("GEMINI_API_KEY")
            if env:
                data = {**data, "gemini_key": env}
        return data

    @model_validator(mode="after")
    def durations_consistent(self) -> Self:
        if self.min_part_duration >= self.part_duration:
            raise ValueError("min_part_duration должен быть меньше part_duration")
        return self

    @field_validator("output_dir", mode="after")
    @classmethod
    def ensure_output_exists(cls, path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path


class AssetsConfig(Strict):
    banner: Path | None = None  # видео баннера; нет ключа = без баннера
    music_dir: Path | None = None  # папка с треками; нет ключа = без музыки
    fonts_dir: Path = Path("fonts/")

    @field_validator("banner")
    @classmethod
    def banner_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_file():
            raise ValueError(f"баннер не найден: {p}")
        return p

    @field_validator("music_dir")
    @classmethod
    def music_dir_exists(cls, p: Path | None) -> Path | None:
        if p is not None and not p.is_dir():
            raise ValueError(f"папка с музыкой не найдена: {p}")
        return p


class RenderConfig(Strict):
    """Поля совпадают с RenderSettings: RenderSettings(**cfg.model_dump())."""

    width: int = Field(default=1080, ge=240)
    height: int = Field(default=1920, ge=426)
    fps: int = Field(default=30, ge=15, le=60)
    mirror: bool = True
    top_zoom: float = Field(default=1.12, ge=1.0, le=2.0)
    contrast: float = Field(default=1.08, ge=0.5, le=2.0)
    saturation: float = Field(default=1.15, ge=0.0, le=3.0)
    brightness: float = Field(default=0.02, ge=-0.3, le=0.3)
    gamma: float = Field(default=1.03, ge=0.5, le=2.0)
    noise_top: int = Field(default=6, ge=0, le=40)
    noise_bottom: int = Field(default=5, ge=0, le=40)
    bottom_mode: Literal["random", "sequential"] = "random"
    blur_sigma: float = Field(default=40.0, ge=1.0, le=100.0)
    banner_width: int = Field(default=900, ge=100)
    duck_volume: float = Field(default=0.10, ge=0.0, le=1.0)
    music_volume: float = Field(default=0.07, ge=0.0, le=1.0)
    cq: int = Field(default=23, ge=10, le=40)

    @model_validator(mode="after")
    def banner_fits(self) -> Self:
        if self.banner_width > self.width:
            raise ValueError("banner_width не может быть больше width")
        return self


class SubtitlesConfig(Strict):
    """Стиль (-> SubtitleStyle) и правила разбивки на строки (-> SubtitleTask)."""

    font: str = "Montserrat"
    font_size: int = Field(default=84, ge=20, le=300)
    spoken_color: str = "&H0000FFFF"
    unspoken_color: str = "&H00FFFFFF"
    outline_color: str = "&H00000000"
    outline: int = Field(default=6, ge=0, le=20)
    alignment: int = Field(default=5, ge=1, le=9)  # 5 = по центру экрана
    margin_v: int = Field(default=0, ge=0)
    margin_h: int = Field(default=60, ge=0)
    uppercase: bool = True

    max_words: int = Field(default=3, ge=1, le=8)
    max_chars: int = Field(default=24, ge=8, le=60)
    max_gap: float = Field(default=0.6, ge=0.1, le=3.0)
    hold: float = Field(default=0.2, ge=0.0, le=2.0)

    @field_validator("spoken_color", "unspoken_color", "outline_color")
    @classmethod
    def valid_ass_color(cls, v: str) -> str:
        if not ASS_COLOR.match(v):
            raise ValueError(f"цвет должен быть в формате &HAABBGGRR, получено: {v}")
        return v


class AppConfig(Strict):
    videos: VideosConfig
    params: PipelineParams
    assets: AssetsConfig = Field(default_factory=AssetsConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    subtitles: SubtitlesConfig = Field(default_factory=SubtitlesConfig)

    @classmethod
    def from_toml(cls, path: str | Path) -> "AppConfig":
        with Path(path).open("rb") as f:
            data = tomllib.load(f)
        return cls.model_validate(data)
