#!/usr/bin/env python3
"""Тестовый запуск сборщика видео с баннером Video-Gen в 60 FPS."""

import tomllib
from pathlib import Path
from video_gen.config import AppConfig
from video_gen.models import Highlight
from video_gen.tasks.ffmpeg_task import FFmpegTask, RenderSettings

ROOT = Path(__file__).resolve().parent.parent


def main():
    config_file = ROOT / "config.toml"
    with open(config_file, "rb") as f:
        cfg = AppConfig.model_validate(tomllib.load(f))

    # Тестовый фрагмент (12 секунд)
    h = Highlight(start=1171.48, end=1183.48, reason="test", score=10)

    render_settings = RenderSettings(**cfg.render.model_dump())
    ffmpeg_task = FFmpegTask(cfg, settings=render_settings)

    out_video = ROOT / "output" / "test_latest.mp4"
    out_video.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Рендерим тестовый ролик в {render_settings.fps} FPS "
        f"(соотношение {render_settings.top_parts}:{render_settings.bottom_parts})..."
    )

    ffmpeg_task.run(
        top_video=ROOT / "input" / "top" / "south_park" / "video1.mkv",
        highlight=h,
        subtitles=ROOT / "output" / "test_preview.ass",
        out_path=out_video,
    )

    print(f"\nГотово! Файл сохранен:\n{out_video}")


if __name__ == "__main__":
    main()
