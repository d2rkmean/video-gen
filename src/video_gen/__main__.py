"""Точка входа: top_videos -> Whisper -> Gemini -> субтитры -> FFmpeg.

Запуск: ``python -m src.video_gen``
"""

import sys
from pathlib import Path

from loguru import logger
from rich.console import Console

from video_gen.config import AppConfig
from video_gen.tasks import (
    FFmpegTask,
    GeminiTask,
    SubtitleStyle,
    SubtitleTask,
    WhisperTask,
)
from video_gen.utils.setup_logger import setup_logger

setup_logger()
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


console = Console()


def iter_videos(top_videos_config) -> list[Path]:
    """Нормализует ``config.videos.top_videos`` в список файлов."""
    result: list[Path] = []
    for item in top_videos_config:
        p = Path(item.source)
        if p.is_dir():
            result.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT))
        elif p.is_file():
            result.append(p)
        else:
            logger.warning(f"Пропускаю, пути не существует: {p}")
    return result


def process_video(
    video: Path,
    config: AppConfig,
    whisper: WhisperTask,
    gemini: GeminiTask,
    subtitles: SubtitleTask,
    ffmpeg: FFmpegTask,
) -> tuple[int, int]:
    """Прогоняет одно видео через весь пайплайн.

    Returns:
        Кортеж ``(успешных клипов, упавших клипов)``.

    Raises:
        Exception: Любая ошибка Whisper или Gemini (ловится выше, в ``main``).
    """
    out_dir = Path(config.params.output_dir) / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    segments = whisper.run(
        video, beam_size=config.params.beam_size, language=config.params.language
    )
    if not segments:
        logger.warning(f"Речи не найдено, пропускаю: {video.name}")
        return 0, 0

    highlights = gemini.run(
        segments,
        max_clips=config.params.parts_count,
        min_duration=config.params.min_part_duration,
        max_duration=config.params.part_duration,
    )
    if not highlights:
        logger.warning(f"Gemini не нашёл подходящих клипов: {video.name}")
        return 0, 0

    ok = failed = 0
    for i, h in enumerate(highlights, 1):
        out_mp4 = out_dir / f"clip_{i:02d}.mp4"
        if out_mp4.exists():
            logger.info(f"Уже готов, пропускаю: {out_mp4.name}")
            ok += 1
            continue

        logger.info(f"Клип {i}/{len(highlights)}: {h.start:.1f}-{h.end:.1f} [{h.score}] {h.reason}")
        try:
            ass = subtitles.run(segments, h, out_dir / f"clip_{i:02d}.ass")
            ffmpeg.run(video, h, ass, out_mp4)
            ok += 1
        except Exception:
            logger.exception(f"Клип {i} не собрался")
            out_mp4.unlink(missing_ok=True)  # не оставляем битый файл
            failed += 1
    return ok, failed


def main() -> int:
    """Запускает ферму. Возвращает код выхода процесса."""
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.toml"
    config = AppConfig.from_toml(config_path)

    videos = iter_videos(config.videos.top_videos)
    if not videos:
        logger.error("Нет видео для обработки (config.videos.top_videos)")
        return 1

    out_root = Path(config.params.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    # Тяжёлые объекты создаются один раз. Падение здесь (например, CUDA)
    # лучше получить сразу, до обработки.
    whisper = WhisperTask(config)
    gemini = GeminiTask(config)
    subtitles = SubtitleTask(
        config,
        style=SubtitleStyle(
            font=config.subtitles.font,
            font_size=config.subtitles.font_size,
            spoken_color=config.subtitles.spoken_color,
            unspoken_color=config.subtitles.unspoken_color,
            outline_color=config.subtitles.outline_color,
            outline=config.subtitles.outline,
            margin_v=config.subtitles.margin_v,
            margin_h=config.subtitles.margin_h,
            uppercase=config.subtitles.uppercase,
        ),
        max_words=config.subtitles.max_words,
        max_chars=config.subtitles.max_chars,
        max_gap=config.subtitles.max_gap,
        hold=config.subtitles.hold,
    )
    ffmpeg = FFmpegTask(config)

    total_ok = total_failed = broken_videos = 0
    logger.info(f"Видео в очереди: {len(videos)}")

    try:
        for n, video in enumerate(videos, 1):
            logger.info(f"[{n}/{len(videos)}] {video.name}")
            try:
                ok, failed = process_video(video, config, whisper, gemini, subtitles, ffmpeg)
            except Exception:
                logger.exception(f"Видео не обработано: {video}")
                broken_videos += 1
                continue
            total_ok += ok
            total_failed += failed
    except KeyboardInterrupt:
        logger.warning("Остановлено пользователем (Ctrl+C)")

    logger.info(
        f"Итог: клипов готово {total_ok}, упало {total_failed}, видео с ошибкой {broken_videos}"
    )
    return 0 if (total_ok or not (total_failed or broken_videos)) else 1


if __name__ == "__main__":
    sys.exit(main())
