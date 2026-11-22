#!/usr/bin/env python3
"""
Скрипт ручной сборки отдельного вертикального шортса с запрещённой речью Кайла и Санты (S14E06 / 201).
Использует SubtitleTask, FFmpegTask, FilmWorker и TelegramTask из video-gen.
"""

import asyncio
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

# Добавляем путь к src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from video_gen.config import AppConfig
from video_gen.models import Highlight, Segment, Word
from video_gen.tasks.ffmpeg_task import FFmpegTask
from video_gen.tasks.film_worker import FilmWorker
from video_gen.tasks.subtitle_task import SubtitleTask
from video_gen.tasks.telegram_task import TelegramTask

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-7s] %(name)s: %(message)s",
    datefmt="%d.%m.%Y %H:%M:%S",
)
logger = logging.getLogger("bake_short")


async def main() -> None:
    work_dir = Path(__file__).resolve().parent.parent
    config_path = work_dir / "config.toml"
    cfg = AppConfig.from_toml(config_path)

    top_video = work_dir / "input/top/south_park/video22.mkv"
    segments_path = work_dir / "input/top/south_park/video22.segments.json"
    out_dir = work_dir / "output/south_park"
    out_dir.mkdir(parents=True, exist_ok=True)

    ass_path = out_dir / "video012203.ass"
    out_mp4 = out_dir / "video012203.mp4"
    out_json = out_dir / "video012203.json"

    logger.info("Загрузка сегментов из %s...", segments_path)
    with open(segments_path, encoding="utf-8") as f:
        data = json.load(f)

    segments = [
        Segment(
            start=s["start"],
            end=s["end"],
            text=s["text"],
            words=[
                Word(
                    start=w["start"],
                    end=w["end"],
                    text=w.get("text") or w.get("word") or "",
                    probability=w.get("probability", 1.0),
                )
                for w in s.get("words", [])
            ],
            avg_logprob=s.get("avg_logprob", -0.5),
            no_speech_prob=s.get("no_speech_prob", 0.0),
            compression_ratio=s.get("compression_ratio", 1.0),
        )
        for s in data["segments"]
    ]

    # Таймкоды речи Кайла, Иисуса и Санты: 1151.8 -> 1195.0 (~43.2 сек)
    highlight = Highlight(
        start=1151.8,
        end=1195.0,
        reason="Легендарная запрещённая речь Кайла, Иисуса и Санты о терроре и запугивании без цензуры",
        score=10,
        hook="Знаете, сегодня я многое понял...",
        viral_title="Та самая запрещённая речь Кайла и Санты!",
        description="Запрещённая мораль Кайла и Санты о терроризме и запугивании без цензуры #southpark #южныйпарк #террор #насилие",
        code="68707-22-3",
        clip_idx=3,
    )

    logger.info(
        "Генерация субтитров SubtitleTask: %s (таймкоды %.2f -> %.2f)...",
        ass_path.name,
        highlight.start,
        highlight.end,
    )
    sub_task = SubtitleTask(cfg)
    sub_task.run(segments, highlight, ass_path)

    logger.info("Сборка вертикального шортса через FFmpegTask (NVENC 60fps)...")
    ffmpeg_task = FFmpegTask(cfg)

    def on_prog(pct: float) -> None:
        logger.info("Прогресс рендера: %.1f%%", pct)

    await asyncio.to_thread(
        ffmpeg_task.run,
        top_video,
        highlight,
        ass_path,
        out_mp4,
        on_prog,
    )
    logger.info("Рендер успешно завершен: %s (%s)", out_mp4, f"{out_mp4.stat().st_size / (1024*1024):.1f} MB")

    # Инициализируем FilmWorker для регистрации в БД
    film_worker = FilmWorker(cfg)
    await film_worker.init_pool()

    try:
        logger.info("Регистрация нарезки 68707-22-3 в PostgreSQL...")
        # Регистрируем клип под кодом серии 68707-22
        reg_res = await film_worker.register_clip(
            episode_code="68707-22",
            clip_idx=3,
            timecode_start=highlight.start,
            timecode_end=highlight.end,
            duration=highlight.end - highlight.start,
            custom_code="68707-22-3",
            hook=highlight.hook,
            viral_title=highlight.viral_title,
            description=highlight.description,
            is_film=False,
            base_code="68707",
            episode=22,
        )
        logger.info("Клип зарегистрирован в БД: id=%s", reg_res.get("id"))

        # Выгрузка в Telegram
        logger.info("Выгрузка клипа в Telegram через TelegramTask...")
        tg_task = TelegramTask(cfg)
        tg_success = await tg_task.upload(
            out_mp4,
            highlight,
            source_idx=1,
            part_idx=3,
            source_part_idx=22,
            set_idx=1,
            thread_id=cfg.telegram.thread_id,
            caption_name="Южный Парк",
            format="json",
            codename="south_park",
            code="68707-22-3",
            film_code="68707",
            base_code="68707",
            is_film=False,
        )

        if tg_success and tg_task.last_uploaded_file_id:
            fid = tg_task.last_uploaded_file_id
            mid = tg_task.last_uploaded_message_id
            logger.info("Успешно загружено в Telegram: file_id=%s, msg_id=%s", fid, mid)
            highlight.tg_file_id = fid
            highlight.tg_message_id = mid

            await film_worker.update_clip_telegram(
                highlight.code,
                tg_file_id=fid,
                tg_message_id=mid,
            )
            logger.info("Данные Telegram обновлены в БД PostgreSQL!")
        else:
            logger.warning("Загрузка в Telegram завершилась со статусом %s", tg_success)

        await tg_task.close()
    finally:
        await film_worker.close()

    # Сохраняем метаданные JSON
    meta_dict = {
        **asdict(highlight),
        "set_idx": 1,
        "source_idx": 1,
        "source_part_idx": 22,
        "part_idx": 3,
        "base_code": "68707",
        "is_film": False,
        "episode": 22,
    }
    out_json.write_text(json.dumps(meta_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Метаданные сохранены в %s", out_json)


if __name__ == "__main__":
    asyncio.run(main())
