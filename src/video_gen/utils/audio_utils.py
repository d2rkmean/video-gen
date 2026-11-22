"""Утилиты для анализа и выбора аудиодорожек мультимедиа файлов."""

import json
import subprocess
from functools import lru_cache
from pathlib import Path

from loguru import logger


@lru_cache(maxsize=128)
def detect_preferred_audio_stream(
    video_path: Path | str,
    target_lang: str = "ru",
) -> int:
    """Определяет индекс подходящей аудиодорожки (0, 1, 2...), предпочитая target_lang.

    В многодорожечных контейнерах (.mkv, .mp4 торренты с несколькими озвучками)
    ищет дорожку с языковым тегом 'rus'/'ru' или ключевыми словами
    ('рус', 'rus', 'mtv', 'paramount', 'гоблин', 'кураж', 'дубляж', 'рыбов')
    в метаданных названия. Если подходящая дорожка найдена, возвращает её индекс (0, 1...).
    Если дорожка одна или совпадений нет, возвращает 0.
    """
    path_str = str(video_path)
    try:
        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=index:stream_tags=language,title",
                "-of",
                "json",
                path_str,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        streams = data.get("streams", [])
        if len(streams) <= 1:
            return 0

        target_lang = target_lang.lower()
        # 1. Поиск по тегу языка
        for i, s in enumerate(streams):
            tags = s.get("tags") or {}
            lang = tags.get("language", "").lower()
            if lang in (target_lang, "rus", "ru"):
                logger.info(
                    f"AudioUtils: выбрана русская аудиодорожка #{i} (tag language={lang}) в {Path(path_str).name}"
                )
                return i

        # 2. Поиск по ключевым словам в названии дорожки
        keywords = ("рус", "rus", "mtv", "paramount", "гоблин", "кураж", "дубляж", "рыбов")
        for i, s in enumerate(streams):
            tags = s.get("tags") or {}
            title = tags.get("title", "").lower()
            if any(kw in title for kw in keywords):
                logger.info(
                    f"AudioUtils: выбрана русская аудиодорожка #{i} (title={title!r}) в {Path(path_str).name}"
                )
                return i

    except Exception as e:  # noqa: BLE001
        logger.debug(f"AudioUtils: не удалось определить аудиодорожки для {path_str}: {e}")

    return 0
