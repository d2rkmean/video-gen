"""Загрузка видео из YouTube через yt-dlp с кэшированием и получением SponsorBlock."""

import json
from pathlib import Path
from typing import Any

import yt_dlp
from loguru import logger

from video_gen.tasks.base_task import BaseTask
from video_gen.utils.ad_filter import fetch_sponsorblock_segments

DEFAULT_YOUTUBE_CACHE = Path("cache/youtube")


class YouTubeTask(BaseTask):
    """Таск загрузки видео с YouTube с поддержкой одиночных видео и плейлистов."""

    def __init__(self, config: Any, cache_dir: Path = DEFAULT_YOUTUBE_CACHE) -> None:
        super().__init__(config)
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger.bind(worker="YouTubeTask")

    def download(self, url_or_id: str) -> list[tuple[Path, dict[str, Any]]]:
        """Скачивает видео или плейлист по ссылке/ID и возвращает список путей к mp4 и метаданных."""
        self.logger.info(f"[YOUTUBE] Обработка источника: {url_or_id}")

        ydl_opts: dict[str, Any] = {
            "format": "bestvideo[height<=1080]+bestaudio[language=ru]/bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[height<=1080]/best",
            "outtmpl": str(self.cache_dir / "%(id)s.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "merge_output_format": "mp4",
        }

        results: list[tuple[Path, dict[str, Any]]] = []

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url_or_id, download=False)
                if info is None:
                    self.logger.error(f"[YOUTUBE] Не удалось извлечь информацию: {url_or_id}")
                    return []

                entries = info.get("entries") if "entries" in info else [info]

                for entry in entries:
                    if not entry:
                        continue
                    video_id = entry.get("id")
                    title = entry.get("title", video_id)
                    duration = float(entry.get("duration", 0.0))
                    target_file = self.cache_dir / f"{video_id}.mp4"
                    meta_file = self.cache_dir / f"{video_id}.json"

                    if (
                        target_file.is_file()
                        and target_file.stat().st_size > 1024
                        and meta_file.is_file()
                    ):
                        try:
                            meta = json.loads(meta_file.read_text(encoding="utf-8"))
                            self.logger.info(
                                f"[CACHE HIT] [YOUTUBE] Видео уже скачано: {title} ({target_file.name})"
                            )
                            results.append((target_file, meta))
                            continue
                        except Exception:
                            pass

                    self.logger.info(f"[YOUTUBE] Скачивание: {title} ({video_id})...")
                    ydl.download(
                        [entry.get("webpage_url") or f"https://www.youtube.com/watch?v={video_id}"]
                    )

                    sponsor_intervals: list[tuple[float, float, str]] = []
                    if self.config.params.sponsorblock_enabled and video_id:
                        sponsor_intervals = fetch_sponsorblock_segments(
                            video_id,
                            categories=self.config.params.sponsorblock_categories,
                        )

                    meta = {
                        "id": video_id,
                        "title": title,
                        "duration": duration,
                        "url": entry.get("webpage_url") or url_or_id,
                        "sponsor_intervals": sponsor_intervals,
                    }
                    meta_file.write_text(
                        json.dumps(meta, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    self.logger.info(
                        f"[YOUTUBE] Успешно загружено: {target_file.name} (рекламных зон: {len(sponsor_intervals)})"
                    )
                    results.append((target_file, meta))

        except Exception:
            self.logger.exception(f"[YOUTUBE] Ошибка при загрузке {url_or_id}")
            raise

        return results
