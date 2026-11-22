"""Универсальная система кэширования для воркеров фермы (Whisper, Gemini, Render).

Каждый шаг конвейера проверяет наличие готового результата по хэшу входных данных
и настроек. Если хэш совпадает — шаг моментально пропускается без нагрузки на GPU и API.
"""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger

from video_gen.models import Highlight, Segment, Word

CHUNK_SIZE = 512 * 1024  # 512 KB
DEFAULT_CACHE_DIR = Path("cache")


def compute_file_hash(path: Path) -> str:
    """Вычисляет быстрый и надёжный хэш файла по размеру, времени модификации и сэмплам данных."""
    if not path.is_file():
        raise FileNotFoundError(f"Файл не найден для хэширования: {path}")

    stat = path.stat()
    size = stat.st_size
    mtime = stat.st_mtime

    hasher = hashlib.sha256()
    hasher.update(f"{size}_{mtime}_".encode())

    with path.open("rb") as f:
        if size <= 2 * 1024 * 1024:  # до 2 МБ читаем целиком
            hasher.update(f.read())
        else:
            # Читаем начало, середину и конец файла
            hasher.update(f.read(CHUNK_SIZE))
            f.seek(size // 2)
            hasher.update(f.read(CHUNK_SIZE))
            f.seek(max(0, size - CHUNK_SIZE))
            hasher.update(f.read(CHUNK_SIZE))

    return hasher.hexdigest()


def compute_data_hash(data: Any) -> str:
    """Вычисляет детерминированный SHA-256 хэш любого JSON-сериализуемого объекта."""
    raw = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class WhisperCache:
    """Кэш результатов распознавания Whisper."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR / "whisper") -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(
        self,
        file_hash: str,
        model: str,
        language: str,
        beam_size: int,
        audio_track: int,
    ) -> str:
        data = {
            "file_hash": file_hash,
            "model": model,
            "language": language,
            "beam_size": beam_size,
            "audio_track": audio_track,
        }
        return compute_data_hash(data)

    def load(self, cache_key: str) -> list[Segment] | None:
        path = self.cache_dir / f"{cache_key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            segments = [
                Segment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    words=[Word(**w) for w in s["words"]],
                    avg_logprob=s.get("avg_logprob", 0.0),
                    no_speech_prob=s.get("no_speech_prob", 0.0),
                    compression_ratio=s.get("compression_ratio", 1.0),
                )
                for s in payload["segments"]
            ]
            return segments
        except Exception as e:
            logger.warning(f"Ошибка чтения кэша Whisper ({path}): {e}")
            return None

    def save(self, cache_key: str, segments: list[Segment], meta: dict[str, Any]) -> Path:
        path = self.cache_dir / f"{cache_key}.json"
        tmp = path.with_suffix(".tmp")
        payload = {
            "meta": meta,
            "segments": [asdict(s) for s in segments],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


class GeminiCache:
    """Кэш анализа и выбора лучших клипов Gemini."""

    def __init__(self, cache_dir: Path = DEFAULT_CACHE_DIR / "gemini") -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(
        self,
        segments_hash: str,
        model: str,
        parts_count: int,
        min_duration: int,
        max_duration: int,
    ) -> str:
        data = {
            "segments_hash": segments_hash,
            "model": model,
            "parts_count": parts_count,
            "min_duration": min_duration,
            "max_duration": max_duration,
        }
        return compute_data_hash(data)

    def load(self, cache_key: str) -> list[Highlight] | None:
        path = self.cache_dir / f"{cache_key}.json"
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            highlights = [
                Highlight(
                    start=h["start"],
                    end=h["end"],
                    reason=h["reason"],
                    score=h["score"],
                    hook=h.get("hook", ""),
                    viral_title=h.get("viral_title", ""),
                    description=h.get("description", ""),
                    code=h.get("code"),
                    clip_idx=h.get("clip_idx"),
                )
                for h in payload["highlights"]
            ]
            return highlights
        except Exception as e:
            logger.warning(f"Ошибка чтения кэша Gemini ({path}): {e}")
            return None

    def save(self, cache_key: str, highlights: list[Highlight], meta: dict[str, Any]) -> Path:
        path = self.cache_dir / f"{cache_key}.json"
        tmp = path.with_suffix(".tmp")
        payload = {
            "meta": meta,
            "highlights": [asdict(h) for h in highlights],
        }
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return path


class RenderCache:
    """Проверка и сохранение кэша рендера видео."""

    @staticmethod
    def get_key(
        top_video_hash: str,
        highlight_start: float,
        highlight_end: float,
        render_settings_dict: dict[str, Any],
        subtitles_dict: dict[str, Any],
    ) -> str:
        data = {
            "top_video_hash": top_video_hash,
            "start": round(highlight_start, 3),
            "end": round(highlight_end, 3),
            "render_settings": render_settings_dict,
            "subtitles": subtitles_dict,
        }
        return compute_data_hash(data)

    @staticmethod
    def is_cached(out_mp4: Path, expected_hash: str) -> bool:
        if not out_mp4.is_file() or out_mp4.stat().st_size < 1024:
            return False
        meta_path = out_mp4.with_suffix(".json")
        if not meta_path.is_file():
            return False
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            return meta.get("render_hash") == expected_hash
        except Exception:
            return False
