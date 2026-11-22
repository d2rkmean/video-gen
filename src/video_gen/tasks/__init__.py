from .base_task import BaseTask
from .best_parts_task import BestPartsTask
from .ffmpeg_task import FFmpegTask, RenderSettings
from .subtitle_task import SubtitleStyle, SubtitleTask
from .transcribe_task import TranscribeTask
from .youtube_task import YouTubeTask

try:
    from .film_worker import FilmCode, FilmWorker

    _film_worker_available = True
except ImportError:
    FilmCode = None  # type: ignore
    FilmWorker = None  # type: ignore
    _film_worker_available = False

# Псевдонимы для обратной совместимости
WhisperTask = TranscribeTask
GeminiTask = BestPartsTask

__all__ = (
    "BaseTask",
    "BestPartsTask",
    "FFmpegTask",
    "FilmCode",
    "FilmWorker",
    "GeminiTask",
    "RenderSettings",
    "SubtitleStyle",
    "SubtitleTask",
    "TranscribeTask",
    "WhisperTask",
    "YouTubeTask",
)
