"""Интерфейс и реализации колбэков жизненного цикла конвейера video-gen."""

import json
import threading
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PipelineCallbacks(Protocol):
    def on_pipeline_started(self, total_jobs: int, job_names: list[str]) -> None: ...
    def on_transcribe_start(self, job_name: str) -> None: ...
    def on_transcribe_progress(self, job_name: str, current: float, total: float) -> None: ...
    def on_transcribe_done(self, job_name: str) -> None: ...
    def on_analyze_start(self, job_name: str) -> None: ...
    def on_analyze_done(self, job_name: str, clips_count: int, clip_titles: list[str]) -> None: ...
    def on_render_start(
        self, job_name: str, clip_name: str, clip_idx: int, total_clips: int
    ) -> None: ...
    def on_render_progress(self, clip_name: str, pct: float) -> None: ...
    def on_render_done(self, clip_name: str, success: bool) -> None: ...
    def on_video_done(self, job_name: str, broken: bool) -> None: ...
    def on_log(self, level: str, message: str) -> None: ...
    def on_error(self, job_name: str | None, error: Exception) -> None: ...
    def on_pipeline_finished(self, stats: dict[str, int]) -> None: ...


class NoopPipelineCallbacks:
    def on_pipeline_started(self, total_jobs: int, job_names: list[str]) -> None:
        pass

    def on_transcribe_start(self, job_name: str) -> None:
        pass

    def on_transcribe_progress(self, job_name: str, current: float, total: float) -> None:
        pass

    def on_transcribe_done(self, job_name: str) -> None:
        pass

    def on_analyze_start(self, job_name: str) -> None:
        pass

    def on_analyze_done(self, job_name: str, clips_count: int, clip_titles: list[str]) -> None:
        pass

    def on_render_start(
        self, job_name: str, clip_name: str, clip_idx: int, total_clips: int
    ) -> None:
        pass

    def on_render_progress(self, clip_name: str, pct: float) -> None:
        pass

    def on_render_done(self, clip_name: str, success: bool) -> None:
        pass

    def on_video_done(self, job_name: str, broken: bool) -> None:
        pass

    def on_log(self, level: str, message: str) -> None:
        pass

    def on_error(self, job_name: str | None, error: Exception) -> None:
        pass

    def on_pipeline_finished(self, stats: dict[str, int]) -> None:
        pass


class JsonProgressCallbacks:
    """Потокобезопасный вывод JSON-событий в stdout для GUI/CLI-интеграций."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def _emit(self, event: str, **data: Any) -> None:
        line = json.dumps({"event": event, **data}, ensure_ascii=False)
        with self._lock:
            print(line, flush=True)

    def on_pipeline_started(self, total_jobs: int, job_names: list[str]) -> None:
        self._emit("pipeline_started", total=total_jobs, jobs=job_names)

    def on_transcribe_start(self, job_name: str) -> None:
        self._emit("transcribe_start", job=job_name)

    def on_transcribe_progress(self, job_name: str, current: float, total: float) -> None:
        self._emit(
            "transcribe_progress",
            job=job_name,
            current=round(current, 2),
            total=round(total, 2),
            pct=round(current / total * 100, 1) if total else 0,
        )

    def on_transcribe_done(self, job_name: str) -> None:
        self._emit("transcribe_done", job=job_name)

    def on_analyze_start(self, job_name: str) -> None:
        self._emit("analyze_start", job=job_name)

    def on_analyze_done(self, job_name: str, clips_count: int, clip_titles: list[str]) -> None:
        self._emit("analyze_done", job=job_name, clips=clips_count, titles=clip_titles)

    def on_render_start(
        self, job_name: str, clip_name: str, clip_idx: int, total_clips: int
    ) -> None:
        self._emit("render_start", job=job_name, clip=clip_name, idx=clip_idx, total=total_clips)

    def on_render_progress(self, clip_name: str, pct: float) -> None:
        self._emit("render_progress", clip=clip_name, pct=round(pct, 1))

    def on_render_done(self, clip_name: str, success: bool) -> None:
        self._emit("render_done", clip=clip_name, success=success)

    def on_video_done(self, job_name: str, broken: bool) -> None:
        self._emit("video_done", job=job_name, broken=broken)

    def on_pipeline_finished(self, stats: dict[str, int]) -> None:
        self._emit("pipeline_finished", **stats)

    def on_log(self, level: str, message: str) -> None:
        self._emit("log", level=level, message=message)

    def on_error(self, job_name: str | None, error: Exception) -> None:
        self._emit("error", job=job_name, error=str(error))
