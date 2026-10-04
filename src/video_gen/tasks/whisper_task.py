import json
from dataclasses import asdict
from pathlib import Path

from faster_whisper import WhisperModel
from loguru import logger
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from video_gen.config import AppConfig
from video_gen.models import Segment, Word
from video_gen.tasks.base_task import BaseTask


class WhisperTask(BaseTask):
    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.model = WhisperModel(
            config.params.whisper_model,
            device=config.params.whisper_device,
            compute_type=config.params.whisper_compute_type,
        )
        self.logger = logger.bind(worker="WhisperTask")

    def run(
        self,
        video_path: Path,
        beam_size: int = 5,
        language: str = "ru",
        use_cache: bool = True,
    ) -> list[Segment]:
        cache_path = video_path.with_suffix(".segments.json")

        if use_cache and cache_path.exists():
            self.logger.info(f"Использован кэш: {cache_path}")
            return self._load(cache_path)

        self.logger.info(f"Генерация субтитров для... {video_path}")
        segments_iter, info = self.model.transcribe(
            str(video_path),
            beam_size=beam_size,
            language=language,
            vad_filter=True,
            word_timestamps=True,
        )
        self.logger.info(f"Длительность: {info.duration:.1f} с")

        segments: list[Segment] = []
        with Progress(
            TextColumn("[bold cyan]Whisper[/] {task.fields[name]}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        ) as progress:
            task_id = progress.add_task("transcribe", total=info.duration, name=video_path.name)
            for s in segments_iter:
                segments.append(
                    Segment(
                        start=s.start,
                        end=s.end,
                        text=s.text.strip(),
                        words=[Word(w.start, w.end, w.word.strip()) for w in (s.words or [])],
                    )
                )
                progress.update(task_id, completed=min(s.end, info.duration))
            progress.update(task_id, completed=info.duration)

        self.logger.info(f"Обработка готова, сегментов: {len(segments)}")

        self._save(cache_path, segments)
        return segments

    @staticmethod
    def _save(path: Path, segments: list[Segment]) -> None:
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps([asdict(s) for s in segments], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(path)

    @staticmethod
    def _load(path: Path) -> list[Segment]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return [
            Segment(
                start=s["start"],
                end=s["end"],
                text=s["text"],
                words=[Word(**w) for w in s["words"]],
            )
            for s in raw
        ]
