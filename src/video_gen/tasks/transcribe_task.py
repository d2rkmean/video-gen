import io
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Any

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
from video_gen.utils.audio_utils import detect_preferred_audio_stream

_CACHE_VERSION = 2


class TranscribeTask(BaseTask):
    """Транскрибирует видео через faster-whisper (локально) или Groq/OpenAI (API)."""

    def __init__(self, config: AppConfig) -> None:
        super().__init__(config)
        self.backend = config.params.transcribe_backend  # "local" | "groq" | "openai"
        self.model_name = config.params.whisper_model
        self.logger = logger.bind(worker="TranscribeTask")

        if self.backend == "local":
            from faster_whisper import WhisperModel

            self.model = WhisperModel(
                self.model_name,
                device=config.params.whisper_device,
                compute_type=config.params.whisper_compute_type,
            )

    def run(
        self,
        video_path: Path,
        beam_size: int = 5,
        language: str = "ru",
        use_cache: bool = True,
        progress_cb: Any = None,
    ) -> list[Segment]:
        if self.backend == "local":
            return self._run_local(video_path, beam_size, language, use_cache, progress_cb)
        else:
            return self._run_api(video_path, language, use_cache, progress_cb)

    def _run_local(
        self,
        video_path: Path,
        beam_size: int = 5,
        language: str = "ru",
        use_cache: bool = True,
        progress_cb: Any = None,
    ) -> list[Segment]:
        audio_idx = detect_preferred_audio_stream(video_path, language)
        cache_path = video_path.with_suffix(".segments.json")
        meta = {
            "version": _CACHE_VERSION,
            "model": self.model_name,
            "language": language,
            "beam_size": beam_size,
            "audio_track": audio_idx,
        }

        if use_cache:
            cached = self._load(cache_path, meta)
            if cached is not None:
                self.logger.info(f"Использован кэш: {cache_path}")
                if progress_cb:
                    progress_cb(1.0, 1.0)
                return cached

        self.logger.info(f"Генерация субтитров для... {video_path}")
        audio_input: str | io.BytesIO = str(video_path)
        if audio_idx > 0:
            self.logger.info(
                f"Транскрибирую русскую аудиодорожку #{audio_idx} из {video_path.name}"
            )
            cmd = [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(video_path),
                "-map",
                f"0:a:{audio_idx}",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-f",
                "wav",
                "-",
            ]
            proc = subprocess.run(cmd, capture_output=True, check=True)
            audio_input = io.BytesIO(proc.stdout)

        segments_iter, info = self.model.transcribe(
            audio_input,
            beam_size=beam_size,
            language=language,
            vad_filter=True,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=2.0,
        )
        self.logger.info(f"Длительность: {info.duration:.1f} с")

        segments: list[Segment] = []
        if progress_cb:
            for s in segments_iter:
                segments.append(
                    Segment(
                        start=s.start,
                        end=s.end,
                        text=s.text.strip(),
                        words=[
                            Word(w.start, w.end, w.word.strip(), w.probability)
                            for w in (s.words or [])
                        ],
                        avg_logprob=s.avg_logprob,
                        no_speech_prob=s.no_speech_prob,
                        compression_ratio=s.compression_ratio,
                    )
                )
                progress_cb(min(s.end, info.duration), info.duration)
            progress_cb(info.duration, info.duration)
        else:
            with Progress(
                TextColumn("[bold cyan]Transcribe[/] {task.fields[name]}"),
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
                            words=[
                                Word(w.start, w.end, w.word.strip(), w.probability)
                                for w in (s.words or [])
                            ],
                            avg_logprob=s.avg_logprob,
                            no_speech_prob=s.no_speech_prob,
                            compression_ratio=s.compression_ratio,
                        )
                    )
                    progress.update(task_id, completed=min(s.end, info.duration))
                progress.update(task_id, completed=info.duration)

        self.logger.info(f"Обработка готова, сегментов: {len(segments)}")
        self._save(cache_path, segments, meta)
        return segments

    def _run_api(
        self,
        video_path: Path,
        language: str,
        use_cache: bool,
        progress_cb: Any = None,
    ) -> list[Segment]:
        from openai import OpenAI

        cache_path = video_path.with_suffix(".segments.api.json")
        meta = {
            "version": _CACHE_VERSION,
            "backend": self.backend,
            "model": self.config.params.transcribe_model,
            "language": language,
        }

        if use_cache:
            cached = self._load(cache_path, meta)
            if cached is not None:
                self.logger.info(f"Использован кэш: {cache_path}")
                if progress_cb:
                    progress_cb(1.0, 1.0)
                return cached

        audio_idx = detect_preferred_audio_stream(video_path, language)
        cmd = ["ffmpeg", "-v", "error", "-i", str(video_path)]
        if audio_idx > 0:
            cmd.extend(["-map", f"0:a:{audio_idx}"])
        cmd.extend(["-ac", "1", "-ar", "16000", "-f", "wav", "-"])
        proc = subprocess.run(cmd, capture_output=True, check=True)
        audio_bytes = proc.stdout

        api_key = self.config.params.transcribe_api_key or os.environ.get(
            "GROQ_API_KEY" if self.backend == "groq" else "OPENAI_API_KEY", ""
        )
        base_url = self.config.params.transcribe_api_base or (
            "https://api.groq.com/openai/v1" if self.backend == "groq" else None
        )
        client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )

        response = client.audio.transcriptions.create(
            model=self.config.params.transcribe_model,
            file=("audio.wav", audio_bytes, "audio/wav"),
            language=language,
            response_format="verbose_json",
            timestamp_granularities=["word", "segment"],
        )

        segments: list[Segment] = []
        resp_segments = getattr(response, "segments", None) or []
        for s in resp_segments:
            start = float(
                s.get("start", 0.0) if isinstance(s, dict) else (getattr(s, "start", 0.0) or 0.0)
            )
            end = float(
                s.get("end", 0.0) if isinstance(s, dict) else (getattr(s, "end", 0.0) or 0.0)
            )
            raw_text = s.get("text") if isinstance(s, dict) else getattr(s, "text", "")
            text = str(raw_text or "").strip()
            words_raw = (s.get("words") if isinstance(s, dict) else getattr(s, "words", None)) or []
            words = []
            for w in words_raw:
                w_start = float(
                    w.get("start", 0.0)
                    if isinstance(w, dict)
                    else (getattr(w, "start", 0.0) or 0.0)
                )
                w_end = float(
                    w.get("end", 0.0) if isinstance(w, dict) else (getattr(w, "end", 0.0) or 0.0)
                )
                raw_w_word = w.get("word") if isinstance(w, dict) else getattr(w, "word", "")
                w_word = str(raw_w_word or "").strip()
                words.append(Word(w_start, w_end, w_word, 1.0))
            avg_logprob = float(
                s.get("avg_logprob", 0.0)
                if isinstance(s, dict)
                else (getattr(s, "avg_logprob", 0.0) or 0.0)
            )
            no_speech_prob = float(
                s.get("no_speech_prob", 0.0)
                if isinstance(s, dict)
                else (getattr(s, "no_speech_prob", 0.0) or 0.0)
            )
            compression_ratio = float(
                s.get("compression_ratio", 1.0)
                if isinstance(s, dict)
                else (getattr(s, "compression_ratio", 1.0) or 1.0)
            )
            segments.append(
                Segment(
                    start=start,
                    end=end,
                    text=text,
                    words=words,
                    avg_logprob=avg_logprob,
                    no_speech_prob=no_speech_prob,
                    compression_ratio=compression_ratio,
                )
            )

        if not segments and getattr(response, "text", None):
            segments.append(Segment(start=0.0, end=0.0, text=response.text.strip(), words=[]))

        if use_cache:
            self._save(cache_path, segments, meta)
        if progress_cb:
            progress_cb(1.0, 1.0)
        return segments

    @staticmethod
    def _save(path: Path, segments: list[Segment], meta: dict[str, Any]) -> None:
        tmp = path.with_suffix(".tmp")
        payload = {"meta": meta, "segments": [asdict(s) for s in segments]}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)

    def _load(self, path: Path, meta: dict[str, Any]) -> list[Segment] | None:
        if not path.exists():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict) or raw.get("meta") != meta:
                self.logger.info(f"Кэш устарел или создан с другими параметрами: {path}")
                return None
            return [
                Segment(
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    words=[Word(**w) for w in s["words"]],
                    avg_logprob=s.get("avg_logprob", 0.0),
                    no_speech_prob=s.get("no_speech_prob", 0.0),
                    compression_ratio=s.get("compression_ratio", 1.0),
                )
                for s in raw["segments"]
            ]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            self.logger.warning(f"Кэш повреждён ({e}), расшифровываю заново: {path}")
            return None
