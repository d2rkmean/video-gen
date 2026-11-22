"""Асинхронный конвейер video-gen (Transcribe -> Analyze -> FFmpeg).

Запуск: ``python -m video_gen [config.json] [--json-progress]`` или ``uv run video-gen``.
"""

import os

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import asyncio
import contextlib
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import ValidationError
from rich.console import Console

from video_gen.callbacks import (
    JsonProgressCallbacks,
    NoopPipelineCallbacks,
    PipelineCallbacks,
)
from video_gen.config import AppConfig, VideoSource, get_effective_config
from video_gen.models import Highlight, Segment, VideoJob
from video_gen.tasks import (
    BestPartsTask,
    FFmpegTask,
    FilmCode,
    FilmWorker,
    SubtitleStyle,
    SubtitleTask,
    TranscribeTask,
    YouTubeTask,
)
from video_gen.utils.cache import (
    GeminiCache,
    RenderCache,
    WhisperCache,
    compute_data_hash,
    compute_file_hash,
)
from video_gen.utils.setup_logger import setup_logger
from video_gen.utils.speech_splitter import split_by_speech_pauses

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


def make_subtitle_task(cfg: AppConfig) -> SubtitleTask:
    return SubtitleTask(
        cfg,
        style=SubtitleStyle(
            font=cfg.subtitles.font,
            font_size=cfg.subtitles.font_size,
            bold=cfg.subtitles.bold,
            italic=cfg.subtitles.italic,
            spoken_color=cfg.subtitles.spoken_color,
            unspoken_color=cfg.subtitles.unspoken_color,
            outline_color=cfg.subtitles.outline_color,
            outline=cfg.subtitles.outline,
            shadow=cfg.subtitles.shadow,
            margin_v=cfg.subtitles.margin_v,
            margin_h=cfg.subtitles.margin_h,
            uppercase=cfg.subtitles.uppercase,
            res_x=cfg.render.width,
            res_y=cfg.render.height,
            animation=cfg.subtitles.animation,
            pop_animation=cfg.subtitles.pop_animation,
            top_parts=cfg.render.top_parts,
            bottom_parts=cfg.render.bottom_parts,
            seam_offset=cfg.subtitles.seam_offset,
        ),
        max_words=cfg.subtitles.max_words,
        max_chars=cfg.subtitles.max_chars,
        max_gap=cfg.subtitles.max_gap,
        hold=cfg.subtitles.hold,
    )


def natural_sort_key(path: Path) -> list[int | str]:
    """Ключ для естественной сортировки файлов (video1, video2, ..., video10)."""
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


def iter_videos(
    top_videos_config: list[VideoSource] | AppConfig,
    global_config: AppConfig | None = None,
) -> list[VideoJob]:
    """Нормализует источники top_videos в список VideoJob с эффективными конфигами."""
    if isinstance(top_videos_config, AppConfig):
        cfg = top_videos_config
        sources = cfg.videos.top_videos
    elif global_config is not None:
        cfg = global_config
        sources = top_videos_config
    else:
        raise ValueError("Не передан глобальный конфиг для вычисления эффективных настроек")

    result: list[VideoJob] = []
    for idx, item in enumerate(sources, 1):
        effective_cfg = get_effective_config(cfg, item)
        set_num = item.source_number or idx
        caption_name = item.caption_name or item.caption
        season = item.season if item.season is not None else 1
        item_base_code = item.base_code or FilmCode.generate_base_code(5)

        if item.type == "youtube":
            is_film = item.is_film if item.is_film is not None else False
            yt_task = YouTubeTask(effective_cfg)
            try:
                downloaded_list = yt_task.download(item.source)
                for part_i, (f, meta) in enumerate(downloaded_list, 1):
                    resolved_name = caption_name or meta.get("title") or f.stem
                    ep = item.episode if item.episode is not None else (None if is_film else part_i)
                    video_code = FilmCode.build_video_code(
                        item_base_code, is_film=is_film, episode=ep
                    )
                    result.append(
                        VideoJob(
                            path=f,
                            config=effective_cfg,
                            set_idx=set_num,
                            part_idx=part_i,
                            source_idx=set_num,
                            metadata=meta,
                            caption_name=resolved_name,
                            caption=item.caption,
                            source_type="youtube",
                            source_number=set_num,
                            codename=item.codename,
                            is_film=is_film,
                            season=season,
                            episode=ep,
                            base_code=item_base_code,
                            video_code=video_code,
                        )
                    )
            except Exception as e:
                logger.error(f"[YOUTUBE] Ошибка при обработке источника {item.source}: {e}")
        elif item.type == "video":
            p = Path(item.source)
            if p.is_dir():
                is_film = item.is_film if item.is_film is not None else False
                files = sorted(
                    (x for x in p.iterdir() if x.suffix.lower() in VIDEO_EXT),
                    key=natural_sort_key,
                )
                if not files:
                    files = sorted(
                        (x for x in p.rglob("*") if x.suffix.lower() in VIDEO_EXT),
                        key=natural_sort_key,
                    )
                for part_i, f in enumerate(files, 1):
                    m = re.search(r"(\d+)", f.stem)
                    actual_part = int(m.group(1)) if m else part_i
                    resolved_name = caption_name or f.stem
                    ep = (
                        item.episode
                        if item.episode is not None
                        else (None if is_film else actual_part)
                    )
                    video_code = FilmCode.build_video_code(
                        item_base_code, is_film=is_film, episode=ep
                    )
                    result.append(
                        VideoJob(
                            path=f,
                            config=effective_cfg,
                            set_idx=set_num,
                            part_idx=actual_part,
                            source_idx=set_num,
                            caption_name=resolved_name,
                            caption=item.caption,
                            source_type="video",
                            source_number=set_num,
                            codename=item.codename,
                            is_film=is_film,
                            season=season,
                            episode=ep,
                            base_code=item_base_code,
                            video_code=video_code,
                        )
                    )
            elif p.is_file():
                is_film = item.is_film if item.is_film is not None else True
                resolved_name = caption_name or p.stem
                ep = item.episode if item.episode is not None else (None if is_film else 1)
                video_code = FilmCode.build_video_code(item_base_code, is_film=is_film, episode=ep)
                result.append(
                    VideoJob(
                        path=p,
                        config=effective_cfg,
                        set_idx=set_num,
                        part_idx=1,
                        source_idx=set_num,
                        caption_name=resolved_name,
                        caption=item.caption,
                        source_type="video",
                        source_number=set_num,
                        codename=item.codename,
                        is_film=is_film,
                        season=season,
                        episode=ep,
                        base_code=item_base_code,
                        video_code=video_code,
                    )
                )
            else:
                logger.warning(f"Пропускаю, пути не существует: {p}")
        else:
            logger.warning(f"Пропускаю top_video с неподдерживаемым типом: {item.type}")
    return result


async def upload_full_worker(
    queue_in: asyncio.Queue[VideoJob | None],
    queue_out: asyncio.Queue[VideoJob | None],
    film_worker: FilmWorker | None,
    _callbacks: PipelineCallbacks | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Асинхронный воркер Этапа 1: каталогизация в базе данных PostgreSQL."""
    while True:
        if stop_event is not None and stop_event.is_set():
            await queue_out.put(None)
            break

        try:
            job = await asyncio.wait_for(queue_in.get(), timeout=0.5)
        except TimeoutError:
            continue

        if job is None:
            await queue_out.put(None)
            queue_in.task_done()
            break

        if stop_event is not None and stop_event.is_set():
            queue_in.task_done()
            await queue_out.put(None)
            break

        video = job.path
        if film_worker is not None:
            logger.info(f"[STAGE 1] Каталогизация видео для {video.name} (код {job.video_code})...")
            try:
                await film_worker.ensure_full_video_uploaded(job)
                logger.info(f"[STAGE 1] Видео {job.video_code} зарегистрировано в базе данных")
            except Exception as e:
                logger.exception(
                    f"[STAGE 1] Ошибка регистрации в базе данных для {video.name}: {e}"
                )

        # Передаем дальше на транскрипцию (Этап 2)
        await queue_out.put(job)
        queue_in.task_done()


async def transcribe_worker(
    queue_in: asyncio.Queue[VideoJob | None],
    queue_out: asyncio.Queue[tuple[VideoJob, list[Segment]] | None],
    transcribe: TranscribeTask,
    callbacks: PipelineCallbacks,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Асинхронный воркер транскрипции с кэшированием."""
    whisper_cache = WhisperCache()

    while True:
        if stop_event is not None and stop_event.is_set():
            await queue_out.put(None)
            break

        try:
            job = await asyncio.wait_for(queue_in.get(), timeout=0.5)
        except TimeoutError:
            continue

        if job is None:
            await queue_out.put(None)
            queue_in.task_done()
            break

        if stop_event is not None and stop_event.is_set():
            queue_in.task_done()
            await queue_out.put(None)
            break

        video = job.path
        cfg = job.config
        callbacks.on_transcribe_start(video.name)

        v_name = video.name

        def on_transcribe_prog(cur: float, total: float, _v: str = v_name) -> None:
            callbacks.on_transcribe_progress(_v, cur, total)

        try:
            file_hash = compute_file_hash(video)
            cache_key = whisper_cache.get_key(
                file_hash,
                cfg.params.whisper_model,
                cfg.params.language,
                cfg.params.beam_size,
                0,
            )
            use_cache = cfg.params.use_whisper_cache
            cached_segments = whisper_cache.load(cache_key) if use_cache else None
            if cached_segments is not None:
                logger.info(
                    f"[CACHE HIT] [TRANSCRIBE] Загружена расшифровка для {video.name} ({len(cached_segments)} сегментов)"
                )
                on_transcribe_prog(1.0, 1.0)
                await queue_out.put((job, cached_segments))
            else:
                if not use_cache:
                    logger.info(
                        f"[CACHE DISABLED] [TRANSCRIBE] Запуск распознавания для {video.name}..."
                    )
                else:
                    logger.info(
                        f"[CACHE MISS] [TRANSCRIBE] Запуск распознавания для {video.name}..."
                    )
                segments = await asyncio.to_thread(
                    transcribe.run,
                    video,
                    beam_size=cfg.params.beam_size,
                    language=cfg.params.language,
                    use_cache=use_cache,
                    progress_cb=on_transcribe_prog,
                )
                if segments:
                    if use_cache:
                        whisper_cache.save(
                            cache_key,
                            segments,
                            {"file_hash": file_hash, "video": video.name},
                        )
                    await queue_out.put((job, segments))
                else:
                    logger.warning(f"В видео {video.name} не найдено речи, пропускаю")
                    callbacks.on_video_done(video.name, broken=True)
        except Exception:
            logger.exception(f"Ошибка транскрипции {video.name}")
            callbacks.on_video_done(video.name, broken=True)

        callbacks.on_transcribe_done(video.name)
        queue_in.task_done()


async def analyze_worker(
    queue_in: asyncio.Queue[tuple[VideoJob, list[Segment]] | None],
    queue_out: asyncio.Queue[tuple[VideoJob, Highlight, Path, Path, int, int, str] | None],
    analyze: BestPartsTask,
    callbacks: PipelineCallbacks,
    stats: dict[str, int],
    stop_event: asyncio.Event | None = None,
    film_worker: FilmWorker | None = None,
    *,
    render_workers: int = 1,
    job_progress: dict[str, dict[str, int]] | None = None,
) -> None:
    gemini_cache = GeminiCache()

    async def _finish_queue() -> None:
        for _ in range(max(1, render_workers)):
            await queue_out.put(None)

    while True:
        if stop_event is not None and stop_event.is_set():
            await _finish_queue()
            break

        try:
            item = await asyncio.wait_for(queue_in.get(), timeout=0.5)
        except TimeoutError:
            continue

        if item is None:
            await _finish_queue()
            queue_in.task_done()
            break

        if stop_event is not None and stop_event.is_set():
            queue_in.task_done()
            await _finish_queue()
            break

        job, segments = item
        video = job.path
        cfg = job.config
        callbacks.on_analyze_start(video.name)

        try:
            sponsor_intervals = job.metadata.get("sponsor_intervals")
            mode = getattr(cfg.params, "mode", "highlights")

            if mode == "speech_split":
                logger.info(
                    f"[SPEECH_SPLIT] Нарезка по паузам в речи для {video.name} "
                    f"(длительность {cfg.params.min_part_duration}-{cfg.params.part_duration} с)..."
                )
                highlights = split_by_speech_pauses(
                    segments,
                    target_duration=float(cfg.params.part_duration),
                    min_duration=float(cfg.params.min_part_duration),
                    max_duration=float(cfg.params.part_duration),
                    sponsor_intervals=sponsor_intervals,
                    title_prefix="Часть",
                    video_title=job.caption_name or video.stem,
                )
            else:
                model_used = cfg.params.analysis_model or cfg.params.gemini_model
                segments_hash = compute_data_hash([s.text for s in segments])
                cache_key = gemini_cache.get_key(
                    segments_hash,
                    model_used,
                    cfg.params.parts_count,
                    cfg.params.min_part_duration,
                    cfg.params.part_duration,
                )
                use_gemini = cfg.params.use_gemini_cache
                cached_highlights = gemini_cache.load(cache_key) if use_gemini else None

                if cached_highlights is not None:
                    logger.info(
                        f"[CACHE HIT] [ANALYZE] Использован кэш анализа для {video.name} ({len(cached_highlights)} клипов)"
                    )
                    highlights = cached_highlights
                else:
                    cache_tag = "[CACHE MISS]" if use_gemini else "[CACHE DISABLED]"
                    logger.info(
                        f"{cache_tag} [ANALYZE] Анализ {video.name} через {model_used} "
                        f"(длительность {cfg.params.min_part_duration}-{cfg.params.part_duration} с, "
                        f"макс. клипов: {cfg.params.parts_count})..."
                    )
                    highlights = await asyncio.to_thread(
                        analyze.run,
                        segments,
                        max_clips=cfg.params.parts_count,
                        min_duration=cfg.params.min_part_duration,
                        max_duration=cfg.params.part_duration,
                        sponsor_intervals=sponsor_intervals,
                    )
                    if highlights and use_gemini:
                        gemini_cache.save(
                            cache_key,
                            highlights,
                            {"segments_hash": segments_hash, "video": video.name},
                        )

            if not highlights:
                logger.warning(f"Не удалось получить клипы для {video.name} (режим: {mode})")
                callbacks.on_analyze_done(video.stem, 0, [])
                callbacks.on_video_done(video.name, broken=True)
                stats["broken_videos"] += 1
            else:
                for i, h in enumerate(highlights, 1):
                    h.clip_idx = i
                    h.code = FilmCode.build_clip_code(
                        base_code=job.base_code or "00000",
                        clip_idx=i,
                        is_film=job.is_film,
                        episode=job.episode,
                    )

                if film_worker is not None:
                    try:
                        await film_worker.process_job_and_highlights(job, highlights)
                    except Exception as e:
                        logger.warning(f"[FILM_WORKER] Ошибка сохранения клипов в базу данных: {e}")

                out_dir = Path(cfg.params.output_dir)
                out_dir.mkdir(parents=True, exist_ok=True)

                clip_names = [
                    f"video{job.set_idx:02d}{job.part_idx:02d}{i:02d}.mp4"
                    for i in range(1, len(highlights) + 1)
                ]
                callbacks.on_analyze_done(video.stem, len(highlights), clip_names)

                sub_task = make_subtitle_task(cfg)
                top_hash = compute_file_hash(video)

                if job_progress is not None:
                    job_progress[video.stem] = {
                        "total": len(highlights),
                        "done": 0,
                        "ok": 0,
                        "failed": 0,
                    }

                queued_any = False
                for i, h in enumerate(highlights, 1):
                    if stop_event is not None and stop_event.is_set():
                        break

                    out_mp4 = out_dir / f"video{job.set_idx:02d}{job.part_idx:02d}{i:02d}.mp4"
                    ass_path = out_dir / f"video{job.set_idx:02d}{job.part_idx:02d}{i:02d}.ass"

                    render_hash = RenderCache.get_key(
                        top_hash,
                        h.start,
                        h.end,
                        cfg.render.model_dump(),
                        cfg.subtitles.model_dump(),
                    )

                    use_render = cfg.params.use_render_cache
                    if use_render and RenderCache.is_cached(out_mp4, render_hash):
                        logger.info(
                            f"[CACHE HIT] [RENDER] Клип уже готов: {out_mp4.name} (хэш совпадает)"
                        )
                        stats["ok"] += 1
                        callbacks.on_render_done(out_mp4.name, success=True)
                        if job_progress is not None:
                            jp = job_progress[video.stem]
                            jp["done"] += 1
                            jp["ok"] += 1
                        continue

                    ass = sub_task.run(segments, h, ass_path)
                    queued_any = True
                    await queue_out.put((job, h, ass, out_mp4, i, len(highlights), render_hash))

                if not queued_any and (stop_event is None or not stop_event.is_set()):
                    callbacks.on_video_done(video.stem, broken=False)
        except Exception:
            logger.exception(f"Ошибка AI-анализа для {video.name}")
            callbacks.on_analyze_done(video.stem, 0, [])
            callbacks.on_video_done(video.name, broken=True)
            stats["broken_videos"] += 1

        queue_in.task_done()


async def render_worker(
    queue_in: asyncio.Queue[tuple[VideoJob, Highlight, Path, Path, int, int, str] | None],
    callbacks: PipelineCallbacks,
    stats: dict[str, int],
    stop_event: asyncio.Event | None = None,
    *,
    job_progress: dict[str, dict[str, int]] | None = None,
) -> None:
    while True:
        if stop_event is not None and stop_event.is_set():
            break

        try:
            item = await asyncio.wait_for(queue_in.get(), timeout=0.5)
        except TimeoutError:
            continue

        if item is None:
            queue_in.task_done()
            break

        if stop_event is not None and stop_event.is_set():
            queue_in.task_done()
            break

        job, highlight, ass, out_mp4, idx, total, render_hash = item
        video = job.path
        cfg = job.config
        task = FFmpegTask(cfg)

        callbacks.on_render_start(video.stem, out_mp4.name, idx, total)

        out_name = out_mp4.name

        def on_ffmpeg_prog(pct: float, _name: str = out_name) -> None:
            callbacks.on_render_progress(_name, pct)

        logger.info(
            f"[RENDER] Клип {idx}/{total}: {highlight.start:.1f}-{highlight.end:.1f} "
            f"[{highlight.score}] {highlight.viral_title or highlight.reason}"
        )

        success = True
        try:
            await asyncio.to_thread(
                task.run,
                video,
                highlight,
                ass,
                out_mp4,
                on_ffmpeg_prog,
            )
            meta_path = out_mp4.with_suffix(".json")
            meta_dict = {
                **asdict(highlight),
                "render_hash": render_hash,
                "set_idx": job.set_idx,
                "source_idx": job.set_idx,
                "source_part_idx": job.part_idx,
                "part_idx": idx,
                "base_code": job.base_code,
                "is_film": job.is_film,
                "episode": job.episode,
            }
            meta_path.write_text(
                json.dumps(meta_dict, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            stats["ok"] += 1
        except Exception:
            logger.exception(f"Клип {idx} не собрался")
            out_mp4.unlink(missing_ok=True)
            stats["failed"] += 1
            success = False

        callbacks.on_render_done(out_mp4.name, success=success)
        if job_progress is not None and video.stem in job_progress:
            jp = job_progress[video.stem]
            jp["done"] += 1
            if success:
                jp["ok"] += 1
            else:
                jp["failed"] += 1
            if jp["done"] >= jp["total"]:
                callbacks.on_video_done(video.stem, broken=(jp["ok"] == 0 and jp["failed"] > 0))
        elif idx == total:
            callbacks.on_video_done(video.stem, broken=not success)
        queue_in.task_done()


async def run_pipeline(
    config: AppConfig,
    videos: Sequence[Path | VideoJob],
    callbacks: PipelineCallbacks | None = None,
    stop_event: asyncio.Event | None = None,
) -> dict[str, int]:
    """Запускает асинхронный конвейер (DB -> Transcribe -> Analyze -> Render)."""
    stats = {"ok": 0, "failed": 0, "broken_videos": 0}
    job_progress: dict[str, dict[str, int]] = {}
    render_concurrency = max(1, int(config.params.concurrent_renders))

    video_jobs = [v if isinstance(v, VideoJob) else VideoJob(path=v, config=config) for v in videos]

    transcribe = TranscribeTask(config)
    analyze = BestPartsTask(config)

    film_worker: FilmWorker | None = None
    if getattr(config, "database", None) and config.database.enabled:
        film_worker = FilmWorker(config)
        try:
            await film_worker.init_pool()
            await film_worker.init_schema()
        except Exception as e:
            logger.warning(
                f"[FILM_WORKER] Не удалось подключиться к {config.database.database} ({e}), продолжаем без БД"
            )
            film_worker = None

    q_upload_full: asyncio.Queue[VideoJob | None] = asyncio.Queue()
    q_transcribe: asyncio.Queue[VideoJob | None] = asyncio.Queue()
    q_analyze: asyncio.Queue[tuple[VideoJob, list[Segment]] | None] = asyncio.Queue()
    q_render: asyncio.Queue[tuple[VideoJob, Highlight, Path, Path, int, int, str] | None] = (
        asyncio.Queue()
    )

    for job in video_jobs:
        await q_upload_full.put(job)
    await q_upload_full.put(None)

    job_names = [job.name for job in video_jobs]

    async def _run_tasks(cb: PipelineCallbacks) -> None:
        u_worker = asyncio.create_task(
            upload_full_worker(
                q_upload_full,
                q_transcribe,
                film_worker,
                cb,
                stop_event,
            )
        )
        t_worker = asyncio.create_task(
            transcribe_worker(q_transcribe, q_analyze, transcribe, cb, stop_event)
        )
        a_worker = asyncio.create_task(
            analyze_worker(
                q_analyze,
                q_render,
                analyze,
                cb,
                stats,
                stop_event,
                film_worker,
                render_workers=render_concurrency,
                job_progress=job_progress,
            )
        )
        r_workers = [
            asyncio.create_task(
                render_worker(q_render, cb, stats, stop_event, job_progress=job_progress)
            )
            for _ in range(render_concurrency)
        ]

        await asyncio.gather(u_worker, t_worker, a_worker, *r_workers)

    try:
        cb = callbacks or NoopPipelineCallbacks()
        cb.on_pipeline_started(len(video_jobs), job_names)

        def _log_sink(message: Any) -> None:
            with contextlib.suppress(Exception):
                rec = message.record
                cb.on_log(rec["level"].name, rec["message"].strip())

        sink_id = logger.add(_log_sink, level="INFO", enqueue=False)
        try:
            await _run_tasks(cb)
        finally:
            with contextlib.suppress(Exception):
                logger.remove(sink_id)

        cb.on_pipeline_finished(stats)
    finally:
        if film_worker:
            await film_worker.close()

    return stats


def _print_help() -> None:
    print("Использование: video-gen [config.json] [--json-progress]")
    print("  config.json      Путь к файлу конфигурации (по умолчанию: config.json)")
    print("  --json-progress  Вывод событий прогресса в формате JSON Lines (или VIDEOGEN_JSON_PROGRESS=1)")
    print("  -h, --help       Показать эту справку")


def _parse_cli_args(raw_args: list[str]) -> tuple[str | None, bool, int | None]:
    if "-h" in raw_args or "--help" in raw_args:
        _print_help()
        return None, False, 0

    use_json = "--json-progress" in raw_args or os.environ.get("VIDEOGEN_JSON_PROGRESS") == "1"
    positional: list[str] = []
    for arg in raw_args:
        if arg == "--json-progress":
            continue
        if arg.startswith("-"):
            print(f"Неизвестный флаг: {arg}", file=sys.stderr)
            _print_help()
            return None, False, 2
        positional.append(arg)

    if len(positional) > 1:
        print(f"Слишком много аргументов: {' '.join(positional)}", file=sys.stderr)
        return None, False, 2

    return (positional[0] if positional else "config.json"), use_json, None


def main() -> int:
    config_path, use_json, exit_code = _parse_cli_args(sys.argv[1:])
    if exit_code is not None or config_path is None:
        return exit_code or 0

    if not use_json:
        setup_logger(out_console=Console(), console_level="INFO")
    else:
        setup_logger(console_level="ERROR")

    try:
        config = AppConfig.from_json(config_path)
    except (FileNotFoundError, json.JSONDecodeError, ValidationError, ValueError) as exc:
        msg = (
            f"Файл конфигурации не найден: {config_path}"
            if isinstance(exc, FileNotFoundError)
            else f"Ошибка конфигурации ({config_path}): {exc}"
        )
        print(msg, file=sys.stderr)
        return 2

    videos = iter_videos(config)
    if not videos:
        logger.error("Нет видео для обработки (config.videos.top_videos)")
        return 1

    out_root = Path(config.params.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    logger.info(f"Видео в очереди: {len(videos)}")

    try:
        callbacks = JsonProgressCallbacks() if use_json else None
        stats = asyncio.run(run_pipeline(config, videos, callbacks=callbacks))
    except KeyboardInterrupt:
        logger.warning("Остановлено пользователем (Ctrl+C)")
        return 130
    except Exception:
        logger.exception("Критическая ошибка конвейера")
        return 1

    logger.info(
        f"Итог: клипов готово {stats['ok']}, упало {stats['failed']}, видео с ошибкой {stats['broken_videos']}"
    )
    return 0 if (stats["ok"] or not (stats["failed"] or stats["broken_videos"])) else 1


if __name__ == "__main__":
    sys.exit(main())
