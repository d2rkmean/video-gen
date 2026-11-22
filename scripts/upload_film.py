#!/usr/bin/env python3
"""Скрипт загрузки видео/полнометражки в Telegram-хранилище и БД PostgreSQL.

Возможности:
- Pre-flight check: проверка подключения к БД и Telegram ПЕРЕД пересжатием
- Аппаратное пересжатие в MP4 (NVIDIA NVENC) с кэшированием (не теряет результат при сбоях)
- Умный выбор: быстрый ремукс (если уже H.264 и < 1.95 GB) или качественное пересжатие
- Отправка в Telegram как нативное потоковое Видео (supports_streaming=True)
- Отказоустойчивая параллельная загрузка с автоповторами SaveBigFilePartRequest
- Генерация и привязка кода в PostgreSQL (episodes)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import asyncpg
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.prompt import Confirm, Prompt
from telethon import TelegramClient, helpers, utils
from telethon.tl import types
from telethon.tl.functions import upload

DB_CONFIG = {
    "host": os.environ.get("POSTGRES_HOST", "127.0.0.1"),
    "port": int(os.environ.get("POSTGRES_PORT", "5432")),
    "user": os.environ.get("POSTGRES_USER", "videogen"),
    "password": os.environ.get("POSTGRES_PASSWORD", ""),
    "database": os.environ.get("POSTGRES_DB", "videogen_db"),
}

TG_CONFIG = {
    "api_id": int(os.environ.get("TG_API_ID", "0")),
    "api_hash": os.environ.get("TG_API_HASH", ""),
    "channel_id": int(os.environ.get("TG_CHANNEL_ID", "0")),
    "session_name": os.environ.get(
        "TG_SESSION_NAME",
        str(Path(__file__).resolve().parent.parent / "telethon_uploader_session"),
    ),
}

BOT_USERNAME = os.environ.get("TG_BOT_USERNAME", "video_catalog_bot")
CACHE_DIR = Path("/tmp/videogen_transcode_cache")

console = Console()


def get_video_info(file_path: Path) -> dict[str, Any]:
    """Извлекает детальные параметры видео (кодек, разрешение, длительность) через ffprobe."""
    try:
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,duration",
            "-show_entries", "format=duration,size",
            "-of", "json",
            str(file_path),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)

        v_stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}

        codec = v_stream.get("codec_name", "")
        width = int(v_stream.get("width") or 1280)
        height = int(v_stream.get("height") or 720)

        dur_val = v_stream.get("duration") or fmt.get("duration") or 0.0
        duration = float(dur_val)
        size_bytes = int(fmt.get("size") or file_path.stat().st_size)

        return {
            "codec": codec.lower(),
            "width": width,
            "height": height,
            "duration": duration,
            "size_bytes": size_bytes,
            "is_h264": codec.lower() in ("h264", "avc1"),
            "is_mp4": file_path.suffix.lower() == ".mp4",
        }
    except Exception as e:
        console.print(f"[yellow]Предупреждение ffprobe:[/] {e}")
        return {
            "codec": "unknown",
            "width": 1280,
            "height": 720,
            "duration": 0.0,
            "size_bytes": file_path.stat().st_size,
            "is_h264": False,
            "is_mp4": file_path.suffix.lower() == ".mp4",
        }


def format_duration(seconds: float | None) -> str:
    """Форматирует длительность в HH:MM:SS."""
    if not seconds:
        return "--:--"
    total_sec = max(0, int(seconds))
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def has_nvenc_support() -> bool:
    """Проверяет доступность энкодера NVIDIA NVENC."""
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in res.stdout
    except Exception:
        return False


def reencode_video_to_mp4(
    input_path: Path,
    output_path: Path,
    quality_cq: int = 23,
) -> Path:
    """Пересжимает видео в веб-оптимизированный MP4 (H.264 + AAC + faststart)."""
    use_nvenc = has_nvenc_support()
    encoder_name = "NVIDIA NVENC (GPU)" if use_nvenc else "libx264 (CPU)"

    console.print(f"[bold cyan]Запуск пересжатия видео в MP4...[/] [dim]({encoder_name})[/]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = output_path.with_suffix(".part.mp4")

    cmd = ["ffmpeg", "-y", "-i", str(input_path)]

    if use_nvenc:
        cmd.extend([
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", str(quality_cq),
            "-b:v", "0",
            "-pix_fmt", "yuv420p",
        ])
    else:
        cmd.extend([
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", str(quality_cq),
            "-pix_fmt", "yuv420p",
        ])

    cmd.extend([
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        str(temp_target),
    ])

    t0 = time.time()
    with console.status(f"[cyan]Кодирование видео через {encoder_name}... Пожалуйста, подождите...[/]"):
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            console.print(f"[bold red]Ошибка FFmpeg:[/] {proc.stderr[-500:]}")
            if temp_target.exists():
                temp_target.unlink()
            raise RuntimeError(f"FFmpeg encoding failed with code {proc.returncode}")

    temp_target.rename(output_path)
    elapsed = time.time() - t0
    in_size_mb = input_path.stat().st_size / (1024 * 1024)
    out_size_mb = output_path.stat().st_size / (1024 * 1024)
    ratio = (1.0 - (out_size_mb / max(0.1, in_size_mb))) * 100.0

    console.print(
        f"[bold green]✔ Пересжатие завершено за {elapsed:.1f} сек![/] "
        f"[dim]{in_size_mb:.1f} MB ➔[/] [bold yellow]{out_size_mb:.1f} MB[/] "
        f"([cyan]{ratio:+.1f}%[/])"
    )
    return output_path


def remux_faststart_mp4(input_path: Path, output_path: Path) -> Path:
    """Быстрый ремуксинг MP4 для добавления флага faststart (без перекодирования)."""
    console.print("[cyan]Быстрая оптимизация MP4 для онлайн-стриминга (faststart)...[/]")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_target = output_path.with_suffix(".part.mp4")
    cmd = [
        "ffmpeg", "-y",
        "-i", str(input_path),
        "-c", "copy",
        "-movflags", "+faststart",
        str(temp_target),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    temp_target.rename(output_path)
    return output_path


async def get_next_film_code(conn: asyncpg.Connection) -> str:
    """Генерирует следующий доступный 5-значный код фильма (начиная с 10001)."""
    query = """
    SELECT code FROM episodes 
    WHERE is_film = true AND code ~ '^[0-9]+$' 
    ORDER BY code::bigint DESC LIMIT 1;
    """
    row = await conn.fetchval(query)
    if row:
        return f"{int(row) + 1:05d}"
    
    chk = await conn.fetchval("SELECT id FROM episodes WHERE code = '10001';")
    if not chk:
        return "10001"
    
    for i in range(10002, 99999):
        code_str = f"{i:05d}"
        if not await conn.fetchval("SELECT id FROM episodes WHERE code = $1;", code_str):
            return code_str
    return "99999"


async def fast_upload_file(
    client: TelegramClient,
    file_path: Path,
    max_workers: int = 12,
    progress_cb: Any = None,
) -> types.InputFile | types.InputFileBig:
    """Многопоточная отказоустойчивая выгрузка файла в Telegram (FastTelethon)."""
    p = Path(file_path)
    file_size = p.stat().st_size
    if file_size == 0:
        raise ValueError(f"Файл {p} пустой")

    part_size = 512 * 1024  # 512 KB
    total_parts = math.ceil(file_size / part_size)
    is_big = file_size > 10 * 1024 * 1024

    file_id = helpers.generate_random_long()
    uploaded_bytes = 0
    lock = asyncio.Lock()

    async def _upload_part(part_index: int, data: bytes) -> None:
        nonlocal uploaded_bytes
        max_retries = 5
        for attempt in range(max_retries):
            try:
                if is_big:
                    req = upload.SaveBigFilePartRequest(
                        file_id=file_id,
                        file_part=part_index,
                        file_total_parts=total_parts,
                        bytes=data,
                    )
                else:
                    req = upload.SaveFilePartRequest(
                        file_id=file_id,
                        file_part=part_index,
                        bytes=data,
                    )
                await client(req)
                break
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                await asyncio.sleep(1.0 + attempt * 0.5)

        if progress_cb:
            async with lock:
                uploaded_bytes += len(data)
                progress_cb(uploaded_bytes, file_size)

    semaphore = asyncio.Semaphore(max_workers)

    async def _worker(part_index: int, offset: int, length: int) -> None:
        async with semaphore:
            with p.open("rb") as f:
                f.seek(offset)
                chunk = f.read(length)
            await _upload_part(part_index, chunk)

    tasks = [
        _worker(i, i * part_size, min(part_size, file_size - i * part_size))
        for i in range(total_parts)
    ]
    await asyncio.gather(*tasks)

    if is_big:
        return types.InputFileBig(id=file_id, parts=total_parts, name=p.name)

    import hashlib
    hasher = hashlib.md5()
    with p.open("rb") as f:
        while chunk_data := f.read(512 * 1024):
            hasher.update(chunk_data)
    return types.InputFile(id=file_id, parts=total_parts, name=p.name, md5_checksum=hasher.hexdigest())


async def preflight_check() -> asyncpg.Connection:
    """Проверяет связь с PostgreSQL и Telegram ПЕРЕД пересжатием."""
    with console.status("[cyan]Проверка подключения к PostgreSQL (Aeza VPS)...[/]"):
        db_url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
        try:
            conn = await asyncpg.connect(db_url, timeout=10.0)
            await conn.fetchval("SELECT 1;")
            await conn.execute("ALTER TABLE episodes ADD COLUMN IF NOT EXISTS description TEXT;")
        except Exception as e:
            console.print(f"[bold red]✘ Ошибка подключения к базе данных {DB_CONFIG['host']}:[/] {e}")
            raise

    with console.status("[cyan]Проверка авторизации в Telegram Telethon...[/]"):
        client = TelegramClient(
            TG_CONFIG["session_name"],
            TG_CONFIG["api_id"],
            TG_CONFIG["api_hash"],
        )
        await client.connect()
        try:
            if not await client.is_user_authorized():
                raise RuntimeError("Сессия Telethon не авторизована! Запустите скрипт авторизации.")
        finally:
            await client.disconnect()

    console.print("[bold green]✔ Pre-flight check пройден:[/] PostgreSQL и Telegram активны.\n")
    return conn


async def run_upload(
    file_path: Path,
    title: str,
    description: str,
    custom_code: str | None = None,
    is_film: bool = True,
    force_reencode: bool = False,
    quality_cq: int = 23,
) -> None:
    src_path = Path(file_path).expanduser().resolve()
    if not src_path.is_file():
        console.print(f"[bold red]Ошибка:[/] Файл не найден: {src_path}")
        return

    # 1. ПРОВЕРКА СВЯЗИ В САМОМ НАЧАЛЕ (не тратим время на пересжатие при сбое сети/БД)
    try:
        conn = await preflight_check()
    except Exception:
        console.print("[bold red]Публикация остановлена из-за ошибки доступности сервисов.[/]")
        return

    try:
        info = get_video_info(src_path)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached_file = CACHE_DIR / f"{src_path.stem}_converted.mp4"

        upload_path = src_path

        # Проверяем, есть ли уже готовый пережатый файл в кэше
        if cached_file.is_file() and cached_file.stat().st_size > 10 * 1024 * 1024:
            cached_mb = cached_file.stat().st_size / (1024 * 1024)
            console.print(f"[bold green]✔ Найден уже пережатый файл в кэше:[/] {cached_file.name} ({cached_mb:.1f} MB)")
            upload_path = cached_file
            info = get_video_info(upload_path)
        else:
            # Решаем: нужно ли полное пересжатие или достаточно быстрого ремукса faststart
            needs_full_encode = force_reencode or not info["is_h264"] or (info["size_bytes"] > 1.95 * 1024 * 1024 * 1024)

            if needs_full_encode:
                upload_path = reencode_video_to_mp4(src_path, cached_file, quality_cq=quality_cq)
                info = get_video_info(upload_path)
            elif not info["is_mp4"]:
                # Уже H.264 и меньше 1.95 GB, но контейнер .mkv/.mov — быстрый ремукс в MP4!
                upload_path = remux_faststart_mp4(src_path, cached_file)
                info = get_video_info(upload_path)
            else:
                # Уже MP4 H.264
                upload_path = remux_faststart_mp4(src_path, cached_file)
                info = get_video_info(upload_path)

        size_bytes = upload_path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        size_gb = size_bytes / (1024 * 1024 * 1024)
        duration_sec = info["duration"]
        width = info["width"]
        height = info["height"]

        # Назначаем код
        if custom_code and custom_code.strip():
            code = custom_code.strip().upper()
            existing = await conn.fetchrow("SELECT id, title FROM episodes WHERE code = $1;", code)
            if existing:
                console.print(f"[yellow]Предупреждение:[/] Код {code} уже существует ('{existing['title']}'). Будет перезаписан.")
        else:
            code = await get_next_film_code(conn)

        console.print(
            Panel.fit(
                f"[bold cyan]Название:[/] {title}\n"
                f"[bold cyan]Формат:[/] MP4 (H.264, {width}x{height}, faststart)\n"
                f"[bold cyan]Файл:[/] {upload_path.name} ({size_gb:.2f} GB / {size_mb:.1f} MB)\n"
                f"[bold cyan]Длительность:[/] {format_duration(duration_sec)}\n"
                f"[bold cyan]Назначенный код:[/] [bold yellow]{code}[/]\n"
                f"[bold cyan]Режим отправки:[/] [bold green]Native Streamable Video (плеер Telegram)[/]\n"
                f"[bold cyan]Канал-хранилище:[/] {TG_CONFIG['channel_id']}",
                title="[bold green]Параметры публикации контента[/]",
                border_style="cyan",
            )
        )

        # 2. Выгрузка в Telegram через Telethon
        client = TelegramClient(
            TG_CONFIG["session_name"],
            TG_CONFIG["api_id"],
            TG_CONFIG["api_hash"],
        )
        await client.connect()

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.fields[name]}[/]"),
                BarColumn(bar_width=40),
                TextColumn("[bold magenta]{task.percentage:>3.1f}%[/]"),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task_id = progress.add_task("upload", total=size_bytes, name=upload_path.name)

                def progress_callback(current: int, total: int) -> None:
                    progress.update(task_id, completed=current, total=total)

                t0 = time.time()
                uploaded_input_file = await fast_upload_file(
                    client=client,
                    file_path=upload_path,
                    max_workers=12,
                    progress_cb=progress_callback,
                )

            caption_meta = json.dumps(
                {
                    "name": title,
                    "code": code,
                    "type": "film" if is_film else "video",
                    "is_film": is_film,
                    "description": description or "",
                },
                ensure_ascii=False,
            )

            # Отправка как нативное Video
            with console.status("[cyan]Отправка видео в канал-хранилище...[/]"):
                video_attributes = [
                    types.DocumentAttributeVideo(
                        duration=int(duration_sec or 0),
                        w=width,
                        h=height,
                        supports_streaming=True,
                    )
                ]
                msg = await client.send_file(
                    entity=TG_CONFIG["channel_id"],
                    file=uploaded_input_file,
                    caption=caption_meta,
                    attributes=video_attributes,
                    supports_streaming=True,
                )

            file_id = None
            if hasattr(msg, "media") and msg.media:
                try:
                    file_id = utils.pack_bot_file_id(msg.media)
                except Exception:
                    file_id = None

            if not file_id:
                file_id = str(getattr(msg.file, "id", msg.id))

            message_id = msg.id

            # 3. Сохранение в базу данных
            insert_query = """
            INSERT INTO episodes (code, base_code, title, description, is_film, tg_file_id, tg_message_id, duration)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (code) DO UPDATE SET
                title = EXCLUDED.title,
                description = EXCLUDED.description,
                is_film = EXCLUDED.is_film,
                tg_file_id = EXCLUDED.tg_file_id,
                tg_message_id = EXCLUDED.tg_message_id,
                duration = EXCLUDED.duration;
            """
            await conn.execute(
                insert_query,
                code,
                code,
                title,
                description or "",
                is_film,
                file_id,
                message_id,
                duration_sec,
            )

            upload_time = time.time() - t0
            speed_mbps = size_mb / max(0.1, upload_time)

            # Успешно сохранили — можно удалить кэшированный пережатый файл
            if cached_file.exists():
                cached_file.unlink()

            deeplink = f"https://t.me/{BOT_USERNAME}?start={code}"
            copy_text = f"🎬 {title}\n🔥 Смотри в боте: @{BOT_USERNAME}\n👉 Код фильма: {code}"

            console.print("\n")
            console.print(
                Panel.fit(
                    f"[bold green]✔ ФИЛЬМ УСПЕШНО ОПУБЛИКОВАН КАК ПОТОКОВОЕ ВИДЕО![/]\n\n"
                    f"[bold white]Код фильма:[/] [bold yellow on black]  {code}  [/]\n"
                    f"[bold white]Прямая ссылка:[/] [cyan]{deeplink}[/]\n"
                    f"[bold white]Название:[/] {title}\n"
                    f"[bold white]Описание:[/] {description or '(не указано)'}\n"
                    f"[bold white]Длительность:[/] {format_duration(duration_sec)} ({width}x{height})\n"
                    f"[bold white]Telegram File ID:[/] [dim]{file_id}[/]\n"
                    f"[bold white]Message ID:[/] [dim]{message_id}[/] (канал {TG_CONFIG['channel_id']})\n"
                    f"[bold white]Скорость выгрузки:[/] {upload_time:.1f} сек ({speed_mbps:.1f} MB/s)\n\n"
                    f"[bold cyan]Готовый текст для описания ролика в TikTok/Shorts:[/]\n"
                    f"[italic]{copy_text}[/]",
                    title="[bold green]Video-Gen Content Publisher[/]",
                    border_style="green",
                )
            )

        finally:
            await client.disconnect()

    finally:
        await conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Загрузка видеофайла/полнометражки в хранилище как Video и генерация кода для каталога."
    )
    parser.add_argument("-f", "--file", help="Путь к видеофайлу (mp4, mkv, mov, etc.)")
    parser.add_argument("-t", "--title", help="Название фильма или серии")
    parser.add_argument("-d", "--desc", "--description", help="Описание фильма / синопсис", default="")
    parser.add_argument("-c", "--code", help="Пользовательский код (если не указан — генерируется 10001+)")
    parser.add_argument("--series", action="store_true", help="Отметить как серию сериала, а не полнометражку")
    parser.add_argument("--reencode", "--compress", action="store_true", help="Принудительно пережать в MP4 H.264")
    parser.add_argument("--cq", type=int, default=23, help="Качество сжатия CQ/CRF (по умолчанию 23)")

    args = parser.parse_args()

    is_interactive = not bool(args.file)

    file_str = args.file
    if not file_str:
        file_str = Prompt.ask("[bold cyan]Введите путь к видеофайлу[/]")

    file_path = Path(file_str).expanduser().resolve()
    while not file_path.is_file():
        console.print(f"[red]Файл не найден:[/] {file_path}")
        file_str = Prompt.ask("[bold cyan]Введите корректный путь к видеофайлу[/]")
        file_path = Path(file_str).expanduser().resolve()

    title = args.title
    if not title:
        default_title = file_path.stem.replace(".", " ").replace("_", " ").title()
        if is_interactive:
            title = Prompt.ask("[bold cyan]Введите название фильма[/]", default=default_title)
        else:
            title = default_title

    description = args.desc
    if is_interactive and not description:
        description = Prompt.ask("[bold cyan]Введите описание (или нажмите Enter для пропуска)[/]", default="")

    custom_code = args.code
    if is_interactive and not custom_code:
        custom_code = Prompt.ask(
            "[bold cyan]Код для бота (Enter для автогенерации, например 10001)[/]",
            default="",
        )

    force_reencode = args.reencode
    if is_interactive and not force_reencode:
        info = get_video_info(file_path)
        if not info["is_h264"] or info["size_bytes"] > 1.95 * 1024 * 1024 * 1024:
            console.print(f"[yellow]Файл ({info['codec']}, {info['size_bytes'] / (1024*1024):.1f} MB) требует пересжатия для укладывания в лимиты Telegram.[/]")
            force_reencode = True
        else:
            force_reencode = Confirm.ask("[bold cyan]Выполнить дополнительное пересжатие для уменьшения веса?[/]", default=False)

    is_film = not args.series

    asyncio.run(
        run_upload(
            file_path=file_path,
            title=title.strip(),
            description=description.strip(),
            custom_code=custom_code.strip() if custom_code else None,
            is_film=is_film,
            force_reencode=force_reencode,
            quality_cq=args.cq,
        )
    )


if __name__ == "__main__":
    main()
