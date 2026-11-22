#!/usr/bin/env python3
"""Скрипт пакетной загрузки полных серий в Telegram и привязки их tg_file_id в PostgreSQL.

Оснащен интерактивным прогресс-баром Rich, расчетом скорости, ETA и подробной сводкой.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import asyncpg
from loguru import logger
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
from rich.table import Table
from telethon import TelegramClient, helpers, utils
from telethon.tl import types
from telethon.tl.functions import upload

from video_gen.config import AppConfig

# Настройка loguru только в файл (чтобы не ломать Rich прогресс-бар в терминале)
logger.remove()
Path("logs").mkdir(exist_ok=True)
logger.add("logs/upload_full_videos.log", rotation="50 MB", retention="7 days", level="INFO")

console = Console()


def natural_sort_key(path: Path) -> list[int | str]:
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r"(\d+)", path.name)]


def extract_episode_num(path: Path) -> int:
    m = re.search(r"(\d+)", path.stem)
    return int(m.group(1)) if m else 1


async def fast_upload_file(
    client: TelegramClient,
    file_path: Path,
    max_workers: int = 12,
    progress_cb: Any = None,
) -> types.InputFile | types.InputFileBig:
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


async def main() -> None:
    cfg = AppConfig.from_toml("config.toml")
    tg = cfg.telegram
    db_cfg = cfg.database

    if not tg.enabled or not tg.channel:
        console.print("[bold red]Ошибка:[/] Telegram выгрузка отключена или не указан channel в config.toml")
        return

    db_url = f"postgresql://{db_cfg.user}:{db_cfg.password}@{db_cfg.host}:{db_cfg.port}/{db_cfg.database}"

    async def query_db(q: str, *args: Any) -> Any:
        conn = await asyncpg.connect(db_url)
        try:
            return await conn.fetchrow(q, *args)
        finally:
            await conn.close()

    async def execute_db(q: str, *args: Any) -> None:
        conn = await asyncpg.connect(db_url)
        try:
            await conn.execute(q, *args)
        finally:
            await conn.close()

    # Проверка доступности БД
    try:
        test_conn = await asyncpg.connect(db_url, timeout=5.0)
        await test_conn.close()
    except Exception as e:
        console.print(f"[bold red]Ошибка подключения к БД {db_cfg.host}:[/] {e}")
        return

    # Заголовок утилиты
    console.print(
        Panel.fit(
            f"[bold cyan]Video-Gen Full Video Uploader[/]\n"
            f"[dim]Канал Telegram:[/] [yellow]{tg.channel}[/]\n"
            f"[dim]База данных:[/] [green]{db_cfg.host}:{db_cfg.port}/{db_cfg.database}[/]\n"
            f"[dim]Параллельных потоков:[/] [magenta]12 воркеров (FastTelethon)[/]",
            title="[bold green]Загрузчик контента Video-Gen[/]",
            border_style="cyan",
        )
    )

    # Подключение к Telethon
    with console.status("[cyan]Подключение к серверам Telegram...[/]"):
        client = TelegramClient("telethon_uploader_session", tg.api_id, tg.api_hash)
        if tg.bot_token:
            await client.start(bot_token=tg.bot_token)
        else:
            await client.start()
    console.print("[bold green]✔[/] Авторизация в Telegram успешна.\n")

    folder_mapping = [
        {"dir": Path("input/top/south_park"), "base_code": "00001", "title": "Южный Парк"},
        {"dir": Path("input/top/familyguy"), "base_code": "00002", "title": "Гриффины"},
        {"dir": Path("input/top/rick_and_morty"), "base_code": "00003", "title": "Рик и Морти"},
    ]

    all_queue: list[dict[str, Any]] = []
    for item in folder_mapping:
        dir_path = item["dir"]
        if not dir_path.is_dir():
            continue
        found = [
            f for f in dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".mkv", ".mp4", ".mov", ".webm"}
        ]
        found.sort(key=natural_sort_key)
        for f in found:
            ep_num = extract_episode_num(f)
            code = f"{item['base_code']}-{ep_num}"
            all_queue.append({
                "file": f,
                "title": item["title"],
                "base_code": item["base_code"],
                "ep_num": ep_num,
                "code": code,
                "size_mb": f.stat().st_size / (1024 * 1024),
            })

    total_count = len(all_queue)
    console.print(f"[bold]Очередь:[/] найдено [cyan]{total_count}[/] видеофайлов для проверки.\n")

    total_uploaded = 0
    total_skipped = 0
    total_failed = 0
    start_time = time.time()

    try:
        for idx, item in enumerate(all_queue, 1):
            f: Path = item["file"]
            code: str = item["code"]
            title: str = item["title"]
            ep_num: int = item["ep_num"]
            base_code: str = item["base_code"]
            size_mb: float = item["size_mb"]

            # 1. Проверяем в БД: есть ли уже tg_file_id
            existing = await query_db(
                "SELECT id, tg_file_id FROM episodes WHERE code = $1;", code
            )

            if existing and existing["tg_file_id"]:
                fid_short = existing["tg_file_id"][:16]
                console.print(
                    f"[{idx}/{total_count}] [dim]↷ [ПРОПУСК][/] [cyan]{code}[/] "
                    f"({title}, {f.name}) [dim]уже в хранилище (ID: {fid_short}...)[/]"
                )
                logger.info(f"[ПРОПУСК] {code} уже выгружен (file_id: {existing['tg_file_id']})")
                total_skipped += 1
                continue

            console.print(
                f"\n[{idx}/{total_count}] [bold yellow]▶ СТАРТ:[/] [bold cyan]{code}[/] "
                f"({title}) [dim]файл: {f.name} ({size_mb:.1f} MB)[/]"
            )
            logger.info(f"[СТАРТ] Выгрузка {code} ({title}) файл: {f.name} [{size_mb:.1f} MB]...")

            file_size_bytes = f.stat().st_size

            # Создаем динамический прогресс-бар Rich
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.fields[name]}[/]"),
                BarColumn(bar_width=40),
                TextColumn("[bold magenta]{task.percentage:>3.1f}%[/]"),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console,
                transient=True,
            ) as progress:
                upload_task = progress.add_task(
                    "upload",
                    total=file_size_bytes,
                    name=f"{code} {title}",
                )

                def on_progress(cur: int, tot: int) -> None:
                    progress.update(upload_task, completed=cur, total=tot)

                try:
                    uploaded = await fast_upload_file(
                        client, f, max_workers=12, progress_cb=on_progress
                    )

                    payload = {
                        "chapter": "full_episodes",
                        "name": f"{title} ({ep_num})",
                        "code": code,
                        "base_code": base_code,
                        "episode": ep_num,
                        "is_film": False,
                    }
                    caption_json = json.dumps(payload, ensure_ascii=False)

                    msg = await client.send_file(
                        entity=tg.channel,
                        file=uploaded,
                        caption=caption_json,
                        force_document=True,
                    )

                    file_id = None
                    if hasattr(msg, "media") and msg.media:
                        try:
                            file_id = utils.pack_bot_file_id(msg.media)
                        except Exception:
                            file_id = None

                    if not file_id:
                        file_id = str(getattr(msg.file, "id", msg.id))

                    # Сохранение в PostgreSQL
                    await execute_db(
                        """
                        INSERT INTO episodes (code, base_code, title, episode, is_film, tg_file_id)
                        VALUES ($1, $2, $3, $4, false, $5)
                        ON CONFLICT (code) DO UPDATE SET tg_file_id = EXCLUDED.tg_file_id;
                        """,
                        code,
                        base_code,
                        title,
                        ep_num,
                        file_id,
                    )

                    console.print(
                        f"[{idx}/{total_count}] [bold green]✔ ВЫГРУЖЕНО:[/] [bold cyan]{code}[/] "
                        f"({title}) [dim]file_id: {file_id}[/]"
                    )
                    logger.info(f"[УСПЕХ] {code} выгружен! file_id: {file_id}")
                    total_uploaded += 1
                    await asyncio.sleep(1.5)

                except Exception as e:
                    console.print(
                        f"[{idx}/{total_count}] [bold red]✘ ОШИБКА:[/] [bold cyan]{code}[/] ({f.name}): {e}"
                    )
                    logger.exception(f"[ОШИБКА] Не удалось выгрузить {code} ({f.name}): {e}")
                    total_failed += 1
                    await asyncio.sleep(5.0)

    finally:
        await client.disconnect()
        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)

        table = Table(title="Итоги сессии выгрузки", border_style="cyan")
        table.add_column("Параметр", style="cyan")
        table.add_column("Значение", style="bold green")

        table.add_row("Успешно выгружено", str(total_uploaded))
        table.add_row("Пропущено (ранее загружено)", str(total_skipped))
        table.add_row("Ошибок", str(total_failed))
        table.add_row("Затраченное время", f"{mins} мин {secs} сек")

        console.print("\n")
        console.print(table)


if __name__ == "__main__":
    asyncio.run(main())
