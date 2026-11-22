"""FilmWorker — модуль каталогизации фильмов, сериалов и нарезок (каталог видео).

Присваивает уникальные иерархические коды формата (XXXXX-{серия}-{нарезка}):
- Для фильма: последние 2 части опциональны
    - Сам фильм: XXXXX (например "04291")
    - Нарезка фильма: XXXXX-{нарезка} (например "04291-1")
- Для сериала: последняя часть опциональна
    - Серия: XXXXX-{серия} (например "04291-1")
    - Нарезка серии: XXXXX-{серия}-{нарезка} (например "04291-1-1")

Сохраняет в базе данных PostgreSQL:
- Таймкоды (timecode_start, timecode_end, duration)
- Иерархические коды
- Метаданные (заголовки, хуки, описания, хэши рендера)
"""

from __future__ import annotations

import secrets
from typing import Any

import asyncpg
from loguru import logger

from video_gen.config import AppConfig, DatabaseConfig
from video_gen.models import Highlight, VideoJob
from video_gen.tasks.base_task import BaseTask


class FilmCode:
    """Генератор, парсер и валидатор иерархических кодов каталога видео."""

    @staticmethod
    def generate_base_code(length: int = 5) -> str:
        """Генерирует случайный цифровой базовый код фиксированной длины (например '04291')."""
        return f"{secrets.randbelow(10**length):0{length}d}"

    @staticmethod
    def format_base_code(val: int | str, length: int = 5) -> str:
        """Форматирует базовый код, дополняя ведущими нулями если это число."""
        if isinstance(val, int) or (isinstance(val, str) and val.isdigit() and len(val) < length):
            return f"{int(val):0{length}d}"
        return str(val)

    @staticmethod
    def build_code(
        base_code: str | int,
        episode: int | None = None,
        clip_idx: int | None = None,
        is_film: bool = False,
    ) -> str:
        """Формирует иерархический код по правилам (XXXXX-{серия}-{нарезка}).

        - Для фильма (is_film=True): последние 2 части опциональны.
            - Полный фильм: XXXXX
            - Нарезка фильма: XXXXX-{нарезка}
        - Для серии (is_film=False): последняя часть опциональна.
            - Серия: XXXXX-{серия}
            - Нарезка серии: XXXXX-{серия}-{нарезка}
        """
        base = FilmCode.format_base_code(base_code)

        if is_film:
            if clip_idx is None:
                return base
            return f"{base}-{clip_idx}"
        else:
            ep = episode if episode is not None else 1
            if clip_idx is None:
                return f"{base}-{ep}"
            return f"{base}-{ep}-{clip_idx}"

    @staticmethod
    def build_video_code(
        base_code: str | int,
        is_film: bool = False,
        episode: int | None = None,
    ) -> str:
        """Формирует код видеоисточника (фильма или серии сериала)."""
        return FilmCode.build_code(
            base_code=base_code, episode=episode, clip_idx=None, is_film=is_film
        )

    @staticmethod
    def build_clip_code(
        base_code: str | int,
        clip_idx: int,
        is_film: bool = False,
        episode: int | None = None,
    ) -> str:
        """Формирует код нарезки."""
        return FilmCode.build_code(
            base_code=base_code,
            episode=episode,
            clip_idx=clip_idx,
            is_film=is_film,
        )

    @staticmethod
    def parse_code(code: str, is_film: bool | None = None) -> dict[str, Any]:
        """Разбирает строку кода на составляющие.

        Returns:
            Словарь с base_code, is_film, episode, clip_idx и type.
        """
        code_str = code.strip()
        parts = code_str.split("-")

        if len(parts) == 1:
            return {
                "code": code_str,
                "base_code": parts[0],
                "is_film": True,
                "episode": None,
                "clip_idx": None,
                "type": "film",
            }
        elif len(parts) == 2:
            part2 = int(parts[1]) if parts[1].isdigit() else parts[1]
            if is_film is True:
                return {
                    "code": code_str,
                    "base_code": parts[0],
                    "is_film": True,
                    "episode": None,
                    "clip_idx": part2,
                    "type": "film_clip",
                }
            elif is_film is False:
                return {
                    "code": code_str,
                    "base_code": parts[0],
                    "is_film": False,
                    "episode": part2,
                    "clip_idx": None,
                    "type": "episode",
                }
            else:
                return {
                    "code": code_str,
                    "base_code": parts[0],
                    "is_film": None,
                    "second_part": part2,
                    "type": "ambiguous_2_parts",
                }
        elif len(parts) == 3:
            ep = int(parts[1]) if parts[1].isdigit() else parts[1]
            clip = int(parts[2]) if parts[2].isdigit() else parts[2]
            return {
                "code": code_str,
                "base_code": parts[0],
                "is_film": False,
                "episode": ep,
                "clip_idx": clip,
                "type": "episode_clip",
            }
        else:
            raise ValueError(f"Некорректный формат кода: {code_str}")


class FilmWorker(BaseTask):
    """Воркер каталогизации фильмов, сериалов и нарезок в базе данных PostgreSQL."""

    def __init__(self, config: AppConfig | DatabaseConfig | None = None) -> None:
        if isinstance(config, AppConfig):
            super().__init__(config)
            self.app_cfg: AppConfig | None = config
            self.db_cfg: DatabaseConfig = getattr(config, "database", DatabaseConfig())
        elif isinstance(config, DatabaseConfig):
            self.app_cfg = None
            self.db_cfg = config
        else:
            self.app_cfg = None
            self.db_cfg = DatabaseConfig()

        self._pool: asyncpg.Pool | None = None
        self.logger = logger.bind(worker="film_worker")

    async def init_pool(self) -> asyncpg.Pool:
        """Инициализирует пул соединений с базой данных PostgreSQL."""
        if self._pool is not None and not self._pool._closed:
            return self._pool

        self.logger.info(
            f"[FILM_WORKER] Подключение к PostgreSQL {self.db_cfg.host}:{self.db_cfg.port}/{self.db_cfg.database}..."
        )
        self._pool = await asyncpg.create_pool(
            host=self.db_cfg.host,
            port=self.db_cfg.port,
            user=self.db_cfg.user,
            password=self.db_cfg.password,
            database=self.db_cfg.database,
            min_size=self.db_cfg.min_connections,
            max_size=self.db_cfg.max_connections,
            command_timeout=self.db_cfg.timeout,
        )
        self.logger.info("[FILM_WORKER] Пул соединений с PostgreSQL успешно инициализирован.")
        return self._pool

    async def get_pool(self) -> asyncpg.Pool:
        """Возвращает активный пул или инициализирует его."""
        if self._pool is None or self._pool._closed:
            return await self.init_pool()
        return self._pool

    async def close(self) -> None:
        """Закрывает пул соединений."""
        if self._pool is not None and not self._pool._closed:
            await self._pool.close()
            self.logger.info("[FILM_WORKER] Пул соединений PostgreSQL закрыт.")
            self._pool = None

    async def __aenter__(self) -> FilmWorker:
        await self.init_pool()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        await self.close()

    async def init_schema(self) -> None:
        """Проверяет и создает структуру таблиц и индексов при необходимости."""
        pool = await self.get_pool()
        schema_sql = """
        CREATE TABLE IF NOT EXISTS episodes (
            id SERIAL PRIMARY KEY,
            code VARCHAR(64) UNIQUE NOT NULL,
            base_code VARCHAR(32),
            title VARCHAR(255) NOT NULL,
            season INT DEFAULT 1,
            episode INT DEFAULT 1,
            stream_url TEXT,
            created_at TIMESTAMPTZ DEFAULT now(),
            is_film BOOLEAN DEFAULT false,
            duration DOUBLE PRECISION
        );

        CREATE TABLE IF NOT EXISTS clips (
            id SERIAL PRIMARY KEY,
            episode_code VARCHAR(64) NOT NULL REFERENCES episodes(code) ON DELETE CASCADE,
            code VARCHAR(64) UNIQUE NOT NULL,
            clip_idx INT,
            timecode_start DOUBLE PRECISION NOT NULL,
            timecode_end DOUBLE PRECISION NOT NULL,
            duration DOUBLE PRECISION,
            hook TEXT,
            viral_title TEXT,
            description TEXT,
            render_hash VARCHAR(255) UNIQUE,
            created_at TIMESTAMPTZ DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_code ON episodes(code);
        CREATE INDEX IF NOT EXISTS idx_episodes_base_code ON episodes(base_code);
        CREATE INDEX IF NOT EXISTS idx_clips_code ON clips(code);
        CREATE INDEX IF NOT EXISTS idx_clips_episode_code ON clips(episode_code);
        """
        async with pool.acquire() as conn:
            await conn.execute(schema_sql)
        self.logger.info("[FILM_WORKER] Схема PostgreSQL проверена и готова.")

    async def register_video(
        self,
        title: str,
        base_code: str | int | None = None,
        is_film: bool = False,
        season: int = 1,
        episode: int | None = None,
        duration: float | None = None,
        stream_url: str | None = None,
        custom_code: str | None = None,
    ) -> dict[str, Any]:
        """Регистрирует фильм или эпизод сериала в таблице episodes.

        Returns:
            Словарь с данными сохраненной записи.
        """
        pool = await self.get_pool()

        if base_code is None:
            base_code_str = FilmCode.generate_base_code()
        else:
            base_code_str = FilmCode.format_base_code(base_code)

        effective_episode = None if is_film else (episode if episode is not None else 1)
        video_code = custom_code or FilmCode.build_video_code(
            base_code=base_code_str,
            is_film=is_film,
            episode=effective_episode,
        )

        query = """
        INSERT INTO episodes (
            code, base_code, title, season, episode, stream_url, is_film, duration
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (code) DO UPDATE SET
            base_code = EXCLUDED.base_code,
            title = EXCLUDED.title,
            season = COALESCE(EXCLUDED.season, episodes.season),
            episode = EXCLUDED.episode,
            stream_url = COALESCE(EXCLUDED.stream_url, episodes.stream_url),
            is_film = EXCLUDED.is_film,
            duration = COALESCE(EXCLUDED.duration, episodes.duration)
        RETURNING id, code, base_code, title, season, episode, stream_url, is_film, duration, created_at;
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                video_code,
                base_code_str,
                title,
                season if not is_film else None,
                effective_episode,
                stream_url,
                is_film,
                float(duration) if duration is not None else None,
            )

        res = dict(row)
        self.logger.info(
            f"[FILM_WORKER] Зарегистрировано видео {video_code} (фильм={is_film}): '{title}' [id={res['id']}]"
        )
        return res

    async def register_clip(
        self,
        episode_code: str,
        clip_idx: int,
        timecode_start: float,
        timecode_end: float,
        duration: float | None = None,
        custom_code: str | None = None,
        hook: str | None = None,
        viral_title: str | None = None,
        description: str | None = None,
        render_hash: str | None = None,
        is_film: bool = False,
        base_code: str | int | None = None,
        episode: int | None = None,
    ) -> dict[str, Any]:
        """Регистрирует нарезку (клип) в таблице clips.

        Returns:
            Словарь с данными сохраненной нарезки.
        """
        pool = await self.get_pool()

        calc_duration = (
            duration if duration is not None else max(0.0, timecode_end - timecode_start)
        )

        if custom_code:
            clip_code = custom_code
        elif base_code is not None:
            clip_code = FilmCode.build_clip_code(
                base_code=base_code,
                clip_idx=clip_idx,
                is_film=is_film,
                episode=episode,
            )
        else:
            # Выводим код из episode_code
            clip_code = f"{episode_code}-{clip_idx}"

        query = """
        INSERT INTO clips (
            episode_code, code, clip_idx, timecode_start, timecode_end, duration,
            hook, viral_title, description, render_hash
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (code) DO UPDATE SET
            episode_code = EXCLUDED.episode_code,
            clip_idx = EXCLUDED.clip_idx,
            timecode_start = EXCLUDED.timecode_start,
            timecode_end = EXCLUDED.timecode_end,
            duration = EXCLUDED.duration,
            hook = COALESCE(EXCLUDED.hook, clips.hook),
            viral_title = COALESCE(EXCLUDED.viral_title, clips.viral_title),
            description = COALESCE(EXCLUDED.description, clips.description),
            render_hash = COALESCE(EXCLUDED.render_hash, clips.render_hash)
        RETURNING id, episode_code, code, clip_idx, timecode_start, timecode_end, duration, hook, viral_title, description, render_hash, created_at;
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                query,
                episode_code,
                clip_code,
                clip_idx,
                float(timecode_start),
                float(timecode_end),
                float(calc_duration),
                hook,
                viral_title,
                description,
                render_hash,
            )

        res = dict(row)
        self.logger.info(
            f"[FILM_WORKER] Зарегистрирована нарезка {clip_code} (таймкоды: {timecode_start:.1f}-{timecode_end:.1f}с, {calc_duration:.1f}с) [id={res['id']}]"
        )
        return res

    async def get_video(self, code: str) -> dict[str, Any] | None:
        """Получает запись видео/эпизода из episodes по коду."""
        pool = await self.get_pool()
        query = "SELECT * FROM episodes WHERE code = $1;"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, code)
        return dict(row) if row else None

    async def find_video_by_title(self, title: str) -> dict[str, Any] | None:
        """Ищет видео в episodes по точному названию."""
        pool = await self.get_pool()
        query = "SELECT * FROM episodes WHERE title = $1 ORDER BY id DESC LIMIT 1;"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, title)
        return dict(row) if row else None

    async def ensure_full_video_uploaded(
        self,
        job: VideoJob,
        *_args: Any,
        **_kwargs: Any,
    ) -> dict[str, Any]:
        """Обеспечивает регистрацию исходного видео в базе данных PostgreSQL."""
        is_film = job.is_film or job.metadata.get("is_film", False)
        title = (
            job.caption_name
            or job.metadata.get("caption_name")
            or job.metadata.get("title")
            or job.path.stem
        )
        season = job.season or job.metadata.get("season", 1)
        episode = None if is_film else (job.episode or job.part_idx)

        # Проверяем, есть ли уже видео с таким названием в БД для сохранения общего base_code
        existing_by_title = await self.find_video_by_title(title)
        if existing_by_title and existing_by_title.get("base_code"):
            base_code = existing_by_title["base_code"]
        elif job.base_code:
            base_code = FilmCode.format_base_code(job.base_code)
        else:
            base_code = FilmCode.generate_base_code(5)

        job.base_code = base_code
        video_code = job.video_code or FilmCode.build_video_code(
            base_code=base_code,
            is_film=is_film,
            episode=episode,
        )
        job.video_code = video_code

        # Проверяем наличие записи по video_code
        existing_rec = await self.get_video(video_code)
        if existing_rec:
            return existing_rec

        video_rec = await self.register_video(
            title=title,
            base_code=base_code,
            is_film=is_film,
            season=season,
            episode=episode,
            duration=job.metadata.get("duration"),
            custom_code=video_code,
        )
        return video_rec

    async def get_clip(self, code: str) -> dict[str, Any] | None:
        """Получает запись нарезки из clips по коду."""
        pool = await self.get_pool()
        query = "SELECT * FROM clips WHERE code = $1;"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, code)
        return dict(row) if row else None

    async def list_clips_for_video(self, video_code: str) -> list[dict[str, Any]]:
        """Возвращает все нарезки, привязанные к данному видео/эпизоду."""
        pool = await self.get_pool()
        query = "SELECT * FROM clips WHERE episode_code = $1 ORDER BY clip_idx ASC, id ASC;"
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, video_code)
        return [dict(r) for r in rows]

    async def get_by_code(self, code: str) -> dict[str, Any] | None:
        """Универсальный поиск по иерархическому коду (для ботов, диплинков /start <code> и API).

        Сначала ищет точное совпадение в нарезках (clips).
        Если не найдено — ищет в видео/эпизодах (episodes) и возвращает объект со списком нарезок.
        """
        pool = await self.get_pool()
        code_str = code.strip()

        # 1. Поиск в нарезках с объединением данных родительского эпизода/фильма
        clip_query = """
        SELECT
            c.id AS clip_id,
            c.code AS clip_code,
            c.clip_idx,
            c.timecode_start,
            c.timecode_end,
            c.duration AS clip_duration,
            c.hook,
            c.viral_title,
            c.description,
            c.render_hash,
            c.created_at AS clip_created_at,
            e.id AS episode_id,
            e.code AS video_code,
            e.base_code,
            e.title AS video_title,
            e.season,
            e.episode,
            e.is_film,
            e.duration AS video_duration,
            e.stream_url
        FROM clips c
        JOIN episodes e ON e.code = c.episode_code
        WHERE c.code = $1;
        """
        async with pool.acquire() as conn:
            clip_row = await conn.fetchrow(clip_query, code_str)
            if clip_row:
                data = dict(clip_row)
                return {
                    "type": "clip",
                    "code": code_str,
                    "clip": {
                        "id": data["clip_id"],
                        "code": data["clip_code"],
                        "clip_idx": data["clip_idx"],
                        "timecode_start": data["timecode_start"],
                        "timecode_end": data["timecode_end"],
                        "duration": data["clip_duration"],
                        "hook": data["hook"],
                        "viral_title": data["viral_title"],
                        "description": data["description"],
                        "render_hash": data["render_hash"],
                        "created_at": data["clip_created_at"],
                    },
                    "video": {
                        "id": data["episode_id"],
                        "code": data["video_code"],
                        "base_code": data["base_code"],
                        "title": data["video_title"],
                        "season": data["season"],
                        "episode": data["episode"],
                        "is_film": data["is_film"],
                        "duration": data["video_duration"],
                        "stream_url": data["stream_url"],
                    },
                }

            # 2. Поиск в видео/фильмах
            ep_query = "SELECT * FROM episodes WHERE code = $1;"
            ep_row = await conn.fetchrow(ep_query, code_str)
            if ep_row:
                ep_data = dict(ep_row)
                clips_rows = await conn.fetch(
                    "SELECT * FROM clips WHERE episode_code = $1 ORDER BY clip_idx ASC, id ASC;",
                    code_str,
                )
                return {
                    "type": "film" if ep_data.get("is_film") else "episode",
                    "code": code_str,
                    "video": ep_data,
                    "clips": [dict(c) for c in clips_rows],
                }

        return None

    async def delete_video(self, code: str) -> bool:
        """Удаляет видео и связанные с ним нарезки (CASCADE)."""
        pool = await self.get_pool()
        query = "DELETE FROM episodes WHERE code = $1 RETURNING id;"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, code)
        return bool(row)

    async def delete_clip(self, code: str) -> bool:
        """Удаляет отдельную нарезку по коду."""
        pool = await self.get_pool()
        query = "DELETE FROM clips WHERE code = $1 RETURNING id;"
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, code)
        return bool(row)

    async def process_job_and_highlights(
        self,
        job: VideoJob,
        highlights: list[Highlight],
        render_hashes: dict[int, str] | None = None,
    ) -> dict[str, Any]:
        """Пакетная обработка и сохранение исходного видео и его нарезок из VideoJob.

        Args:
            job: Объект VideoJob (содержит путь, конфиг, метаданные, base_code, is_film).
            highlights: Список распознанных/сгенерированных фрагментов.
            render_hashes: Опциональная карта {clip_idx: render_hash}.

        Returns:
            Словарь со структурой зарегистрированного видео и списка нарезок.
        """
        is_film = job.is_film or job.metadata.get("is_film", False)
        base_code = (
            job.base_code
            or job.codename
            or FilmCode.format_base_code(job.source_number or job.set_idx)
        )
        season = job.season or job.metadata.get("season", 1)
        episode = None if is_film else (job.episode or job.part_idx)
        title = (
            job.caption_name
            or job.metadata.get("caption_name")
            or job.metadata.get("title")
            or job.path.stem
        )
        duration = job.metadata.get("duration")

        # Регистрируем исходник
        video_rec = await self.register_video(
            title=title,
            base_code=base_code,
            is_film=is_film,
            season=season,
            episode=episode,
            duration=duration,
            stream_url=job.metadata.get("stream_url"),
            custom_code=job.video_code,
        )
        video_code = video_rec["code"]

        registered_clips: list[dict[str, Any]] = []
        for i, h in enumerate(highlights, 1):
            clip_code = FilmCode.build_clip_code(
                base_code=base_code,
                clip_idx=i,
                is_film=is_film,
                episode=episode,
            )
            # Обновляем сам объект Highlight для сквозной передачи
            h.code = clip_code
            h.clip_idx = i

            r_hash = (render_hashes or {}).get(i)

            clip_rec = await self.register_clip(
                episode_code=video_code,
                clip_idx=i,
                timecode_start=h.start,
                timecode_end=h.end,
                duration=h.end - h.start,
                custom_code=clip_code,
                hook=h.hook or h.reason,
                viral_title=h.viral_title,
                description=h.description,
                render_hash=r_hash,
                is_film=is_film,
                base_code=base_code,
                episode=episode,
            )
            registered_clips.append(clip_rec)

        return {
            "video": video_rec,
            "clips": registered_clips,
            "total_clips": len(registered_clips),
        }
