"""Тесты для FilmWorker и FilmCode: генерация кодов, парсинг и работа с PostgreSQL."""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from video_gen.config import AppConfig, DatabaseConfig
from video_gen.models import Highlight, VideoJob
from video_gen.tasks.film_worker import FilmCode, FilmWorker


class TestFilmCode:
    """Тестирование генерации, форматирования и парсинга иерархических кодов."""

    def test_generate_base_code(self) -> None:
        code = FilmCode.generate_base_code()
        assert len(code) == 5
        assert code.isdigit()

        code_7 = FilmCode.generate_base_code(length=7)
        assert len(code_7) == 7
        assert code_7.isdigit()

    def test_format_base_code(self) -> None:
        assert FilmCode.format_base_code(42) == "00042"
        assert FilmCode.format_base_code("123") == "00123"
        assert FilmCode.format_base_code("04291") == "04291"
        assert FilmCode.format_base_code("SOUTH") == "SOUTH"

    def test_build_code_film(self) -> None:
        # Для фильма последние 2 части (-{серия}-{нарезка}) опциональны:
        # 1. Сам фильм: только XXXXX
        assert FilmCode.build_code("04291", is_film=True) == "04291"
        assert FilmCode.build_video_code("04291", is_film=True) == "04291"

        # 2. Нарезка фильма: XXXXX-{нарезка}
        assert FilmCode.build_code("04291", clip_idx=1, is_film=True) == "04291-1"
        assert FilmCode.build_code("04291", clip_idx=5, is_film=True) == "04291-5"
        assert FilmCode.build_clip_code("04291", clip_idx=2, is_film=True) == "04291-2"

    def test_build_code_series(self) -> None:
        # Для сериала последняя часть (-{нарезка}) опциональна:
        # 1. Серия: XXXXX-{серия}
        assert FilmCode.build_code("04291", episode=1, is_film=False) == "04291-1"
        assert FilmCode.build_code("04291", episode=3, is_film=False) == "04291-3"
        assert FilmCode.build_video_code("04291", episode=2, is_film=False) == "04291-2"

        # 2. Нарезка серии: XXXXX-{серия}-{нарезка}
        assert FilmCode.build_code("04291", episode=1, clip_idx=1, is_film=False) == "04291-1-1"
        assert FilmCode.build_code("04291", episode=2, clip_idx=4, is_film=False) == "04291-2-4"
        assert (
            FilmCode.build_clip_code("04291", clip_idx=3, is_film=False, episode=2) == "04291-2-3"
        )

    def test_parse_code(self) -> None:
        # 1 часть: Фильм
        res = FilmCode.parse_code("04291")
        assert res["type"] == "film"
        assert res["base_code"] == "04291"
        assert res["is_film"] is True
        assert res["episode"] is None
        assert res["clip_idx"] is None

        # 2 части (фильм): Нарезка фильма
        res_fc = FilmCode.parse_code("04291-2", is_film=True)
        assert res_fc["type"] == "film_clip"
        assert res_fc["base_code"] == "04291"
        assert res_fc["clip_idx"] == 2
        assert res_fc["is_film"] is True

        # 2 части (сериал): Серия
        res_ep = FilmCode.parse_code("04291-2", is_film=False)
        assert res_ep["type"] == "episode"
        assert res_ep["base_code"] == "04291"
        assert res_ep["episode"] == 2
        assert res_ep["is_film"] is False

        # 3 части: Нарезка серии сериала
        res_sc = FilmCode.parse_code("04291-2-5")
        assert res_sc["type"] == "episode_clip"
        assert res_sc["base_code"] == "04291"
        assert res_sc["episode"] == 2
        assert res_sc["clip_idx"] == 5
        assert res_sc["is_film"] is False

        # Невалидный код
        with pytest.raises(ValueError):
            FilmCode.parse_code("1-2-3-4")


@pytest.mark.anyio
class TestFilmWorkerDatabase:
    """Интеграционные тесты сохранения и извлечения из базы данных PostgreSQL."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        return "asyncio"

    @pytest.fixture
    def db_config(self) -> DatabaseConfig:
        host = os.environ.get("POSTGRES_HOST")
        if not host:
            pytest.skip("Set POSTGRES_HOST to run live PostgreSQL integration tests")
        return DatabaseConfig(
            host=host,
            port=int(os.environ.get("POSTGRES_PORT", "5432")),
            user=os.environ.get("POSTGRES_USER", "videogen"),
            password=os.environ.get("POSTGRES_PASSWORD", ""),
            database=os.environ.get("POSTGRES_DB", "videogen_db"),
            timeout=10.0,
        )

    async def test_film_worker_lifecycle(self, db_config: DatabaseConfig) -> None:
        worker = FilmWorker(db_config)
        try:
            pool = await worker.init_pool()
            assert pool is not None
            await worker.init_schema()

            test_base_film = "99001"
            test_base_series = "99002"

            # 1. Регистрация фильма и его нарезки
            film = await worker.register_video(
                title="Интерстеллар (Тест)",
                base_code=test_base_film,
                is_film=True,
                duration=10140.0,
            )
            assert film["code"] == test_base_film
            assert film["is_film"] is True

            clip_film = await worker.register_clip(
                episode_code=film["code"],
                clip_idx=1,
                timecode_start=120.0,
                timecode_end=175.5,
                is_film=True,
                base_code=test_base_film,
                hook="Невероятный стык в космосе",
                viral_title="Они сделали это в невесомости!",
            )
            assert clip_film["code"] == f"{test_base_film}-1"
            assert clip_film["duration"] == pytest.approx(55.5)
            assert clip_film["hook"] == "Невероятный стык в космосе"

            # Поиск по коду нарезки фильма
            found_clip = await worker.get_by_code(clip_film["code"])
            assert found_clip is not None
            assert found_clip["type"] == "clip"
            assert found_clip["clip"]["code"] == f"{test_base_film}-1"
            assert found_clip["video"]["title"] == "Интерстеллар (Тест)"

            # Поиск по коду самого фильма
            found_film = await worker.get_by_code(test_base_film)
            assert found_film is not None
            assert found_film["type"] == "film"
            assert len(found_film["clips"]) >= 1

            # 2. Регистрация сериала и его нарезки
            series_ep = await worker.register_video(
                title="Южный Парк S01E01 (Тест)",
                base_code=test_base_series,
                is_film=False,
                season=1,
                episode=1,
                duration=1320.0,
            )
            assert series_ep["code"] == f"{test_base_series}-1"
            assert series_ep["is_film"] is False

            clip_series = await worker.register_clip(
                episode_code=series_ep["code"],
                clip_idx=2,
                timecode_start=300.0,
                timecode_end=350.0,
                is_film=False,
                base_code=test_base_series,
                episode=1,
                hook="Картман и зонд пришельцев",
                viral_title="Что они с ним сделали?!",
            )
            assert clip_series["code"] == f"{test_base_series}-1-2"
            assert clip_series["duration"] == 50.0

            # Поиск по коду нарезки серии
            found_ep_clip = await worker.get_by_code(clip_series["code"])
            assert found_ep_clip is not None
            assert found_ep_clip["type"] == "clip"
            assert found_ep_clip["clip"]["code"] == f"{test_base_series}-1-2"
            assert found_ep_clip["video"]["title"] == "Южный Парк S01E01 (Тест)"

            # 3. Очистка тестовых записей (CASCADE удалит и clips)
            del_film = await worker.delete_video(test_base_film)
            del_series = await worker.delete_video(series_ep["code"])
            assert del_film is True
            assert del_series is True

            # Проверяем, что нарезки тоже удалились по каскаду
            assert await worker.get_clip(clip_film["code"]) is None
            assert await worker.get_clip(clip_series["code"]) is None

        finally:
            await worker.close()

    async def test_process_job_and_highlights(self, db_config: DatabaseConfig) -> None:
        cfg = AppConfig.model_validate(
            {
                "videos": {"top_videos": [{"type": "video", "source": "input"}]},
                "params": {"gemini_key": "test"},
            }
        )
        worker = FilmWorker(db_config)
        try:
            await worker.init_pool()

            test_base = "99003"
            job = VideoJob(
                path=Path("dummy_movie.mp4"),
                config=cfg,
                is_film=True,
                base_code=test_base,
                caption_name="Тестовый Полный Фильм",
                metadata={"duration": 7200.0},
            )

            highlights = [
                Highlight(start=10.0, end=40.0, reason="Хайлайт 1", score=9, viral_title="Шок 1"),
                Highlight(start=100.0, end=150.0, reason="Хайлайт 2", score=8, viral_title="Шок 2"),
            ]

            res = await worker.process_job_and_highlights(
                job,
                highlights,
            )

            assert res["total_clips"] == 2
            assert res["video"]["code"] == test_base
            assert highlights[0].code == f"{test_base}-1"
            assert highlights[1].code == f"{test_base}-2"

            # Очистка
            await worker.delete_video(test_base)
        finally:
            await worker.close()

    async def test_ensure_full_video_uploaded(
        self, db_config: DatabaseConfig, tmp_path: Path
    ) -> None:
        cfg = AppConfig.model_validate(
            {
                "videos": {"top_videos": [{"type": "video", "source": "input"}]},
                "params": {"gemini_key": "test"},
            }
        )
        worker = FilmWorker(db_config)
        try:
            await worker.init_pool()

            test_base = "88112"
            movie_file = tmp_path / "movie.mp4"
            movie_file.write_bytes(b"dummy")

            job = VideoJob(
                path=movie_file,
                config=cfg,
                is_film=True,
                base_code=test_base,
                caption_name="Тестовый Фильм Для Загрузки",
                metadata={"duration": 3600.0},
            )

            # Первый вызов — регистрация
            rec1 = await worker.ensure_full_video_uploaded(job)
            assert rec1["code"] == test_base
            assert rec1["title"] == "Тестовый Фильм Для Загрузки"

            # Второй вызов — должен пропустить и вернуть из базы
            rec2 = await worker.ensure_full_video_uploaded(job)
            assert rec2["code"] == test_base

            # Очистка
            await worker.delete_video(test_base)
        finally:
            await worker.close()


def test_iter_videos_film_and_series_rules(tmp_path: Path) -> None:
    from video_gen.__main__ import iter_videos

    # 1. Сериал (директория с сериями)
    series_dir = tmp_path / "my_series"
    series_dir.mkdir()
    (series_dir / "ep1.mp4").write_bytes(b"dummy")
    (series_dir / "ep2.mp4").write_bytes(b"dummy")

    # 2. Фильм (одиночный файл)
    film_file = tmp_path / "standalone_movie.mp4"
    film_file.write_bytes(b"dummy")

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {"type": "video", "source": str(series_dir)},
                    {"type": "video", "source": str(film_file)},
                ]
            },
            "params": {"gemini_key": "test"},
        }
    )

    jobs = iter_videos(cfg)
    assert len(jobs) == 3

    job_s1 = jobs[0]
    assert job_s1.is_film is False
    assert job_s1.base_code is not None
    assert len(job_s1.base_code) == 5
    assert job_s1.video_code == f"{job_s1.base_code}-1"

    job_s2 = jobs[1]
    assert job_s2.is_film is False
    assert job_s2.base_code == job_s1.base_code
    assert job_s2.video_code == f"{job_s1.base_code}-2"

    job_film = jobs[2]
    assert job_film.is_film is True
    assert job_film.base_code is not None
    assert len(job_film.base_code) == 5
    assert job_film.video_code == job_film.base_code


@pytest.mark.anyio
class TestFilmWorkerOffline:
    """Офлайн unit-тесты FilmWorker с моком asyncpg пула."""

    @pytest.fixture
    def anyio_backend(self) -> str:
        return "asyncio"

    @staticmethod
    def _mock_pool(fetchrow_side_effect: object) -> tuple[MagicMock, AsyncMock]:
        conn = AsyncMock()
        conn.fetchrow = AsyncMock(side_effect=fetchrow_side_effect)
        acquire_cm = MagicMock()
        acquire_cm.__aenter__ = AsyncMock(return_value=conn)
        acquire_cm.__aexit__ = AsyncMock(return_value=False)

        pool = MagicMock()
        pool._closed = False
        pool.acquire.return_value = acquire_cm
        return pool, conn

    async def test_register_video_offline(self) -> None:
        pool, conn = self._mock_pool(
            [
                {
                    "id": 1,
                    "code": "12345",
                    "base_code": "12345",
                    "title": "Тестовый фильм",
                    "season": None,
                    "episode": None,
                    "stream_url": None,
                    "is_film": True,
                    "duration": 3600.0,
                    "created_at": None,
                }
            ]
        )
        worker = FilmWorker(DatabaseConfig(host="127.0.0.1", database="videogen_db"))
        worker._pool = pool

        res = await worker.register_video(
            title="Тестовый фильм",
            base_code="12345",
            is_film=True,
            duration=3600.0,
        )
        assert res["id"] == 1
        assert res["code"] == "12345"
        assert res["is_film"] is True
        conn.fetchrow.assert_awaited_once()

    async def test_register_clip_offline(self) -> None:
        pool, conn = self._mock_pool(
            [
                {
                    "id": 7,
                    "episode_code": "12345-1",
                    "code": "12345-1-2",
                    "clip_idx": 2,
                    "timecode_start": 10.0,
                    "timecode_end": 45.5,
                    "duration": 35.5,
                    "hook": "Хук",
                    "viral_title": "Заголовок",
                    "description": "Описание",
                    "render_hash": "abc",
                    "created_at": None,
                }
            ]
        )
        worker = FilmWorker(DatabaseConfig(host="127.0.0.1", database="videogen_db"))
        worker._pool = pool

        res = await worker.register_clip(
            episode_code="12345-1",
            clip_idx=2,
            timecode_start=10.0,
            timecode_end=45.5,
            hook="Хук",
            viral_title="Заголовок",
            description="Описание",
            render_hash="abc",
            is_film=False,
            base_code="12345",
            episode=1,
        )
        assert res["code"] == "12345-1-2"
        assert res["duration"] == pytest.approx(35.5)
        call_args = conn.fetchrow.await_args.args
        assert call_args[1] == "12345-1"
        assert call_args[2] == "12345-1-2"
        assert call_args[6] == pytest.approx(35.5)

    async def test_process_job_and_highlights_offline(self) -> None:
        pool, conn = self._mock_pool(
            [
                {
                    "id": 10,
                    "code": "55555",
                    "base_code": "55555",
                    "title": "Мой фильм",
                    "season": None,
                    "episode": None,
                    "stream_url": None,
                    "is_film": True,
                    "duration": 1800.0,
                    "created_at": None,
                },
                {
                    "id": 11,
                    "episode_code": "55555",
                    "code": "55555-1",
                    "clip_idx": 1,
                    "timecode_start": 5.0,
                    "timecode_end": 25.0,
                    "duration": 20.0,
                    "hook": "Момент 1",
                    "viral_title": "Титл 1",
                    "description": None,
                    "render_hash": "h1",
                    "created_at": None,
                },
                {
                    "id": 12,
                    "episode_code": "55555",
                    "code": "55555-2",
                    "clip_idx": 2,
                    "timecode_start": 30.0,
                    "timecode_end": 55.0,
                    "duration": 25.0,
                    "hook": "Момент 2",
                    "viral_title": "Титл 2",
                    "description": None,
                    "render_hash": "h2",
                    "created_at": None,
                },
            ]
        )
        cfg = AppConfig.model_validate(
            {
                "videos": {"top_videos": [{"type": "video", "source": "input"}]},
                "params": {"gemini_key": "test"},
            }
        )
        worker = FilmWorker(DatabaseConfig(host="127.0.0.1", database="videogen_db"))
        worker._pool = pool

        job = VideoJob(
            path=Path("movie.mp4"),
            config=cfg,
            is_film=True,
            base_code="55555",
            caption_name="Мой фильм",
            metadata={"duration": 1800.0},
        )
        highlights = [
            Highlight(start=5.0, end=25.0, reason="Момент 1", score=9, viral_title="Титл 1"),
            Highlight(start=30.0, end=55.0, reason="Момент 2", score=8, viral_title="Титл 2"),
        ]
        res = await worker.process_job_and_highlights(
            job, highlights, render_hashes={1: "h1", 2: "h2"}
        )

        assert res["total_clips"] == 2
        assert res["video"]["code"] == "55555"
        assert highlights[0].code == "55555-1"
        assert highlights[1].code == "55555-2"
        assert conn.fetchrow.await_count == 3
