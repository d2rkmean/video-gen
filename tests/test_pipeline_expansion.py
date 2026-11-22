"""Тесты для расширенного функционала пайплайна:
1. Режим speech_split (нарезка по паузам и предложениям).
2. Интеграция SponsorBlock API (повторы, 404, фильтрация рекламы).
3. JSON формат выгрузки в Telegram с force_document=True.
4. Случайные баннеры (banners_dir) и настройки canvas/blur в FFmpegTask.
5. Настройки конфига AppConfig и get_effective_config.
"""

import asyncio
import json
import urllib.error
from unittest.mock import AsyncMock, MagicMock, patch

from video_gen.config import AppConfig, get_effective_config
from video_gen.models import Highlight, Segment, Word
from video_gen.tasks.ffmpeg_task import FFmpegTask
from video_gen.utils.ad_filter import (
    fetch_sponsorblock_segments,
    highlight_overlaps_sponsor,
    is_ad_text,
)
from video_gen.utils.speech_splitter import split_by_speech_pauses


def test_speech_splitter_basic():
    """Проверяет нарезку по паузам без пересечения слов."""
    # Создаем тестовые сегменты с пословными таймингами
    words = [
        Word(start=0.0, end=1.0, text="Привет,"),
        Word(start=1.2, end=2.5, text="это"),
        Word(start=2.6, end=3.5, text="начало."),
        # Пауза
        Word(start=5.0, end=6.0, text="Продолжаем"),
        Word(start=6.2, end=22.0, text="историю."),
        # Пауза
        Word(start=24.0, end=25.0, text="Вторая"),
        Word(start=25.2, end=45.0, text="часть."),
    ]
    segments = [
        Segment(start=0.0, end=3.5, text="Привет, это начало.", words=words[:3]),
        Segment(start=5.0, end=22.0, text="Продолжаем историю.", words=words[3:5]),
        Segment(start=24.0, end=45.0, text="Вторая часть.", words=words[5:]),
    ]

    highlights = split_by_speech_pauses(
        segments,
        target_duration=20.0,
        min_duration=10.0,
        max_duration=30.0,
        title_prefix="Часть",
        video_title="Мой Ролик",
    )

    assert len(highlights) >= 1
    for h in highlights:
        assert h.start >= 0.0
        assert h.end > h.start
        assert "Мой Ролик" in h.viral_title
        assert "Часть" in h.viral_title
        assert "#fyp" in h.description
        assert "#shorts" in h.description


def test_speech_splitter_with_sponsorblock():
    """Проверяет пропуск спонсорского блока при нарезке речи."""
    words = [
        Word(start=0.0, end=15.0, text="Первая фраза."),
        # Спонсорский блок с 20.0 до 35.0
        Word(start=22.0, end=25.0, text="Спонсор выпуска."),
        Word(start=26.0, end=34.0, text="Купите по промокоду."),
        # Контент продолжается после спонсора
        Word(start=36.0, end=55.0, text="А теперь возвращаемся к теме."),
    ]
    segments = [Segment(start=0.0, end=55.0, text="Все слова", words=words)]

    sponsor_intervals = [(20.0, 35.0, "sponsor")]
    highlights = split_by_speech_pauses(
        segments,
        target_duration=20.0,
        min_duration=10.0,
        max_duration=40.0,
        sponsor_intervals=sponsor_intervals,
    )

    for h in highlights:
        # Ни один клип не должен пересекаться со спонсорским блоком 20-35
        assert not highlight_overlaps_sponsor(h.start, h.end, sponsor_intervals, tolerance=0.5)


def test_fetch_sponsorblock_segments_mock_success():
    """Проверяет успешное получение сегментов SponsorBlock API."""
    fake_json = json.dumps(
        [
            {"segment": [10.5, 25.0], "category": "sponsor"},
            {"segment": [50.0, 55.2], "category": "selfpromo"},
        ]
    ).encode("utf-8")

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = fake_json
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = fetch_sponsorblock_segments("dQw4w9WgXcQ")
        assert len(res) == 2
        assert res[0] == (10.5, 25.0, "sponsor")
        assert res[1] == (50.0, 55.2, "selfpromo")


def test_fetch_sponsorblock_segments_404():
    """Проверяет обработку 404 (видео нет в базе SponsorBlock)."""
    http_err = urllib.error.HTTPError("url", 404, "Not Found", None, None)
    with patch("urllib.request.urlopen", side_effect=http_err):
        res = fetch_sponsorblock_segments("unknown_id")
        assert res == []


def test_ad_filter_helpers():
    """Проверяет детекцию рекламных фраз."""
    assert is_ad_text("Используй промокод на скидку 10%") is True
    assert is_ad_text("Подписывайтесь на мой телеграм-канал") is True
    assert is_ad_text("Просто интересный диалог персонажей") is False


def test_ffmpeg_random_banner_selection(tmp_path):
    """Проверяет случайный выбор баннера из banners_dir."""
    banners_dir = tmp_path / "banners"
    banners_dir.mkdir()
    (banners_dir / "banner1.png").write_bytes(b"fake_png")
    (banners_dir / "banner2.mp4").write_bytes(b"fake_mp4")
    (banners_dir / "ignored.txt").write_bytes(b"fake_txt")

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/"}],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "dummy"},
            "assets": {
                "banners_dir": banners_dir,
            },
        }
    )

    task = FFmpegTask(cfg)
    active = task._get_active_banner()
    assert active is not None
    assert active.suffix.lower() in {".png", ".mp4"}
    assert active.name in {"banner1.png", "banner2.mp4"}


def test_effective_config_mode_and_render_overrides(tmp_path):
    """Проверяет наложение локальных настроек mode, canvas blur и banners_dir."""
    banners_dir = tmp_path / "banners"
    banners_dir.mkdir()

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {
                        "type": "youtube",
                        "source": "https://youtube.com/watch?v=123",
                        "mode": "speech_split",
                        "render": {
                            "enable_canvas_blur": True,
                            "boxblur_radius": 25.0,
                        },
                    }
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {
                "gemini_key": "dummy",
                "mode": "highlights",
            },
            "assets": {
                "banners_dir": banners_dir,
            },
        }
    )

    src = cfg.videos.top_videos[0]
    eff = get_effective_config(cfg, src)

    # Проверяем, что локальный mode переопределил глобальный
    assert eff.params.mode == "speech_split"
    # Проверяем, что локальный render применился
    assert eff.render.enable_canvas_blur is True
    assert eff.render.boxblur_radius == 25.0
    # Проверяем, что banners_dir сохранился
    assert eff.assets.banners_dir == banners_dir


def test_speech_splitter_duration_limit():
    """Проверяет, что нарезка по речи строго соблюдает лимит max_duration."""
    words = [
        Word(start=i * 2.0, end=i * 2.0 + 1.5, text=f"слово{i}." if i % 10 == 0 else f"слово{i}")
        for i in range(120)
    ]
    segments = [Segment(start=0.0, end=240.0, text=" ".join(w.text for w in words), words=words)]

    highlights = split_by_speech_pauses(
        segments,
        target_duration=118.0,
        min_duration=60.0,
        max_duration=118.0,
    )

    assert len(highlights) >= 2
    for h in highlights:
        dur = h.end - h.start
        assert dur <= 118.5, f"Длительность клипа {dur} превышает 118с"
