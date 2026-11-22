import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

from video_gen.__main__ import iter_videos
from video_gen.config import AppConfig, VideoSource, get_effective_config
from video_gen.gameplay.generator import generate_pygame_video
from video_gen.models import Highlight, Segment, Word
from video_gen.tasks.ffmpeg_task import FFmpegTask
from video_gen.tasks.youtube_task import YouTubeTask
from video_gen.utils.ad_filter import (
    filter_highlights_ads,
    highlight_overlaps_sponsor,
    is_ad_text,
)
from video_gen.utils.cache import (
    GeminiCache,
    RenderCache,
    WhisperCache,
    compute_data_hash,
    compute_file_hash,
)
from video_gen.utils.setup_logger import logcat_formatter


# -------------------------------------------------------------
# 1. YouTube Source Tests
# -------------------------------------------------------------
def test_youtube_config_validation():
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {"type": "youtube", "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )
    assert cfg.videos.top_videos[0].type == "youtube"


def test_youtube_task_mock_download(tmp_path, monkeypatch):
    cache_dir = tmp_path / "yt_cache"
    fake_video = cache_dir / "dQw4w9WgXcQ.mp4"
    cache_dir.mkdir(parents=True)
    fake_video.write_bytes(b"dummy mp4 data" * 100)

    mock_ydl = MagicMock()
    mock_ydl.extract_info.return_value = {
        "id": "dQw4w9WgXcQ",
        "title": "Rick Astley",
        "duration": 212.0,
        "webpage_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    }
    mock_ydl.download.return_value = 0

    monkeypatch.setattr(
        "yt_dlp.YoutubeDL",
        lambda *args, **kwargs: MagicMock(__enter__=lambda s: mock_ydl, __exit__=lambda *a: None),
    )
    monkeypatch.setattr(
        "video_gen.tasks.youtube_task.fetch_sponsorblock_segments",
        lambda vid, **_kw: [(10.0, 20.0, "sponsor")],
    )

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {"type": "youtube", "source": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )

    task = YouTubeTask(cfg, cache_dir=cache_dir)
    res = task.download("https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    assert len(res) == 1
    file_path, meta = res[0]
    assert file_path == fake_video
    assert meta["id"] == "dQw4w9WgXcQ"
    assert meta["sponsor_intervals"] == [(10.0, 20.0, "sponsor")]


def test_iter_videos_with_youtube(tmp_path, monkeypatch):
    cache_file = tmp_path / "yt_vid.mp4"
    cache_file.write_bytes(b"video data")

    mock_download = MagicMock(
        return_value=[(cache_file, {"id": "yt_vid", "sponsor_intervals": []})]
    )
    monkeypatch.setattr("video_gen.tasks.youtube_task.YouTubeTask.download", mock_download)

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "youtube", "source": "https://youtube.com/watch?v=test"}],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )

    jobs = iter_videos(cfg)
    assert len(jobs) == 1
    assert jobs[0].path == cache_file
    assert jobs[0].source_idx == 1
    assert jobs[0].metadata["id"] == "yt_vid"


# -------------------------------------------------------------
# 2. Ad and Sponsor Filter Tests
# -------------------------------------------------------------
def test_ad_text_heuristics():
    assert is_ad_text("Промокод на скидку 20% в описании")
    assert is_ad_text("Спонсор сегодняшнего ролика — онлайн игра")
    assert is_ad_text("Переходи по ссылке в закреплённом комментарии")
    assert is_ad_text("Подписывайтесь на наш телеграм-канал")
    assert is_ad_text("1xbet большие выигрыши")
    assert is_ad_text("Ставьте лайк и жмите колокольчик!")
    assert not is_ad_text("Картман, отойди от микроволновки, это опасно!")
    assert not is_ad_text("Сегодня на уроке мистер Гаррисон расскажет про историю")


def test_highlight_overlaps_sponsor():
    intervals = [(15.0, 30.0, "sponsor"), (100.0, 110.0, "intro")]
    assert highlight_overlaps_sponsor(20.0, 25.0, intervals)
    assert highlight_overlaps_sponsor(10.0, 22.0, intervals)
    assert not highlight_overlaps_sponsor(40.0, 60.0, intervals)


def test_filter_highlights_ads():
    segments = [
        Segment(start=10.0, end=15.0, text="Спонсор выпуска — магазин...", words=[]),
        Segment(start=40.0, end=50.0, text="Обычный диалог персонажей", words=[]),
    ]
    h_ad = Highlight(start=10.0, end=15.0, reason="ad", score=10)
    h_clean = Highlight(start=40.0, end=50.0, reason="clean", score=10)
    sponsor_intervals = [(10.0, 20.0, "sponsor")]

    filtered = filter_highlights_ads([h_ad, h_clean], segments, sponsor_intervals)
    assert len(filtered) == 1
    assert filtered[0].start == 40.0


# -------------------------------------------------------------
# 3. PyGame Generative Bottom Video Tests
# -------------------------------------------------------------
def test_pygame_config_and_generation(tmp_path):
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/"}],
                "bottom_videos": [{"type": "pygame", "source": "runner"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )
    assert cfg.videos.bottom_videos[0].type == "pygame"

    out_file = tmp_path / "pygame_test.mp4"
    # Быстрый тест генерации 1 секунды 320x240 на 15 fps
    res = generate_pygame_video(
        preset="runner",
        width=320,
        height=240,
        fps=15,
        duration=1.0,
        output_path=out_file,
    )
    assert res.is_file()
    assert res.stat().st_size > 1000

    # Проверка кэширования
    cached_res = generate_pygame_video(
        preset="runner",
        width=320,
        height=240,
        fps=15,
        duration=1.0,
        output_path=out_file,
    )
    assert cached_res == res


def test_ffmpeg_task_bottom_files_with_pygame(tmp_path, monkeypatch):
    fake_py_video = tmp_path / "py_bg.mp4"
    fake_py_video.write_bytes(b"dummy")

    monkeypatch.setattr(
        "video_gen.tasks.ffmpeg_task.generate_pygame_video",
        lambda *args, **kwargs: fake_py_video,
    )

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/"}],
                "bottom_videos": [{"type": "pygame", "source": "runner"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )
    task = FFmpegTask(cfg)
    bottom_files = task._bottom_files(need=20.0)
    assert len(bottom_files) == 1
    assert bottom_files[0] == fake_py_video


# -------------------------------------------------------------
# 4. Worker Caching Tests
# -------------------------------------------------------------
def test_cache_hashing_and_lifecycle(tmp_path):
    test_file = tmp_path / "sample.bin"
    test_file.write_bytes(b"A" * 10000)

    hash1 = compute_file_hash(test_file)
    assert len(hash1) == 64

    # WhisperCache
    w_cache = WhisperCache(tmp_path / "cache_w")
    key = w_cache.get_key(hash1, "large-v3", "ru", 5, 0)
    assert w_cache.load(key) is None

    segs = [Segment(start=0.0, end=2.0, text="Привет", words=[Word(0.0, 2.0, "Привет")])]
    w_cache.save(key, segs, {"test": 1})
    loaded_segs = w_cache.load(key)
    assert loaded_segs is not None
    assert len(loaded_segs) == 1
    assert loaded_segs[0].text == "Привет"

    # GeminiCache
    g_cache = GeminiCache(tmp_path / "cache_g")
    g_key = g_cache.get_key(compute_data_hash("test"), "gemini-3.5-flash-lite", 5, 20, 60)
    assert g_cache.load(g_key) is None

    highs = [Highlight(start=1.0, end=5.0, reason="good", score=9)]
    g_cache.save(g_key, highs, {})
    loaded_highs = g_cache.load(g_key)
    assert loaded_highs is not None
    assert loaded_highs[0].score == 9

    # RenderCache
    mp4_file = tmp_path / "clip.mp4"
    mp4_file.write_bytes(b"dummy video data" * 200)
    r_key = RenderCache.get_key(hash1, 1.0, 5.0, {"w": 1080}, {"font": "Arial"})

    assert not RenderCache.is_cached(mp4_file, r_key)

    meta_file = mp4_file.with_suffix(".json")
    meta_file.write_text(json.dumps({"render_hash": r_key}), encoding="utf-8")
    assert RenderCache.is_cached(mp4_file, r_key)


# -------------------------------------------------------------
# 5. Output Dir & Naming Convention Tests
# -------------------------------------------------------------
def test_per_source_output_dir_and_naming(tmp_path):
    custom_out = tmp_path / "custom_output"
    source = VideoSource(
        type="video",
        source=str(tmp_path / "v1.mp4"),
        output_dir=custom_out,
        source_number=7,
    )
    base_cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/"}],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "k", "output_dir": str(tmp_path / "default_out")},
        }
    )

    effective = get_effective_config(base_cfg, source)
    assert effective.params.output_dir == custom_out


def test_iter_videos_caption_propagation(tmp_path):
    sub = tmp_path / "series"
    sub.mkdir()
    f1 = sub / "ep1.mp4"
    f1.write_bytes(b"data")

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {
                        "type": "video",
                        "source": str(sub),
                        "caption_name": "Тестовый Сериал",
                        "caption": "$caption_name, часть $source_idx, нарезка $part_idx",
                    }
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "k"},
        }
    )
    jobs = iter_videos(cfg)
    assert len(jobs) == 1
    assert jobs[0].caption_name == "Тестовый Сериал"
    assert jobs[0].caption == "$caption_name, часть $source_idx, нарезка $part_idx"


# -------------------------------------------------------------
# 7. Structured Logs Formatter Tests
# -------------------------------------------------------------
def test_logcat_formatter_badges():
    record = {
        "extra": {"worker": "TestWorker"},
        "name": "test",
        "level": MagicMock(name="INFO", icon="I"),
        "message": "[CACHE HIT] [YOUTUBE] Файл video0101.mp4 загружен",
    }
    record["level"].name = "INFO"
    record["level"].icon = "I"

    formatted = logcat_formatter(record)
    assert "CACHE HIT" in formatted
    assert "YOUTUBE" in formatted
    assert "TestWorker" in formatted
