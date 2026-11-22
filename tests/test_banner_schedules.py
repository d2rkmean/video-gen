import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from video_gen.config import (
    AppConfig,
    PartialRenderConfig,
    RenderConfig,
    get_effective_config,
)
from video_gen.models import Highlight
from video_gen.tasks.ffmpeg_task import (
    FFmpegTask,
    RenderSettings,
    _compute_banner_timestamps,
    _FreezePlan,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Config & PartialRenderConfig Validation
# ─────────────────────────────────────────────────────────────────────────────


def test_render_config_banner_schedule_defaults():
    rc = RenderConfig()
    assert rc.banner_mode == "center"
    assert rc.banner_interval_minutes == 1.0
    assert rc.banner_interval_percent == 25.0
    assert rc.banner_freeze is True


def test_render_config_banner_schedule_valid():
    rc_min = RenderConfig(
        banner_mode="interval_minutes",
        banner_interval_minutes=2.5,
        banner_freeze=False,
    )
    assert rc_min.banner_mode == "interval_minutes"
    assert rc_min.banner_interval_minutes == 2.5
    assert rc_min.banner_freeze is False

    rc_pct = RenderConfig(
        banner_mode="interval_percent",
        banner_interval_percent=33.3,
        banner_freeze=True,
    )
    assert rc_pct.banner_mode == "interval_percent"
    assert rc_pct.banner_interval_percent == 33.3
    assert rc_pct.banner_freeze is True


def test_render_config_banner_schedule_bounds():
    with pytest.raises(ValidationError):
        RenderConfig(banner_interval_minutes=0.0)
    with pytest.raises(ValidationError):
        RenderConfig(banner_interval_minutes=-1.0)
    with pytest.raises(ValidationError):
        RenderConfig(banner_interval_minutes=181.0)

    with pytest.raises(ValidationError):
        RenderConfig(banner_interval_percent=0.0)
    with pytest.raises(ValidationError):
        RenderConfig(banner_interval_percent=-5.0)
    with pytest.raises(ValidationError):
        RenderConfig(banner_interval_percent=101.0)


def test_partial_render_config_banner_schedule_merge():
    prc = PartialRenderConfig(
        banner_mode="interval_percent",
        banner_interval_minutes=0.5,
        banner_interval_percent=20.0,
        banner_freeze=False,
    )
    assert prc.banner_mode == "interval_percent"
    assert prc.banner_interval_minutes == 0.5
    assert prc.banner_interval_percent == 20.0
    assert prc.banner_freeze is False

    with pytest.raises(ValidationError):
        PartialRenderConfig(banner_interval_minutes=0.0)
    with pytest.raises(ValidationError):
        PartialRenderConfig(banner_interval_percent=120.0)

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {
                        "type": "video",
                        "source": "input/",
                        "render": {
                            "banner_mode": "interval_minutes",
                            "banner_interval_minutes": 0.25,
                            "banner_freeze": False,
                        },
                    }
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "test"},
        }
    )
    eff = get_effective_config(cfg, cfg.videos.top_videos[0])
    assert eff.render.banner_mode == "interval_minutes"
    assert eff.render.banner_interval_minutes == 0.25
    assert eff.render.banner_interval_percent == 25.0
    assert eff.render.banner_freeze is False


# ─────────────────────────────────────────────────────────────────────────────
# 2. _compute_banner_timestamps Edge Cases
# ─────────────────────────────────────────────────────────────────────────────


def test_compute_banner_timestamps():
    assert _compute_banner_timestamps(0.0, "center", 1.0, 25.0) == []
    assert _compute_banner_timestamps(-5.0, "interval_percent", 1.0, 25.0) == []
    assert _compute_banner_timestamps(10.0, "overlay", 1.0, 25.0) == []

    assert _compute_banner_timestamps(10.0, "center", 1.0, 25.0) == [5.0]
    assert _compute_banner_timestamps(10.0, "end", 1.0, 25.0) == [10.0]

    # interval_minutes: клип короче интервала -> пусто
    assert _compute_banner_timestamps(30.0, "interval_minutes", 1.0, 25.0) == []
    # interval_minutes: точное кратное (3s шаг на 6s клипе -> 3s и 6s)
    assert _compute_banner_timestamps(6.0, "interval_minutes", 0.05, 25.0) == pytest.approx(
        [3.0, 6.0]
    )
    # interval_minutes: некратная длина (3s шаг на 7s клипе -> 3s и 6s)
    assert _compute_banner_timestamps(7.0, "interval_minutes", 0.05, 25.0) == pytest.approx(
        [3.0, 6.0]
    )

    # interval_percent: 25% на 8s -> 25%, 50%, 75% = [2.0, 4.0, 6.0]
    assert _compute_banner_timestamps(8.0, "interval_percent", 1.0, 25.0) == pytest.approx(
        [2.0, 4.0, 6.0]
    )
    # interval_percent: 50% на 10s -> [5.0]
    assert _compute_banner_timestamps(10.0, "interval_percent", 1.0, 50.0) == pytest.approx([5.0])
    # interval_percent: 100% -> в самом конце [d]
    assert _compute_banner_timestamps(10.0, "interval_percent", 1.0, 100.0) == pytest.approx([10.0])


# ─────────────────────────────────────────────────────────────────────────────
# 3. Filter Graph Generation (Freeze & Non-Freeze Multi-Banner)
# ─────────────────────────────────────────────────────────────────────────────


def test_filter_graph_multi_banner_freeze():
    s = RenderSettings(banner_mode="interval_percent", banner_interval_percent=25.0, banner_freeze=True)
    freezes = [_FreezePlan(ta=2.0, bd=1.5), _FreezePlan(ta=4.0, bd=1.5), _FreezePlan(ta=6.0, bd=1.5)]
    windows = [(2.0, 3.5), (5.5, 7.0), (9.0, 10.5)]
    info = {
        "i_top": 0,
        "top_has_audio": True,
        "bottom_idx": [1],
        "i_banner": 2,
        "b_audio": True,
        "banner_mode": "interval_percent",
        "banner_freeze": True,
        "windows": windows,
        "freezes": freezes,
        "d_src": 8.0,
        "i_music": None,
    }
    graph, has_audio = FFmpegTask._build_filter_graph(s, 12.5, info, None, Path("fonts"))
    assert has_audio
    assert "gte((PTS*TB),2.000000)" in graph
    assert "gte((PTS*TB),4.000000)" in graph
    assert "gte((PTS*TB),6.000000)" in graph
    assert "split=3[bns0][bns1][bns2]" in graph
    assert "between(t,2.000,3.500)+between(t,5.500,7.000)+between(t,9.000,10.500)" in graph
    assert "sendcmd=" in graph
    assert "gblur@bl" in graph
    assert "asplit=3[abs0][abs1][abs2]" in graph
    assert "adelay=2000:all=1[ab0]" in graph
    assert "adelay=5500:all=1[ab1]" in graph
    assert "adelay=9000:all=1[ab2]" in graph
    assert "concat=n=7:v=0:a=1" in graph


def test_filter_graph_multi_banner_freeze_at_end_uses_tpad():
    s = RenderSettings(banner_mode="interval_minutes", banner_freeze=True)
    freezes = [_FreezePlan(ta=3.0, bd=1.5), _FreezePlan(ta=6.0, bd=1.5)]
    windows = [(3.0, 4.5), (7.5, 9.0)]
    info = {
        "i_top": 0,
        "top_has_audio": True,
        "bottom_idx": [1],
        "i_banner": 2,
        "b_audio": False,
        "banner_mode": "interval_minutes",
        "banner_freeze": True,
        "windows": windows,
        "freezes": freezes,
        "d_src": 6.0,
        "i_music": None,
    }
    graph, has_audio = FFmpegTask._build_filter_graph(s, 9.0, info, None, Path("fonts"))
    assert has_audio
    assert "tpad=stop_mode=clone:stop_duration=1.500[padded]" in graph
    # Trailing segment at 6.0..6.0 skipped -> 2 speech segs + 2 gaps = 4 concat inputs
    assert "concat=n=4:v=0:a=1" in graph


def test_filter_graph_multi_banner_non_freeze():
    s = RenderSettings(
        banner_mode="interval_percent",
        banner_interval_percent=25.0,
        banner_freeze=False,
        banner_opacity=0.8,
        duck_volume=0.15,
    )
    windows = [(2.0, 3.5), (4.0, 5.5), (6.0, 7.5)]
    info = {
        "i_top": 0,
        "top_has_audio": True,
        "bottom_idx": [1],
        "i_banner": 2,
        "b_audio": True,
        "banner_mode": "interval_percent",
        "banner_freeze": False,
        "windows": windows,
        "freezes": [],
        "d_src": 8.0,
        "i_music": None,
    }
    graph, has_audio = FFmpegTask._build_filter_graph(s, 8.0, info, None, Path("fonts"))
    assert has_audio
    assert "tpad=" not in graph
    assert "gblur" not in graph
    assert "colorchannelmixer=aa=0.800" in graph
    assert "split=3[bns0][bns1][bns2]" in graph
    assert "volume='0.15':enable='between(t,2.000,3.500)+between(t,4.000,5.500)+between(t,6.000,7.500)'" in graph
    assert "asplit=3[abs0][abs1][abs2]" in graph
    assert "anullsrc" not in graph
    assert "v=0:a=1" not in graph


# ─────────────────────────────────────────────────────────────────────────────
# 4. End-to-End Real FFmpeg Render Tests (libx264, no NVENC GPU required)
# ─────────────────────────────────────────────────────────────────────────────


def _probe_duration(path: Path) -> float:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(json.loads(proc.stdout)["format"]["duration"])


@pytest.fixture
def synthetic_media(tmp_path: Path) -> dict[str, Path]:
    top = tmp_path / "top.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=10:size=640x360:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=800:duration=10",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(top),
        ],
        check=True,
        capture_output=True,
    )

    bottom_dir = tmp_path / "bottom"
    bottom_dir.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=blue:duration=10:size=640x360:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(bottom_dir / "bg.mp4"),
        ],
        check=True,
        capture_output=True,
    )

    banner = tmp_path / "banner.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:duration=1.5:size=320x180:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=1.5",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(banner),
        ],
        check=True,
        capture_output=True,
    )

    return {"top": top, "bottom_dir": bottom_dir, "banner": banner}


def _make_render_task(
    bottom_dir: Path,
    banner: Path,
    mode: str,
    interval_minutes: float,
    interval_percent: float,
    freeze: bool,
) -> FFmpegTask:
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input"}],
                "bottom_videos": [{"type": "video", "source": str(bottom_dir)}],
            },
            "params": {"gemini_key": "k", "output_dir": str(bottom_dir.parent / "out")},
            "assets": {"banner": str(banner), "fonts_dir": str(bottom_dir.parent / "fonts")},
            "render": {
                "width": 360,
                "height": 640,
                "fps": 30,
                "video_codec": "libx264",
                "preset": "ultrafast",
                "cq": 28,
                "enable_antidetect": False,
                "speed_jitter": False,
                "face_tracking": False,
                "banner_width": 240,
                "banner_chromakey": False,
                "banner_autocrop": False,
                "banner_mode": mode,
                "banner_interval_minutes": interval_minutes,
                "banner_interval_percent": interval_percent,
                "banner_freeze": freeze,
            },
        }
    )
    return FFmpegTask(cfg)


@pytest.mark.parametrize(
    ("mode", "interval_min", "interval_pct", "freeze", "clip_dur", "expected_dur"),
    [
        # 0.05 min = 3.0s на 7s клипе -> 2 баннера по 1.5s => 7 + 3.0 = 10.0s (freeze) / 7.0s (non-freeze)
        ("interval_minutes", 0.05, 25.0, True, 7.0, 10.0),
        ("interval_minutes", 0.05, 25.0, False, 7.0, 7.0),
        # 25% на 8s клипе -> 3 баннера (2s, 4s, 6s) по 1.5s => 8 + 4.5 = 12.5s (freeze) / 8.0s (non-freeze)
        ("interval_percent", 1.0, 25.0, True, 8.0, 12.5),
        ("interval_percent", 1.0, 25.0, False, 8.0, 8.0),
    ],
)
def test_e2e_ffmpeg_banner_schedules(
    synthetic_media: dict[str, Path],
    tmp_path: Path,
    mode: str,
    interval_min: float,
    interval_pct: float,
    freeze: bool,
    clip_dur: float,
    expected_dur: float,
):
    task = _make_render_task(
        synthetic_media["bottom_dir"],
        synthetic_media["banner"],
        mode=mode,
        interval_minutes=interval_min,
        interval_percent=interval_pct,
        freeze=freeze,
    )
    out = tmp_path / f"out_{mode}_{freeze}.mp4"
    highlight = Highlight(start=0.0, end=clip_dur, reason="test", score=10)
    task.run(synthetic_media["top"], highlight, None, out)

    assert out.is_file()
    actual_dur = _probe_duration(out)
    assert abs(actual_dur - expected_dur) < 0.25
