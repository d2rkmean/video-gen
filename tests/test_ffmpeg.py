import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from video_gen.config import AppConfig
from video_gen.tasks.ffmpeg_task import FFmpegTask, RenderSettings
from video_gen.utils.audio_utils import detect_preferred_audio_stream


def test_ffmpeg_plan_bottom():
    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    task = FFmpegTask(cfg)

    # mock _bottom_files and _probe
    with (
        patch.object(task, "_bottom_files", return_value=[Path("a.mp4"), Path("b.mp4")]),
        patch("video_gen.tasks.ffmpeg_task._probe", side_effect=lambda x: (10.0, False)),
    ):
        plan = task._plan_bottom(15.0)
        # need 15.0 seconds, each video is 10.0s. It should pick 2 segments.
        assert len(plan) == 2
        total = sum(take for _, off, take in plan)
        assert abs(total - 15.0) < 0.01


def test_ffmpeg_build_filter_graph():
    s = RenderSettings()
    info = {
        "i_top": 0,
        "top_has_audio": True,
        "bottom_idx": [1, 2],
        "i_banner": None,
        "b_audio": False,
        "t0": 0.0,
        "t1": 0.0,
        "i_music": None,
    }
    f, has_audio = FFmpegTask._build_filter_graph(s, 10.0, info, None, Path("fonts"))
    assert "[top]" in f
    assert "[bot]" in f
    assert "vstack=inputs=2" in f
    assert "unsharp=" in f
    assert "eq=contrast=" in f
    assert has_audio

    info["i_banner"] = 3
    info["b_audio"] = True
    info["t0"] = 2.0
    info["t1"] = 5.0
    f2, _ = FFmpegTask._build_filter_graph(s, 10.0, info, None, Path("fonts"))
    assert "overlay" in f2
    assert "gblur" in f2
    assert "split[main][tmp]" not in f2  # Verify memory leak fix (no buffering split)
    assert "enable='between(t,2.000,5.000)'" in f2

    info["i_music"] = 4
    f3, _ = FFmpegTask._build_filter_graph(s, 10.0, info, None, Path("fonts"))
    assert "[amus]" in f3
    assert "amix" in f3


def test_detect_preferred_audio_stream():
    mock_out = json.dumps(
        {
            "streams": [
                {"index": 0, "tags": {"language": "eng", "title": "English"}},
                {"index": 1, "tags": {"language": "rus", "title": "MTV Дубляж"}},
            ]
        }
    )
    with patch("subprocess.run", return_value=MagicMock(stdout=mock_out)):
        # Clear lru_cache for fresh mock run
        detect_preferred_audio_stream.cache_clear()
        idx = detect_preferred_audio_stream(Path("south_park.mkv"))
        assert idx == 1
