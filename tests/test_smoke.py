import json
import subprocess

import pytest

from video_gen.config import AppConfig
from video_gen.models import Highlight
from video_gen.tasks.ffmpeg_task import FFmpegTask


@pytest.fixture
def smoke_env(tmp_path):
    top_video = tmp_path / "top.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=20:size=1280x720:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=1000:duration=20",
            "-c:v",
            "h264_nvenc",
            "-c:a",
            "aac",
            str(top_video),
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
            "testsrc=duration=10:size=1280x720:rate=30",
            "-c:v",
            "h264_nvenc",
            str(bottom_dir / "b1.mp4"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=10:size=1280x720:rate=30",
            "-c:v",
            "h264_nvenc",
            str(bottom_dir / "b2.mp4"),
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
            "testsrc=duration=5:size=1280x720:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=500:duration=5",
            "-c:v",
            "h264_nvenc",
            "-c:a",
            "aac",
            str(banner),
        ],
        check=True,
        capture_output=True,
    )

    music_dir = tmp_path / "music"
    music_dir.mkdir()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=400:duration=5",
            "-c:a",
            "aac",
            str(music_dir / "m1.m4a"),
        ],
        check=True,
        capture_output=True,
    )

    subtitles = tmp_path / "sub.ass"
    subtitles.write_text("""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,84,&H0000FFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,0,2,60,60,420,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,{\\kf100}TEST
""")

    fonts_dir = tmp_path / "fonts"
    fonts_dir.mkdir()

    return {
        "top_video": top_video,
        "bottom_dir": bottom_dir,
        "banner": banner,
        "music_dir": music_dir,
        "subtitles": subtitles,
        "fonts_dir": fonts_dir,
    }


@pytest.mark.ffmpeg
def test_smoke_render(smoke_env, tmp_path):
    # BASE
    cfg_base = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input"}],
                "buttom_videos": [{"type": "video", "source": str(smoke_env["bottom_dir"])}],
            },
            "params": {"gemini_key": "k", "output_dir": "out"},
            "assets": {"fonts_dir": str(smoke_env["fonts_dir"])},
        }
    )

    # +BANNER
    cfg_banner = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input"}],
                "buttom_videos": [{"type": "video", "source": str(smoke_env["bottom_dir"])}],
            },
            "params": {"gemini_key": "k", "output_dir": "out"},
            "assets": {
                "fonts_dir": str(smoke_env["fonts_dir"]),
                "banner": str(smoke_env["banner"]),
            },
        }
    )

    # +BANNER+MUSIC
    cfg_all = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input"}],
                "buttom_videos": [{"type": "video", "source": str(smoke_env["bottom_dir"])}],
            },
            "params": {"gemini_key": "k", "output_dir": "out"},
            "assets": {
                "fonts_dir": str(smoke_env["fonts_dir"]),
                "banner": str(smoke_env["banner"]),
                "music_dir": str(smoke_env["music_dir"]),
            },
        }
    )

    highlight = Highlight(start=2.0, end=7.0, reason="test", score=10)  # 5 seconds

    for i, cfg in enumerate([cfg_base, cfg_banner, cfg_all]):
        task = FFmpegTask(cfg)
        out = tmp_path / f"out_{i}.mp4"
        task.run(smoke_env["top_video"], highlight, smoke_env["subtitles"], out)

        proc = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(out),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(proc.stdout)

        dur = float(data["format"]["duration"])
        expected_dur = 8.0 if cfg.assets.banner else 5.0
        assert abs(dur - expected_dur) < 0.2

        vstream = next(s for s in data["streams"] if s["codec_type"] == "video")

        assert vstream["width"] == 1080
        assert vstream["height"] == 1920
        assert vstream["codec_name"] == "h264"
        assert vstream["pix_fmt"] == "yuv420p"
