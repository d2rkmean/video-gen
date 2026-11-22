"""Агрессивные интеграционные и CLI-тесты video-gen (subprocess, FFmpeg, конкурентность, кэш)."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from video_gen.__main__ import _parse_cli_args, make_subtitle_task, run_pipeline
from video_gen.callbacks import JsonProgressCallbacks
from video_gen.config import AppConfig, PartialPipelineParams, VideoSource, get_effective_config
from video_gen.models import Highlight, Segment, Word
from video_gen.tasks.ffmpeg_task import FFmpegTask, _resolve_codec_preset, _video_profile
from video_gen.utils.cache import GeminiCache

CLI_BIN = Path(sys.executable).parent / "video-gen"


def _run_cli(args: list[str], cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    return subprocess.run(
        [str(CLI_BIN), *args],
        cwd=str(cwd),
        env=run_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


def test_cli_flags_and_error_exit_codes(tmp_path: Path) -> None:
    res_help = _run_cli(["--help"], tmp_path)
    assert res_help.returncode == 0
    assert "--json-progress" in res_help.stdout

    res_h = _run_cli(["-h"], tmp_path)
    assert res_h.returncode == 0

    res_unknown = _run_cli(["--definitely-unknown"], tmp_path)
    assert res_unknown.returncode == 2
    assert "Неизвестный флаг" in res_unknown.stderr

    res_extra = _run_cli(["a.json", "b.json"], tmp_path)
    assert res_extra.returncode == 2

    res_missing = _run_cli(["nonexistent.json"], tmp_path)
    assert res_missing.returncode == 2
    assert "Traceback" not in res_missing.stderr
    assert "не найден" in res_missing.stderr

    bad_json = tmp_path / "broken.json"
    bad_json.write_text("{not valid json", encoding="utf-8")
    res_bad_json = _run_cli([str(bad_json)], tmp_path)
    assert res_bad_json.returncode == 2
    assert "Traceback" not in res_bad_json.stderr

    strict_bad = tmp_path / "strict_bad.json"
    strict_bad.write_text(
        json.dumps(
            {
                "videos": {
                    "top_videos": [{"type": "video", "source": "input/"}],
                    "bottom_videos": [{"type": "video", "source": "bottom/"}],
                },
                "params": {"unknown_typo_field": 123},
            }
        ),
        encoding="utf-8",
    )
    res_strict = _run_cli([str(strict_bad)], tmp_path)
    assert res_strict.returncode == 2
    assert "Traceback" not in res_strict.stderr

    empty_src = tmp_path / "empty_src.json"
    empty_src.write_text(
        json.dumps(
            {
                "videos": {
                    "top_videos": [{"type": "video", "source": str(tmp_path / "no_such_dir")}],
                    "bottom_videos": [{"type": "video", "source": str(tmp_path / "no_bot")}],
                },
                "params": {"output_dir": str(tmp_path / "out")},
            }
        ),
        encoding="utf-8",
    )
    res_empty = _run_cli([str(empty_src), "--json-progress"], tmp_path)
    assert res_empty.returncode == 1


def test_parse_cli_args_env_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VIDEOGEN_JSON_PROGRESS", "1")
    cfg_path, use_json, rc = _parse_cli_args(["custom.json"])
    assert cfg_path == "custom.json"
    assert use_json is True
    assert rc is None


def test_partial_pipeline_params_merge_adjusts_durations(tmp_path: Path) -> None:
    global_cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/"}],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {
                "output_dir": str(tmp_path / "out"),
                "min_part_duration": 25,
                "part_duration": 60,
            },
        }
    )
    src = VideoSource(
        type="video",
        source="input/short.mp4",
        params=PartialPipelineParams(part_duration=15),
    )
    eff = get_effective_config(global_cfg, src)
    assert eff.params.part_duration == 15
    assert eff.params.min_part_duration < eff.params.part_duration

    src_high_min = VideoSource(
        type="video",
        source="input/long.mp4",
        params=PartialPipelineParams(min_part_duration=80),
    )
    eff2 = get_effective_config(global_cfg, src_high_min)
    assert eff2.params.min_part_duration == 80
    assert eff2.params.part_duration > eff2.params.min_part_duration


def test_from_toml_reads_real_toml(tmp_path: Path) -> None:
    toml_file = tmp_path / "legacy.toml"
    out_dir = tmp_path / "out"
    toml_file.write_text(
        f'''
[[videos.top_videos]]
type = "video"
source = "input/"

[[videos.buttom_videos]]
type = "video"
source = "bottom/"

[params]
output_dir = "{out_dir}"
parts_count = 3

[telegram]
enabled = false
''',
        encoding="utf-8",
    )
    cfg = AppConfig.from_toml(toml_file)
    assert cfg.params.parts_count == 3
    assert cfg.params.output_dir == out_dir


def test_gemini_cache_preserves_description_and_codes(tmp_path: Path) -> None:
    cache = GeminiCache(cache_dir=tmp_path / "gem_cache")
    h = Highlight(
        start=1.0,
        end=25.0,
        reason="punchline",
        score=9,
        hook="wait for it",
        viral_title="Wild ending",
        description="Watch till the end #shorts",
        code="04291-1",
        clip_idx=1,
    )
    key = cache.get_key("seg_hash", "gpt-4o-mini", 5, 20, 60)
    cache.save(key, [h], {"video": "demo.mp4"})
    loaded = cache.load(key)
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded[0].description == "Watch till the end #shorts"
    assert loaded[0].code == "04291-1"
    assert loaded[0].clip_idx == 1


def test_make_subtitle_task_propagates_resolution(tmp_path: Path) -> None:
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/"}],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"output_dir": str(tmp_path / "out")},
            "render": {"width": 720, "height": 1280},
        }
    )
    task = make_subtitle_task(cfg)
    assert task.style.res_x == 720
    assert task.style.res_y == 1280


def test_codec_preset_and_profile_helpers() -> None:
    assert _resolve_codec_preset("libx264", "p4") == "fast"
    assert _resolve_codec_preset("libx265", "p1") == "ultrafast"
    assert _resolve_codec_preset("h264_nvenc", "p4") == "p4"
    assert _video_profile("h264_nvenc") == "high"
    assert _video_profile("libx264") == "high"
    assert _video_profile("libx265") == "main"
    assert _video_profile("h265_nvenc") == "main"


@pytest.fixture
def synthetic_media(tmp_path: Path) -> dict[str, Path]:
    top_video = tmp_path / "top.mp4"
    bottom_dir = tmp_path / "bottom"
    bottom_dir.mkdir()
    bottom_video = bottom_dir / "bg.mp4"

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=duration=12:size=640x360:rate=24",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:duration=12",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-c:a",
            "aac",
            str(top_video),
        ],
        check=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=duration=12:size=640x360:rate=24",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            str(bottom_video),
        ],
        check=True,
    )
    return {"top": top_video, "bottom_dir": bottom_dir}


@pytest.mark.ffmpeg
@pytest.mark.parametrize("codec", ["libx264", "libx265"])
def test_real_ffmpeg_cpu_codecs_with_nvenc_default_preset_and_strip(
    synthetic_media: dict[str, Path], tmp_path: Path, codec: str
) -> None:
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": str(synthetic_media["top"])}],
                "bottom_videos": [{"type": "video", "source": str(synthetic_media["bottom_dir"])}],
            },
            "params": {"output_dir": str(tmp_path / "out")},
            "assets": {"fonts_dir": str(tmp_path / "fonts")},
            "render": {
                "width": 360,
                "height": 640,
                "fps": 24,
                "video_codec": codec,
                "preset": "p4",
                "strip_metadata": True,
                "face_tracking": False,
                "enable_antidetect": False,
            },
        }
    )
    out = tmp_path / f"clip_{codec}.mp4"
    pcts: list[float] = []
    FFmpegTask(cfg).run(
        synthetic_media["top"],
        Highlight(start=1.0, end=4.0, reason="test", score=8),
        None,
        out,
        progress_cb=pcts.append,
    )
    assert out.is_file() and out.stat().st_size > 1024
    assert pcts and pcts[-1] == 100.0

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_streams",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    vstream = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    expected_name = "hevc" if codec == "libx265" else "h264"
    assert vstream["codec_name"] == expected_name
    assert vstream["width"] == 360
    assert vstream["height"] == 640


@pytest.mark.ffmpeg
def test_pipeline_concurrent_renders_and_cache_hit_video_done(
    synthetic_media: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "out"
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": str(synthetic_media["top"])}],
                "bottom_videos": [{"type": "video", "source": str(synthetic_media["bottom_dir"])}],
            },
            "params": {
                "output_dir": str(out_dir),
                "mode": "highlights",
                "concurrent_renders": 2,
                "min_part_duration": 5,
                "part_duration": 10,
                "use_whisper_cache": False,
                "use_gemini_cache": False,
                "use_render_cache": True,
                "transcribe_backend": "openai",
            },
            "assets": {"fonts_dir": str(tmp_path / "fonts")},
            "render": {
                "width": 360,
                "height": 640,
                "fps": 24,
                "video_codec": "libx264",
                "preset": "p1",
                "face_tracking": False,
                "enable_antidetect": False,
            },
        }
    )

    fake_segments = [
        Segment(
            start=0.5,
            end=5.0,
            text="Первая тестовая фраза.",
            words=[Word(0.5, 2.0, "Первая"), Word(2.1, 3.5, "тестовая"), Word(3.6, 5.0, "фраза.")],
        ),
        Segment(
            start=5.5,
            end=10.5,
            text="Вторая тестовая фраза.",
            words=[Word(5.5, 7.0, "Вторая"), Word(7.1, 8.8, "тестовая"), Word(8.9, 10.5, "фраза.")],
        ),
    ]
    fake_highlights = [
        Highlight(start=0.5, end=5.0, reason="c1", score=9, viral_title="Clip 1"),
        Highlight(start=5.5, end=10.5, reason="c2", score=8, viral_title="Clip 2"),
    ]

    transcribe_use_cache_flags: list[bool] = []

    def _fake_transcribe_run(_self: object, _video: Path, **kwargs: object) -> list[Segment]:
        transcribe_use_cache_flags.append(bool(kwargs.get("use_cache", True)))
        return fake_segments

    def _fake_analyze_run(_self: object, _segs: list[Segment], **_kwargs: object) -> list[Highlight]:
        return [Highlight(**h.__dict__) for h in fake_highlights]

    monkeypatch.setattr("video_gen.tasks.TranscribeTask.run", _fake_transcribe_run)
    monkeypatch.setattr("video_gen.tasks.BestPartsTask.run", _fake_analyze_run)

    events_run1: list[dict[str, object]] = []

    class _CaptureCb(JsonProgressCallbacks):
        def __init__(self, sink: list[dict[str, object]]) -> None:
            super().__init__()
            self._sink = sink

        def _emit(self, event: str, **data: object) -> None:
            self._sink.append({"event": event, **data})

    stats1 = asyncio.run(
        run_pipeline(cfg, [synthetic_media["top"]], callbacks=_CaptureCb(events_run1))
    )
    assert stats1 == {"ok": 2, "failed": 0, "broken_videos": 0}
    assert transcribe_use_cache_flags == [False]
    video_done_1 = [e for e in events_run1 if e["event"] == "video_done"]
    assert len(video_done_1) == 1
    assert video_done_1[0]["broken"] is False

    events_run2: list[dict[str, object]] = []
    stats2 = asyncio.run(
        run_pipeline(cfg, [synthetic_media["top"]], callbacks=_CaptureCb(events_run2))
    )
    assert stats2 == {"ok": 2, "failed": 0, "broken_videos": 0}
    video_done_2 = [e for e in events_run2 if e["event"] == "video_done"]
    assert len(video_done_2) == 1
    assert video_done_2[0]["broken"] is False
