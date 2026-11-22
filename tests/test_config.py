import json
import pytest
from pathlib import Path
from video_gen.config import AppConfig

MINIMAL = {
    "videos": {
        "top_videos": [{"source": "test.mp4", "type": "video", "caption": "Test"}],
        "bottom_videos": [{"source": "game.mp4", "type": "video"}],
    },
    "params": {
        "analysis_api_key": "test-key",
        "output_dir": "/tmp/vg_test",
    },
}


def test_minimal_config():
    cfg = AppConfig.model_validate(MINIMAL)
    assert cfg.params.analysis_model == "gpt-4o-mini"
    assert not hasattr(cfg, "telegram")


def test_from_json(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(MINIMAL), encoding="utf-8")
    cfg = AppConfig.from_json(p)
    assert cfg.render.width == 1080


def test_transcribe_backends():
    for backend in ("local", "groq", "openai"):
        data = dict(MINIMAL)
        data["params"] = {**MINIMAL["params"], "transcribe_backend": backend}
        cfg = AppConfig.model_validate(data)
        assert cfg.params.transcribe_backend == backend


def test_render_params():
    data = dict(MINIMAL)
    data["render"] = {"width": 720, "height": 1280, "fps": 60, "cq": 20}
    cfg = AppConfig.model_validate(data)
    assert cfg.render.fps == 60
    assert cfg.render.cq == 20


def test_subtitle_params():
    data = dict(MINIMAL)
    data["subtitles"] = {"font_size": 100, "bold": True, "pop_scale": 1.5}
    cfg = AppConfig.model_validate(data)
    assert cfg.subtitles.font_size == 100
    assert cfg.subtitles.bold == True


def test_no_telegram_section():
    cfg = AppConfig.model_validate(MINIMAL)
    assert not hasattr(cfg, "telegram")
