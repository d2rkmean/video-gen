import os

import pytest
from pydantic import ValidationError

from video_gen.config import AppConfig


def test_config_typo():
    data = {
        "videos": {"top_videos": [{"type": "video", "source": "input"}]},
        "params": {"output_dir": "out", "gemini_key": "k", "gemini_model_typo": "model"},
    }

    # wait, AppConfig.from_toml reads file. I can use model_validate
    with pytest.raises(ValidationError):
        AppConfig.model_validate(data)


def test_config_min_duration_error():
    data = {
        "videos": {"top_videos": [{"type": "video", "source": "input"}]},
        "params": {
            "output_dir": "out",
            "gemini_key": "k",
            "min_part_duration": 60,
            "part_duration": 50,
        },
    }
    with pytest.raises(ValidationError, match="min_part_duration должен быть меньше part_duration"):
        AppConfig.model_validate(data)


def test_config_gemini_key_env():
    data = {
        "videos": {"top_videos": [{"type": "video", "source": "input"}]},
        "params": {
            "output_dir": "out",
        },
    }
    os.environ["GEMINI_API_KEY"] = "secret_env_key"
    cfg = AppConfig.model_validate(data)
    assert cfg.params.gemini_key == "secret_env_key"
    del os.environ["GEMINI_API_KEY"]
