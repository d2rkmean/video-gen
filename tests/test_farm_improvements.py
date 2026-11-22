import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from video_gen.__main__ import iter_videos
from video_gen.config import (
    AppConfig,
    PartialPipelineParams,
    PartialRenderConfig,
    PipelineParams,
    RenderConfig,
    VideoSource,
    get_effective_config,
)
from video_gen.models import Highlight, VideoJob
from video_gen.tasks.ffmpeg_task import FFmpegTask, RenderSettings, _anchor_expr


# ─────────────────────────────────────────────────────────────────────────────
# 1. Тесты расширенных настроек баннера и режима overlay
# ─────────────────────────────────────────────────────────────────────────────


def test_banner_mode_defaults():
    rc = RenderConfig()
    assert rc.banner_mode == "center"
    assert rc.banner_anchor == "center"
    assert rc.banner_x is None
    assert rc.banner_y is None
    assert rc.banner_opacity == 1.0
    assert rc.banner_scale == 1.0


def test_banner_mode_custom_valid():
    rc = RenderConfig(
        banner_mode="overlay",
        banner_anchor="bottom-right",
        banner_x=-20,
        banner_y=-30,
        banner_opacity=0.85,
        banner_scale=1.5,
    )
    assert rc.banner_mode == "overlay"
    assert rc.banner_anchor == "bottom-right"
    assert rc.banner_x == -20
    assert rc.banner_y == -30
    assert rc.banner_opacity == 0.85
    assert rc.banner_scale == 1.5


def test_banner_mode_invalid():
    with pytest.raises(ValidationError):
        RenderConfig(banner_mode="unsupported_mode")


def test_banner_anchor_invalid():
    with pytest.raises(ValidationError):
        RenderConfig(banner_anchor="diagonal")


def test_banner_opacity_out_of_bounds():
    with pytest.raises(ValidationError):
        RenderConfig(banner_opacity=1.5)
    with pytest.raises(ValidationError):
        RenderConfig(banner_opacity=-0.1)


def test_banner_scale_out_of_bounds():
    with pytest.raises(ValidationError):
        RenderConfig(banner_scale=0.05)
    with pytest.raises(ValidationError):
        RenderConfig(banner_scale=10.0)


def test_partial_render_config_banner_fields():
    prc = PartialRenderConfig(
        banner_mode="overlay",
        banner_anchor="top-left",
        banner_opacity=0.5,
        banner_scale=2.0,
    )
    assert prc.banner_mode == "overlay"
    assert prc.banner_anchor == "top-left"
    assert prc.banner_opacity == 0.5
    assert prc.banner_scale == 2.0


def test_anchor_expr_calculations():
    # Без смещений
    assert _anchor_expr("top-left") == ("0", "0")
    assert _anchor_expr("top-right") == ("W-w", "0")
    assert _anchor_expr("bottom-left") == ("0", "H-h")
    assert _anchor_expr("bottom-right") == ("W-w", "H-h")
    assert _anchor_expr("center") == ("(W-w)/2", "(H-h)/2")

    # С положительными смещениями
    assert _anchor_expr("top-left", 15, 25) == ("(0)+15", "(0)+25")
    # С отрицательными смещениями
    assert _anchor_expr("bottom-right", -10, -20) == ("(W-w)-10", "(H-h)-20")


def test_ffmpeg_build_banner_overlay_mode():
    s = RenderSettings(
        banner_mode="overlay",
        banner_anchor="top-left",
        banner_x=10,
        banner_y=20,
        banner_opacity=0.8,
        banner_scale=1.2,
        banner_width=500,
        banner_chromakey=False,
    )
    info = {
        "i_banner": 2,
        "banner_mode": "overlay",
        "active_banner": Path("banner.mp4"),
    }
    filters, cur = FFmpegTask._build_banner_overlay(s, "cur_layer", info)
    assert cur == "ov"
    full_f = ";".join(filters)
    assert "colorchannelmixer=aa=0.800" in full_f
    assert "scale=600:-2" in full_f  # 500 * 1.2 = 600
    assert "overlay=(0)+10:(0)+20" in full_f


# ─────────────────────────────────────────────────────────────────────────────
# 2. Тесты codename для источников и выгрузки в Telegram
# ─────────────────────────────────────────────────────────────────────────────


def test_video_source_codename():
    src = VideoSource(type="video", source="input/test", codename="GGSEL")
    assert src.codename == "GGSEL"

    src_no_code = VideoSource(type="video", source="input/test")
    assert src_no_code.codename is None


def test_video_job_codename():
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [{"type": "video", "source": "input/test", "codename": "PLAYEROK"}],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )
    job = VideoJob(
        path=Path("input/test/clip.mp4"),
        config=cfg,
        codename="PLAYEROK",
    )
    assert job.codename == "PLAYEROK"


def test_iter_videos_propagates_codename(tmp_path):
    video_dir = tmp_path / "top_dir"
    video_dir.mkdir()
    (video_dir / "ep1.mp4").touch()

    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {
                        "type": "video",
                        "source": str(video_dir),
                        "caption_name": "Тест Сериал",
                        "codename": "GGSEL",
                    }
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {"gemini_key": "test_key"},
        }
    )
    jobs = iter_videos(cfg)
    assert len(jobs) == 1
    assert jobs[0].codename == "GGSEL"
    assert jobs[0].caption_name == "Тест Сериал"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Тесты флагов Use Cache для каждого воркера
# ─────────────────────────────────────────────────────────────────────────────


def test_pipeline_params_cache_flags_defaults():
    params = PipelineParams(gemini_key="test")
    assert params.use_whisper_cache is True
    assert params.use_gemini_cache is True
    assert params.use_render_cache is True


def test_pipeline_params_cache_flags_override():
    params = PipelineParams(
        gemini_key="test",
        use_whisper_cache=False,
        use_gemini_cache=False,
        use_render_cache=False,
    )
    assert params.use_whisper_cache is False
    assert params.use_gemini_cache is False
    assert params.use_render_cache is False


def test_partial_pipeline_params_cache_flags():
    partial = PartialPipelineParams(
        use_whisper_cache=False,
        use_gemini_cache=True,
    )
    assert partial.use_whisper_cache is False
    assert partial.use_gemini_cache is True
    assert partial.use_render_cache is None


def test_effective_config_cache_flags_inheritance():
    cfg = AppConfig.model_validate(
        {
            "videos": {
                "top_videos": [
                    {
                        "type": "video",
                        "source": "input/",
                        "params": {"use_whisper_cache": False},
                    }
                ],
                "bottom_videos": [{"type": "video", "source": "bottom/"}],
            },
            "params": {
                "gemini_key": "test_key",
                "use_whisper_cache": True,
                "use_gemini_cache": True,
            },
        }
    )
    eff = get_effective_config(cfg, cfg.videos.top_videos[0])
    assert eff.params.use_whisper_cache is False
    assert eff.params.use_gemini_cache is True
    assert eff.params.use_render_cache is True


def test_render_settings_banner_overlay_attrs():
    rs = RenderSettings(
        banner_mode="overlay",
        banner_anchor="bottom-left",
        banner_x=50,
        banner_y=-50,
        banner_opacity=0.75,
        banner_scale=2.5,
    )
    assert rs.banner_mode == "overlay"
    assert rs.banner_anchor == "bottom-left"
    assert rs.banner_x == 50
    assert rs.banner_y == -50
    assert rs.banner_opacity == 0.75
    assert rs.banner_scale == 2.5


def test_video_source_all_new_fields():
    src = VideoSource(
        type="video",
        source="input/test",
        codename="GGSEL",
        render=PartialRenderConfig(
            banner_mode="overlay",
            banner_anchor="top-right",
            banner_opacity=0.9,
            banner_scale=1.1,
        ),
        params=PartialPipelineParams(
            use_whisper_cache=False,
            use_gemini_cache=True,
            use_render_cache=False,
        ),
    )
    assert src.codename == "GGSEL"
    assert src.render.banner_mode == "overlay"
    assert src.render.banner_anchor == "top-right"
    assert src.params.use_whisper_cache is False
    assert src.params.use_render_cache is False
