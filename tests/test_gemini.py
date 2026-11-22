from video_gen.config import AppConfig
from video_gen.tasks.best_parts_task import BestPartsTask as GeminiTask, _HighlightSchema, _Unit


def test_gemini_bounds():
    units = [
        _Unit(start=1.0, end=2.0, text="A"),
        _Unit(start=3.0, end=4.0, text="B"),
        _Unit(start=5.0, end=6.0, text="C"),
    ]
    # For unit 1 ("B"): i=1, j=1.
    # gap_before = 3.0 - 2.0 = 1.0 -> pad_before is min(0.1, 0.5) = 0.1 -> start = 2.9
    # gap_after = 5.0 - 4.0 = 1.0 -> pad_after is min(0.1, 0.5) = 0.1 -> end = 4.1
    s, e = GeminiTask._bounds(1, 1, units, 10.0, pad_before=0.1, pad_after=0.1)
    assert abs(s - 2.9) < 0.01
    assert abs(e - 4.1) < 0.01


def test_gemini_validate():
    raw = [
        _HighlightSchema(
            start_line="L0", end_line="L0", reason="first", score=10, hook="h1", viral_title="t1"
        ),
        _HighlightSchema(
            start_line="L0", end_line="L1", reason="overlap", score=9
        ),  # overlaps with L0
        _HighlightSchema(
            start_line="L2", end_line="L2", reason="ok", score=8, hook="h2", viral_title="t2"
        ),
        _HighlightSchema(start_line="L4", end_line="L3", reason="invalid end<start", score=7),
    ]
    units = [
        _Unit(start=1.0, end=3.0, text="A"),
        _Unit(start=3.5, end=5.5, text="B"),
        _Unit(start=6.0, end=8.0, text="C"),
    ]

    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    task = GeminiTask(cfg)

    highlights = task._validate(
        raw, units, 15.0, max_clips=5, min_dur=1.0, max_dur=10.0, expand_short=False
    )
    assert len(highlights) == 2
    assert highlights[0].score == 10
    assert highlights[0].viral_title == "t1"
    assert highlights[1].score == 8
    assert highlights[1].viral_title == "t2"
