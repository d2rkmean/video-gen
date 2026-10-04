from video_gen.config import AppConfig
from video_gen.models import Segment
from video_gen.tasks.gemini_task import GeminiTask, _HighlightSchema


def test_gemini_snap():
    segments = [
        Segment(start=1.0, end=2.0, text="A"),
        Segment(start=3.0, end=4.0, text="B"),
        Segment(start=5.0, end=6.0, text="C"),
    ]
    # start in first gap, end in second gap
    s, e = GeminiTask._snap(2.5, 4.5, segments, 10.0, pad_before=0.1, pad_after=0.1)
    # snap to nearest phrase inside range, which is "B" (3.0-4.0).
    # pad before max 0.1, gap_before is 3.0 - 2.0 = 1.0 (half is 0.5), so pad_before=0.1 -> 3.0 - 0.1 = 2.9
    # pad after max 0.1, gap_after is 5.0 - 4.0 = 1.0 (half is 0.5), so pad_after=0.1 -> 4.0 + 0.1 = 4.1
    assert abs(s - 2.9) < 0.01
    assert abs(e - 4.1) < 0.01


def test_gemini_validate():
    # raw: list[_HighlightSchema], segments, video_end, max_clips, min_dur, max_dur
    raw = [
        _HighlightSchema(start=1.0, end=3.0, reason="a", score=10),
        _HighlightSchema(start=2.0, end=4.0, reason="b", score=9),  # overlap
        _HighlightSchema(start=8.0, end=10.0, reason="c", score=8),  # ok
        _HighlightSchema(start=12.0, end=10.0, reason="d", score=7),  # invalid end<=start
    ]
    segments = [
        Segment(start=1.5, end=2.5, text="A"),
        Segment(start=3.5, end=4.5, text="B"),
        Segment(start=8.5, end=9.5, text="C"),
    ]
    # wait, validation also snaps.
    # r1: 1.0-3.0 -> snaps to A (1.5-2.5) -> pad 0.3/0.5 -> 1.2-3.0 (with max gap logic)
    # let's just make a dummy instance or call classmethod
    # validate is instance method? Wait, _validate is a normal method or static?
    # Let me mock logger and call it.

    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    task = GeminiTask(cfg)

    highlights = task._validate(raw, segments, 15.0, 5, min_dur=0.5, max_dur=10.0)
    assert len(highlights) == 2
    assert highlights[0].score == 10
    assert highlights[1].score == 8
