from video_gen.config import AppConfig
from video_gen.models import Highlight, Segment, Word
from video_gen.tasks.subtitle_task import SubtitleStyle, SubtitleTask


def test_subtitle_ts():
    assert SubtitleTask._ts(0) == "0:00:00.00"
    assert SubtitleTask._ts(61.5) == "0:01:01.50"
    assert SubtitleTask._ts(3605.123) == "1:00:05.12"


def test_subtitle_collect_words():
    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    task = SubtitleTask(cfg)
    segments = [
        Segment(
            start=1.0,
            end=3.0,
            text="hello world",
            words=[Word(1.0, 2.0, "hello"), Word(2.0, 3.0, "world")],
        )
    ]
    h = Highlight(start=1.5, end=3.0, reason="", score=10)
    words = task._collect_words(segments, h)
    assert len(words) == 2
    # shift by 1.5, bound to duration (1.5)
    # first word "hello" is 1.0-2.0. Relative to 1.5 -> -0.5 to 0.5 -> bounded 0.0 to 0.5
    assert words[0].start == 0.0
    assert words[0].end == 0.5
    assert words[0].text == "HELLO"


def test_subtitle_chunk():
    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    task = SubtitleTask(cfg, max_words=3)
    words = [
        Word(0.0, 1.0, "A"),
        Word(1.0, 2.0, "B"),
        Word(2.0, 3.0, "C"),
        Word(3.0, 4.0, "D"),
        Word(4.0, 5.0, "E"),
    ]
    chunks = task._chunk(words)
    assert len(chunks) == 2
    assert len(chunks[0]) == 3
    assert len(chunks[1]) == 2


def test_subtitle_dialogue():
    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    task = SubtitleTask(cfg)
    chunk = [Word(0.0, 1.0, "HELLO"), Word(1.0, 2.0, "WORLD")]
    d = task._dialogue(chunk, None, 5.0)
    assert "Dialogue:" in d
    assert "{\\kf100}HELLO {\\kf100}WORLD" in d
    assert r"\fscx92\fscy92\t(0,75,\fscx100\fscy100)" in d


def test_subtitle_animations():
    cfg = AppConfig.model_validate(
        {
            "videos": {"top_videos": [{"type": "video", "source": "input"}]},
            "params": {"gemini_key": "k"},
        }
    )
    # bounce
    task_bounce = SubtitleTask(cfg, style=SubtitleStyle(animation="bounce"))
    d_bounce = task_bounce._dialogue([Word(0.0, 1.0, "BOUNCE")], None, 2.0)
    assert r"\fscx88\fscy88" in d_bounce

    # none
    task_none = SubtitleTask(cfg, style=SubtitleStyle(animation="none"))
    d_none = task_none._dialogue([Word(0.0, 1.0, "STATIC")], None, 2.0)
    assert r"\fscx" not in d_none

    # header checks
    header = task_bounce._header()
    assert "Unbounded" in header
    assert "ScaledBorderAndShadow: yes" in header
