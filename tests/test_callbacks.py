import json
from io import StringIO
from unittest.mock import patch
from video_gen.callbacks import JsonProgressCallbacks


def capture(fn):
    buf = StringIO()
    with patch("builtins.print", side_effect=lambda *a, **kw: buf.write(str(a[0]) + "\n")):
        fn()
    lines = [line for line in buf.getvalue().strip().splitlines() if line]
    return [json.loads(line) for line in lines]


def test_pipeline_started():
    cb = JsonProgressCallbacks()
    events = capture(lambda: cb.on_pipeline_started(2, ["a.mp4", "b.mp4"]))
    assert events[0]["event"] == "pipeline_started"
    assert events[0]["total"] == 2


def test_render_progress():
    cb = JsonProgressCallbacks()
    events = capture(lambda: cb.on_render_progress("clip1", 55.5))
    assert events[0]["pct"] == 55.5


def test_pipeline_finished():
    cb = JsonProgressCallbacks()
    events = capture(lambda: cb.on_pipeline_finished({"ok": 5, "failed": 1, "broken_videos": 0}))
    assert events[0]["ok"] == 5


def test_log_event():
    cb = JsonProgressCallbacks()
    events = capture(lambda: cb.on_log("INFO", "test message"))
    assert events[0]["level"] == "INFO"
