"""Deadline classification with fake thread, clock and HTTP responses."""

import io
import json
from types import SimpleNamespace

import pytest
import niah_boundary_probe as probe


@pytest.mark.parametrize("completed,expected,deadline,stall_ticks", [(0.5, 0, 1, 0), (1.0, 0, 1, 0), (1.001, 4, 1, 0), (2.0, 4, 1, 0), (None, 4, 1, 0), (None, 4, 45, 1)])
def test_deadline_never_accepts_a_late_response(monkeypatch, completed, expected, deadline, stall_ticks):
    now = [0.0]
    joins = []
    monkeypatch.setattr(
        probe,
        "ARGS",
        SimpleNamespace(
            seed=1,
            depth=1,
            api="http://invalid",
            api_key="",
            model="fixture",
            deadline=deadline,
            stall_ticks=stall_ticks,
        ),
    )
    monkeypatch.setattr(probe, "build_archive", lambda *args: ("prompt", "12345678"))
    monkeypatch.setattr(probe, "prompt_counter", lambda *args, **kwargs: None)
    monkeypatch.setattr(probe.time, "monotonic", lambda: now[0])
    document = {
        "choices": [{"message": {"content": "12345678"}, "finish_reason": "stop"}]
    }
    monkeypatch.setattr(
        probe.urllib.request,
        "urlopen",
        lambda *args, **kwargs: io.BytesIO(json.dumps(document).encode()),
    )

    class Thread:
        def __init__(self, target, daemon):
            self.target = target
            self.alive = True

        def start(self):
            pass

        def is_alive(self):
            return self.alive

        def join(self, timeout):
            joins.append(timeout)
            now[0] = completed if completed is not None else now[0] + timeout
            if completed is not None:
                self.target()
                self.alive = False

    monkeypatch.setattr(probe.threading, "Thread", Thread)
    with pytest.raises(SystemExit) as result:
        probe.main()
    assert result.value.code == expected
    assert joins and all(0 < timeout <= min(20, deadline) for timeout in joins)
