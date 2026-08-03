from __future__ import annotations

import exl3_live_gate


def test_completion_payload_is_deterministic():
    payload = exl3_live_gate.completion_payload("model", "prompt", 64)
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 20260731
    assert payload["max_tokens"] == 64


def test_performance_cell_reports_aggregate_rate(monkeypatch):
    monkeypatch.setattr(
        exl3_live_gate,
        "one_completion",
        lambda *_args, **_kwargs: exl3_live_gate.Completion("ok", 32),
    )
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr(exl3_live_gate.time, "perf_counter", lambda: next(ticks))
    cell = exl3_live_gate.performance_cell(
        "http://example", "model", "prompt", 32, 1.0, 2
    )
    assert cell["completion_tokens"] == 64
    assert cell["elapsed_seconds"] == 2.0
    assert cell["aggregate_tokens_per_second"] == 32.0
