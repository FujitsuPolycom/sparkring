from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


def _load():
    spec = importlib.util.spec_from_file_location(
        "glm53_liveness_gate", HERE / "glm53_liveness_gate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_probe_disables_thinking_and_puts_nonce_first() -> None:
    module = _load()

    payload = module.chat_payload("glm-5.3-flash", "abc123", 10, 1)

    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["messages"][0]["content"].startswith("abc123 ")
    assert payload["temperature"] == 0


def test_idle_requires_no_requests_or_capture_ownership() -> None:
    module = _load()
    baseline = {"kv_usage": 0.20}
    idle = {
        "running": 0,
        "waiting": 0,
        "kv_usage": 0.20,
        "capture_delayed": 0,
        "capture_pages": 0,
        "capture_uncertain": 0,
    }

    assert module.idle_satisfied(idle, baseline, kv_tolerance=0.005)
    assert not module.idle_satisfied(
        {**idle, "capture_pages": 1}, baseline, kv_tolerance=0.005
    )
    assert not module.idle_satisfied(
        {**idle, "kv_usage": 0.21}, baseline, kv_tolerance=0.005
    )


def test_metrics_without_sparkcache_are_supported() -> None:
    module = _load()
    text = "\n".join(
        (
            'vllm:num_requests_running{engine="0"} 0',
            'vllm:num_requests_waiting{engine="0"} 0',
            'vllm:kv_cache_usage_perc{engine="0"} 0.125',
        )
    )

    assert module.parse_metrics(text) == {
        "running": 0.0,
        "waiting": 0.0,
        "kv_usage": 0.125,
        "capture_delayed": 0.0,
        "capture_pages": 0.0,
        "capture_uncertain": 0.0,
    }


@pytest.mark.parametrize("counter", ["running", "waiting", "capture_pages", "capture_delayed", "capture_uncertain"])
def test_every_outstanding_request_or_capture_counter_blocks_idle(counter):
    module = _load()
    idle = dict(running=0, waiting=0, capture_pages=0, capture_delayed=0,
                capture_uncertain=0, kv_usage=0)
    assert module.idle_satisfied(idle, {"kv_usage": 0}, kv_tolerance=0.125)
    assert not module.idle_satisfied({**idle, counter: 1}, {"kv_usage": 0}, kv_tolerance=0.125)


def test_kv_usage_tolerance_includes_its_exact_boundary():
    module = _load()
    idle = dict(running=0, waiting=0, capture_pages=0, capture_delayed=0,
                capture_uncertain=0, kv_usage=0.125)
    # Binary-exact fractions keep this boundary test independent of decimal rounding.
    assert module.idle_satisfied(idle, {"kv_usage": 0}, kv_tolerance=0.125)
    assert not module.idle_satisfied({**idle, "kv_usage": 0.25}, {"kv_usage": 0}, kv_tolerance=0.125)


@pytest.mark.parametrize("value", ["NaN", "+Inf", "1e999", "-1"])
def test_invalid_metric_values_cannot_establish_idle(value):
    module = _load()
    with pytest.raises((ValueError, RuntimeError)):
        module._metric_sum("counter " + value, "counter", required=False)


def test_cache_enabled_gate_requires_capture_metrics():
    module = _load()
    text = "vllm:num_requests_running 0\nvllm:num_requests_waiting 0\nvllm:kv_cache_usage_perc 0"
    with pytest.raises(RuntimeError, match="sparkcache"):
        module.parse_metrics(text, require_capture_metrics=True)


@pytest.mark.parametrize("document", [{"error": "failed"}, {"choices": []},
    {"choices": [{"message": {"role": "assistant"}, "finish_reason": None}], "usage": {"completion_tokens": 1}}])
def test_http_success_without_completed_chat_is_not_liveness(monkeypatch, document):
    import json
    module = _load()
    client = module.Client("http://invalid", None)
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: json.dumps(document).encode())
    with pytest.raises(RuntimeError):
        client.chat({})


def test_busy_baseline_rejects_before_chat(monkeypatch):
    from types import SimpleNamespace
    module = _load()
    idle = dict(running=1, waiting=0, capture_pages=0, capture_delayed=0, capture_uncertain=0, kv_usage=0.2)
    monkeypatch.setattr(module, "Client", lambda *args, **kwargs: SimpleNamespace(metrics=lambda: idle, chat=lambda *args: pytest.fail("submitted while busy")))
    args = SimpleNamespace(duration_seconds=0, drain_timeout_seconds=1, kv_tolerance=0.005, endpoint="unused", api_key_file=None)
    with pytest.raises(RuntimeError, match="baseline must be idle"):
        module.run(args)


def test_completed_chat_returns_latency_without_http(monkeypatch):
    import json
    module = _load()
    client = module.Client("http://invalid", None)
    document = {"choices": [{"message": {"role": "assistant"}, "finish_reason": "length"}], "usage": {"completion_tokens": 1}}
    monkeypatch.setattr(client, "_request", lambda *args, **kwargs: json.dumps(document).encode())
    assert client.chat({}) >= 0


def test_idle_response_after_drain_deadline_is_rejected(monkeypatch):
    from types import SimpleNamespace
    module = _load()
    clock = [0]
    samples = [0]
    idle = dict(running=0, waiting=0, capture_pages=0, capture_delayed=0, capture_uncertain=0, kv_usage=0)
    def metrics(timeout=10):
        samples[0] += 1
        if samples[0] > 1:
            clock[0] = 2
        return idle
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(module, "Client", lambda *args, **kwargs: SimpleNamespace(metrics=metrics, chat=lambda payload: 0.1))
    args = SimpleNamespace(duration_seconds=0, drain_timeout_seconds=1, kv_tolerance=0, endpoint="unused", api_key_file=None,
        cycles=1, concurrency=1, prompt_words=1, max_tokens=1, model="fixture")
    with pytest.raises(RuntimeError, match="deadline"):
        module.run(args)
