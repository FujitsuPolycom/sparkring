from __future__ import annotations

import importlib.util
from pathlib import Path


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
