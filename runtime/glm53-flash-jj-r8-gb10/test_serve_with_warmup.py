from __future__ import annotations

import importlib.util
import copy
import io
import json
import sys
import threading
from email.message import Message
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


class _Stream(io.BytesIO):
    status = 200

    def __init__(self, raw):
        super().__init__(raw)
        self.headers = Message()
        self.headers["Content-Type"] = "text/event-stream; charset=utf-8"


def _chunk(choices, **kwargs):
    return {"id": "warmup-response", "object": "chat.completion.chunk",
            "choices": choices, **kwargs}


def _events(*, finish_reason="stop", completion_tokens=2):
    return [
        _chunk([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]),
        _chunk([{"index": 0, "delta": {"reasoning": "Several tokens in one chunk"},
                 "finish_reason": None}]),
        _chunk([{"index": 0, "delta": {}, "finish_reason": finish_reason}]),
        _chunk([], usage={"prompt_tokens": 12, "completion_tokens": completion_tokens,
                          "total_tokens": 12 + completion_tokens}),
        "[DONE]",
    ]


def _stream(events=None, **kwargs):
    events = _events(**kwargs) if events is None else events
    return _Stream("".join("data: " + (event if isinstance(event, str) else json.dumps(event))
                          + "\n\n" for event in events).encode())


def test_sampling_sweep_covers_filter_and_seed_paths_with_completed_outputs(monkeypatch):
    wrapper, _ = _load_module(monkeypatch)
    observed = []
    clock = [100.0]
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: clock[0])
    monkeypatch.setenv("SPARKRING_STARTUP_TOKEN", "internal-secret")

    def urlopen(request, timeout):
        assert timeout == 10 - len(observed)
        assert request.get_header("Authorization") == "Bearer secret"
        assert request.get_header("X-sparkring-startup-token") == "internal-secret"
        observed.append(json.loads(request.data))
        clock[0] += 1
        return _stream()

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    result = wrapper.warmup_sampling("http://localhost", "model", 16, 10, "secret")
    assert {(body["top_k"], body["top_p"]) for body in observed} == {
        (-1, 1.0), (40, 1.0), (-1, 0.9), (40, 0.9),
    }
    assert any(body["temperature"] == 0.7 and body["top_k"] == -1 and body["top_p"] == 1.0 for body in observed)
    assert any(body.get("seed") == 0 and body["top_k"] == 40 and body["top_p"] == 0.9 for body in observed)
    assert all(body["min_p"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": True} for body in observed)
    assert all(body["stream"] is True and body["n"] == 1 and body["stream_options"] == {
        "include_usage": True, "continuous_usage_stats": False} for body in observed)
    assert len({body["messages"][0]["content"] for body in observed}) == len(observed) == 6
    assert result["coverage"] == "limited-c1-only"
    assert result["concurrent_cases"] == []
    assert result["jit_coverage_verified"] is False
    assert result["stream"] is True
    assert all(case["completion_tokens"] == 2 for case in result["cases"])


@pytest.mark.parametrize("failed_index", range(6))
def test_every_sampling_arm_must_complete_before_readiness(tmp_path, monkeypatch, failed_index):
    wrapper, warmup = _load_module(monkeypatch)
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())
    calls = []

    def urlopen(request, timeout):
        calls.append(request)
        count = 0 if len(calls) == failed_index + 1 else 1
        return _stream(completion_tokens=count)

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    ready = tmp_path / "ready"
    ready.touch()
    with pytest.raises(RuntimeError, match="generated no tokens"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1,), shape_words=(8,), max_tokens=16,
            timeout_seconds=30, credential=None, ready_path=ready,
        )
    assert not ready.exists()
    assert len(calls) == failed_index + 1


@pytest.mark.parametrize("choices", [
    [None], ["invalid"], [{}],
    [{"index": 0, "finish_reason": "error", "delta": {"content": "partial"}}],
    [{"index": 0, "finish_reason": None, "delta": {"content": "partial"}}],
    [{"delta": {"content": "partial"}}],
    [{"index": 0, "finish_reason": "tool_calls", "delta": {"tool_calls": []}}],
    [{"finish_reason": "stop"}],
    [{"finish_reason": "stop", "message": None}],
    [{"finish_reason": "stop", "message": {}}] * 2,
])
def test_invalid_or_unfinished_sampling_choice_withholds_readiness(tmp_path, monkeypatch, choices):
    wrapper, warmup = _load_module(monkeypatch)
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())
    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: _stream([_chunk(choices), "[DONE]"]))
    ready = tmp_path / "ready"
    ready.touch()
    with pytest.raises(RuntimeError, match="Sampling warmup stream"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1,), shape_words=(8,), max_tokens=16,
            timeout_seconds=30, credential=None, ready_path=ready,
        )
    assert not ready.exists()


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_sampling_accepts_completed_stop_or_token_limit(monkeypatch, finish_reason):
    wrapper, _ = _load_module(monkeypatch)
    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: _stream(finish_reason=finish_reason))
    result = wrapper.warmup_sampling("http://localhost", "model", 16, 30, None)
    assert result["coverage"] == "limited-c1-only"
    assert all(case["finish_reason"] == finish_reason for case in result["cases"])


def test_sampling_accepts_comments_blank_lines_and_multiline_data(monkeypatch):
    wrapper, _ = _load_module(monkeypatch)
    frames = []
    for event in _events():
        payload = event if isinstance(event, str) else json.dumps(event, indent=2)
        frames.append(": keepalive\r\n\r\n" + "".join(
            "data: " + line + "\r\n" for line in payload.splitlines()) + "\r\n")
    raw = "\r\n".join(frames).encode()
    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: _Stream(raw))
    result = wrapper.warmup_sampling("http://localhost", "model", 16, 30, None)
    # A chunk can contain several tokens or only metadata; usage supplies totals.
    assert all(case["completion_tokens"] == 2 and case["total_tokens"] == 14
               for case in result["cases"])


def test_concurrent_filters_follow_c1_cases_with_fixed_decode_and_http_overlap(monkeypatch):
    wrapper, _ = _load_module(monkeypatch)
    observed = []
    both_started = threading.Barrier(2)

    def urlopen(request, **kwargs):
        body = json.loads(request.data)
        observed.append(body)
        if body.get("min_tokens") == 128:
            both_started.wait(timeout=5)
            return _stream(completion_tokens=128, finish_reason="length")
        return _stream()

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    # Any requested concurrency above one enables fixed two-request sampler
    # pairs; separate shape warmup exercises the requested concurrency values.
    result = wrapper.warmup_sampling("http://localhost", "model", 16, 30, None, (1, 3))
    assert len(observed) == 12
    assert all("min_tokens" not in body and "ignore_eos" not in body for body in observed[:6])
    for start, filters in ((6, (40, 1.0)), (8, (-1, 0.9)), (10, (40, 0.9))):
        for body in observed[start:start + 2]:
            assert (body["top_k"], body["top_p"]) == filters
            assert body["temperature"] == 1.0 and body["min_p"] == 0.0
            assert "seed" not in body
            assert body["max_tokens"] == body["min_tokens"] == 128
            assert body["ignore_eos"] is True and body["stream"] is True
    assert result["coverage"] == "request-recipe-complete"
    assert result["jit_coverage_verified"] is False
    assert len(result["concurrent_cases"]) == 3
    for case in result["concurrent_cases"]:
        assert case["concurrency"] == 2 and case["http_overlap_seconds"] > 0
        assert [request["completion_tokens"] for request in case["requests"]] == [128, 128]


@pytest.mark.parametrize("filters", [(40, 1.0), (-1, 0.9), (40, 0.9)])
@pytest.mark.parametrize("peer", [0, 1])
def test_each_concurrent_peer_must_complete_before_readiness(tmp_path, monkeypatch, filters, peer):
    wrapper, warmup = _load_module(monkeypatch)
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())
    both_started = threading.Barrier(2)

    def urlopen(request, **kwargs):
        body = json.loads(request.data)
        if body.get("min_tokens") != 128:
            return _stream()
        both_started.wait(timeout=5)
        fail = ((body["top_k"], body["top_p"]) == filters
                and f"peer={peer}" in body["messages"][0]["content"])
        return _stream(completion_tokens=127 if fail else 128, finish_reason="length")

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    ready = tmp_path / "ready"
    ready.touch()
    with pytest.raises(RuntimeError, match="fixed token span"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1, 2), shape_words=(8,), max_tokens=16,
            timeout_seconds=30, credential=None, ready_path=ready,
        )
    assert not ready.exists()


def test_concurrent_stages_do_not_reset_the_startup_deadline(tmp_path, monkeypatch):
    wrapper, warmup = _load_module(monkeypatch)
    clock = [0.0]
    lock = threading.Lock()
    both_started = threading.Barrier(2)
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())

    def urlopen(request, **kwargs):
        paired = json.loads(request.data).get("min_tokens") == 128
        with lock:
            clock[0] += 3 if paired else 1
        if paired:
            both_started.wait(timeout=5)
        return _stream(completion_tokens=128 if paired else 2, finish_reason="length")

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    ready = tmp_path / "ready"
    with pytest.raises(RuntimeError, match="deadline"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1, 2), shape_words=(8,), max_tokens=16,
            timeout_seconds=10, credential=None, ready_path=ready,
        )
    assert clock[0] == 12
    assert not ready.exists()


def _invalid_streams():
    events = _events()
    return [
        b"data: not-json\n\n",
        b"data: \xff\n\n",
        _stream(events[:-1]).getvalue(),
        _stream(events[:2] + [{"error": {"message": "generation failed"}}, "[DONE]"]).getvalue(),
        b"event: error\ndata: {}\n\n",
        _stream(events[:2] + events[3:]).getvalue(),
        _stream(events[:3] + ["[DONE]"]).getvalue(),
        _stream(events[:3] + events[2:]).getvalue(),
        _stream(events[:4] + events[3:]).getvalue(),
        _stream(events + ["[DONE]"]).getvalue(),
        _stream(events + [events[1]]).getvalue(),
        _stream(events[:2] + [_chunk([{"index": 1, "delta": {}, "finish_reason": "stop"}])]).getvalue(),
        _stream(events[:2] + [{**events[2], "id": "other-response"}]).getvalue(),
        b"data: " + b"x" * 65536 + b"\n\n",
    ]


@pytest.mark.parametrize("raw", _invalid_streams(), ids=[
    "json", "utf8", "truncated", "engine-error", "error-event", "missing-finish",
    "missing-usage", "duplicate-finish", "duplicate-usage", "duplicate-done",
    "after-done", "choice-index", "response-id", "oversized-line",
])
def test_malformed_or_incomplete_stream_withholds_readiness(tmp_path, monkeypatch, raw):
    wrapper, warmup = _load_module(monkeypatch)
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())
    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: _Stream(raw))
    ready = tmp_path / "ready"
    ready.touch()
    with pytest.raises(RuntimeError, match="Sampling warmup stream"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1,), shape_words=(8,), max_tokens=16,
            timeout_seconds=30, credential=None, ready_path=ready,
        )
    assert not ready.exists()


@pytest.mark.parametrize("field,value", [
    ("completion_tokens", True), ("completion_tokens", "2"),
    ("completion_tokens", 0), ("completion_tokens", 17),
    ("prompt_tokens", True), ("prompt_tokens", -1),
    ("total_tokens", True), ("total_tokens", 15),
])
def test_sampling_rejects_invalid_or_inconsistent_usage(monkeypatch, field, value):
    wrapper, _ = _load_module(monkeypatch)
    events = copy.deepcopy(_events())
    events[-2]["usage"][field] = value
    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: _stream(events))
    with pytest.raises(RuntimeError, match="Sampling warmup"):
        wrapper.warmup_sampling("http://localhost", "model", 16, 30, None)


def test_stream_consumption_uses_the_remaining_startup_deadline(tmp_path, monkeypatch):
    wrapper, warmup = _load_module(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())

    class SlowStream(_Stream):
        def readline(self, limit):
            clock[0] += 1
            return super().readline(limit)

    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: SlowStream(_stream().getvalue()))
    ready = tmp_path / "ready"
    with pytest.raises(RuntimeError, match="deadline"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1,), shape_words=(8,), max_tokens=16,
            timeout_seconds=3, credential=None, ready_path=ready,
        )
    assert clock[0] == 3
    assert not ready.exists()


def test_api_shapes_and_sampling_share_one_readiness_budget(tmp_path, monkeypatch):
    wrapper, warmup = _load_module(monkeypatch)
    clock = [0.0]
    monkeypatch.setattr(wrapper.time, "monotonic", lambda: clock[0])
    observed = []

    def api(_endpoint, timeout, _credential):
        observed.append(timeout)
        clock[0] += 3

    def shapes(*args):
        observed.append(args[4])
        clock[0] += 4
        return ()

    def sampling(*args):
        observed.append(args[3])
        clock[0] += 4
        return {}

    monkeypatch.setattr(warmup, "wait_for_api", api)
    monkeypatch.setattr(warmup, "run_warmup", shapes)
    monkeypatch.setattr(wrapper, "warmup_sampling", sampling)
    ready = tmp_path / "ready"
    with pytest.raises(RuntimeError, match="deadline"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model", warmup_enabled=True,
            concurrencies=(1,), shape_words=(8,), max_tokens=16,
            timeout_seconds=10, credential=None, ready_path=ready,
        )
    assert observed == [10, 7, 3]
    assert not ready.exists()


def _load_module(monkeypatch):
    warmup_spec = importlib.util.spec_from_file_location(
        "warmup_dflash", HERE / "warmup_dflash.py"
    )
    assert warmup_spec is not None and warmup_spec.loader is not None
    warmup = importlib.util.module_from_spec(warmup_spec)
    warmup_spec.loader.exec_module(warmup)
    monkeypatch.setitem(sys.modules, "warmup_dflash", warmup)
    liveness_spec = importlib.util.spec_from_file_location(
        "scheduler_liveness", HERE / "scheduler_liveness.py"
    )
    assert liveness_spec is not None and liveness_spec.loader is not None
    liveness = importlib.util.module_from_spec(liveness_spec)
    liveness_spec.loader.exec_module(liveness)
    monkeypatch.setitem(sys.modules, "scheduler_liveness", liveness)
    spec = importlib.util.spec_from_file_location(
        "serve_with_warmup", HERE / "serve_with_warmup.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, warmup


def test_rank_zero_marks_ready_only_after_warmup(tmp_path: Path, monkeypatch) -> None:
    wrapper, warmup = _load_module(monkeypatch)
    ready = tmp_path / "ready"
    events = []
    monkeypatch.setattr(
        warmup,
        "wait_for_api",
        lambda *_args: events.append("api"),
    )

    def run(*_args):
        assert not ready.exists()
        events.append("warmup")
        return ({"concurrency": 2},)

    monkeypatch.setattr(warmup, "run_warmup", run)
    monkeypatch.setattr(
        wrapper, "warmup_sampling",
        lambda *_args: events.append("sampling") or {},
    )

    wrapper.complete_readiness(
        rank=0,
        endpoint="http://127.0.0.1:8015",
        model="glm-5.3-flash",
        warmup_enabled=True,
        concurrencies=(1, 2),
        shape_words=(8, 24),
        max_tokens=16,
        timeout_seconds=10,
        credential=None,
        ready_path=ready,
    )

    assert events == ["api", "warmup", "sampling"]
    assert ready.is_file()


def test_sampling_request_uses_auth_temperature_and_reasoning(monkeypatch):
    wrapper, _ = _load_module(monkeypatch)
    observed = []

    def urlopen(request, timeout):
        observed.append(request)
        assert 0 < timeout <= 10
        return _stream(completion_tokens=1)

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    result = wrapper.warmup_sampling("http://localhost/", "model", 16, 10, "secret")
    body = json.loads(observed[0].data)
    assert observed[0].full_url == "http://localhost/v1/chat/completions"
    assert observed[0].get_header("Authorization") == "Bearer secret"
    assert body["temperature"] == 1.0
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == 16
    assert result["enable_thinking"] is True


def test_sampling_failure_withholds_readiness_marker(tmp_path, monkeypatch):
    import pytest

    wrapper, warmup = _load_module(monkeypatch)
    ready = tmp_path / "ready"
    ready.touch()
    monkeypatch.setattr(warmup, "wait_for_api", lambda *_args: None)
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())
    monkeypatch.setattr(wrapper.urllib.request, "urlopen",
                        lambda *_args, **_kwargs: _stream([_chunk([])]))
    with pytest.raises(RuntimeError, match="Sampling warmup stream"):
        wrapper.complete_readiness(
            rank=0, endpoint="http://localhost", model="model",
            warmup_enabled=True, concurrencies=(1,), shape_words=(8,),
            max_tokens=16, timeout_seconds=10, credential=None, ready_path=ready,
        )
    assert not ready.exists()


def test_headless_rank_does_not_call_http_warmup(tmp_path: Path, monkeypatch) -> None:
    wrapper, warmup = _load_module(monkeypatch)
    ready = tmp_path / "ready"
    monkeypatch.setattr(
        warmup,
        "wait_for_api",
        lambda *_args: (_ for _ in ()).throw(AssertionError("headless API wait")),
    )

    wrapper.complete_readiness(
        rank=3,
        endpoint="http://127.0.0.1:8015",
        model="glm-5.3-flash",
        warmup_enabled=True,
        concurrencies=(1,),
        shape_words=(8,),
        max_tokens=16,
        timeout_seconds=10,
        credential=None,
        ready_path=ready,
    )

    assert ready.is_file()


def test_rank_zero_probes_api_with_warmup_credential(tmp_path: Path, monkeypatch) -> None:
    wrapper, warmup = _load_module(monkeypatch)
    ready = tmp_path / "ready"
    probes = []
    monkeypatch.setattr(
        warmup,
        "wait_for_api",
        lambda endpoint, timeout, credential=None: probes.append(
            (endpoint, timeout, credential)
        ),
    )
    monkeypatch.setattr(warmup, "run_warmup", lambda *_args: ())

    wrapper.complete_readiness(
        rank=0,
        endpoint="http://127.0.0.1:8015",
        model="glm-5.3-flash",
        warmup_enabled=False,
        concurrencies=(1,),
        shape_words=(8,),
        max_tokens=16,
        timeout_seconds=10,
        credential="secret",
        ready_path=ready,
    )

    assert len(probes) == 1
    assert probes[0][0] == "http://127.0.0.1:8015"
    assert 0 < probes[0][1] <= 10
    assert probes[0][2] == "secret"
    assert ready.is_file()


def test_rank_zero_starts_scheduler_liveness_service(monkeypatch) -> None:
    wrapper, _warmup = _load_module(monkeypatch)
    liveness = sys.modules["scheduler_liveness"]
    observed = []
    service = object()
    monkeypatch.setattr(
        liveness,
        "start_liveness_service",
        lambda **kwargs: observed.append(kwargs) or service,
    )
    monkeypatch.setenv("SPARKRING_LIVENESS_ENABLED", "1")
    monkeypatch.setenv("SPARKRING_LIVENESS_PORT", "9016")
    monkeypatch.setenv("SPARKRING_LIVENESS_BLOCKED_SECONDS", "75")
    monkeypatch.setenv("SPARKRING_LIVENESS_OUTPUT_SECONDS", "900")
    monkeypatch.setenv("SPARKRING_IDLE_KV_WARN_SECONDS", "360")
    monkeypatch.setenv("SPARKRING_LIVENESS_STALE_SECONDS", "20")
    monkeypatch.setenv("SPARKRING_LIVENESS_SAMPLE_SECONDS", "3")

    result = wrapper.start_rank_liveness(
        rank=0,
        endpoint="http://127.0.0.1:8015",
        credential="secret",
    )

    assert result is service
    assert observed == [
        {
            "metrics_url": "http://127.0.0.1:8015/metrics",
            "port": 9016,
            "blocked_timeout_seconds": 75.0,
            "output_timeout_seconds": 900.0,
            "idle_kv_warn_seconds": 360.0,
            "stale_sample_seconds": 20.0,
            "sample_interval_seconds": 3.0,
            "credential": "secret",
        }
    ]


def test_headless_rank_does_not_start_scheduler_liveness(monkeypatch) -> None:
    wrapper, _warmup = _load_module(monkeypatch)
    monkeypatch.setenv("SPARKRING_LIVENESS_ENABLED", "1")

    assert (
        wrapper.start_rank_liveness(
            rank=2,
            endpoint="http://127.0.0.1:8015",
            credential=None,
        )
        is None
    )


def test_wrapper_launches_gate_before_readiness_and_removes_marker(tmp_path, monkeypatch):
    wrapper, _ = _load_module(monkeypatch)
    ready = tmp_path / "ready"
    ready.touch()
    monkeypatch.setattr(wrapper, "READY_PATH", ready)
    monkeypatch.setenv("SPARKRING_STARTUP_TOKEN", "stale-token")
    monkeypatch.setenv("SPARKRING_LIVENESS_ENABLED", "0")
    monkeypatch.setattr(wrapper.sys, "argv", ["serve-with-warmup.py", "model"])
    monkeypatch.setattr(wrapper.signal, "signal", lambda *_: None)
    started = []

    class Child:
        def wait(self):
            assert ready.exists()
            return 0

    def popen(argv, env):
        assert not ready.exists()
        assert argv[-2:] == ["--middleware", "startup_admission.StartupAdmission"]
        assert env["SPARKRING_STARTUP_TOKEN"] != "stale-token"
        assert len(env["SPARKRING_STARTUP_TOKEN"]) >= 40
        assert env["SPARKRING_STARTUP_TOKEN"] not in argv
        assert str(HERE) in env["PYTHONPATH"].split(wrapper.os.pathsep)
        assert env["SPARKRING_READY_PATH"] == str(ready)
        started.append(env["SPARKRING_STARTUP_TOKEN"])
        return Child()

    def complete(**kwargs):
        assert started == [wrapper.os.environ["SPARKRING_STARTUP_TOKEN"]]
        ready.touch()

    monkeypatch.setattr(wrapper.subprocess, "Popen", popen)
    monkeypatch.setattr(wrapper, "complete_readiness", complete)
    assert wrapper.main() == 0
    assert not ready.exists()


def test_both_warmup_requests_carry_internal_token_and_api_auth(monkeypatch):
    wrapper, warmup = _load_module(monkeypatch)
    monkeypatch.setenv("SPARKRING_STARTUP_TOKEN", "internal-secret")
    observed = []

    def urlopen(request, **kwargs):
        observed.append(request)
        if json.loads(request.data).get("stream"):
            return _stream(completion_tokens=1)
        return io.BytesIO(b'{"choices":[{"finish_reason":"stop","message":{"content":"ok"}}],"usage":{"completion_tokens":1}}')

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    wrapper.warmup_sampling("http://localhost", "model", 16, 10, "api-secret")
    warmup.send_warmup_request("http://localhost", "model", "nonce", 16, 10, 8, "api-secret")
    assert len(observed) == 7
    for request in observed:
        assert request.get_header("X-sparkring-startup-token") == "internal-secret"
        assert request.get_header("Authorization") == "Bearer api-secret"
