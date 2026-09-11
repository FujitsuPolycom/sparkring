"""Exercise request timing and readiness gates without contacting a server."""

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture(
    params=sorted(
        path for path in HERE.glob("*_checks.py") if not path.name.startswith("test_")
    )
)
def harness(request):
    spec = importlib.util.spec_from_file_location(
        "prefill_harness_" + request.param.stem, request.param
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "health,running,waiting,expected",
    [
        ("healthy", 0, 0, True),
        ("starting", 0, 0, False),
        ("healthy", 1, 0, False),
        ("healthy", 0, 1, False),
    ],
)
def test_readiness_requires_healthy_idle_service(
    harness, monkeypatch, health, running, waiting, expected
):
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *a, **kw: health)
    monkeypatch.setattr(
        harness,
        "get",
        lambda path: (
            f'vllm:num_requests_running{{model="m"}} {running}\n'
            f'vllm:num_requests_waiting{{model="m"}} {waiting}\n'
        ),
    )
    assert harness.ready() is expected


def test_missing_metrics_do_not_count_as_zero(harness, monkeypatch):
    monkeypatch.setattr(harness.subprocess, "check_output", lambda *a, **kw: "healthy")
    monkeypatch.setattr(harness, "get", lambda path: "")
    assert harness.ready() is False


@pytest.mark.parametrize(
    "tokens,cached,rejected", [(8192, 0, False), (8191, 0, True), (8192, 1, True)]
)
def test_timing_rejects_wrong_prompt_length_or_cached_work(
    harness, monkeypatch, tokens, cached, rejected
):
    clock = iter([10.0, 10.25])
    clock_name = (
        "perf_counter"
        if harness.__name__.endswith("mhc_precise_checks")
        else "monotonic"
    )
    monkeypatch.setattr(harness.time, clock_name, lambda: next(clock))
    monkeypatch.setattr(
        harness, "calibrate", lambda *args: [{"role": "user", "content": "test"}]
    )
    chunks = [
        {"choices": [{"delta": {"content": ""}}]},
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": "length"}]},
        {
            "choices": [],
            "usage": {
                "prompt_tokens": tokens,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": cached},
            },
        },
    ]
    stream = (
        b"".join(b"data: " + json.dumps(chunk).encode() + b"\n" for chunk in chunks)
        + b"data: [DONE]\n"
    )
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(stream)
    )
    if rejected:
        with pytest.raises(ValueError):
            harness.prefill(8192)
    else:
        result = harness.prefill(8192)
        assert result["ttft_seconds"] == 0.25
        assert result["tokens_per_second"] == 32768


@pytest.mark.parametrize(
    "details",
    [
        "omitted",
        None,
        {},
        {"cached_tokens": False},
        {"cached_tokens": "0"},
        {"cached_tokens": -1},
    ],
)
def test_timing_rejects_unproven_cache_accounting(harness, monkeypatch, details):
    clock = iter([10.0, 10.25])
    clock_name = (
        "perf_counter"
        if harness.__name__.endswith("mhc_precise_checks")
        else "monotonic"
    )
    monkeypatch.setattr(harness.time, clock_name, lambda: next(clock))
    monkeypatch.setattr(
        harness, "calibrate", lambda *args: [{"role": "user", "content": "test"}]
    )
    usage = {"prompt_tokens": 8192, "completion_tokens": 1}
    if details != "omitted":
        usage["prompt_tokens_details"] = details
    chunks = [
        {"choices": [{"delta": {"content": "answer"}, "finish_reason": "length"}]},
        {"choices": [], "usage": usage},
    ]
    stream = (
        b"".join(b"data: " + json.dumps(chunk).encode() + b"\n" for chunk in chunks)
        + b"data: [DONE]\n"
    )
    monkeypatch.setattr(
        harness.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(stream)
    )
    with pytest.raises(ValueError, match="cached_tokens"):
        harness.prefill(8192)


@pytest.mark.parametrize('failure', ['truncated', 'missing-finish', 'error-event', 'missing-completion'])
def test_incomplete_stream_cannot_qualify_prefill(harness, monkeypatch, failure):
    ticks = iter([10.0, 10.25])
    clock_name = 'perf_counter' if harness.__name__.endswith('mhc_precise_checks') else 'monotonic'
    monkeypatch.setattr(harness.time, clock_name, lambda: next(ticks))
    monkeypatch.setattr(harness, 'calibrate', lambda *args: [{'role': 'user', 'content': 'test'}])
    choice = {'delta': {'content': 'answer'}, 'finish_reason': 'length'}
    usage = {'prompt_tokens': 8192, 'completion_tokens': 1, 'prompt_tokens_details': {'cached_tokens': 0}}
    if failure == 'missing-finish':
        choice.pop('finish_reason')
    if failure == 'missing-completion':
        usage.pop('completion_tokens')
    stream = b'data: ' + json.dumps({'choices': [choice], 'usage': usage}).encode() + b'\n'
    if failure == 'error-event':
        stream += b'data: {"error":{"message":"failed"}}\n'
    if failure != 'truncated':
        stream += b'data: [DONE]\n'
    monkeypatch.setattr(harness.urllib.request, 'urlopen', lambda *args, **kwargs: io.BytesIO(stream))
    with pytest.raises((ValueError, RuntimeError)):
        harness.prefill(8192)


def test_zero_samples_refuses_before_any_readiness_request(harness, monkeypatch, tmp_path):
    arguments = ['harness', 'prefill', '--output', str(tmp_path / 'receipt.json'), '--samples', '0']
    if harness.__name__.endswith('mhc_precise_checks'):
        arguments += ['--container-prefix', 'sparkring-test']
    monkeypatch.setattr(sys, 'argv', arguments)
    monkeypatch.setattr(harness, 'ready', lambda: pytest.fail('Readiness queried before validating sample count'))
    with pytest.raises((ValueError, SystemExit)):
        harness.main()
