"""Exercise request timing and readiness gates without contacting a server."""

import io
import json
import sys

import pytest

from performance.harnesses.vllm.prefill_checks import harness as shared


@pytest.fixture(params=[("monotonic", False), ("perf_counter", True)])
def harness(request):
    from performance.harnesses.vllm.prefill_checks.harness import Harness

    clock_name, require_prefix = request.param
    return Harness(clock_name=clock_name, require_prefix=require_prefix)


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
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *a, **kw: health)
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
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *a, **kw: "healthy")
    monkeypatch.setattr(harness, "get", lambda path: "")
    assert harness.ready() is False


@pytest.mark.parametrize(
    "tokens,cached,rejected", [(8192, 0, False), (8191, 0, True), (8192, 1, True)]
)
def test_timing_rejects_wrong_prompt_length_or_cached_work(
    harness, monkeypatch, tokens, cached, rejected
):
    clock = iter([10.0, 10.25])
    monkeypatch.setattr(harness, "clock", lambda: next(clock))
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
        shared.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(stream)
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
    monkeypatch.setattr(harness, "clock", lambda: next(clock))
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
        shared.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(stream)
    )
    with pytest.raises(ValueError, match="cached_tokens"):
        harness.prefill(8192)


@pytest.mark.parametrize(
    "failure", ["truncated", "missing-finish", "error-event", "missing-completion"]
)
def test_incomplete_stream_cannot_qualify_prefill(harness, monkeypatch, failure):
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(harness, "clock", lambda: next(ticks))
    monkeypatch.setattr(
        harness, "calibrate", lambda *args: [{"role": "user", "content": "test"}]
    )
    choice = {"delta": {"content": "answer"}, "finish_reason": "length"}
    usage = {
        "prompt_tokens": 8192,
        "completion_tokens": 1,
        "prompt_tokens_details": {"cached_tokens": 0},
    }
    if failure == "missing-finish":
        choice.pop("finish_reason")
    if failure == "missing-completion":
        usage.pop("completion_tokens")
    stream = (
        b"data: " + json.dumps({"choices": [choice], "usage": usage}).encode() + b"\n"
    )
    if failure == "error-event":
        stream += b'data: {"error":{"message":"failed"}}\n'
    if failure != "truncated":
        stream += b"data: [DONE]\n"
    monkeypatch.setattr(
        shared.urllib.request, "urlopen", lambda *args, **kwargs: io.BytesIO(stream)
    )
    with pytest.raises((ValueError, RuntimeError)):
        harness.prefill(8192)


def test_zero_samples_refuses_before_any_readiness_request(
    harness, monkeypatch, tmp_path
):
    arguments = [
        "harness",
        "prefill",
        "--output",
        str(tmp_path / "receipt.json"),
        "--samples",
        "0",
    ]
    if harness.require_prefix:
        arguments += ["--container-prefix", "sparkring-test"]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(
        harness,
        "ready",
        lambda: pytest.fail("Readiness queried before validating sample count"),
    )
    with pytest.raises((ValueError, SystemExit)):
        harness.main()


def test_healthy_container_before_api_listener_is_retryable(harness, monkeypatch):
    monkeypatch.setattr(shared.subprocess, "check_output", lambda *a, **kw: "healthy")

    def unavailable(path):
        raise OSError("API socket not listening")

    monkeypatch.setattr(harness, "get", unavailable)
    assert harness.ready() is False


def test_invalid_container_never_reaches_ssh(harness, monkeypatch):
    harness.container = "bad;command"
    monkeypatch.setattr(
        shared.subprocess, "check_output", lambda *a, **kw: pytest.fail("SSH invoked")
    )
    with pytest.raises(ValueError, match="Container name"):
        harness.ready()


def test_invalid_prefix_does_not_mutate_configuration(monkeypatch, tmp_path):
    instance = shared.Harness(require_prefix=True, clock_name="perf_counter")
    before = instance.container
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "harness",
            "prefill",
            "--output",
            str(tmp_path / "receipt.json"),
            "--container-prefix",
            "invalid;prefix",
        ],
    )
    with pytest.raises(SystemExit) as error:
        instance.main()
    assert error.value.code == 2
    assert instance.container == before
    assert not (tmp_path / "receipt.json").exists()


def test_both_entrypoints_preserve_configuration_without_import_side_effects(
    monkeypatch,
):
    import importlib

    monkeypatch.setenv("BENCH_RANK0_CONTAINER", "invalid;name")
    seen = []

    class FakeHarness:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def main(self):
            seen.append("main")

    for module_name in ("continuation_serve_checks", "mhc_precise_checks"):
        module = importlib.import_module(
            "performance.harnesses.vllm.prefill_checks." + module_name
        )
        monkeypatch.setattr(module, "Harness", FakeHarness)
        module.main()
    assert seen == [
        {},
        "main",
        {"require_prefix": True, "clock_name": "perf_counter"},
        "main",
    ]


def test_main_retains_warm_and_measured_output_rows(harness, monkeypatch, tmp_path):
    output = tmp_path / "receipt.json"
    args = [
        "harness",
        "prefill",
        "--output",
        str(output),
        "--sizes",
        "8,16",
        "--samples",
        "2",
    ]
    if harness.require_prefix:
        args += ["--container-prefix", "sparkring-test"]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(harness, "ready", lambda: True)
    monkeypatch.setattr(
        harness, "prefill", lambda tokens: {"tokens": tokens, "ttft_seconds": 0.25}
    )
    harness.main()
    receipt = json.loads(output.read_text())
    assert receipt["phase"] == "prefill"
    assert [
        (row["phase"], row["sample"], row["tokens"]) for row in receipt["rows"]
    ] == [
        ("warm", 0, 8),
        ("warm", 0, 16),
        ("measured", 0, 8),
        ("measured", 0, 16),
        ("measured", 1, 8),
        ("measured", 1, 16),
    ]
