from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).resolve().parent


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
        return io.BytesIO(b'{"choices":[{"message":{"reasoning":"ok"}}],"usage":{"completion_tokens":2}}')

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    result = wrapper.warmup_sampling("http://localhost", "model", 16, 10, "secret")
    assert {(body["top_k"], body["top_p"]) for body in observed} == {
        (-1, 1.0), (40, 1.0), (-1, 0.9), (40, 0.9),
    }
    assert any(body["temperature"] == 0.7 and body["top_k"] == -1 and body["top_p"] == 1.0 for body in observed)
    assert any(body.get("seed") == 0 and body["top_k"] == 40 and body["top_p"] == 0.9 for body in observed)
    assert all(body["min_p"] == 0.0 and body["chat_template_kwargs"] == {"enable_thinking": True} for body in observed)
    assert len({body["messages"][0]["content"] for body in observed}) == len(observed) == 6
    assert result["coverage"] == "request-recipe-complete"
    assert result["jit_coverage_verified"] is False
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
        return io.BytesIO(json.dumps({"choices": [{"message": {"reasoning": "ok"}}], "usage": {"completion_tokens": count}}).encode())

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
        return io.BytesIO(b'{"choices":[{"message":{"reasoning":"ok"}}],"usage":{"completion_tokens":1}}')

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
                        lambda *_args, **_kwargs: io.BytesIO(b'{"choices":[]}'))
    with pytest.raises(RuntimeError, match="Sampling warmup response has no completion"):
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
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}],"usage":{"completion_tokens":1}}')

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    wrapper.warmup_sampling("http://localhost", "model", 16, 10, "api-secret")
    warmup.send_warmup_request("http://localhost", "model", "nonce", 16, 10, 8, "api-secret")
    assert len(observed) == 7
    for request in observed:
        assert request.get_header("X-sparkring-startup-token") == "internal-secret"
        assert request.get_header("Authorization") == "Bearer api-secret"
