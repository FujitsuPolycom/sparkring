from __future__ import annotations

import importlib.util
import io
import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


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
        assert timeout == 10
        return io.BytesIO(b'{"choices":[{"message":{"reasoning":"ok"}}]}')

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

    assert probes == [("http://127.0.0.1:8015", 10, "secret")]
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
        return io.BytesIO(b'{"choices":[{"message":{"content":"ok"}}]}')

    monkeypatch.setattr(wrapper.urllib.request, "urlopen", urlopen)
    wrapper.warmup_sampling("http://localhost", "model", 16, 10, "api-secret")
    warmup.send_warmup_request("http://localhost", "model", "nonce", 16, 10, 8, "api-secret")
    assert len(observed) == 2
    for request in observed:
        assert request.get_header("X-sparkring-startup-token") == "internal-secret"
        assert request.get_header("Authorization") == "Bearer api-secret"
