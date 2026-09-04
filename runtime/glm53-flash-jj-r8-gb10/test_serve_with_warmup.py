from __future__ import annotations

import importlib.util
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

    assert events == ["api", "warmup"]
    assert ready.is_file()


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
