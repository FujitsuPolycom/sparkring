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
