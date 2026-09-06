from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "scheduler_liveness", HERE / "scheduler_liveness.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metrics(*, running: float, waiting: float, kv: float, uncertain: float = 0) -> str:
    return "\n".join(
        (
            f'vllm:num_requests_running{{engine="0"}} {running}',
            f'vllm:num_requests_waiting{{engine="0"}} {waiting}',
            f'vllm:kv_cache_usage_perc{{engine="0"}} {kv}',
            "vllm:sparkcache_capture_ownership_uncertain_ranks"
            f'{{engine="0"}} {uncertain}',
        )
    )


def test_waiting_without_running_becomes_unhealthy_after_timeout() -> None:
    module = _load_module()
    now = [100.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60.0,
        idle_kv_warn_seconds=330.0,
        stale_sample_seconds=15.0,
        clock=lambda: now[0],
    )

    monitor.observe(_metrics(running=0, waiting=2, kv=0.98))
    assert monitor.snapshot()["healthy"] is True
    now[0] += 61.0
    monitor.observe(_metrics(running=0, waiting=2, kv=0.98))

    snapshot = monitor.snapshot()
    assert snapshot["healthy"] is False
    assert snapshot["reason"] == "scheduler_capacity_stall"
    assert snapshot["blocked_seconds"] == 61.0


def test_idle_kv_retention_warns_without_declaring_scheduler_dead() -> None:
    module = _load_module()
    now = [100.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60.0,
        idle_kv_warn_seconds=330.0,
        stale_sample_seconds=15.0,
        clock=lambda: now[0],
    )

    monitor.observe(_metrics(running=0, waiting=0, kv=0.20))
    now[0] += 331.0
    monitor.observe(_metrics(running=0, waiting=0, kv=0.20))

    snapshot = monitor.snapshot()
    assert snapshot["healthy"] is True
    assert snapshot["warnings"] == ["idle_kv_not_falling"]
    assert snapshot["idle_kv_nonfall_seconds"] == 331.0


def test_falling_idle_kv_restarts_the_warning_window() -> None:
    module = _load_module()
    now = [100.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60.0,
        idle_kv_warn_seconds=330.0,
        stale_sample_seconds=15.0,
        clock=lambda: now[0],
    )

    monitor.observe(_metrics(running=0, waiting=0, kv=0.20))
    now[0] += 300.0
    monitor.observe(_metrics(running=0, waiting=0, kv=0.10))
    now[0] += 31.0
    monitor.observe(_metrics(running=0, waiting=0, kv=0.10))

    snapshot = monitor.snapshot()
    assert snapshot["healthy"] is True
    assert snapshot["warnings"] == []
    assert snapshot["idle_kv_nonfall_seconds"] == 31.0


def test_uncertain_capture_ownership_is_unhealthy_immediately() -> None:
    module = _load_module()
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60.0,
        idle_kv_warn_seconds=330.0,
        stale_sample_seconds=15.0,
        clock=lambda: 100.0,
    )

    monitor.observe(_metrics(running=0, waiting=0, kv=0.10, uncertain=1))

    assert monitor.snapshot()["reason"] == "capture_ownership_uncertain"
    assert monitor.http_status() == 503


def test_liveness_metrics_are_machine_readable() -> None:
    module = _load_module()
    now = [100.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60.0,
        idle_kv_warn_seconds=330.0,
        stale_sample_seconds=15.0,
        clock=lambda: now[0],
    )
    monitor.observe(_metrics(running=0, waiting=3, kv=0.98))
    now[0] += 61.0
    monitor.observe(_metrics(running=0, waiting=3, kv=0.98))

    rendered = monitor.prometheus()

    assert "sparkring:scheduler_liveness 0" in rendered
    assert "sparkring:scheduler_blocked_seconds 61.0" in rendered
    assert "sparkring:idle_kv_nonfall_seconds 61.0" in rendered
    json.dumps(monitor.snapshot())


def test_initial_unavailable_snapshot_is_strict_json() -> None:
    module = _load_module()
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60.0,
        idle_kv_warn_seconds=330.0,
        stale_sample_seconds=15.0,
        clock=lambda: 100.0,
    )

    snapshot = monitor.snapshot()

    assert snapshot["healthy"] is False
    assert snapshot["sample_age_seconds"] is None
    json.dumps(snapshot, allow_nan=False)
