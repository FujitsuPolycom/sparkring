from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


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


def _output_metrics(*, running=3, iterations=10):
    return _metrics(running=running, waiting=0, kv=0.2) + (
        f'\nvllm:iteration_tokens_total_count{{engine="0"}} {iterations}'
    )


def test_fresh_scrapes_do_not_hide_running_output_stall() -> None:
    module = _load_module()
    now = [100.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    monitor.observe(_output_metrics())
    for elapsed in range(10, 301, 10):
        now[0] = 100.0 + elapsed
        monitor.observe(_output_metrics())
    assert monitor.http_status() == 503
    assert monitor.snapshot()["reason"] == "engine_output_stall"
    assert monitor.snapshot()["output_stalled_seconds"] == 300


def test_output_progress_and_idle_restart_the_stall_window() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    for timestamp, running, iterations in (
        (0, 3, 10), (290, 3, 11), (580, 3, 12),
        (870, 0, 12), (2000, 3, 12), (2290, 3, 12),
        (2300, 3, 1), (2590, 3, 1),
    ):
        now[0] = timestamp
        monitor.observe(_output_metrics(running=running, iterations=iterations))
        assert monitor.http_status() == 200


def test_running_stall_timeout_is_independent_of_capacity_timeout() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, output_timeout_seconds=900,
        idle_kv_warn_seconds=330, stale_sample_seconds=15,
        clock=lambda: now[0],
    )
    monitor.observe(_output_metrics())
    now[0] = 899
    monitor.observe(_output_metrics())
    assert monitor.http_status() == 200
    now[0] = 900
    monitor.observe(_output_metrics())
    assert monitor.http_status() == 503
    assert "sparkring:engine_output_stalled_seconds 900" in monitor.prometheus()


def test_missing_progress_metric_does_not_refresh_sample() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    monitor.observe(_output_metrics())
    now[0] = 16
    with pytest.raises(ValueError, match="iteration_tokens_total_count") as error:
        monitor.observe(_metrics(running=3, waiting=0, kv=0.2))
    monitor.observe_error(error.value)
    assert monitor.snapshot()["reason"] == "metrics_unavailable"
    assert monitor.http_status() == 503


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_output_timeout_is_rejected(timeout) -> None:
    with pytest.raises(ValueError, match="output timeout"):
        _load_module().SchedulerLiveness(
            blocked_timeout_seconds=60, output_timeout_seconds=timeout,
            idle_kv_warn_seconds=330, stale_sample_seconds=15,
        )


def test_progress_recovers_unhealthy_monitor() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    monitor.observe(_output_metrics())
    now[0] = 300
    monitor.observe(_output_metrics())
    assert monitor.http_status() == 503
    now[0] += 10
    monitor.observe(_output_metrics(iterations=11))
    assert monitor.http_status() == 200
    assert monitor.snapshot()["output_stalled_seconds"] == 0
