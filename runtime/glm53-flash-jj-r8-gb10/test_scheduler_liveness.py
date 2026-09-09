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


def _output_metrics(*, running=3, iterations=10, kv=0.2):
    return _metrics(running=running, waiting=0, kv=kv) + (
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


def test_issue231_growing_prefill_cache_does_not_trip_output_stall() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    for elapsed in range(0, 601, 10):
        now[0] = float(elapsed)
        monitor.observe(_output_metrics(running=1, iterations=0, kv=elapsed / 1000))
        assert monitor.http_status() == 200
    assert monitor.snapshot()["output_stalled_seconds"] == 600


def test_issue231_prefill_progress_stopping_still_detects_a_stall() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    for elapsed in range(0, 301, 10):
        now[0] = float(elapsed)
        monitor.observe(_output_metrics(running=1, iterations=0, kv=elapsed / 1000))
    assert monitor.http_status() == 200
    for elapsed in range(310, 601, 10):
        now[0] = float(elapsed)
        monitor.observe(_output_metrics(running=1, iterations=0, kv=0.3))
    assert monitor.snapshot()["reason"] == "engine_output_stall"


def test_kv_reallocation_and_request_churn_do_not_renew_progress() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    for timestamp, running, kv in (
        (0, 1, 0.4), (100, 1, 0.6), (200, 3, 0.3),
        (300, 2, 0.59), (399, 1, 0.6000000001),
    ):
        now[0] = timestamp
        monitor.observe(_output_metrics(running=running, kv=kv))
        assert monitor.http_status() == 200
    now[0] = 400
    monitor.observe(_output_metrics(running=2, kv=0.6))
    snapshot = monitor.snapshot()
    assert snapshot["reason"] == "engine_output_stall"
    assert snapshot["progress_stalled_seconds"] == 300
    assert snapshot["output_stalled_seconds"] == 400
    assert snapshot["kv_allocation_high_water"] == 0.6


def test_optional_prompt_counter_growth_and_loss_have_distinct_meanings() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    for timestamp, prompt in ((0, 100), (100, 200), (200, None), (300, 0), (399, 200)):
        now[0] = timestamp
        metrics = _output_metrics()
        if prompt is not None:
            metrics += f'\nvllm:prompt_tokens_total{{engine="0"}} {prompt}'
        monitor.observe(metrics)
        assert monitor.http_status() == 200
    now[0] = 400
    monitor.observe(_output_metrics() + '\nvllm:prompt_tokens_total{engine="0"} 200')
    assert monitor.snapshot()["reason"] == "engine_output_stall"
    now[0] = 401
    monitor.observe(_output_metrics() + '\nvllm:prompt_tokens_total{engine="0"} 201')
    snapshot = monitor.snapshot()
    assert snapshot["healthy"] is True
    assert snapshot["last_progress_signal"] == "prompt_token_counter"
    assert snapshot["output_stalled_seconds"] == 401
    assert snapshot["progress_stalled_seconds"] == 0


def test_first_optional_counter_sample_does_not_prove_progress() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    monitor.observe(_output_metrics())
    now[0] = 300
    monitor.observe(_output_metrics() + '\nvllm:prompt_tokens_total{engine="0"} 1000')
    assert monitor.snapshot()["reason"] == "engine_output_stall"


def test_output_or_idle_starts_a_fresh_kv_progress_epoch() -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    for timestamp, running, iterations, kv in (
        (0, 1, 10, 0.8), (100, 1, 11, 0.2), (390, 1, 11, 0.3),
        (680, 1, 11, 0.4), (700, 0, 11, 0.4), (1000, 1, 11, 0.1),
        (1290, 1, 11, 0.2), (1580, 1, 11, 0.3),
    ):
        now[0] = timestamp
        monitor.observe(_output_metrics(running=running, iterations=iterations, kv=kv))
        assert monitor.http_status() == 200
    assert 'sparkring:engine_progress_stalled_seconds 0' in monitor.prometheus()
    assert 'sparkring:engine_output_stalled_seconds 580' in monitor.prometheus()


@pytest.mark.parametrize("value", ["-1", "nan", "inf"])
def test_invalid_optional_prompt_counter_rejects_the_sample(value: str) -> None:
    module = _load_module()
    now = [0.0]
    monitor = module.SchedulerLiveness(
        blocked_timeout_seconds=60, idle_kv_warn_seconds=330,
        stale_sample_seconds=15, clock=lambda: now[0],
    )
    monitor.observe(_output_metrics())
    now[0] = 16
    with pytest.raises(ValueError):
        monitor.observe(_output_metrics() + f'\nvllm:prompt_tokens_total{{engine="0"}} {value}')
    assert monitor.snapshot()["reason"] == "metrics_unavailable"
