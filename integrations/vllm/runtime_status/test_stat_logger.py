"""CPU contract tests against the extracted image's stat-logger interface."""
import importlib.util
import json
from pathlib import Path
import sys
import threading
import time
from types import ModuleType, SimpleNamespace as NS

import pytest


@pytest.fixture
def logger_module(monkeypatch):
    # Keep this test's vLLM stub scoped: other plugin tests inspect imports.
    base = ModuleType("vllm.v1.metrics.loggers")
    base.StatLoggerBase = type("StatLoggerBase", (), {})
    monkeypatch.setitem(sys.modules, "vllm.v1.metrics.loggers", base)
    path = Path(__file__).parent / "sparkring_runtime_status/stat_logger.py"
    spec = importlib.util.spec_from_file_location("tested_stat_logger", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stats():
    return (NS(num_running_reqs=1, num_waiting_reqs=2, kv_cache_usage=.25,
               iteration_details=NS(iteration_index=1, is_dummy=False, elapsed_ms=25),
               spec_decoding_stats=NS(num_accepted_tokens=2, num_drafts=1, num_draft_tokens=3)),
            NS(prompt_token_stats=NS(computed=100), num_generation_tokens=3,
               time_to_first_tokens_iter=[.2]))


def make(module, monkeypatch, tmp_path):
    monkeypatch.setenv("SPARKRING_STATS_PATH", str(tmp_path / "snapshot.json"))
    monkeypatch.setenv("SPARKRING_STATS_DEPLOYMENT", "test")
    monkeypatch.setattr(module.threading.Thread, "start", lambda self: None)
    return module.SparkRingStatLogger(NS(parallel_config=NS(data_parallel_size=1)))


def test_totals_throttle_idle_and_restart(logger_module, monkeypatch, tmp_path):
    log = make(logger_module, monkeypatch, tmp_path)
    clock = [10.0]
    monkeypatch.setattr(logger_module.time, "monotonic", lambda: clock[0])
    scheduler, iteration = stats()
    log.record(scheduler, iteration)
    first = log.pending
    assert first["prompt"] == 100 and first["gen"] == 3 and first["steps"] == 1
    assert first["ttft_sum"] == .2 and first["step_seconds"] == .025
    for _ in range(100):
        log.record(None, iteration)
    assert log.pending is first
    clock[0] += .5
    log.record(None, None)
    assert log.pending["prompt"] == 10100
    clock[0] += 10
    log.log()  # vLLM's existing logger tick emits idle counter samples.
    assert log.pending["monotonic"] == 20.5 and log.pending["gen"] == 303
    log._write(log.pending)
    assert json.loads(Path(log.path).read_text())["run_id"] == first["run_id"]
    assert Path(log.path).stat().st_size < 1024
    restarted = make(logger_module, monkeypatch, tmp_path)
    assert restarted.identity["run_id"] != log.identity["run_id"]
    assert restarted.values["prompt"] == 0


@pytest.mark.parametrize("error", [OSError(28, "full disk"), PermissionError("read only")])
def test_storage_failure_dropped(logger_module, monkeypatch, tmp_path, error):
    log = make(logger_module, monkeypatch, tmp_path)
    def fail(*args, **kwargs):
        raise error
    monkeypatch.setattr(Path, "write_text", fail)
    log.record(*stats())
    log._write(log.pending)
    assert log.enabled


def test_record_budget_with_blocked_storage(logger_module, monkeypatch, tmp_path):
    start_thread = threading.Thread.start
    log = make(logger_module, monkeypatch, tmp_path)
    blocked, release = threading.Event(), threading.Event()
    def slow_write(snapshot):
        blocked.set()
        release.wait(5)
    monkeypatch.setattr(log, "_write", slow_write)
    # Exercise a real blocked writer without leaving an immortal test thread.
    worker = threading.Thread(target=lambda: log._write({}))
    start_thread(worker)
    assert blocked.wait(1)
    log.enabled = True
    log.next_write = float("inf")
    scheduler, iteration = stats()
    count = 50000
    try:
        started = time.perf_counter()
        for _ in range(count):
            scheduler.iteration_details.iteration_index += 1
            log.record(scheduler, iteration)
        cost = (time.perf_counter() - started) / count
        assert cost < 25e-6, f"record cost {cost * 1e6:.2f} us/call exceeds 25 us budget"
        assert log.values["gen"] == count * 3
    finally:
        release.set()
        worker.join(2)


def test_missing_path_or_dp_disables_writer(logger_module, monkeypatch):
    monkeypatch.delenv("SPARKRING_STATS_PATH", raising=False)
    log = logger_module.SparkRingStatLogger(NS(parallel_config=NS(data_parallel_size=1)))
    assert not log.enabled
    log.record(object(), object())
    monkeypatch.setenv("SPARKRING_STATS_PATH", "/unused")
    assert not logger_module.SparkRingStatLogger(NS(parallel_config=NS(data_parallel_size=2))).enabled


def test_steps_are_real_distinct_iterations_not_generation_counts(logger_module, monkeypatch, tmp_path):
    log = make(logger_module, monkeypatch, tmp_path)
    scheduler, iteration = stats()
    log.record(scheduler, iteration)
    assert log.values["steps"] == 1 and log.values["step_accepted"] == 2
    log.record(scheduler, None)  # duplicate scheduler report
    assert log.values["steps"] == 1 and log.values["step_accepted"] == 2
    assert log.values["accepted"] == 2 and log.values["draft_tokens"] == 3
    scheduler.iteration_details.iteration_index = 2
    scheduler.iteration_details.is_dummy = True
    log.record(scheduler, None)
    assert log.values["steps"] == 1
    scheduler.iteration_details = None
    iteration.num_generation_tokens = 10000
    log.record(scheduler, iteration)
    assert log.values["steps"] == 1 and log.values["step_accepted"] == 2
    # The logger is per API engine, never per TP/DCP rank.
    before = dict(log.values)
    log.record(scheduler, iteration, engine_idx=1)
    assert log.values == before


def test_absent_optional_iteration_fields_keep_token_logger_enabled(logger_module, monkeypatch, tmp_path):
    log = make(logger_module, monkeypatch, tmp_path)
    scheduler, iteration = stats()
    del scheduler.iteration_details
    del scheduler.spec_decoding_stats
    log.record(scheduler, iteration)
    assert log.enabled and log.values["gen"] == 3
    assert not log.pending["step_timing"]
