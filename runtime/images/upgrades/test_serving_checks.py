"""Qualification fixtures and measurements fail closed without a live model."""

import json

import pytest

from .contracts import Refused
from .serving_checks import (
    benchmark_measurements,
    cached_tokens,
    needle_fixture,
    verify_needle,
)


def test_needle_fixture_is_reproducible_but_not_shared_across_seeds():
    first = needle_fixture("trial-A")
    assert first == needle_fixture("trial-A")
    assert first["expected"] != needle_fixture("trial-B")["expected"]
    assert first["messages"][0]["content"].count(first["expected"]) == 1


def test_restart_credit_and_answer_are_both_required():
    fixture = needle_fixture("trial-A")
    value = dict(
        content=fixture["expected"],
        usage={"prompt_tokens": 8192, "prompt_tokens_details": {"cached_tokens": 4096}},
    )
    assert verify_needle(value, fixture, minimum_cached=4096) == value
    value["usage"]["prompt_tokens_details"]["cached_tokens"] = 0
    with pytest.raises(Refused, match="persisted prefix"):
        verify_needle(value, fixture, minimum_cached=4096)


def test_absent_cache_metrics_are_not_zero_or_a_pass():
    with pytest.raises(Refused, match="exact cached"):
        cached_tokens(dict(usage={}))


def write_bench(path, **overrides):
    row = dict(
        concurrency=8,
        context_tokens=32768,
        effective_concurrency=7.9,
        aggregate_tps=190,
        server_steps_per_s=81,
        capacity_limited=True,
        queue_fraction=0.05,
        num_errors=0,
    )
    row.update(overrides)
    path.write_text(json.dumps(dict(results=[row])))


def test_measured_full_cells_keep_transient_queue_warning(tmp_path):
    files = [tmp_path / f"sample-{i}.json" for i in range(3)]
    for path in files:
        write_bench(path)
    result = benchmark_measurements(files)
    assert result["measurements"]["decode-32768-c8"] == [190, 190, 190]
    assert len(result["warnings"]) == 3


@pytest.mark.parametrize(
    "override",
    [
        {"num_errors": 1},
        {"loop_detected": True},
        {"effective_concurrency": 6},
        {"warmup_timed_out": True},
        {"aggregate_tps": float("nan")},
        {"failure_reason": "load failed"},
    ],
)
def test_invalid_benchmark_rows_cannot_establish_parity(tmp_path, override):
    files = [tmp_path / f"sample-{i}.json" for i in range(3)]
    for path in files:
        write_bench(path, **override)
    with pytest.raises(Refused):
        benchmark_measurements(files)
