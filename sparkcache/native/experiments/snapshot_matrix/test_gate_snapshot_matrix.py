from __future__ import annotations

import copy

import pytest

from sparkcache.native.experiments.snapshot_matrix.gate_snapshot_matrix import (
    GLM52_PRODUCTION_PAYLOAD_BYTES,
    MatrixError,
    evaluate_matrix,
)


def _cell(mode: str, depth: int) -> dict:
    payload_bytes = 48 * 1024 * 1024
    byte_checks = [8] * 4
    return {
        "arena_mode": mode,
        "ring_depth": depth,
        "slot_bytes": 64 * 1024 * 1024,
        "rank_count": 4,
        "safety": {
            "byte_mismatches": 0,
            "cuda_errors": 0,
            "stale_tickets": 0,
            "leaked_slots": 0,
            "leaked_leases": 0,
            "timeouts": 0,
            "request_errors": 0,
            "worker_restarts": 0,
            "shutdown_clean": True,
        },
        "memory": {
            "peak_delta_bytes_per_rank": [150_000_000] * 4,
            "minimum_free_bytes_per_rank": [3_000_000_000] * 4,
            "kv_capacity_tokens": 458_752,
        },
        "ring": {
            "submissions": 10_000,
            "would_block": 10,
            "gather_p95_ms": 3.0,
            "gather_p99_ms": 4.0,
            "submit_p99_us": 100.0,
            "completion_pause_p95_ms": 200.0,
            "managed_fault_events": 0,
        },
        "standalone": {
            "production_payload_bytes": payload_bytes,
            "production_byte_checks_per_rank": byte_checks,
            "cpu_readback_bytes_per_rank": [
                payload_bytes * count for count in byte_checks
            ],
            "cpu_readback_mismatches": 0,
            "saturation_cycles_per_rank": [100] * 4,
            "max_outstanding_per_rank": [depth] * 4,
            "distinct_slots_observed_per_rank": [depth] * 4,
            "depth_plus_one_would_block_per_rank": [True] * 4,
            "cpu_read_during_gpu_fill_samples_per_rank": [100] * 4,
            "cpu_first_touch_p95_ms": 4.0,
            "cpu_warm_read_p95_ms": 1.0,
            "end_to_end_p95_ms": 8.0,
            "end_to_end_p99_ms": 10.0,
        },
        "serving": {
            "prefill_tps": 792.0,
            "decode_c1_tps": 19.8,
            "decode_c8_tps": 49.5,
            "inter_token_p99_ms": 82.0,
        },
    }


def _matrix() -> dict:
    return {
        "schema": "sparkring-snapshot-matrix/v2",
        "live_candidate": {
            "arena_mode": "mapped",
            "ring_depth": 2,
        },
        "baseline": {
            "prefill_tps": 800.0,
            "decode_c1_tps": 20.0,
            "decode_c8_tps": 50.0,
            "inter_token_p99_ms": 80.0,
            "kv_capacity_tokens": 458_752,
        },
        "cells": [
            _cell(mode, depth)
            for mode in ("mapped", "managed")
            for depth in (2, 3)
        ],
    }


def test_conservative_passing_matrix_promotes_mapped_depth_two() -> None:
    decision = evaluate_matrix(_matrix())
    assert decision["recommended"] == {
        "arena_mode": "mapped",
        "ring_depth": 2,
    }
    assert decision["promotable"] == [
        {"arena_mode": "mapped", "ring_depth": 2}
    ]


def test_depth_three_requires_measured_backpressure_reduction() -> None:
    payload = _matrix()
    mapped2 = payload["cells"][0]
    mapped3 = payload["cells"][1]
    mapped2["ring"]["would_block"] = 200
    mapped3["ring"]["would_block"] = 40
    payload["live_candidate"] = {
        "arena_mode": "mapped",
        "ring_depth": 3,
    }

    decision = evaluate_matrix(payload)

    assert {"arena_mode": "mapped", "ring_depth": 3} in decision["promotable"]
    assert decision["recommended"] == {
        "arena_mode": "mapped",
        "ring_depth": 3,
    }


def test_managed_requires_speed_win_and_no_serving_regression() -> None:
    payload = _matrix()
    managed2 = payload["cells"][2]
    managed2["standalone"]["end_to_end_p95_ms"] = 7.0
    managed2["serving"]["prefill_tps"] = 788.0
    payload["live_candidate"] = {
        "arena_mode": "managed",
        "ring_depth": 2,
    }

    decision = evaluate_matrix(payload)

    assert decision["recommended"] == {
        "arena_mode": "managed",
        "ring_depth": 2,
    }


def test_safety_failure_rejects_standalone_cell() -> None:
    payload = _matrix()
    payload["baseline"] = None
    payload["live_candidate"] = None
    payload["cells"][0]["safety"]["leaked_leases"] = 1

    decision = evaluate_matrix(payload)
    mapped2 = decision["cells"][0]

    assert not mapped2["standalone_passed"]
    assert "leaked_leases must be zero" in mapped2["reasons"]


def test_memory_failure_rejects_live_candidate() -> None:
    payload = _matrix()
    payload["cells"][0]["memory"]["minimum_free_bytes_per_rank"][3] = 100

    decision = evaluate_matrix(payload)
    mapped2 = decision["cells"][0]

    assert not mapped2["passed"]
    assert "a rank has less than 1 GiB free memory" in mapped2["reasons"]


def test_managed_cpu_path_rejects_standalone_cell() -> None:
    payload = _matrix()
    payload["baseline"] = None
    payload["live_candidate"] = None
    managed2 = payload["cells"][2]
    managed2["standalone"]["cpu_readback_bytes_per_rank"][0] = 1

    decision = evaluate_matrix(payload)
    result = decision["cells"][2]

    assert not result["standalone_passed"]
    assert (
        "CPU readback did not consume every checked production payload"
        in result["reasons"]
    )


def test_managed_live_interference_rejects_candidate() -> None:
    payload = _matrix()
    managed2 = payload["cells"][2]
    managed2["standalone"]["end_to_end_p95_ms"] = 7.0
    managed2["serving"]["decode_c8_tps"] = 40.0
    payload["live_candidate"] = {
        "arena_mode": "managed",
        "ring_depth": 2,
    }

    decision = evaluate_matrix(payload)
    result = decision["cells"][2]

    assert not result["passed"]
    assert "decode_c8_tps is below 98% of baseline" in result["reasons"]


def test_serial_probe_cannot_claim_to_have_exercised_depth() -> None:
    payload = _matrix()
    mapped3 = payload["cells"][1]
    mapped3["standalone"]["max_outstanding_per_rank"] = [1] * 4
    mapped3["standalone"]["distinct_slots_observed_per_rank"] = [1] * 4
    mapped3["standalone"]["depth_plus_one_would_block_per_rank"] = [False] * 4

    decision = evaluate_matrix(payload)
    result = decision["cells"][1]

    assert not result["standalone_passed"]
    assert (
        "configured ring depth was not simultaneously outstanding"
        in result["standalone_reasons"]
    )
    assert (
        "configured ring depth did not use distinct slots"
        in result["standalone_reasons"]
    )


def test_payload_below_exact_glm52_boundary_is_not_production_evidence() -> None:
    payload = _matrix()
    payload["baseline"] = None
    payload["live_candidate"] = None
    payload["cells"][0]["standalone"]["production_payload_bytes"] = (
        GLM52_PRODUCTION_PAYLOAD_BYTES - 1
    )

    decision = evaluate_matrix(payload)
    result = decision["cells"][0]

    assert not result["standalone_passed"]
    assert (
        "production payload is below the pinned GLM-5.2 1024-row "
        "payload or above 64 MiB"
        in result["standalone_reasons"]
    )


def test_exact_glm52_payload_boundary_passes_with_full_readback() -> None:
    payload = _matrix()
    payload["baseline"] = None
    payload["live_candidate"] = None
    standalone = payload["cells"][0]["standalone"]
    standalone["production_payload_bytes"] = GLM52_PRODUCTION_PAYLOAD_BYTES
    standalone["cpu_readback_bytes_per_rank"] = [
        GLM52_PRODUCTION_PAYLOAD_BYTES * check_count
        for check_count in standalone["production_byte_checks_per_rank"]
    ]

    decision = evaluate_matrix(payload)

    assert decision["cells"][0]["standalone_passed"]


def test_standalone_matrix_selects_one_reload_candidate() -> None:
    payload = _matrix()
    payload["baseline"] = None
    payload["live_candidate"] = None
    for cell in payload["cells"]:
        cell.pop("memory")
        cell.pop("serving")
    payload["cells"][2]["standalone"]["end_to_end_p95_ms"] = 7.0

    decision = evaluate_matrix(payload)

    assert decision["standalone_recommended"] == {
        "arena_mode": "managed",
        "ring_depth": 2,
    }
    assert decision["recommended"] is None
    assert all(not cell["live_tested"] for cell in decision["cells"])


def test_incomplete_or_duplicate_matrix_is_invalid() -> None:
    missing = _matrix()
    missing["cells"].pop()
    with pytest.raises(MatrixError, match="exactly four"):
        evaluate_matrix(missing)

    duplicate = _matrix()
    duplicate["cells"][-1] = copy.deepcopy(duplicate["cells"][0])
    with pytest.raises(MatrixError, match="duplicate"):
        evaluate_matrix(duplicate)
