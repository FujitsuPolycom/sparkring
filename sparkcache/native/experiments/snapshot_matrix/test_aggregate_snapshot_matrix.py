from __future__ import annotations

import copy
import json

import pytest

from sparkcache.native.experiments.snapshot_matrix.aggregate_snapshot_matrix import (
    GLM52_PAYLOAD_BYTES,
    AggregationError,
    aggregate_jsonl,
    aggregate_records,
)
from sparkcache.native.experiments.snapshot_matrix.gate_snapshot_matrix import (
    evaluate_matrix,
)


def _latency(samples: int, scale: float) -> dict:
    return {
        "samples": samples,
        "p50": scale,
        "p95": scale * 2,
        "p99": scale * 3,
    }


def _stats(
    submissions: int,
    *,
    payload_bytes: int = GLM52_PAYLOAD_BYTES,
    claims: int | None = None,
    releases: int | None = None,
    would_block: int = 0,
    abandoned: int = 0,
) -> dict:
    claimed = submissions if claims is None else claims
    released = submissions if releases is None else releases
    return {
        "submitted_bytes": payload_bytes * submissions,
        "completed_bytes": payload_bytes * submissions,
        "released_bytes": payload_bytes * released,
        "submissions": submissions,
        "claims": claimed,
        "releases": released,
        "would_block": would_block,
        "abandoned": abandoned,
        "stale_tickets": 0,
    }


def _probe(mode: str, depth: int, rank: int) -> dict:
    iterations = 10_000
    checks = 9
    overlap_samples = 107
    consume_passes = 108
    cycles = 100
    free = 8_000_000_000 - rank * 10_000_000
    return {
        "schema": "sparkcache.snapshot_matrix.v1",
        "passed": True,
        "error": None,
        "config": {
            "arena": mode,
            "slots": depth,
            "rank": rank,
            "rows": 1024,
            "iterations": iterations,
            "compare_every": 1000,
            "pipeline_depth": depth,
            "writer_hold_us": 100,
            "saturation_cycles": cycles,
            "profile": "glm52",
            "slot_bytes": 64 * 1024 * 1024,
        },
        "memory": {
            "nominal_arena_bytes": 64 * 1024 * 1024 * depth,
            "before_create": {"free": free, "total": 128_000_000_000},
            "after_configure": {
                "free": free - depth * 64 * 1024 * 1024,
                "total": 128_000_000_000,
            },
            "after_shutdown": {
                "free": free - 1024,
                "total": 128_000_000_000,
            },
        },
        "would_block": {"intentional": cycles, "unexpected": 0},
        "saturation": {
            "passed": True,
            "cycles_completed": cycles,
            "max_outstanding": depth,
            "distinct_slots_observed": depth,
            "stats": _stats(
                depth * cycles,
                payload_bytes=0,
                claims=0,
                releases=0,
                would_block=cycles,
                abandoned=depth * cycles,
            ),
        },
        "latency_us": {
            "submit": _latency(iterations, 10 + rank),
            "gather": _latency(iterations, 1000 + rank * 100),
            "total": _latency(iterations, 1100 + rank * 100),
        },
        "cpu_consume_ms": {
            "first_touch": _latency(consume_passes, 1 + rank),
            "warm_read": _latency(checks, 0.5 + rank),
            "end_to_end": _latency(consume_passes, 4 + rank),
        },
        "cpu_readback": {
            "bytes": GLM52_PAYLOAD_BYTES * consume_passes,
            "consume_passes": consume_passes,
            "warm_read_bytes": GLM52_PAYLOAD_BYTES * checks,
            "warm_read_passes": checks,
            "exact_check_bytes": GLM52_PAYLOAD_BYTES * checks,
            "checksum": rank + 1,
            "mismatches": 0,
            "checks": checks,
            "read_during_gpu_fill_samples": overlap_samples,
        },
        "byte_checked_iterations": [
            0,
            1250,
            2500,
            3750,
            5000,
            6250,
            7500,
            8750,
            9999,
        ],
        "geometry": {
            "record_mask": 0b011,
            "record_offsets": [0, 29_769_728, 0, 0],
            "record_lengths": [29_769_728, 2_973_696, 0, 0],
            "used_bytes": GLM52_PAYLOAD_BYTES,
        },
        "stats": _stats(iterations),
        "mismatch_count": 0,
        "mismatches": [],
    }


def _records() -> list[dict]:
    return [
        _probe(mode, depth, rank)
        for mode in ("mapped", "managed")
        for depth in (2, 3)
        for rank in range(4)
    ]


def test_aggregates_all_cells_in_gate_v2_shape_conservatively() -> None:
    result = aggregate_records(_records())

    assert result["schema"] == "sparkring-snapshot-matrix/v2"
    assert result["baseline"] is None
    assert result["live_candidate"] is None
    assert len(result["cells"]) == 4
    mapped2 = result["cells"][0]
    assert (mapped2["arena_mode"], mapped2["ring_depth"]) == ("mapped", 2)
    assert mapped2["ring"]["submissions"] == 40_000
    assert mapped2["ring"]["gather_p95_ms"] == 2.6
    assert mapped2["ring"]["gather_p99_ms"] == 3.9
    assert mapped2["ring"]["submit_p99_us"] == 39
    assert mapped2["standalone"]["cpu_first_touch_p95_ms"] == 8
    assert mapped2["standalone"]["cpu_warm_read_p95_ms"] == 7
    assert mapped2["standalone"]["end_to_end_p95_ms"] == 14
    assert mapped2["standalone"]["end_to_end_p99_ms"] == 21
    assert mapped2["standalone"]["production_byte_checks_per_rank"] == [9] * 4
    assert mapped2["standalone"]["cpu_readback_bytes_per_rank"] == [
        GLM52_PAYLOAD_BYTES * 9
    ] * 4
    assert mapped2["standalone"]["saturation_cycles_per_rank"] == [100] * 4
    assert mapped2["standalone"][
        "cpu_read_during_gpu_fill_samples_per_rank"
    ] == [107] * 4
    assert mapped2["standalone"][
        "depth_plus_one_would_block_per_rank"
    ] == [True] * 4
    assert mapped2["memory"]["peak_delta_bytes_per_rank"] == [
        2 * 64 * 1024 * 1024
    ] * 4
    assert result["aggregation"]["field_map"][
        "standalone.production_payload_bytes"
    ] == "common(geometry.used_bytes)"


def test_real_9_check_108_consume_107_overlap_pattern_passes() -> None:
    result = aggregate_records(_records())
    standalone = result["cells"][0]["standalone"]

    assert standalone["production_byte_checks_per_rank"] == [9] * 4
    assert standalone["cpu_read_during_gpu_fill_samples_per_rank"] == [107] * 4


def test_jsonl_order_and_blank_lines_do_not_matter() -> None:
    records = list(reversed(_records()))
    lines = ["", *(json.dumps(record) for record in records), "  "]
    result = aggregate_jsonl(lines)
    assert [
        (cell["arena_mode"], cell["ring_depth"]) for cell in result["cells"]
    ] == [
        ("mapped", 2),
        ("mapped", 3),
        ("managed", 2),
        ("managed", 3),
    ]


def test_aggregated_document_is_accepted_by_the_v2_grader() -> None:
    decision = evaluate_matrix(aggregate_records(_records()))

    assert decision["standalone_recommended"] == {
        "arena_mode": "mapped",
        "ring_depth": 2,
    }
    assert decision["recommended"] is None


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda records: records.pop(), "exactly 16 probes"),
        (
            lambda records: records.__setitem__(15, copy.deepcopy(records[0])),
            "duplicate probe",
        ),
        (
            lambda records: records[0]["config"].__setitem__("slot_bytes", 32 << 20),
            "nominal arena bytes disagree",
        ),
        (
            lambda records: records[0].__setitem__("passed", False),
            "failed probe",
        ),
        (
            lambda records: records[0]["geometry"].__setitem__(
                "used_bytes", GLM52_PAYLOAD_BYTES - 1
            ),
            "used_bytes disagrees",
        ),
        (
            lambda records: records[0]["cpu_readback"].__setitem__("checks", 7),
            "byte_checked_iterations disagrees",
        ),
        (
            lambda records: records[0]["saturation"].__setitem__(
                "max_outstanding", 1
            ),
            "did not reach configured depth",
        ),
    ],
)
def test_rejects_incomplete_inconsistent_or_failed_inputs(
    mutation,
    match: str,
) -> None:
    records = _records()
    mutation(records)
    with pytest.raises(AggregationError, match=match):
        aggregate_records(records)


def test_rejects_invalid_json_with_line_number() -> None:
    with pytest.raises(AggregationError, match="line 2"):
        aggregate_jsonl([json.dumps(_records()[0]), "{not-json"])


def test_sums_safety_counters_instead_of_hiding_them() -> None:
    records = _records()
    records[0]["stats"]["stale_tickets"] = 1
    with pytest.raises(AggregationError, match="stats.stale_tickets"):
        aggregate_records(records)


def test_uses_lowest_observed_free_memory_and_largest_observed_delta() -> None:
    records = _records()
    rank0 = records[0]
    rank0["memory"]["before_create"]["free"] = 10_000
    rank0["memory"]["after_configure"]["free"] = 8_000
    rank0["memory"]["after_shutdown"]["free"] = 7_000
    result = aggregate_records(records)
    mapped2 = result["cells"][0]
    assert mapped2["memory"]["peak_delta_bytes_per_rank"][0] == 3_000
    assert mapped2["memory"]["minimum_free_bytes_per_rank"][0] == 7_000


@pytest.mark.parametrize(
    "field",
    (
        "consume_passes",
        "warm_read_bytes",
        "warm_read_passes",
        "exact_check_bytes",
    ),
)
def test_rejects_old_readback_schema_without_population_accounting(
    field: str,
) -> None:
    records = _records()
    records[0]["cpu_readback"].pop(field)

    with pytest.raises(AggregationError, match=f"cpu_readback.{field}"):
        aggregate_records(records)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda probe: probe["cpu_consume_ms"]["first_touch"].__setitem__(
                "samples", 9
            ),
            "first_touch.samples disagrees with consume_passes",
        ),
        (
            lambda probe: probe["cpu_consume_ms"]["end_to_end"].__setitem__(
                "samples", 9
            ),
            "end_to_end.samples disagrees with consume_passes",
        ),
        (
            lambda probe: probe["cpu_consume_ms"]["warm_read"].__setitem__(
                "samples", 108
            ),
            "warm_read.samples disagrees with byte checks",
        ),
        (
            lambda probe: probe["cpu_readback"].__setitem__(
                "bytes", GLM52_PAYLOAD_BYTES * 9
            ),
            "cpu_readback.bytes disagrees with consume_passes",
        ),
        (
            lambda probe: probe["cpu_readback"].__setitem__(
                "warm_read_bytes", GLM52_PAYLOAD_BYTES * 8
            ),
            "warm_read_bytes disagrees with warm_read_passes",
        ),
        (
            lambda probe: probe["cpu_readback"].__setitem__(
                "exact_check_bytes", GLM52_PAYLOAD_BYTES * 8
            ),
            "exact_check_bytes disagrees with byte checks",
        ),
        (
            lambda probe: probe["cpu_readback"].__setitem__(
                "read_during_gpu_fill_samples", 109
            ),
            "overlap samples exceed consume passes",
        ),
        (
            lambda probe: probe["cpu_readback"].__setitem__(
                "consume_passes", 117
            ),
            "consume passes exceed checks plus overlap",
        ),
    ],
)
def test_rejects_partial_or_inconsistent_consumer_populations(
    mutation,
    match: str,
) -> None:
    records = _records()
    mutation(records[0])

    with pytest.raises(AggregationError, match=match):
        aggregate_records(records)
