"""Grade a supplied SparkCache mapped/managed, depth-2/depth-3 matrix."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

SCHEMA = "sparkring-snapshot-matrix/v2"
MODES = ("mapped", "managed")
DEPTHS = (2, 3)
MIB = 1024 * 1024
GIB = 1024 * MIB
GLM52_PRODUCTION_PAYLOAD_BYTES = 32_743_424
MAXIMUM_PRODUCTION_SLOT_BYTES = 64 * MIB


class MatrixError(ValueError):
    pass


def _number(container: dict[str, Any], name: str) -> float:
    value = container.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MatrixError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise MatrixError(f"{name} must be finite and non-negative")
    return value


def _integers(container: dict[str, Any], name: str, count: int) -> list[int]:
    value = container.get(name)
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise MatrixError(f"{name} must contain {count} integers")
    if any(item < 0 for item in value):
        raise MatrixError(f"{name} values must be non-negative")
    return value


def _booleans(container: dict[str, Any], name: str, count: int) -> list[bool]:
    value = container.get(name)
    if (
        not isinstance(value, list)
        or len(value) != count
        or any(not isinstance(item, bool) for item in value)
    ):
        raise MatrixError(f"{name} must contain {count} booleans")
    return value


def _ratio(value: float, baseline: float, name: str) -> float:
    if baseline <= 0:
        raise MatrixError(f"baseline {name} must be positive")
    return value / baseline


def _evaluate_cell(
    cell: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    live_required: bool,
) -> dict[str, Any]:
    mode = cell.get("arena_mode")
    depth = cell.get("ring_depth")
    if mode not in MODES or depth not in DEPTHS:
        raise MatrixError(f"invalid cell key: {mode!r}/depth-{depth!r}")
    rank_count = cell.get("rank_count")
    if rank_count != 4:
        raise MatrixError(f"{mode}/depth-{depth}: rank_count must equal 4")
    slot_bytes = int(_number(cell, "slot_bytes"))
    if slot_bytes <= 0:
        raise MatrixError(f"{mode}/depth-{depth}: slot_bytes must be positive")

    safety = cell.get("safety")
    ring = cell.get("ring")
    standalone = cell.get("standalone")
    if not all(isinstance(item, dict) for item in (safety, ring, standalone)):
        raise MatrixError(
            f"{mode}/depth-{depth}: missing safety, ring, or standalone section"
        )

    standalone_reasons: list[str] = []
    for metric in (
        "byte_mismatches",
        "cuda_errors",
        "stale_tickets",
        "leaked_slots",
        "leaked_leases",
        "timeouts",
        "request_errors",
        "worker_restarts",
    ):
        if _number(safety, metric) != 0:
            standalone_reasons.append(f"{metric} must be zero")
    if safety.get("shutdown_clean") is not True:
        standalone_reasons.append("shutdown_clean must be true")

    production_payload_bytes = int(
        _number(standalone, "production_payload_bytes")
    )
    if not (
        GLM52_PRODUCTION_PAYLOAD_BYTES
        <= production_payload_bytes
        <= MAXIMUM_PRODUCTION_SLOT_BYTES
    ):
        standalone_reasons.append(
            "production payload is below the pinned GLM-5.2 1024-row "
            "payload or above 64 MiB"
        )
    if production_payload_bytes > slot_bytes:
        standalone_reasons.append("production payload exceeds slot capacity")
    checks = _integers(
        standalone, "production_byte_checks_per_rank", rank_count
    )
    if min(checks) < 8:
        standalone_reasons.append(
            "fewer than eight production-payload byte checks on a rank"
        )
    readback_bytes = _integers(
        standalone, "cpu_readback_bytes_per_rank", rank_count
    )
    if any(
        observed < production_payload_bytes * check_count
        for observed, check_count in zip(readback_bytes, checks, strict=True)
    ):
        standalone_reasons.append(
            "CPU readback did not consume every checked production payload"
        )
    if _number(standalone, "cpu_readback_mismatches") != 0:
        standalone_reasons.append("CPU readback mismatches must be zero")

    saturation_cycles = _integers(
        standalone, "saturation_cycles_per_rank", rank_count
    )
    if min(saturation_cycles) < 100:
        standalone_reasons.append(
            "fewer than 100 depth-saturation cycles on a rank"
        )
    max_outstanding = _integers(
        standalone, "max_outstanding_per_rank", rank_count
    )
    if any(observed != depth for observed in max_outstanding):
        standalone_reasons.append(
            "configured ring depth was not simultaneously outstanding"
        )
    distinct_slots = _integers(
        standalone, "distinct_slots_observed_per_rank", rank_count
    )
    if any(observed != depth for observed in distinct_slots):
        standalone_reasons.append(
            "configured ring depth did not use distinct slots"
        )
    depth_plus_one_blocked = _booleans(
        standalone, "depth_plus_one_would_block_per_rank", rank_count
    )
    if not all(depth_plus_one_blocked):
        standalone_reasons.append(
            "depth-plus-one submission did not WOULD_BLOCK on every rank"
        )
    overlap_samples = _integers(
        standalone, "cpu_read_during_gpu_fill_samples_per_rank", rank_count
    )
    if min(overlap_samples) < 100:
        standalone_reasons.append(
            "CPU consumption was not overlapped with enough subsequent GPU fills"
        )
    cpu_first_touch_p95_ms = _number(
        standalone, "cpu_first_touch_p95_ms"
    )
    cpu_warm_read_p95_ms = _number(standalone, "cpu_warm_read_p95_ms")
    end_to_end_p95_ms = _number(standalone, "end_to_end_p95_ms")
    if _number(standalone, "end_to_end_p99_ms") > 100:
        standalone_reasons.append(
            "standalone gather-plus-CPU-consume p99 exceeds 100 ms"
        )

    submissions = _number(ring, "submissions")
    would_block = _number(ring, "would_block")
    if submissions < 10_000:
        standalone_reasons.append("fewer than 10,000 measured submissions")
    would_block_rate = would_block / submissions if submissions else math.inf
    if would_block_rate > 0.005:
        standalone_reasons.append("would-block rate exceeds 0.5%")
    gather_p95_ms = _number(ring, "gather_p95_ms")
    if _number(ring, "gather_p99_ms") > 50:
        standalone_reasons.append("gather p99 exceeds 50 ms")
    if _number(ring, "submit_p99_us") > 500:
        standalone_reasons.append("submit p99 exceeds 500 us")
    if _number(ring, "completion_pause_p95_ms") > 500:
        standalone_reasons.append("completion pause p95 exceeds 500 ms")
    # Unified/managed allocations are expected to migrate when the CPU first
    # consumes GPU-written bytes.  The gate therefore prices the measured
    # first-touch and end-to-end path instead of demanding an impossible or
    # tool-dependent zero-fault counter.
    _number(ring, "managed_fault_events")

    live_reasons: list[str] = []
    ratios: dict[str, float] = {}
    latency_ratio: float | None = None
    memory_ceiling = max(384 * MIB, math.ceil(1.25 * slot_bytes * depth))
    if live_required:
        if baseline is None:
            raise MatrixError("live candidate requires a baseline")
        memory = cell.get("memory")
        serving = cell.get("serving")
        if not isinstance(memory, dict) or not isinstance(serving, dict):
            raise MatrixError(
                f"{mode}/depth-{depth}: live matrix requires memory and serving"
            )
        peak_delta = _integers(memory, "peak_delta_bytes_per_rank", rank_count)
        minimum_free = _integers(
            memory, "minimum_free_bytes_per_rank", rank_count
        )
        if max(peak_delta) > memory_ceiling:
            live_reasons.append("peak memory delta exceeds slot-budget ceiling")
        if min(minimum_free) < GIB:
            live_reasons.append("a rank has less than 1 GiB free memory")
        baseline_kv = int(_number(baseline, "kv_capacity_tokens"))
        if int(_number(memory, "kv_capacity_tokens")) != baseline_kv:
            live_reasons.append("configured KV capacity changed")

        ratios = {
            metric: _ratio(
                _number(serving, metric),
                _number(baseline, metric),
                metric,
            )
            for metric in ("prefill_tps", "decode_c1_tps", "decode_c8_tps")
        }
        for metric, ratio in ratios.items():
            if ratio < 0.98:
                live_reasons.append(f"{metric} is below 98% of baseline")
        latency_ratio = _ratio(
            _number(serving, "inter_token_p99_ms"),
            _number(baseline, "inter_token_p99_ms"),
            "inter_token_p99_ms",
        )
        if latency_ratio > 1.05:
            live_reasons.append("inter-token p99 exceeds 105% of baseline")

    standalone_passed = not standalone_reasons
    final_reasons = [*standalone_reasons, *live_reasons]

    return {
        "arena_mode": mode,
        "ring_depth": depth,
        "standalone_passed": standalone_passed,
        "standalone_reasons": standalone_reasons,
        "live_tested": live_required,
        "passed": live_required and not final_reasons,
        "reasons": final_reasons,
        "would_block_rate": would_block_rate,
        "gather_p95_ms": gather_p95_ms,
        "cpu_first_touch_p95_ms": cpu_first_touch_p95_ms,
        "cpu_warm_read_p95_ms": cpu_warm_read_p95_ms,
        "end_to_end_p95_ms": end_to_end_p95_ms,
        "minimum_serving_ratio": (
            None if not ratios else min(ratios.values())
        ),
        "latency_ratio": latency_ratio,
        "memory_ceiling_bytes": memory_ceiling,
    }


def evaluate_matrix(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != SCHEMA:
        raise MatrixError(f"schema must equal {SCHEMA}")
    baseline = payload.get("baseline")
    cells = payload.get("cells")
    if baseline is not None and not isinstance(baseline, dict):
        raise MatrixError("baseline must be an object or null")
    if not isinstance(cells, list):
        raise MatrixError("cells are required")
    live_candidate_raw = payload.get("live_candidate")
    live_candidate: tuple[str, int] | None = None
    if live_candidate_raw is not None:
        if not isinstance(live_candidate_raw, dict):
            raise MatrixError("live_candidate must be an object or null")
        candidate_mode = live_candidate_raw.get("arena_mode")
        candidate_depth = live_candidate_raw.get("ring_depth")
        if candidate_mode not in MODES or candidate_depth not in DEPTHS:
            raise MatrixError("live_candidate has an invalid mode or depth")
        live_candidate = (candidate_mode, candidate_depth)
    if (baseline is None) != (live_candidate is None):
        raise MatrixError(
            "baseline and live_candidate must either both be supplied or both be null"
        )

    evaluated: dict[tuple[str, int], dict[str, Any]] = {}
    for cell in cells:
        if not isinstance(cell, dict):
            raise MatrixError("every cell must be an object")
        raw_mode = cell.get("arena_mode")
        raw_depth = cell.get("ring_depth")
        result = _evaluate_cell(
            cell,
            baseline,
            live_required=live_candidate == (raw_mode, raw_depth),
        )
        key = (result["arena_mode"], result["ring_depth"])
        if key in evaluated:
            raise MatrixError(f"duplicate cell: {key[0]}/depth-{key[1]}")
        evaluated[key] = result
    expected = {(mode, depth) for mode in MODES for depth in DEPTHS}
    if set(evaluated) != expected:
        missing = sorted(expected - set(evaluated))
        raise MatrixError(f"matrix must contain exactly four cells; missing={missing}")

    def select_candidates(
        pass_key: str,
        *,
        require_live_parity: bool,
    ) -> tuple[list[tuple[str, int]], tuple[str, int] | None]:
        depth_candidates: set[tuple[str, int]] = set()
        for mode in MODES:
            depth2 = evaluated[(mode, 2)]
            depth3 = evaluated[(mode, 3)]
            if depth2[pass_key]:
                depth_candidates.add((mode, 2))
            if depth3[pass_key]:
                depth3_useful = (
                    depth2["would_block_rate"] > 0.005
                    and depth3["would_block_rate"]
                    <= depth2["would_block_rate"] * 0.5
                )
                if depth3_useful:
                    depth_candidates.add((mode, 3))

        candidates: list[tuple[str, int]] = []
        for depth in DEPTHS:
            mapped = evaluated[("mapped", depth)]
            managed = evaluated[("managed", depth)]
            if ("mapped", depth) in depth_candidates:
                candidates.append(("mapped", depth))
            if (
                ("managed", depth) in depth_candidates
                and ("mapped", depth) in depth_candidates
            ):
                managed_wins = (
                    managed["end_to_end_p95_ms"]
                    <= mapped["end_to_end_p95_ms"] * 0.9
                )
                if require_live_parity:
                    mapped_ratio = mapped["minimum_serving_ratio"]
                    managed_ratio = managed["minimum_serving_ratio"]
                    managed_wins = (
                        managed_wins
                        and mapped_ratio is not None
                        and managed_ratio is not None
                        and managed_ratio >= mapped_ratio - 0.005
                    )
                if managed_wins:
                    candidates.append(("managed", depth))

        recommended = candidates[0] if candidates else None
        if recommended is not None:
            managed_peer = ("managed", recommended[1])
            if managed_peer in candidates:
                recommended = managed_peer
        return candidates, recommended

    standalone_promotable, standalone_recommended = select_candidates(
        "standalone_passed", require_live_parity=False
    )
    promotable: list[tuple[str, int]] = []
    recommended: tuple[str, int] | None = None
    if live_candidate is not None:
        if live_candidate != standalone_recommended:
            raise MatrixError(
                "live_candidate must equal the standalone-recommended cell"
            )
        if evaluated[live_candidate]["passed"]:
            promotable = [live_candidate]
            recommended = live_candidate
    ordered_results = [
        evaluated[(mode, depth)] for mode in MODES for depth in DEPTHS
    ]
    return {
        "schema": "sparkring-snapshot-matrix-decision/v2",
        "standalone_promotable": [
            {"arena_mode": mode, "ring_depth": depth}
            for mode, depth in standalone_promotable
        ],
        "standalone_recommended": (
            None
            if standalone_recommended is None
            else {
                "arena_mode": standalone_recommended[0],
                "ring_depth": standalone_recommended[1],
            }
        ),
        "promotable": [
            {"arena_mode": mode, "ring_depth": depth}
            for mode, depth in promotable
        ],
        "recommended": (
            None
            if recommended is None
            else {"arena_mode": recommended[0], "ring_depth": recommended[1]}
        ),
        "cells": ordered_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.input.read_text("utf-8"))
        decision = evaluate_matrix(payload)
    except (OSError, json.JSONDecodeError, MatrixError) as error:
        print(f"snapshot-matrix: INVALID: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(decision, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    if decision["recommended"] is not None:
        return 0
    if decision["standalone_recommended"] is not None:
        return 3
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
