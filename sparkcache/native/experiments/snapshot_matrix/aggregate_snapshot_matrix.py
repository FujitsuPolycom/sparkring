"""Aggregate strict per-rank snapshot-probe JSONL into the v2 matrix schema."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


RAW_SCHEMA = "sparkcache.snapshot_matrix.v1"
OUTPUT_SCHEMA = "sparkring-snapshot-matrix/v2"
AGGREGATION_SCHEMA = "sparkring-snapshot-matrix-aggregation/v1"
MODES = ("mapped", "managed")
DEPTHS = (2, 3)
RANKS = (0, 1, 2, 3)
MIB = 1024 * 1024

GLM52_RECORD_MASK = 0b011
GLM52_RECORD_OFFSETS = (0, 29_769_728, 0, 0)
GLM52_RECORD_LENGTHS = (29_769_728, 2_973_696, 0, 0)
GLM52_PAYLOAD_BYTES = 32_743_424

# This table is deliberately machine-readable in the output. It documents
# exactly how producer fields become gate_snapshot_matrix.py v2 fields.
RAW_FIELD_MAP = {
    "safety.byte_mismatches": "sum(mismatch_count)",
    "safety.cuda_errors": (
        "0 after rejecting any passed!=true or error!=null producer record"
    ),
    "safety.stale_tickets": (
        "sum(stats.stale_tickets + saturation.stats.stale_tickets)"
    ),
    "safety.leaked_slots": (
        "0 after exact main/saturation terminal-stat validation and clean shutdown"
    ),
    "safety.leaked_leases": (
        "0 after exact claims/releases and clean-shutdown validation"
    ),
    "safety.timeouts": (
        "0 after rejecting any passed!=true or error!=null producer record"
    ),
    "safety.request_errors": "0 for the model-down standalone producer",
    "safety.worker_restarts": "0 for the model-down standalone producer",
    "safety.shutdown_clean": "all(passed and error is null)",
    "memory.peak_delta_bytes_per_rank": (
        "before_create.free - min(after_configure.free, after_shutdown.free),"
        " floored at zero"
    ),
    "memory.minimum_free_bytes_per_rank": (
        "min(before_create.free, after_configure.free, after_shutdown.free)"
    ),
    "ring.submissions": "sum(stats.submissions)",
    "ring.would_block": "sum(stats.would_block)",
    "ring.gather_p95_ms": "max_rank(latency_us.gather.p95) / 1000",
    "ring.gather_p99_ms": "max_rank(latency_us.gather.p99) / 1000",
    "ring.submit_p99_us": "max_rank(latency_us.submit.p99)",
    "ring.completion_pause_p95_ms": (
        "max_rank(cpu_consume_ms.end_to_end.p95)"
    ),
    "ring.managed_fault_events": (
        "0; producer has no portable counter, explicitly marked uninstrumented"
    ),
    "standalone.production_payload_bytes": "common(geometry.used_bytes)",
    "standalone.production_byte_checks_per_rank": "cpu_readback.checks",
    "standalone.cpu_readback_bytes_per_rank": "cpu_readback.exact_check_bytes",
    "standalone.cpu_readback_mismatches": "sum(cpu_readback.mismatches)",
    "standalone.saturation_cycles_per_rank": "saturation.cycles_completed",
    "standalone.max_outstanding_per_rank": "saturation.max_outstanding",
    "standalone.distinct_slots_observed_per_rank": (
        "saturation.distinct_slots_observed"
    ),
    "standalone.depth_plus_one_would_block_per_rank": (
        "saturation.passed and would_block.intentional == "
        "saturation.cycles_completed"
    ),
    "standalone.cpu_read_during_gpu_fill_samples_per_rank": (
        "cpu_readback.read_during_gpu_fill_samples"
    ),
    "standalone.cpu_first_touch_p95_ms": (
        "max_rank(cpu_consume_ms.first_touch.p95)"
    ),
    "standalone.cpu_warm_read_p95_ms": (
        "max_rank(cpu_consume_ms.warm_read.p95)"
    ),
    "standalone.end_to_end_p95_ms": (
        "max_rank(cpu_consume_ms.end_to_end.p95)"
    ),
    "standalone.end_to_end_p99_ms": (
        "max_rank(cpu_consume_ms.end_to_end.p99)"
    ),
    "validation.cpu_first_touch_population": (
        "cpu_consume_ms.first_touch.samples == cpu_readback.consume_passes"
    ),
    "validation.cpu_warm_read_population": (
        "cpu_consume_ms.warm_read.samples == cpu_readback.warm_read_passes "
        "== cpu_readback.checks"
    ),
    "validation.end_to_end_population": (
        "cpu_consume_ms.end_to_end.samples == cpu_readback.consume_passes"
    ),
    "validation.consumer_union": (
        "overlap_samples <= consume_passes <= checks + overlap_samples"
    ),
    "validation.consumer_bytes": (
        "cpu_readback.bytes == payload * consume_passes; "
        "warm_read_bytes == payload * warm_read_passes; "
        "exact_check_bytes == payload * checks"
    ),
}


class AggregationError(ValueError):
    """The raw matrix is incomplete, inconsistent, or unsafe."""


def _object(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise AggregationError(f"{path} must be an object")
    return value


def _path(record: Mapping[str, Any], dotted: str) -> Any:
    value: Any = record
    for component in dotted.split("."):
        if not isinstance(value, Mapping) or component not in value:
            raise AggregationError(f"missing producer field {dotted}")
        value = value[component]
    return value


def _integer(record: Mapping[str, Any], dotted: str) -> int:
    value = _path(record, dotted)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AggregationError(
            f"producer field {dotted} must be a non-negative integer"
        )
    return value


def _number(record: Mapping[str, Any], dotted: str) -> float:
    value = _path(record, dotted)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AggregationError(f"producer field {dotted} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise AggregationError(
            f"producer field {dotted} must be finite and non-negative"
        )
    return result


def _boolean(record: Mapping[str, Any], dotted: str) -> bool:
    value = _path(record, dotted)
    if not isinstance(value, bool):
        raise AggregationError(f"producer field {dotted} must be boolean")
    return value


def _integer_list(
    record: Mapping[str, Any],
    dotted: str,
    *,
    length: int,
) -> tuple[int, ...]:
    value = _path(record, dotted)
    if (
        not isinstance(value, list)
        or len(value) != length
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in value
        )
    ):
        raise AggregationError(
            f"producer field {dotted} must contain {length}"
            " non-negative integers"
        )
    return tuple(value)


def _load_jsonl(lines: Iterable[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise AggregationError(
                f"line {line_number} is not valid JSON: {error.msg}"
            ) from error
        records.append(_object(value, f"line {line_number}"))
    if not records:
        raise AggregationError("input JSONL contains no probe records")
    return records


def _config(record: Mapping[str, Any]) -> dict[str, Any]:
    config = _object(_path(record, "config"), "config")
    expected_types = {
        "arena": str,
        "slots": int,
        "rank": int,
        "rows": int,
        "iterations": int,
        "compare_every": int,
        "pipeline_depth": int,
        "writer_hold_us": int,
        "saturation_cycles": int,
        "profile": str,
        "slot_bytes": int,
    }
    for name, expected_type in expected_types.items():
        value = config.get(name)
        if expected_type is int:
            if isinstance(value, bool) or not isinstance(value, int):
                raise AggregationError(f"config.{name} must be an integer")
        elif not isinstance(value, expected_type):
            raise AggregationError(
                f"config.{name} must be {expected_type.__name__}"
            )
    return config


def _validate_probe(record: Mapping[str, Any]) -> tuple[str, int, int]:
    if record.get("schema") != RAW_SCHEMA:
        raise AggregationError(f"producer schema must equal {RAW_SCHEMA}")
    config = _config(record)
    mode = config["arena"]
    depth = config["slots"]
    rank = config["rank"]
    if mode not in MODES or depth not in DEPTHS or rank not in RANKS:
        raise AggregationError(
            f"invalid producer cell/rank {mode!r}/depth-{depth!r}/rank-{rank!r}"
        )
    if record.get("passed") is not True or record.get("error") is not None:
        raise AggregationError(
            f"failed probe at {mode}/depth-{depth}/rank-{rank}: "
            f"passed={record.get('passed')!r} error={record.get('error')!r}"
        )
    if config["profile"] != "glm52" or config["rows"] != 1024:
        raise AggregationError(
            f"{mode}/depth-{depth}/rank-{rank} is not a glm52/1024-row run"
        )
    if config["pipeline_depth"] != depth:
        raise AggregationError(
            f"{mode}/depth-{depth}/rank-{rank} did not exercise full depth"
        )
    if config["slot_bytes"] not in (32 * MIB, 64 * MIB):
        raise AggregationError("production slot must be exactly 32 or 64 MiB")
    if config["iterations"] <= 0 or config["compare_every"] <= 0:
        raise AggregationError("iterations and compare_every must be positive")
    if config["saturation_cycles"] < 100:
        raise AggregationError("producer configured fewer than 100 saturation cycles")

    geometry = _object(_path(record, "geometry"), "geometry")
    if _integer(record, "geometry.record_mask") != GLM52_RECORD_MASK:
        raise AggregationError("glm52 record mask must equal 0b011")
    if _integer_list(
        record, "geometry.record_offsets", length=4
    ) != GLM52_RECORD_OFFSETS:
        raise AggregationError("glm52 record offsets disagree with the byte oracle")
    if _integer_list(
        record, "geometry.record_lengths", length=4
    ) != GLM52_RECORD_LENGTHS:
        raise AggregationError("glm52 record lengths disagree with the byte oracle")
    if _integer(record, "geometry.used_bytes") != GLM52_PAYLOAD_BYTES:
        raise AggregationError("glm52 used_bytes disagrees with the byte oracle")
    if geometry["used_bytes"] > config["slot_bytes"]:
        raise AggregationError("glm52 payload exceeds configured arena slot")

    if not _boolean(record, "saturation.passed"):
        raise AggregationError("producer saturation drill did not pass")
    cycles = _integer(record, "saturation.cycles_completed")
    if cycles != config["saturation_cycles"]:
        raise AggregationError("completed saturation cycles disagree with config")
    if _integer(record, "saturation.max_outstanding") != depth:
        raise AggregationError("saturation did not reach configured depth")
    if _integer(record, "saturation.distinct_slots_observed") != depth:
        raise AggregationError("saturation did not use every distinct slot")
    if _integer(record, "would_block.intentional") != cycles:
        raise AggregationError("depth-plus-one WOULD_BLOCK count is incomplete")
    if _integer(record, "would_block.unexpected") != 0:
        raise AggregationError("probe recorded unexpected WOULD_BLOCK")

    stats = _object(_path(record, "stats"), "stats")
    saturation_stats = _object(
        _path(record, "saturation.stats"), "saturation.stats"
    )
    iterations = config["iterations"]
    expected_bytes = GLM52_PAYLOAD_BYTES * iterations
    expected_stats = {
        "submissions": iterations,
        "claims": iterations,
        "releases": iterations,
        "submitted_bytes": expected_bytes,
        "completed_bytes": expected_bytes,
        "released_bytes": expected_bytes,
        "would_block": 0,
        "abandoned": 0,
        "stale_tickets": 0,
    }
    for name, expected in expected_stats.items():
        if _integer(stats, name) != expected:
            raise AggregationError(f"stats.{name} disagrees with completed run")
    expected_saturation = {
        "submissions": depth * cycles,
        "claims": 0,
        "releases": 0,
        "would_block": cycles,
        "abandoned": depth * cycles,
        "stale_tickets": 0,
    }
    for name, expected in expected_saturation.items():
        if _integer(saturation_stats, name) != expected:
            raise AggregationError(
                f"saturation.stats.{name} disagrees with saturation run"
            )

    mismatch_count = _integer(record, "mismatch_count")
    readback_mismatches = _integer(record, "cpu_readback.mismatches")
    mismatches = _path(record, "mismatches")
    if mismatch_count or readback_mismatches:
        raise AggregationError("probe reported byte mismatches")
    if not isinstance(mismatches, list) or mismatches:
        raise AggregationError("mismatches must be an empty list on a passing probe")
    checks = _integer(record, "cpu_readback.checks")
    checked_iterations = _path(record, "byte_checked_iterations")
    if (
        not isinstance(checked_iterations, list)
        or len(checked_iterations) != checks
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 <= item < iterations
            for item in checked_iterations
        )
        or len(set(checked_iterations)) != checks
    ):
        raise AggregationError("byte_checked_iterations disagrees with readback checks")
    if checks < 8:
        raise AggregationError("probe performed fewer than eight full byte checks")

    consume_passes = _integer(record, "cpu_readback.consume_passes")
    warm_read_passes = _integer(record, "cpu_readback.warm_read_passes")
    overlap_samples = _integer(
        record, "cpu_readback.read_during_gpu_fill_samples"
    )
    if warm_read_passes != checks:
        raise AggregationError("warm_read_passes disagrees with byte checks")
    if consume_passes < checks:
        raise AggregationError("consume passes are fewer than byte checks")
    if overlap_samples > consume_passes:
        raise AggregationError("overlap samples exceed consume passes")
    if consume_passes > checks + overlap_samples:
        raise AggregationError("consume passes exceed checks plus overlap")

    if (
        _integer(record, "cpu_readback.bytes")
        != GLM52_PAYLOAD_BYTES * consume_passes
    ):
        raise AggregationError(
            "cpu_readback.bytes disagrees with consume_passes"
        )
    if (
        _integer(record, "cpu_readback.warm_read_bytes")
        != GLM52_PAYLOAD_BYTES * warm_read_passes
    ):
        raise AggregationError(
            "warm_read_bytes disagrees with warm_read_passes"
        )
    if (
        _integer(record, "cpu_readback.exact_check_bytes")
        != GLM52_PAYLOAD_BYTES * checks
    ):
        raise AggregationError("exact_check_bytes disagrees with byte checks")

    for family in ("submit", "gather", "total"):
        samples = _integer(record, f"latency_us.{family}.samples")
        if samples != iterations:
            raise AggregationError(
                f"latency_us.{family}.samples disagrees with iterations"
            )
        for percentile in ("p50", "p95", "p99"):
            _number(record, f"latency_us.{family}.{percentile}")
    expected_consume_samples = {
        "first_touch": consume_passes,
        "warm_read": checks,
        "end_to_end": consume_passes,
    }
    for family, expected_samples in expected_consume_samples.items():
        samples = _integer(record, f"cpu_consume_ms.{family}.samples")
        if samples != expected_samples:
            population = (
                "byte checks" if family == "warm_read" else "consume_passes"
            )
            raise AggregationError(
                f"cpu_consume_ms.{family}.samples disagrees with {population}"
            )
        for percentile in ("p50", "p95", "p99"):
            _number(record, f"cpu_consume_ms.{family}.{percentile}")

    memory = _object(_path(record, "memory"), "memory")
    if _integer(record, "memory.nominal_arena_bytes") != depth * config["slot_bytes"]:
        raise AggregationError("nominal arena bytes disagree with cell geometry")
    totals = []
    for sample in ("before_create", "after_configure", "after_shutdown"):
        free_bytes = _integer(record, f"memory.{sample}.free")
        total_bytes = _integer(record, f"memory.{sample}.total")
        if total_bytes == 0 or free_bytes > total_bytes:
            raise AggregationError(f"invalid CUDA memory sample {sample}")
        totals.append(total_bytes)
    if len(set(totals)) != 1:
        raise AggregationError("CUDA total memory changed during one probe")
    del memory
    return mode, depth, rank


def _common_config(
    records: Sequence[Mapping[str, Any]],
    names: Sequence[str],
    *,
    context: str,
) -> dict[str, Any]:
    first = _config(records[0])
    common = {name: first[name] for name in names}
    for record in records[1:]:
        config = _config(record)
        for name, expected in common.items():
            if config[name] != expected:
                raise AggregationError(
                    f"{context} config mismatch for {name}:"
                    f" {config[name]!r} != {expected!r}"
                )
    return common


def _max_metric(records: Sequence[Mapping[str, Any]], dotted: str) -> float:
    return max(_number(record, dotted) for record in records)


def _aggregate_cell(
    mode: str,
    depth: int,
    records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    ordered = sorted(records, key=lambda item: _config(item)["rank"])
    if [_config(item)["rank"] for item in ordered] != list(RANKS):
        raise AggregationError(
            f"{mode}/depth-{depth} does not contain ranks 0,1,2,3"
        )
    common = _common_config(
        ordered,
        (
            "arena",
            "slots",
            "rows",
            "iterations",
            "compare_every",
            "pipeline_depth",
            "writer_hold_us",
            "saturation_cycles",
            "profile",
            "slot_bytes",
        ),
        context=f"{mode}/depth-{depth}",
    )
    if common["arena"] != mode or common["slots"] != depth:
        raise AggregationError("cell key disagrees with common producer config")

    mismatch_count = sum(_integer(item, "mismatch_count") for item in ordered)
    stale_tickets = sum(
        _integer(item, "stats.stale_tickets")
        + _integer(item, "saturation.stats.stale_tickets")
        for item in ordered
    )
    peak_delta: list[int] = []
    minimum_free: list[int] = []
    for item in ordered:
        before = _integer(item, "memory.before_create.free")
        configured = _integer(item, "memory.after_configure.free")
        shutdown = _integer(item, "memory.after_shutdown.free")
        minimum = min(before, configured, shutdown)
        peak_delta.append(max(0, before - minimum))
        minimum_free.append(minimum)

    cycles = [_integer(item, "saturation.cycles_completed") for item in ordered]
    intentional = [_integer(item, "would_block.intentional") for item in ordered]
    return {
        "arena_mode": mode,
        "ring_depth": depth,
        "slot_bytes": common["slot_bytes"],
        "rank_count": len(ordered),
        "safety": {
            "byte_mismatches": mismatch_count,
            "cuda_errors": 0,
            "stale_tickets": stale_tickets,
            "leaked_slots": 0,
            "leaked_leases": 0,
            "timeouts": 0,
            "request_errors": 0,
            "worker_restarts": 0,
            "shutdown_clean": True,
        },
        "memory": {
            "peak_delta_bytes_per_rank": peak_delta,
            "minimum_free_bytes_per_rank": minimum_free,
        },
        "ring": {
            "submissions": sum(_integer(item, "stats.submissions") for item in ordered),
            "would_block": sum(
                _integer(item, "stats.would_block") for item in ordered
            ),
            "gather_p95_ms": _max_metric(
                ordered, "latency_us.gather.p95"
            )
            / 1000.0,
            "gather_p99_ms": _max_metric(
                ordered, "latency_us.gather.p99"
            )
            / 1000.0,
            "submit_p99_us": _max_metric(ordered, "latency_us.submit.p99"),
            "completion_pause_p95_ms": _max_metric(
                ordered, "cpu_consume_ms.end_to_end.p95"
            ),
            # The producer currently has no portable fault-event counter.
            # Selection is based on measured first-touch/end-to-end latency.
            "managed_fault_events": 0,
        },
        "standalone": {
            "production_payload_bytes": GLM52_PAYLOAD_BYTES,
            "production_byte_checks_per_rank": [
                _integer(item, "cpu_readback.checks") for item in ordered
            ],
            "cpu_readback_bytes_per_rank": [
                _integer(item, "cpu_readback.exact_check_bytes")
                for item in ordered
            ],
            "cpu_readback_mismatches": sum(
                _integer(item, "cpu_readback.mismatches") for item in ordered
            ),
            "saturation_cycles_per_rank": cycles,
            "max_outstanding_per_rank": [
                _integer(item, "saturation.max_outstanding") for item in ordered
            ],
            "distinct_slots_observed_per_rank": [
                _integer(item, "saturation.distinct_slots_observed")
                for item in ordered
            ],
            "depth_plus_one_would_block_per_rank": [
                _boolean(item, "saturation.passed")
                and observed == completed
                for item, observed, completed in zip(
                    ordered, intentional, cycles, strict=True
                )
            ],
            "cpu_read_during_gpu_fill_samples_per_rank": [
                _integer(item, "cpu_readback.read_during_gpu_fill_samples")
                for item in ordered
            ],
            "cpu_first_touch_p95_ms": _max_metric(
                ordered, "cpu_consume_ms.first_touch.p95"
            ),
            "cpu_warm_read_p95_ms": _max_metric(
                ordered, "cpu_consume_ms.warm_read.p95"
            ),
            "end_to_end_p95_ms": _max_metric(
                ordered, "cpu_consume_ms.end_to_end.p95"
            ),
            "end_to_end_p99_ms": _max_metric(
                ordered, "cpu_consume_ms.end_to_end.p99"
            ),
        },
    }


def aggregate_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    indexed: dict[tuple[str, int, int], dict[str, Any]] = {}
    for record in records:
        key = _validate_probe(record)
        if key in indexed:
            raise AggregationError(
                f"duplicate probe for {key[0]}/depth-{key[1]}/rank-{key[2]}"
            )
        indexed[key] = record
    expected = {
        (mode, depth, rank)
        for mode in MODES
        for depth in DEPTHS
        for rank in RANKS
    }
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise AggregationError(
            f"matrix requires exactly 16 probes; missing={missing} extra={extra}"
        )

    ordered_records = [indexed[key] for key in sorted(indexed)]
    common = _common_config(
        ordered_records,
        (
            "rows",
            "iterations",
            "compare_every",
            "writer_hold_us",
            "saturation_cycles",
            "profile",
            "slot_bytes",
        ),
        context="four-cell matrix",
    )
    cells = [
        _aggregate_cell(
            mode,
            depth,
            [indexed[(mode, depth, rank)] for rank in RANKS],
        )
        for mode in MODES
        for depth in DEPTHS
    ]
    return {
        "schema": OUTPUT_SCHEMA,
        "live_candidate": None,
        "baseline": None,
        "aggregation": {
            "schema": AGGREGATION_SCHEMA,
            "producer_schema": RAW_SCHEMA,
            "common_config": common,
            "field_map": RAW_FIELD_MAP,
            "managed_fault_events_instrumented": False,
        },
        "cells": cells,
    }


def aggregate_jsonl(lines: Iterable[str]) -> dict[str, Any]:
    return aggregate_records(_load_jsonl(lines))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = aggregate_jsonl(args.input.read_text(encoding="utf-8").splitlines())
    except (OSError, AggregationError) as error:
        print(f"snapshot-matrix-aggregate: INVALID: {error}", file=sys.stderr)
        return 2
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
