"""Fail-closed qualification contract for the tiled TP4 native probe.

This module is offline and research-only. It plans a bounded four-rank matrix
and validates JSON receipts emitted by a standalone native probe. It never
opens network connections, launches CUDA, or enables serving.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

from spark_transport.experiments.tiled_prefill.gpu_harness import (
    CONTROL_THREADS,
    PAYLOAD_TILE_BYTES,
    SLOTS_PER_EDGE,
    WORKER_CTA_BYTES,
    WORKER_THREADS,
    build_harness_plan,
)


PLAN_SCHEMA = "sparkring-tp4-tiled-prefill-qualification-plan/v1"
RECEIPT_SCHEMA = "sparkring-tp4-tiled-prefill-probe/v1"
AGGREGATE_SCHEMA = "sparkring-tp4-tiled-prefill-rank-gate/v1"
RECEIPT_PREFIX = "TP4_TILED_PREFILL_RECEIPT "
WORLD_SIZE = 4
EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE = 8_389_632
GUARD_BYTES = 4096
EXPECTED_POISON_EXIT_CODE = 42
PRIMARY_QUERY_ROWS = (40, 512, 1024, 4096)
BOUNDARY_QUERY_ROWS = (513, 1025)
GLM_QUERY_ROWS = (1024, 2048, 4096, 8192)
GPU_ABI_SYMBOLS = (
    "TiledBulkDescriptor",
    "TiledTimingReceipt",
    "TiledCorrectnessReceipt",
    "launch_stage_phase1_tile",
    "launch_reduce_phase1_tile",
    "launch_reduce_phase2_tile",
)


class ReceiptValidationError(ValueError):
    """A probe receipt failed a qualification invariant."""


@dataclass(frozen=True)
class QualificationArm:
    """One four-rank invocation with one unambiguous evidence purpose."""

    arm_id: str
    purpose: str
    query_rows: int
    timing_mode: str
    warmup_operations: int
    measured_operations: int
    elements_per_row: int = 6144
    guard_bytes: int = GUARD_BYTES
    credit_delay_edge: int = -1
    credit_delay_us: int = 0
    poison: str = "none"
    expected_exit_code: int = 0

    @property
    def is_poison(self) -> bool:
        return self.poison != "none"

    @property
    def is_backpressure(self) -> bool:
        return self.credit_delay_edge in (0, 1)


def qualification_arms() -> tuple[QualificationArm, ...]:
    """Return the fixed evidence matrix in deterministic execution order."""

    iterations = {40: 200, 512: 50, 1024: 25, 4096: 10}
    arms: list[QualificationArm] = []
    for query_rows in PRIMARY_QUERY_ROWS:
        for timing_mode in ("isolated", "steady"):
            arms.append(
                QualificationArm(
                    arm_id=f"q{query_rows}_{timing_mode}",
                    purpose=(
                        "single-operation output readiness and retirement"
                        if timing_mode == "isolated"
                        else "back-to-back credit-limited throughput"
                    ),
                    query_rows=query_rows,
                    timing_mode=timing_mode,
                    warmup_operations=10,
                    measured_operations=iterations[query_rows],
                )
            )
    for query_rows in BOUNDARY_QUERY_ROWS:
        arms.append(
            QualificationArm(
                arm_id=f"q{query_rows}_boundary_correctness",
                purpose=(
                    "validate capacity transition, generation reuse, and "
                    "partial-tail sentinels without a performance claim"
                ),
                query_rows=query_rows,
                timing_mode="correctness",
                warmup_operations=2,
                measured_operations=4,
            )
        )
    glm_iterations = {1024: 25, 2048: 20, 4096: 10, 8192: 6}
    for query_rows in GLM_QUERY_ROWS:
        arms.append(
            QualificationArm(
                arm_id=f"glm4096_q{query_rows}_steady",
                purpose="GLM-5.3 width-4096 steady prefill qualification",
                query_rows=query_rows,
                timing_mode="steady",
                warmup_operations=10,
                measured_operations=glm_iterations[query_rows],
                elements_per_row=4096,
            )
        )
    for edge in (0, 1):
        arms.append(
            QualificationArm(
                arm_id=f"q4096_backpressure_edge{edge}",
                purpose=(
                    f"prove edge-{edge} delayed credit bounds slot reuse "
                    "without delaying output readiness"
                ),
                query_rows=4096,
                timing_mode="steady",
                warmup_operations=2,
                measured_operations=8,
                credit_delay_edge=edge,
                credit_delay_us=1000,
            )
        )
    for poison, query_rows in (
        ("unexpected_generation", 40),
        ("credit_regression", 512),
    ):
        arms.append(
            QualificationArm(
                arm_id=f"q{query_rows}_poison_{poison}",
                purpose=f"prove symmetric process-fatal {poison}",
                query_rows=query_rows,
                timing_mode="poison",
                warmup_operations=0,
                measured_operations=1,
                poison=poison,
                expected_exit_code=EXPECTED_POISON_EXIT_CODE,
            )
        )
    return tuple(arms)


def arm_by_id(arm_id: str) -> QualificationArm:
    """Return one exact matrix arm or fail without accepting ad-hoc values."""

    matches = [arm for arm in qualification_arms() if arm.arm_id == arm_id]
    if len(matches) != 1:
        raise ValueError(f"unknown tiled-prefill qualification arm: {arm_id}")
    return matches[0]


def _arm_plan(arm: QualificationArm) -> dict[str, object]:
    gpu = build_harness_plan(
        arm.query_rows, elements_per_row=arm.elements_per_row
    )
    total_operations = arm.warmup_operations + arm.measured_operations
    return {
        **asdict(arm),
        "active_bytes": gpu.payload_bytes,
        "logical_tile_count": gpu.tile_count,
        "tail_active_bytes": (
            gpu.descriptors[-1].active_bytes
            if gpu.partial_tail_bytes
            else PAYLOAD_TILE_BYTES
        ),
        "partial_tail": gpu.partial_tail_bytes != 0,
        "worker_ctas_per_bulk_phase": gpu.worker_ctas_per_bulk_phase,
        "total_operations": total_operations,
        "expected_tiles_acquired": (
            None if arm.is_poison else total_operations * gpu.tile_count
        ),
        "requires_slot_reuse": gpu.tile_count > SLOTS_PER_EDGE,
        "requires_output_ready_before_retirement": arm.is_backpressure,
        "probe_arguments": [
            "--arm-id",
            arm.arm_id,
            "--query-rows",
            str(arm.query_rows),
            "--elements-per-row",
            str(arm.elements_per_row),
            "--timing-mode",
            arm.timing_mode,
            "--warmup-operations",
            str(arm.warmup_operations),
            "--measured-operations",
            str(arm.measured_operations),
            "--guard-bytes",
            str(arm.guard_bytes),
            "--credit-delay-edge",
            str(arm.credit_delay_edge),
            "--credit-delay-us",
            str(arm.credit_delay_us),
            "--inject-poison",
            arm.poison,
            "--receipt-schema",
            RECEIPT_SCHEMA,
            "--receipt-prefix",
            RECEIPT_PREFIX.rstrip(),
            "--expected-payload-tile-bytes",
            str(PAYLOAD_TILE_BYTES),
            "--expected-slots-per-edge",
            str(SLOTS_PER_EDGE),
            "--expected-lanes-per-edge",
            str(gpu.storage.lanes_per_endpoint),
            "--expected-registered-tile-storage-bytes-per-edge",
            str(EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE),
        ],
    }


def qualification_plan() -> dict[str, object]:
    """Return the complete explicit-execution four-rank evidence contract."""

    storage = build_harness_plan(4096).storage
    if (
        storage.registered_tile_storage_bytes_per_edge
        != EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE
    ):
        raise RuntimeError("GPU and qualification storage geometry disagree")
    return {
        "schema": PLAN_SCHEMA,
        "status": "research-only",
        "runnable": True,
        "world_size": WORLD_SIZE,
        "receipt_schema": RECEIPT_SCHEMA,
        "receipt_prefix": RECEIPT_PREFIX,
        "gpu_abi_symbols": list(GPU_ABI_SYMBOLS),
        "geometry": {
            "payload_tile_bytes": PAYLOAD_TILE_BYTES,
            "worker_cta_bytes": WORKER_CTA_BYTES,
            "worker_threads": WORKER_THREADS,
            "control_threads": CONTROL_THREADS,
            "slots_per_edge": SLOTS_PER_EDGE,
            "lanes_per_edge": storage.lanes_per_endpoint,
            "logical_payload_capacity_bytes_per_edge": (
                storage.logical_payload_capacity_bytes_per_edge
            ),
            "registered_tile_storage_bytes_per_edge": (
                storage.registered_tile_storage_bytes_per_edge
            ),
            "descriptor_storage_included": False,
        },
        "arms": [_arm_plan(arm) for arm in qualification_arms()],
        "execution_contract": {
            "default_action": "plan-only",
            "remote_execution_requires_explicit_switch": True,
            "binary_sha256_must_match_across_ranks": True,
            "one_receipt_per_rank_per_arm": True,
            "watchdog_required": True,
            "serving_integration": False,
            "native_executor_available": True,
            "protocol_nodes_status": "standalone-native-executor-bound",
            "first_executor_stream_policy": (
                "single_stream_correctness_only"
            ),
            "performance_claim_allowed": False,
            "performance_removal_gate": (
                "a reviewed multi-stream assignment must preserve every DAG "
                "edge and a shared-CQ dispatcher must preserve completion "
                "ownership while allowing independent edge work"
            ),
        },
        "qualification_gates": {
            "active_tail": (
                "Q40/Q513/Q1025 sentinel-fill both inactive capacity and "
                "outer input/output guards and report zero corruptions"
            ),
            "backpressure": (
                "each physical edge is delayed independently; in-flight tiles "
                "remain bounded by eight and output becomes ready while final "
                "retirement is pending at least once"
            ),
            "poison": (
                "unexpected generation and regressing credit each emit one "
                "poison receipt per rank and exit with code 42"
            ),
            "timing": (
                "isolated min/p50/p95 and steady-state mean are distinct; "
                "diagnostic phase envelopes are not summed and cannot support "
                "a performance claim under the single-stream executor"
            ),
            "association": (
                "the lower half reports xor1_then_xor3 and the upper half "
                "reports xor3_then_xor1 under an exact integer BF16 oracle"
            ),
            "teardown": (
                "every successful arm drains both edge credits and reports "
                "zero pending tiles and operations before process exit"
            ),
        },
    }


def _required(receipt: Mapping[str, object], name: str) -> object:
    if name not in receipt:
        raise ReceiptValidationError(f"receipt is missing {name}")
    return receipt[name]


def _integer(receipt: Mapping[str, object], name: str) -> int:
    value = _required(receipt, name)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ReceiptValidationError(f"receipt {name} must be an integer")
    return value


def _number(
    receipt: Mapping[str, object], name: str, *, positive: bool = False
) -> float:
    value = _required(receipt, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReceiptValidationError(f"receipt {name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ReceiptValidationError(f"receipt {name} must be finite")
    if result < 0 or (positive and result <= 0):
        comparator = "positive" if positive else "nonnegative"
        raise ReceiptValidationError(
            f"receipt {name} must be {comparator}"
        )
    return result


def _equals(
    receipt: Mapping[str, object], name: str, expected: object
) -> None:
    actual = _required(receipt, name)
    if actual != expected:
        raise ReceiptValidationError(
            f"receipt {name} expected {expected!r}, got {actual!r}"
        )


_ZERO_ON_SUCCESS = (
    "mismatched_active_elements",
    "input_guard_corruptions",
    "output_guard_corruptions",
    "inactive_input_sentinel_corruptions",
    "inactive_output_sentinel_corruptions",
    "unexpected_generation_count",
    "poisoned_slot_count",
    "descriptor_overflow_count",
    "credit_regression_count",
    "generation_overflow_count",
)


def _validate_common(receipt: Mapping[str, object], arm: QualificationArm) -> None:
    gpu = build_harness_plan(
        arm.query_rows, elements_per_row=arm.elements_per_row
    )
    _equals(receipt, "schema", RECEIPT_SCHEMA)
    _equals(receipt, "arm_id", arm.arm_id)
    _equals(receipt, "world_size", WORLD_SIZE)
    rank = _integer(receipt, "rank")
    if rank not in range(WORLD_SIZE):
        raise ReceiptValidationError("receipt rank must be in [0, 3]")
    _equals(receipt, "query_rows", arm.query_rows)
    _equals(receipt, "elements_per_row", arm.elements_per_row)
    _equals(receipt, "timing_mode", arm.timing_mode)
    _equals(receipt, "active_bytes", gpu.payload_bytes)
    _equals(receipt, "logical_tile_count", gpu.tile_count)
    _equals(receipt, "tail_active_bytes", gpu.descriptors[-1].active_bytes)
    _equals(receipt, "payload_tile_bytes", PAYLOAD_TILE_BYTES)
    _equals(receipt, "slots_per_edge", SLOTS_PER_EDGE)
    _equals(receipt, "lanes_per_edge", gpu.storage.lanes_per_endpoint)
    _equals(
        receipt,
        "registered_tile_storage_bytes_per_edge",
        EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE,
    )
    _equals(receipt, "descriptor_storage_included", False)
    _equals(receipt, "guard_bytes", arm.guard_bytes)
    _equals(receipt, "warmup_operations", arm.warmup_operations)
    _equals(receipt, "measured_operations", arm.measured_operations)
    _equals(receipt, "credit_delay_edge", arm.credit_delay_edge)
    _equals(receipt, "credit_delay_us", arm.credit_delay_us)
    _equals(receipt, "poison_injection", arm.poison)
    _equals(receipt, "association", "counter_rotating_halves")
    _equals(receipt, "lower_half_association", "xor1_then_xor3")
    _equals(receipt, "upper_half_association", "xor3_then_xor1")
    _equals(receipt, "numerical_oracle", "exact_integer_bf16")
    _equals(
        receipt,
        "executor_stream_policy",
        "single_stream_correctness_only",
    )
    _equals(receipt, "performance_claim_allowed", False)
    _equals(receipt, "protocol_node_implementation", "native_executor")


def _validate_success(
    receipt: Mapping[str, object], arm: QualificationArm
) -> None:
    gpu = build_harness_plan(
        arm.query_rows, elements_per_row=arm.elements_per_row
    )
    operations = arm.warmup_operations + arm.measured_operations
    expected_tiles = operations * gpu.tile_count
    _equals(receipt, "passed", True)
    _equals(receipt, "exit_code", 0)
    _equals(receipt, "fatal_state", "none")
    _equals(receipt, "completed_operations", operations)
    _equals(receipt, "tiles_acquired", expected_tiles)
    _equals(receipt, "tiles_recycled", expected_tiles)
    _equals(receipt, "highest_issued_ordinal", expected_tiles - 1)
    _equals(receipt, "highest_retired_ordinal_edge0", expected_tiles - 1)
    _equals(receipt, "highest_retired_ordinal_edge1", expected_tiles - 1)
    _equals(receipt, "teardown_state", "drained")
    _equals(receipt, "teardown_pending_tiles", 0)
    _equals(receipt, "teardown_pending_operations", 0)
    peak = _integer(receipt, "peak_tiles_in_flight")
    if not 1 <= peak <= SLOTS_PER_EDGE:
        raise ReceiptValidationError(
            "peak_tiles_in_flight must remain in [1, 8]"
        )
    for name in _ZERO_ON_SUCCESS:
        _equals(receipt, name, 0)
    _number(receipt, "host_submit_us_per_operation")
    for name in (
        "stage_phase1_gpu_us_p50",
        "phase1_remote_wait_us_p50",
        "reduce_phase1_gpu_us_p50",
        "phase2_remote_wait_us_p50",
        "reduce_phase2_gpu_us_p50",
    ):
        _number(receipt, name)
    if arm.timing_mode == "isolated":
        minimum = _number(
            receipt, "device_output_ready_us_min", positive=True
        )
        median = _number(
            receipt, "device_output_ready_us_p50", positive=True
        )
        p95 = _number(
            receipt, "device_output_ready_us_p95", positive=True
        )
        retired_median = _number(
            receipt, "device_fully_retired_us_p50", positive=True
        )
        retired_p95 = _number(
            receipt, "device_fully_retired_us_p95", positive=True
        )
        if not minimum <= median <= p95:
            raise ReceiptValidationError(
                "isolated output-ready quantiles must be monotonic"
            )
        if retired_median < median or retired_p95 < p95:
            raise ReceiptValidationError(
                "full retirement cannot precede output readiness"
            )
        _number(receipt, "active_payload_gib_per_s_p50", positive=True)
    elif arm.timing_mode == "steady":
        _number(
            receipt, "steady_state_device_us_per_operation", positive=True
        )
        _number(
            receipt,
            "steady_state_active_payload_gib_per_s",
            positive=True,
        )
    elif arm.timing_mode == "correctness":
        output_ready = _number(
            receipt, "device_output_ready_us_p50", positive=True
        )
        fully_retired = _number(
            receipt, "device_fully_retired_us_p50", positive=True
        )
        if fully_retired < output_ready:
            raise ReceiptValidationError(
                "full retirement cannot precede output readiness"
            )
    else:
        raise ReceiptValidationError("success arm has invalid timing mode")
    wait_events = _integer(receipt, "slot_credit_wait_events")
    if wait_events < 0:
        raise ReceiptValidationError(
            "slot_credit_wait_events must be nonnegative"
        )
    ready_before_retirement = _integer(
        receipt, "output_ready_before_final_retirement_count"
    )
    if not 0 <= ready_before_retirement <= operations:
        raise ReceiptValidationError(
            "output-ready-before-retirement count is out of range"
        )
    if arm.is_backpressure:
        if wait_events == 0:
            raise ReceiptValidationError(
                "backpressure arm did not observe a slot-credit wait"
            )
        if ready_before_retirement == 0:
            raise ReceiptValidationError(
                "backpressure arm did not separate output readiness from "
                "final retirement"
            )
        _number(receipt, "slot_credit_wait_us_p50", positive=True)
        _number(receipt, "slot_credit_wait_us_p95", positive=True)


def _validate_poison(
    receipt: Mapping[str, object], arm: QualificationArm
) -> None:
    _equals(receipt, "passed", False)
    _equals(receipt, "exit_code", EXPECTED_POISON_EXIT_CODE)
    _equals(receipt, "fatal_state", "poisoned")
    _equals(receipt, "completed_operations", 0)
    for name in (
        "mismatched_active_elements",
        "input_guard_corruptions",
        "output_guard_corruptions",
        "inactive_input_sentinel_corruptions",
        "inactive_output_sentinel_corruptions",
        "descriptor_overflow_count",
        "generation_overflow_count",
    ):
        _equals(receipt, name, 0)
    _equals(receipt, "poisoned_slot_count", 1)
    _equals(receipt, "teardown_state", "fatal_poisoned")
    if arm.poison == "unexpected_generation":
        _equals(receipt, "unexpected_generation_count", 1)
        _equals(receipt, "credit_regression_count", 0)
    elif arm.poison == "credit_regression":
        _equals(receipt, "unexpected_generation_count", 0)
        _equals(receipt, "credit_regression_count", 1)
    else:
        raise ReceiptValidationError("unknown poison contract")
    acquired = _integer(receipt, "tiles_acquired")
    recycled = _integer(receipt, "tiles_recycled")
    if acquired < 0 or recycled < 0 or recycled > acquired:
        raise ReceiptValidationError("poison tile counters are inconsistent")
    if acquired > build_harness_plan(
        arm.query_rows, elements_per_row=arm.elements_per_row
    ).tile_count:
        raise ReceiptValidationError(
            "poison injection occurred after more than one operation"
        )


def validate_receipt(
    receipt: Mapping[str, object], arm: QualificationArm
) -> None:
    """Validate one rank receipt against one exact matrix arm."""

    _validate_common(receipt, arm)
    if arm.is_poison:
        _validate_poison(receipt, arm)
    else:
        _validate_success(receipt, arm)


def validate_rank_receipts(
    receipts: Sequence[Mapping[str, object]], arm: QualificationArm
) -> dict[str, object]:
    """Require exactly one mutually consistent receipt from every TP4 rank."""

    if len(receipts) != WORLD_SIZE:
        raise ReceiptValidationError(
            f"expected {WORLD_SIZE} receipts, got {len(receipts)}"
        )
    for receipt in receipts:
        validate_receipt(receipt, arm)
    ranks = [_integer(receipt, "rank") for receipt in receipts]
    if sorted(ranks) != list(range(WORLD_SIZE)):
        raise ReceiptValidationError(
            f"receipt ranks must be exactly [0, 1, 2, 3], got {ranks}"
        )
    invariant_fields = (
        "schema",
        "arm_id",
        "world_size",
        "query_rows",
        "timing_mode",
        "active_bytes",
        "logical_tile_count",
        "tail_active_bytes",
        "registered_tile_storage_bytes_per_edge",
        "warmup_operations",
        "measured_operations",
        "credit_delay_edge",
        "credit_delay_us",
        "poison_injection",
        "lower_half_association",
        "upper_half_association",
        "executor_stream_policy",
        "performance_claim_allowed",
        "protocol_node_implementation",
        "teardown_state",
    )
    for name in invariant_fields:
        values = {json.dumps(receipt[name], sort_keys=True) for receipt in receipts}
        if len(values) != 1:
            raise ReceiptValidationError(
                f"rank receipts disagree on {name}"
            )
    return {
        "schema": AGGREGATE_SCHEMA,
        "status": "pass",
        "arm_id": arm.arm_id,
        "ranks": sorted(ranks),
        "query_rows": arm.query_rows,
        "timing_mode": arm.timing_mode,
        "poison": arm.poison,
        "expected_exit_code": arm.expected_exit_code,
        "executor_stream_policy": "single_stream_correctness_only",
        "performance_claim_allowed": False,
        "teardown_state": (
            "fatal_poisoned" if arm.is_poison else "drained"
        ),
        "registered_tile_storage_bytes_per_edge": (
            EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE
        ),
    }


def load_receipts(path: Path) -> list[Mapping[str, object]]:
    """Load plain JSONL or native probe lines bearing the receipt prefix."""

    receipts: list[Mapping[str, object]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(RECEIPT_PREFIX):
            line = line[len(RECEIPT_PREFIX) :]
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ReceiptValidationError(
                f"invalid JSON receipt at line {line_number}: {error}"
            ) from error
        if not isinstance(value, dict):
            raise ReceiptValidationError(
                f"receipt line {line_number} must be a JSON object"
            )
        receipts.append(value)
    return receipts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="print the offline matrix")
    plan.add_argument("--compact", action="store_true")
    validate = subparsers.add_parser(
        "validate", help="validate four native rank receipts"
    )
    validate.add_argument("--arm-id", required=True)
    validate.add_argument("--receipts", type=Path, required=True)
    return parser


def main(arguments: Iterable[str] | None = None) -> int:
    options = _build_parser().parse_args(arguments)
    try:
        if options.command == "plan":
            print(
                json.dumps(
                    qualification_plan(),
                    indent=None if options.compact else 2,
                    sort_keys=True,
                )
            )
            return 0
        arm = arm_by_id(options.arm_id)
        receipts = load_receipts(options.receipts)
        print(
            json.dumps(
                validate_rank_receipts(receipts, arm),
                sort_keys=True,
            )
        )
        return 0
    except (OSError, ValueError) as error:
        print(f"qualification error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
