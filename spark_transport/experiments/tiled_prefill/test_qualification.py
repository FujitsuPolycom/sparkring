"""Offline receipt gates for the four-rank tiled-prefill probe."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from spark_transport.experiments.tiled_prefill.gpu_harness import (
    build_harness_plan,
)
from spark_transport.experiments.tiled_prefill.qualification import (
    EXPECTED_POISON_EXIT_CODE,
    EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE,
    GPU_ABI_SYMBOLS,
    PLAN_SCHEMA,
    RECEIPT_PREFIX,
    RECEIPT_SCHEMA,
    ReceiptValidationError,
    arm_by_id,
    load_receipts,
    main,
    qualification_arms,
    qualification_plan,
    validate_rank_receipts,
)


def receipt_fixture(arm_id: str, rank: int) -> dict[str, object]:
    arm = arm_by_id(arm_id)
    gpu = build_harness_plan(
        arm.query_rows, elements_per_row=arm.elements_per_row
    )
    operations = arm.warmup_operations + arm.measured_operations
    tiles = operations * gpu.tile_count
    receipt: dict[str, object] = {
        "schema": RECEIPT_SCHEMA,
        "arm_id": arm.arm_id,
        "rank": rank,
        "world_size": 4,
        "query_rows": arm.query_rows,
        "elements_per_row": arm.elements_per_row,
        "timing_mode": arm.timing_mode,
        "active_bytes": gpu.payload_bytes,
        "logical_tile_count": gpu.tile_count,
        "tail_active_bytes": gpu.descriptors[-1].active_bytes,
        "payload_tile_bytes": 512 * 1024,
        "slots_per_edge": 8,
        "lanes_per_edge": 2,
        "registered_tile_storage_bytes_per_edge": (
            EXPECTED_REGISTERED_TILE_STORAGE_BYTES_PER_EDGE
        ),
        "descriptor_storage_included": False,
        "guard_bytes": arm.guard_bytes,
        "warmup_operations": arm.warmup_operations,
        "measured_operations": arm.measured_operations,
        "credit_delay_edge": arm.credit_delay_edge,
        "credit_delay_us": arm.credit_delay_us,
        "poison_injection": arm.poison,
        "association": "counter_rotating_halves",
        "lower_half_association": "xor1_then_xor3",
        "upper_half_association": "xor3_then_xor1",
        "numerical_oracle": "exact_integer_bf16",
        "executor_stream_policy": "single_stream_correctness_only",
        "performance_claim_allowed": False,
        "protocol_node_implementation": "native_executor",
        "mismatched_active_elements": 0,
        "input_guard_corruptions": 0,
        "output_guard_corruptions": 0,
        "inactive_input_sentinel_corruptions": 0,
        "inactive_output_sentinel_corruptions": 0,
        "descriptor_overflow_count": 0,
        "generation_overflow_count": 0,
    }
    if arm.is_poison:
        acquired = 0 if arm.poison == "unexpected_generation" else 2
        receipt.update(
            passed=False,
            exit_code=EXPECTED_POISON_EXIT_CODE,
            fatal_state="poisoned",
            completed_operations=0,
            tiles_acquired=acquired,
            tiles_recycled=0,
            poisoned_slot_count=1,
            unexpected_generation_count=(
                1 if arm.poison == "unexpected_generation" else 0
            ),
            credit_regression_count=(
                1 if arm.poison == "credit_regression" else 0
            ),
            teardown_state="fatal_poisoned",
        )
        return receipt

    receipt.update(
        passed=True,
        exit_code=0,
        fatal_state="none",
        completed_operations=operations,
        tiles_acquired=tiles,
        tiles_recycled=tiles,
        highest_issued_ordinal=tiles - 1,
        highest_retired_ordinal_edge0=tiles - 1,
        highest_retired_ordinal_edge1=tiles - 1,
        teardown_state="drained",
        teardown_pending_tiles=0,
        teardown_pending_operations=0,
        peak_tiles_in_flight=8 if gpu.tile_count > 8 else 1,
        unexpected_generation_count=0,
        poisoned_slot_count=0,
        credit_regression_count=0,
        slot_credit_wait_events=(12 if arm.is_backpressure else 0),
        output_ready_before_final_retirement_count=(
            operations if arm.is_backpressure else 0
        ),
        host_submit_us_per_operation=2.0,
        stage_phase1_gpu_us_p50=10.0,
        phase1_remote_wait_us_p50=20.0,
        reduce_phase1_gpu_us_p50=10.0,
        phase2_remote_wait_us_p50=20.0,
        reduce_phase2_gpu_us_p50=10.0,
    )
    if arm.timing_mode == "isolated":
        receipt.update(
            device_output_ready_us_min=90.0,
            device_output_ready_us_p50=100.0,
            device_output_ready_us_p95=120.0,
            device_fully_retired_us_p50=110.0,
            device_fully_retired_us_p95=140.0,
            active_payload_gib_per_s_p50=4.5,
        )
    else:
        receipt.update(
            steady_state_device_us_per_operation=80.0,
            steady_state_active_payload_gib_per_s=5.5,
        )
    if arm.timing_mode == "correctness":
        receipt.update(
            device_output_ready_us_p50=100.0,
            device_fully_retired_us_p50=120.0,
        )
    if arm.is_backpressure:
        receipt.update(
            slot_credit_wait_us_p50=1000.0,
            slot_credit_wait_us_p95=1100.0,
        )
    return receipt


def rank_fixtures(arm_id: str) -> list[dict[str, object]]:
    return [receipt_fixture(arm_id, rank) for rank in range(4)]


def test_plan_covers_requested_timing_backpressure_and_poison_matrix() -> None:
    plan = qualification_plan()

    assert plan["schema"] == PLAN_SCHEMA
    assert plan["status"] == "research-only"
    assert plan["runnable"] is True
    assert plan["execution_contract"]["native_executor_available"] is True
    assert (
        plan["execution_contract"]["protocol_nodes_status"]
        == "standalone-native-executor-bound"
    )
    assert plan["world_size"] == 4
    assert tuple(plan["gpu_abi_symbols"]) == GPU_ABI_SYMBOLS
    assert plan["geometry"] == {
        "payload_tile_bytes": 512 * 1024,
        "worker_cta_bytes": 64 * 1024,
        "worker_threads": 256,
        "control_threads": 32,
        "slots_per_edge": 8,
        "lanes_per_edge": 2,
        "logical_payload_capacity_bytes_per_edge": 4 * 1024 * 1024,
        "registered_tile_storage_bytes_per_edge": 8_389_632,
        "descriptor_storage_included": False,
    }
    arms = plan["arms"]
    assert [arm["arm_id"] for arm in arms] == [
        "q40_isolated",
        "q40_steady",
        "q512_isolated",
        "q512_steady",
        "q1024_isolated",
        "q1024_steady",
        "q4096_isolated",
        "q4096_steady",
        "q513_boundary_correctness",
        "q1025_boundary_correctness",
        "glm4096_q1024_steady",
        "glm4096_q2048_steady",
        "glm4096_q4096_steady",
        "glm4096_q8192_steady",
        "q4096_backpressure_edge0",
        "q4096_backpressure_edge1",
        "q40_poison_unexpected_generation",
        "q512_poison_credit_regression",
    ]
    q40 = arms[0]
    assert q40["active_bytes"] == 491_520
    assert q40["tail_active_bytes"] == 491_520
    assert q40["partial_tail"] is True
    assert q40["guard_bytes"] == 4096
    assert all(
        "--expected-registered-tile-storage-bytes-per-edge"
        in arm["probe_arguments"]
        for arm in arms
    )
    glm_q8192 = next(
        arm for arm in arms if arm["arm_id"] == "glm4096_q8192_steady"
    )
    assert glm_q8192["elements_per_row"] == 4096
    assert glm_q8192["active_bytes"] == 64 * 1024 * 1024


def test_plan_gpu_abi_symbols_exist_in_the_research_interface() -> None:
    directory = Path(__file__).parent
    interface = (directory / "tiled_bulk_kernels.cuh").read_text(
        encoding="utf-8"
    )
    abi = (directory / "tiled_bulk_abi.hpp").read_text(encoding="utf-8")
    source = (directory / "tiled_bulk_kernels.cu").read_text(
        encoding="utf-8"
    )

    assert f"struct {GPU_ABI_SYMBOLS[0]}" in abi
    for symbol in GPU_ABI_SYMBOLS[1:3]:
        assert f"struct {symbol}" in interface
    for symbol in GPU_ABI_SYMBOLS[3:]:
        assert symbol in interface
        assert symbol in source


@pytest.mark.parametrize(
    "arm_id",
    [
        "q40_isolated",
        "q512_steady",
        "q1024_isolated",
        "q4096_steady",
        "q513_boundary_correctness",
        "q1025_boundary_correctness",
        "q4096_backpressure_edge0",
        "q4096_backpressure_edge1",
        "q40_poison_unexpected_generation",
        "q512_poison_credit_regression",
    ],
)
def test_valid_four_rank_receipts_pass(arm_id: str) -> None:
    gate = validate_rank_receipts(rank_fixtures(arm_id), arm_by_id(arm_id))

    assert gate["status"] == "pass"
    assert gate["ranks"] == [0, 1, 2, 3]
    assert gate["arm_id"] == arm_id


def test_geometry_drift_from_registered_storage_fails_closed() -> None:
    receipts = rank_fixtures("q512_steady")
    receipts[2]["registered_tile_storage_bytes_per_edge"] = 4 * 1024 * 1024

    with pytest.raises(
        ReceiptValidationError,
        match="registered_tile_storage_bytes_per_edge",
    ):
        validate_rank_receipts(receipts, arm_by_id("q512_steady"))


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("output_guard_corruptions", 1, "output_guard_corruptions"),
        ("peak_tiles_in_flight", 9, "peak_tiles_in_flight"),
        ("tiles_recycled", 1, "tiles_recycled"),
        ("credit_regression_count", 1, "credit_regression_count"),
        ("steady_state_device_us_per_operation", float("nan"), "finite"),
    ],
)
def test_success_receipt_faults_fail_closed(
    field: str, value: object, match: str
) -> None:
    receipts = rank_fixtures("q4096_steady")
    receipts[0][field] = value

    with pytest.raises(ReceiptValidationError, match=match):
        validate_rank_receipts(receipts, arm_by_id("q4096_steady"))


def test_backpressure_requires_wait_and_completion_retirement_separation() -> None:
    receipts = rank_fixtures("q4096_backpressure_edge1")
    receipts[3]["slot_credit_wait_events"] = 0

    with pytest.raises(ReceiptValidationError, match="slot-credit wait"):
        validate_rank_receipts(
            receipts, arm_by_id("q4096_backpressure_edge1")
        )

    receipts = rank_fixtures("q4096_backpressure_edge1")
    receipts[3]["output_ready_before_final_retirement_count"] = 0
    with pytest.raises(ReceiptValidationError, match="output readiness"):
        validate_rank_receipts(
            receipts, arm_by_id("q4096_backpressure_edge1")
        )


def test_poison_receipts_require_exact_counter_and_exit_code() -> None:
    receipts = rank_fixtures("q40_poison_unexpected_generation")
    receipts[1]["exit_code"] = 0
    with pytest.raises(ReceiptValidationError, match="exit_code"):
        validate_rank_receipts(
            receipts, arm_by_id("q40_poison_unexpected_generation")
        )

    receipts = rank_fixtures("q512_poison_credit_regression")
    receipts[1]["credit_regression_count"] = 0
    with pytest.raises(ReceiptValidationError, match="credit_regression_count"):
        validate_rank_receipts(
            receipts, arm_by_id("q512_poison_credit_regression")
        )


def test_rank_set_and_cross_rank_contract_are_exact() -> None:
    receipts = rank_fixtures("q40_isolated")
    receipts[3]["rank"] = 2
    with pytest.raises(ReceiptValidationError, match="exactly"):
        validate_rank_receipts(receipts, arm_by_id("q40_isolated"))

    receipts = rank_fixtures("q40_isolated")
    receipts[3]["active_bytes"] = 1
    with pytest.raises(ReceiptValidationError, match="active_bytes"):
        validate_rank_receipts(receipts, arm_by_id("q40_isolated"))


def test_prefixed_jsonl_loader_and_cli_gate(tmp_path, capsys) -> None:
    arm_id = "q1024_steady"
    path = tmp_path / "receipts.jsonl"
    path.write_text(
        "\n".join(
            RECEIPT_PREFIX + json.dumps(receipt, sort_keys=True)
            for receipt in rank_fixtures(arm_id)
        ),
        encoding="utf-8",
    )

    assert len(load_receipts(path)) == 4
    assert main(
        ["validate", "--arm-id", arm_id, "--receipts", str(path)]
    ) == 0
    gate = json.loads(capsys.readouterr().out)
    assert gate["status"] == "pass"


def test_cli_returns_two_for_corrupt_receipt(tmp_path, capsys) -> None:
    receipts = rank_fixtures("q40_steady")
    corrupt = deepcopy(receipts)
    corrupt[0]["input_guard_corruptions"] = 1
    path = tmp_path / "corrupt.jsonl"
    path.write_text(
        "\n".join(json.dumps(receipt) for receipt in corrupt),
        encoding="utf-8",
    )

    assert main(
        [
            "validate",
            "--arm-id",
            "q40_steady",
            "--receipts",
            str(path),
        ]
    ) == 2
    assert "input_guard_corruptions" in capsys.readouterr().err


def test_arm_ids_are_unique_and_lookup_rejects_ad_hoc_arms() -> None:
    arm_ids = [arm.arm_id for arm in qualification_arms()]
    assert len(arm_ids) == len(set(arm_ids))
    with pytest.raises(ValueError, match="unknown"):
        arm_by_id("q513_steady")
