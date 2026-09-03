"""Offline contracts for the research-only tiled prefill GPU slice."""

from __future__ import annotations

import json
from pathlib import Path

from spark_transport.experiments.tiled_prefill.gpu_harness import (
    PAYLOAD_TILE_BYTES,
    SLOTS_PER_EDGE,
    TIMING_FIELD_SEMANTICS,
    WORKER_CTA_BYTES,
    build_harness_plan,
    main,
    validate_harness_plan,
)


HERE = Path(__file__).parent


def test_q512_and_q4096_have_bounded_512k_multi_cta_geometry() -> None:
    q512 = build_harness_plan(512)
    q4096 = build_harness_plan(4096)

    assert q512.payload_bytes == 6 * 1024 * 1024
    assert q512.tile_count == 12
    assert q512.worker_ctas_per_bulk_phase == 12 * 8
    assert q512.active_worker_ctas_per_bulk_phase == 12 * 8
    assert q4096.payload_bytes == 48 * 1024 * 1024
    assert q4096.tile_count == 96
    assert q4096.worker_ctas_per_bulk_phase == 96 * 8
    assert q4096.active_worker_ctas_per_bulk_phase == 96 * 8
    assert all(
        descriptor.active_bytes == PAYLOAD_TILE_BYTES
        for descriptor in q512.descriptors + q4096.descriptors
    )
    assert q512.descriptors[-1].ticket.generation == 2
    assert q4096.descriptors[-1].ticket.generation == 12


def test_partial_tail_is_exact_and_inactive_ctas_cannot_expand_it() -> None:
    plan = build_harness_plan(513)
    tail = plan.descriptors[-1]

    assert plan.tile_count == 13
    assert tail.input_offset_bytes == 12 * PAYLOAD_TILE_BYTES
    assert tail.output_offset_bytes == 12 * PAYLOAD_TILE_BYTES
    assert tail.active_bytes == 6144 * 2
    assert tail.stripe_active_bytes == 6144
    assert tail.worker_ctas_per_bulk_phase == 8
    assert tail.active_worker_ctas_per_bulk_phase == 2
    assert plan.partial_tail_bytes == 6144 * 2
    assert sum(tile.active_bytes for tile in plan.descriptors) == 513 * 6144 * 2
    assert tail.input_offset_bytes + tail.active_bytes == plan.payload_bytes


def test_every_q1_q4096_retains_exact_active_ranges_and_worker_bounds() -> None:
    for query_rows in range(1, 4097):
        plan = build_harness_plan(query_rows)
        assert plan.payload_bytes == query_rows * 6144 * 2
        assert plan.descriptors[0].input_offset_bytes == 0
        assert plan.descriptors[0].output_offset_bytes == 0
        assert (
            plan.descriptors[-1].output_offset_bytes
            + plan.descriptors[-1].active_bytes
            == plan.payload_bytes
        )
        assert all(
            0 < descriptor.active_bytes <= PAYLOAD_TILE_BYTES
            and descriptor.active_bytes % 4 == 0
            and descriptor.worker_ctas_per_bulk_phase == 8
            and 0 < descriptor.active_worker_ctas_per_bulk_phase <= 8
            for descriptor in plan.descriptors
        )


def test_registered_storage_is_not_confused_with_logical_payload_capacity() -> None:
    storage = build_harness_plan(512).storage

    assert storage.payload_tile_bytes == 512 * 1024
    assert storage.stripe_capacity_bytes == 256 * 1024
    assert storage.lanes_per_endpoint == 2
    assert storage.lane_receive_offset == 256 * 1024
    assert storage.lane_control_offset == 512 * 1024
    assert storage.lane_stride == 512 * 1024 + 64
    assert storage.slot_stride == 1024 * 1024 + 128
    assert storage.logical_payload_capacity_bytes_per_edge == 4 * 1024 * 1024
    assert storage.registered_tile_storage_bytes_per_edge == 8 * (
        1024 * 1024 + 128
    )
    assert (
        storage.registered_tile_storage_bytes_per_edge
        > storage.logical_payload_capacity_bytes_per_edge
    )


def test_pipeline_requires_exact_prior_slot_credit_and_separates_completion() -> None:
    plan = build_harness_plan(512)
    by_id = {node.node_id: node for node in plan.nodes}

    for index in range(SLOTS_PER_EDGE):
        assert by_id[f"tile{index:03d}.acquire_credit"].dependencies == ()
    for index in range(SLOTS_PER_EDGE, plan.tile_count):
        assert by_id[f"tile{index:03d}.acquire_credit"].dependencies == (
            f"tile{index - SLOTS_PER_EDGE:03d}.retire_slot",
        )

    assert by_id[plan.output_ready_node].dependencies == tuple(
        f"tile{index:03d}.release_output"
        for index in range(plan.tile_count)
    )
    assert by_id[plan.fully_retired_node].dependencies == tuple(
        f"tile{index:03d}.retire_slot"
        for index in range(plan.tile_count)
    )
    final_generation = range(plan.tile_count - SLOTS_PER_EDGE, plan.tile_count)
    assert not {
        f"tile{index:03d}.retire_slot" for index in final_generation
    } & set(by_id[plan.output_ready_node].dependencies)
    validate_harness_plan(plan)


def test_timing_contract_names_scope_and_non_additive_diagnostics() -> None:
    receipt = build_harness_plan(4096).receipt()

    assert set(receipt["timing_fields"]) == set(TIMING_FIELD_SEMANTICS)
    assert "device_output_ready_us_p50" in receipt["timing_fields"]
    assert "steady_state_device_us_per_operation" in receipt["timing_fields"]
    assert "isolated_measured_operations" in receipt["timing_fields"]
    assert "steady_state_measured_operations" in receipt["timing_fields"]
    assert "device_fully_retired_us_p95" in receipt["timing_fields"]
    assert "slot_credit_wait_us_p95" in receipt["timing_fields"]
    assert "active_payload_gib_per_s_p50" in receipt["timing_fields"]
    assert receipt["timing_modes_required"] == [
        "isolated_output_ready_and_retirement",
        "back_to_back_steady_state_throughput",
    ]
    assert receipt["correctness_fields_required"] == [
        "mismatched_active_elements",
        "input_guard_corruptions",
        "output_guard_corruptions",
        "inactive_input_sentinel_corruptions",
        "inactive_output_sentinel_corruptions",
        "unexpected_generation_count",
        "credit_regression_count",
        "output_ready_before_final_retirement_count",
        "teardown_pending_tiles",
        "teardown_pending_operations",
    ]
    assert any(
        "must not be summed" in limitation
        for limitation in receipt["limitations"]
    )


def test_cli_emits_q512_and_q4096_offline_receipts(capsys) -> None:
    assert main(["--query-rows", "512", "4096"]) == 0
    receipts = json.loads(capsys.readouterr().out)

    assert [receipt["query_rows"] for receipt in receipts] == [512, 4096]
    assert [receipt["tile_count"] for receipt in receipts] == [12, 96]
    assert all(receipt["executor_available"] is False for receipt in receipts)
    assert all("research-only" in receipt["status"] for receipt in receipts)


def test_cuda_bulk_source_keeps_workers_bounded_and_protocol_free() -> None:
    source = (HERE / "tiled_bulk_kernels.cu").read_text(encoding="utf-8")
    interface = (HERE / "tiled_bulk_kernels.cuh").read_text(encoding="utf-8")
    bulk_abi = (HERE / "tiled_bulk_abi.hpp").read_text(encoding="utf-8")

    assert "kPayloadTileBytes = 512U * 1024U" in source
    assert "kWorkerCtaBytes = 64U * 1024U" in source
    assert "static_assert(kWorkerCtasPerTile == 8U)" in source
    assert source.count("if (!span.active)") == 3
    assert "packet_end = end - end % kPacketBytes" in source
    assert "pair_end" in source
    assert "__hadd2" in source
    assert "__hadd(" in source
    assert source.count(
        "<<<kWorkerCtasPerTile, kWorkerThreads, 0, stream>>>"
    ) == 3
    assert "tile_count * kWorkerCtasPerTile" not in source
    assert "batching descriptors" in source
    assert "Worker CTAs never poll a" in source
    assert "while (" not in source
    for forbidden in (
        "producer_sequence",
        "remote_sequence",
        "acknowledgement_sequence",
        "DoorbellControl",
    ):
        assert forbidden not in source
    for kernel in (
        "stage_phase1_tile",
        "reduce_phase1_tile",
        "reduce_phase2_tile",
    ):
        assert f"__global__ void {kernel}(" in source
    assert "struct TiledTimingReceipt" in interface
    assert "struct TiledCorrectnessReceipt" in interface
    assert "output_guard_corruptions" in interface
    assert "inactive_input_sentinel_corruptions" in interface
    assert "inactive_output_sentinel_corruptions" in interface
    assert "output_ready_before_final_retirement_count" in interface
    assert "TiledHalfAssociation lower_half_association" in interface
    assert "TiledHalfAssociation upper_half_association" in interface
    assert "kSingleStreamCorrectnessOnly" in bulk_abi
    assert "teardown_pending_tiles" in interface
    assert "must not batch later generations" in interface
    assert source.count("descriptor_pointer[0]") == 3


def test_worker_slice_divides_one_payload_tile_exactly() -> None:
    assert PAYLOAD_TILE_BYTES % WORKER_CTA_BYTES == 0
    assert PAYLOAD_TILE_BYTES // WORKER_CTA_BYTES == 8


def test_gpu_contract_is_not_linked_into_production_transport() -> None:
    cmake = (HERE.parents[1] / "CMakeLists.txt").read_text(encoding="utf-8")

    production_library = cmake[
        cmake.index("add_library(spark_transport")
        : cmake.index("add_library(spark_transport_capi")
    ]
    assert "tiled_bulk_kernels" not in production_library
    assert "tp4_tiled_executor_test" in cmake
    production_interfaces = cmake[
        cmake.index("add_library(spark_transport_capi")
        : cmake.index("add_executable(spark_tp4_tiled_prefill_probe")
    ]
    assert "experiments/tiled_prefill" not in production_interfaces
    assert "tiled_executor" not in production_interfaces
