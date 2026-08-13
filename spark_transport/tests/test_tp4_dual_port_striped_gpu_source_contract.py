"""Offline wiring checks for the research-only striped GPU schedule.

These checks preserve the reviewed graph-node order and lane ownership. They
do not prove CUDA execution, mapped-memory ordering, RDMA progress, numerical
correctness, or performance on a four-rank Spark cluster.
"""

from pathlib import Path
import re


_TRANSPORT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (_TRANSPORT / relative).read_text(encoding="utf-8")


def _function_body(source: str, signature: str) -> str:
    start = source.index(signature)
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError(f"unterminated function: {signature}")


def test_striped_graph_launches_the_reviewed_eight_node_dag() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    enqueue = _function_body(source, "void GpuTp4TensorWorker::enqueue_graph(")
    striped_start = enqueue.index(
        "if (schedule_ == Tp4AllreduceSchedule::kDualPortStriped)"
    )
    sequential_start = enqueue.index(
        "if (tp4_graph_kernel_uses_split(graph_kernel_strategy_, q))"
    )
    striped = enqueue[striped_start:sequential_start]
    expected = [
        "tp4_striped_claim_and_gate<<<",
        "tp4_striped_stage_phase1<<<",
        "tp4_striped_publish_phase1<<<",
        "tp4_striped_wait_phase1<<<",
        "tp4_striped_reduce_phase1<<<",
        "tp4_striped_handoff_phase2<<<",
        "tp4_striped_reduce_phase2<<<",
        "tp4_striped_finish<<<",
    ]

    assert re.findall(r"tp4_striped_[a-z0-9_]+<<<", striped) == expected
    assert striped_start < sequential_start
    assert "active_payload_bytes != payload_bytes_" in striped
    assert "tp4_protocol_uses_deferred_ack(protocol_)" in striped
    assert "graph_kernel_strategy_ != Tp4GraphKernelStrategy::kFused" in striped
    assert "striped_graph_state_ == nullptr" in striped
    assert "bulk_blocks = 2 * stripe_blocks" in striped
    assert "kSplitGraphTileBytes = 64U * 1024U" in source


def test_striped_claim_gates_all_four_lanes_before_phase1_writes() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    claim = _function_body(source, "__global__ void tp4_striped_claim_and_gate(")
    stage = _function_body(source, "__global__ void tp4_striped_stage_phase1(")

    assert "__syncthreads();" in claim
    assert "tp4_expected_reuse_credit(state->sequence, protocol)" in claim
    assert claim.count("->acknowledgement_sequence") == 4
    for lane in (
        "endpoint0_lower",
        "endpoint0_upper",
        "endpoint1_lower",
        "endpoint1_upper",
    ):
        assert f"&{lane}->acknowledgement_sequence" in claim
    assert source.index("__global__ void tp4_striped_claim_and_gate(") < source.index(
        "__global__ void tp4_striped_stage_phase1("
    )
    assert "send[index] = input[input_offset + index]" in stage


def test_striped_lane_mapping_and_phase_handoffs_match_the_wire_schedule() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    stage = _function_body(source, "__global__ void tp4_striped_stage_phase1(")
    publish1 = _function_body(source, "__global__ void tp4_striped_publish_phase1(")
    wait1 = _function_body(source, "__global__ void tp4_striped_wait_phase1(")
    reduce1 = _function_body(source, "__global__ void tp4_striped_reduce_phase1(")
    handoff2 = _function_body(source, "__global__ void tp4_striped_handoff_phase2(")
    reduce2 = _function_body(source, "__global__ void tp4_striped_reduce_phase2(")
    finish = _function_body(source, "__global__ void tp4_striped_finish(")

    # Phase 1 stages lower/A on endpoint 0 and upper/B on endpoint 1.
    assert "? endpoint0_buffer\n          : endpoint1_buffer" in stage
    assert "&endpoint0_lower->producer_sequence" in publish1
    assert "&endpoint1_upper->producer_sequence" in publish1
    assert publish1.count("->producer_sequence") == 2
    assert "&endpoint0_lower->remote_sequence" in wait1
    assert "&endpoint1_upper->remote_sequence" in wait1

    # Cross-buffer reduction places lower/A on endpoint 1 and upper/B on
    # endpoint 0 for the opposite-order second phase.
    assert "source_endpoint" in reduce1
    assert "destination_endpoint" in reduce1
    assert "? endpoint1_buffer\n          : endpoint0_buffer" in reduce1
    assert "phase2_send[index] = __hadd2(send[index], receive[index])" in reduce1
    assert "? endpoint1_buffer\n          : endpoint0_buffer" in reduce2

    # GPU consumption releases phase 1. Only the host publishes phase-2
    # producers after observing both releases.
    assert "&endpoint0_lower->consumer_sequence" in handoff2
    assert "&endpoint1_upper->consumer_sequence" in handoff2
    assert "->producer_sequence" not in handoff2
    assert "&endpoint1_lower->remote_sequence" in handoff2
    assert "&endpoint0_upper->remote_sequence" in handoff2

    assert "&endpoint1_lower->consumer_sequence" in finish
    assert "&endpoint0_upper->consumer_sequence" in finish
    assert "&endpoint1_lower->observed_sequence" in finish
    assert "&endpoint0_upper->observed_sequence" in finish


def test_striped_schedule_is_graph_only_and_default_remains_sequential() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    worker_header = _read("include/spark_transport/gpu_tp4_tensor.hpp")
    schedule_header = _read(
        "include/spark_transport/tp4_dual_port_striped_allreduce.hpp"
    )
    eager = _function_body(source, "void GpuTp4TensorWorker::enqueue(")
    constructor = _function_body(source, "GpuTp4TensorWorker::GpuTp4TensorWorker(")

    assert "schedule_ != Tp4AllreduceSchedule::kSequential" in eager
    assert "restricted to graph all-reduce" in eager
    assert "tp4_protocol_uses_deferred_ack(protocol_)" in constructor
    assert "graph_kernel_strategy_ != Tp4GraphKernelStrategy::kFused" in constructor
    assert "Tp4AllreduceSchedule::kSequential" in worker_header
    assert 'return "sequential";' in schedule_header
    assert 'return "dual_port_striped";' in schedule_header
