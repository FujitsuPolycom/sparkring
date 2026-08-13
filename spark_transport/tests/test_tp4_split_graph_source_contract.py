"""Offline wiring checks for split and tiered graph-kernel strategies.

These checks keep the research selector narrow and preserve the reviewed
transport handoffs. They do not prove CUDA execution, mapped-memory ordering,
RDMA progress, numerical correctness, or performance.
"""

from pathlib import Path


_TRANSPORT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (_TRANSPORT / relative).read_text(encoding="utf-8")


def _function(source: str, start: str, end: str) -> str:
    return source[source.index(start) : source.index(end)]


def test_split_graph_launches_the_reviewed_eight_node_dag() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    enqueue = _function(
        source,
        "void GpuTp4TensorWorker::enqueue_graph(",
        "}  // namespace spark_transport",
    )
    launches = [
        "tp4_split_claim_command<<<",
        "tp4_split_stage_round0<<<",
        "tp4_split_publish_round0<<<",
        "tp4_split_wait_round0<<<",
        "tp4_split_reduce_round0<<<",
        "tp4_split_handoff_round1<<<",
        "tp4_split_reduce_round1<<<",
        "tp4_split_finish<<<",
    ]

    positions = [enqueue.index(launch) for launch in launches]
    assert positions == sorted(positions)
    assert enqueue.count("tp4_split_") == len(launches)
    assert "kSplitGraphTileBytes = 64U * 1024U" in source


def test_split_round_handoffs_retain_system_fences_and_cpu_round1_publish() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    publish0 = _function(
        source,
        "__global__ void tp4_split_publish_round0(",
        "__global__ void tp4_split_wait_round0(",
    )
    handoff1 = _function(
        source,
        "__global__ void tp4_split_handoff_round1(",
        "__global__ void tp4_split_reduce_round1(",
    )
    finish = _function(
        source,
        "__global__ void tp4_split_finish(",
        "}  // namespace",
    )

    assert "publish_sequence_block(&control0->producer_sequence" in publish0
    assert "publish_sequence_block(&control0->consumer_sequence" in handoff1
    assert "wait_for_sequence_block(&control0->acknowledgement_sequence" in handoff1
    assert "if (!tp4_protocol_uses_deferred_ack(protocol))" in handoff1
    assert "wait_for_sequence_block(&control1->remote_sequence" in handoff1
    assert "producer_sequence" not in handoff1
    assert "publish_sequence_block(&control1->consumer_sequence" in finish
    assert "wait_for_sequence_block(&control1->acknowledgement_sequence" in finish
    assert "if (!tp4_protocol_uses_deferred_ack(protocol))" in finish
    assert "publish_sequence_block(&control1->observed_sequence" in finish

    helper = _function(
        source,
        "__device__ void publish_sequence_block(",
        "__global__ void tp4_tensor_all_reduce(",
    )
    assert "__threadfence_system();" in helper


def test_tiered_selector_is_explicit_and_legacy_c_api_defaults_to_fused() -> None:
    session_header = _read("include/spark_transport/tp4_session.hpp")
    c_api_header = _read("include/spark_transport/tp4_c_api.h")
    c_api_source = _read("src/tp4_c_api.cpp")
    probe = _read("app/tp4_graph_q1_probe.cu")

    assert "Tp4GraphKernelStrategy::kFused" in session_header
    assert "spark_tp4_create_with_protocol_and_graph_kernel" in c_api_header
    assert "SPARK_TP4_GRAPH_KERNEL_TIERED_64K" in c_api_header
    assert "SPARK_TP4_GRAPH_STATUS_TIERED_64K" in c_api_header
    assert "SPARK_TP4_GRAPH_KERNEL_FUSED" in c_api_source
    assert "options.graph_kernel_strategy" in c_api_source
    assert 'argument == "--graph-kernel"' in probe
    assert 'graph_kernel == "tiered_64k"' in probe
    assert "fixed Q1 through Q512" in probe
    assert "options.operations_per_graph != 1" in probe
    assert '<< " graph_kernel="' in probe
    assert '<< " kernel_fused_nodes="' in probe
    assert '<< " kernel_split_64k_nodes="' in probe


def test_probe_runner_forwards_and_attests_graph_kernel_strategy() -> None:
    runner = _read("scripts/run_tp4_graph_q1_probe.ps1")

    assert '[ValidateSet("fused", "split_64k", "tiered_64k")]' in runner
    assert '[string]$GraphKernel = "fused"' in runner
    assert '"--graph-kernel $GraphKernel"' in runner
    assert '"graph_kernel=$GraphKernel(?:\\s|$)"' in runner
    assert '"kernel_fused_nodes=$expectedKernelFusedNodes' in runner
    assert '"kernel_split_64k_nodes=$expectedKernelSplitNodes' in runner
    assert '"slot_reuse_exercised=$expectedSlotReuse' in runner


def test_tiered_worker_rejects_eager_but_accepts_both_graph_protocols() -> None:
    worker = _read("src/gpu_tp4_tensor.cu")
    session = _read("src/tp4_session.cpp")
    strategy = _read("include/spark_transport/tp4_graph_kernel_strategy.hpp")

    assert "split_graph_q_supported(q)" in worker
    assert "research TP4 kernel strategies are restricted to graph all-reduce" in worker
    enqueue_graph = _function(
        worker,
        "void GpuTp4TensorWorker::enqueue_graph(",
        "}  // namespace spark_transport",
    )
    sequential_graph = enqueue_graph[
        enqueue_graph.index(
            "if (tp4_graph_kernel_uses_split(graph_kernel_strategy_, q))"
        ) :
    ]
    assert "tp4_protocol_uses_deferred_ack" not in sequential_graph
    assert "tp4_graph_kernel_uses_split(graph_kernel_strategy_, q)" in enqueue_graph
    assert "q >= 1 && q <= kTp4GraphAllreduceMaximumQ" in worker
    assert "tp4_graph_kernel_strategy_is_graph_only" in session
    assert "graph_capacity_supported(options_.payload_bytes)" in session
    assert "kTp4TieredSplitMinimumQ = 7" in strategy
    assert "q >= kTp4TieredSplitMinimumQ" in strategy


def test_deferred_split_waits_for_exact_prior_slot_generations() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    claim = _function(
        source,
        "__global__ void tp4_split_claim_command(",
        "__global__ void tp4_split_stage_round0(",
    )
    wait0 = _function(
        source,
        "__global__ void tp4_split_wait_round0(",
        "__global__ void tp4_split_reduce_round0(",
    )
    enqueue = _function(
        source,
        "void GpuTp4TensorWorker::enqueue_graph(",
        "}  // namespace spark_transport",
    )

    for barrier in (claim, wait0):
        assert "tp4_expected_reuse_credit(state->sequence, protocol)" in barrier
        assert "acknowledgement_sequence" in barrier
        assert "state->sequence" in barrier
    assert source.index("__global__ void tp4_split_claim_command(") < source.index(
        "__global__ void tp4_split_stage_round0("
    )
    assert source.index("__global__ void tp4_split_wait_round0(") < source.index(
        "__global__ void tp4_split_reduce_round0("
    )
    assert enqueue.index("tp4_split_claim_command<<<") < enqueue.index(
        "tp4_split_stage_round0<<<"
    )
    assert enqueue.index("tp4_split_wait_round0<<<") < enqueue.index(
        "tp4_split_reduce_round0<<<"
    )


def test_probe_receipt_requires_sequence_three_for_deferred_slot_reuse() -> None:
    probe = _read("app/tp4_graph_q1_probe.cu")
    runner = _read("scripts/run_tp4_graph_q1_probe.ps1")

    assert "status.completed_sequence >= 3" in probe
    assert '<< " payload_slots=" << payload_slots' in probe
    assert '<< " slot_reuse_exercised="' in probe
    assert '$expectedSlotReuse = if ($expected -ge 3)' in runner
    assert '$expectedPayloadSlots = 2' in runner


def test_probe_input_epoch_is_not_aligned_with_two_slot_reuse() -> None:
    probe = _read("app/tp4_graph_q1_probe.cu")

    assert "kInputEpochPeriod = 7" in probe
    assert "kInputEpochPeriod % 2 != 0" in probe
    assert "input_epoch % kInputEpochPeriod" in probe
    assert "kMaximumInputValue == 32" in probe
    assert "kMaximumReducedValue == 128" in probe
    assert "kBfloat16ConsecutiveIntegerLimit = 256" in probe

    # A physical slot's immediately prior generation is S-2. The odd period
    # guarantees a different exactly representable input epoch at every reuse.
    for sequence in range(3, 4097):
        assert sequence % 7 != (sequence - 2) % 7

    # Reproduce the probe's integer-valued inputs. BF16 has an eight-bit
    # significand, so every integer through 256 is exact; both reduction
    # stages remain below that consecutive-integer limit.
    for epoch in range(7):
        for element in range(8):
            values = [
                ((element * 3 + rank * 5) & 7) + 1 + epoch * 4
                for rank in range(4)
            ]
            assert max(values) <= 32
            assert values[0] + values[1] <= 64
            assert values[2] + values[3] <= 64
            assert sum(values) <= 128


def test_mixed_graph_restages_and_validates_each_node_epoch() -> None:
    probe = _read("app/tp4_graph_q1_probe.cu")
    capture = _function(
        probe,
        "CapturedGraph capture_mixed_q_graph(",
        "bool attempt_post_replay_capture(",
    )

    prepare = capture.index("prepare_mixed_node_input<<<")
    collective = capture.index("session.capture_all_reduce(")
    validate = capture.index("validate_mixed_node_output<<<")
    assert prepare < collective < validate
    assert "graph_epoch_offset + operation" in capture
    assert capture.count("operations_per_cycle);") == 2

    epoch = _function(
        probe,
        "__device__ unsigned long long mixed_node_input_epoch(",
        "__global__ void prepare_replay(",
    )
    assert "(replay - 1ULL) / 2ULL" in epoch
    assert "operations_per_cycle" in epoch
    assert "node_epoch_offset" in epoch

    # Reproduce the accepted A/B launch order and prove the generated node
    # epoch is exactly the logical sequence, including graph boundaries.
    graph_a = 3
    graph_b = 128
    operations_per_cycle = graph_a + graph_b
    epochs: list[int] = []
    replay = 0
    for _ in range(10):
        replay += 1
        cycle = (replay - 1) // 2
        epochs.extend(
            cycle * operations_per_cycle + operation + 1
            for operation in range(graph_a)
        )
        replay += 1
        cycle = (replay - 1) // 2
        epochs.extend(
            cycle * operations_per_cycle + graph_a + operation + 1
            for operation in range(graph_b)
        )
    assert epochs == list(range(1, 1311))
    assert all(
        epochs[index] % 7 != epochs[index - 2] % 7
        for index in range(2, len(epochs))
    )
