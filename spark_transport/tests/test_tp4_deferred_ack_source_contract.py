"""Offline wiring checks for the graph-only deferred-ACK experiment.

These checks do not prove CUDA, RDMA, or numerical correctness. They keep the
reviewed ownership barriers connected until the native Spark probe can compile
and run on the target hardware.
"""

from pathlib import Path


_TRANSPORT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (_TRANSPORT / relative).read_text(encoding="utf-8")


def test_gpu_reuse_credit_precedes_each_slot_send_write() -> None:
    source = _read("src/gpu_tp4_tensor.cu")
    kernel = source[
        source.index("__global__ void tp4_tensor_all_reduce") : source.index(
            "GpuTp4TensorWorker::GpuTp4TensorWorker"
        )
    ]

    round0_credit = kernel.index(
        "wait_for_sequence_block(&control0->acknowledgement_sequence"
    )
    round0_write = kernel.index("send0[index] = input[index]")
    round1_credit = kernel.index(
        "wait_for_sequence_block(&control1->acknowledgement_sequence"
    )
    round1_write = kernel.index("send1_pairs[index] =")

    assert round0_credit < round0_write
    assert round1_credit < round1_write
    assert "tp4_payload_slot_index(sequence, protocol)" in kernel
    assert "round0_buffer += slot * round0_layout.total_bytes" in kernel
    assert "round1_buffer += slot * round1_layout.total_bytes" in kernel


def test_host_posts_signaled_raw_credit_and_drains_final_generation() -> None:
    source = _read("src/tp4_session.cpp")

    assert "store_sequence(&control.reserved, sequence)" in source
    assert "offsetof(DoorbellControl, reserved)" in source
    assert "offsetof(DoorbellControl, acknowledgement_sequence)" in source
    assert "sizeof(sequence), doorbell_token, true" in source
    assert "endpoint.wait_for_send_through(doorbell_token)" in source
    assert "drain_deferred_credits();" in source
    assert "round-0 deferred-credit retirement" in source
    assert "round-1 deferred-credit retirement" in source


def test_deferred_endpoint_version_rejects_mixed_binaries_before_data() -> None:
    session = _read("src/tp4_session.cpp")
    endpoint_header = _read("include/spark_transport/verbs_endpoint.hpp")

    assert "kTwoSlotDeferredEndpointVersion = 2" in session
    assert "local.version = kTwoSlotDeferredEndpointVersion" in session
    assert "remote.version != local.version" in session
    assert "endpoint.connect(remote, local.version)" in session
    assert "expected_version = kEndpointVersion" in endpoint_header


def test_default_constructor_stays_on_serial_ack_protocol() -> None:
    header = _read("include/spark_transport/tp4_c_api.h")
    source = _read("src/tp4_c_api.cpp")

    assert "spark_tp4_create_with_protocol" in header
    default_create = source[
        source.index('extern "C" spark_tp4_handle spark_tp4_create(') :
        source.index(
            'extern "C" spark_tp4_handle spark_tp4_create_with_protocol('
        )
    ]
    assert "SPARK_TP4_ALLREDUCE_PROTOCOL_SERIAL_ACK" in default_create


def test_native_contract_targets_are_registered() -> None:
    cmake = _read("CMakeLists.txt")
    assert "tp4_allreduce_protocol_test" in cmake
    assert "tp4_c_api_layout_test" in cmake


def test_graph_runner_forwards_and_attests_selected_protocol() -> None:
    runner = _read("scripts/run_tp4_graph_q1_probe.ps1")
    assert '[ValidateSet("serial_ack", "two_slot_deferred_ack")]' in runner
    assert '"--allreduce-protocol $AllreduceProtocol"' in runner
    assert "allreduce_protocol=$AllreduceProtocol" in runner
    assert "protocol_status_match=true" in runner
    assert '[ValidateSet("burst", "isolated")]' in runner
    assert '"--timing-mode $TimingMode"' in runner
    assert '"--fixed-q $FixedQ"' in runner
    assert '"--entrypoint /usr/bin/env"' in runner
    assert '"--device0 $($node.Device0) --device1 $($node.Device1)"' in runner
    assert runner.count('Device0 = "rocep1s0f1"') == 2
    assert runner.count('Device1 = "rocep1s0f0"') == 2
    assert "device_output_ready_single_replay" in runner
    assert "device_output_ready_replay_throughput" in runner
    assert "p95_device_output_ready_us_per_graph" in runner


def test_isolated_timing_excludes_input_preparation_and_validation() -> None:
    source = _read("app/tp4_graph_q1_probe.cu")
    launch = source[
        source.index("const auto launch_graph =") : source.index(
            "for (int iteration = 0; iteration < options.warmup"
        )
    ]

    prepare = launch.index("prepare_replay<<<")
    start = launch.index("cudaEventRecord(start, stream)")
    graph = launch.index("cudaGraphLaunch(graph.executable, stream)")
    stop = launch.index("cudaEventRecord(stop, stream)")
    validation = launch.index("validate_active_output<<<")
    assert prepare < start < graph < stop < validation
    assert "options.timing_mode == TimingMode::kBurst" in source
