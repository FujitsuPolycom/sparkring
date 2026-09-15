"""Offline wiring checks for the research-only tiled-prefill native probe.

These checks inspect build definitions and protocol source guards. They do not
compile or run the probe. CUDA, RDMA, numerical, and performance qualification
require the four-rank qualification runner and one validated receipt per rank.
"""

from pathlib import Path
import re

from spark_transport.experiments.tiled_prefill.qualification import RECEIPT_SCHEMA


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_cmake_builds_an_isolated_native_tiled_prefill_probe() -> None:
    cmake = _read("CMakeLists.txt")

    target = cmake[
        cmake.index("add_executable(spark_tp4_tiled_prefill_probe") :
        cmake.index("add_executable(spark_tp4_bidirectional_prefill_probe")
    ]
    assert "app/tp4_tiled_prefill_probe.cu" in target
    assert "experiments/tiled_prefill/tiled_executor.cpp" in target
    assert "experiments/tiled_prefill/tiled_bulk_kernels.cu" in target
    assert "experiments/tiled_prefill/tiled_correctness_kernels.cu" in target
    assert "spark_transport_capi" not in target
    assert "integrations/vllm" not in target.lower()


def test_probe_binds_the_executor_to_cuda_and_two_nonblocking_verbs_ports() -> None:
    probe = _read("app/tp4_tiled_prefill_probe.cu")

    assert "class CudaTiledBulkPort final" in probe
    assert "class VerbsTiledEdgePort final" in probe
    assert "TiledExecutor executor" in probe
    assert "launch_stage_phase1_tile(" in probe
    fill = probe.index("launch_fill_correctness_input(")
    stage = probe.index("launch_stage_phase1_tile(")
    assert fill < stage
    assert "launch_reduce_phase1_tile(" in probe
    assert "launch_reduce_phase2_tile(" in probe
    assert "poll_send_completions(" in probe
    assert "pending_signaled_work_" in probe
    assert "logical_exchanges_" in probe
    remote_ready = probe.index(
        "load_word(word(request.remote_doorbell_offset))"
    )
    local_retired = probe.index(
        "pending_signaled_work_.count(request.work_id)", remote_ready
    )
    logical_complete = probe.index(
        "logical_exchanges_.erase(request.work_id)", local_retired
    )
    assert remote_ready < local_retired < logical_complete
    assert "drain_all()" in probe
    assert "TiledSignaledWorkGate signaled_work_" in probe
    assert "signaled_work_.try_begin_credit(now_ticks())" in probe
    assert "inject_peer_wire_credit(2, credit_offset)" in probe
    assert "inject_peer_wire_credit(1, credit_offset)" in probe
    assert "__atomic_load_n" in probe
    assert "__atomic_store_n" in probe


def test_probe_exchanges_exact_rank_and_operation_geometry_before_connect() -> None:
    probe = _read("app/tp4_tiled_prefill_probe.cu")

    assert "struct TiledGeometryInfo" in probe
    for field in (
        "capacity_class",
        "rank",
        "world_size",
        "query_rows",
        "elements_per_row",
        "tile_count",
        "slots_per_edge",
        "lanes_per_edge",
        "bytes_per_row",
        "active_bytes",
        "tile_payload_bytes",
        "slot_stride",
        "registered_bytes_per_edge",
    ):
        assert field in probe
    geometry_exchange = probe.index(
        "channel.exchange(local_geometry)"
    )
    endpoint_connect = probe.index("endpoint.connect(remote, local.version)")
    assert geometry_exchange < endpoint_connect
    assert "remote_geometry.rank != plan.peer_rank" in probe
    assert "tiled-prefill geometry mismatch between edge peers" in probe


def test_probe_consumes_the_fail_closed_qualification_cli_and_receipt_schema() -> None:
    probe = _read("app/tp4_tiled_prefill_probe.cu")

    for option in (
        "--arm-id",
        "--query-rows",
        "--elements-per-row",
        "--timing-mode",
        "--warmup-operations",
        "--measured-operations",
        "--guard-bytes",
        "--credit-delay-edge",
        "--credit-delay-us",
        "--inject-poison",
        "--receipt-schema",
        "--receipt-prefix",
        "--expected-payload-tile-bytes",
        "--expected-slots-per-edge",
        "--expected-lanes-per-edge",
        "--expected-registered-tile-storage-bytes-per-edge",
    ):
        assert f'argument == "{option}"' in probe

    assert RECEIPT_SCHEMA == "sparkring-tp4-tiled-prefill-probe/v2"
    assert f'"{RECEIPT_SCHEMA}"' in probe
    assert '"TP4_TILED_PREFILL_RECEIPT"' in probe
    assert "protocol_node_implementation\\\":\\\"native_executor" in probe
    assert "performance_claim_allowed\\\":false" in probe


def test_probe_guards_active_tails_and_drains_before_endpoint_teardown() -> None:
    probe = _read("app/tp4_tiled_prefill_probe.cu")

    assert "launch_fill_correctness_sentinels(" in probe
    assert "launch_fill_correctness_input(" in probe
    assert "launch_validate_correctness(" in probe
    assert "launch_validate_correctness_sentinels(" in probe
    operation_loop = probe[
        probe.index("for (std::uint64_t op = 0;") : probe.index(
            "launch_validate_correctness_sentinels("
        )
    ]
    assert "validate_operation(final_status)" in operation_loop
    assert operation_loop.index("executor.drain()") < operation_loop.index(
        "validate_operation(final_status)"
    )
    assert "executor.drain()" in probe
    assert "safe_to_release_registered_storage" in probe
    assert "return kPoisonExitCode" in probe


def test_steady_timing_excludes_warmup_and_repeated_validation() -> None:
    probe = _read("app/tp4_tiled_prefill_probe.cu")

    assert "elapsed_us / timing_operations" in probe
    assert "options.measured_operations" in probe
    assert "const double measured_stop = now_us()" in probe
    assert "if (!validate_each_operation)" in probe
    assert "validate_operation(final_status)" in probe


def test_probe_receipt_exposes_only_measured_host_timings() -> None:
    # Source wiring check only: runtime tests of qualification.py independently
    # reject invented measurements. CUDA execution remains a hardware check.
    probe = _read("app/tp4_tiled_prefill_probe.cu")
    receipt = probe[probe.index("std::string make_receipt(") : probe.index("int main(")]
    emitted_fields = set(re.findall(r'\\"([a-z0-9_]+)\\":', receipt))
    assert {"host_operation_us_min", "host_operation_us_p50",
            "host_operation_us_p95", "host_measured_window_us_per_operation",
            "device_timing_measured", "slot_credit_wait_timing_measured"} <= emitted_fields
    assert not any(field.startswith(("device_output_", "device_fully_",
        "steady_state_device_", "slot_credit_wait_us_", "stage_phase1_gpu_",
        "phase1_remote_wait_", "reduce_phase1_gpu_", "phase2_remote_wait_",
        "reduce_phase2_gpu_", "active_payload_gib_", "steady_state_active_payload_"))
        for field in emitted_fields)
    assert "std::chrono::steady_clock::now()" in probe
    assert "latencies.push_back(operation_end - operation_start)" in probe
    assert "positive_or_epsilon" not in receipt


def test_poison_receipt_requires_observed_executor_failure() -> None:
    probe = _read("app/tp4_tiled_prefill_probe.cu")
    arm = probe[probe.index("if (options.poison != PoisonInjection::kNone) {") :
                probe.index("const bool validate_each_operation")]
    guard = re.search(r"if\s*\(!final_status\.poisoned\)\s*\{([^}]+)\}", arm)
    assert guard is not None
    assert "throw std::runtime_error(" in guard.group(1)
    assert arm.index("final_status = executor.status()") < guard.start()
    assert guard.end() < arm.index("make_receipt(") < arm.index("return kPoisonExitCode")
