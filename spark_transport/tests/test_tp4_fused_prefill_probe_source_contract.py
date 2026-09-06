"""Source contract for the research-only four-node fused verbs probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "tp4_fused_prefill_probe.cu"


def test_probe_is_opt_in_and_absent_from_serving_abi() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    assert "SPARK_TP4_ENABLE_FUSED_PREFILL_PROBE" in cmake
    target = cmake[cmake.index("if(SPARK_TP4_ENABLE_FUSED_PREFILL_PROBE)") :]
    assert "app/tp4_fused_prefill_probe.cu" in target
    assert "fused_prefill_verbs_proxy.cpp" in target
    assert "fused_prefill_kernels.cu" in target
    assert "spark_transport_capi" not in target.split("endif()", 1)[0]


def test_probe_requires_four_distinct_dual_rail_endpoints_and_mtu_4096() -> None:
    source = SOURCE.read_text()
    for option in (
        "--primary-device0",
        "--primary-device1",
        "--secondary-device0",
        "--secondary-device1",
        "--secondary-peer0",
        "--secondary-peer1",
        "--primary-gid0",
        "--secondary-gid1",
        "--primary-port0",
        "--secondary-port1",
    ):
        assert option in source
    assert "active_mtu_bytes() != kRequiredMtu" in source
    assert "peer.outgoing_direction != -direction" in source
    assert "peer.arena_bytes != geometry.arena_bytes" in source
    assert "*arena_buffer" in source
    assert "primary[clockwise_link].get()" in source
    assert "secondary[counterclockwise_link].get()" in source


def test_probe_launches_cooperatively_and_reports_three_timing_boundaries() -> None:
    source = SOURCE.read_text()
    assert "cudaDevAttrCooperativeLaunch" in source
    assert "cudaStreamNonBlocking" in source
    assert "launch_fused_prefill_q8192_n4" in source
    assert "cudaEventElapsedTime" in source
    assert "enqueue_us_p50" in source
    assert "device_us_p50" in source
    assert "full_proxy_retirement_us_p50" in source
    assert "proxy_worker.start(sequence)" in source
    assert "proxy_worker.wait()" in source


def test_probe_checks_exact_and_noninteger_outputs_guards_and_wire_bytes() -> None:
    source = SOURCE.read_text()
    assert "run_case(\n        false" in source
    assert "run_case(\n        true" in source
    assert "bf16_add" in source
    assert "output_mismatches" in source
    assert "input_mismatches" in source
    assert "input_guard_corruptions" in source
    assert "output_guard_corruptions" in source
    assert "payload_wire_bytes" in source
    assert "doorbell_wire_bytes" in source
    assert "credit_wire_bytes" in source
    assert "cq_completions" in source
    assert "per-endpoint accounting mismatch" in source
    assert '"clockwise_primary"' in source
    assert '"counterclockwise_secondary"' in source
    assert "proxy_receipt.payload_bytes[endpoint]" in source
