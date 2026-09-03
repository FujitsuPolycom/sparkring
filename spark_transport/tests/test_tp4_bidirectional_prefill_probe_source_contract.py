"""Source contracts for the research-only bidirectional prefill probe."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_probe_is_standalone_and_admits_only_the_reviewed_q_matrix() -> None:
    cmake = (ROOT / "CMakeLists.txt").read_text()
    source = (ROOT / "app" / "tp4_bidirectional_prefill_probe.cu").read_text()
    assert "add_executable(spark_tp4_bidirectional_prefill_probe" in cmake
    assert "PRIVATE spark_transport)" in cmake
    assert "spark_transport_capi" not in cmake[
        cmake.index("add_executable(spark_tp4_bidirectional_prefill_probe") :
        cmake.index("include(CTest)")
    ]
    assert "--query-rows 1024|2048|4096|8192" in source
    assert "make_bidirectional_ring_geometry" in source
    assert "geometry.elements_per_row" in source
    assert "geometry.tile_bytes" in source


def test_probe_enforces_geometry_ordering_and_fail_stop_contracts() -> None:
    source = (ROOT / "app" / "tp4_bidirectional_prefill_probe.cu").read_text()
    assert "bidirectional geometry handshake mismatch" in source
    assert "make_bidirectional_post_exchange_descriptor" in source
    assert "upload bidirectional descriptor" not in source
    assert "BidirectionalBulkDescriptor* descriptor_" not in source
    assert "request.incoming_doorbell_offset" in source
    assert "request.consumed_doorbell_token" in source
    assert "observed > request.doorbell_token" in source
    assert "next_completion_id_[endpoint][rail]++" in source
    assert "safe_to_release_registered_storage" in source
    assert "bidirectional verbs drain failed" in source


def test_probe_receipt_is_honest_about_host_timing_and_wire_bytes() -> None:
    source = (ROOT / "app" / "tp4_bidirectional_prefill_probe.cu").read_text()
    assert "host_wall_us_p50" in source
    assert "device_us" not in source
    assert "endpoint0_rail0_payload_bytes" in source
    assert "endpoint0_rail1_payload_bytes" in source
    assert "endpoint1_doorbell_bytes" in source
    assert "endpoint0_credit_bytes" in source
    assert "output_mismatches" in source
    assert "output_guard_corruptions" in source


def test_probe_dual_rail_is_opt_in_atomic_and_observable() -> None:
    source = (ROOT / "app" / "tp4_bidirectional_prefill_probe.cu").read_text()
    assert "SPARK_TP4_BIDIRECTIONAL_RAIL_MODE" in source
    assert "RailMode::kSingle" in source
    assert "secondary_device0" in source
    assert "secondary_endpoint0" in source
    assert "active_mtu_bytes() != 4096" in source
    preflight = source.index("Preflight every rail")
    first_write = source.index("->write(", preflight)
    assert source.index("maximum_send_work_requests()", preflight) < first_write
    assert "DoorbellControl, command_sequence" in source
    assert "DoorbellControl, remote_sequence" in source
    assert "DoorbellControl, mismatch_count" in source
    assert "DoorbellControl, reserved" in source
    assert "endpoint0_rail1_cqes" in source
    assert "endpoint1_rail1_doorbell_bytes" in source
    assert "endpoint0_rail1_credit_bytes" in source
