from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_session_is_additive_shape_fixed_and_fail_closed() -> None:
    header = (
        ROOT
        / "include"
        / "spark_transport"
        / "tp4_bidirectional_prefill_session.hpp"
    ).read_text()
    source = (ROOT / "src" / "tp4_bidirectional_prefill_session.cpp").read_text()
    assert "class Tp4BidirectionalPrefillSession" in header
    assert "void all_reduce(const void* input, void* output" in header
    assert "make_bidirectional_ring_geometry" in source
    assert "geometry mismatch" in source
    assert "session is poisoned" in source
    assert "operation timed out" in source
    assert "cudaStreamSynchronize" in source
    assert "drain_edge" in source
    assert "secondary_endpoint0_" in source
    assert "DualRailEdgePort" in source
    assert "active_mtu_bytes() != 4096" in (
        ROOT / "src" / "tp4_bidirectional_prefill_dual_edge.inc"
    ).read_text()


def test_c_api_is_separate_and_uses_v2_exact_geometry() -> None:
    header = (
        ROOT
        / "include"
        / "spark_transport"
        / "tp4_bidirectional_prefill_c_api.h"
    ).read_text()
    source = (ROOT / "src" / "tp4_bidirectional_prefill_c_api.cpp").read_text()
    for symbol in (
        "spark_tp4_bidirectional_prefill_create",
        "spark_tp4_bidirectional_prefill_all_reduce",
        "spark_tp4_bidirectional_prefill_get_health_status",
        "spark_tp4_bidirectional_prefill_destroy",
    ):
        assert symbol in header
        assert symbol in source
    assert "spark_tp4_bidirectional_prefill_config_v1" in header
    assert "primary.base.payload_bytes != expected_payload" in source
    assert "primary.bytes_per_row != primary.elements_per_row * 2U" in source
    assert "single-rail config must omit secondary topology" in source
    assert "dual-rail config requires secondary topology" in source
    assert "rail_count must be one or two" in source
    assert "graph CPU affinity" in source


def test_cuda_default_stream_handle_is_valid() -> None:
    source = (ROOT / "src" / "tp4_bidirectional_prefill_session.cpp").read_text()
    assert "stream_pointer == nullptr" not in source
    assert "stream == nullptr" not in source
    assert "stream_ == nullptr" not in source
    assert "CUDA's valid legacy/default stream" in source
