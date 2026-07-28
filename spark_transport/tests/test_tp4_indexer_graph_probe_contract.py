from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[1]
PROBE = (ROOT / "app" / "tp4_indexer_graph_probe.cu").read_text(
    encoding="utf-8"
)
CMAKE = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")


def test_probe_captures_three_separate_indexer_graph_executables() -> None:
    assert "constexpr std::array<std::uint32_t, 3> kQPattern{1U, 23U, 40U}" in PROBE
    assert "std::array<CapturedGraph, kQPattern.size()> graphs" in PROBE
    assert "graphs[index] = capture_graph(" in PROBE
    assert "session.capture_all_gather(input, output, q, stream)" in PROBE
    assert "q_pattern=1,23,40" in PROBE
    assert "captured_nodes=" in PROBE
    assert "captured_q_mask=" in PROBE
    assert "census_q1=1 census_q23=1 census_q40=1" in PROBE


def test_probe_is_byte_exact_and_checks_rank_order_on_every_replay() -> None:
    assert "validate_rank_ordered_output" in PROBE
    assert "source_rank" in PROBE
    assert "source_element" in PROBE
    assert "output[index] != expected" in PROBE
    assert "cudaStreamSynchronize(stream)" in PROBE
    assert "mismatched_int32=" in PROBE
    assert "byte_exact=" in PROBE
    assert "validated_bytes=" in PROBE
    assert "monotonic_sequences=" in PROBE


def test_probe_requires_two_ring_wraps_and_exact_status() -> None:
    assert "constexpr std::uint64_t kRequiredRingWraps = 2" in PROBE
    assert "kTp4GraphCommandCapacity" in PROBE
    assert "normal probe requires enough cycles for two ring wraps" in PROBE
    for field in (
        "published_sequence == expected_sequence",
        "consumed_sequence == expected_sequence",
        "completed_sequence == expected_sequence",
        "overflow_sequence == 0",
        "host_native_atomics_supported",
        "submit_affinity_verified",
        "progress_affinity_verified",
        "graph_submit_cpu",
        "graph_progress_cpu",
    ):
        assert field in PROBE


def test_destructive_mismatch_mode_is_double_opt_in_and_never_normal() -> None:
    assert "--destructive-mismatch-q" in PROBE
    assert "--i-understand-mismatch-may-abort" in PROBE
    assert "options.destructive_mismatch_q !=" in PROBE
    assert "options.transport.rank % 2 == 0 ? 0 : 1" in PROBE
    assert "expected_outcome=bounded_transport_abort" in PROBE
    assert "TP4_INDEXER_GRAPH_MISMATCH_UNEXPECTED" in PROBE
    assert "if (options.destructive_mismatch_q)" in PROBE


def test_probe_is_a_first_class_cmake_target() -> None:
    assert "add_executable(spark_tp4_indexer_graph_probe" in CMAKE
    assert "app/tp4_indexer_graph_probe.cu" in CMAKE
    assert "target_link_libraries(spark_tp4_indexer_graph_probe" in CMAKE
