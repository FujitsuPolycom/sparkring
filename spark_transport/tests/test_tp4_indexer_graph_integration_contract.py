from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_gpu_worker_uses_graph_command_ring_not_eager_sequence() -> None:
    header = _read("include/spark_transport/gpu_tp4_allgather.hpp")
    source = _read("src/gpu_tp4_allgather.cu")

    assert "void enqueue_graph(" in header
    assert "gpu_graph_command::publish_command(" in source
    assert "Tp4GraphCommandKind::kIndexerAllgather" in source
    assert "kTp4IndexerGraphDescriptorVersion" in source
    assert "tp4_graph_doorbell_token(sequence, q)" in source
    assert source.count(
        "gpu_graph_command::wait_for_sequence_block("
    ) >= 4
    assert "q, 0, command_ring, trace" in source


def test_graph_session_is_fixed_arena_graph_only_and_strict() -> None:
    header = _read(
        "include/spark_transport/tp4_indexer_graph_session.hpp"
    )
    source = _read("src/tp4_indexer_graph_session.cpp")

    assert "class Tp4IndexerGraphSession" in header
    assert "void capture_all_gather(" in header
    assert "void all_gather(" not in header
    assert (
        "make_tp4_allgather_buffer_layout(\n"
        "            kTp4IndexerGraphMaximumInputBytes)"
    ) in source
    assert "tp4_graph_command_try_consume_tagged_layout(" in source
    assert "Tp4GraphCommandKind::kIndexerAllgather" in source
    assert "kTp4IndexerGraphBytesPerRow" in source
    assert "kTp4IndexerGraphMaximumQ" in source
    assert "cudaStreamCaptureStatusActive" in source
    assert "cudaDevAttrHostNativeAtomicSupported" in source


def test_c_abi_exposes_capture_and_progress_status() -> None:
    header = _read(
        "include/spark_transport/tp4_indexer_graph_c_api.h"
    )
    source = _read("src/tp4_indexer_graph_c_api.cpp")

    for symbol in (
        "spark_tp4_indexer_graph_create",
        "spark_tp4_indexer_capture_allgather",
        "spark_tp4_indexer_get_graph_status",
        "spark_tp4_indexer_graph_destroy",
    ):
        assert symbol in header
        assert symbol in source
    for field in (
        "captured_q_mask",
        "published_sequence",
        "consumed_sequence",
        "completed_sequence",
        "overflow_sequence",
    ):
        assert field in header
        assert field in source


def test_adapter_status_decodes_affinity_and_replay_counters() -> None:
    source = _read(
        "integrations/vllm/spark_tp4_allgather_backend.py"
    )

    assert '"submit_cpu": (' in source
    assert '"progress_cpu": (' in source
    assert "graph_submit_cpu_plus_one" in source
    assert "graph_progress_cpu_plus_one" in source
    assert '"replay_caught_up": (' in source
