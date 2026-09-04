from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_health_snapshot_is_additive_and_host_only() -> None:
    header = (ROOT / "include/spark_transport/tp4_c_api.h").read_text()
    source = (ROOT / "src/tp4_c_api.cpp").read_text()
    session = (ROOT / "src/tp4_session.cpp").read_text()

    assert "spark_tp4_health_status" in header
    assert "spark_tp4_get_health_status" in header
    assert "spark_tp4_get_health_status" in source
    body = session.split(
        "Tp4HealthStatus health_status() const noexcept", 1
    )[1].split("\n  }", 1)[0]
    assert "cudaStreamSynchronize" not in body
    assert "submission_mutex_" not in body
    assert "progress_thread_running_" in body
    assert "poisoned_" in body
    assert "spark_tp4_get_abi_version" in header
    assert "spark_tp4_get_abi_version" in source


def test_fused_health_reads_mapped_device_poison_without_cuda_sync() -> None:
    source = (ROOT / "src" / "tp4_fused_prefill_session.cpp").read_text()
    body = source.split(
        "Tp4FusedPrefillHealthStatus health_status() const noexcept", 2
    )[2].split("\n  }", 1)[0]

    assert "poison_sequence" in body
    assert "load_mapped_poison" in body
    assert "failing_sequence" in body
    assert "failing_stage" in body
    assert "cudaStreamSynchronize" not in body


def test_bidirectional_health_is_host_only_and_exposed_through_c_api() -> None:
    header = (
        ROOT
        / "include"
        / "spark_transport"
        / "tp4_bidirectional_prefill_c_api.h"
    ).read_text()
    source = (
        ROOT / "src" / "tp4_bidirectional_prefill_session.cpp"
    ).read_text()
    body = source.split(
        "Tp4BidirectionalPrefillHealthStatus health_status() const noexcept",
        1,
    )[1].split("\n  }", 1)[0]

    assert "spark_tp4_bidirectional_prefill_get_health_status" in header
    assert "submitted_sequence_" in body
    assert "poisoned_" in body
    assert "cudaStreamSynchronize" not in body
