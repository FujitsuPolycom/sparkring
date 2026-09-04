from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_health_snapshot_is_additive_and_host_only() -> None:
    header = (ROOT / "include/spark_transport/tp4_c_api.h").read_text()
    source = (ROOT / "src/tp4_c_api.cpp").read_text()
    session = (ROOT / "src/tp4_session.cpp").read_text()

    assert "spark_tp4_health_status" in header
    assert "spark_tp4_get_health_status" in header
    assert "spark_tp4_get_health_status" in source
    body = session.split("Tp4HealthStatus health_status()", 1)[1].split(
        "\n  }", 1
    )[0]
    assert "cudaStreamSynchronize" not in body
    assert "progress_thread_running_" in body
    assert "poisoned_" in body
