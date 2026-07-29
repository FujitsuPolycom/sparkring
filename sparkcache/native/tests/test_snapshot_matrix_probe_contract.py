from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "app" / "spark_cache_snapshot_matrix_probe.cu"
CMAKE = ROOT / "CMakeLists.txt"


def test_matrix_probe_is_a_separate_cuda_target() -> None:
    cmake = CMAKE.read_text(encoding="utf-8")
    assert "spark_cache_snapshot_matrix_probe" in cmake
    assert "app/spark_cache_snapshot_matrix_probe.cu" in cmake
    assert "PRIVATE spark_cache_snapshot CUDA::cudart" in cmake


def test_matrix_probe_exposes_the_exact_live_matrix() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    for argument in (
        "--arena",
        "--slots",
        "--rank",
        "--rows",
        "--iterations",
        "--compare-every",
        "--pipeline-depth",
        "--writer-hold-us",
        "--profile",
        "--slot-mib",
        "--saturation-cycles",
        "--overlap-samples",
    ):
        assert argument in source
    assert "--arena must be mapped or managed" in source
    assert "--slots must be 2 or 3" in source
    assert "--rank must be in [0, 3]" in source
    assert "--rows must be 64 or 1024" in source
    assert "--pipeline-depth cannot exceed --slots" in source
    assert "--profile must be compact or glm52" in source
    assert "--slot-mib must be 2, 32, or 64" in source


def test_matrix_probe_matches_documented_fixture_formula() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "constexpr std::uint32_t kSourceRows = 2048" in source
    assert "fixture.host.assign(" in source
    assert "0xEE" in source
    assert "kind * 71U + layer * 43U + row * 17U + byte * 29U" in source
    assert "(11U + rank * 53U + index * 37U) % 2039U" in source
    for name in (
        "model.layers.00.mla",
        "model.layers.01.mla",
        "model.layers.00.indexer",
        "model.layers.00.mtp",
    ):
        assert name in source


def test_matrix_probe_has_exact_glm52_production_inventory() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "fixtures.reserve(101)" in source
    assert "layer < 79" in source
    assert "layer < 22" in source
    assert "SPARK_CACHE_SNAPSHOT_TARGET_CKV" in source
    assert "SPARK_CACHE_SNAPSHOT_SPARSE_INDEXER" in source
    assert "79U : 2U" in source
    assert "22U : 1U" in source
    assert "0b011U : 0b111U" in source
    assert "glm52 profile requires --slot-mib 32 or 64" in source


def test_matrix_probe_checks_every_declared_slice_and_reports_json() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "compare_fixture(" in source
    assert "payload[payload_offset]" in source
    assert "sparkcache.snapshot_matrix.v1" in source
    for field in (
        '"submit"',
        '"gather"',
        '"total"',
        "mismatch_count",
        "mismatches",
        "byte_checked_iterations",
        "submitted_bytes",
        "completed_bytes",
        "released_bytes",
        "intentional_would_block",
        "unexpected_would_block",
        "saturation_passed",
        "nominal_arena_bytes",
        "device_free_before_create",
        "device_free_after_configure",
        "device_free_after_shutdown",
        "cpu_readback_bytes",
        "cpu_readback_checksum",
        "cpu_consume_passes",
        "cpu_warm_read_bytes",
        "cpu_warm_read_passes",
        "cpu_exact_check_bytes",
        "cpu_read_during_gpu_fill_samples",
        "cpu_first_touch_ms",
        "cpu_warm_read_ms",
        "end_to_end_ms",
        "standalone_rank",
        "production_payload_bytes",
        "depth_plus_one_would_block",
    ):
        assert field in source


def test_matrix_probe_comparison_cadence_keeps_first_and_last() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "iteration == 0" in source
    assert "iteration + 1 == options.iterations" in source
    assert "iteration % options.compare_every == 0" in source


def test_matrix_probe_uses_external_stream_and_checked_shutdown() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "cudaStreamCreateWithFlags" in source
    assert "reinterpret_cast<std::uintptr_t>(stream)" in source
    assert "spark_cache_snapshot_shutdown(snapshot)" in source
    assert "spark_cache_snapshot_destroy(snapshot)" in source
    assert "cudaDeviceSynchronize" not in source


def test_matrix_probe_pipelines_tickets_fifo_with_per_ticket_metadata() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "std::deque<PendingTicket> pending" in source
    assert "pending.size() < options.pipeline_depth" in source
    assert "PendingTicket current = pending.front()" in source
    assert "pending.pop_front()" in source
    assert "current.context_sequence" in source
    assert "current.logical_start" in source
    assert "current.started" in source
    assert "current.submitted" in source
    assert "std::chrono::microseconds(options.writer_hold_us)" in source


def test_matrix_probe_has_isolated_explicit_saturation_drill() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert source.count("create_and_configure(&snapshot)") == 2
    assert "saturation fill submit" in source
    assert "saturation extra submit" in source
    assert "SPARK_CACHE_SNAPSHOT_WOULD_BLOCK" in source
    assert "spark_cache_snapshot_abandon_context(" in source
    assert "result.saturation_stats.would_block ==" in source
    assert "options.saturation_cycles" in source
    assert "result.saturation_stats.abandoned ==" in source
    assert "result.distinct_slots_observed == options.slot_count" in source


def test_matrix_probe_reports_cuda_memory_across_runtime_lifecycle() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "cudaMemGetInfo(&free_value, &total_value)" in source
    assert "cudaMemGetInfo(before create)" in source
    assert "cudaMemGetInfo(after configure)" in source
    assert "cudaMemGetInfo(after shutdown)" in source
    assert "options.slot_bytes * options.slot_count" in source


def test_matrix_probe_reports_v2_cpu_consumer_evidence() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "consume_payload_vectorized(ready)" in source
    assert "const std::uint64_t lane_count" in source
    assert "lane_checksum += lanes[lane]" in source
    assert "1099511628211ULL" not in source
    assert "poll later ticket" in source
    assert "later_status == SPARK_CACHE_SNAPSHOT_NOT_READY" in source
    assert '"first_touch"' in source
    assert '"warm_read"' in source
    assert '"end_to_end"' in source


def test_overlap_consume_uses_one_pass_and_sparse_checks_get_warm_pass() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    first = source.index("consume_payload_vectorized(ready)")
    exact = source.index("if (exact_check)", first)
    second = source.index("consume_payload_vectorized(ready)", first + 1)
    assert first < exact < second
    assert "result.cpu_exact_check_bytes += ready.used_bytes" in source
    assert "result.cpu_warm_read_passes += 1" in source


def test_overlap_detection_scans_every_later_pending_ticket() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    overlap_start = source.index("bool later_fill_in_flight = false")
    overlap_end = source.index(
        "if (consume_on_cpu && initialized)", overlap_start
    )
    overlap = source[overlap_start:overlap_end]
    assert "for (const auto& later : pending)" in overlap
    assert "&later.ticket" in overlap
    assert "pending.front().ticket" not in overlap
    assert "later_status == SPARK_CACHE_SNAPSHOT_NOT_READY" in overlap
    assert "later_status != SPARK_CACHE_SNAPSHOT_OK" in overlap
