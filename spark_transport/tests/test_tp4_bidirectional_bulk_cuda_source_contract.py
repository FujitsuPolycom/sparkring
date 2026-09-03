"""Source contracts for the adaptive width-4096 RS/AG CUDA slice."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "tiled_prefill"


def test_descriptor_keeps_four_tiles_and_carries_adaptive_geometry() -> None:
    abi = (EXPERIMENT / "bidirectional_bulk_abi.hpp").read_text()
    assert "kBidirectionalBulkTilesPerShard = 4" in abi
    for query_rows, tile_kib in ((1024, 256), (2048, 512), (4096, 1024)):
        assert (
            f"bidirectional_bulk_tile_bytes({query_rows}) == "
            f"{tile_kib}U * 1024U"
        ) in abi
    assert (
        "bidirectional_bulk_tile_bytes(8192) == 2U * 1024U * 1024U"
        in abi
    )
    assert "static_assert(kBidirectionalBulkTilesPerShard == 4)" in abi
    assert "sizeof(BidirectionalBulkDescriptor) == 96" in abi
    assert "query_rows" in abi
    assert "elements_per_row" in abi
    assert "inbound_doorbell_offset_bytes" in abi
    assert "expected_doorbell_token" in abi
    kernels = (EXPERIMENT / "bidirectional_bulk_kernels.cuh").read_text()
    assert "exchange_stage - 1" in kernels
    assert "expected_doorbell_token == 0" in kernels
    assert "inbound_doorbell_offset_bytes % alignof(std::uint64_t)" in kernels


def test_kernels_cover_the_exact_rs_ag_phase_sequence() -> None:
    source = (EXPERIMENT / "bidirectional_bulk_kernels.cu").read_text()
    names = (
        "stage_initial_kernel",
        "reduce_forward_kernel",
        "reduce_finalize_seed_gather_kernel",
        "gather_forward_kernel",
        "gather_finish_kernel",
    )
    positions = [source.index(name) for name in names]
    assert positions == sorted(positions)
    assert "descriptor_valid(descriptor, 0, 0)" in source
    assert "descriptor_valid(descriptor, 0, 1)" in source
    assert "descriptor_valid(descriptor, 2, 2)" in source
    assert "descriptor_valid(descriptor, 3, 4)" in source
    assert "descriptor_valid(descriptor, 5, 5)" in source
    assert "descriptor.tensor_offset_bytes == expected_offset" in source
    assert "descriptor.active_bytes % kBidirectionalBulkCtaBytes == 0" in source
    assert "launch_blocks(query_rows)" in source
    assert "BidirectionalBulkDescriptor descriptor" in source
    assert "descriptor_pointer" not in source


def test_nic_written_payload_uses_explicit_system_scope_loads() -> None:
    source = (EXPERIMENT / "bidirectional_bulk_kernels.cu").read_text()
    assert "ld.relaxed.sys.global.v4.u32" in source
    assert source.count("load_relaxed_system(") >= 4
    assert source.count("__threadfence_system();") >= 4
    assert "secondary_expected_doorbell_token" in source
    assert "secondary_inbound_doorbell_offset_bytes" in source
    assert "ld.acquire.sys.global.u64" in source
    assert source.count("require_inbound_doorbell(incoming, descriptor)") == 4
    assert 'asm volatile("trap;")' in source


def test_smoke_emulates_all_six_hops_and_checks_exact_integer_sum() -> None:
    smoke = (
        EXPERIMENT / "bidirectional_bulk_cuda_smoke_test.cu"
    ).read_text()
    assert "cudaHostAllocMapped" in smoke
    assert "__atomic_store_n" in smoke
    assert "make_bidirectional_stage_initial_descriptor" in smoke
    assert "for (std::uint32_t hop = 0; hop < 2; ++hop)" in smoke
    assert "launch_bidirectional_reduce_finalize_seed_gather" in smoke
    assert "launch_bidirectional_gather_finish" in smoke
    assert "__bfloat162float(host[element]) != 10.0F" in smoke
    assert "query_shapes{1024, 2048, 4096" in smoke
    assert "exact_mismatches[shape] == 0" in smoke
    assert "deterministic_noninteger_input" in smoke
    assert "compare_to_rounded_fp32" in smoke
    assert "noninteger_max_abs=" in smoke
    assert "noninteger_mismatches=" in smoke
    assert "kAbsoluteTolerance" in smoke
    assert "kRelativeTolerance" in smoke
    assert "upload_descriptor" not in smoke
    assert "cudaMalloc(&descriptors" not in smoke
