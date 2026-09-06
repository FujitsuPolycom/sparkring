"""Source contract for the probe-only fused Q8192/N4 CUDA implementation."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "tiled_prefill"


def test_fixed_geometry_and_control_abi() -> None:
    abi = (EXPERIMENT / "fused_prefill_abi.hpp").read_text()
    assert "kFusedPrefillQueryRows = 8192" in abi
    assert "kFusedPrefillFlows = 8" in abi
    assert "kFusedPrefillCtasPerFlow = 4" in abi
    assert "kFusedPrefillParitySlots = 2" in abi
    assert "kFusedPrefillCtaBytes == 512U * 1024U" in abi
    assert "kFusedPrefillRailPlaneBytes == 2U * 1024U * 1024U" in abi
    assert "disjoint, 128-byte-aligned 2 MiB ranges" in abi
    assert "sizeof(FusedPrefillHostControl) == 128" in abi
    assert "alignof(FusedPrefillHostControl) == 128" in abi
    assert "FusedPrefillDeviceSync" in abi
    assert "kFusedPrefillBarrierPhases = 7" in abi
    assert "initial_tensor_offset_bytes" in abi
    for token in (
        "producer",
        "primary_doorbell",
        "secondary_doorbell",
        "consumer",
        "reuse",
        "poison_sequence",
    ):
        assert token in abi


def test_cooperative_kernel_has_six_independent_flow_stages() -> None:
    source = (EXPERIMENT / "fused_prefill_kernels.cu").read_text()
    assert "cudaLaunchCooperativeKernel" in source
    assert "cudaOccupancyMaxActiveBlocksPerMultiprocessor" in source
    assert "requested_blocks" in source
    assert "kFusedPrefillFlows * kFusedPrefillCtasPerFlow" in source
    assert "stage < kFusedPrefillStages" in source
    assert "blockIdx.x / kFusedPrefillCtasPerFlow" in source
    assert "descriptor.device_sync->arrivals[phase]" in source
    assert "descriptor.device_sync->sense[phase]" in source
    assert "descriptor.operation_sequence + 1U" in source
    assert "descriptor.operation_sequence < UINT32_MAX" in source
    assert "atomicExch(&descriptor.device_sync->arrivals[phase], 0U)" in source
    assert "aligned_and_disjoint" in source
    assert "left_end <= planes[right] || right_end <= planes[left]" in source


def test_protocol_waits_both_rails_and_fails_closed() -> None:
    source = (EXPERIMENT / "fused_prefill_kernels.cu").read_text()
    assert "ld.acquire.sys.global.u64" in source
    assert "primary_doorbell[parity]" in source
    assert "secondary_doorbell[parity]" in source
    assert "producer[0]" in source
    assert "producer[next_parity]" in source
    assert "reuse[next_parity]" in source
    assert "consumer[parity]" in source
    assert "observed > expected" in source
    assert "descriptor.spin_limit" in source
    assert "publish_poison(descriptor, stage)" in source
    assert "The GPU, not the CPU proxy, originates exchange zero" in source
    assert "wait_previous_operation_reuse(descriptor)" in source
    assert "fused_prefill_stage_token(previous, 4)" in source
    assert "fused_prefill_stage_token(previous, 5)" in source
    assert "wait_final_reuse(descriptor)" in source
    assert "descriptor.operation_sequence, 4" in source
    assert "descriptor.operation_sequence, 5" in source


def test_bf16_order_and_six_stage_rs_ag_actions() -> None:
    source = (EXPERIMENT / "fused_prefill_kernels.cu").read_text()
    assert "__hadd2(remote.pairs[pair], local.pairs[pair])" in source
    assert "stage < 2U" in source
    assert "stage == 2U" in source
    assert "stage < kFusedPrefillStages - 1U" in source
    assert "__threadfence_system();" in source
    assert "st.release.sys.global.u64" in source
