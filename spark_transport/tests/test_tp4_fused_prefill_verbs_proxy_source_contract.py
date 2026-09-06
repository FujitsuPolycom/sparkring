"""Source contract for the isolated fused prefill real-verbs proxy."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "tiled_prefill"


def test_global_arena_layout_is_flow_parity_and_rail_exact() -> None:
    header = (EXPERIMENT / "fused_prefill_verbs_proxy.hpp").read_text()
    assert "kFusedPrefillEndpointCount = 4" in header
    assert "kFusedPrefillArenaBytes" in header
    assert "64U * 1024U * 1024U + 1024U" in header
    assert "fused_prefill_flow" in header
    assert "fused_prefill_slot_offset" in header
    assert "FusedPrefillPlane::kOutgoingPrimary" in header
    assert "FusedPrefillPlane::kIncomingSecondary" in header
    assert "same CUDA-mapped" in header


def test_proxy_posts_payload_then_distinct_signaled_doorbell_per_rail() -> None:
    source = (EXPERIMENT / "fused_prefill_verbs_proxy.cpp").read_text()
    payload = source.index("payload_id, false")
    doorbell = source.index("doorbell_id, true")
    assert payload < doorbell
    assert "primary_doorbell" in source
    assert "secondary_doorbell" in source
    assert "require_sq(primary_endpoint, 2)" in source
    assert "require_sq(secondary_endpoint, 2)" in source
    assert "poll_send_completions" in source
    assert "kCompletionBatch = 32" in source


def test_reuse_requires_both_cqes_and_exact_peer_credit() -> None:
    abi = (EXPERIMENT / "fused_prefill_abi.hpp").read_text()
    source = (EXPERIMENT / "fused_prefill_verbs_proxy.cpp").read_text()
    assert "peer_credit[kFusedPrefillParitySlots]" in abi
    assert "state.exchange_cqe[0]" in source
    assert "state.exchange_cqe[1]" in source
    assert "state.credit_cqe" in source
    assert "control->peer_credit[parity]" in source
    assert "control->reuse[parity]" in source
    assert "future peer-credit token" in source


def test_operation_slot_offsets_and_reuse_distance_are_explicit() -> None:
    abi = (EXPERIMENT / "fused_prefill_abi.hpp").read_text()
    kernel = (EXPERIMENT / "fused_prefill_kernels.cu").read_text()
    header = (EXPERIMENT / "fused_prefill_verbs_proxy.hpp").read_text()
    source = (EXPERIMENT / "fused_prefill_verbs_proxy.cpp").read_text()
    assert "operation_slots{1}" in abi
    assert "operation_sequence - descriptor.operation_slots" in kernel
    assert "operation_sequence < descriptor.operation_slots" in kernel
    assert "registered_offset" in header
    assert "operation_slot" in header
    assert "operation_slot) * kFusedPrefillArenaBytes" in source
    assert "arena_.registered_offset +" in source


def test_credit_uses_reverse_direction_primary_qp_and_exact_consumer_source() -> None:
    source = (EXPERIMENT / "fused_prefill_verbs_proxy.cpp").read_text()
    assert "reverse_direction = -flow_direction(flow)" in source
    assert "fused_prefill_endpoint_index(reverse_direction, 0)" in source
    assert "offsetof(FusedPrefillHostControl, consumer)" in source
    assert "offsetof(FusedPrefillHostControl, peer_credit)" in source
    assert "credit_id, true" in source


def test_fail_stop_covers_monotonic_ids_sq_and_future_tokens() -> None:
    source = (EXPERIMENT / "fused_prefill_verbs_proxy.cpp").read_text()
    assert "operation sequence is not monotonic" in source
    assert "first fused prefill operation sequence must be zero" in source
    assert "work ID exhausted" in source
    assert "SQ capacity exhausted" in source
    assert "stale, unknown, or future" in source
    assert "future producer token" in source
    assert "future consumer token" in source
    assert "poison_sequence" in source
    assert "poison_after_exception" in source
    assert "drain all four CQs" in source
    assert 'asm volatile("yield"' in source
    assert "config_.cpu >= 0" in source
    assert "std::terminate()" in source


def test_receipt_accounts_exact_bytes_and_completions_per_qp() -> None:
    header = (EXPERIMENT / "fused_prefill_verbs_proxy.hpp").read_text()
    source = (EXPERIMENT / "fused_prefill_verbs_proxy.cpp").read_text()
    for field in (
        "payload_bytes",
        "doorbell_bytes",
        "credit_bytes",
        "cq_completions",
    ):
        assert field in header
        assert f"receipt_.{field}[endpoint]" in source


def test_real_verbs_target_is_probe_only_and_opt_in() -> None:
    cmake = (EXPERIMENT / "CMakeLists.txt").read_text()
    assert "SPARK_TILED_PREFILL_ENABLE_FUSED_PREFILL_VERBS_PROXY" in cmake
    assert "spark_fused_prefill_verbs_proxy" in cmake
    assert "fused_prefill_verbs_proxy.cpp" in cmake
