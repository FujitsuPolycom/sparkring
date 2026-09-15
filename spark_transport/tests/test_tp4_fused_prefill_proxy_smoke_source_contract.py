"""Source contract for the opt-in fused prefill mapped-memory proxy smoke."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "tiled_prefill"


def test_proxy_smoke_is_isolated_and_opt_in() -> None:
    cmake = (EXPERIMENT / "CMakeLists.txt").read_text()
    assert "SPARK_TILED_PREFILL_ENABLE_FUSED_PREFILL_PROXY_SMOKE" in cmake
    assert "tp4_fused_prefill_proxy_smoke_test" in cmake
    assert "fused_prefill_proxy_smoke_test.cu" in cmake
    assert "COMMAND tp4_fused_prefill_proxy_smoke_test --operations=100" in cmake


def test_proxy_orders_two_rail_visibility_consumption_and_delayed_reuse() -> None:
    source = (EXPERIMENT / "fused_prefill_proxy_smoke_test.cu").read_text()
    assert "wait_exact(&control->producer[parity]" in source
    assert "primary_doorbell[parity]" in source
    assert "secondary_doorbell[parity]" in source
    assert "wait_exact(&control->consumer[parity]" in source
    ordered_steps = (
        'wait_exact(&control->producer[parity]',
        'store_release(&control->primary_doorbell[parity], token)',
        'store_release(&control->secondary_doorbell[parity], token)',
        'wait_exact(&control->consumer[parity]',
        'delay_us(config.cqe_delay_us)',
        'delay_us(config.credit_delay_us)',
        'store_release(&control->reuse[parity], token)',
    )
    positions = [source.index(step) for step in ordered_steps]
    assert positions == sorted(positions)
    assert "control->reuse[parity]" in source
    assert "kFusedPrefillRailBytes" in source
    assert "flow_tile(flow_index) == 3U" in source
    assert "expected_outgoing" in source
    assert "outgoing_mismatches" in source
    assert "std::memcmp(observed_primary" in source


def test_proxy_runs_repeated_exact_and_noninteger_bitwise_oracles() -> None:
    source = (EXPERIMENT / "fused_prefill_proxy_smoke_test.cu").read_text()
    assert "operations{100}" in source
    assert "proxy smoke requires at least 100 operations" in source
    assert "run_case(false" in source
    assert "run_case(true" in source
    assert "bf16_add" in source
    assert "bitwise_mismatches" in source
    assert "mismatches != 0" in source
    assert "std::thread proxy" in source
    assert "std::rethrow_exception(proxy_error)" in source
