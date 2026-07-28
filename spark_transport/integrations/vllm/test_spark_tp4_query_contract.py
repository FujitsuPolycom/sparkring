from __future__ import annotations

import importlib

import pytest

import spark_tp4_query_contract as contract


def test_default_preserves_current_q6_runtime() -> None:
    assert contract.MAX_QUERY_ROWS == 6
    assert contract.SUPPORTED_QUERY_ROWS == frozenset(range(1, 7))


def test_c8_capture_plan_bounds_padding_and_preserves_full_widths() -> None:
    assert contract.capture_sizes(8) == [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        12,
        15,
        16,
        20,
        24,
        25,
        30,
        32,
        35,
        40,
    ]


def test_capture_buckets_preserve_c2_adaptive_widths() -> None:
    assert contract.capture_buckets(10, uniform_query_rows=5) == [
        1,
        2,
        3,
        4,
        5,
        6,
        8,
        10,
    ]


@pytest.mark.parametrize("value", [0, 9, True, "8"])
def test_capture_plan_rejects_invalid_concurrency(value: object) -> None:
    with pytest.raises(ValueError):
        contract.capture_sizes(value)  # type: ignore[arg-type]


def test_explicit_q40_runtime_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VLLM_SPARK_MAX_QUERY_ROWS", "40")
    reloaded = importlib.reload(contract)
    try:
        assert reloaded.MAX_QUERY_ROWS == 40
        assert reloaded.SUPPORTED_QUERY_ROWS == frozenset(range(1, 41))
    finally:
        monkeypatch.delenv("VLLM_SPARK_MAX_QUERY_ROWS")
        importlib.reload(contract)


@pytest.mark.parametrize("value", ["0", "41", "invalid"])
def test_runtime_query_limit_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("VLLM_SPARK_MAX_QUERY_ROWS", value)
    with pytest.raises(ValueError):
        importlib.reload(contract)
    monkeypatch.delenv("VLLM_SPARK_MAX_QUERY_ROWS")
    importlib.reload(contract)
