from __future__ import annotations

import pytest

import spark_cudagraph_bucket_contract as contract
import spark_tp4_query_contract as transport_contract


def test_initial_prefill_policy_covers_observed_shapes_with_bounded_padding() -> None:
    assert contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS == (
        48,
        72,
        144,
        224,
        288,
        352,
        432,
        512,
    )
    assert contract.prefill_padding_plan(
        contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
    ) == {
        48: 48,
        69: 72,
        72: 72,
        143: 144,
        210: 224,
        279: 288,
        348: 352,
        417: 432,
        486: 512,
    }
    assert contract.maximum_observed_padding(
        contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
    ) == 26


def test_current_decode_and_prefill_buckets_remain_separate() -> None:
    assert contract.DECODE_CAPTURE_BUCKETS == (
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
    )
    assert contract.FULL_SPECULATIVE_DECODE_BUCKETS == (
        5,
        10,
        15,
        20,
        25,
        30,
        35,
        40,
    )
    assert max(contract.DECODE_CAPTURE_BUCKETS) == 40
    assert min(contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS) > 40
    assert contract.combined_capture_buckets(
        contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
    ) == contract.DECODE_CAPTURE_BUCKETS + (
        48,
        72,
        144,
        224,
        288,
        352,
        432,
        512,
    )


def test_prefill_graph_policy_does_not_expand_native_transport() -> None:
    assert transport_contract.ABSOLUTE_MAX_QUERY_ROWS == 40
    assert max(contract.DECODE_CAPTURE_BUCKETS) == (
        transport_contract.ABSOLUTE_MAX_QUERY_ROWS
    )
    assert min(contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS) > (
        transport_contract.ABSOLUTE_MAX_QUERY_ROWS
    )


def test_policy_allows_a_future_broader_bounded_bucket_set() -> None:
    broader = (64, 80, 160, 224, 288, 352, 448, 512)
    assert contract.validate_prefill_piecewise_buckets(broader) == broader
    assert contract.maximum_observed_padding(broader) == 31


@pytest.mark.parametrize(
    "buckets",
    [
        (),
        (48, 72, 144, 224, 288, 352, 432),
        (40, 72, 144, 224, 288, 352, 432, 512),
        (48, 72, 72, 224, 288, 352, 432, 512),
        (72, 48, 144, 224, 288, 352, 432, 512),
        (48, 72, 144, 224, 288, 352, 512),
        tuple(range(48, 65)) + (512,),
    ],
)
def test_prefill_policy_fails_closed_on_unbounded_or_malformed_buckets(
    buckets: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError):
        contract.validate_prefill_piecewise_buckets(buckets)


@pytest.mark.parametrize(
    "buckets",
    [
        (5, 10, 15, 20, 25, 30, 35),
        (5, 10, 15, 20, 25, 30, 35, 40, 45),
        (1, 5, 10, 15, 20, 25, 30, 35, 40),
    ],
)
def test_full_decode_contract_fails_closed_outside_q5_through_q40(
    buckets: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="FULL speculative decode"):
        contract.validate_full_speculative_decode_buckets(buckets)


def test_environment_contract_reconciles_split_and_combined_lists() -> None:
    configured = contract.contract_from_environment(
        {
            contract.DECODE_CAPTURE_ENV: contract.format_buckets(
                contract.DECODE_CAPTURE_BUCKETS
            ),
            contract.FULL_DECODE_CAPTURE_ENV: contract.format_buckets(
                contract.FULL_SPECULATIVE_DECODE_BUCKETS
            ),
            contract.PREFILL_PIECEWISE_CAPTURE_ENV: contract.format_buckets(
                contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
            ),
            contract.COMBINED_CAPTURE_ENV: contract.format_buckets(
                contract.combined_capture_buckets(
                    contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
                )
            ),
        }
    )
    assert configured.prefill_piecewise == (
        48,
        72,
        144,
        224,
        288,
        352,
        432,
        512,
    )
    assert configured.full_speculative_decode[-1] == 40
    assert configured.combined[-1] == 512


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("missing combined", None),
        ("noncanonical CSV", "1, 2"),
        ("combined disagreement", "1,2,3"),
    ],
)
def test_environment_contract_fails_closed(
    name: str, value: str | None
) -> None:
    del name
    environment = {
        contract.DECODE_CAPTURE_ENV: contract.format_buckets(
            contract.DECODE_CAPTURE_BUCKETS
        ),
        contract.FULL_DECODE_CAPTURE_ENV: contract.format_buckets(
            contract.FULL_SPECULATIVE_DECODE_BUCKETS
        ),
        contract.PREFILL_PIECEWISE_CAPTURE_ENV: contract.format_buckets(
            contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
        ),
        contract.COMBINED_CAPTURE_ENV: contract.format_buckets(
            contract.combined_capture_buckets(
                contract.DEFAULT_PREFILL_PIECEWISE_BUCKETS
            )
        ),
    }
    if value is None:
        del environment[contract.COMBINED_CAPTURE_ENV]
    elif value == "1, 2":
        environment[contract.PREFILL_PIECEWISE_CAPTURE_ENV] = value
    else:
        environment[contract.COMBINED_CAPTURE_ENV] = value
    with pytest.raises(ValueError):
        contract.contract_from_environment(environment)
