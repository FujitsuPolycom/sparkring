from __future__ import annotations

import pytest

from . import decode_payload_contract as contract


EXPECTED_MAXIMA = (
    (5, 61_440, 81_920, 92_160, 1_280, 327_680, 387_200),
    (10, 122_880, 163_840, 184_320, 2_560, 655_360, 774_400),
    (15, 184_320, 245_760, 276_480, 3_840, 983_040, 1_161_600),
    (20, 245_760, 327_680, 368_640, 5_120, 1_310_720, 1_548_800),
    (25, 307_200, 409_600, 460_800, 6_400, 1_638_400, 1_936_000),
    (30, 368_640, 491_520, 552_960, 7_680, 1_966_080, 2_323_200),
    (35, 430_080, 573_440, 645_120, 8_960, 2_293_760, 2_710_400),
    (40, 491_520, 655_360, 737_280, 10_240, 2_621_440, 3_097_600),
)

EXPECTED_FUSED_MAXIMA = (
    (82_560, 41_280),
    (165_120, 82_560),
    (247_680, 123_840),
    (330_240, 165_120),
    (412_800, 206_400),
    (495_360, 247_680),
    (577_920, 288_960),
    (660_480, 330_240),
)

EXPECTED_INDEXER_Q1_Q40 = (
    16_384,
    32_768,
    49_152,
    65_536,
    81_920,
    98_304,
    114_688,
    131_072,
    147_456,
    163_840,
    180_224,
    196_608,
    212_992,
    229_376,
    245_760,
    262_144,
    278_528,
    294_912,
    311_296,
    327_680,
    344_064,
    360_448,
    376_832,
    393_216,
    409_600,
    425_984,
    442_368,
    458_752,
    475_136,
    491_520,
    507_904,
    524_288,
    540_672,
    557_056,
    573_440,
    589_824,
    606_208,
    622_592,
    638_976,
    655_360,
)

EXPECTED_CAPTURE_ROWS = (
    (1, 3, 5),
    (1, 2, 3, 4, 5, 6, 8, 10),
    (1, 2, 3, 4, 5, 6, 8, 10, 12, 15),
    (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20),
    (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25),
    (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30),
    (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 16, 20, 24, 25, 30, 32, 35),
    (
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
    ),
)


@pytest.mark.parametrize(
    ("concurrency", "expected"),
    enumerate(EXPECTED_MAXIMA, start=1),
)
def test_c1_c8_collective_maxima(
    concurrency: int,
    expected: tuple[int, ...],
) -> None:
    (
        maximum_q,
        tp_ar,
        indexer,
        dcp_query,
        dcp_lse,
        dcp_output,
        vocabulary,
    ) = expected
    assert contract.maximum_query_rows(concurrency) == maximum_q
    assert contract.legal_query_rows(concurrency) == tuple(
        range(1, maximum_q + 1)
    )
    maximum = contract.maximum_payloads(concurrency)
    assert (
        maximum["tp_ar"],
        maximum["indexer"],
        maximum["dcp_query"],
        maximum["dcp_lse"],
        maximum["dcp_output"],
        maximum["vocabulary"],
    ) == (
        tp_ar,
        indexer,
        dcp_query,
        dcp_lse,
        dcp_output,
        vocabulary,
    )


@pytest.mark.parametrize(
    ("concurrency", "expected"),
    enumerate(EXPECTED_FUSED_MAXIMA, start=1),
)
def test_c1_c8_fused_combine_maxima(
    concurrency: int,
    expected: tuple[int, int],
) -> None:
    maximum = contract.maximum_payloads(concurrency)
    assert (
        maximum["fused_combine_0"],
        maximum["fused_combine_1"],
    ) == expected


def test_every_indexer_payload_q1_q40() -> None:
    assert tuple(
        contract.payload_bytes("indexer", query_rows)
        for query_rows in range(1, 41)
    ) == EXPECTED_INDEXER_Q1_Q40


def test_current_capture_matrix_c1_c8() -> None:
    assert tuple(
        contract.capture_query_rows(concurrency)
        for concurrency in range(1, 9)
    ) == EXPECTED_CAPTURE_ROWS


@pytest.mark.parametrize("value", [0, 9, True, 1.5])
def test_concurrency_is_bounded_to_c1_c8(value: object) -> None:
    with pytest.raises(ValueError):
        contract.maximum_query_rows(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [0, 41, True, 1.5])
def test_query_rows_are_bounded_to_q1_q40(value: object) -> None:
    with pytest.raises(ValueError):
        contract.payload_bytes("indexer", value)  # type: ignore[arg-type]
