"""Separate CUDA-graph bucket contracts for decode and prefill.

The native TP4 query transport remains bounded at Q40.  PIECEWISE prefill
graphs may pad larger observed query shapes into a small, separately attested
bucket set through Q512; those buckets must never become FULL speculative
decode widths.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


DECODE_CAPTURE_ENV = "VLLM_SPARK_DECODE_CAPTURE_SIZES"
FULL_DECODE_CAPTURE_ENV = "VLLM_SPARK_FULL_DECODE_CAPTURE_SIZES"
PREFILL_PIECEWISE_CAPTURE_ENV = (
    "VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES"
)
COMBINED_CAPTURE_ENV = "VLLM_SPARK_GRAPH_CAPTURE_SIZES"

MAX_FULL_DECODE_QUERY_ROWS = 40
MAX_PREFILL_PIECEWISE_QUERY_ROWS = 512
MAX_PREFILL_BUCKET_COUNT = 16
MAX_OBSERVED_PREFILL_PADDING_ROWS = 32

# Existing C8 adaptive-MTP4 graph plan. Only its Q5 multiples are eligible
# for FULL speculative decode; the other values remain decode padding buckets.
DECODE_CAPTURE_BUCKETS = (
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
FULL_SPECULATIVE_DECODE_BUCKETS = tuple(
    range(5, MAX_FULL_DECODE_QUERY_ROWS + 1, 5)
)

OBSERVED_PREFILL_QUERY_ROWS = (48, 69, 72, 143, 210, 279, 348, 417, 486)
DEFAULT_PREFILL_PIECEWISE_BUCKETS = (
    48,
    72,
    144,
    224,
    288,
    352,
    432,
    512,
)


@dataclass(frozen=True)
class CUDAGraphBucketContract:
    decode: tuple[int, ...]
    full_speculative_decode: tuple[int, ...]
    prefill_piecewise: tuple[int, ...]
    combined: tuple[int, ...]


def _strict_ordered_buckets(
    buckets: Sequence[int], *, field: str
) -> tuple[int, ...]:
    values = tuple(buckets)
    if (
        not values
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in values
        )
        or tuple(sorted(set(values))) != values
    ):
        raise ValueError(
            f"{field} must be nonempty, strictly increasing unique integers"
        )
    return values


def validate_full_speculative_decode_buckets(
    buckets: Sequence[int],
) -> tuple[int, ...]:
    values = _strict_ordered_buckets(
        buckets, field="FULL speculative decode buckets"
    )
    if values != FULL_SPECULATIVE_DECODE_BUCKETS:
        raise ValueError(
            "FULL speculative decode buckets must be Q5 multiples through "
            f"Q{MAX_FULL_DECODE_QUERY_ROWS}"
        )
    return values


def prefill_padding_plan(
    buckets: Sequence[int],
) -> dict[int, int]:
    values = tuple(buckets)
    plan: dict[int, int] = {}
    for query_rows in OBSERVED_PREFILL_QUERY_ROWS:
        padded = next(
            (bucket for bucket in values if bucket >= query_rows),
            None,
        )
        if padded is None:
            raise ValueError(
                f"PIECEWISE prefill buckets do not cover Q{query_rows}"
            )
        plan[query_rows] = padded
    return plan


def maximum_observed_padding(buckets: Sequence[int]) -> int:
    return max(
        padded - query_rows
        for query_rows, padded in prefill_padding_plan(buckets).items()
    )


def validate_prefill_piecewise_buckets(
    buckets: Sequence[int],
) -> tuple[int, ...]:
    values = _strict_ordered_buckets(
        buckets, field="PIECEWISE prefill buckets"
    )
    if len(values) > MAX_PREFILL_BUCKET_COUNT:
        raise ValueError(
            "PIECEWISE prefill bucket count exceeds "
            f"{MAX_PREFILL_BUCKET_COUNT}"
        )
    if values[0] <= MAX_FULL_DECODE_QUERY_ROWS:
        raise ValueError(
            "PIECEWISE prefill buckets must remain above the Q40 decode "
            "contract"
        )
    if values[-1] != MAX_PREFILL_PIECEWISE_QUERY_ROWS:
        raise ValueError(
            "PIECEWISE prefill buckets must terminate exactly at Q512"
        )
    if any(value > MAX_PREFILL_PIECEWISE_QUERY_ROWS for value in values):
        raise ValueError("PIECEWISE prefill buckets may not exceed Q512")
    padding = maximum_observed_padding(values)
    if padding > MAX_OBSERVED_PREFILL_PADDING_ROWS:
        raise ValueError(
            "PIECEWISE prefill buckets exceed observed-shape padding bound: "
            f"{padding} > {MAX_OBSERVED_PREFILL_PADDING_ROWS}"
        )
    return values


def combined_capture_buckets(
    prefill_piecewise: Sequence[int],
) -> tuple[int, ...]:
    prefill = validate_prefill_piecewise_buckets(prefill_piecewise)
    return DECODE_CAPTURE_BUCKETS + prefill


def format_buckets(buckets: Sequence[int]) -> str:
    return ",".join(str(value) for value in buckets)


def parse_bucket_csv(raw: object, *, field: str) -> tuple[int, ...]:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{field} must be an explicit canonical CSV list")
    try:
        values = tuple(int(item) for item in raw.split(","))
    except ValueError as error:
        raise ValueError(
            f"{field} must be an explicit canonical CSV list"
        ) from error
    if format_buckets(values) != raw:
        raise ValueError(f"{field} must be an explicit canonical CSV list")
    return values


def contract_from_environment(
    environment: Mapping[str, str],
) -> CUDAGraphBucketContract:
    decode = parse_bucket_csv(
        environment.get(DECODE_CAPTURE_ENV),
        field=DECODE_CAPTURE_ENV,
    )
    if decode != DECODE_CAPTURE_BUCKETS:
        raise ValueError(
            f"{DECODE_CAPTURE_ENV} must preserve the current 18 decode "
            "buckets"
        )
    full = validate_full_speculative_decode_buckets(
        parse_bucket_csv(
            environment.get(FULL_DECODE_CAPTURE_ENV),
            field=FULL_DECODE_CAPTURE_ENV,
        )
    )
    prefill = validate_prefill_piecewise_buckets(
        parse_bucket_csv(
            environment.get(PREFILL_PIECEWISE_CAPTURE_ENV),
            field=PREFILL_PIECEWISE_CAPTURE_ENV,
        )
    )
    combined = parse_bucket_csv(
        environment.get(COMBINED_CAPTURE_ENV),
        field=COMBINED_CAPTURE_ENV,
    )
    expected_combined = combined_capture_buckets(prefill)
    if combined != expected_combined:
        raise ValueError(
            f"{COMBINED_CAPTURE_ENV} does not equal decode + PIECEWISE "
            "prefill buckets"
        )
    return CUDAGraphBucketContract(
        decode=decode,
        full_speculative_decode=full,
        prefill_piecewise=prefill,
        combined=combined,
    )
