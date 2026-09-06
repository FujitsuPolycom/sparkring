"""Shared GLM-5.2 TP4 query-width contract.

One scheduler step can mix ordinary decode rows and speculative target rows.
With at most eight active sequences and maximum adaptive MTP depth four, the
total target-query width is bounded by 8 * (4 + 1) = 40. Every custom
collective must accept that complete interval; otherwise a legal mixed batch
can silently fall back to NCCL Socket.
"""

from __future__ import annotations

import os

MAX_CONCURRENT_SEQUENCES = 8
MAX_SPECULATIVE_TOKENS = 4
ABSOLUTE_MAX_QUERY_ROWS = MAX_CONCURRENT_SEQUENCES * (
    MAX_SPECULATIVE_TOKENS + 1
)
_DEFAULT_MAX_QUERY_ROWS = 6
_BASE_CAPTURE_BUCKETS = (1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40)


def _configured_max_query_rows() -> int:
    raw = os.getenv(
        "VLLM_SPARK_MAX_QUERY_ROWS",
        str(_DEFAULT_MAX_QUERY_ROWS),
    )
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(
            "VLLM_SPARK_MAX_QUERY_ROWS must be an integer"
        ) from error
    if not 1 <= value <= ABSOLUTE_MAX_QUERY_ROWS:
        raise ValueError(
            "VLLM_SPARK_MAX_QUERY_ROWS must be in "
            f"[1, {ABSOLUTE_MAX_QUERY_ROWS}]"
        )
    return value


MAX_QUERY_ROWS = _configured_max_query_rows()
SUPPORTED_QUERY_ROWS = frozenset(range(1, MAX_QUERY_ROWS + 1))


def capture_buckets(
    max_query_rows: int,
    *,
    uniform_query_rows: int,
) -> list[int]:
    """Return mixed-batch buckets plus exact FULL-decode widths."""

    if not 1 <= max_query_rows <= ABSOLUTE_MAX_QUERY_ROWS:
        raise ValueError(
            f"max_query_rows must be in [1, {ABSOLUTE_MAX_QUERY_ROWS}]"
        )
    if not 1 <= uniform_query_rows <= max_query_rows:
        raise ValueError("uniform_query_rows must be in [1, max_query_rows]")

    buckets = {value for value in _BASE_CAPTURE_BUCKETS if value <= max_query_rows}
    buckets.add(max_query_rows)
    buckets.update(range(uniform_query_rows, max_query_rows + 1, uniform_query_rows))
    return sorted(buckets)


def capture_sizes(max_num_seqs: int) -> list[int]:
    """Return bounded-padding CUDA-graph coverage for adaptive MTP2/4."""

    if (
        isinstance(max_num_seqs, bool)
        or not isinstance(max_num_seqs, int)
        or not 1 <= max_num_seqs <= MAX_CONCURRENT_SEQUENCES
    ):
        raise ValueError(
            "max_num_seqs must be an integer in "
            f"[1, {MAX_CONCURRENT_SEQUENCES}]"
        )
    maximum_q = max_num_seqs * (MAX_SPECULATIVE_TOKENS + 1)
    if max_num_seqs == 1:
        return [1, 3, 5]
    return capture_buckets(maximum_q, uniform_query_rows=5)
