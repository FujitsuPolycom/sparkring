from __future__ import annotations

import pytest

from . import nccl_graph_allreduce as probe


def test_parse_query_rows_accepts_sorted_unique_positive_values() -> None:
    assert probe.parse_query_rows("8,16,32,64,128") == (8, 16, 32, 64, 128)


@pytest.mark.parametrize("value", ["", "0", "8,8", "16,8", "8,nope"])
def test_parse_query_rows_rejects_invalid_values(value: str) -> None:
    with pytest.raises(ValueError, match="query rows"):
        probe.parse_query_rows(value)


def test_nearest_rank_percentile_returns_observed_samples() -> None:
    samples = [7.0, 1.0, 5.0, 3.0]
    assert probe.nearest_rank(samples, 0.50) == 3.0
    assert probe.nearest_rank(samples, 0.95) == 7.0


@pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.1])
def test_nearest_rank_rejects_invalid_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        probe.nearest_rank([1.0], fraction)
