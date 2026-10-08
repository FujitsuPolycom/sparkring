"""vLLM's rank layout, reproduced without vLLM, against vLLM's own tensor operations."""

from __future__ import annotations

import numpy as np
import pytest

from sparkring_sircl.groups import dcp_groups, nodes_of, tp_groups


def _vllm_dcp_groups(world, tp, dcp, pcp, pp, dp):
    # The operations of vllm/distributed/parallel_state.py initialize_model_parallel.
    all_ranks = np.arange(world).reshape(-1, dp, pp, pcp, tp)
    ranks = all_ranks
    if dcp > 1:
        ranks = ranks.swapaxes(-1, -2)
    return tuple(tuple(int(x) for x in row) for row in ranks.reshape(-1, dcp))


@pytest.mark.parametrize("world,tp,dcp,pcp,pp,dp", [
    (8, 8, 1, 1, 1, 1), (8, 8, 2, 1, 1, 1), (8, 8, 4, 1, 1, 1), (8, 8, 8, 1, 1, 1),
    (8, 4, 2, 1, 1, 2), (8, 4, 4, 2, 1, 1), (8, 2, 2, 2, 2, 1), (6, 6, 3, 1, 1, 1),
    (4, 2, 2, 2, 1, 1),
])
def test_dcp_groups_match_vllm(world, tp, dcp, pcp, pp, dp):
    assert dcp_groups(world, tp=tp, dcp=dcp, pcp=pcp, pp=pp, dp=dp) == _vllm_dcp_groups(
        world, tp, dcp, pcp, pp, dp)


def test_tp_groups_are_contiguous():
    assert tp_groups(8, tp=4) == ((0, 1, 2, 3), (4, 5, 6, 7))
    assert tp_groups(4, tp=2, pp=2) == ((0, 1), (2, 3))
    with pytest.raises(ValueError):
        tp_groups(6, tp=4)


def test_nodes_of():
    assert nodes_of((1, 3), ("a", "b", "c", "d")) == ("b", "d")
    with pytest.raises(ValueError):
        nodes_of((5,), ("a",))
