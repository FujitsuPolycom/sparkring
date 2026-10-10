"""vLLM's rank layout for tensor-parallel and decode-context-parallel groups.

vLLM numbers global ranks in the order ExternalDP x DP x PP x PCP x TP and
builds its groups from that layout (``vllm/distributed/parallel_state.py``,
``initialize_model_parallel``). Tensor-parallel groups are contiguous rank
blocks. Decode-context-parallel (DCP) groups span the prefill-context-parallel
dimension first and then tensor-parallel ranks: with PCP size 1 they are
contiguous blocks of ``dcp`` ranks inside each TP group.

These functions reproduce that layout without importing vLLM so that route
tables for every group can be planned and checked offline. The vLLM adapter
uses the ranks vLLM actually passes to each communicator; tests compare both.
Model constraints such as attention heads divisible by the TP size belong to
vLLM and are not checked here.
"""

from __future__ import annotations

from collections.abc import Sequence


def _layout(
    world: int, *, tp: int, pcp: int, pp: int, dp: int
) -> list[list[list[list[list[int]]]]]:
    if min(world, tp, pcp, pp, dp) < 1:
        raise ValueError("group sizes must be positive")
    inner = dp * pp * pcp * tp
    if world % inner:
        raise ValueError(f"world size {world} is not a multiple of DP*PP*PCP*TP = {inner}")
    outer = world // inner
    ranks = iter(range(world))
    return [
        [[[[next(ranks) for _ in range(tp)] for _ in range(pcp)] for _ in range(pp)]
         for _ in range(dp)]
        for _ in range(outer)
    ]


def tp_groups(world: int, *, tp: int, pcp: int = 1, pp: int = 1, dp: int = 1) -> tuple[tuple[int, ...], ...]:
    """Global ranks of every tensor-parallel group."""
    layout = _layout(world, tp=tp, pcp=pcp, pp=pp, dp=dp)
    return tuple(
        tuple(group)
        for external in layout for data in external for stage in data for group in stage
    )


def dcp_groups(
    world: int, *, tp: int, dcp: int, pcp: int = 1, pp: int = 1, dp: int = 1
) -> tuple[tuple[int, ...], ...]:
    """Global ranks of every decode-context-parallel group, as vLLM forms them."""
    layout = _layout(world, tp=tp, pcp=pcp, pp=pp, dp=dp)
    if dcp < 1:
        raise ValueError("dcp must be positive")
    flat: list[int] = []
    for external in layout:
        for data in external:
            for stage in data:
                if dcp > 1:
                    # transpose(-1, -2): iterate TP ranks, then PCP ranks.
                    for t in range(tp):
                        for c in range(pcp):
                            flat.append(stage[c][t])
                else:
                    for group in stage:
                        flat.extend(group)
    if len(flat) % dcp:
        raise ValueError(f"{len(flat)} ranks do not divide into DCP groups of {dcp}")
    return tuple(tuple(flat[index:index + dcp]) for index in range(0, len(flat), dcp))


def nodes_of(ranks: Sequence[int], rank_nodes: Sequence[str]) -> tuple[str, ...]:
    """Topology node names of ``ranks`` given each global rank's node."""
    try:
        return tuple(rank_nodes[rank] for rank in ranks)
    except IndexError:
        raise ValueError(
            f"ranks {tuple(ranks)} exceed the {len(rank_nodes)} configured rank nodes"
        ) from None
