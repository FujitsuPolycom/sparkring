"""The NCCL communicators a stock serving runtime creates, and how their ranks lie on a ring of Sparks.

Both runtimes derive their process groups from the parallelism widths alone,
and torch gives every group a NCCL communicator of that group's ranks.
libsircl creates each communicator as a child of the deployment's through
``ncclCommSplit``, at the child's first collective.

Under ``LIBSIRCL_RING_ONLY=1`` a process has queue pairs only to its two ring
neighbours, so a child communicator can start only when its ranks occupy one
contiguous arc of the ring, or the whole ring. A child's ring neighbours are
the members adjacent to it in the group's own position order
(``spark_transport/libsircl/src/engine.c``, "the chain order"), and
``tools/site_routes.py --ring-only`` prints a route map naming only the
deployment ring's neighbours: a rank whose group neighbour lies beyond its
cable neighbour has no route entry, and the child's setup fails before any
collective runs.

The widths and rank lists here are transcriptions of
SGLang v0.5.21 ``srt/distributed/parallel_state.py`` and
``srt/runtime_context.py``, and of vLLM v0.19.1
``vllm/distributed/parallel_state.py``: the same widths and the same ranks,
in the order each runtime builds them. Nothing here imports either runtime.
"""
from dataclasses import dataclass

# A group's ranks as they lie on the ring.
SINGLE = "single"          # one rank: no wire, so no ring to cross
ARC = "arc"                # a contiguous run of ring positions
WHOLE = "whole ring"       # every position, in ring order
STRIDED = "strided"        # two or more runs: a member has a neighbour it cannot cable


@dataclass(frozen=True)
class Group:
    """One process group of a runtime: its name there, and its ranks in that group's order."""
    name: str
    ranks: tuple


def rank_sets(groups):
    """``[(names, ranks)]`` of the distinct rank sets, names joined.

    A runtime builds one communicator per group name even where two names
    cover the same ranks, so this deduplicates for the ring question only:
    equal rank sets have one ring verdict. Count communicators from the group
    list itself, not from here.
    """
    order, names = [], {}
    for group in groups:
        if group.ranks not in names:
            order.append(group.ranks)
            names[group.ranks] = []
        if group.name not in names[group.ranks]:
            names[group.ranks].append(group.name)
    return [("/".join(names[ranks]), ranks) for ranks in order]


def _span(base, count, stride=1):
    return tuple(range(base, base + count * stride, stride))


def _blocks(base, width, count):
    return [_span(base + index * width, width) for index in range(count)]


def sglang_world_size(*, tp, pp=1, dp=1, enable_dp_attention=False):
    """The world SGLang launches: ``(1 if enable_dp_attention else dp) * tp * pp``.

    With DP attention the replicas divide the tensor-parallel width, so the
    world is ``tp * pp`` and ``dp`` must divide ``tp``; without it every
    replica is a world of its own size and ``dp`` multiplies.
    """
    return (1 if enable_dp_attention else dp) * tp * pp


def sglang_groups(*, tp, pp=1, dp=1, enable_dp_attention=False, attn_cp=1, moe_ep=1, moe_dp=1):
    """Every process group SGLang's ``initialize_model_parallel`` builds, in its order.

    ``tp`` is ``--tp-size``; ``dp`` is ``--dp-size`` (the attention replicas
    when ``enable_dp_attention``, else whole replicas); ``attn_cp`` is
    ``--attn-cp-size``; ``moe_ep`` is ``--ep-size`` and ``moe_dp`` is
    ``--moe-dp-size``. Attention widths derive as
    ``attn_dp = dp if enable_dp_attention else 1``, ``attn_tp = tp // attn_dp
    // attn_cp``; the MoE widths as ``moe_tp = tp // moe_ep // moe_dp``.

    A width that equals ``tp`` aliases the tensor-parallel group, which is the
    same communicator under another name; a width of 1 builds one-rank groups,
    which cross no wire and are left out.
    """
    world = sglang_world_size(tp=tp, pp=pp, dp=dp, enable_dp_attention=enable_dp_attention)
    if world % tp:
        raise ValueError(f"a world of {world} ranks is not a multiple of tp {tp}")
    attn_dp = dp if enable_dp_attention else 1
    if tp % (attn_dp * attn_cp):
        raise ValueError(f"tp {tp} is not divisible by attn_dp {attn_dp} x attn_cp {attn_cp}")
    attn_tp = tp // attn_dp // attn_cp
    if tp % (moe_ep * moe_dp):
        raise ValueError(f"tp {tp} is not divisible by moe_ep {moe_ep} x moe_dp {moe_dp}")
    moe_tp = tp // moe_ep // moe_dp
    groups = [Group("world", tuple(range(world)))]
    for stage in _blocks(0, tp, world // tp):
        tp_ranks = stage
        groups.append(Group("tp", tp_ranks))
        # _ATTN_CP: one group per (attention replica, tensor index) with stride attn_tp; attn_cp == tp is _TP.
        attn_cp_groups = ([tp_ranks] if attn_cp == tp else
                          [_span(tp_ranks[0] + replica * attn_tp * attn_cp + index, attn_cp, attn_tp)
                           for replica in range(attn_dp) for index in range(attn_tp)] if attn_cp > 1 else [])
        for ranks in attn_cp_groups:
            groups.append(Group("attn_cp", ranks))
        # _ATTN_TP: blocks of attn_tp, one per (context, replica) pair of a stage; attn_tp == tp is _TP.
        if attn_tp == tp:
            groups.append(Group("attention_tp", tp_ranks))
        else:
            for index in range(attn_cp * attn_dp):
                groups.append(Group("attention_tp", _span(tp_ranks[0] + index * attn_tp, attn_tp)))
        # _MOE_EP: strided by moe_tp within each replica; moe_ep == tp is _TP.
        if moe_ep == tp:
            groups.append(Group("moe_ep", tp_ranks))
        else:
            for replica in range(moe_dp):
                for index in range(moe_tp):
                    base = tp_ranks[0] + replica * moe_ep * moe_tp + index
                    groups.append(Group("moe_ep", _span(base, moe_ep, moe_tp)))
        # _MOE_TP: blocks of moe_tp; moe_tp == tp is _TP.
        if moe_tp == tp:
            groups.append(Group("moe_tp", tp_ranks))
        else:
            for index in range(moe_ep * moe_dp):
                groups.append(Group("moe_tp", _span(tp_ranks[0] + index * moe_tp, moe_tp)))
        # _MOE_DP: strided by moe_tp * moe_ep; moe_dp == tp is _TP, and moe_dp below attn_cp is _ATTN_CP.
        if moe_dp == tp:
            groups.append(Group("moe_dp", tp_ranks))
        elif attn_cp > moe_dp:
            for ranks in attn_cp_groups:
                groups.append(Group("moe_dp", ranks))
        else:
            for index in range(moe_tp * moe_ep):
                groups.append(Group("moe_dp", _span(tp_ranks[0] + index, moe_dp, moe_tp * moe_ep)))
    return groups


def vllm_world_size(*, tp, pp=1, dp=1, pcp=1):
    """The world vLLM launches: ``dp * pp * pcp * tp``, data parallel outermost."""
    return dp * pp * pcp * tp


def vllm_groups(*, tp, pp=1, dp=1, pcp=1, dcp=1, enable_expert_parallel=False):
    """Every process group vLLM's ``initialize_model_parallel`` builds, in its order.

    vLLM lays its ranks out as ``(dp, pp, pcp, tp)`` with ``tp`` fastest, then
    builds each group by transposing one dimension to the end
    (``vllm/distributed/parallel_state.py``): the tensor-parallel and decode
    context-parallel groups are contiguous, the pipeline, prefill context and
    data-parallel groups are strided, and the expert-parallel group is one
    contiguous run of ``dp * pcp * tp`` ranks per pipeline stage. ``dcp``
    subdivides a tensor-parallel group. The expert-parallel group exists only
    for a mixture-of-experts model.
    """
    world = vllm_world_size(tp=tp, pp=pp, dp=dp, pcp=pcp)
    if tp % dcp:
        raise ValueError(f"tp {tp} is not divisible by dcp {dcp}")

    def rank(data, pipeline, context, tensor):
        return ((data * pp + pipeline) * pcp + context) * tp + tensor

    groups = [Group("world", tuple(range(world)))]
    for data in range(dp):
        for pipeline in range(pp):
            for context in range(pcp):
                stage = tuple(rank(data, pipeline, context, tensor) for tensor in range(tp))
                groups.append(Group("tp", stage))
                for index in range(tp // dcp):
                    groups.append(Group("dcp", stage[index * dcp:(index + 1) * dcp]))
    for data in range(dp):
        for pipeline in range(pp):
            for tensor in range(tp):
                groups.append(Group("pcp", tuple(rank(data, pipeline, context, tensor) for context in range(pcp))))
    for data in range(dp):
        for context in range(pcp):
            for tensor in range(tp):
                groups.append(Group("pp", tuple(rank(data, pipeline, context, tensor) for pipeline in range(pp))))
    for pipeline in range(pp):
        for context in range(pcp):
            for tensor in range(tp):
                groups.append(Group("dp", tuple(rank(data, pipeline, context, tensor) for data in range(dp))))
        if enable_expert_parallel:
            groups.append(Group("ep", tuple(rank(data, pipeline, context, tensor)
                                             for data in range(dp) for context in range(pcp) for tensor in range(tp))))
    return groups


def _strided(placed, world):
    """Whether these sorted positions, one arc's worth or not, lie in more than one run of the ring."""
    if len(placed) == 1 or len(placed) == world:
        return False
    gaps = [(placed[(index + 1) % len(placed)] - placed[index]) % world for index in range(len(placed))]
    return gaps.count(1) != len(placed) - 1


def ring_kind(group, positions):
    """How ``group`` lies on the ring: ``SINGLE``, ``ARC``, ``WHOLE`` or ``STRIDED``.

    ``positions[rank]`` is the ring position of process ``rank``, a permutation
    of ``0 .. len(positions) - 1``. The group's members are read in position
    order, which is the order ``engine.c`` gives a communicator's ring.
    """
    world = len(positions)
    if sorted(positions) != list(range(world)):
        raise ValueError(f"positions {list(positions)} are not a permutation of 0-{world - 1}")
    if len(set(group.ranks)) != len(group.ranks):
        raise ValueError(f"group {group.name} {list(group.ranks)} names a rank twice")
    placed = sorted(positions[rank] for rank in group.ranks)
    if len(placed) == 1:
        return SINGLE
    if len(placed) == world:
        return WHOLE
    return STRIDED if _strided(placed, world) else ARC


def arc_problems(groups, positions):
    """Why these groups cannot start on this ring under ``LIBSIRCL_RING_ONLY=1``; empty when they all can.

    One message per distinct rank set that is not one arc: the child's ring
    neighbours are read from its own members, and the ring-only route map
    names only the deployment ring's neighbours, so the child is refused at
    setup rather than left to fail a collective.
    """
    problems = []
    for name, ranks in rank_sets(groups):
        if ring_kind(Group(name, ranks), positions) != STRIDED:
            continue
        placed = sorted(positions[rank] for rank in ranks)
        problems.append(f"the {name} communicator over ranks {list(ranks)} lies at positions {placed}, which is "
                        f"{len(ranks)} separate runs of the {len(positions)}-Sparks ring; LIBSIRCL_RING_ONLY=1 gives "
                        "a rank queue pairs to its ring neighbours only, so this child of the deployment's "
                        "communicator has no route to its next member (use a relayed group, or a rank order whose "
                        "communicators are arcs)")
    return problems


def assignment(groups, world):
    """``{rank: position}`` under which every group is one arc, or None when no order is.

    The installer chooses each process's ring position (``LIBSIRCL_POSITION``),
    so a group list that breaks on the ranks' own order may still fit: at four
    Sparks ``dp2 x tp2`` fits, with the tensor-parallel pairs on cables 0-1 and
    2-3 and the data-parallel pairs on 1-2 and 3-0. The search is exhaustive
    over the ring's positions, pruned as each group completes: it is exact,
    and eight positions are at most 40,320 orders.
    """
    world = len(range(world))
    by_rank = [[] for _ in range(world)]
    for group in groups:
        for rank in group.ranks:
            if not 0 <= rank < world:
                raise ValueError(f"group {group.name} names rank {rank}, outside 0-{world - 1}")
            by_rank[rank].append(group)
    placed = {}

    def completes():
        for group in groups:
            if all(rank in placed for rank in group.ranks):
                if _strided(sorted(placed[rank] for rank in group.ranks), world):
                    return False
        return True

    def search(rank):
        if rank == world:
            return True
        for position in range(world):
            if position in placed.values():
                continue
            placed[rank] = position
            if completes() and search(rank + 1):
                return True
            del placed[rank]
        return False

    return dict(placed) if search(0) else None
