"""The NCCL communicators a stock serving runtime creates, and how their ranks lie on a ring of Sparks.

Both runtimes derive their process groups from the parallelism widths alone,
and torch gives every group a NCCL communicator of that group's ranks.
libsircl creates each communicator as a child of the deployment's through
``ncclCommSplit``, at the child's first collective.

Under ``LIBSIRCL_RING_ONLY=1`` a process has queue pairs only to its two ring
neighbours, open-chain collectives are off (``e->chain_on = 0`` in
``spark_transport/libsircl/src/engine.c``), and every communicator closes its
ring modulo its own member count ``K`` (``ring_next = order[(i + 1) % K]``).
``roce_set_links`` (``sircl_roce_proxy.c``) refuses setup whenever ``ring_prev``
or ``ring_next`` lacks a connected lane. On a ring of ``W`` Sparks, only
subgroups with ``K in (1, 2, W)`` can close on direct cables:

- ``K == 1`` (``SINGLE``): no wire.
- ``K == 2`` on cable neighbours (``PAIR``): both ``ring_next`` and
  ``ring_prev`` cross the same single cable.
- ``K == W`` (``WHOLE``): closes on the physical wrap-around cable
  ``(W - 1) <-> 0``.

A contiguous proper arc of ``3 <= K < W`` positions (``OPEN_ARC``, such as
``[0, 1, 2, 3]`` on an 8-Spark ring) has direct cables inside the run, but its
closing edge ``placed[-1] -> placed[0]`` crosses multiple hops with no cable
and no relay, so ``LIBSIRCL_RING_ONLY=1`` refuses it just like a ``STRIDED``
group. Relayed mode (default) carries both ``OPEN_ARC`` and ``STRIDED`` groups
over ConnectX hairpin relays, subject to forward-window budgets.

The widths and rank lists here are transcriptions of
SGLang v0.5.21 ``srt/distributed/parallel_state.py`` and
``srt/runtime_context.py``, and of vLLM v0.19.1
``vllm/distributed/parallel_state.py``: the same widths and the same ranks,
in the order each runtime builds them. Nothing here imports either runtime.
"""
from dataclasses import dataclass

# A group's ranks as they lie on the ring.
SINGLE = "single"          # one rank: no wire, so no ring to cross
PAIR = "pair"              # two cable-adjacent ranks: closes a 2-rank ring over one cable
OPEN_ARC = "open arc"      # 3..W-1 contiguous positions: interior cables only, closing edge has no cable
WHOLE = "whole ring"       # every position, in ring order
STRIDED = "strided"        # two or more runs: a member has a neighbour it cannot cable
RING_ONLY_OK = frozenset((SINGLE, PAIR, WHOLE))


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
    """The total Sparks SGLang launches: ``(1 if enable_dp_attention else dp) * tp * pp``.

    With DP attention the replicas divide the tensor-parallel width, so the
    world is ``tp * pp`` and ``dp`` must divide ``tp``; without it SGLang runs
    ``dp`` independent ``tp * pp`` worlds and ``dp`` multiplies.
    """
    return (1 if enable_dp_attention else dp) * tp * pp


def sglang_groups(*, tp, pp=1, dp=1, enable_dp_attention=False, attn_cp=1, moe_ep=1, moe_dp=1):
    """Every process group SGLang's ``initialize_model_parallel`` builds, in its order.

    ``tp`` is ``--tp-size``; ``dp`` is ``--dp-size`` (the attention replicas
    when ``enable_dp_attention``, else independent ``tp * pp`` replicas);
    ``pp`` is ``--pp-size``; ``attn_cp`` is ``--attn-cp-size``; ``moe_ep`` is
    ``--ep-size`` and ``moe_dp`` is ``--moe-dp-size``. Attention widths derive
    as ``attn_dp = dp if enable_dp_attention else 1``, ``attn_tp = tp //
    attn_dp // attn_cp``; the MoE widths as ``moe_tp = tp // moe_ep // moe_dp``.

    Matches SGLang v0.5.21 ``srt/distributed/parallel_state.py``: each
    independent replica has ``world_size = tp * pp``, builds ``tp``,
    ``attn_cp``, ``attention_tp``, ``moe_ep``, ``moe_tp``, ``moe_dp`` inside
    each stage, and builds ``pp`` across the ``pp`` stages with stride ``tp``.
    A width that equals ``tp`` aliases ``_TP``.
    """
    world = sglang_world_size(tp=tp, pp=pp, dp=dp, enable_dp_attention=enable_dp_attention)
    replica_count = 1 if enable_dp_attention else dp
    replica_size = tp * pp
    if world != replica_count * replica_size:
        raise ValueError(f"a world of {world} ranks is not {replica_count} replicas of tp {tp} x pp {pp}")
    attn_dp = dp if enable_dp_attention else 1
    if tp % (attn_dp * attn_cp):
        raise ValueError(f"tp {tp} is not divisible by attn_dp {attn_dp} x attn_cp {attn_cp}")
    attn_tp = tp // attn_dp // attn_cp
    if tp % (moe_ep * moe_dp):
        raise ValueError(f"tp {tp} is not divisible by moe_ep {moe_ep} x moe_dp {moe_dp}")
    moe_tp = tp // moe_ep // moe_dp
    groups = []
    for dp_replica in range(replica_count):
        replica_base = dp_replica * replica_size
        groups.append(Group("world", _span(replica_base, replica_size)))
        for tp_ranks in _blocks(replica_base, tp, pp):
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
        # _PP: stride tp across the pp stages of this replica (parallel_state.py:2933-2941).
        for index in range(tp):
            groups.append(Group("pp", _span(replica_base + index, pp, tp)))
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


def _gaps(placed, world):
    return [(placed[(index + 1) % len(placed)] - placed[index]) % world for index in range(len(placed))]


def _classify(placed, world):
    """Classify sorted ring positions ``placed`` on a ``world``-Spark ring."""
    count = len(placed)
    if count == 1:
        return SINGLE
    if count == world:
        return WHOLE
    gaps = _gaps(placed, world)
    if count == 2:
        return PAIR if 1 in gaps else STRIDED
    return OPEN_ARC if gaps.count(1) == count - 1 else STRIDED


def _ring_only_ok(placed, world):
    """Whether sorted positions ``placed`` close on direct cables under ``LIBSIRCL_RING_ONLY=1``."""
    return _classify(placed, world) in RING_ONLY_OK


def ring_kind(group, positions):
    """How ``group`` lies on the ring: ``SINGLE``, ``PAIR``, ``OPEN_ARC``, ``WHOLE`` or ``STRIDED``.

    ``positions[rank]`` is the ring position of process ``rank``, a permutation
    of ``0 .. len(positions) - 1``. The group's members are read in position
    order, which is the order ``engine.c`` gives a communicator's ring. Only
    ``SINGLE``, ``PAIR`` and ``WHOLE`` (``RING_ONLY_OK``) close on direct
    cables under ``LIBSIRCL_RING_ONLY=1``.
    """
    world = len(positions)
    if sorted(positions) != list(range(world)):
        raise ValueError(f"positions {list(positions)} are not a permutation of 0-{world - 1}")
    if len(set(group.ranks)) != len(group.ranks):
        raise ValueError(f"group {group.name} {list(group.ranks)} names a rank twice")
    placed = sorted(positions[rank] for rank in group.ranks)
    return _classify(placed, world)


def arc_problems(groups, positions):
    """Why these groups cannot start on this ring under ``LIBSIRCL_RING_ONLY=1``; empty when they all can.

    One message per distinct rank set whose communicator does not close on
    direct cables: ``LIBSIRCL_RING_ONLY=1`` disables open-chain collectives
    and closes every communicator modulo its own size, so only ``SINGLE``,
    ``PAIR`` (2 cable-adjacent positions) and ``WHOLE`` can start; both
    ``OPEN_ARC`` (3..W-1 contiguous positions, missing the closing cable) and
    ``STRIDED`` fail setup in ``roce_set_links``.
    """
    world = len(positions)
    problems = []
    for name, ranks in rank_sets(groups):
        kind = ring_kind(Group(name, ranks), positions)
        if kind in RING_ONLY_OK:
            continue
        placed = sorted(positions[rank] for rank in ranks)
        if kind == OPEN_ARC:
            gaps = _gaps(placed, world)
            gap_at = next(index for index, gap in enumerate(gaps) if gap != 1)
            start_pos, end_pos = placed[gap_at], placed[(gap_at + 1) % len(placed)]
            problems.append(
                f"the {name} communicator over ranks {list(ranks)} lies at positions {placed}, an open arc of "
                f"{len(ranks)} on the {world}-Sparks ring; LIBSIRCL_RING_ONLY=1 disables open-chain collectives and "
                f"closes every communicator's ring modulo its size, so position {start_pos} has no direct cable to "
                f"position {end_pos} (use a relayed group)")
        else:
            runs = sum(1 for gap in _gaps(placed, world) if gap != 1)
            problems.append(
                f"the {name} communicator over ranks {list(ranks)} lies at positions {placed}, which is "
                f"{runs} separate runs of the {world}-Sparks ring; LIBSIRCL_RING_ONLY=1 gives a rank queue pairs to "
                "its ring neighbours only, so this child of the deployment's communicator has no route to its next "
                "member (use a relayed group, or a rank order whose communicators close on cables)")
    return problems


def assignment(groups, world):
    """``{rank: position}`` under which every group closes on cables under ``LIBSIRCL_RING_ONLY=1``, or None.

    A subgroup of ``3 <= K < world`` ranks can never close on cables of a
    ``world``-cycle in any rank order, so such shapes return None immediately.
    For shapes whose proper subgroups are all pairs (e.g. ``dp2 x tp2`` or
    ``pp2 x tp2`` at four Sparks, where TP pairs take cables 0-1 / 2-3 and
    DP/PP pairs take 1-2 / 3-0), the search is exhaustive over the ring's
    positions, pruned as each group completes.

    The map is realised through the Spark each rank runs on, not through the
    position list a plan hands to the route tool: that list traces the cable
    ring (consecutive entries are cabled neighbours) and ``tools/site_routes.py``
    numbers each row by its index in it (``"LIBSIRCL_POSITION": str(rank)``), so
    a permuted list would describe a ring over cables that do not exist. The
    installer therefore orders the ranks' Sparks by this map and leaves the
    trace alone.
    """
    world = len(range(world))
    by_rank = [[] for _ in range(world)]
    for group in groups:
        if len(set(group.ranks)) != len(group.ranks):
            raise ValueError(f"group {group.name} {list(group.ranks)} names a rank twice")
        for rank in group.ranks:
            if not 0 <= rank < world:
                raise ValueError(f"group {group.name} names rank {rank}, outside 0-{world - 1}")
            by_rank[rank].append(group)
    # Any proper subgroup with 3 <= K < world cannot form a closed cycle on a world-cycle in any order.
    for group in groups:
        if 3 <= len(group.ranks) < world:
            return None
    placed = {}

    def completes(rank):
        for group in by_rank[rank]:
            if all(member in placed for member in group.ranks):
                if not _ring_only_ok(sorted(placed[member] for member in group.ranks), world):
                    return False
        return True

    def search(rank):
        if rank == world:
            return True
        for position in range(world):
            if position in placed.values():
                continue
            placed[rank] = position
            if completes(rank) and search(rank + 1):
                return True
            del placed[rank]
        return False

    return dict(placed) if search(0) else None

