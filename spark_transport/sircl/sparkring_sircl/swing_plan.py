"""Swing all-reduce plan and relay load (torch-free).

The Swing all-reduce (De Sensi et al., "Swing: Short-cutting Rings for Higher
Bandwidth Allreduce", NSDI 2024) of a group of ``W = 2^s`` ranks runs ``s``
reduce-scatter steps and then the same steps in reverse as an all-gather. At
step ``k`` rank ``r`` exchanges with ``protocol.swing_peer(W, r, k)``: offsets
+1, -1, +3, -5, ... with the sign flipped for odd ranks. The message is cut
into ``W`` chunk positions; position ``i`` is the pack range
``[floor(i * P / W), floor((i + 1) * P / W))`` of a ``P``-pack message, and
``protocol.swing_chunk_owners`` orders the positions so that every step moves
one contiguous position range to one peer.

:func:`swing_steps` lists, per reduce-scatter step, the peer, the positions
the rank sends (its partial sums of the positions the peer keeps) and the
positions it keeps. After the last step the rank holds the complete sum of its
own position. All-gather step ``k`` sends the kept range of step ``k`` to the
same peer and receives the peer's kept range, which is the range the rank sent
at that step. The network phases of one op are the reduce-scatter steps
``0 .. s-1`` followed by the all-gather steps ``s-1 .. 0``; the progress thread
posts each as one described phase (``protocol.swing_phases`` gives their
descriptors: position range, peer, flag namespace 0 for the reduce-scatter and
1 for the all-gather).

Arithmetic. At every reduce-scatter step a rank adds its partial sums and the
peer's in float32 and rounds once to the dtype. One step adds exactly two
values, so the order inside a step does not matter, and the result depends only
on the schedule. Each position is reduced by one rank and copied to the others,
so every rank holds identical bits; they can differ in the last bits from the
rank-ordered sum of the one-shot and two-shot all-reduce.

Send volume (:func:`phase_send_bytes`). Reduce-scatter step ``k`` sends
``1 / 2^(k+1)`` of the message to its peer and the all-gather sends the same
sizes in reverse: ``2 (W - 1) / W`` of the message leaves every rank and as
much enters it, the volume of the two-shot all-reduce, but each phase goes to
one peer at ring distance ``|rho(k)|``. On the ring of eight the phases send
1/2, 1/4, 1/8, 1/8, 1/4 and 1/2 of the message to peers at distances 1, 1, 3,
3, 1 and 1. Divided by a Spark's NIC host rate this is each phase's least
time through the host interface; ``sparkring_sircl.bounds`` (schedule
``swing``) bounds the whole op by its busiest host interface or cable
direction: 1.75 of the message through every host interface on the ring of
eight.

Phase-serial bound (:func:`phase_seconds`). The kernel runs the phases one
after another, and each phase's transfers alone have a least time of their
own: the larger of the busiest host interface and the busiest cable direction
in that phase. Their sum bounds an op of the kernel from below and is never
less than the whole-op bound. On the ring of eight the two distance-3 phases
put two transfers of 1/8 of the message on every other cable direction, so
they take at least 1/4 of the message at the cable rate, and the six phases
add up to 2 of the message at 24 GB/s: the two-shot all-reduce's bound.

Relay load (:func:`relay_safe_message_bytes`). Without forward windows a
relayed lane posts its whole stripe at once, so the bytes a relay hairpin queue
may hold during one op are bounded by the sum, over every phase of the op, of
the stripes of the lanes that cross the queue. The function returns the largest
message whose bound stays within ``share`` of the queue, or None when no lane of
any Swing phase crosses a relay.

Status: implemented; CPU tests in ``tests/test_swing_plan.py``.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from . import bounds as bounds_mod
from . import protocol as proto
from . import routes as routes_mod
from .protocol import PACK_BYTES


@dataclasses.dataclass(frozen=True)
class SwingStep:
    """One reduce-scatter step of one rank: peer and chunk-position ranges ``[first, end)``."""

    step: int
    peer: int
    send: tuple[int, int]
    keep: tuple[int, int]


def available(world: int, multi_phase: bool = True) -> bool:
    """Swing runs on power-of-two groups of 2 to 16 ranks whose session can post described ops."""
    world = int(world)
    return bool(multi_phase) and 2 <= world <= proto.MAX_WORLD and world & (world - 1) == 0


def swing_steps(world: int, rank: int) -> tuple[SwingStep, ...]:
    """The reduce-scatter steps of ``rank`` (raises ``protocol.ProtocolError`` for other world sizes)."""
    phases = proto.swing_phases(int(world), int(rank))
    steps = len(phases) // 2
    result = []
    for step in range(steps):
        sent = phases[step]
        kept = phases[2 * steps - 1 - step]
        if sent.peer != kept.peer or sent.namespace != 0 or kept.namespace != 1:
            raise proto.ProtocolError(f"Swing phases of rank {rank} of {world} do not pair at step {step}")
        result.append(SwingStep(step, sent.peer, (sent.first, sent.end), (kept.first, kept.end)))
    return tuple(result)


def phase_count(world: int) -> int:
    return 2 * (int(world).bit_length() - 1)


def descriptor_words(world: int, rank: int) -> tuple[int, ...]:
    """The 32-bit descriptor of every network phase of ``rank``, phase 0 first."""
    return tuple(phase.word(int(world)) for phase in proto.swing_phases(int(world), int(rank)))


def phase_range(world: int, rank: int, phase: int) -> tuple[int, int, int]:
    """``(peer, first position, end position)`` that ``rank`` sends in network phase ``phase``."""
    steps = swing_steps(world, rank)
    count = len(steps)
    if not 0 <= phase < 2 * count:
        raise proto.ProtocolError(f"a Swing op of {world} ranks has phases 0-{2 * count - 1}, got {phase}")
    if phase < count:
        step = steps[phase]
        return step.peer, step.send[0], step.send[1]
    step = steps[2 * count - 1 - phase]
    return step.peer, step.keep[0], step.keep[1]


def phase_send_bytes(world: int, nbytes: float) -> tuple[float, ...]:
    """Bytes one rank sends in each network phase of a Swing all-reduce of ``nbytes`` (every rank alike)."""
    sends = [[nbytes * (phase.end - phase.first) / world for phase in proto.swing_phases(world, rank)]
             for rank in range(world)]
    return tuple(max(column) for column in zip(*sends))


def phase_seconds(layout: routes_mod.Layout, lanes: int, nbytes: float, *,
                  host_gbps: float = bounds_mod.HOST_CAP_GBPS,
                  cable_gbps: float = bounds_mod.CABLE_GBPS) -> tuple[float, ...]:
    """Least time of every network phase of a Swing all-reduce of ``nbytes`` on ``layout``, in seconds.

    Each phase's transfers alone, on the layout's lanes (``bounds.loads``): the larger of the busiest host
    interface and the busiest cable direction. Raises ``protocol.ProtocolError`` without a Swing schedule.
    """
    world = layout.world
    group_routes = routes_mod.derive_routes(layout, int(lanes))
    phases = [proto.swing_phases(world, rank) for rank in range(world)]
    result = []
    for phase in range(len(phases[0])):
        transfers = [(rank, phases[rank][phase].peer,
                      nbytes * (phases[rank][phase].end - phases[rank][phase].first) / world)
                     for rank in range(world)]
        result.append(bounds_mod.loads(group_routes, transfers).seconds(host_gbps, cable_gbps))
    return tuple(result)


def phase_distances(world: int) -> tuple[int, ...]:
    """Ring distance between the two ranks of every network phase (both phases of step ``k``: ``|rho(k)|``)."""
    steps = phase_count(world) // 2
    forward = [min(abs(proto.swing_offset(step)) % world, world - abs(proto.swing_offset(step)) % world)
               for step in range(steps)]
    return tuple(forward + forward[::-1])


def position_packs(packs: int, world: int, first: int, end: int) -> tuple[int, int]:
    """Pack range ``[lo, hi)`` of chunk positions ``[first, end)`` of a ``packs``-pack message."""
    return int(first) * int(packs) // int(world), int(end) * int(packs) // int(world)


def relay_safe_message_bytes(
    layout: routes_mod.Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    lane_count: int,
    *,
    queue_bytes: int = routes_mod.DEFAULT_HAIRPIN_QUEUE,
    share: float = routes_mod.RELAY_QUEUE_SHARE,
    roles: Mapping[str, routes_mod.Role] | None = None,
) -> int | None:
    """Largest Swing message (bytes, a multiple of 16) that keeps every relay queue within ``share``.

    For a message of ``M`` bytes, a lane that carries positions ``[first,
    end)`` of a phase posts at most ``(end - first) * M / (W * L) + 32`` bytes
    (chunk boundaries and the stripe remainder add at most two packs). The
    bound of a queue is the sum over every phase and every lane through the
    queue; the result is the largest ``M`` whose bound is at most ``share *
    queue_bytes`` for every queue. None when no lane of any phase crosses a
    relay.
    """
    world = layout.world
    if not available(world):
        raise proto.ProtocolError(f"the Swing all-reduce needs a power-of-two group, got {world} ranks")
    queues = routes_mod.relay_queues(layout, route_maps, roles=roles)
    plans = [swing_steps(world, rank) for rank in range(world)]
    lanes = int(lane_count)
    budget = float(share) * int(queue_bytes)
    limit: float | None = None
    for members in queues.values():
        fraction = 0.0
        slack = 0
        for rank, peer, _lane in members:
            for step in plans[rank]:
                if step.peer != peer:
                    continue
                for first, end in (step.send, step.keep):
                    if end > first:
                        fraction += (end - first) / (world * lanes)
                        slack += 2 * PACK_BYTES
        if fraction == 0.0:
            continue
        bound = (budget - slack) / fraction
        limit = bound if limit is None else min(limit, bound)
    if limit is None:
        return None
    return max(PACK_BYTES, int(limit) // PACK_BYTES * PACK_BYTES)


__all__ = ["SwingStep", "available", "descriptor_words", "phase_count", "phase_distances", "phase_range",
           "phase_seconds", "phase_send_bytes", "position_packs", "relay_safe_message_bytes", "swing_steps"]
