"""Bandwidth bounds of a ring session's collectives (torch-free).

Three rates bound a collective on a ring of DGX Sparks (measured by the
fabric's operator; ``STATUS.md`` keeps the measurements):

- the host interface: a Spark's NIC moves at most about 24 GB/s from host
  memory to the network (sending, 22.4-24 GB/s measured) and about 26.8 GB/s
  from the network to host memory (receiving), summed over all its RDMA
  functions. Bytes a relay forwards stay inside the relaying NIC and do not
  count against the relaying Spark;
- the cable rate: a cable moves at most about 24 GB/s in each direction over
  both of its lanes.

:func:`flows` lists the point-to-point transfers ``(source rank, destination
rank, bytes)`` a schedule makes for one collective; :func:`loads` adds them up
per rank (bytes leaving and entering its host) and per cable direction,
routing each transfer over its lanes (:func:`sparkring_sircl.routes.derive_routes`,
bytes split evenly over the lanes); :func:`bound_seconds` is the largest load
divided by its rate (sending, receiving or cable), the time below which the
collective cannot finish whatever its pipelining. A caller that models one
host rate for both directions passes it as ``host_gbps`` (``HOST_CAP_GBPS``,
the sending rate, is the lower of the two). Message bytes are the all-reduce message, the
reduce-scatter input of one rank (``W`` chunks) and the all-gather output
(``W`` shards).

Schedules (``SCHEDULES``):

- ``oneshot``: every rank writes the whole message to every peer;
- ``twoshot``: every rank writes chunk ``p`` to rank ``p``, then its reduced
  chunk to every peer (the two-shot all-reduce, also its pieces);
- ``scatter``: every rank writes chunk ``p`` to rank ``p`` (the scatter-op
  reduce-scatter and all-to-all; the tiled and one-shot all-gather, whose
  shards are the chunks);
- ``chain``: the chain all-reduce (half the message reduces along the chain
  and returns, the other half the opposite way), the chain reduce-scatter
  (partial sums travel toward each chunk's owner from both ends) and the
  chain all-gather (each shard travels toward both ends), between neighbors
  in ``order``;
- ``ring``: a ring over ``order`` closed by its last rank's lanes to its
  first (through relays on a path): the reduce-scatter and the all-gather
  move ``W - 1`` chunks to the next rank, the all-reduce both;
- ``swing``: the Swing all-reduce of a power-of-two group
  (``protocol.swing_phases``): ``2 log2(W)`` phases, in each of which every
  rank sends one contiguous range of chunk positions to that phase's peer, at
  ring distance ``|rho(k)|`` (1, 1, 3, 5, ...): half the message, a quarter,
  ... one chunk, then the same sizes in reverse. ``2 (W - 1) / W`` of the
  message leaves every rank, as in the two-shot all-reduce, but toward one
  peer per phase.

The host cap applies both ways: the chain reduce-scatter sends ``W - 1``
chunks from every rank but receives up to ``W + 1`` at a middle rank (partial
sums for the owners on its far side pass through it from both ends), the
mirror of the chain all-gather; a ring moves ``W - 1`` chunks into and out
of every rank.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from . import protocol
from . import routes as routes_mod

HOST_SEND_GBPS = 24.0    # bytes per second (10^9) from one Spark's host memory to the network
HOST_RECV_GBPS = 26.8    # bytes per second (10^9) from the network into one Spark's host memory
HOST_CAP_GBPS = HOST_SEND_GBPS  # one host rate for both directions: the lower of the two
CABLE_GBPS = 24.0        # bytes per second (10^9) per cable direction over both lanes
COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather", "all_to_all")
SCHEDULES = ("oneshot", "twoshot", "scatter", "chain", "ring", "swing")


class BoundError(ValueError):
    """A collective and schedule this model does not describe."""


@dataclasses.dataclass(frozen=True)
class Loads:
    """Bytes leaving and entering every rank's host, and crossing every cable direction."""

    egress: tuple[float, ...]
    ingress: tuple[float, ...]
    cables: dict[tuple[int, int], float]      # (from position, to position) -> bytes

    def times(self, host_gbps: float | None = None, cable_gbps: float = CABLE_GBPS, *,
              send_gbps: float = HOST_SEND_GBPS, recv_gbps: float = HOST_RECV_GBPS) -> dict[str, float]:
        """Seconds the busiest sender, receiver and cable direction need (``host_gbps``, when given, is
        the host rate of both directions)."""
        if host_gbps is not None:
            send_gbps = recv_gbps = host_gbps
        return {"send": max(self.egress, default=0.0) / (send_gbps * 1e9),
                "receive": max(self.ingress, default=0.0) / (recv_gbps * 1e9),
                "cable": max(self.cables.values(), default=0.0) / (cable_gbps * 1e9)}

    def seconds(self, host_gbps: float | None = None, cable_gbps: float = CABLE_GBPS, *,
                send_gbps: float = HOST_SEND_GBPS, recv_gbps: float = HOST_RECV_GBPS) -> float:
        return max(self.times(host_gbps, cable_gbps, send_gbps=send_gbps, recv_gbps=recv_gbps).values())

    def binding(self, host_gbps: float | None = None, cable_gbps: float = CABLE_GBPS, *,
                send_gbps: float = HOST_SEND_GBPS, recv_gbps: float = HOST_RECV_GBPS) -> str:
        """Which rate sets the bound: ``send``, ``receive`` or ``cable``."""
        times = self.times(host_gbps, cable_gbps, send_gbps=send_gbps, recv_gbps=recv_gbps)
        return max(times, key=times.get)

    def limit(self, host_gbps: float | None = None, cable_gbps: float = CABLE_GBPS, *,
              send_gbps: float = HOST_SEND_GBPS, recv_gbps: float = HOST_RECV_GBPS) -> str:
        """Which resource sets the bound: ``host`` (its interface, either direction) or ``cable``."""
        rate = self.binding(host_gbps, cable_gbps, send_gbps=send_gbps, recv_gbps=recv_gbps)
        return "cable" if rate == "cable" else "host"


def flows(collective: str, schedule: str, world: int, nbytes: float,
          order: Sequence[int] | None = None) -> list[tuple[int, int, float]]:
    """The transfers of one collective of ``nbytes`` message bytes under ``schedule``."""
    if collective not in COLLECTIVES or schedule not in SCHEDULES:
        raise BoundError(f"no traffic model for {collective} with the {schedule} schedule")
    if world < 2:
        return []
    peers = [(src, dst) for src in range(world) for dst in range(world) if src != dst]
    part = nbytes / world
    if schedule == "oneshot":
        if collective != "all_reduce":
            raise BoundError("the one-shot schedule is an all-reduce schedule")
        return [(src, dst, float(nbytes)) for src, dst in peers]
    if schedule == "twoshot":
        if collective != "all_reduce":
            raise BoundError("the two-shot schedule is an all-reduce schedule")
        return [(src, dst, 2 * part) for src, dst in peers]
    if schedule == "scatter":
        if collective == "all_reduce":
            raise BoundError("the scatter schedule moves each chunk once; an all-reduce needs two phases")
        return [(src, dst, part) for src, dst in peers]
    if schedule == "swing":
        if collective != "all_reduce":
            raise BoundError("the Swing schedule is an all-reduce schedule")
        try:
            phases = [protocol.swing_phases(world, rank) for rank in range(world)]
        except protocol.ProtocolError as error:
            raise BoundError(str(error)) from None
        return [(rank, phase.peer, part * (phase.end - phase.first))
                for rank in range(world) for phase in phases[rank]]
    if order is None or sorted(order) != list(range(world)):
        raise BoundError("chain and ring schedules need the chain order of every rank")
    if schedule == "ring":
        if collective not in ("all_reduce", "reduce_scatter", "all_gather"):
            raise BoundError("the ring schedule carries all-reduces, reduce-scatters and all-gathers")
        steps = 2 * (world - 1) if collective == "all_reduce" else world - 1
        return [(order[i], order[(i + 1) % world], steps * part) for i in range(world)]
    result: list[tuple[int, int, float]] = []
    for i in range(world - 1):
        right, left = order[i], order[i + 1]          # the link between chain indices i and i + 1
        if collective == "all_reduce":
            # Half A's partials and half B's results rightward; the mirror leftward.
            result += [(right, left, float(nbytes)), (left, right, float(nbytes))]
        elif collective == "reduce_scatter":
            # Rightward: partials for owners i + 1 .. W - 1; leftward: for owners 0 .. i.
            result += [(right, left, (world - 1 - i) * part), (left, right, (i + 1) * part)]
        elif collective == "all_gather":
            # Rightward: shards of chain indices 0 .. i; leftward: of i + 1 .. W - 1.
            result += [(right, left, (i + 1) * part), (left, right, (world - 1 - i) * part)]
        else:
            raise BoundError("the chain schedule carries all-reduces, reduce-scatters and all-gathers")
    return result


def loads(group_routes: routes_mod.GroupRoutes, transfers: Sequence[tuple[int, int, float]]) -> Loads:
    """Per-rank host bytes and per-cable-direction bytes of ``transfers`` on the group's lanes."""
    world = group_routes.world
    egress = [0.0] * world
    ingress = [0.0] * world
    cables: dict[tuple[int, int], float] = {}
    for src, dst, size in transfers:
        egress[src] += size
        ingress[dst] += size
        lanes = group_routes.lanes_to(src, dst)
        for lane in lanes:
            for step in lane.steps:
                key = (step.src, step.dst)
                cables[key] = cables.get(key, 0.0) + size / len(lanes)
    return Loads(tuple(egress), tuple(ingress), cables)


def bound_seconds(layout: routes_mod.Layout, lanes: int, collective: str, schedule: str, nbytes: float,
                  order: Sequence[int] | None = None, *, host_gbps: float | None = None,
                  send_gbps: float = HOST_SEND_GBPS, recv_gbps: float = HOST_RECV_GBPS,
                  cable_gbps: float = CABLE_GBPS) -> float:
    """The time below which one collective of ``nbytes`` cannot finish on ``layout`` under ``schedule``
    (``host_gbps``, when given, is the host rate of both directions)."""
    group_routes = routes_mod.derive_routes(layout, lanes)
    return loads(group_routes, flows(collective, schedule, layout.world, nbytes, order)).seconds(
        host_gbps, cable_gbps, send_gbps=send_gbps, recv_gbps=recv_gbps)


__all__ = ["CABLE_GBPS", "COLLECTIVES", "HOST_CAP_GBPS", "HOST_RECV_GBPS", "HOST_SEND_GBPS", "SCHEDULES", "BoundError",
           "Loads", "bound_seconds", "flows", "loads"]
