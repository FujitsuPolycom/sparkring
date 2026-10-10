"""Forward windows of point-to-point lanes within the relay hairpin rule (torch-free).

A relay forwards a lane's frames through one hairpin queue per (relay Spark,
function class, egress port), and the queue cannot pause its sender. SIRCL
keeps the bytes of every lane through one queue within ``RELAY_QUEUE_SHARE``
(75 %) of its size (``SIRCL_HAIRPIN_QUEUE_BYTES``, 512 KiB). Collective
sessions take their share first; point-to-point lanes share what is left.

:func:`session_reservation` gives the bytes a collective session's relayed
lanes may hold in each queue: its forward windows (``routes.forward_windows``
with the session's settings), or, where a ``ring`` schedule is named, the
larger of those and its ring window (``routes.ring_window``); a session with
forward windows off reserves the whole share of every queue it uses.

:func:`allocate` gives every relayed point-to-point lane its window, level by
level (PP groups before the channels of groups with a collective session):
each queue's remainder divided evenly among the level's lanes through it; a
lane's window is the smallest share over its queues, capped at the largest
window and rounded down to whole chunks. A lane left with less than one chunk
has no window; its channel is unavailable, and :class:`Allocation` names the
queue, its reservations and its lanes.

Queues are keyed by physical position (``(relay position, secondary
function, egress port)``, as :func:`routes.relay_queues` keys them), so lanes
of groups on different layouts of one ring share a key when they share a
queue. Every rank of a vLLM instance computes the same allocation from
inputs every rank shares (the adapter's ``p2p_plan``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence

from .. import routes as routes_mod

QueueKey = tuple[int, bool, int]
LaneKey = tuple[str, int, int, int]          # (group name, rank, peer, lane)


class BudgetError(ValueError):
    """Point-to-point lanes that the relay rule leaves without a window."""


@dataclasses.dataclass(frozen=True)
class LaneSet:
    """One group's lanes: its layout, every rank's full route map and the ordered pairs that carry traffic."""

    name: str
    layout: routes_mod.Layout
    route_maps: tuple[Mapping[int, tuple[str, ...]], ...]
    pairs: frozenset[tuple[int, int]]

    @classmethod
    def of(cls, name: str, layout: routes_mod.Layout, lanes: int = 2,
           pairs: Iterable[tuple[int, int]] | None = None) -> "LaneSet":
        derived = routes_mod.derive_routes(layout, lanes)
        maps = tuple(derived.route_map(rank) for rank in range(layout.world))
        every = {(a, b) for a in range(layout.world) for b in range(layout.world) if a != b}
        return cls(name, layout, maps, frozenset(every if pairs is None else
                                                 {(int(a), int(b)) for a, b in pairs}))

    def filtered_maps(self) -> list[dict[int, tuple[str, ...]]]:
        return [{peer: devices for peer, devices in routes.items() if (rank, peer) in self.pairs}
                for rank, routes in enumerate(self.route_maps)]

    def queues(self) -> dict[QueueKey, list[tuple[int, int, int]]]:
        """Every relay queue the set's pairs cross, with the (rank, peer, lane) of each lane through it."""
        return routes_mod.relay_queues(self.layout, self.filtered_maps())


@dataclasses.dataclass(frozen=True)
class SessionSettings:
    """The settings of a collective session that size its relayed lanes' share."""

    forward_window: int = routes_mod.DEFAULT_FORWARD_WINDOW
    forward_chunk: int = routes_mod.DEFAULT_FORWARD_CHUNK
    ring_schedule: bool = False          # a large, gather or scatter schedule is ``ring``

    @classmethod
    def from_env(cls, environ: Mapping[str, str]) -> "SessionSettings":
        def number(name: str, default: int) -> int:
            raw = environ.get(name, "").strip()
            return int(raw, 0) if raw else default

        ring = any(environ.get(name, "").strip() == "ring"
                   for name in ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE"))
        return cls(number("SIRCL_FORWARD_WINDOW_BYTES", routes_mod.DEFAULT_FORWARD_WINDOW),
                   number("SIRCL_FORWARD_CHUNK_BYTES", routes_mod.DEFAULT_FORWARD_CHUNK), ring)


def queue_share(queue_bytes: int) -> int:
    return int(routes_mod.RELAY_QUEUE_SHARE * queue_bytes)


def session_reservation(lanes: LaneSet, settings: SessionSettings, *,
                        queue_bytes: int = routes_mod.DEFAULT_HAIRPIN_QUEUE) -> dict[QueueKey, int]:
    """Bytes the collective session of ``lanes`` may hold in every relay queue its lanes cross."""
    queues = lanes.queues()
    if not queues:
        return {}
    if settings.forward_window <= 0:
        return {key: queue_share(queue_bytes) for key in queues}
    maps = lanes.filtered_maps()
    tables = [routes_mod.forward_windows(lanes.layout, maps, rank, max_window=settings.forward_window,
                                         chunk=settings.forward_chunk, queue_bytes=queue_bytes)
              for rank in range(lanes.layout.world)]
    reserved = {key: sum(tables[rank][peer][lane] for rank, peer, lane in members)
                for key, members in queues.items()}
    if settings.ring_schedule:
        order = routes_mod.chain_order(lanes.layout, maps)
        if order is not None:
            window, problems = routes_mod.ring_window(lanes.layout, maps, order, queue_bytes=queue_bytes)
            if not problems:
                for key, members in routes_mod.ring_queues(lanes.layout, maps, order).items():
                    reserved[key] = max(reserved.get(key, 0), window * len(members))
    return reserved


@dataclasses.dataclass(frozen=True)
class Allocation:
    """Windows of every relayed point-to-point lane, and the lanes left without one."""

    windows: Mapping[LaneKey, int]
    unavailable: Mapping[LaneKey, str]
    reserved: Mapping[QueueKey, int]

    def table(self, lanes: LaneSet, lane_count: int) -> list[list[list[int]]]:
        """``table[rank][peer][lane]`` of one group (0: a direct lane or no channel)."""
        world = lanes.layout.world
        table = [[[0] * lane_count for _ in range(world)] for _ in range(world)]
        for (name, rank, peer, lane), window in self.windows.items():
            if name == lanes.name:
                table[rank][peer][lane] = window
        return table

    def unavailable_pairs(self, name: str) -> dict[tuple[int, int], str]:
        """Ordered pairs of group ``name`` whose channel has a lane without a window, with the reason."""
        found: dict[tuple[int, int], str] = {}
        for (group, rank, peer, _), reason in sorted(self.unavailable.items()):
            if group == name:
                found.setdefault((rank, peer), reason)
        return found


def _describe(key: QueueKey) -> str:
    return f"relay {key[0]} {'secondary' if key[1] else 'primary'} function toward port {key[2]}"


def allocate(levels: Sequence[Sequence[LaneSet]], *, reserved: Mapping[QueueKey, int] | None = None,
             max_window: int, chunk: int, queue_bytes: int = routes_mod.DEFAULT_HAIRPIN_QUEUE) -> Allocation:
    """Windows of the point-to-point lanes of every level, after ``reserved`` (see the module docstring)."""
    if chunk <= 0 or chunk % 16:
        raise BudgetError(f"forward chunk {chunk} must be a positive multiple of 16")
    share = queue_share(queue_bytes)
    used: dict[QueueKey, int] = dict(reserved or {})
    windows: dict[LaneKey, int] = {}
    unavailable: dict[LaneKey, str] = {}
    for level in levels:
        through: dict[QueueKey, list[LaneKey]] = {}
        for lanes in level:
            for key, members in lanes.queues().items():
                through.setdefault(key, []).extend((lanes.name, rank, peer, lane) for rank, peer, lane in members)
        lane_queues: dict[LaneKey, list[QueueKey]] = {}
        for key, members in through.items():
            for member in members:
                lane_queues.setdefault(member, []).append(key)
        for member, keys in sorted(lane_queues.items()):
            limit = max_window if max_window > 0 else 0
            reason = "" if max_window > 0 else "point-to-point windows are off (SIRCL_P2P_WINDOW_BYTES=0)"
            for key in keys:
                room = max(0, share - used.get(key, 0)) // len(through[key])
                if room < limit:
                    limit = room
                    if room < chunk:
                        reason = (f"{_describe(key)} has {max(0, share - used.get(key, 0))} of its {share} bytes "
                                  f"left for {len(through[key])} point-to-point lanes "
                                  f"({used.get(key, 0)} bytes reserved before them)")
            window = limit // chunk * chunk
            if window < chunk:
                unavailable[member] = reason or f"its window is below one chunk of {chunk} bytes"
                continue
            windows[member] = window
        for key, members in through.items():
            used[key] = used.get(key, 0) + sum(windows.get(member, 0) for member in members)
    return Allocation(windows, unavailable, dict(reserved or {}))


def group_windows(lanes: LaneSet, lane_count: int, *, max_window: int, chunk: int,
                  queue_bytes: int = routes_mod.DEFAULT_HAIRPIN_QUEUE) -> tuple[list[list[list[int]]], dict]:
    """Windows of one group's lanes when they are the only relayed traffic: ``(table, unavailable pairs)``."""
    allocation = allocate([[lanes]], max_window=max_window, chunk=chunk, queue_bytes=queue_bytes)
    return allocation.table(lanes, lane_count), allocation.unavailable_pairs(lanes.name)


def pp_groups(world: int, *, tp: int, pp: int, pcp: int = 1, dp: int = 1) -> tuple[tuple[int, ...], ...]:
    """Global ranks of every pipeline-parallel group, as vLLM forms them (layout ExternalDP x DP x PP x PCP x TP;
    a PP group holds one rank of every stage, ``initialize_model_parallel``)."""
    if min(world, tp, pp, pcp, dp) < 1:
        raise ValueError("group sizes must be positive")
    inner = dp * pp * pcp * tp
    if world % inner:
        raise ValueError(f"world size {world} is not a multiple of DP*PP*PCP*TP = {inner}")
    groups = []
    stage = pcp * tp
    for outer in range(world // inner):
        for data in range(dp):
            base = (outer * dp + data) * pp * stage
            for offset in range(stage):
                groups.append(tuple(base + s * stage + offset for s in range(pp)))
    return tuple(groups)


__all__ = ["Allocation", "BudgetError", "LaneSet", "QueueKey", "SessionSettings", "allocate", "group_windows",
           "pp_groups", "queue_share", "session_reservation"]
