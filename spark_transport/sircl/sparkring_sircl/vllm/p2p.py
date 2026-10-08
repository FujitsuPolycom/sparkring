"""SIRCL point-to-point channels of vLLM groups: which groups get them, and their forward windows.

A group whose kind ``SIRCL_P2P_GROUPS`` names (default ``pp,tp,dcp``) gets
point-to-point channels between every pair of its ranks
(:mod:`sparkring_sircl.p2p`): every pipeline-parallel (PP) group of two or
more ranks, and every tensor-parallel or decode-context-parallel group that
has its own SIRCL session. A group whose global ranks equal, in order, those
of a group with channels shares them (the EP and EPLB groups of a
mixture-of-experts model share the TP group's).

Placement. A PP group's lanes route over the instance's fabric (the cables
between the Sparks of every global rank,
``fabric.describe_group(layout, members, parent=<every position>)``), so a
pair of stages that shares no cable is joined through the relays of the
Sparks between them; a session group's channels use its session's fabric.

Windows. Relayed lanes stay within 75 % of every relay hairpin queue they
cross (:mod:`sparkring_sircl.p2p.budget`): the collective sessions of the
instance reserve their forward windows (and ring windows under a ``ring``
schedule) first, PP lanes share what is left, and the channels of session
groups share what PP lanes leave. :class:`InstancePlan` computes that for
every group of the instance from vLLM's parallel sizes, ``SIRCL_FABRIC``, the
rank positions and the session settings, which every rank shares, so every
rank of a group passes the same window table (the channels' setup agreement
compares it). A pair left without one chunk of window has no channel; send and
receive to it are refused and the receipt names the queue. Without vLLM's
parallel sizes (tools, tests) the plan covers the group and its own session
only.

Status: implemented; CPU tests (``tests/test_vllm_p2p.py``); not run on a
ring.
"""

from __future__ import annotations

import dataclasses
import functools
import os
from collections.abc import Mapping, Sequence
from typing import Any

from .. import groups as groups_mod
from .. import routes as routes_mod
from ..p2p import budget
from ..p2p.settings import P2PSettings
from . import fabric
from .fabric import FabricError, GroupTopology, Layout

P2P_KINDS = ("pp", "tp", "dcp")
SESSION_KINDS = ("tp", "dcp")
# Variables whose values shape the instance plan (the cache key holds them).
PLAN_VARIABLES = ("SIRCL_FORWARD_WINDOW_BYTES", "SIRCL_FORWARD_CHUNK_BYTES", "SIRCL_HAIRPIN_QUEUE_BYTES",
                  "SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE",
                  "SIRCL_P2P_WINDOW_BYTES", "SIRCL_P2P_CHUNK_BYTES", "SIRCL_MAX_RELAYS")


@dataclasses.dataclass(frozen=True)
class Shape:
    """vLLM's parallel sizes of the instance."""

    world: int
    tp: int
    pp: int = 1
    dcp: int = 1
    pcp: int = 1

    def __post_init__(self) -> None:
        if min(self.world, self.tp, self.pp, self.dcp, self.pcp) < 1 or self.world % (self.tp * self.pp * self.pcp):
            raise ValueError(f"vLLM sizes TP {self.tp} x PP {self.pp} x PCP {self.pcp} do not divide the world "
                             f"of {self.world}")

    @property
    def dp(self) -> int:
        return self.world // (self.tp * self.pp * self.pcp)

    def tp_groups(self) -> tuple[tuple[int, ...], ...]:
        return groups_mod.tp_groups(self.world, tp=self.tp, pcp=self.pcp, pp=self.pp, dp=self.dp)

    def dcp_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.dcp <= 1:
            return ()
        return groups_mod.dcp_groups(self.world, tp=self.tp, dcp=self.dcp, pcp=self.pcp, pp=self.pp, dp=self.dp)

    def pp_groups(self) -> tuple[tuple[int, ...], ...]:
        if self.pp <= 1:
            return ()
        return budget.pp_groups(self.world, tp=self.tp, pp=self.pp, pcp=self.pcp, dp=self.dp)


def shape_of(vllm_config: Any, world: int) -> Shape | None:
    """vLLM's parallel sizes from its config, or None when the config does not state them."""
    parallel = getattr(vllm_config, "parallel_config", None)
    if parallel is None:
        return None
    try:
        return Shape(int(world), int(getattr(parallel, "tensor_parallel_size", 1) or 1),
                     int(getattr(parallel, "pipeline_parallel_size", 1) or 1),
                     int(getattr(parallel, "decode_context_parallel_size", 1) or 1),
                     int(getattr(parallel, "prefill_context_parallel_size", 1) or 1))
    except (TypeError, ValueError):
        return None


def name_of(kind: str, global_ranks: Sequence[int]) -> str:
    return f"{kind}{list(int(r) for r in global_ranks)}"


@dataclasses.dataclass(frozen=True)
class GroupChannels:
    """One group's channel plan, the same on every rank of the group."""

    name: str
    kind: str
    global_ranks: tuple[int, ...]
    topology: GroupTopology
    windows: tuple[tuple[tuple[int, ...], ...], ...]   # [rank][peer][lane]
    unavailable: Mapping[tuple[int, int], str]
    basis: str                                         # "instance" or "group"

    @property
    def layout_text(self) -> str:
        return self.topology.session_layout()

    def route_map(self, rank: int) -> dict[int, tuple[str, ...]]:
        return self.topology.route_map(rank)

    def relayed_peers(self, rank: int) -> frozenset[int]:
        return frozenset(peer for peer in range(len(self.global_ranks))
                         if peer != rank and self.topology.hops(rank, peer) > 1)

    def window_text(self) -> str:
        values = sorted({w for row in self.windows for lanes in row for w in lanes if w})
        if not values:
            return "direct"
        return f"windows={values[0]}" if len(values) == 1 else f"windows={values[0]}-{values[-1]}"


def topology_for(kind: str, layout: Layout, positions: Sequence[int], *, instance: Sequence[int],
                 parent: Sequence[int] | None = None) -> GroupTopology:
    """Where a group's channels route: a PP group over the instance's fabric, a session group as its session."""
    if kind == "pp":
        return fabric.describe_group(layout, positions, parent=tuple(sorted(set(instance))))
    if kind == "dcp" and parent is not None:
        return fabric.describe_group(layout, positions, parent=parent)
    return fabric.describe_group(layout, positions)


def _lane_set(name: str, topology: GroupTopology) -> budget.LaneSet:
    layout = routes_mod.Layout.parse(topology.session_layout())
    world = len(topology.members)
    maps = tuple(topology.route_map(rank) for rank in range(world))
    pairs = frozenset((a, b) for a in range(world) for b in range(world) if a != b)
    return budget.LaneSet(name, layout, maps, pairs)


class InstancePlan:
    """Channel plans of every group of one vLLM instance (see the module docstring)."""

    def __init__(self, layout: Layout, positions: Sequence[int], session_groups: Sequence[str],
                 p2p_groups: Sequence[str], shape: Shape | None, environ: Mapping[str, str], *,
                 only: tuple[str, tuple[int, ...], tuple[int, ...] | None] | None = None) -> None:
        self.layout = layout
        self.positions = tuple(int(p) for p in positions)
        self.shape = shape
        self.settings = P2PSettings.from_env(environ)
        session_settings = budget.SessionSettings.from_env(environ)
        queue = self.settings.hairpin_queue_bytes
        sessions: list[tuple[str, tuple[int, ...], GroupTopology]] = []
        levels: list[list[tuple[str, str, tuple[int, ...], GroupTopology]]] = [[], []]
        instance = self.positions

        def place(kind: str, ranks: Sequence[int], parent: Sequence[int] | None = None) -> GroupTopology:
            return topology_for(kind, layout, [self.positions[r] for r in ranks], instance=instance,
                                parent=None if parent is None else [self.positions[r] for r in parent])

        if shape is not None:
            self.basis = "instance"
            tp_of = {}
            for ranks in shape.tp_groups():
                for r in ranks:
                    tp_of[r] = ranks
                if len(ranks) > 1 and "tp" in session_groups:
                    sessions.append(("tp", ranks, place("tp", ranks)))
            for ranks in shape.dcp_groups():
                if len(ranks) > 1 and "dcp" in session_groups:
                    sessions.append(("dcp", ranks, place("dcp", ranks, tp_of[ranks[0]])))
            if "pp" in p2p_groups:
                for ranks in shape.pp_groups():
                    levels[0].append(("pp", name_of("pp", ranks), ranks, place("pp", ranks)))
        else:
            # Without vLLM's sizes: the one group and its own session.
            self.basis = "group"
            if only is None:
                raise ValueError("an instance plan without vLLM's parallel sizes needs the group it plans")
            kind, ranks, parent = only
            topology = place(kind, ranks, parent)
            if kind in SESSION_KINDS and kind in session_groups:
                sessions.append((kind, ranks, topology))
            if kind == "pp" and "pp" in p2p_groups:
                levels[0].append((kind, name_of(kind, ranks), ranks, topology))
        for kind, ranks, topology in sessions:
            if kind in p2p_groups:
                levels[1].append((kind, name_of(kind, ranks), ranks, topology))
        reserved: dict[budget.QueueKey, int] = {}
        for kind, ranks, topology in sessions:
            for key, value in budget.session_reservation(_lane_set(name_of(kind, ranks), topology),
                                                         session_settings, queue_bytes=queue).items():
                reserved[key] = reserved.get(key, 0) + value
        lane_sets = {name: _lane_set(name, topology) for level in levels for _, name, _, topology in level}
        self.allocation = budget.allocate([[lane_sets[name] for _, name, _, _ in level] for level in levels],
                                          reserved=reserved, max_window=self.settings.window_bytes,
                                          chunk=self.settings.chunk_bytes, queue_bytes=queue)
        self._groups: dict[str, GroupChannels] = {}
        for level in levels:
            for kind, name, ranks, topology in level:
                table = self.allocation.table(lane_sets[name], topology.lane_count)
                self._groups[name] = GroupChannels(
                    name, kind, tuple(ranks), topology,
                    tuple(tuple(tuple(lanes) for lanes in row) for row in table),
                    self.allocation.unavailable_pairs(name), self.basis)

    def channels_for(self, kind: str, global_ranks: Sequence[int]) -> GroupChannels | None:
        return self._groups.get(name_of(kind, global_ranks))

    def groups(self) -> list[GroupChannels]:
        return list(self._groups.values())


@functools.lru_cache(maxsize=32)
def _cached_plan(layout: Layout, positions: tuple[int, ...], session_groups: tuple[str, ...],
                 p2p_groups: tuple[str, ...], shape: Shape | None, environ: tuple[tuple[str, str], ...],
                 only: tuple[str, tuple[int, ...], tuple[int, ...] | None] | None) -> InstancePlan:
    return InstancePlan(layout, positions, session_groups, p2p_groups, shape, dict(environ), only=only)


def plan(layout: Layout, positions: Sequence[int], session_groups: Sequence[str], p2p_groups: Sequence[str],
         shape: Shape | None, *, kind: str, global_ranks: Sequence[int], parent_ranks: Sequence[int] | None = None,
         environ: Mapping[str, str] | None = None) -> GroupChannels | None:
    """The channel plan of one group (None when the group gets no channels of its own)."""
    env = os.environ if environ is None else environ
    snapshot = tuple((name, env.get(name, "")) for name in PLAN_VARIABLES + tuple(
        v for v in ("SIRCL_P2P_SLOTS", "SIRCL_P2P_SLOT_BYTES", "SIRCL_P2P_BLOCKS", "SIRCL_P2P_THREADS",
                    "SIRCL_P2P_UNROLL", "SIRCL_STARTUP_WAIT_S", "SIRCL_SERVING_WAIT_S")))
    ranks = tuple(int(r) for r in global_ranks)
    only = None if shape is not None else (kind, ranks, None if parent_ranks is None else tuple(parent_ranks))
    return _cached_plan(layout, tuple(positions), tuple(session_groups), tuple(p2p_groups), shape, snapshot,
                        only).channels_for(kind, ranks)


__all__ = ["FabricError", "GroupChannels", "InstancePlan", "P2P_KINDS", "Shape", "name_of", "plan", "shape_of",
           "topology_for"]
