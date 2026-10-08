"""Which Sparks of a vLLM group share a cable, and what that allows.

This module answers three questions for every vLLM process group, from the
physical fabric (``SIRCL_FABRIC``) and the fabric position of every global rank
(``SIRCL_RANK_POSITIONS``):

1. **Group fabric.** A top-level group (tensor
   parallel, expert parallel, ...) owns the cables between its members, so its
   members must form a path or a cycle of the physical fabric: four
   consecutive Sparks of an eight-Spark ring form a path, all eight form a
   cycle. A subgroup (a decode-context-parallel group inside a
   tensor-parallel group) routes over its parent's fabric.
2. **NCCL policy.** NCCL chooses its devices and addresses without the relay
   table, so it can connect only Sparks that share a cable; its queue-pair
   setup to any other Spark times out. A group whose every pair of members
   shares a cable (a pair, a triangle) may run any NCCL collective
   (:attr:`NcclPolicy.ALL`). A group whose consecutive ranks, including the
   last and the first, share cables may run NCCL's ring algorithm only
   (:attr:`NcclPolicy.RING`: ``NCCL_ALGO=Ring`` with tree connection setup
   skipped, the repository's patched-NCCL cycle contract). Every other group
   may run no NCCL collective at all (:attr:`NcclPolicy.NONE`): four
   consecutive Sparks of a larger ring are such a group, because rank 3 reaches
   rank 0 only through two relays.
3. **Route map and relay load.** Every rank's lane
   devices toward every other rank, and the relay-load factor ``f``: the
   busiest relay hairpin queue holds about ``f * b`` bytes when every rank
   sends ``b`` bytes to every peer.

Everything here is pure data computed identically on every rank; nothing
imports torch or vLLM.

Status: implemented; the route rules reproduce every map of the reference
vectors in ``tests/data/routes.json`` (CPU test ``test_vllm_fabric``).
"""

from __future__ import annotations

import dataclasses
import enum
import itertools
from collections import deque
from collections.abc import Mapping, Sequence

from .. import routes as _routes

PRIMARY = "primary"
SECONDARY = "secondary"
CLASSES = (PRIMARY, SECONDARY)

# RDMA device of each (port, function class): the roles of sparkring_sircl.routes, which are
# the DGX OS names unless SIRCL_FABRIC_DOCUMENT names a fabric document with others. Port 0
# traffic travels clockwise on a ring, port 1 traffic counter-clockwise.
CANONICAL_DEVICES: Mapping[tuple[int, str], str] = {
    (role.port, SECONDARY if role.secondary else PRIMARY): role.device for role in _routes.ROLES}
ROLE_NAMES: Mapping[tuple[int, str], str] = {
    (0, PRIMARY): "cw_primary",
    (0, SECONDARY): "cw_secondary",
    (1, PRIMARY): "ccw_primary",
    (1, SECONDARY): "ccw_secondary",
}
MAX_LANES = 2
DEFAULT_MAX_RELAYS = 3
HAIRPIN_QUEUE_BYTES = 512 * 1024
RELAY_FILL_LIMIT = 0.75
# Per-peer op sizes measured drop-free on the eight-Spark ring,
# keyed by the relay-load factor of the layout they were measured on: 3 for
# the ring of eight, 1 for a path of four.
MEASURED_PER_PEER_BYTES: Mapping[float, int] = {3.0: 131072, 1.0: 262144}


class FabricError(ValueError):
    """The layout, a group's placement on it or a route map is inconsistent."""


class NcclPolicy(str, enum.Enum):
    """What NCCL may do on a group, from its cabling."""

    ALL = "all"      # every pair of members shares a cable
    RING = "ring"    # consecutive members share cables, last to first included
    NONE = "none"    # some consecutive pair shares no cable

    def allows(self, operation: str) -> bool:
        """Whether NCCL may run ``operation`` on a group of this policy.

        Ring-shaped operations (all-reduce, all-gather, reduce-scatter,
        broadcast, reduce) connect each rank to its ring neighbours only;
        everything else (all-to-all, gather, scatter, point-to-point between
        arbitrary ranks) may connect any two ranks.
        """
        if self is NcclPolicy.ALL:
            return True
        if self is NcclPolicy.NONE:
            return False
        return operation in RING_OPERATIONS


RING_OPERATIONS = frozenset({
    "all_reduce", "all_gather", "all_gather_into_tensor", "reduce_scatter",
    "reduce_scatter_tensor", "broadcast", "reduce", "barrier", "all_gather_object",
    "all_reduce_coalesced", "all_gather_coalesced",
})


@dataclasses.dataclass(frozen=True)
class Cable:
    """A cable between port ``a_port`` of position ``a`` and port ``b_port`` of position ``b``."""

    a: int
    a_port: int
    b: int
    b_port: int

    def ends(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.a, self.a_port), (self.b, self.b_port)

    def label(self) -> str:
        return f"{self.a}.port{self.a_port}-{self.b}.port{self.b_port}"


@dataclasses.dataclass(frozen=True)
class Layout:
    """The physical fabric: Spark positions and the cables between their ports."""

    kind: str                  # ring, path or pair
    size: int
    cables: tuple[Cable, ...]

    @classmethod
    def ring(cls, size: int) -> "Layout":
        if size < 2:
            raise FabricError("a ring needs at least two Sparks")
        return cls("ring", size, tuple(Cable(i, 0, (i + 1) % size, 1) for i in range(size)))

    @classmethod
    def path(cls, size: int) -> "Layout":
        if size < 2:
            raise FabricError("a path needs at least two Sparks")
        return cls("path", size, tuple(Cable(i, 0, i + 1, 1) for i in range(size - 1)))

    @classmethod
    def pair(cls, cables: int = 1) -> "Layout":
        if cables not in (1, 2):
            raise FabricError("a pair has one or two cables")
        return cls("pair", 2, tuple(Cable(0, port, 1, port) for port in range(cables)))

    @classmethod
    def parse(cls, text: str) -> "Layout":
        """``ring:N``, ``path:N``, ``pair`` or ``pair:2``."""
        kind, _, count = text.strip().lower().partition(":")
        try:
            number = int(count) if count else None
        except ValueError:
            raise FabricError(f"layout {text!r}: {count!r} is not a number") from None
        if kind == "ring" and number is not None:
            return cls.ring(number)
        if kind == "path" and number is not None:
            return cls.path(number)
        if kind == "pair":
            return cls.pair(1 if number is None else number)
        raise FabricError(f"layout {text!r} must be ring:N, path:N, pair or pair:2")

    def describe(self) -> str:
        return f"{self.kind}:{self.size}" if self.kind != "pair" else f"pair:{len(self.cables)}"


@dataclasses.dataclass(frozen=True)
class Fabric:
    """The positions and cables a group may route over."""

    kind: str                  # cycle, path or pair
    order: tuple[int, ...]     # positions in cabling order
    cables: tuple[Cable, ...]

    def adjacent(self, a: int, b: int) -> bool:
        return any({a, b} == {cable.a, cable.b} for cable in self.cables)

    def describe(self) -> str:
        return f"{self.kind}:{'-'.join(str(p) for p in self.order)}"


def group_fabric(layout: Layout, members: Sequence[int]) -> Fabric:
    """The fabric a top-level group owns: its members must be a path or cycle of the layout."""
    members = tuple(members)
    _check_positions(layout, members)
    wanted = set(members)
    if len(members) == 1:
        return Fabric("path", members, ())
    if layout.kind == "pair":
        return Fabric("pair", (0, 1), layout.cables)
    if layout.kind == "ring" and wanted == set(range(layout.size)):
        kind = "cycle" if layout.size >= 3 else "pair"
        return Fabric(kind, tuple(range(layout.size)), layout.cables)
    starts = range(layout.size) if layout.kind == "ring" else range(layout.size - len(members) + 1)
    for start in starts:
        arc = tuple((start + step) % layout.size for step in range(len(members)))
        if set(arc) == wanted:
            # Cables in path order, so a path across the ring's last cable (7-0-1-2)
            # lists 7-0 first, as the ring harness's layout text does.
            owned = tuple(cable for a, b in zip(arc, arc[1:]) for cable in layout.cables
                          if {cable.a, cable.b} == {a, b})
            return Fabric("pair" if len(arc) == 2 else "path", arc, owned)
    raise FabricError(
        f"positions {list(members)} are not consecutive Sparks of the {layout.describe()} "
        "layout; a group owns the cables between its members and needs them to form a path "
        "or a cycle"
    )


def _check_positions(layout: Layout, members: Sequence[int]) -> None:
    if len(set(members)) != len(members):
        raise FabricError(f"group positions {list(members)} repeat a Spark")
    outside = [p for p in members if not 0 <= p < layout.size]
    if outside:
        raise FabricError(f"positions {outside} are outside the {layout.describe()} layout")


@dataclasses.dataclass(frozen=True)
class Lane:
    """One lane of a rank toward a peer."""

    peer: int
    index: int
    port: int                  # local port the lane leaves through
    function_class: str        # primary or secondary
    remote_port: int           # port at the peer the lane arrives on
    positions: tuple[int, ...]  # Sparks along the lane, origin first
    cables: tuple[str, ...]

    @property
    def hops(self) -> int:
        return len(self.cables)

    @property
    def relays(self) -> tuple[int, ...]:
        return self.positions[1:-1]

    def device(self, devices: Mapping[tuple[int, str], str] = CANONICAL_DEVICES) -> str:
        return devices[(self.port, self.function_class)]

    def remote_device(self, devices: Mapping[tuple[int, str], str] = CANONICAL_DEVICES) -> str:
        return devices[(self.remote_port, self.function_class)]

    @property
    def role(self) -> str:
        return ROLE_NAMES[(self.port, self.function_class)]


# A step along a cable: (from position, port left, to position, port arrived, cable)
_Step = tuple[int, int, int, int, Cable]


def _shortest_paths(fabric: Fabric, origin: int, destination: int) -> list[list[_Step]]:
    links: dict[int, list[tuple[int, int, int, Cable]]] = {}
    for cable in fabric.cables:
        links.setdefault(cable.a, []).append((cable.a_port, cable.b, cable.b_port, cable))
        links.setdefault(cable.b, []).append((cable.b_port, cable.a, cable.a_port, cable))
    distance = {destination: 0}
    queue = deque([destination])
    while queue:
        node = queue.popleft()
        for _, other, _, _ in links.get(node, ()):
            if other not in distance:
                distance[other] = distance[node] + 1
                queue.append(other)
    if origin not in distance:
        return []
    paths: list[list[_Step]] = []

    def extend(node: int, steps: list[_Step]) -> None:
        if node == destination:
            paths.append(list(steps))
            return
        for port, other, other_port, cable in links.get(node, ()):
            if distance.get(other) == distance[node] - 1:
                steps.append((node, port, other, other_port, cable))
                extend(other, steps)
                steps.pop()

    extend(origin, [])
    return paths


def lanes_toward(fabric: Fabric, members: Sequence[int], rank: int, peer: int,
                 lane_count: int = MAX_LANES) -> tuple[Lane, ...]:
    """The lanes of group rank ``rank`` toward group rank ``peer``."""
    if lane_count not in (1, 2):
        raise FabricError("a route has one or two lanes per peer")
    origin, destination = members[rank], members[peer]
    paths = _shortest_paths(fabric, origin, destination)
    if not paths:
        raise FabricError(f"position {origin} cannot reach position {destination} over the "
                          f"group's fabric ({fabric.describe()})")
    if len(paths) == 1:
        chosen = [(paths[0], PRIMARY), (paths[0], SECONDARY)]
    elif len(paths) == 2:
        # Path A leaves the end with the smaller position through its port 0;
        # seen from the larger end it arrives there on port 0.
        smaller_is_origin = origin < destination

        def is_a(path: list[_Step]) -> bool:
            return path[0][1] == 0 if smaller_is_origin else path[-1][3] == 0

        a_paths = [path for path in paths if is_a(path)]
        b_paths = [path for path in paths if not is_a(path)]
        if len(a_paths) != 1 or len(b_paths) != 1:
            raise FabricError(f"positions {origin} and {destination} have two shortest paths "
                              "that the port-0 rule cannot order")
        chosen = [(a_paths[0], PRIMARY), (b_paths[0], SECONDARY)]
    else:
        raise FabricError(f"positions {origin} and {destination} have {len(paths)} shortest "
                          "paths; Spark fabrics have at most two")
    lanes = []
    for index, (path, function_class) in enumerate(chosen[:lane_count]):
        positions = (path[0][0],) + tuple(step[2] for step in path)
        lanes.append(Lane(peer, index, path[0][1], function_class, path[-1][3], positions,
                          tuple(step[4].label() for step in path)))
    return tuple(lanes)


@dataclasses.dataclass(frozen=True)
class GroupTopology:
    """A vLLM group placed on the fabric: everything rank-invariant the adapter needs."""

    layout: Layout
    fabric: Fabric
    members: tuple[int, ...]           # fabric position of each group rank
    lane_count: int
    max_relays_allowed: int
    subgroup: bool

    # -- NCCL ----------------------------------------------------------------

    @property
    def nccl_policy(self) -> NcclPolicy:
        return self._nccl()[0]

    @property
    def nccl_reason(self) -> str:
        return self._nccl()[1]

    def uncabled_pairs(self) -> tuple[tuple[int, int], ...]:
        """Consecutive group-rank pairs (last to first included) that share no cable."""
        n = len(self.members)
        if n < 2:
            return ()
        pairs = [(i, (i + 1) % n) for i in range(n if n > 2 else 1)]
        return tuple((a, b) for a, b in pairs
                     if not self.fabric.adjacent(self.members[a], self.members[b]))

    def _nccl(self) -> tuple[NcclPolicy, str]:
        n = len(self.members)
        if n < 2:
            return NcclPolicy.ALL, "single rank"
        if all(self.fabric.adjacent(a, b) for a, b in itertools.combinations(self.members, 2)):
            return NcclPolicy.ALL, "every pair of ranks shares a cable"
        missing = self.uncabled_pairs()
        if not missing:
            return NcclPolicy.RING, ("consecutive ranks share cables around the whole group; "
                                     "NCCL ring algorithm only")
        described = ", ".join(f"ranks {a}-{b} (positions {self.members[a]}-{self.members[b]}, "
                              f"{self.hops(a, b) - 1} relays)" for a, b in missing)
        return NcclPolicy.NONE, f"no cable between {described}"

    def pair_cabled(self, a: int, b: int) -> bool:
        return self.fabric.adjacent(self.members[a], self.members[b])

    # -- routes ------------------------------------------------------------------

    def lanes(self, rank: int, peer: int) -> tuple[Lane, ...]:
        return lanes_toward(self.fabric, self.members, rank, peer, self.lane_count)

    def hops(self, rank: int, peer: int) -> int:
        return self.lanes(rank, peer)[0].hops

    def route_map(self, rank: int,
                  devices: Mapping[tuple[int, str], str] = CANONICAL_DEVICES) -> dict[int, tuple[str, ...]]:
        """Rank ``rank``'s route map: peer -> local lane devices."""
        return {peer: tuple(lane.device(devices) for lane in self.lanes(rank, peer))
                for peer in range(len(self.members)) if peer != rank}

    def max_relays(self) -> int:
        n = len(self.members)
        return max((lane.hops - 1 for a in range(n) for b in range(n) if a != b
                    for lane in self.lanes(a, b)), default=0)

    def relay_factor(self) -> float:
        """Relay-load factor ``f``: lanes through the busiest (relay, port, class) over the lane count."""
        counts: dict[tuple[int, int, str], int] = {}
        n = len(self.members)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                for lane in self.lanes(a, b):
                    for position, cable in zip(lane.positions[1:-1], lane.cables[1:]):
                        port = _port_at(cable, position)
                        key = (position, port, lane.function_class)
                        counts[key] = counts.get(key, 0) + 1
        return max(counts.values(), default=0) / self.lane_count

    def per_peer_op_bytes(self, override: int | None = None) -> tuple[int | None, str]:
        """Largest per-peer bytes of one gather or scatter op, and the basis of that value.

        ``None`` means no relay constraint (no lane crosses a relay).
        """
        if override is not None:
            return override, "SIRCL_RELAY_PER_PEER_BYTES"
        f = self.relay_factor()
        if f == 0:
            return None, "no relayed lane"
        rule = int(HAIRPIN_QUEUE_BYTES * RELAY_FILL_LIMIT / f) // 16 * 16
        measured = MEASURED_PER_PEER_BYTES.get(f)
        if measured is not None:
            return min(rule, measured), f"measured drop-free value for relay-load factor {f:g}"
        return rule, f"relay-load rule (75 % of a 512 KiB hairpin queue) for factor {f:g} (unmeasured layout)"

    def session_layout(self) -> str:
        """The session's layout in the session package's explicit form.

        ``cables=<cable>,...;positions=<p>,...``: the cables the group may route
        over (its own for a top-level group, its parent's for a subgroup) and
        the fabric position of every group rank.
        """
        return ("cables=" + ",".join(cable.label() for cable in self.fabric.cables)
                + ";positions=" + ",".join(str(p) for p in self.members))

    def identity(self) -> dict[str, object]:
        """The layout identity the session checks, equal on every rank of the group."""
        return {"layout": self.layout.describe(), "fabric": self.fabric.kind,
                "fabric_order": list(self.fabric.order), "positions": list(self.members),
                "lanes": self.lane_count}

    def describe(self) -> str:
        return (f"{'sub' if self.subgroup else ''}group at positions {list(self.members)} on "
                f"{self.fabric.describe()} of {self.layout.describe()}")


def _port_at(cable_label: str, position: int) -> int:
    """Port of ``position`` on the cable labelled ``a.portX-b.portY`` (the egress at a relay)."""
    left, right = cable_label.split("-")
    for end in (left, right):
        node, _, port = end.partition(".port")
        if int(node) == position:
            return int(port)
    raise FabricError(f"position {position} is not an end of cable {cable_label}")


def nccl_policy_of(layout: Layout, positions: Sequence[int]) -> tuple[NcclPolicy, str]:
    """What NCCL may do on a group of any shape, from the physical cables between its ranks.

    Unlike :func:`describe_group` this accepts members that do not form a path
    or cycle of their own (a pipeline-parallel group, the world group), because
    NCCL's reach depends only on which ranks share a cable.
    """
    positions = tuple(int(p) for p in positions)
    _check_positions(layout, positions)
    fabric = Fabric(layout.kind, tuple(range(layout.size)), layout.cables)
    n = len(positions)
    if n < 2:
        return NcclPolicy.ALL, "single rank"
    if all(fabric.adjacent(a, b) for a, b in itertools.combinations(positions, 2)):
        return NcclPolicy.ALL, "every pair of ranks shares a cable"
    pairs = [(i, (i + 1) % n) for i in range(n if n > 2 else 1)]
    missing = [(a, b) for a, b in pairs if not fabric.adjacent(positions[a], positions[b])]
    if not missing:
        return NcclPolicy.RING, ("consecutive ranks share cables around the whole group; NCCL "
                                 "ring algorithm only")
    described = ", ".join(f"ranks {a}-{b} (positions {positions[a]}-{positions[b]})"
                          for a, b in missing)
    return NcclPolicy.NONE, f"no cable between {described}"


def describe_group(
    layout: Layout,
    members: Sequence[int],
    *,
    parent: Sequence[int] | None = None,
    lane_count: int = MAX_LANES,
    max_relays: int = DEFAULT_MAX_RELAYS,
) -> GroupTopology:
    """Place a group on the layout and check its lanes against the relay limit."""
    members = tuple(int(p) for p in members)
    if parent is None:
        fabric = group_fabric(layout, members)
        subgroup = False
    else:
        parent = tuple(int(p) for p in parent)
        if not set(members) <= set(parent):
            raise FabricError(f"subgroup positions {list(members)} are not inside the parent "
                              f"group {list(parent)}")
        fabric = group_fabric(layout, parent)
        subgroup = True
    topology = GroupTopology(layout, fabric, members, lane_count, max_relays, subgroup)
    relays = topology.max_relays() if len(members) > 1 else 0
    if relays > max_relays:
        raise FabricError(
            f"the {topology.describe()} has lanes through {relays} relays; the qualified limit "
            f"is {max_relays} (SIRCL_MAX_RELAYS)"
        )
    return topology


# -- route map text: peer=device[/device],... ---------------------------------------


def format_routes(routes: Mapping[int, Sequence[str]]) -> str:
    return ",".join(f"{peer}={'/'.join(devices)}" for peer, devices in sorted(routes.items()))


def parse_routes(text: str, *, world: int, rank: int, name: str = "SIRCL_PEER_ROUTES") -> dict[int, tuple[str, ...]]:
    """Parse ``peer=dev[/dev],...`` for group rank ``rank``.

    Every other rank once, one or two distinct devices for each, the same lane
    count for every peer, at most four devices in all.
    """
    routes: dict[int, tuple[str, ...]] = {}
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            continue
        key, separator, value = entry.partition("=")
        if not separator:
            raise FabricError(f"{name} entry {entry!r} is not peer=device[/device]")
        try:
            peer = int(key.strip())
        except ValueError:
            raise FabricError(f"{name} entry {entry!r}: {key.strip()!r} is not a rank") from None
        devices = tuple(item.strip() for item in value.split("/"))
        if peer in routes:
            raise FabricError(f"{name} names rank {peer} twice")
        if not 1 <= len(devices) <= MAX_LANES or not all(devices) or len(set(devices)) != len(devices):
            raise FabricError(f"{name} entry for rank {peer} must list 1 to {MAX_LANES} distinct "
                              f"device names")
        routes[peer] = devices
    expected = set(range(world)) - {rank}
    if rank in routes:
        raise FabricError(f"{name} of rank {rank} names its own rank")
    if set(routes) != expected:
        raise FabricError(f"{name} of rank {rank} must name ranks {sorted(expected)}, names "
                          f"{sorted(routes)}")
    if len({len(devices) for devices in routes.values()}) > 1:
        raise FabricError(f"{name} of rank {rank} gives peers different lane counts")
    if len({device for devices in routes.values() for device in devices}) > 4:
        raise FabricError(f"{name} of rank {rank} names more than four devices")
    return routes


def check_routes(topology: GroupTopology, rank: int, routes: Mapping[int, Sequence[str]],
                 devices: Mapping[tuple[int, str], str] = CANONICAL_DEVICES) -> None:
    """A given map must equal the map the layout derives for this rank."""
    expected = topology.route_map(rank, devices)
    lanes = {len(value) for value in routes.values()}
    if lanes and lanes != {topology.lane_count}:
        raise FabricError(f"route map of rank {rank} has {sorted(lanes)} lanes per peer, the "
                          f"layout {topology.lane_count}")
    for peer, wanted in expected.items():
        given = tuple(routes.get(peer, ()))
        if given != wanted:
            lanes_detail = "; ".join(
                f"lane {lane.index}: {lane.role} through {lane.hops - 1} relays"
                for lane in topology.lanes(rank, peer))
            raise FabricError(
                f"route map of rank {rank} sends rank {peer} over {'/'.join(given) or 'nothing'}; "
                f"the {topology.describe()} needs {'/'.join(wanted)} ({lanes_detail})"
            )


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def chunk_ranges(total: int, chunk: int, align: int = 1) -> list[tuple[int, int]]:
    """``(offset, length)`` pieces of ``total`` units of at most ``chunk`` units each.

    ``chunk`` is rounded down to a multiple of ``align`` (at least ``align``).
    """
    if total <= 0:
        return []
    step = max(align, chunk // align * align)
    return [(offset, min(step, total - offset)) for offset in range(0, total, step)]


__all__ = [
    "CANONICAL_DEVICES", "Cable", "Fabric", "FabricError", "GroupTopology", "Lane", "Layout",
    "MAX_LANES", "NcclPolicy", "RING_OPERATIONS", "ceil_div", "check_routes", "chunk_ranges",
    "describe_group", "format_routes", "group_fabric", "lanes_toward", "nccl_policy_of",
    "parse_routes",
]
