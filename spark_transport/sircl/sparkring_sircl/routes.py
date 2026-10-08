"""Fabric layouts and route maps of SIRCL ring sessions.

Physical model. A DGX Spark has one ConnectX-7 with ports 0 and 1; each port
appears as a primary and a secondary PCIe function, each with its own RDMA
device. The repository names the four functions by role:

    cw_primary     rocep1s0f0     port 0, primary
    cw_secondary   roceP2p1s0f0   port 0, secondary
    ccw_primary    rocep1s0f1     port 1, primary
    ccw_secondary  roceP2p1s0f1   port 1, secondary

These are the names DGX OS gives the functions (:data:`DEFAULT_ROLES`).
When ``SIRCL_FABRIC_DOCUMENT`` names a fabric document (schema
``sparkring-fabric/v1``, written by SparkRing's setup), the devices and
network interfaces come from its port functions instead
(:func:`load_roles`); every Spark of the fabric must name them alike.

A cable joins one port of one Spark to one port of another and carries two
links (the primary functions of its ends and the secondary functions). A
group's fabric is the set of cables it owns, written as
``"<position>.port<p>-<position>.port<q>"``; every Spark has two ports, so a
fabric is a path or a cycle. Ring cabling joins position ``i``'s port 0 to
position ``i+1``'s port 1.

Route maps. A rank's route map names, for every other rank of its session,
the local RDMA devices of lanes ``0 .. L-1`` (``L`` is 1 or 2). Textual form
(``SIRCL_PEER_ROUTES``): ``<peer>=<device>[/<device>][,...]``.

Derivation (:func:`derive_lanes`), for a lane set from position ``a`` to
position ``b``: take the shortest paths in the fabric (one, or two when the
two ends are opposite on an even cycle or joined by two cables). With one
path, lane 0 uses the primary and lane 1 the secondary function of the path's
first cable at ``a``. With two paths, path A is the one that leaves the end
with the smaller position through its port 0; lane 0 uses the primary function
on path A and lane 1 the secondary function on path B. A lane's remote device
is the function of the same class at the far end of its last cable. Subgroup
sessions use their parent group's fabric.

Validation (:func:`validate_route_map`) checks one rank's map before setup;
:func:`check_complementary` checks every ordered pair of ranks against the
layout; :func:`relay_load` gives the relay queue load factor of a group.

Status: implemented; checked against the reference route vectors of the
tests (``tests/data/routes.json``).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from collections import deque
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

from .protocol import MAX_DEVICES, MAX_LANES

DEFAULT_MAX_RELAYS = 3


class RouteError(ValueError):
    """A route map or layout breaks a route rule (the message names the rank, entry and rule)."""


@dataclasses.dataclass(frozen=True)
class Role:
    """One PCIe function of a Spark's ConnectX-7."""

    name: str
    device: str
    netdev: str
    port: int
    secondary: bool

    @property
    def label(self) -> str:
        """The hyphenated spelling of the role, e.g. ``cw-primary``."""
        return self.name.replace("_", "-")


# The four functions under the names DGX OS gives a DGX Spark's ConnectX-7.
DEFAULT_ROLES = (
    Role("cw_primary", "rocep1s0f0", "enp1s0f0np0", 0, False),
    Role("cw_secondary", "roceP2p1s0f0", "enP2p1s0f0np0", 0, True),
    Role("ccw_primary", "rocep1s0f1", "enp1s0f1np1", 1, False),
    Role("ccw_secondary", "roceP2p1s0f1", "enP2p1s0f1np1", 1, True),
)
FABRIC_DOCUMENT_SCHEMA = "sparkring-fabric/v1"
FABRIC_DOCUMENT_VARIABLE = "SIRCL_FABRIC_DOCUMENT"


def roles_from_fabric_document(document: Mapping[str, object]) -> tuple[Role, ...]:
    """The four function roles a ``sparkring-fabric/v1`` document names.

    Every entry of ``positions`` names, under ``ports.<port>.functions.<function>``
    (port ``"0"`` or ``"1"``, function ``primary`` or ``secondary``), the RDMA
    device (``rdma``) and the network interface (``netdev``). A route map names
    devices, and a lane's remote device is the function of the same role on the
    far Spark, so every position must name each role alike; a document whose
    positions differ is refused, naming the position and the role.
    """
    if not isinstance(document, Mapping) or document.get("schema") != FABRIC_DOCUMENT_SCHEMA:
        raise RouteError(f"the fabric document's schema is not {FABRIC_DOCUMENT_SCHEMA}")
    positions = document.get("positions")
    if not isinstance(positions, Sequence) or isinstance(positions, str) or not positions:
        raise RouteError("the fabric document lists no positions")
    roles: tuple[Role, ...] | None = None
    for entry in positions:
        position = entry.get("position") if isinstance(entry, Mapping) else None
        named = []
        for default in DEFAULT_ROLES:
            function = "secondary" if default.secondary else "primary"
            try:
                names = entry["ports"][str(default.port)]["functions"][function]
                device, netdev = names["rdma"], names["netdev"]
            except (KeyError, TypeError, IndexError):
                raise RouteError(f"position {position} of the fabric document does not name the RDMA "
                                 f"device and network interface of port {default.port} {function}") from None
            if not (isinstance(device, str) and device and isinstance(netdev, str) and netdev):
                raise RouteError(f"position {position} of the fabric document names an empty device or "
                                 f"interface for port {default.port} {function}")
            named.append(dataclasses.replace(default, device=device, netdev=netdev))
        if roles is None:
            roles = tuple(named)
        elif tuple(named) != roles:
            differing = next(mine for mine, first in zip(named, roles) if mine != first)
            raise RouteError(f"position {position} of the fabric document names port {differing.port} "
                             f"{'secondary' if differing.secondary else 'primary'} differently from the "
                             "first position; SIRCL needs every Spark of a fabric to name its functions "
                             "alike")
    assert roles is not None
    if len({role.device for role in roles}) != len(roles) or len({role.netdev for role in roles}) != len(roles):
        raise RouteError("the fabric document names one device or interface for two functions")
    return roles


def load_roles(environ: Mapping[str, str] | None = None) -> tuple[Role, ...]:
    """The roles of the document ``SIRCL_FABRIC_DOCUMENT`` names, else :data:`DEFAULT_ROLES`."""
    environ = os.environ if environ is None else environ
    path = (environ.get(FABRIC_DOCUMENT_VARIABLE) or "").strip()
    if not path:
        return DEFAULT_ROLES
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RouteError(f"{FABRIC_DOCUMENT_VARIABLE}={path}: {error}") from None
    return roles_from_fabric_document(document)


# The roles every derivation and check in this process uses, read once at import.
ROLES = load_roles()
_BY_DEVICE = {role.device: role for role in ROLES}
_BY_NAME = {role.name: role for role in ROLES}


def role_at(port: int, secondary: bool) -> Role:
    for role in ROLES:
        if role.port == port and role.secondary == secondary:
            return role
    raise RouteError(f"no function on port {port}")


def role_of(device: str, roles: Mapping[str, Role] | None = None) -> Role | None:
    """The role of an RDMA device name (canonical names unless ``roles`` maps others)."""
    if roles is not None and device in roles:
        return roles[device]
    return _BY_DEVICE.get(device)


# -- fabric --------------------------------------------------------------------

_CABLE = re.compile(r"^\s*(\d+)\.port([01])\s*-\s*(\d+)\.port([01])\s*$")


@dataclasses.dataclass(frozen=True)
class Cable:
    a: int
    a_port: int
    b: int
    b_port: int

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise RouteError(f"cable {self.text} joins a Spark to itself")

    @property
    def text(self) -> str:
        return f"{self.a}.port{self.a_port}-{self.b}.port{self.b_port}"

    @classmethod
    def parse(cls, text: str) -> "Cable":
        match = _CABLE.match(text)
        if match is None:
            raise RouteError(f"cable {text!r} is not written <position>.port<0|1>-<position>.port<0|1>")
        a, a_port, b, b_port = (int(group) for group in match.groups())
        return cls(a, a_port, b, b_port)

    def ends(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.a, self.a_port), (self.b, self.b_port)


@dataclasses.dataclass(frozen=True)
class Step:
    """One cable crossed from (``src``, ``src_port``) to (``dst``, ``dst_port``)."""

    cable: Cable
    src: int
    src_port: int
    dst: int
    dst_port: int

    @property
    def text(self) -> str:
        return f"{self.src}.port{self.src_port}-{self.dst}.port{self.dst_port}"


@dataclasses.dataclass(frozen=True)
class Fabric:
    """The cables a group owns."""

    cables: tuple[Cable, ...]

    def __post_init__(self) -> None:
        seen: dict[tuple[int, int], Cable] = {}
        for cable in self.cables:
            for end in cable.ends():
                if end in seen:
                    raise RouteError(f"port {end[1]} of position {end[0]} has two cables")
                seen[end] = cable

    @classmethod
    def ring(cls, size: int) -> "Fabric":
        """A ring of ``size`` Sparks: position ``i`` port 0 to position ``i+1`` port 1."""
        if size < 2:
            raise RouteError("a ring needs at least two Sparks")
        return cls(tuple(Cable(i, 0, (i + 1) % size, 1) for i in range(size)))

    @classmethod
    def path(cls, positions: Sequence[int]) -> "Fabric":
        """Consecutive Sparks of a ring, each cabled port 0 to the next one's port 1."""
        positions = tuple(positions)
        if len(positions) < 2:
            raise RouteError("a path needs at least two Sparks")
        return cls(tuple(Cable(a, 0, b, 1) for a, b in zip(positions, positions[1:])))

    @classmethod
    def parse(cls, texts: Iterable[str]) -> "Fabric":
        return cls(tuple(Cable.parse(text) for text in texts))

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted({end[0] for cable in self.cables for end in cable.ends()}))

    @property
    def kind(self) -> str:
        """``cycle`` when every Spark has two cables and they close one loop, else ``path``."""
        degree: dict[int, int] = {}
        for cable in self.cables:
            for position, _ in cable.ends():
                degree[position] = degree.get(position, 0) + 1
        if degree and all(count == 2 for count in degree.values()) and len(self.cables) == len(degree):
            return "cycle"
        return "path"

    def cable_at(self, position: int, port: int) -> Cable | None:
        for cable in self.cables:
            if (position, port) in cable.ends():
                return cable
        return None

    def cross(self, position: int, port: int) -> Step | None:
        """Leave ``position`` through ``port``; None when that port has no cable in this fabric."""
        cable = self.cable_at(position, port)
        if cable is None:
            return None
        if (cable.a, cable.a_port) == (position, port):
            return Step(cable, position, port, cable.b, cable.b_port)
        return Step(cable, position, port, cable.a, cable.a_port)

    def shortest_paths(self, a: int, b: int) -> tuple[tuple[Step, ...], ...]:
        """Every shortest cable path from ``a`` to ``b``."""
        if a == b:
            raise RouteError("a lane joins two different Sparks")
        distance = {b: 0}
        queue = deque([b])
        while queue:
            here = queue.popleft()
            for port in (0, 1):
                step = self.cross(here, port)
                if step is not None and step.dst not in distance:
                    distance[step.dst] = distance[here] + 1
                    queue.append(step.dst)
        if a not in distance:
            return ()
        paths: list[tuple[Step, ...]] = []

        def extend(here: int, steps: list[Step]) -> None:
            if here == b:
                paths.append(tuple(steps))
                return
            for port in (0, 1):
                step = self.cross(here, port)
                if step is not None and distance.get(step.dst) == distance[here] - 1:
                    steps.append(step)
                    extend(step.dst, steps)
                    steps.pop()

        extend(a, [])
        return tuple(paths)

    def walk(self, start: int, port: int, target: int, limit: int) -> tuple[Step, ...] | None:
        """Follow the fabric from ``start`` out of ``port`` (relays leave through their other port).

        Returns the steps up to ``target``, or None when the walk leaves the
        fabric or exceeds ``limit`` cables first.
        """
        steps: list[Step] = []
        here, out = start, port
        while len(steps) < limit:
            step = self.cross(here, out)
            if step is None:
                return None
            steps.append(step)
            if step.dst == target:
                return tuple(steps)
            here, out = step.dst, 1 - step.dst_port
        return None


@dataclasses.dataclass(frozen=True)
class Layout:
    """A session's layout identity: its fabric and the fabric position of every rank."""

    fabric: Fabric
    positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if len(set(self.positions)) != len(self.positions):
            raise RouteError("ranks need distinct fabric positions")
        known = set(self.fabric.positions)
        missing = [p for p in self.positions if p not in known]
        if missing:
            raise RouteError(f"positions {missing} are not on the fabric")

    @property
    def world(self) -> int:
        return len(self.positions)

    def identity(self) -> dict[str, object]:
        return {
            "kind": self.fabric.kind,
            "size": len(self.fabric.positions),
            "positions": list(self.positions),
            "cables": [cable.text for cable in self.fabric.cables],
        }

    @classmethod
    def parse(cls, text: str) -> "Layout":
        """``ring:<n>[:<positions>]``, ``path:<first>-<last>[:<positions>]`` or
        ``cables=<cable>,...;positions=<p>,...``."""
        text = text.strip()
        try:
            if text.startswith("cables="):
                cables_part, _, positions_part = text.partition(";")
                fabric = Fabric.parse(cables_part[len("cables="):].split(","))
                if not positions_part.startswith("positions="):
                    raise RouteError("an explicit layout names positions=<p>,...")
                positions = tuple(int(p) for p in positions_part[len("positions="):].split(","))
                return cls(fabric, positions)
            kind, _, rest = text.partition(":")
            span, _, ranks = rest.partition(":")
            if kind == "ring":
                fabric = Fabric.ring(int(span))
            elif kind == "path":
                first, _, last = span.partition("-")
                fabric = Fabric.path(range(int(first), int(last) + 1))
            else:
                raise RouteError(f"layout {text!r} is not ring:, path: or cables=")
            positions = (tuple(int(p) for p in ranks.split(",")) if ranks
                         else fabric.positions)
            return cls(fabric, positions)
        except ValueError as error:
            if isinstance(error, RouteError):
                raise
            raise RouteError(f"layout {text!r} is malformed: {error}") from None


# -- derivation -------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class LaneRoute:
    """One lane from a rank toward a peer."""

    lane: int
    local: Role
    remote: Role
    steps: tuple[Step, ...]

    @property
    def hops(self) -> int:
        return len(self.steps)

    @property
    def relays(self) -> tuple[int, ...]:
        return tuple(step.dst for step in self.steps[:-1])

    @property
    def cables(self) -> tuple[str, ...]:
        return tuple(step.text for step in self.steps)

    def to_json(self) -> dict[str, object]:
        return {
            "lane": self.lane,
            "local": self.local.device,
            "remote": self.remote.device,
            "local_role": self.local.label,
            "remote_role": self.remote.label,
            "hops": self.hops,
            "relays": list(self.relays),
            "cables": list(self.cables),
        }


def derive_lanes(fabric: Fabric, a: int, b: int, lanes: int = 2) -> tuple[LaneRoute, ...]:
    """The lanes from position ``a`` to position ``b`` (derivation rules above)."""
    if lanes not in (1, 2):
        raise RouteError("a session has one or two lanes per peer")
    paths = fabric.shortest_paths(a, b)
    if not paths:
        raise RouteError(f"position {b} cannot be reached from position {a} on this fabric")
    if len(paths) == 1:
        chosen = (paths[0], paths[0])
    elif len(paths) == 2:
        smaller = min(a, b)

        def leaves_smaller_through_port0(path: tuple[Step, ...]) -> bool:
            if smaller == a:
                return path[0].src_port == 0
            return path[-1].dst_port == 0

        path_a = [path for path in paths if leaves_smaller_through_port0(path)]
        if len(path_a) != 1:
            raise RouteError(f"cannot tell path A from path B between positions {a} and {b}")
        path_b = next(path for path in paths if path is not path_a[0])
        chosen = (path_a[0], path_b)
    else:
        raise RouteError(f"{len(paths)} shortest paths between positions {a} and {b}")
    result = []
    for lane in range(lanes):
        path = chosen[lane]
        secondary = lane == 1
        result.append(LaneRoute(lane, role_at(path[0].src_port, secondary),
                                role_at(path[-1].dst_port, secondary), path))
    return tuple(result)


@dataclasses.dataclass(frozen=True)
class GroupRoutes:
    """Every rank's lanes for one session."""

    layout: Layout
    lanes: int
    ranks: tuple[Mapping[int, tuple[LaneRoute, ...]], ...]

    @property
    def world(self) -> int:
        return self.layout.world

    def route_map(self, rank: int) -> dict[int, tuple[str, ...]]:
        """Peer rank to lane device names."""
        return {peer: tuple(lane.local.device for lane in lanes)
                for peer, lanes in sorted(self.ranks[rank].items())}

    def route_text(self, rank: int) -> str:
        return format_peer_routes(self.route_map(rank))

    def lanes_to(self, rank: int, peer: int) -> tuple[LaneRoute, ...]:
        return self.ranks[rank][peer]

    def max_relays(self) -> int:
        return max((len(lane.relays) for rank in self.ranks for lanes in rank.values() for lane in lanes),
                   default=0)


def derive_routes(layout: Layout, lanes: int = 2) -> GroupRoutes:
    """Route maps of every rank of a session on ``layout``."""
    ranks = []
    for rank, here in enumerate(layout.positions):
        ranks.append({peer: derive_lanes(layout.fabric, here, there, lanes)
                      for peer, there in enumerate(layout.positions) if peer != rank})
    return GroupRoutes(layout, lanes, tuple(ranks))


# -- textual form -----------------------------------------------------------------


def parse_peer_routes(text: str) -> dict[int, tuple[str, ...]]:
    """Parse ``<peer>=<device>[/<device>],...``; duplicate peers raise."""
    routes: dict[int, tuple[str, ...]] = {}
    if text is None or not text.strip():
        raise RouteError("the route map is empty")
    for entry in text.split(","):
        entry = entry.strip()
        if not entry:
            raise RouteError(f"route map {text!r} has an empty entry")
        peer_text, separator, devices_text = entry.partition("=")
        if not separator:
            raise RouteError(f"route map entry {entry!r} is not <peer>=<device>[/<device>]")
        try:
            peer = int(peer_text.strip())
        except ValueError:
            raise RouteError(f"route map entry {entry!r} names peer {peer_text.strip()!r}, not a rank") from None
        if peer in routes:
            raise RouteError(f"route map names rank {peer} twice")
        routes[peer] = tuple(device.strip() for device in devices_text.split("/"))
    return routes


def format_peer_routes(routes: Mapping[int, Sequence[str]]) -> str:
    return ",".join(f"{peer}={'/'.join(devices)}" for peer, devices in sorted(routes.items()))


# -- validation -------------------------------------------------------------------


def validate_route_map(
    rank: int,
    world: int,
    routes: Mapping[int, Sequence[str]],
    *,
    layout: Layout | None = None,
    max_relays: int = DEFAULT_MAX_RELAYS,
    available_devices: Iterable[str] | None = None,
    roles: Mapping[str, Role] | None = None,
) -> int:
    """Check one rank's route map; return its lane count ``L``.

    Rules: every other rank named exactly once and the own rank not at all;
    every entry with the same lane count, 1 or 2, of distinct non-empty device
    names; 1 to 4 devices in all, each available when ``available_devices``
    is given; with a layout, every lane stays inside the fabric and crosses at
    most ``max_relays`` relays. Errors name the rank, the entry and the rule.
    """
    if not 0 <= rank < world:
        raise RouteError(f"rank {rank} is outside a session of {world}")
    if rank in routes:
        raise RouteError(f"rank {rank}'s route map names its own rank")
    missing = [peer for peer in range(world) if peer != rank and peer not in routes]
    if missing:
        raise RouteError(f"rank {rank}'s route map has no entry for rank {missing[0]}")
    extra = sorted(peer for peer in routes if not 0 <= peer < world)
    if extra:
        raise RouteError(f"rank {rank}'s route map names rank {extra[0]}, outside the session")
    counts = {len(devices) for devices in routes.values()}
    if len(counts) != 1:
        raise RouteError(f"rank {rank}'s route map entries have different lane counts {sorted(counts)}")
    lanes = counts.pop()
    if not 1 <= lanes <= MAX_LANES:
        raise RouteError(f"rank {rank}'s route map has {lanes} lanes per peer; one or two are supported")
    for peer, devices in sorted(routes.items()):
        if any(not device for device in devices):
            raise RouteError(f"rank {rank}'s entry for rank {peer} has an empty device name")
        if len(set(devices)) != len(devices):
            raise RouteError(f"rank {rank}'s entry for rank {peer} names device {devices[0]} twice")
    named = list(dict.fromkeys(device for _, devices in sorted(routes.items()) for device in devices))
    if not 1 <= len(named) <= MAX_DEVICES:
        raise RouteError(f"rank {rank}'s route map names {len(named)} devices; 1 to {MAX_DEVICES} are supported")
    if available_devices is not None:
        available = set(available_devices)
        for device in named:
            if device not in available:
                raise RouteError(f"rank {rank}'s route map names device {device}, which is not an active "
                                 "RDMA device of this host")
    if layout is not None:
        if layout.world != world:
            raise RouteError(f"layout has {layout.world} ranks, the session {world}")
        start = layout.positions[rank]
        for peer, devices in sorted(routes.items()):
            for lane, device in enumerate(devices):
                role = role_of(device, roles)
                if role is None:
                    raise RouteError(f"rank {rank} lane {lane} toward rank {peer} uses device {device}, "
                                     "whose port and function the layout check cannot tell")
                steps = layout.fabric.walk(start, role.port, layout.positions[peer],
                                           limit=len(layout.fabric.positions))
                if steps is None:
                    raise RouteError(
                        f"rank {rank} lane {lane} toward rank {peer} leaves through {device} (port "
                        f"{role.port}) and crosses a Spark outside the group's fabric")
                if len(steps) - 1 > max_relays:
                    raise RouteError(
                        f"rank {rank} lane {lane} toward rank {peer} crosses {len(steps) - 1} relays; the "
                        f"qualified limit is {max_relays} (SIRCL_MAX_RELAYS raises it, research-only)")
    return lanes


def check_complementary(
    layout: Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    *,
    roles: Mapping[str, Role] | None = None,
) -> list[str]:
    """Problems with the cross-rank pairing of lanes (empty when every lane pairs).

    Lane ``l`` of rank ``r`` toward ``p`` and lane ``l`` of ``p`` toward ``r``
    must cross the same cables in opposite directions with the same function
    class: ``p``'s device must be the function of ``r``'s class on the port
    where ``r``'s lane arrives.
    """
    problems = []
    for rank, routes in enumerate(route_maps):
        for peer, devices in sorted(routes.items()):
            for lane, device in enumerate(devices):
                mine = role_of(device, roles)
                theirs_device = route_maps[peer][rank][lane] if lane < len(route_maps[peer].get(rank, ())) else None
                if mine is None or theirs_device is None:
                    problems.append(f"lane {lane} between ranks {rank} and {peer} cannot be checked")
                    continue
                steps = layout.fabric.walk(layout.positions[rank], mine.port, layout.positions[peer],
                                           limit=len(layout.fabric.positions))
                if steps is None:
                    problems.append(f"lane {lane} of rank {rank} toward rank {peer} does not reach it")
                    continue
                expected = role_at(steps[-1].dst_port, mine.secondary)
                theirs = role_of(theirs_device, roles)
                if theirs != expected:
                    problems.append(
                        f"lane {lane} of rank {rank} leaves through port {mine.port} and arrives on rank "
                        f"{peer}'s port {steps[-1].dst_port} ({expected.device}), but rank {peer} lists "
                        f"{theirs_device} for lane {lane}")
    return problems


def relay_load(routes: GroupRoutes) -> tuple[int, float]:
    """``(lanes through the busiest relay queue, load factor f)``.

    A relay queue is one (relay Spark, function class, direction). When every
    rank sends ``b`` bytes to every peer, split over ``L`` lanes, the busiest
    queue holds about ``f * b`` bytes with ``f`` = lanes / ``L``.
    """
    load: dict[tuple[int, bool, int], int] = {}
    for rank in routes.ranks:
        for lanes in rank.values():
            for lane in lanes:
                for step in lane.steps[:-1]:
                    # Arriving on port 1 continues clockwise (out of port 0).
                    key = (step.dst, lane.local.secondary, step.dst_port)
                    load[key] = load.get(key, 0) + 1
    busiest = max(load.values(), default=0)
    return busiest, busiest / routes.lanes


DEFAULT_FORWARD_WINDOW = 131072
DEFAULT_FORWARD_CHUNK = 32768
DEFAULT_HAIRPIN_QUEUE = 524288
RELAY_QUEUE_SHARE = 0.75


def _lane_steps(layout: Layout, rank: int, peer: int, device: str,
                roles: Mapping[str, Role] | None) -> tuple[Role, tuple[Step, ...]] | None:
    role = role_of(device, roles)
    if role is None:
        return None
    steps = layout.fabric.walk(layout.positions[rank], role.port, layout.positions[peer],
                               limit=len(layout.fabric.positions))
    return (role, steps) if steps is not None else None


def relay_queues(
    layout: Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    *,
    roles: Mapping[str, Role] | None = None,
) -> dict[tuple[int, bool, int], list[tuple[int, int, int]]]:
    """Lanes through every relay hairpin queue of a group.

    Key: (relay position, secondary function, egress port); value: the
    ``(rank, peer, lane)`` of every lane whose packets that queue forwards.
    """
    queues: dict[tuple[int, bool, int], list[tuple[int, int, int]]] = {}
    for rank, routes in enumerate(route_maps):
        for peer, devices in sorted(routes.items()):
            for lane, device in enumerate(devices):
                found = _lane_steps(layout, rank, peer, device, roles)
                if found is None:
                    continue
                role, steps = found
                for step in steps[:-1]:
                    key = (step.dst, role.secondary, 1 - step.dst_port)
                    queues.setdefault(key, []).append((rank, peer, lane))
    return queues


def ring_queues(
    layout: Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    order: Sequence[int],
    *,
    roles: Mapping[str, Role] | None = None,
) -> dict[tuple[int, bool, int], list[tuple[int, int, int]]]:
    """Lanes of the ring over ``order`` (rank ``order[i]`` to ``order[(i + 1) % W]``) through every
    relay hairpin queue, keyed as in :func:`relay_queues`; empty when every ring edge is a cable."""
    queues: dict[tuple[int, bool, int], list[tuple[int, int, int]]] = {}
    world = len(order)
    for i, rank in enumerate(order):
        peer = order[(i + 1) % world]
        if peer == rank:
            continue
        for lane, device in enumerate(route_maps[rank].get(peer, ())):
            found = _lane_steps(layout, rank, peer, device, roles)
            if found is None:
                continue
            role, steps = found
            for step in steps[:-1]:
                key = (step.dst, role.secondary, 1 - step.dst_port)
                queues.setdefault(key, []).append((rank, peer, lane))
    return queues


def ring_window(
    layout: Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    order: Sequence[int],
    *,
    chunk: int = DEFAULT_FORWARD_CHUNK,
    queue_bytes: int = DEFAULT_HAIRPIN_QUEUE,
    roles: Mapping[str, Role] | None = None,
) -> tuple[int, list[str]]:
    """``(window, problems)`` of the ring over ``order``: the bytes each ring lane through relays
    keeps unacknowledged (0 when every ring edge is a cable), and why the ring cannot run.

    During a ring op the ring's lanes are a group's only relayed traffic, so a relay hairpin queue
    holds the bytes of the ring lanes through it; the window is the largest multiple of ``chunk``
    within ``RELAY_QUEUE_SHARE`` of a queue when every queue carries at most one ring lane (relays
    forward each function's lanes through that function's own queue). A queue that carries two
    ring lanes is a problem.
    """
    queues = ring_queues(layout, route_maps, order, roles=roles)
    problems = [f"relay {key[0]} {'secondary' if key[1] else 'primary'} function toward port {key[2]} carries "
                f"ring lanes {members}" for key, members in sorted(queues.items()) if len(members) > 1]
    if not queues:
        return 0, problems
    window = int(RELAY_QUEUE_SHARE * queue_bytes) // chunk * chunk
    if window < chunk:
        problems.append(f"a relay queue of {queue_bytes} bytes holds less than one {chunk}-byte chunk")
    return window, problems


def forward_windows(
    layout: Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    rank: int,
    *,
    max_window: int = DEFAULT_FORWARD_WINDOW,
    chunk: int = DEFAULT_FORWARD_CHUNK,
    queue_bytes: int = DEFAULT_HAIRPIN_QUEUE,
    roles: Mapping[str, Role] | None = None,
) -> list[list[int]]:
    """Forward window in bytes of every lane of ``rank`` (``[peer][lane]``; 0 for direct lanes).

    A lane through relays gets the largest multiple of ``chunk`` that is at
    most ``max_window`` and, for every relay queue it crosses, at most
    ``RELAY_QUEUE_SHARE * queue_bytes / n``, where ``n`` lanes share that queue;
    never less than one chunk. A ``max_window`` of 0 disables windows.
    """
    world = layout.world
    lanes = max((len(devices) for devices in route_maps[rank].values()), default=0)
    table = [[0] * lanes for _ in range(world)]
    if max_window <= 0:
        return table
    if chunk <= 0 or chunk % 16 or max_window < chunk:
        raise RouteError(f"forward chunk {chunk} must be a positive multiple of 16 within the window {max_window}")
    queues = relay_queues(layout, route_maps, roles=roles)
    load: dict[tuple[int, int, int], int] = {}
    for members in queues.values():
        for member in members:
            load[member] = max(load.get(member, 0), len(members))
    for peer, devices in sorted(route_maps[rank].items()):
        for lane in range(len(devices)):
            sharing = load.get((rank, peer, lane), 0)
            if sharing == 0:
                continue
            limit = min(max_window, int(RELAY_QUEUE_SHARE * queue_bytes / sharing))
            table[peer][lane] = max(chunk, limit // chunk * chunk)
    return table


def chain_order(
    layout: Layout,
    route_maps: Sequence[Mapping[int, Sequence[str]]],
    *,
    roles: Mapping[str, Role] | None = None,
) -> tuple[int, ...] | None:
    """The ranks in chain order when the group's ranks form a chain of cable neighbors, else None.

    Every Spark of the fabric must host a rank. On a path the chain starts at the
    end with the smaller position; on a cycle it starts at rank 0 and follows
    port 0, leaving the cable that closes the cycle unused. Every lane between
    consecutive ranks of the chain, in both directions, must be a direct lane
    (no relay).
    """
    fabric = layout.fabric
    if sorted(layout.positions) != list(fabric.positions):
        return None
    rank_at = {position: rank for rank, position in enumerate(layout.positions)}
    if fabric.kind == "path":
        degree: dict[int, int] = {}
        for cable in fabric.cables:
            for position, _ in cable.ends():
                degree[position] = degree.get(position, 0) + 1
        start = min(position for position in fabric.positions if degree.get(position, 0) <= 1)
    else:
        start = layout.positions[0]
    order = [start]
    previous = None
    while len(order) < len(fabric.positions):
        here = order[-1]
        steps = [fabric.cross(here, port) for port in (0, 1)]
        candidates = [step.dst for step in steps if step is not None and step.dst != previous
                      and step.dst not in order]
        if fabric.kind != "path" and len(order) == 1:
            first = fabric.cross(here, 0)
            candidates = [first.dst] if first is not None else []
        if not candidates:
            return None
        previous = here
        order.append(candidates[0])
    ranks = tuple(rank_at[position] for position in order)
    for a, b in zip(ranks, ranks[1:]):
        for source, target in ((a, b), (b, a)):
            devices = route_maps[source].get(target)
            if not devices:
                return None
            for device in devices:
                found = _lane_steps(layout, source, target, device, roles)
                if found is None or len(found[1]) != 1:
                    return None
    return ranks


def isolation_problems(groups: Sequence[GroupRoutes]) -> list[str]:
    """Cables or relays shared by independent groups on one ring (must be empty)."""
    problems = []
    for first in range(len(groups)):
        for second in range(first + 1, len(groups)):
            a, b = groups[first], groups[second]
            cables_a = {step.cable for rank in a.ranks for lanes in rank.values() for lane in lanes
                        for step in lane.steps}
            cables_b = {step.cable for rank in b.ranks for lanes in rank.values() for lane in lanes
                        for step in lane.steps}
            for cable in sorted(cables_a & cables_b, key=lambda c: c.text):
                problems.append(f"groups {first} and {second} both use cable {cable.text}")
            sparks_a = set(a.layout.positions) | {relay for rank in a.ranks for lanes in rank.values()
                                                  for lane in lanes for relay in lane.relays}
            sparks_b = set(b.layout.positions) | {relay for rank in b.ranks for lanes in rank.values()
                                                  for lane in lanes for relay in lane.relays}
            for spark in sorted(sparks_a & sparks_b):
                problems.append(f"groups {first} and {second} both use Spark {spark}")
    return problems
