"""The relay plan of a layout, derived from SIRCL's route module.

Carried paths. For every ordered pair of members of a group the plan carries
every shortest path of the group's fabric (``routes.Fabric.shortest_paths``)
on both function classes. SIRCL's lanes (``routes.derive_routes``) are a
subset: lane 0 is the primary function and lane 1 the secondary function of
one of these paths. On a path of Sparks both sets are equal. On a ring of even
size the plan also carries the other function class of each opposite pair's
two paths, which makes the whole-ring plan the universal relay table: every
Spark reaches all four addresses of every Spark it shares no cable with. Every
lane of ``routes.derive_routes`` is checked to be carried, so this plan and
SIRCL's route maps cannot disagree.

Objects, for a carried path with ``h >= 2`` hops from ``a`` to ``b`` on
function class ``c``:

- an origin route at ``a``: ``b``'s address on the arrival function as a /32
  route, scope link, over ``a``'s network device of class ``c`` on the path's
  first port, source ``a``'s address on that device, and a permanent
  neighbour entry holding the adjacent Spark's MAC on that cable;
- a marker rule at ``a``: the RDMA device of that function rewrites the
  EtherType of RoCE packets to that destination into ``TAG(h - 1)``;
- a relay filter at every intermediate Spark: on the ingress of the device
  that receives the path (class ``c``), match ``TAG(k)`` (``k`` relays left),
  set the next Spark's MAC of class ``c``, rewrite the tag to ``TAG(k - 1)``
  and redirect out of the other port (relay egress ``same`` or ``sibling``,
  :mod:`.layouts`).

``TAG(0)`` is IPv4 (0x0800) and ``TAG(k) = 0x88b4 + k``. Relay filters do not
depend on the sender: one per (Spark, ingress device, tag).

Measured constraints that the plan states and applies:

- a relay's hardware hairpin queue holds 512 KiB, the device maximum, and
  cannot pause its sender, so frames beyond it are dropped. The plan lists
  the SIRCL lanes through every queue, the relay load factor ``f`` and the
  per-peer op bound ``0.75 * 512 KiB / f``;
- relay egress through the sibling Socket Direct function of the egress port
  lowered burst loss to 0.06 % at 2 MiB (:mod:`.layouts`);
- queue pairs use flow label 0 and tags are chosen by destination address,
  never by UDP port: the sender NIC computes the ICRC after the RDMA-TX
  rewrite, so a rewritten UDP field would be covered by the ICRC and could not
  be restored by a relay, and with flow label 0 the UDP source port does not
  identify a lane. The EtherType lies outside the ICRC.

Isolation: routes and tags exist only for destinations in the
same group and only on its members; a relay filter exists only on a member
that lies strictly inside a carried path, and both cables it forwards between
belong to the group. :func:`trace` follows every origin route through the
plan's objects over the physical ring and must reach the destination without
touching a Spark outside the group; :func:`isolation_problems` checks the
rules within and across the groups of a layout.

Everything here is offline. Facts (addresses and MACs) come from the Sparks
(:mod:`.state`), from a facts file (schema ``sparkring-fabric-facts/v1``), or
as symbolic placeholders that make a plan reviewable without them.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import re
from collections.abc import Iterable, Mapping, Sequence

from .. import routes
from .layouts import RELAY_EGRESS, FabricError, FabricLayout, Group

ETH_P_IP = 0x0800
TAG_BASE = 0x88B4
MAX_TAG = 7                                  # tags TAG(1)..TAG(7); relay filter preferences 11..17
FILTER_PREF_BASE = 10
RESERVED_PREFS = range(FILTER_PREF_BASE + 1, FILTER_PREF_BASE + MAX_TAG + 1)
HAIRPIN_QUEUE_BYTES = routes.DEFAULT_HAIRPIN_QUEUE
RELAY_QUEUE_SHARE = routes.RELAY_QUEUE_SHARE
FABRIC_NETDEVS = tuple(role.netdev for role in routes.ROLES)
RDMA_DEVICES = tuple(role.device for role in routes.ROLES)
FACTS_SCHEMA = "sparkring-fabric-facts/v1"
PLAN_SCHEMA = "sparkring-fabric-plan/v1"
_ROLE_BY_NETDEV = {role.netdev: role for role in routes.ROLES}
_MAC = re.compile(r"^[0-9a-f]{2}(?::[0-9a-f]{2}){5}$")


def tag(k: int) -> int:
    """The EtherType of a frame with ``k`` relays left."""
    if not 0 <= k <= MAX_TAG:
        raise FabricError(f"relay count {k} is outside the tags 0 to {MAX_TAG}")
    return ETH_P_IP if k == 0 else TAG_BASE + k


def relays_left(ethertype: int) -> int | None:
    """``k`` for ``TAG(k)``, or None for an EtherType that is not a relay tag."""
    if ethertype == ETH_P_IP:
        return 0
    k = ethertype - TAG_BASE
    return k if 1 <= k <= MAX_TAG else None


def role_of_netdev(netdev: str) -> routes.Role:
    role = _ROLE_BY_NETDEV.get(netdev)
    if role is None:
        raise FabricError(f"{netdev} is not one of the fabric network devices {', '.join(FABRIC_NETDEVS)}")
    return role


def class_name(secondary: bool) -> str:
    return "secondary" if secondary else "primary"


def _address_key(text: str) -> tuple:
    try:
        return (0, int(ipaddress.IPv4Address(text)))
    except ValueError:
        return (1, text)


# -- facts --------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Port:
    """One fabric function's network device on one Spark: its MAC and IPv4 address."""

    netdev: str
    mac: str
    address: str
    prefixlen: int | None = None

    def network(self) -> ipaddress.IPv4Network | None:
        if self.prefixlen is None:
            return None
        try:
            return ipaddress.IPv4Interface(f"{self.address}/{self.prefixlen}").network
        except ValueError:
            return None

    def to_json(self) -> dict[str, object]:
        address = self.address if self.prefixlen is None else f"{self.address}/{self.prefixlen}"
        return {"mac": self.mac, "address": address}


@dataclasses.dataclass(frozen=True)
class SparkFacts:
    position: int
    name: str
    ports: tuple[Port, ...]

    def port(self, netdev: str) -> Port | None:
        for port in self.ports:
            if port.netdev == netdev:
                return port
        return None


class Facts:
    """Addresses and MACs of the fabric functions of the Sparks a plan uses."""

    def __init__(self, sparks: Iterable[SparkFacts], *, symbolic: bool = False) -> None:
        self.symbolic = symbolic
        self._sparks = {spark.position: spark for spark in sparks}
        self._addresses: dict[str, list[tuple[int, str]]] = {}
        self._macs: dict[str, list[tuple[int, str]]] = {}
        for spark in self._sparks.values():
            for port in spark.ports:
                self._addresses.setdefault(port.address, []).append((spark.position, port.netdev))
                self._macs.setdefault(port.mac.lower(), []).append((spark.position, port.netdev))

    @property
    def positions(self) -> tuple[int, ...]:
        return tuple(sorted(self._sparks))

    def has(self, position: int) -> bool:
        return position in self._sparks

    def spark(self, position: int) -> SparkFacts:
        if position not in self._sparks:
            raise FabricError(f"no facts for the Spark at ring position {position}")
        return self._sparks[position]

    def name(self, position: int) -> str:
        return self._sparks[position].name if position in self._sparks else f"position {position}"

    def port(self, position: int, netdev: str) -> Port:
        port = self.spark(position).port(netdev)
        if port is None:
            raise FabricError(f"{self.name(position)} has no address or MAC for {netdev}")
        return port

    def find_address(self, address: str) -> tuple[int, str] | None:
        owners = self._addresses.get(address, [])
        return owners[0] if len(owners) == 1 else None

    def find_mac(self, mac: str) -> tuple[int, str] | None:
        owners = self._macs.get(mac.lower(), [])
        return owners[0] if len(owners) == 1 else None

    def problems(self) -> list[str]:
        """Addresses or MACs that more than one function holds."""
        found = []
        for label, table in (("address", self._addresses), ("MAC", self._macs)):
            for value, owners in sorted(table.items()):
                if len(owners) > 1:
                    holders = ", ".join(f"{self.name(position)} {netdev}" for position, netdev in owners)
                    found.append(f"{label} {value} is held by {holders}")
        return found

    @classmethod
    def symbolic_for(cls, names: Sequence[str]) -> "Facts":
        """Placeholders ``<name:netdev>`` and ``<name:netdev:mac>`` for every Spark of a ring."""
        return cls((SparkFacts(position, name, tuple(Port(netdev, f"<{name}:{netdev}:mac>", f"<{name}:{netdev}>")
                                                     for netdev in FABRIC_NETDEVS))
                    for position, name in enumerate(names)), symbolic=True)

    @classmethod
    def from_json(cls, document: Mapping[str, object]) -> "Facts":
        if document.get("schema") != FACTS_SCHEMA:
            raise FabricError(f"facts schema must be {FACTS_SCHEMA}")
        sparks = []
        for entry in document.get("sparks", []):
            position, name = int(entry["position"]), str(entry["name"])
            ports = []
            for netdev, values in sorted(dict(entry.get("ports", {})).items()):
                role_of_netdev(netdev)
                ports.append(port_from_text(netdev, str(values["mac"]), str(values["address"]), owner=name))
            sparks.append(SparkFacts(position, name, tuple(ports)))
        return cls(sparks)

    def to_json(self) -> dict[str, object]:
        return {"schema": FACTS_SCHEMA, "sparks": [
            {"position": spark.position, "name": spark.name,
             "ports": {port.netdev: port.to_json() for port in spark.ports}}
            for spark in sorted(self._sparks.values(), key=lambda s: s.position)]}


def port_from_text(netdev: str, mac: str, address: str, *, owner: str) -> Port:
    """A validated :class:`Port` from a MAC and ``a.b.c.d`` or ``a.b.c.d/len``."""
    mac = mac.strip().lower()
    if not _MAC.match(mac):
        raise FabricError(f"{owner} {netdev}: {mac!r} is not a MAC address")
    try:
        if "/" in address:
            interface = ipaddress.IPv4Interface(address.strip())
            return Port(netdev, mac, str(interface.ip), interface.network.prefixlen)
        return Port(netdev, mac, str(ipaddress.IPv4Address(address.strip())))
    except ValueError:
        raise FabricError(f"{owner} {netdev}: {address!r} is not an IPv4 address") from None


# -- plan objects -----------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class OriginRoute:
    """A /32 route, its permanent neighbour and its marker rule at the Spark where a path starts."""

    destination: str
    netdev: str
    device: str
    source: str
    next_mac: str
    tag: int
    peer: int
    peer_netdev: str
    hops: int
    relays: tuple[int, ...]
    cables: tuple[str, ...]              # cables crossed, in the direction of travel
    owned_cables: tuple[str, ...]        # the same cables in canonical spelling
    lanes: tuple[str, ...] = ()          # SIRCL lanes it carries: "rank r->p lane l"

    def to_json(self) -> dict[str, object]:
        return {"destination": self.destination, "netdev": self.netdev, "device": self.device,
                "source": self.source, "next_mac": self.next_mac, "tag": f"0x{self.tag:04x}", "peer": self.peer,
                "peer_netdev": self.peer_netdev, "hops": self.hops, "relays": list(self.relays),
                "cables": list(self.cables), "sircl_lanes": list(self.lanes)}


@dataclasses.dataclass(frozen=True)
class RelayFilter:
    """A hardware ingress filter of a relay: ``TAG(k)`` in, ``TAG(k - 1)`` out of the other port."""

    in_netdev: str
    k: int
    out_netdev: str
    next_mac: str
    next_position: int
    next_netdev: str
    direction: str                       # cw (out of port 0) or ccw (out of port 1)
    lane_class: str                      # primary or secondary: the class of the lanes it relays

    @property
    def pref(self) -> int:
        return FILTER_PREF_BASE + self.k

    @property
    def handle(self) -> int:
        return self.k

    @property
    def protocol(self) -> int:
        return tag(self.k)

    @property
    def new_type(self) -> int:
        return tag(self.k - 1)

    def to_json(self) -> dict[str, object]:
        return {"in_netdev": self.in_netdev, "k": self.k, "pref": self.pref, "protocol": f"0x{self.protocol:04x}",
                "new_type": f"0x{self.new_type:04x}", "out_netdev": self.out_netdev, "next_mac": self.next_mac,
                "next_position": self.next_position, "next_netdev": self.next_netdev, "direction": self.direction,
                "lane_class": self.lane_class}


@dataclasses.dataclass(frozen=True)
class MarkerRules:
    """The RDMA-TX tag rules of one RDMA device: (destination, EtherType) pairs."""

    device: str
    rules: tuple[tuple[str, int], ...]

    def to_json(self) -> dict[str, object]:
        return {"device": self.device, "rules": {destination: f"0x{value:04x}" for destination, value in self.rules}}


@dataclasses.dataclass(frozen=True)
class SparkPlan:
    position: int
    name: str
    group: str
    routes: tuple[OriginRoute, ...]
    filters: tuple[RelayFilter, ...]

    @property
    def markers(self) -> tuple[MarkerRules, ...]:
        by_device: dict[str, list[tuple[str, int]]] = {}
        for route in self.routes:
            by_device.setdefault(route.device, []).append((route.destination, route.tag))
        return tuple(MarkerRules(device, tuple(sorted(rules, key=lambda rule: _address_key(rule[0]))))
                     for device, rules in sorted(by_device.items(), key=lambda item: RDMA_DEVICES.index(item[0])))

    def marker(self, device: str) -> MarkerRules | None:
        for rules in self.markers:
            if rules.device == device:
                return rules
        return None

    @property
    def filter_netdevs(self) -> tuple[str, ...]:
        return tuple(netdev for netdev in FABRIC_NETDEVS if any(f.in_netdev == netdev for f in self.filters))

    def route_to(self, destination: str) -> OriginRoute | None:
        for route in self.routes:
            if route.destination == destination:
                return route
        return None

    def filter_for(self, netdev: str, k: int) -> RelayFilter | None:
        for relay_filter in self.filters:
            if relay_filter.in_netdev == netdev and relay_filter.k == k:
                return relay_filter
        return None

    def to_json(self) -> dict[str, object]:
        return {"position": self.position, "name": self.name, "group": self.group,
                "routes": [route.to_json() for route in self.routes],
                "filters": [relay_filter.to_json() for relay_filter in self.filters],
                "markers": [rules.to_json() for rules in self.markers]}


@dataclasses.dataclass(frozen=True)
class RelayQueue:
    """One hairpin queue of a relay (ingress device to egress device) and the SIRCL lanes through it."""

    position: int
    in_netdev: str
    out_netdev: str
    lanes: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class GroupPlan:
    group: Group
    layout: routes.Layout
    sircl: routes.GroupRoutes
    relay_egress: str
    max_relays: int
    sparks: tuple[SparkPlan, ...]
    queues: tuple[RelayQueue, ...]
    busiest_queue: int
    load_factor: float
    digest: str

    @property
    def per_peer_bytes(self) -> int | None:
        """The largest per-peer op size that keeps the busiest relay queue at 75 %; None without relays."""
        if not self.load_factor:
            return None
        return int(RELAY_QUEUE_SHARE * HAIRPIN_QUEUE_BYTES / self.load_factor)

    def spark(self, position: int) -> SparkPlan | None:
        for spark in self.sparks:
            if spark.position == position:
                return spark
        return None

    def to_json(self) -> dict[str, object]:
        return {
            "group": self.group.label, "members": list(self.group.members), "cables": list(self.group.cables),
            "relay_egress": self.relay_egress, "max_relays": self.max_relays, "digest": self.digest,
            "route_maps": [self.sircl.route_text(rank) for rank in range(self.sircl.world)],
            "relay_queues": [{"position": queue.position, "in": queue.in_netdev, "out": queue.out_netdev,
                              "sircl_lanes": list(queue.lanes)} for queue in self.queues],
            "busiest_queue_lanes": self.busiest_queue, "load_factor": self.load_factor,
            "per_peer_bytes": self.per_peer_bytes, "sparks": [spark.to_json() for spark in self.sparks],
        }


@dataclasses.dataclass(frozen=True)
class LayoutPlan:
    layout: FabricLayout
    groups: tuple[GroupPlan, ...]
    symbolic: bool

    def find(self, position: int) -> tuple[GroupPlan, SparkPlan] | None:
        for group in self.groups:
            spark = group.spark(position)
            if spark is not None:
                return group, spark
        return None

    def to_json(self) -> dict[str, object]:
        return {"schema": PLAN_SCHEMA, "layout": self.layout.name, "ring_size": self.layout.ring_size,
                "relay_egress": self.layout.relay_egress, "symbolic_facts": self.symbolic,
                "constraints": list(CONSTRAINTS), "groups": [group.to_json() for group in self.groups]}


CONSTRAINTS = (
    "relay hairpin queue: 512 KiB per (relay, ingress device, egress device), the device maximum; it cannot "
    "pause its sender, so frames beyond it are dropped",
    "relay egress: sibling Socket Direct function of the egress port lowered burst loss to 0.06 % at 2 MiB; "
    "the whole-ring layout keeps same-function egress",
    "queue pairs use flow label 0",
    "tags are EtherTypes chosen by destination address, never by UDP port (the sender NIC computes the ICRC "
    "after the RDMA-TX rewrite)",
)


# -- derivation ---------------------------------------------------------------------------------


def _carry(path: tuple[routes.Step, ...], secondary: bool, facts: Facts, relay_egress: str,
           routes_at: dict[int, dict[str, OriginRoute]], filters_at: dict[int, dict[tuple[str, int], RelayFilter]],
           label: str) -> None:
    hops = len(path)
    origin, target = path[0].src, path[-1].dst
    local = routes.role_at(path[0].src_port, secondary)
    remote = routes.role_at(path[-1].dst_port, secondary)
    first = routes.role_at(path[0].dst_port, secondary)
    route = OriginRoute(
        destination=facts.port(target, remote.netdev).address, netdev=local.netdev, device=local.device,
        source=facts.port(origin, local.netdev).address, next_mac=facts.port(path[0].dst, first.netdev).mac,
        tag=tag(hops - 1), peer=target, peer_netdev=remote.netdev, hops=hops,
        relays=tuple(step.dst for step in path[:-1]), cables=tuple(step.text for step in path),
        owned_cables=tuple(step.cable.text for step in path))
    existing = routes_at[origin].get(route.destination)
    if existing is not None and existing != route:
        raise FabricError(f"group {label}: two paths from Spark {origin} reach {route.destination} differently")
    routes_at[origin][route.destination] = route
    for index in range(hops - 1):
        arrive, leave = path[index], path[index + 1]
        relay = arrive.dst
        if leave.src != relay or leave.src_port != 1 - arrive.dst_port:
            raise FabricError(f"group {label}: the path {route.cables} does not pass through Spark {relay}")
        in_role = routes.role_at(arrive.dst_port, secondary)
        out_role = routes.role_at(leave.src_port, secondary if relay_egress == "same" else not secondary)
        next_role = routes.role_at(leave.dst_port, secondary)
        relay_filter = RelayFilter(
            in_netdev=in_role.netdev, k=hops - 1 - index, out_netdev=out_role.netdev,
            next_mac=facts.port(leave.dst, next_role.netdev).mac, next_position=leave.dst,
            next_netdev=next_role.netdev, direction="cw" if leave.src_port == 0 else "ccw",
            lane_class=class_name(secondary))
        key = (relay_filter.in_netdev, relay_filter.k)
        known = filters_at[relay].get(key)
        if known is not None and known != relay_filter:
            raise FabricError(f"group {label}: Spark {relay} would need two relay filters for TAG({relay_filter.k}) "
                              f"on {relay_filter.in_netdev}")
        filters_at[relay][key] = relay_filter


def _digest(label: str, relay_egress: str, sparks: Sequence[SparkPlan]) -> str:
    canonical = json.dumps({"group": label, "relay_egress": relay_egress,
                            "sparks": [spark.to_json() for spark in sparks]}, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def build_group_plan(group: Group, facts: Facts, *, relay_egress: str,
                     max_relays: int = routes.DEFAULT_MAX_RELAYS) -> GroupPlan:
    """The relay plan of one group (module docstring), checked by :func:`trace` on every origin route."""
    if relay_egress not in RELAY_EGRESS:
        raise FabricError(f"relay egress must be one of {', '.join(RELAY_EGRESS)}")
    if not 0 <= max_relays <= MAX_TAG:
        raise FabricError(f"the relay limit must be between 0 and {MAX_TAG} (one tag per relay count)")
    layout = group.layout()
    members = layout.positions
    routes_at: dict[int, dict[str, OriginRoute]] = {position: {} for position in members}
    filters_at: dict[int, dict[tuple[str, int], RelayFilter]] = {position: {} for position in members}
    for origin in members:
        for target in members:
            if origin == target:
                continue
            for path in layout.fabric.shortest_paths(origin, target):
                if len(path) < 2:
                    continue
                if len(path) - 1 > max_relays:
                    raise FabricError(
                        f"group {group.label}: Spark {target} is {len(path)} cables from Spark {origin}, "
                        f"{len(path) - 1} relays; the qualified limit is {max_relays} ("
                        "--max-relays raises it, research-only)")
                for secondary in (False, True):
                    _carry(path, secondary, facts, relay_egress, routes_at, filters_at, group.label)
    sircl = routes.derive_routes(layout, 2)
    lanes_of: dict[tuple[int, str], list[str]] = {}
    for rank, peers in enumerate(sircl.ranks):
        origin = layout.positions[rank]
        for peer, lanes in sorted(peers.items()):
            for lane in lanes:
                if lane.hops < 2:
                    continue
                destination = facts.port(layout.positions[peer], lane.remote.netdev).address
                route = routes_at[origin].get(destination)
                if (route is None or route.netdev != lane.local.netdev or route.cables != lane.cables
                        or route.tag != tag(len(lane.relays))):
                    raise FabricError(f"group {group.label}: SIRCL lane {lane.lane} from rank {rank} to rank {peer} "
                                      f"({lane.local.device}, {lane.cables}) is not carried by the relay plan")
                lanes_of.setdefault((origin, destination), []).append(f"rank {rank}->{peer} lane {lane.lane}")
    sparks = []
    for position in group.order:
        planned = sorted(routes_at[position].values(),
                         key=lambda r: (FABRIC_NETDEVS.index(r.netdev), r.hops, _address_key(r.destination)))
        planned = [dataclasses.replace(route, lanes=tuple(lanes_of.get((position, route.destination), ())))
                   for route in planned]
        filters = sorted(filters_at[position].values(), key=lambda f: (FABRIC_NETDEVS.index(f.in_netdev), f.k))
        sparks.append(SparkPlan(position, facts.name(position), group.label, tuple(planned), tuple(filters)))
    by_position = {spark.position: spark for spark in sparks}
    for spark in sparks:
        for route in spark.routes:
            result = trace(by_position, facts, group.ring_size, spark.position, route.destination)
            if result.outcome != "delivered" or result.hops[-1] != (route.peer, route.peer_netdev):
                raise FabricError(f"group {group.label}: the route from {spark.name} to {route.destination} does "
                                  f"not deliver through the plan's relays ({result.outcome}: {result.detail})")
            outside = [position for position, _ in result.hops if position not in members]
            if outside:
                raise FabricError(f"group {group.label}: the route from {spark.name} to {route.destination} "
                                  f"crosses Sparks {outside} outside the group")
    maps = [sircl.route_map(rank) for rank in range(layout.world)]
    queues = []
    for (relay, secondary, egress_port), members_of_queue in sorted(routes.relay_queues(layout, maps).items()):
        in_role = routes.role_at(1 - egress_port, secondary)
        out_role = routes.role_at(egress_port, secondary if relay_egress == "same" else not secondary)
        queues.append(RelayQueue(relay, in_role.netdev, out_role.netdev,
                                 tuple(f"rank {r}->{p} lane {lane}" for r, p, lane in members_of_queue)))
    busiest, load = routes.relay_load(sircl)
    return GroupPlan(group, layout, sircl, relay_egress, max_relays, tuple(sparks), tuple(queues), busiest, load,
                     _digest(group.label, relay_egress, sparks))


def build_layout_plan(layout: FabricLayout, facts: Facts, *, groups: Sequence[Group] | None = None,
                      max_relays: int = routes.DEFAULT_MAX_RELAYS) -> LayoutPlan:
    """Plans of the layout's groups (or of ``groups``), checked for isolation across the whole layout."""
    problems = layout_isolation_problems(layout)
    chosen = tuple(groups) if groups is not None else layout.groups
    plans = tuple(build_group_plan(group, facts, relay_egress=layout.relay_egress, max_relays=max_relays)
                  for group in chosen)
    problems += isolation_problems(plans)
    if problems:
        raise FabricError("isolation: " + "; ".join(problems))
    return LayoutPlan(layout, plans, facts.symbolic)


# -- checks ----------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Trace:
    """Where a frame sent on an origin route went: ``delivered``, ``dropped``, ``misdelivered``,
    ``unrouted`` or ``looped``; ``hops`` lists the (position, receiving device) of every cable crossed."""

    outcome: str
    hops: tuple[tuple[int, str], ...]
    detail: str = ""


def trace(sparks: Mapping[int, SparkPlan], facts: Facts, ring_size: int, origin: int, destination: str,
          *, limit: int = 2 * MAX_TAG) -> Trace:
    """Follow a frame from ``origin`` to ``destination`` through the plan's objects on the physical ring.

    Port 0 of position ``i`` is cabled to port 1 of position ``i + 1``. A frame
    is received by the function whose MAC it carries on the far end of the
    cable; a tagged frame continues only through a relay filter for its tag.
    """
    spark = sparks.get(origin)
    route = spark.route_to(destination) if spark is not None else None
    if route is None:
        return Trace("unrouted", (), f"no origin route to {destination} at position {origin}")
    mac, ethertype, here, out_netdev = route.next_mac, route.tag, origin, route.netdev
    hops: list[tuple[int, str]] = []
    for _ in range(limit):
        port = role_of_netdev(out_netdev).port
        there = (here + 1) % ring_size if port == 0 else (here - 1) % ring_size
        receiver = facts.find_mac(mac)
        if receiver is None or receiver[0] != there or role_of_netdev(receiver[1]).port != 1 - port:
            return Trace("misdelivered", tuple(hops), f"a frame for {mac} leaves position {here} port {port}; "
                         f"the cable ends at position {there} port {1 - port}")
        hops.append(receiver)
        netdev = receiver[1]
        k = relays_left(ethertype)
        if k == 0:
            if facts.port(there, netdev).address == destination:
                return Trace("delivered", tuple(hops))
            return Trace("misdelivered", tuple(hops), f"IPv4 frame for {destination} reached {netdev} at {there}")
        relay = sparks.get(there)
        relay_filter = relay.filter_for(netdev, k) if (relay is not None and k is not None) else None
        if relay_filter is None:
            return Trace("dropped", tuple(hops), f"no relay filter for EtherType 0x{ethertype:04x} on {netdev} "
                         f"at position {there}")
        mac, ethertype, here, out_netdev = relay_filter.next_mac, tag(k - 1), there, relay_filter.out_netdev
    return Trace("looped", tuple(hops), f"more than {limit} cables")


def isolation_problems(plans: Sequence[GroupPlan]) -> list[str]:
    """Breaks of the group isolation rules within and across group plans; empty when isolated."""
    problems = []
    for plan in plans:
        members = set(plan.group.members)
        owned = set(plan.group.cables)
        fabric = plan.layout.fabric
        for spark in plan.sparks:
            if spark.position not in members:
                problems.append(f"{plan.group.label}: {spark.name} is not a member but has objects")
            for route in spark.routes:
                if route.peer not in members:
                    problems.append(f"{plan.group.label}: {spark.name} routes to {route.destination} of Spark "
                                    f"{route.peer}, outside the group")
                if not set(route.relays) <= members or not set(route.owned_cables) <= owned:
                    problems.append(f"{plan.group.label}: the route from {spark.name} to {route.destination} "
                                    f"crosses {route.cables}, outside the group's cables {sorted(owned)}")
            for relay_filter in spark.filters:
                for netdev in (relay_filter.in_netdev, relay_filter.out_netdev):
                    port = role_of_netdev(netdev).port
                    if fabric.cable_at(spark.position, port) is None:
                        problems.append(f"{plan.group.label}: {spark.name} relays through {netdev} (port {port}), "
                                        "whose cable the group does not own")
                if relay_filter.next_position not in members:
                    problems.append(f"{plan.group.label}: {spark.name} relays toward Spark "
                                    f"{relay_filter.next_position}, outside the group")
    for first in range(len(plans)):
        for second in range(first + 1, len(plans)):
            a, b = plans[first], plans[second]
            shared = set(a.group.members) & set(b.group.members)
            if shared:
                problems.append(f"{a.group.label} and {b.group.label} share Sparks {sorted(shared)}")
            used_a = {cable for spark in a.sparks for route in spark.routes for cable in route.owned_cables}
            used_b = {cable for spark in b.sparks for route in spark.routes for cable in route.owned_cables}
            for cable in sorted((set(a.group.cables) | used_a) & (set(b.group.cables) | used_b)):
                problems.append(f"{a.group.label} and {b.group.label} both use cable {cable}")
    problems += routes.isolation_problems([plan.sircl for plan in plans])
    return problems


def layout_isolation_problems(layout: FabricLayout) -> list[str]:
    """SIRCL's lane isolation check (``routes.isolation_problems``) over every group of a layout."""
    return routes.isolation_problems([routes.derive_routes(group.layout(), 2) for group in layout.groups])


# -- rendering ----------------------------------------------------------------------------------------


def render_text(plan: LayoutPlan, facts: Facts, *, details: bool = True) -> str:
    layout = plan.layout
    lines = [f"layout {layout.name} on a ring of {layout.ring_size}: {len(plan.groups)} group(s) planned of "
             f"{len(layout.groups)}, relay egress {layout.relay_egress}"
             + (" (symbolic facts: placeholders stand for addresses and MACs)" if plan.symbolic else "")]
    for note in CONSTRAINTS:
        lines.append(f"  constraint: {note}")
    for group in plan.groups:
        counts = (sum(len(s.routes) for s in group.sparks), sum(len(s.filters) for s in group.sparks),
                  sum(len(s.markers) for s in group.sparks))
        lines.append(f"group {group.group.label}: Sparks {list(group.group.members)}, cables "
                     f"{list(group.group.cables) or 'none'}; {counts[0]} routes, {counts[1]} relay filters, "
                     f"{counts[2]} markers; plan {group.digest}")
        if group.per_peer_bytes is None:
            lines.append("  relay queues: none (no lane crosses a relay)")
        else:
            lines.append(f"  relay queues: busiest {group.busiest_queue} SIRCL lanes, load factor "
                         f"{group.load_factor:g}; keep per-peer op sizes at or below {group.per_peer_bytes} bytes "
                         f"({RELAY_QUEUE_SHARE:.0%} of {HAIRPIN_QUEUE_BYTES >> 10} KiB / {group.load_factor:g})")
        for rank in range(group.sircl.world):
            lines.append(f"  rank {rank} (position {group.layout.positions[rank]}): "
                         f"SIRCL_PEER_ROUTES={group.sircl.route_text(rank)}")
        if not details:
            continue
        for queue in group.queues:
            lines.append(f"  queue {facts.name(queue.position)} {queue.in_netdev} -> {queue.out_netdev}: "
                         f"{len(queue.lanes)} SIRCL lanes")
        for spark in group.sparks:
            lines.append(f"  {spark.name} (position {spark.position}): {len(spark.routes)} routes, "
                         f"{len(spark.filters)} relay filters, {len(spark.markers)} markers")
            for route in spark.routes:
                carried = f"; {', '.join(route.lanes)}" if route.lanes else "; not a SIRCL lane"
                lines.append(f"    route {route.destination}/32 dev {route.netdev} src {route.source} neighbour "
                             f"{route.next_mac} -> {facts.name(route.peer)} {route.peer_netdev}, {route.hops} hops "
                             f"through {list(route.relays)}, tag 0x{route.tag:04x}{carried}")
            for rules in spark.markers:
                lines.append(f"    marker {rules.device}: "
                             + " ".join(f"{destination}=0x{value:04x}" for destination, value in rules.rules))
            for relay_filter in spark.filters:
                lines.append(f"    filter {relay_filter.in_netdev} pref {relay_filter.pref} "
                             f"0x{relay_filter.protocol:04x} -> 0x{relay_filter.new_type:04x}, dst {relay_filter.next_mac} "
                             f"({facts.name(relay_filter.next_position)} {relay_filter.next_netdev}), out "
                             f"{relay_filter.out_netdev} ({relay_filter.direction}, {relay_filter.lane_class} lanes)")
    return "\n".join(lines)
