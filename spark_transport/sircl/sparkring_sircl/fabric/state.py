"""What a Spark holds: the parsed output of the installer's read script.

The read script (:func:`.commands.read_script`) prints ``@@<section>`` blocks:
the host name, the iproute2 versions, ``ip -j addr show``, the network device
of each RDMA device, ``ip -j -d route show table main``,
``ip -j neigh show nud permanent``, ``tc -j qdisc show`` and
``tc -j filter show ... ingress`` of the four fabric network devices, the
marker processes (``pgrep -a -x``), the marker executable's SHA-256, the head
of every marker log and the installer's record. :func:`parse` turns them into
a :class:`HostState`.

Classification of what the host holds:

- an *owned* route or neighbour carries routing protocol 82;
- a *relay* route is a /32 route, scope link, without gateway, over a fabric
  network device to an address outside that device's own subnet, and a relay
  neighbour is a permanent neighbour entry on a fabric device for such an
  address. Relay objects without the mark are *unowned*: some other tooling
  installed them. ``--adopt`` lets ``up`` and ``down`` take them over;
- relay filters are the filters at the reserved preferences 11 to 17 of chain
  0 on the ingress of a fabric device; marker processes are the processes
  named after the marker executable. Both are the installer's by their
  reserved identifiers.

The parsers accept the JSON spellings of iproute2 releases 5.x and 6.x: a
filter protocol as a name, ``[decimal]`` or hexadecimal; a handle as a number
or ``0x`` string; pedit keys under either mask convention.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
from collections.abc import Mapping

from . import commands
from .layouts import FabricError
from .plan import (ETH_P_IP, FABRIC_NETDEVS, RDMA_DEVICES, RESERVED_PREFS, Port, RelayFilter, port_from_text,
                   relays_left)

_PROTOCOL_NAMES = {
    "unspec": 0, "redirect": 1, "kernel": 2, "boot": 3, "static": 4, "gated": 8, "ra": 9, "mrt": 10, "zebra": 11,
    "bird": 12, "dnrouted": 13, "xorp": 14, "ntk": 15, "dhcp": 16, "mrouted": 17, "keepalived": 18, "babel": 42,
    "openr": 99, "bgp": 186, "isis": 187, "ospf": 188, "rip": 189, "eigrp": 192,
}
_ETHERTYPE_NAMES = {"ip": 0x0800, "ipv4": 0x0800, "arp": 0x0806, "rarp": 0x8035, "ipv6": 0x86DD, "802.1q": 0x8100,
                    "802.1ad": 0x88A8, "all": 0x0003, "lldp": 0x88CC, "mpls_uc": 0x8847, "mpls_mc": 0x8848}


def _int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 0)
        except ValueError:
            return None
    return None


def _hex(value: object) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, str):
        try:
            return int(value.strip(), 16)
        except ValueError:
            return None
    return None


def protocol_number(text: object) -> int | None:
    """A routing protocol from iproute2's name or number; None for a name this table does not know."""
    if isinstance(text, int):
        return text
    if text is None:
        return None
    word = str(text).strip().lower()
    if word.isdigit():
        return int(word)
    return _PROTOCOL_NAMES.get(word)


def ethertype(text: object) -> int | None:
    """A filter protocol as tc prints it: a name, ``[34997]``, ``0x88b5`` or a number."""
    if isinstance(text, int) and not isinstance(text, bool):
        return text
    if not isinstance(text, str):
        return None
    word = text.strip().lower()
    if word.startswith("[") and word.endswith("]") and word[1:-1].isdigit():
        return int(word[1:-1])
    if word.startswith("0x"):
        return _hex(word)
    if word in _ETHERTYPE_NAMES:
        return _ETHERTYPE_NAMES[word]
    if word.isdigit():
        return int(word)
    return None


def _key_ethertype(text: object) -> int | None:
    """Flower's ``eth_type`` key: ``ipv4``, ``88b5`` (hexadecimal) and so on."""
    if isinstance(text, str) and len(text.strip()) == 4:
        return _hex(text)
    return ethertype(text)


# -- entries ---------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Link:
    name: str
    mac: str
    addresses: tuple[tuple[str, int], ...]     # IPv4 (address, prefix length)
    operstate: str

    def networks(self) -> tuple[ipaddress.IPv4Network, ...]:
        return tuple(ipaddress.IPv4Interface(f"{address}/{length}").network for address, length in self.addresses)


@dataclasses.dataclass(frozen=True)
class RouteEntry:
    destination: str
    prefixlen: int
    dev: str
    source: str | None
    scope: str
    protocol: int | None
    protocol_text: str
    metric: int
    gateway: str | None
    kind: str

    @property
    def owned(self) -> bool:
        return self.protocol == commands.ROUTE_PROTOCOL

    def text(self) -> str:
        parts = [f"{self.destination}/{self.prefixlen}"]
        if self.gateway:
            parts.append(f"via {self.gateway}")
        parts.append(f"dev {self.dev}")
        if self.source:
            parts.append(f"src {self.source}")
        parts.append(f"scope {self.scope} proto {self.protocol_text}")
        if self.metric:
            parts.append(f"metric {self.metric}")
        return " ".join(parts)


@dataclasses.dataclass(frozen=True)
class NeighbourEntry:
    destination: str
    dev: str
    mac: str
    states: tuple[str, ...]
    protocol: int | None
    protocol_text: str

    @property
    def owned(self) -> bool:
        return self.protocol == commands.ROUTE_PROTOCOL

    @property
    def permanent(self) -> bool:
        return "PERMANENT" in self.states

    def text(self) -> str:
        return (f"{self.destination} dev {self.dev} lladdr {self.mac or 'none'} {'/'.join(self.states)}"
                + (f" proto {self.protocol_text}" if self.protocol_text else ""))


@dataclasses.dataclass(frozen=True)
class FilterEntry:
    netdev: str
    pref: int
    chain: int
    kind: str
    protocol: int | None
    handle: int | None
    skip_sw: bool
    in_hw: bool
    dst_mac: str | None
    new_type: int | None
    out_dev: str | None
    problem: str | None                       # why the actions are not a relay filter's

    @property
    def reserved(self) -> bool:
        return self.chain == 0 and self.pref in RESERVED_PREFS

    def matches(self, planned: RelayFilter) -> bool:
        return (self.problem is None and self.kind == "flower" and self.chain == 0 and self.pref == planned.pref
                and self.protocol == planned.protocol and self.handle == planned.handle and self.skip_sw
                and self.dst_mac == planned.next_mac.lower() and self.new_type == planned.new_type
                and self.out_dev == planned.out_netdev)

    def text(self) -> str:
        protocol = f"0x{self.protocol:04x}" if self.protocol is not None else "unknown"
        if self.problem:
            body = f"not a relay filter ({self.problem})"
        else:
            new_type = f"0x{self.new_type:04x}" if self.new_type is not None else "?"
            body = f"-> {new_type}, dst {self.dst_mac}, out {self.out_dev}"
        flags = " ".join(flag for flag, on in (("skip_sw", self.skip_sw), ("in_hw", self.in_hw)) if on)
        return f"{self.netdev} pref {self.pref} handle {self.handle} {protocol} {body} {flags}".rstrip()


@dataclasses.dataclass(frozen=True)
class MarkerProcess:
    pid: int
    argv: tuple[str, ...]
    device: str | None
    rules: frozenset[tuple[str, int]]
    managed: bool
    problem: str | None

    def text(self) -> str:
        rules = " ".join(f"{d}=0x{v:04x}" for d, v in sorted(self.rules))
        return f"pid {self.pid} {self.device or '?'}: {rules}" + ("" if self.managed else " (not --managed)")


# -- decoders ----------------------------------------------------------------------------------------


def _json_list(text: str, what: str, problems: list[str]) -> list:
    text = text.strip()
    if not text:
        return []
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        problems.append(f"{what}: not JSON ({error.msg})")
        return []
    if not isinstance(value, list):
        problems.append(f"{what}: not a JSON list")
        return []
    return [entry for entry in value if isinstance(entry, dict)]


def parse_links(text: str, problems: list[str]) -> dict[str, Link]:
    links = {}
    for entry in _json_list(text, "ip -j addr show", problems):
        name = str(entry.get("ifname", ""))
        addresses = tuple((str(info.get("local")), int(info.get("prefixlen", 32)))
                          for info in entry.get("addr_info", []) if isinstance(info, dict)
                          and info.get("family") == "inet" and info.get("local"))
        links[name] = Link(name, str(entry.get("address", "")).lower(), addresses, str(entry.get("operstate", "")))
    return links


def parse_routes(text: str, problems: list[str]) -> list[RouteEntry]:
    found = []
    for entry in _json_list(text, "ip -j -d route show", problems):
        destination = str(entry.get("dst", ""))
        if destination == "default":
            address, length = "0.0.0.0", 0
        elif "/" in destination:
            address, _, length_text = destination.partition("/")
            length = int(length_text) if length_text.isdigit() else -1
        else:
            address, length = destination, 32
        try:
            ipaddress.IPv4Address(address)
        except ValueError:
            continue
        protocol_text = str(entry.get("protocol", "boot"))
        found.append(RouteEntry(address, length, str(entry.get("dev", "")), entry.get("prefsrc"),
                                str(entry.get("scope", "global")), protocol_number(protocol_text), protocol_text,
                                _int(entry.get("metric")) or 0, entry.get("gateway"),
                                str(entry.get("type", "unicast"))))
    return found


def parse_neighbours(text: str, problems: list[str]) -> list[NeighbourEntry]:
    found = []
    for entry in _json_list(text, "ip -j neigh show", problems):
        states = entry.get("state", [])
        states = tuple(states) if isinstance(states, list) else (str(states),)
        protocol_text = str(entry.get("protocol", "")) if entry.get("protocol") is not None else ""
        found.append(NeighbourEntry(str(entry.get("dst", "")), str(entry.get("dev", "")),
                                    str(entry.get("lladdr", "")).lower(), states,
                                    protocol_number(protocol_text) if protocol_text else None, protocol_text))
    return found


def parse_ingress_qdisc(text: str, problems: list[str], netdev: str) -> str | None:
    for entry in _json_list(text, f"tc -j qdisc show dev {netdev}", problems):
        if entry.get("kind") in ("ingress", "clsact"):
            return str(entry["kind"])
    return None


def _header_bytes(keys: list, keep_mask: bool) -> dict[int, int] | None:
    header: dict[int, int] = {}
    for key in keys:
        offset, value, mask = _int(key.get("offset")), _hex(key.get("val")), _hex(key.get("mask"))
        if offset is None or value is None or mask is None or offset % 4 or offset < 0:
            return None
        for index in range(4):
            shift = 24 - 8 * index
            byte_mask = (mask >> shift) & 0xFF
            written = byte_mask == 0 if keep_mask else byte_mask == 0xFF
            if written:
                header[offset + index] = (value >> shift) & 0xFF
            elif byte_mask not in (0, 0xFF):
                return None
    return header


def decode_filter(netdev: str, entry: Mapping[str, object]) -> FilterEntry | None:
    """One ``tc -j filter show`` entry; None for the per-preference summary entry without options."""
    options = entry.get("options")
    if not isinstance(options, dict):
        return None
    pref = _int(entry.get("pref"))
    chain = _int(entry.get("chain")) or 0
    kind = str(entry.get("kind", ""))
    keys = options.get("keys") if isinstance(options.get("keys"), dict) else {}
    protocol = ethertype(entry.get("protocol"))
    if protocol is None:
        protocol = _key_ethertype(keys.get("eth_type"))
    actions = [action for action in options.get("actions", []) or [] if isinstance(action, dict)]
    problem = None
    pedit_keys = []
    for action in actions:
        if action.get("kind") == "pedit":
            for key in action.get("keys", []) or []:
                if not isinstance(key, dict) or key.get("htype", "eth") != "eth" or key.get("cmd", "set") != "set":
                    problem = "a pedit key is not an Ethernet header set"
                pedit_keys.append(key)
    mirreds = [action for action in actions if action.get("kind") == "mirred"]
    others = [action.get("kind") for action in actions if action.get("kind") not in ("pedit", "mirred")]
    header = None
    for keep_mask in (True, False):
        candidate = _header_bytes(pedit_keys, keep_mask) if problem is None else None
        if candidate is not None and set(candidate) == {0, 1, 2, 3, 4, 5, 12, 13}:
            header = candidate
            break
    if problem is None and header is None:
        problem = "the pedit actions do not set exactly the destination MAC and the EtherType"
    if problem is None and others:
        problem = f"other actions {others}"
    out_dev = None
    if problem is None:
        if (len(mirreds) == 1 and actions[-1] is mirreds[0] and mirreds[0].get("mirred_action") == "redirect"
                and mirreds[0].get("direction") == "egress"):
            out_dev = str(mirreds[0].get("to_dev", ""))
        else:
            problem = "the last action is not one egress redirect"
    dst_mac = ":".join(f"{header[index]:02x}" for index in range(6)) if header else None
    new_type = (header[12] << 8) | header[13] if header else None
    return FilterEntry(netdev, pref if pref is not None else -1, chain, kind, protocol, _int(options.get("handle")),
                       bool(options.get("skip_sw")), bool(options.get("in_hw")), dst_mac, new_type, out_dev,
                       problem)


def parse_filters(text: str, problems: list[str], netdev: str) -> list[FilterEntry]:
    found = []
    for entry in _json_list(text, f"tc -j filter show dev {netdev} ingress", problems):
        decoded = decode_filter(netdev, entry)
        if decoded is not None:
            found.append(decoded)
    return found


def parse_markers(text: str) -> list[MarkerProcess]:
    found = []
    for line in text.splitlines():
        words = line.split()
        if len(words) < 2 or not words[0].isdigit():
            continue
        argv = tuple(words[1:])
        device, rules, problem = None, set(), None
        index = 1
        while index < len(argv):
            word = argv[index]
            if word == "--device" and index + 1 < len(argv):
                device = argv[index + 1]
                index += 2
                continue
            if word == "--rule" and index + 1 < len(argv):
                destination, _, value = argv[index + 1].partition("=")
                number = _int(value)
                if number is None:
                    problem = f"rule {argv[index + 1]!r} is not ADDRESS=ETHERTYPE"
                else:
                    rules.add((destination, number))
                index += 2
                continue
            index += 1
        found.append(MarkerProcess(int(words[0]), argv, device, frozenset(rules), "--managed" in argv, problem))
    return found


# -- host state ---------------------------------------------------------------------------------------


@dataclasses.dataclass
class HostState:
    """Everything the read script found on one Spark."""

    position: int
    name: str
    ssh: str
    reachable: bool
    error: str = ""
    hostname: str = ""
    versions: str = ""
    links: dict[str, Link] = dataclasses.field(default_factory=dict)
    rdma: dict[str, str] = dataclasses.field(default_factory=dict)
    routes: list[RouteEntry] = dataclasses.field(default_factory=list)
    neighbours: list[NeighbourEntry] = dataclasses.field(default_factory=list)
    ingress_qdisc: dict[str, str | None] = dataclasses.field(default_factory=dict)
    filters: dict[str, list[FilterEntry]] = dataclasses.field(default_factory=dict)
    markers: list[MarkerProcess] = dataclasses.field(default_factory=list)
    marker_binary: str | None = None
    marker_logs: dict[str, dict | None] = dataclasses.field(default_factory=dict)
    record: dict | None = None
    record_problem: str | None = None
    raw: dict[str, str] = dataclasses.field(default_factory=dict)
    parse_problems: list[str] = dataclasses.field(default_factory=list)

    # -- facts --

    def port(self, netdev: str) -> Port | None:
        link = self.links.get(netdev)
        if link is None or len(link.addresses) != 1 or not link.mac:
            return None
        address, length = link.addresses[0]
        try:
            return port_from_text(netdev, link.mac, f"{address}/{length}", owner=self.name)
        except FabricError:
            return None

    def ports(self) -> tuple[Port, ...]:
        return tuple(port for port in (self.port(netdev) for netdev in FABRIC_NETDEVS) if port is not None)

    def fact_problems(self, *, check_hostname: bool = True) -> list[str]:
        if not self.reachable:
            return [f"{self.name}: unreachable ({self.error})"]
        problems = [f"{self.name}: {problem}" for problem in self.parse_problems]
        if check_hostname and self.hostname != self.name:
            problems.append(f"{self.name}: the host at {self.ssh} is named {self.hostname or 'nothing'}, the site "
                            "lists it as " + self.name)
        for device, netdev in zip(RDMA_DEVICES, FABRIC_NETDEVS):
            if self.rdma.get(device) != netdev:
                problems.append(f"{self.name}: RDMA device {device} belongs to network device "
                                f"{self.rdma.get(device) or 'none'}, expected {netdev}")
        for netdev in FABRIC_NETDEVS:
            link = self.links.get(netdev)
            if link is None:
                problems.append(f"{self.name}: network device {netdev} is missing")
            elif len(link.addresses) != 1:
                problems.append(f"{self.name}: {netdev} has {len(link.addresses)} IPv4 addresses; one is needed")
            elif self.port(netdev) is None:
                problems.append(f"{self.name}: {netdev} has no usable MAC or IPv4 address")
        return problems

    # -- classification --

    def on_link(self, netdev: str, address: str) -> bool:
        link = self.links.get(netdev)
        if link is None:
            return False
        try:
            value = ipaddress.IPv4Address(address)
        except ValueError:
            return False
        return any(value in network for network in link.networks())

    def is_relay_route(self, route: RouteEntry) -> bool:
        return (route.prefixlen == 32 and route.dev in FABRIC_NETDEVS and route.scope == "link"
                and not route.gateway and route.kind == "unicast" and not self.on_link(route.dev, route.destination))

    def relay_routes(self) -> list[RouteEntry]:
        """Owned routes and unowned relay routes."""
        return [route for route in self.routes if route.owned or self.is_relay_route(route)]

    def is_relay_neighbour(self, neighbour: NeighbourEntry) -> bool:
        return (neighbour.permanent and neighbour.dev in FABRIC_NETDEVS
                and (neighbour.owned or not self.on_link(neighbour.dev, neighbour.destination)))

    def relay_neighbours(self) -> list[NeighbourEntry]:
        return [neighbour for neighbour in self.neighbours if neighbour.owned or self.is_relay_neighbour(neighbour)]

    def reserved_filters(self, netdev: str) -> list[FilterEntry]:
        return [entry for entry in self.filters.get(netdev, []) if entry.reserved]

    def foreign_tag_filters(self) -> list[FilterEntry]:
        """Filters outside the reserved preferences that match a relay tag (they could take relayed frames)."""
        return [entry for netdev in FABRIC_NETDEVS for entry in self.filters.get(netdev, [])
                if not entry.reserved and entry.protocol is not None and entry.protocol != ETH_P_IP
                and relays_left(entry.protocol)]

    def unowned_relay_objects(self) -> tuple[list[RouteEntry], list[NeighbourEntry]]:
        return ([route for route in self.relay_routes() if not route.owned],
                [neighbour for neighbour in self.relay_neighbours() if not neighbour.owned])


def parse_sections(stdout: str) -> dict[str, str]:
    """``@@<name>`` blocks of a script's output, by name."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in stdout.splitlines():
        if line.startswith(commands.SECTION):
            current = line[len(commands.SECTION):].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {key: "\n".join(value) for key, value in sections.items()}


def parse(position: int, name: str, ssh: str, *, returncode: int, stdout: str, stderr: str) -> HostState:
    """The :class:`HostState` of one Spark from its read script's output."""
    state = HostState(position, name, ssh, reachable=True)
    sections = parse_sections(stdout)
    if "end" not in sections:
        state.reachable = bool(sections)
        detail = (stderr.strip().splitlines() or [f"exit code {returncode}"])[-1]
        state.error = detail if not sections else f"the read script stopped early ({detail})"
        if not state.reachable:
            return state
        state.parse_problems.append(state.error)
    state.raw = sections
    raw = state.raw
    problems = state.parse_problems
    state.hostname = raw.get("hostname", "").strip()
    state.versions = " | ".join(line.strip() for line in raw.get("versions", "").splitlines() if line.strip())
    state.links = parse_links(raw.get("links", ""), problems)
    for device in RDMA_DEVICES:
        netdevs = raw.get(f"rdma:{device}", "").split()
        state.rdma[device] = netdevs[0] if netdevs else ""
    state.routes = parse_routes(raw.get("routes", ""), problems)
    state.neighbours = parse_neighbours(raw.get("neighbours", ""), problems)
    for netdev in FABRIC_NETDEVS:
        state.ingress_qdisc[netdev] = parse_ingress_qdisc(raw.get(f"qdisc:{netdev}", ""), problems, netdev)
        state.filters[netdev] = parse_filters(raw.get(f"filters:{netdev}", ""), problems, netdev)
    state.markers = parse_markers(raw.get("markers", ""))
    binary = raw.get("marker-binary", "").split()
    state.marker_binary = binary[0] if binary and len(binary[0]) == 64 else None
    for device in RDMA_DEVICES:
        text = raw.get(f"marker-log:{device}", "").strip()
        first = text.splitlines()[0] if text else ""
        try:
            state.marker_logs[device] = json.loads(first) if first.startswith("{") else (
                {"error": text[:300]} if text else None)
        except json.JSONDecodeError:
            state.marker_logs[device] = {"error": text[:300]}
    record_text = raw.get("record", "").strip()
    if record_text:
        try:
            record = json.loads(record_text)
            if not isinstance(record, dict) or record.get("schema") != commands.RECORD_SCHEMA:
                raise ValueError(f"schema is not {commands.RECORD_SCHEMA}")
            state.record = record
        except (ValueError, json.JSONDecodeError) as error:
            state.record_problem = f"the record {commands.RECORD_PATH} is unreadable: {error}"
    return state
