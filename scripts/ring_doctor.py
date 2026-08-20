#!/usr/bin/env python3
"""Discover, diagnose, repair, and verify a four-node SparkRing fabric.

The default operation is read-only. Remote changes require ``--apply``. Every
route next hop and every boot-time address guard comes from addresses observed
through SSH during the same invocation.

Repair covers the fabric interfaces and the routes between them. The launch
endpoints a distributed job names — the rendezvous address and the per-node
NCCL and GLOO socket interfaces — are reported against what each node observes
and are never changed, because they can sit on networks outside this fabric.
The NetworkManager reconnection settings of wireless management interfaces are
read and diagnosed the same way, never changed.
"""

from __future__ import annotations

import argparse
import dataclasses
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

FABRIC_INTERFACES = ("enp1s0f0np0", "enp1s0f1np1")
SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")
INTERFACE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
SECTION_RE = re.compile(r"^__RING_DOCTOR_([A-Z_]+)__$")
UNAVAILABLE = "__UNAVAILABLE__"


class ConfigurationError(ValueError):
    """A ring-doctor input cannot identify a safe discovery operation."""


@dataclasses.dataclass(frozen=True)
class NodeSpec:
    """One management SSH target and its optional discovery jump host.

    ``socket_interfaces`` holds the interface names a distributed launch gives
    to ``NCCL_SOCKET_IFNAME`` and ``GLOO_SOCKET_IFNAME`` on this node. They are
    inputs to be checked against the node, never values this tool sets.
    """

    name: str
    target: str
    proxy_jump: str | None = None
    socket_interfaces: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class InterfaceState:
    """Observed IPv4 addresses and link state for one fabric interface."""

    name: str
    addresses: tuple[ipaddress.IPv4Interface, ...]
    oper_state: str
    mtu: int | None = None

    def address_on(self, network: ipaddress.IPv4Network) -> ipaddress.IPv4Address | None:
        for address in self.addresses:
            if address.ip in network:
                return address.ip
        return None


@dataclasses.dataclass(frozen=True)
class Route:
    """One IPv4 kernel route relevant to fabric reachability."""

    destination: ipaddress.IPv4Network
    gateway: ipaddress.IPv4Address | None
    device: str | None
    raw: str


@dataclasses.dataclass(frozen=True)
class RendezvousRoute:
    """One node's kernel route selection for a distributed rendezvous address.

    A distributed launch names a single rendezvous address that every rank
    connects to; torch passes it as ``--master-addr`` and its TCPStore listens
    there. ``routable`` is true only when the kernel selected an egress device
    for that address on the observed node.
    """

    address: ipaddress.IPv4Address
    routable: bool
    device: str | None
    source: ipaddress.IPv4Address | None
    gateway: ipaddress.IPv4Address | None
    raw: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "address": str(self.address),
            "routable": self.routable,
            "device": self.device,
            "source": str(self.source) if self.source else None,
            "gateway": str(self.gateway) if self.gateway else None,
            "raw": self.raw,
        }


@dataclasses.dataclass(frozen=True)
class WifiInterfaceState:
    """Reconnection settings observed for one wireless management interface.

    An interface appears here only when a NetworkManager connection was active
    on it during the probe. The three profile settings hold the raw ``nmcli``
    values (``autoconnect_retries`` uses ``0`` for retry-forever), or None
    when the per-profile query produced no value. ``driver_power_save`` holds
    the ``iw`` power-save state, ``on`` or ``off``, or None when ``iw`` was
    unavailable on the node.
    """

    device: str
    connection: str
    autoconnect: str | None
    autoconnect_retries: str | None
    powersave: str | None
    driver_power_save: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class WifiObservation:
    """Wireless management reconnection evidence for one node.

    ``nmcli_available`` is False when the node has no ``nmcli``, in which case
    no wireless interface can be enumerated there. ``interfaces`` holds every
    wireless interface that had an active NetworkManager connection when
    probed; a node whose management runs over a wired path has none.
    """

    nmcli_available: bool
    interfaces: tuple[WifiInterfaceState, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "nmcli_available": self.nmcli_available,
            "interfaces": [
                interface.to_dict() for interface in self.interfaces
            ],
        }


@dataclasses.dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    detail: str = ""


class Runner(Protocol):
    """An SSH command runner used by discovery, repair, and verification."""

    def run(
        self, target: str, command: str, proxy_jump: str | None = None
    ) -> CommandResult:
        ...


class SshRunner:
    """Run non-interactive SSH commands with an optional ProxyJump target."""

    def __init__(self, timeout_seconds: int = 20, connect_timeout: int = 5) -> None:
        self.timeout_seconds = timeout_seconds
        self.connect_timeout = connect_timeout

    def run(
        self, target: str, command: str, proxy_jump: str | None = None
    ) -> CommandResult:
        argv = [
            "ssh",
            "-o",
            f"ConnectTimeout={self.connect_timeout}",
            "-o",
            "BatchMode=yes",
        ]
        if proxy_jump:
            argv.extend(("-J", proxy_jump))
        argv.extend((target, command))
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                False, detail=f"SSH timed out after {self.timeout_seconds} seconds"
            )
        except OSError as exc:
            return CommandResult(False, detail=f"SSH could not run: {exc}")
        detail = ""
        if completed.returncode:
            detail = (
                f"SSH exited {completed.returncode}: "
                f"{completed.stderr.strip()[:300]}"
            )
        return CommandResult(
            completed.returncode == 0,
            completed.stdout,
            completed.stderr,
            detail,
        )


@dataclasses.dataclass
class NodeObservation:
    """Discovery evidence for one configured node, including probe failures.

    ``interfaces`` holds only the two fabric interfaces the ring is built from.
    ``host_interfaces`` holds every interface the node reported, so a name that
    a launch configuration gives to NCCL or GLOO can be checked for existence
    and for an IPv4 address without a second remote round. ``wifi`` holds the
    wireless management reconnection evidence gathered by the same probe, or
    None when the probe output carried no wifi section.
    """

    spec: NodeSpec
    reachable: bool
    hostname: str | None = None
    interfaces: dict[str, InterfaceState] = dataclasses.field(default_factory=dict)
    routes: tuple[Route, ...] = ()
    ip_forward: bool | None = None
    forward_policy: str | None = None
    docker_user_rules: tuple[str, ...] | None = None
    error: str = ""
    proxy_jump_used: str | None = None
    host_interfaces: dict[str, InterfaceState] = dataclasses.field(
        default_factory=dict
    )
    rendezvous_route: RendezvousRoute | None = None
    wifi: WifiObservation | None = None

    @property
    def label(self) -> str:
        return self.hostname or self.spec.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.spec.name,
            "target": self.spec.target,
            "reachable": self.reachable,
            "hostname": self.hostname,
            "proxy_jump_used": self.proxy_jump_used,
            "error": self.error,
            "interfaces": {
                name: {
                    "addresses": [str(address) for address in state.addresses],
                    "oper_state": state.oper_state,
                    "mtu": state.mtu,
                }
                for name, state in sorted(self.interfaces.items())
            },
            "routes": [
                {
                    "destination": str(route.destination),
                    "gateway": str(route.gateway) if route.gateway else None,
                    "device": route.device,
                    "raw": route.raw,
                }
                for route in self.routes
            ],
            "host_interfaces": {
                name: {
                    "addresses": [str(address) for address in state.addresses],
                    "oper_state": state.oper_state,
                    "mtu": state.mtu,
                }
                for name, state in sorted(self.host_interfaces.items())
            },
            "rendezvous_route": (
                self.rendezvous_route.to_dict() if self.rendezvous_route else None
            ),
            "wifi": self.wifi.to_dict() if self.wifi else None,
            "ip_forward": self.ip_forward,
            "forward_policy": self.forward_policy,
            "docker_user_rules": (
                list(self.docker_user_rules)
                if self.docker_user_rules is not None
                else None
            ),
        }


@dataclasses.dataclass(frozen=True)
class Finding:
    """One diagnosed condition and the observation that proves it."""

    severity: str
    code: str
    node: str | None
    message: str
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class ProposedRoute:
    """A route whose gateway is an address observed on an adjacent node."""

    destination: ipaddress.IPv4Network
    gateway: ipaddress.IPv4Address
    device: str
    path: tuple[str, ...]

    def command(self, sudo: bool = True) -> str:
        prefix = "sudo -n " if sudo else ""
        return (
            f"{prefix}ip route replace {self.destination} via {self.gateway} "
            f"dev {self.device}"
        )


@dataclasses.dataclass
class NodePlan:
    """Idempotent route, forwarding, and firewall operations for one node."""

    node: str
    routes: list[ProposedRoute] = dataclasses.field(default_factory=list)
    relay_directions: set[tuple[str, str]] = dataclasses.field(default_factory=set)

    def commands(self, sudo: bool = True) -> list[str]:
        prefix = "sudo -n " if sudo else ""
        commands = [route.command(sudo=sudo) for route in self.routes]
        if self.relay_directions:
            commands.append(f"{prefix}sysctl -w net.ipv4.ip_forward=1")
        for ingress, egress in sorted(self.relay_directions):
            rule = f"-i {ingress} -o {egress} -j ACCEPT"
            commands.append(
                f"{prefix}iptables -C DOCKER-USER {rule} || "
                f"{prefix}iptables -I DOCKER-USER 1 {rule}"
            )
        return commands

    def to_dict(self) -> dict[str, Any]:
        return {
            "node": self.node,
            "routes": [
                {
                    "destination": str(route.destination),
                    "gateway": str(route.gateway),
                    "device": route.device,
                    "path": list(route.path),
                    "command": route.command(),
                }
                for route in self.routes
            ],
            "relay_directions": [
                {"in": ingress, "out": egress}
                for ingress, egress in sorted(self.relay_directions)
            ],
            "commands": self.commands(),
        }


@dataclasses.dataclass
class Topology:
    """Adjacency and subnet ownership inferred only from reachable nodes."""

    adjacency: dict[str, set[str]]
    subnet_nodes: dict[ipaddress.IPv4Network, tuple[str, ...]]
    cycle: tuple[str, ...]
    valid_cycle: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "adjacency": {
                node: sorted(neighbors)
                for node, neighbors in sorted(self.adjacency.items())
            },
            "subnets": {
                str(network): list(nodes)
                for network, nodes in sorted(
                    self.subnet_nodes.items(), key=lambda item: int(item[0].network_address)
                )
            },
            "valid_cycle": self.valid_cycle,
            "cycle": list(self.cycle),
            "reason": self.reason,
        }


def _validate_ssh_target(value: str, field: str) -> str:
    if not SSH_TARGET_RE.fullmatch(value):
        raise ConfigurationError(f"{field} must have the form user@host")
    return value


def _validate_ssh_jump_chain(value: str, field: str) -> str:
    """Validate a ProxyJump value, which may chain several hops.

    A rank whose management interface is unavailable is still reachable by
    hopping along the fabric cabling, so a jump specification is a comma
    separated list of hops in the order ssh -J expects. Each hop is validated
    on its own; an empty hop is rejected rather than silently dropped.
    """
    hops = value.split(",")
    if not hops or any(not hop for hop in hops):
        raise ConfigurationError(f"{field} contains an empty jump hop")
    for index, hop in enumerate(hops):
        _validate_ssh_target(hop, f"{field} hop {index + 1}")
    return value


def _validate_interface(value: str, field: str) -> str:
    if not INTERFACE_RE.fullmatch(value):
        raise ConfigurationError(f"{field} contains an unsupported interface name")
    return value


def _default_name(target: str) -> str:
    return target.split("@", 1)[1]


def parse_node_specs(document: Mapping[str, Any]) -> tuple[NodeSpec, ...]:
    """Parse the JSON node list accepted by ``--config``."""
    raw_nodes = document.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ConfigurationError("config.nodes must be a non-empty list")
    specs: list[NodeSpec] = []
    for index, raw in enumerate(raw_nodes):
        where = f"config.nodes[{index}]"
        if isinstance(raw, str):
            target = _validate_ssh_target(raw, where)
            specs.append(NodeSpec(_default_name(target), target))
            continue
        if not isinstance(raw, Mapping):
            raise ConfigurationError(f"{where} must be a string or object")
        unknown = set(raw) - {"name", "target", "proxy_jump", "socket_interfaces"}
        if unknown:
            raise ConfigurationError(
                f"{where} has unsupported keys: {', '.join(sorted(unknown))}"
            )
        target_value = raw.get("target")
        if not isinstance(target_value, str):
            raise ConfigurationError(f"{where}.target must be a string")
        target = _validate_ssh_target(target_value, f"{where}.target")
        name_value = raw.get("name", _default_name(target))
        if not isinstance(name_value, str) or not NAME_RE.fullmatch(name_value):
            raise ConfigurationError(f"{where}.name contains unsupported characters")
        jump_value = raw.get("proxy_jump")
        if jump_value is not None:
            if not isinstance(jump_value, str):
                raise ConfigurationError(f"{where}.proxy_jump must be a string")
            jump_value = _validate_ssh_jump_chain(jump_value, f"{where}.proxy_jump")
        socket_value = raw.get("socket_interfaces")
        sockets: tuple[str, ...] = ()
        if socket_value is not None:
            if not isinstance(socket_value, list) or not socket_value:
                raise ConfigurationError(
                    f"{where}.socket_interfaces must be a non-empty list of names"
                )
            names: list[str] = []
            for socket_index, name in enumerate(socket_value):
                if not isinstance(name, str):
                    raise ConfigurationError(
                        f"{where}.socket_interfaces[{socket_index}] must be a string"
                    )
                names.append(
                    _validate_interface(
                        name, f"{where}.socket_interfaces[{socket_index}]"
                    )
                )
            sockets = tuple(names)
        specs.append(NodeSpec(name_value, target, jump_value, sockets))
    _require_unique_specs(specs)
    return tuple(specs)


def _require_unique_specs(specs: Sequence[NodeSpec]) -> None:
    names = [spec.name for spec in specs]
    targets = [spec.target for spec in specs]
    if len(names) != len(set(names)):
        raise ConfigurationError("node names must be unique")
    if len(targets) != len(set(targets)):
        raise ConfigurationError("node SSH targets must be unique")


# Shell for the probe section holding wireless management reconnection
# evidence. It prints the raw ``nmcli`` terse device rows, then, for every
# row whose TYPE is exactly ``wifi`` and whose CONNECTION is non-empty, a
# ``PROFILE <device> <connection>`` line, the three per-profile setting values
# (one per line), and an ``IW <device> <on|off|unavailable>`` driver
# power-save line. The per-profile query needs the connection names the same
# section just discovered, so the shell loops over them itself instead of
# adding a second remote contact. ``nmcli -t`` escapes ``:`` and ``\\`` in
# connection names with a backslash; ``sed`` removes those escapes before the
# name is reused. A node without ``nmcli`` prints the unavailable sentinel and
# never fails the probe.
_WIFI_PROBE = f"""printf '%s\\n' '__RING_DOCTOR_WIFI__'
if command -v nmcli >/dev/null 2>&1; then
  rd_wifi_rows=$(nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device status 2>/dev/null)
  printf '%s\\n' "$rd_wifi_rows"
  printf '%s\\n' "$rd_wifi_rows" | while IFS= read -r rd_row; do
    rd_dev=${{rd_row%%:*}}
    rd_rest=${{rd_row#*:}}
    rd_type=${{rd_rest%%:*}}
    rd_rest=${{rd_rest#*:}}
    rd_name=$(printf '%s\\n' "${{rd_rest#*:}}" | sed 's/\\\\\\(.\\)/\\1/g')
    [ "$rd_type" = wifi ] || continue
    [ -n "$rd_name" ] || continue
    printf 'PROFILE %s %s\\n' "$rd_dev" "$rd_name"
    nmcli -t -g connection.autoconnect,connection.autoconnect-retries,802-11-wireless.powersave connection show "$rd_name" 2>/dev/null || printf '%s\\n' '{UNAVAILABLE}'
    if command -v iw >/dev/null 2>&1; then
      case "$(iw dev "$rd_dev" get power_save 2>/dev/null)" in
        *': on') printf 'IW %s on\\n' "$rd_dev" ;;
        *': off') printf 'IW %s off\\n' "$rd_dev" ;;
        *) printf 'IW %s unavailable\\n' "$rd_dev" ;;
      esac
    else
      printf 'IW %s unavailable\\n' "$rd_dev"
    fi
  done
else
  printf '%s\\n' '{UNAVAILABLE}'
fi
"""


def build_probe_command(
    interfaces: Sequence[str],
    rendezvous_address: ipaddress.IPv4Address | None = None,
) -> str:
    """Build the read-only remote probe that emits sectioned evidence.

    Naming a rendezvous address adds one route lookup to the same probe, so a
    node is still contacted once per discovery attempt. The wifi section
    likewise rides the same contact: it enumerates wireless interfaces and
    reads each active profile's reconnection settings inside this one command.
    """
    for index, interface in enumerate(interfaces):
        _validate_interface(interface, f"fabric_interfaces[{index}]")
    names = " ".join(shlex.quote(interface) for interface in interfaces)
    rendezvous_probe = ""
    if rendezvous_address is not None:
        destination = shlex.quote(str(rendezvous_address))
        rendezvous_probe = (
            "printf '%s\\n' '__RING_DOCTOR_RENDEZVOUS__'\n"
            f"ip -4 route get {destination} 2>&1\n"
        )
    return f"""set +e
printf '%s\\n' '__RING_DOCTOR_HOSTNAME__'
hostname
printf '%s\\n' '__RING_DOCTOR_ADDR__'
ip -br -4 addr show
printf '%s\\n' '__RING_DOCTOR_LINK__'
ip -o link show
printf '%s\\n' '__RING_DOCTOR_ROUTE__'
ip -4 route show
printf '%s\\n' '__RING_DOCTOR_FORWARD__'
sysctl -n net.ipv4.ip_forward 2>/dev/null || printf '%s\\n' '{UNAVAILABLE}'
printf '%s\\n' '__RING_DOCTOR_FORWARD_CHAIN__'
sudo -n iptables -S FORWARD 2>/dev/null || printf '%s\\n' '{UNAVAILABLE}'
printf '%s\\n' '__RING_DOCTOR_DOCKER_USER__'
sudo -n iptables -S DOCKER-USER 2>/dev/null || printf '%s\\n' '{UNAVAILABLE}'
{_WIFI_PROBE}{rendezvous_probe}printf '%s\\n' '__RING_DOCTOR_INTERFACES__'
printf '%s\\n' {names}
"""


def _split_sections(text: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    active: str | None = None
    for line in text.splitlines():
        match = SECTION_RE.fullmatch(line.strip())
        if match:
            active = match.group(1)
            sections.setdefault(active, [])
        elif active is not None:
            sections[active].append(line)
    return {name: "\n".join(lines).strip() for name, lines in sections.items()}


def parse_all_addresses(text: str) -> dict[str, InterfaceState]:
    """Parse ``ip -br -4 addr`` output for every interface a node reports.

    An interface with no IPv4 address still occupies a row, so an empty
    address tuple distinguishes an addressless interface from an absent one.
    """
    states: dict[str, InterfaceState] = {}
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 2:
            continue
        addresses: list[ipaddress.IPv4Interface] = []
        for field in fields[2:]:
            try:
                address = ipaddress.ip_interface(field)
            except ValueError:
                continue
            if isinstance(address, ipaddress.IPv4Interface):
                addresses.append(address)
        states[fields[0]] = InterfaceState(
            fields[0], tuple(addresses), fields[1].upper()
        )
    return states


def parse_brief_addresses(
    text: str, fabric_interfaces: Sequence[str] = FABRIC_INTERFACES
) -> dict[str, InterfaceState]:
    """Parse ``ip -br -4 addr`` output for the selected fabric interfaces."""
    selected = set(fabric_interfaces)
    return {
        name: state
        for name, state in parse_all_addresses(text).items()
        if name in selected
    }


def parse_link_details(text: str) -> dict[str, tuple[str, int | None]]:
    """Parse operational state and MTU from ``ip -o link show`` output."""
    details: dict[str, tuple[str, int | None]] = {}
    for line in text.splitlines():
        match = re.match(r"^\d+:\s+([^:@]+)(?:@[^:]+)?:\s+.*$", line)
        if not match:
            continue
        name = match.group(1)
        state_match = re.search(r"\bstate\s+(\S+)", line)
        mtu_match = re.search(r"\bmtu\s+(\d+)", line)
        details[name] = (
            state_match.group(1).upper() if state_match else "UNKNOWN",
            int(mtu_match.group(1)) if mtu_match else None,
        )
    return details


def parse_routes(text: str) -> tuple[Route, ...]:
    """Parse IPv4 routes while retaining each source line as evidence."""
    routes: list[Route] = []
    for raw in text.splitlines():
        fields = raw.split()
        if not fields:
            continue
        destination_text = "0.0.0.0/0" if fields[0] == "default" else fields[0]
        try:
            destination = ipaddress.ip_network(destination_text, strict=False)
        except ValueError:
            continue
        if not isinstance(destination, ipaddress.IPv4Network):
            continue
        gateway: ipaddress.IPv4Address | None = None
        device: str | None = None
        if "via" in fields:
            try:
                parsed_gateway = ipaddress.ip_address(fields[fields.index("via") + 1])
                if isinstance(parsed_gateway, ipaddress.IPv4Address):
                    gateway = parsed_gateway
            except (ValueError, IndexError):
                pass
        if "dev" in fields:
            try:
                device = fields[fields.index("dev") + 1]
            except IndexError:
                pass
        routes.append(Route(destination, gateway, device, raw))
    return tuple(routes)


def _field_after(fields: Sequence[str], keyword: str) -> str | None:
    if keyword not in fields:
        return None
    index = list(fields).index(keyword) + 1
    return fields[index] if index < len(fields) else None


def _address_after(
    fields: Sequence[str], keyword: str
) -> ipaddress.IPv4Address | None:
    value = _field_after(fields, keyword)
    if value is None:
        return None
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        return None
    return address if isinstance(address, ipaddress.IPv4Address) else None


def parse_route_get(
    address: ipaddress.IPv4Address, text: str
) -> RendezvousRoute:
    """Parse ``ip -4 route get`` output for one destination address.

    On success the kernel prints the selected egress device, the source
    address it would use, and any gateway. It prints an ``RTNETLINK`` error
    when no route exists, and prefixes the destination with ``unreachable``,
    ``prohibit``, ``blackhole``, or ``throw`` when a route denies the
    destination. Any of those, and any output without a device, is reported as
    not routable, with the kernel's own wording retained as evidence.
    """
    raw = " ".join(
        line.strip() for line in text.strip().splitlines() if line.strip()
    )
    if not raw or UNAVAILABLE in raw:
        return RendezvousRoute(
            address, False, None, None, None, "the route probe produced no output"
        )
    fields = raw.split()
    denied = fields[0] in {"unreachable", "prohibit", "blackhole", "throw"}
    if denied or raw.startswith("RTNETLINK answers:"):
        return RendezvousRoute(address, False, None, None, None, raw)
    device = _field_after(fields, "dev")
    return RendezvousRoute(
        address,
        device is not None,
        device,
        _address_after(fields, "src"),
        _address_after(fields, "via"),
        raw,
    )


_NMCLI_ESCAPE_RE = re.compile(r"\\(.)")


def parse_wifi_observation(text: str) -> WifiObservation | None:
    """Parse the wifi probe section into per-interface reconnection evidence.

    The section body holds the raw ``nmcli -t -f DEVICE,TYPE,STATE,CONNECTION
    device status`` rows, then one block per wireless interface with an active
    connection: a ``PROFILE <device> <connection>`` line, the three values
    printed one per line by ``nmcli -t -g connection.autoconnect,connection.
    autoconnect-retries,802-11-wireless.powersave connection show``, and an
    ``IW <device> <on|off|unavailable>`` line holding the driver power-save
    state. Only rows whose TYPE is exactly ``wifi`` name a wireless
    interface: the ``wifi-p2p`` row NetworkManager adds for the same radio,
    whose DEVICE carries a ``p2p-dev-`` prefix, is a sub-device, never an
    interface. Returns None for an absent or empty section, and an
    observation with ``nmcli_available=False`` when the section holds the
    sentinel a node without ``nmcli`` prints.
    """
    stripped = text.strip()
    if not stripped:
        return None
    if stripped == UNAVAILABLE:
        return WifiObservation(nmcli_available=False)
    active: dict[str, str] = {}
    settings: dict[str, tuple[str | None, str | None, str | None]] = {}
    driver: dict[str, str | None] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("PROFILE "):
            head = line.split(" ", 2)
            device = head[1] if len(head) > 1 else ""
            index += 1
            values: list[str] = []
            while (
                index < len(lines)
                and len(values) < 3
                and not lines[index].startswith(("PROFILE ", "IW "))
            ):
                values.append(lines[index].strip())
                index += 1
            if UNAVAILABLE in values:
                settings[device] = (None, None, None)
            else:
                padded = values + [""] * (3 - len(values))
                settings[device] = (
                    padded[0] or None,
                    padded[1] or None,
                    padded[2] or None,
                )
            continue
        if line.startswith("IW "):
            fields = line.split()
            if len(fields) >= 3:
                driver[fields[1]] = (
                    fields[2] if fields[2] in {"on", "off"} else None
                )
            index += 1
            continue
        fields = line.split(":", 3)
        if len(fields) == 4:
            device, device_type, _state, connection = fields
            connection = _NMCLI_ESCAPE_RE.sub(r"\1", connection)
            if device_type == "wifi" and connection and device not in active:
                active[device] = connection
        index += 1
    interfaces = tuple(
        WifiInterfaceState(
            device,
            connection,
            *settings.get(device, (None, None, None)),
            driver_power_save=driver.get(device),
        )
        for device, connection in active.items()
    )
    return WifiObservation(True, interfaces)


def parse_probe_output(
    spec: NodeSpec,
    text: str,
    interfaces: Sequence[str] = FABRIC_INTERFACES,
    proxy_jump_used: str | None = None,
    rendezvous_address: ipaddress.IPv4Address | None = None,
) -> NodeObservation:
    """Convert one successful probe record into typed discovery evidence."""
    sections = _split_sections(text)
    interface_states = parse_brief_addresses(sections.get("ADDR", ""), interfaces)
    host_interfaces = parse_all_addresses(sections.get("ADDR", ""))
    for name, (oper_state, mtu) in parse_link_details(
        sections.get("LINK", "")
    ).items():
        if name in interface_states:
            state = interface_states[name]
            interface_states[name] = dataclasses.replace(
                state, oper_state=oper_state, mtu=mtu
            )
        existing = host_interfaces.get(name)
        host_interfaces[name] = (
            dataclasses.replace(existing, oper_state=oper_state, mtu=mtu)
            if existing is not None
            else InterfaceState(name, (), oper_state, mtu)
        )
    rendezvous_route: RendezvousRoute | None = None
    if rendezvous_address is not None and "RENDEZVOUS" in sections:
        rendezvous_route = parse_route_get(
            rendezvous_address, sections["RENDEZVOUS"]
        )
    forward_text = sections.get("FORWARD", "")
    ip_forward: bool | None = None
    if forward_text.strip() in {"0", "1"}:
        ip_forward = forward_text.strip() == "1"
    chain_text = sections.get("FORWARD_CHAIN", "")
    policy_match = re.search(r"^-P FORWARD\s+(\S+)", chain_text, re.MULTILINE)
    forward_policy = policy_match.group(1).upper() if policy_match else None
    docker_text = sections.get("DOCKER_USER", "")
    docker_rules = None
    if docker_text and UNAVAILABLE not in docker_text:
        docker_rules = tuple(
            line.strip() for line in docker_text.splitlines() if line.strip()
        )
    hostname = sections.get("HOSTNAME", "").splitlines()
    return NodeObservation(
        spec=spec,
        reachable=True,
        hostname=hostname[0].strip() if hostname and hostname[0].strip() else None,
        interfaces=interface_states,
        routes=parse_routes(sections.get("ROUTE", "")),
        ip_forward=ip_forward,
        forward_policy=forward_policy,
        docker_user_rules=docker_rules,
        proxy_jump_used=proxy_jump_used,
        host_interfaces=host_interfaces,
        rendezvous_route=rendezvous_route,
        wifi=parse_wifi_observation(sections.get("WIFI", "")),
    )


def discover_nodes(
    specs: Sequence[NodeSpec],
    runner: Runner,
    interfaces: Sequence[str] = FABRIC_INTERFACES,
    rendezvous_address: ipaddress.IPv4Address | None = None,
) -> dict[str, NodeObservation]:
    """Probe nodes directly, then retry failures through discovered jump hosts."""
    command = build_probe_command(interfaces, rendezvous_address)
    observations: dict[str, NodeObservation] = {}
    attempted: dict[str, list[str]] = {spec.name: [] for spec in specs}

    def attempt(spec: NodeSpec, jump: str | None) -> bool:
        label = jump or "direct"
        if label in attempted[spec.name]:
            return False
        attempted[spec.name].append(label)
        result = runner.run(spec.target, command, jump)
        if result.ok:
            observations[spec.name] = parse_probe_output(
                spec, result.stdout, interfaces, jump, rendezvous_address
            )
            return True
        observations[spec.name] = NodeObservation(
            spec, False, error=result.detail or result.stderr.strip() or "SSH failed"
        )
        return False

    for spec in specs:
        attempt(spec, None)
    for spec in specs:
        if not observations[spec.name].reachable and spec.proxy_jump:
            attempt(spec, spec.proxy_jump)

    progress = True
    while progress:
        progress = False
        jump_targets = [
            observation.spec.target
            for observation in observations.values()
            if observation.reachable
        ]
        for spec in specs:
            if observations[spec.name].reachable:
                continue
            for jump in jump_targets:
                if jump == spec.target:
                    continue
                if attempt(spec, jump):
                    progress = True
                    break
    for spec in specs:
        observation = observations[spec.name]
        if not observation.reachable:
            observation.error = (
                f"{observation.error}; attempted paths: "
                f"{', '.join(attempted[spec.name])}"
            )
    return observations


def _interface_networks(
    observation: NodeObservation,
) -> dict[ipaddress.IPv4Network, str]:
    networks: dict[ipaddress.IPv4Network, str] = {}
    for interface in observation.interfaces.values():
        for address in interface.addresses:
            networks[address.network] = interface.name
    return networks


def infer_adjacency(
    observations: Mapping[str, NodeObservation],
) -> tuple[dict[str, set[str]], dict[ipaddress.IPv4Network, tuple[str, ...]]]:
    """Infer adjacency when two reachable nodes share an observed subnet."""
    reachable = {
        name: observation
        for name, observation in observations.items()
        if observation.reachable
    }
    subnet_members: dict[ipaddress.IPv4Network, list[str]] = {}
    for name, observation in reachable.items():
        for network in _interface_networks(observation):
            subnet_members.setdefault(network, []).append(name)
    adjacency = {name: set() for name in reachable}
    subnet_nodes: dict[ipaddress.IPv4Network, tuple[str, ...]] = {}
    for network, members in subnet_members.items():
        unique = tuple(sorted(set(members)))
        subnet_nodes[network] = unique
        if len(unique) == 2:
            left, right = unique
            adjacency[left].add(right)
            adjacency[right].add(left)
    return adjacency, subnet_nodes


def validate_cycle(
    adjacency: Mapping[str, set[str]],
) -> tuple[bool, tuple[str, ...], str]:
    """Validate one connected simple cycle and return a deterministic order."""
    if len(adjacency) < 3:
        return False, (), "fewer than three nodes were available"
    wrong_degree = sorted(node for node, neighbors in adjacency.items() if len(neighbors) != 2)
    if wrong_degree:
        return (
            False,
            (),
            "cycle nodes must have two neighbors; invalid nodes: "
            + ", ".join(wrong_degree),
        )
    start = min(adjacency)
    cycle = [start]
    previous: str | None = None
    current = start
    while True:
        candidates = sorted(adjacency[current] - ({previous} if previous else set()))
        if not candidates:
            return False, (), "adjacency traversal ended before closing the cycle"
        following = candidates[0]
        if following == start:
            break
        if following in cycle:
            return False, (), "adjacency contains a closed component below ring size"
        cycle.append(following)
        previous, current = current, following
    if len(cycle) != len(adjacency):
        return False, (), "adjacency contains more than one connected component"
    return True, tuple(cycle), "single cycle"


def infer_topology(observations: Mapping[str, NodeObservation]) -> Topology:
    adjacency, subnet_nodes = infer_adjacency(observations)
    invalid_nodes = []
    for name, observation in observations.items():
        if not observation.reachable:
            continue
        states = tuple(observation.interfaces.values())
        if (
            len(states) != 2
            or any(len(state.addresses) != 1 for state in states)
            or any(
                address.network.prefixlen != 24
                for state in states
                for address in state.addresses
            )
        ):
            invalid_nodes.append(name)
    if invalid_nodes:
        return Topology(
            adjacency,
            subnet_nodes,
            (),
            False,
            "each reachable node must expose one /24 address on each of two "
            "fabric interfaces; invalid nodes: " + ", ".join(sorted(invalid_nodes)),
        )
    invalid_subnets = [
        network
        for network, nodes in subnet_nodes.items()
        if network.prefixlen != 24 or len(nodes) != 2
    ]
    if invalid_subnets:
        details = ", ".join(
            f"{network} ({len(subnet_nodes[network])} observed endpoints)"
            for network in sorted(
                invalid_subnets, key=lambda item: int(item.network_address)
            )
        )
        return Topology(
            adjacency,
            subnet_nodes,
            (),
            False,
            f"fabric subnets must be /24 links with two endpoints: {details}",
        )
    valid, cycle, reason = validate_cycle(adjacency)
    return Topology(adjacency, subnet_nodes, cycle, valid, reason)


def _shared_network(
    left: str,
    right: str,
    topology: Topology,
) -> ipaddress.IPv4Network | None:
    matches = [
        network
        for network, nodes in topology.subnet_nodes.items()
        if left in nodes and right in nodes
    ]
    return min(matches, key=lambda network: int(network.network_address)) if matches else None


def _interface_for_network(
    observation: NodeObservation, network: ipaddress.IPv4Network
) -> str | None:
    return _interface_networks(observation).get(network)


def _address_for_network(
    observation: NodeObservation, network: ipaddress.IPv4Network
) -> ipaddress.IPv4Address | None:
    for interface in observation.interfaces.values():
        address = interface.address_on(network)
        if address is not None:
            return address
    return None


def _shortest_path(
    adjacency: Mapping[str, set[str]], source: str, targets: set[str]
) -> tuple[str, ...] | None:
    queue: deque[tuple[str, ...]] = deque([(source,)])
    visited = {source}
    while queue:
        path = queue.popleft()
        if path[-1] in targets:
            return path
        for neighbor in sorted(adjacency.get(path[-1], set())):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + (neighbor,))
    return None


def select_route(
    source: str,
    destination: ipaddress.IPv4Network,
    observations: Mapping[str, NodeObservation],
    topology: Topology,
) -> ProposedRoute | None:
    """Choose a route through an observed neighbor address, or return unknown."""
    owners = set(topology.subnet_nodes.get(destination, ()))
    if not owners or source in owners:
        return None
    path = _shortest_path(topology.adjacency, source, owners)
    if path is None or len(path) < 2:
        return None
    neighbor = path[1]
    shared = _shared_network(source, neighbor, topology)
    if shared is None:
        return None
    gateway = _address_for_network(observations[neighbor], shared)
    device = _interface_for_network(observations[source], shared)
    if gateway is None or device is None:
        return None
    return ProposedRoute(destination, gateway, device, path)


def build_plans(
    observations: Mapping[str, NodeObservation], topology: Topology
) -> dict[str, NodePlan]:
    """Build complete idempotent plans when the observed graph is a cycle."""
    plans = {name: NodePlan(name) for name in topology.adjacency}
    if not topology.valid_cycle:
        return plans
    networks = sorted(
        topology.subnet_nodes, key=lambda network: int(network.network_address)
    )
    for source in sorted(topology.adjacency):
        for network in networks:
            route = select_route(source, network, observations, topology)
            if route is None:
                continue
            plans[source].routes.append(route)
            path = route.path
            for index in range(1, len(path)):
                relay = path[index]
                incoming_network = _shared_network(path[index - 1], relay, topology)
                if incoming_network is None:
                    continue
                ingress = _interface_for_network(observations[relay], incoming_network)
                if index + 1 < len(path):
                    outgoing_network = _shared_network(relay, path[index + 1], topology)
                else:
                    outgoing_network = network
                if outgoing_network is None:
                    continue
                egress = _interface_for_network(observations[relay], outgoing_network)
                if ingress and egress and ingress != egress:
                    plans[relay].relay_directions.add((ingress, egress))
    return plans


def _route_covering(
    routes: Sequence[Route], network: ipaddress.IPv4Network
) -> list[Route]:
    covering = [
        route
        for route in routes
        if route.destination.prefixlen > 0 and network.subnet_of(route.destination)
    ]
    return sorted(covering, key=lambda route: route.destination.prefixlen, reverse=True)


def _observed_gateway(
    source: str,
    route: Route,
    observations: Mapping[str, NodeObservation],
    topology: Topology,
) -> bool:
    if route.gateway is None or route.device is None:
        return False
    for neighbor in topology.adjacency.get(source, set()):
        shared = _shared_network(source, neighbor, topology)
        if shared is None:
            continue
        address = _address_for_network(observations[neighbor], shared)
        device = _interface_for_network(observations[source], shared)
        if route.gateway == address and route.device == device:
            return True
    return False


def docker_user_accepts(
    rules: Sequence[str] | None, ingress: str, egress: str
) -> bool:
    """Return whether an unrestricted ACCEPT covers one cross-interface flow."""
    if rules is None:
        return False
    for rule in rules:
        try:
            fields = shlex.split(rule)
        except ValueError:
            continue
        if len(fields) < 4 or fields[:2] != ["-A", "DOCKER-USER"]:
            continue
        constraints: dict[str, str] = {}
        harmless = {"-A", "-i", "-o", "-j"}
        index = 2
        supported = True
        while index < len(fields):
            option = fields[index]
            if option not in harmless or index + 1 >= len(fields):
                supported = False
                break
            constraints[option] = fields[index + 1]
            index += 2
        if not supported or constraints.get("-j") != "ACCEPT":
            continue
        if constraints.get("-i", ingress) != ingress:
            continue
        if constraints.get("-o", egress) != egress:
            continue
        return True
    return False


def diagnose_launch_endpoints(
    specs: Sequence[NodeSpec],
    observations: Mapping[str, NodeObservation],
    rendezvous_address: ipaddress.IPv4Address | None = None,
) -> list[Finding]:
    """Check the address and interfaces a distributed launch names per node.

    Two launch inputs are checked against each reachable node: the single
    rendezvous address every rank must connect to, and the socket interface
    names that node gives to NCCL and GLOO for out-of-band bootstrap. Both are
    supplied by the caller. The check is read-only: it reports a node that
    cannot route to the rendezvous address, or that lacks a named interface, or
    whose named interface holds no IPv4 address, and it never proposes a change
    to a route or an interface.

    When at least one node was checked and no such condition was found, the
    result carries one ``info`` finding recording the affirmative outcome, so a
    passing check is distinguishable from a check that did not run. An ``info``
    finding does not affect the process exit code.
    """
    findings: list[Finding] = []
    checked: list[str] = []
    evidence: list[str] = []
    for spec in specs:
        observation = observations.get(spec.name)
        if observation is None or not observation.reachable:
            continue
        if rendezvous_address is None and not spec.socket_interfaces:
            continue
        node_evidence: list[str] = []
        faulted = False
        if rendezvous_address is not None:
            route = observation.rendezvous_route
            if route is None:
                faulted = True
                findings.append(
                    Finding(
                        "warning",
                        "rendezvous-route-unobserved",
                        spec.name,
                        "The route to rendezvous address "
                        f"{rendezvous_address} could not be read, so the node "
                        "is neither confirmed able nor confirmed unable to "
                        "join the rendezvous.",
                        "The probe returned no ip route get section.",
                    )
                )
            elif not route.routable:
                faulted = True
                findings.append(
                    Finding(
                        "error",
                        "rendezvous-unroutable",
                        spec.name,
                        "The node has no route to rendezvous address "
                        f"{rendezvous_address}. Ranks placed here cannot reach "
                        "the rendezvous store, so the launch stalls until the "
                        "rendezvous timeout expires on every other node.",
                        f"ip route get {rendezvous_address} reported: {route.raw}",
                    )
                )
            else:
                node_evidence.append(
                    f"{rendezvous_address} via dev {route.device} "
                    f"src {route.source if route.source else 'unreported'}"
                )
        for name in spec.socket_interfaces:
            state = observation.host_interfaces.get(name)
            if state is None:
                faulted = True
                findings.append(
                    Finding(
                        "error",
                        "socket-interface-absent",
                        spec.name,
                        f"Socket interface {name} is named for NCCL and GLOO "
                        "bootstrap on this node but does not exist here, so "
                        "bootstrap cannot bind and the ranks on this node "
                        "never join the job.",
                        "observed interfaces: "
                        + (
                            ", ".join(sorted(observation.host_interfaces))
                            if observation.host_interfaces
                            else "none"
                        ),
                    )
                )
            elif not state.addresses:
                faulted = True
                findings.append(
                    Finding(
                        "error",
                        "socket-interface-unaddressed",
                        spec.name,
                        f"Socket interface {name} is named for NCCL and GLOO "
                        "bootstrap on this node and exists here, but carries "
                        "no IPv4 address, so bootstrap has no address to bind "
                        "and advertise.",
                        f"observed state={state.oper_state}, IPv4 addresses=none",
                    )
                )
            else:
                node_evidence.append(
                    f"{name}="
                    + ",".join(str(address) for address in state.addresses)
                )
        checked.append(spec.name)
        if not faulted:
            evidence.append(f"{spec.name}: " + "; ".join(node_evidence))
    if not checked or findings:
        return findings
    clauses = []
    if rendezvous_address is not None:
        clauses.append(f"routes to rendezvous address {rendezvous_address}")
    if any(spec.socket_interfaces for spec in specs):
        clauses.append(
            "holds every socket interface named for it with an IPv4 address"
        )
    findings.append(
        Finding(
            "info",
            "launch-endpoints-observed",
            None,
            "Every reachable node " + " and ".join(clauses) + ".",
            " | ".join(evidence),
        )
    )
    return findings


def diagnose_wifi_resilience(
    specs: Sequence[NodeSpec],
    observations: Mapping[str, NodeObservation],
) -> list[Finding]:
    """Check that wireless management interfaces survive an access-point blip.

    A wireless management interface recovers from an access-point restart on
    its own only when its active NetworkManager profile autoconnects, retries
    without a cap, and runs with Wi-Fi power save disabled. Any other
    combination is a latent fault: every reachability check passes while the
    interface is associated, then one blip leaves the node off the management
    network until someone reconnects it by hand. Every wireless interface with
    an active NetworkManager connection on a reachable node is checked, from
    evidence the discovery probe gathered in its single contact per node. The
    check reports observations only and never proposes a profile change. A
    node with no such interface contributes nothing, because wired management
    is not a fault; a node without ``nmcli`` is reported as unobserved,
    because nothing about its wireless configuration could be read.

    When at least one wireless interface was checked and nothing was faulted
    or unreadable, the result carries one ``info`` finding recording the
    affirmative outcome, so a passing check is distinguishable from a site
    with wired-only management, where the check contributes nothing. An
    ``info`` finding does not affect the process exit code.
    """
    findings: list[Finding] = []
    sound: list[str] = []
    for spec in specs:
        observation = observations.get(spec.name)
        if observation is None or not observation.reachable:
            continue
        wifi = observation.wifi
        if wifi is None:
            continue
        if not wifi.nmcli_available:
            findings.append(
                Finding(
                    "warning",
                    "wifi-resilience-unobserved",
                    spec.name,
                    "nmcli is not installed on this node, so no wireless "
                    "management interface here can be checked for the "
                    "NetworkManager settings that reconnect it after an "
                    "access-point restart.",
                    "command -v nmcli found no executable",
                )
            )
            continue
        for interface in wifi.interfaces:
            if (
                interface.autoconnect is None
                or interface.autoconnect_retries is None
                or interface.powersave is None
            ):
                findings.append(
                    Finding(
                        "warning",
                        "wifi-resilience-unobserved",
                        spec.name,
                        f"NetworkManager reports profile "
                        f"{interface.connection} active on wireless "
                        f"interface {interface.device}, but the profile's "
                        "reconnection settings could not be read, so the "
                        "interface is neither confirmed resilient nor "
                        "confirmed faulted.",
                        "nmcli -t -g connection.autoconnect,"
                        "connection.autoconnect-retries,"
                        "802-11-wireless.powersave connection show "
                        f"{interface.connection} returned no values",
                    )
                )
                continue
            faulted = False
            if interface.autoconnect != "yes":
                faulted = True
                findings.append(
                    Finding(
                        "warning",
                        "wifi-autoconnect-disabled",
                        spec.name,
                        f"NetworkManager profile {interface.connection} on "
                        f"wireless interface {interface.device} sets "
                        f"connection.autoconnect={interface.autoconnect}, so "
                        "after any disconnect NetworkManager does not rejoin "
                        "the network and the node stays off the management "
                        "network until the profile is activated by hand.",
                        f"connection.autoconnect={interface.autoconnect}",
                    )
                )
            try:
                retries: int | None = int(interface.autoconnect_retries)
            except ValueError:
                retries = None
            if retries is None:
                faulted = True
                findings.append(
                    Finding(
                        "warning",
                        "wifi-resilience-unobserved",
                        spec.name,
                        f"NetworkManager profile {interface.connection} on "
                        f"wireless interface {interface.device} reports "
                        "connection.autoconnect-retries="
                        f"{interface.autoconnect_retries}, which is not an "
                        "integer, so the reconnection cap here cannot be "
                        "determined.",
                        "connection.autoconnect-retries="
                        f"{interface.autoconnect_retries}",
                    )
                )
            elif retries != 0:
                faulted = True
                if retries == -1:
                    cap_clause = (
                        "connection.autoconnect-retries=-1, the global "
                        "default of four reconnection attempts"
                    )
                    count = "four"
                else:
                    cap_clause = (
                        f"connection.autoconnect-retries={retries}, a finite "
                        f"cap of {retries} reconnection attempts"
                    )
                    count = str(retries)
                findings.append(
                    Finding(
                        "warning",
                        "wifi-retries-limited",
                        spec.name,
                        f"NetworkManager profile {interface.connection} on "
                        f"wireless interface {interface.device} sets "
                        f"{cap_clause}. nm-settings defines zero as retry "
                        f"forever; after {count} failed attempts autoconnect "
                        "blocks until a NetworkManager timeout expires, so an "
                        "access-point outage that outlasts them leaves the "
                        "node off the management network for that window.",
                        "connection.autoconnect-retries="
                        f"{interface.autoconnect_retries}",
                    )
                )
            profile_powersave_on = interface.powersave in {"3", "enable"}
            driver_powersave_on = interface.driver_power_save == "on"
            if profile_powersave_on or driver_powersave_on:
                faulted = True
                if profile_powersave_on:
                    cause = (
                        f"NetworkManager profile {interface.connection} on "
                        f"wireless interface {interface.device} enables "
                        "Wi-Fi power save (802-11-wireless.powersave="
                        f"{interface.powersave})"
                    )
                else:
                    cause = (
                        "The wireless driver reports power save on for "
                        f"interface {interface.device}, whose active "
                        f"NetworkManager profile {interface.connection} sets "
                        "802-11-wireless.powersave="
                        f"{interface.powersave}"
                    )
                findings.append(
                    Finding(
                        "warning",
                        "wifi-powersave-enabled",
                        spec.name,
                        f"{cause}. Wi-Fi power save causes silent "
                        "de-association from the access point, so the node "
                        "can drop off the management network with no failure "
                        "recorded.",
                        f"802-11-wireless.powersave={interface.powersave}; "
                        "iw power_save="
                        + (interface.driver_power_save or "unobserved"),
                    )
                )
            if not faulted:
                sound.append(
                    f"{spec.name}:{interface.device} profile "
                    f"{interface.connection} autoconnect="
                    f"{interface.autoconnect} retries="
                    f"{interface.autoconnect_retries} powersave="
                    f"{interface.powersave} driver_power_save="
                    + (interface.driver_power_save or "unobserved")
                )
    if not sound or findings:
        return findings
    findings.append(
        Finding(
            "info",
            "wifi-resilience-observed",
            None,
            "Every wireless interface with an active NetworkManager "
            "connection on a reachable node autoconnects, retries without "
            "limit, and runs with Wi-Fi power save disabled, so management "
            "over Wi-Fi survives an access-point restart without manual "
            "action.",
            " | ".join(sound),
        )
    )
    return findings


def diagnose(
    specs: Sequence[NodeSpec],
    observations: Mapping[str, NodeObservation],
    topology: Topology,
    plans: Mapping[str, NodePlan],
    interfaces: Sequence[str] = FABRIC_INTERFACES,
    rendezvous_address: ipaddress.IPv4Address | None = None,
) -> list[Finding]:
    """Evaluate required ring invariants and retain evidence for every fault."""
    findings: list[Finding] = []
    for spec in specs:
        observation = observations[spec.name]
        if not observation.reachable:
            findings.append(
                Finding(
                    "error",
                    "node-unreachable",
                    spec.name,
                    "The node did not answer a non-interactive SSH probe. "
                    "Hostname, fabric interface, route, IPv4 forwarding, "
                    "FORWARD policy, and DOCKER-USER checks are unknown.",
                    observation.error,
                )
            )
            continue
        for interface_name in interfaces:
            state = observation.interfaces.get(interface_name)
            if state is None:
                findings.append(
                    Finding(
                        "error",
                        "interface-unobserved",
                        spec.name,
                        f"Fabric interface {interface_name} was not observed.",
                        "The ip -br -4 addr output contained no matching row.",
                    )
                )
            elif state.oper_state != "UP":
                findings.append(
                    Finding(
                        "error",
                        "interface-down",
                        spec.name,
                        f"Fabric interface {interface_name} is not operationally up.",
                        f"observed state={state.oper_state}",
                    )
                )
    if not topology.valid_cycle:
        findings.append(
            Finding(
                "error",
                "topology-not-cycle",
                None,
                "The discovered fabric does not form one cycle.",
                topology.reason,
            )
        )

    for network, members in topology.subnet_nodes.items():
        if len(members) != 2:
            findings.append(
                Finding(
                    "warning",
                    "subnet-endpoints-unknown",
                    None,
                    f"Fabric subnet {network} does not have two observed endpoints.",
                    f"observed nodes={', '.join(members) if members else 'none'}",
                )
            )
            continue
        left, right = members
        left_interface = _interface_for_network(observations[left], network)
        right_interface = _interface_for_network(observations[right], network)
        if not left_interface or not right_interface:
            continue
        left_mtu = observations[left].interfaces[left_interface].mtu
        right_mtu = observations[right].interfaces[right_interface].mtu
        if left_mtu is not None and right_mtu is not None and left_mtu != right_mtu:
            findings.append(
                Finding(
                    "error",
                    "link-mtu-mismatch",
                    None,
                    f"Fabric subnet {network} has mismatched endpoint MTUs.",
                    f"{left}:{left_interface}={left_mtu}, "
                    f"{right}:{right_interface}={right_mtu}",
                )
            )

    if topology.valid_cycle:
        for name, plan in plans.items():
            observation = observations[name]
            for proposed in plan.routes:
                covering = _route_covering(observation.routes, proposed.destination)
                if not covering:
                    findings.append(
                        Finding(
                            "error",
                            "route-missing",
                            name,
                            f"No kernel route covers fabric subnet {proposed.destination}.",
                            "ip -4 route show returned no covering prefix",
                        )
                    )
                elif not any(
                    _observed_gateway(name, route, observations, topology)
                    for route in covering
                ):
                    findings.append(
                        Finding(
                            "error",
                            "route-next-hop-unobserved",
                            name,
                            f"The route to {proposed.destination} does not use an "
                            "observed neighbor address and matching fabric interface.",
                            "; ".join(route.raw for route in covering),
                        )
                    )
            if not plan.relay_directions:
                continue
            if observation.ip_forward is False:
                findings.append(
                    Finding(
                        "error",
                        "ip-forward-disabled",
                        name,
                        "IPv4 forwarding is disabled on a node required to relay traffic.",
                        "net.ipv4.ip_forward=0",
                    )
                )
            elif observation.ip_forward is None:
                findings.append(
                    Finding(
                        "warning",
                        "ip-forward-unknown",
                        name,
                        "IPv4 forwarding could not be read on a required relay.",
                        "The sysctl probe returned no boolean value.",
                    )
                )
            if observation.forward_policy is None:
                findings.append(
                    Finding(
                        "warning",
                        "forward-policy-unknown",
                        name,
                        "The FORWARD chain policy could not be read.",
                        "The iptables FORWARD probe returned no policy.",
                    )
                )
            if observation.docker_user_rules is None:
                findings.append(
                    Finding(
                        "warning",
                        "docker-user-rules-unknown",
                        name,
                        "The DOCKER-USER rules could not be read.",
                        "The iptables DOCKER-USER probe was unavailable.",
                    )
                )
            if observation.forward_policy == "DROP":
                for ingress, egress in sorted(plan.relay_directions):
                    if not docker_user_accepts(
                        observation.docker_user_rules, ingress, egress
                    ):
                        findings.append(
                            Finding(
                                "error",
                                "docker-user-cross-accept-missing",
                                name,
                                "The FORWARD policy is DROP and no unrestricted "
                                f"DOCKER-USER ACCEPT covers {ingress} to {egress}.",
                                "Required relay direction: "
                                f"-i {ingress} -o {egress}; observed rules: "
                                + (
                                    " | ".join(observation.docker_user_rules)
                                    if observation.docker_user_rules
                                    else "none"
                                ),
                            )
                        )
    findings.extend(
        diagnose_launch_endpoints(specs, observations, rendezvous_address)
    )
    findings.extend(diagnose_wifi_resilience(specs, observations))
    return findings


def _probe_by_name(
    specs: Sequence[NodeSpec],
    runner: Runner,
    interfaces: Sequence[str],
    rendezvous_address: ipaddress.IPv4Address | None = None,
) -> tuple[dict[str, NodeObservation], Topology, dict[str, NodePlan], list[Finding]]:
    observations = discover_nodes(specs, runner, interfaces, rendezvous_address)
    topology = infer_topology(observations)
    plans = build_plans(observations, topology)
    findings = diagnose(
        specs, observations, topology, plans, interfaces, rendezvous_address
    )
    return observations, topology, plans, findings


def apply_plans(
    observations: Mapping[str, NodeObservation],
    plans: Mapping[str, NodePlan],
    runner: Runner,
) -> list[Finding]:
    """Execute complete plans through the SSH paths proven by discovery."""
    findings: list[Finding] = []
    for name, plan in sorted(plans.items()):
        commands = plan.commands()
        if not commands:
            continue
        observation = observations[name]
        result = runner.run(
            observation.spec.target,
            "set -e\n" + "\n".join(commands),
            observation.proxy_jump_used,
        )
        if not result.ok:
            findings.append(
                Finding(
                    "error",
                    "apply-failed",
                    name,
                    "The idempotent repair plan did not complete.",
                    result.detail or result.stderr.strip() or "remote command failed",
                )
            )
    return findings


def _unit_program(observation: NodeObservation, plan: NodePlan) -> str:
    expected = {
        name: sorted(str(address) for address in interface.addresses)
        for name, interface in sorted(observation.interfaces.items())
    }
    route_argv = [
        [
            "ip",
            "route",
            "replace",
            str(route.destination),
            "via",
            str(route.gateway),
            "dev",
            route.device,
        ]
        for route in plan.routes
    ]
    directions = [list(direction) for direction in sorted(plan.relay_directions)]
    return f'''#!/usr/bin/env python3
"""Reapply a recorded SparkRing fabric plan after exact address validation."""

import ipaddress
import json
import subprocess
import sys

EXPECTED = {expected!r}
ROUTES = {route_argv!r}
RELAY_DIRECTIONS = {directions!r}


def observed_addresses(interface):
    result = subprocess.run(
        ["ip", "-j", "-4", "addr", "show", "dev", interface],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(result.stdout)
    addresses = []
    for row in rows:
        for item in row.get("addr_info", []):
            if item.get("family") == "inet":
                addresses.append(
                    str(ipaddress.ip_interface(
                        f"{{item['local']}}/{{item['prefixlen']}}"
                    ))
                )
    return sorted(addresses)


def main():
    for interface, expected in EXPECTED.items():
        try:
            actual = observed_addresses(interface)
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            print(f"ADDRESS CHECK FAILED {{interface}}: {{exc}}", file=sys.stderr)
            return 1
        if actual != expected:
            print(
                f"ADDRESS MISMATCH {{interface}}: expected={{expected}} actual={{actual}}",
                file=sys.stderr,
            )
            return 1
    for command in ROUTES:
        subprocess.run(command, check=True)
    if RELAY_DIRECTIONS:
        subprocess.run(
            ["sysctl", "-w", "net.ipv4.ip_forward=1"], check=True
        )
    for ingress, egress in RELAY_DIRECTIONS:
        rule = ["-i", ingress, "-o", egress, "-j", "ACCEPT"]
        check = subprocess.run(
            ["iptables", "-C", "DOCKER-USER", *rule], check=False
        )
        if check.returncode:
            subprocess.run(
                ["iptables", "-I", "DOCKER-USER", "1", *rule], check=True
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def emit_units(
    directory: Path,
    observations: Mapping[str, NodeObservation],
    plans: Mapping[str, NodePlan],
) -> list[str]:
    """Write one fail-closed systemd service and program per observed node."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    for name, plan in sorted(plans.items()):
        observation = observations[name]
        slug = re.sub(r"[^A-Za-z0-9_.-]", "-", name)
        program_path = (directory / f"ring-doctor-{slug}.py").resolve()
        unit_path = directory / f"ring-doctor-{slug}.service"
        program_path.write_text(_unit_program(observation, plan), encoding="utf-8")
        os.chmod(program_path, 0o755)
        unit = f"""[Unit]
Description=Reapply verified SparkRing fabric plan for {name}
Wants=network-online.target
After=network-online.target docker.service

[Service]
Type=oneshot
ExecStart=/usr/bin/python3 {program_path}
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
"""
        unit_path.write_text(unit, encoding="utf-8")
        written.extend((str(program_path), str(unit_path.resolve())))
    return written


def verify_reachability(
    specs: Sequence[NodeSpec],
    observations: Mapping[str, NodeObservation],
    runner: Runner,
) -> dict[str, Any]:
    """Ping every observed fabric address from every reachable node."""
    matrix: dict[str, dict[str, bool | None]] = {}
    probes: list[dict[str, Any]] = []
    for source_spec in specs:
        source = observations[source_spec.name]
        matrix[source_spec.name] = {}
        for destination_spec in specs:
            destination = observations[destination_spec.name]
            if source_spec.name == destination_spec.name:
                matrix[source_spec.name][destination_spec.name] = (
                    True if source.reachable else None
                )
                continue
            if not source.reachable or not destination.reachable:
                matrix[source_spec.name][destination_spec.name] = None
                continue
            addresses = sorted(
                {
                    str(address.ip)
                    for interface in destination.interfaces.values()
                    for address in interface.addresses
                },
                key=ipaddress.ip_address,
            )
            if not addresses:
                matrix[source_spec.name][destination_spec.name] = None
                continue
            pair_ok = True
            for address in addresses:
                result = runner.run(
                    source.spec.target,
                    f"ping -n -c 1 -W 2 {address}",
                    source.proxy_jump_used,
                )
                probes.append(
                    {
                        "source": source_spec.name,
                        "destination": destination_spec.name,
                        "address": address,
                        "reachable": result.ok,
                        "evidence": result.detail or result.stderr.strip(),
                    }
                )
                pair_ok = pair_ok and result.ok
            matrix[source_spec.name][destination_spec.name] = pair_ok
    failed = any(value is False for row in matrix.values() for value in row.values())
    unknown = any(value is None for row in matrix.values() for value in row.values())
    return {"matrix": matrix, "probes": probes, "failed": failed, "unknown": unknown}


def _render_matrix(matrix: Mapping[str, Mapping[str, bool | None]]) -> list[str]:
    nodes = list(matrix)
    width = max(7, *(len(node) for node in nodes))
    lines = ["Reachability matrix:"]
    lines.append(" " * (width + 1) + " ".join(f"{node:>{width}}" for node in nodes))
    for source in nodes:
        cells = []
        for destination in nodes:
            value = matrix[source][destination]
            cells.append("PASS" if value is True else "FAIL" if value is False else "UNKNOWN")
        lines.append(
            f"{source:>{width}} " + " ".join(f"{cell:>{width}}" for cell in cells)
        )
    return lines


def render_text(report: Mapping[str, Any]) -> str:
    """Render the complete report for an operator without hidden context."""
    lines = ["SparkRing fabric diagnosis"]
    topology = report["topology"]
    if topology["valid_cycle"]:
        cycle = topology["cycle"]
        lines.append("Ring cycle: " + " -> ".join(cycle + cycle[:1]))
    else:
        lines.append(f"Ring cycle: UNKNOWN ({topology['reason']})")
    rendezvous_address = report.get("rendezvous_address")
    lines.append(
        f"Rendezvous address: {rendezvous_address}"
        if rendezvous_address
        else "Rendezvous address: none named, so no node was checked for a route to one"
    )
    lines.append("Nodes:")
    for name, node in report["nodes"].items():
        status = "reachable" if node["reachable"] else "unreachable"
        hostname = f" hostname={node['hostname']}" if node["hostname"] else ""
        jump = (
            f" proxy_jump={node['proxy_jump_used']}"
            if node["proxy_jump_used"]
            else ""
        )
        lines.append(f"  {name}: {status}{hostname}{jump}")
    findings = report["findings"]
    lines.append(f"Findings: {len(findings)}")
    for finding in findings:
        scope = f" {finding['node']}" if finding["node"] else ""
        lines.append(
            f"  [{finding['severity'].upper()}] {finding['code']}{scope}: "
            f"{finding['message']}"
        )
        lines.append(f"    evidence: {finding['evidence']}")
    lines.append("Repair plan (read-only output unless --apply is present):")
    commands_printed = 0
    for name, plan in report["plans"].items():
        for command in plan["commands"]:
            lines.append(f"  {name}: {command}")
            commands_printed += 1
    if not commands_printed:
        lines.append("  No verified repair commands are available.")
    if report.get("unit_files"):
        lines.append("Generated systemd files:")
        lines.extend(f"  {path}" for path in report["unit_files"])
    apply_status = report["apply"]
    if apply_status["requested"]:
        outcome = "executed" if apply_status["executed"] else "withheld"
        lines.append(f"Repair application: {outcome}")
    if report.get("verification"):
        lines.extend(_render_matrix(report["verification"]["matrix"]))
    return "\n".join(lines)


def _parse_rendezvous_address(value: str, field: str) -> ipaddress.IPv4Address:
    try:
        address = ipaddress.ip_address(value)
    except ValueError as exc:
        raise ConfigurationError(
            f"{field} must be an IPv4 address literal"
        ) from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise ConfigurationError(f"{field} must be an IPv4 address literal")
    return address


def _load_inputs(
    args: argparse.Namespace,
) -> tuple[tuple[NodeSpec, ...], tuple[str, ...], int, ipaddress.IPv4Address | None]:
    interfaces = tuple(args.interface or FABRIC_INTERFACES)
    timeout = args.ssh_timeout
    rendezvous_address = (
        _parse_rendezvous_address(args.rendezvous_address, "--rendezvous-address")
        if args.rendezvous_address
        else None
    )
    if args.config:
        try:
            document = json.loads(Path(args.config).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(f"could not read --config JSON: {exc}") from exc
        if not isinstance(document, Mapping):
            raise ConfigurationError("config root must be a JSON object")
        specs = parse_node_specs(document)
        configured_interfaces = document.get("fabric_interfaces")
        if configured_interfaces is not None:
            if not isinstance(configured_interfaces, list) or len(configured_interfaces) != 2:
                raise ConfigurationError("config.fabric_interfaces must contain two names")
            checked_interfaces: list[str] = []
            for index, value in enumerate(configured_interfaces):
                if not isinstance(value, str):
                    raise ConfigurationError(
                        f"config.fabric_interfaces[{index}] must be a string"
                    )
                checked_interfaces.append(
                    _validate_interface(
                        value, f"config.fabric_interfaces[{index}]"
                    )
                )
            interfaces = tuple(checked_interfaces)
        configured_timeout = document.get("ssh_timeout_seconds")
        if configured_timeout is not None:
            if not isinstance(configured_timeout, int) or not 1 <= configured_timeout <= 3600:
                raise ConfigurationError(
                    "config.ssh_timeout_seconds must be an integer from 1 to 3600"
                )
            timeout = configured_timeout
        configured_rendezvous = document.get("rendezvous_address")
        if configured_rendezvous is not None:
            if not isinstance(configured_rendezvous, str):
                raise ConfigurationError(
                    "config.rendezvous_address must be a string"
                )
            rendezvous_address = _parse_rendezvous_address(
                configured_rendezvous, "config.rendezvous_address"
            )
    else:
        if not args.node:
            raise ConfigurationError("provide --config or at least one --node user@host")
        sockets: dict[str, tuple[str, ...]] = {}
        for item in args.socket_interface or []:
            if "=" not in item:
                raise ConfigurationError(
                    "--socket-interface must have the form NODE=NAME[,NAME]"
                )
            socket_target, socket_names = item.split("=", 1)
            socket_target = _validate_ssh_target(
                socket_target, "--socket-interface NODE"
            )
            parsed_names = tuple(name for name in socket_names.split(",") if name)
            if not parsed_names:
                raise ConfigurationError(
                    f"--socket-interface {socket_target} names no interface"
                )
            for index, name in enumerate(parsed_names):
                _validate_interface(
                    name, f"--socket-interface {socket_target} name {index + 1}"
                )
            sockets[socket_target] = parsed_names
        jumps: dict[str, str] = {}
        for item in args.proxy_jump or []:
            if "=" not in item:
                raise ConfigurationError("--proxy-jump must have the form NODE=JUMP")
            target, jump = item.split("=", 1)
            jumps[_validate_ssh_target(target, "--proxy-jump NODE")] = (
                _validate_ssh_jump_chain(jump, "--proxy-jump JUMP")
            )
        specs = tuple(
            NodeSpec(
                _default_name(_validate_ssh_target(target, "--node")),
                target,
                jumps.get(target),
                sockets.get(target, ()),
            )
            for target in args.node
        )
        _require_unique_specs(specs)
        unused = sorted(set(jumps) - {spec.target for spec in specs})
        if unused:
            raise ConfigurationError(
                "--proxy-jump names targets absent from --node: " + ", ".join(unused)
            )
        unused_sockets = sorted(set(sockets) - {spec.target for spec in specs})
        if unused_sockets:
            raise ConfigurationError(
                "--socket-interface names targets absent from --node: "
                + ", ".join(unused_sockets)
            )
    if len(interfaces) != 2 or len(set(interfaces)) != 2:
        raise ConfigurationError("exactly two distinct fabric interfaces are required")
    for index, interface in enumerate(interfaces):
        _validate_interface(interface, f"fabric_interfaces[{index}]")
    if not isinstance(timeout, int) or not 1 <= timeout <= 3600:
        raise ConfigurationError("--ssh-timeout must be an integer from 1 to 3600")
    return specs, interfaces, timeout, rendezvous_address


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ring-doctor",
        description=(
            "Discover and diagnose a four-node switchless SparkRing fabric. "
            "The default operation is read-only and prints a repair plan."
        ),
    )
    parser.add_argument("--config", help="JSON file containing node SSH targets")
    parser.add_argument(
        "--node", action="append", metavar="USER@HOST", help="seed SSH target; repeatable"
    )
    parser.add_argument(
        "--proxy-jump",
        action="append",
        metavar="NODE=JUMP",
        help="per-node ProxyJump assignment for --node inputs; repeatable",
    )
    parser.add_argument(
        "--interface",
        action="append",
        metavar="NAME",
        help="fabric interface name; specify exactly twice to override defaults",
    )
    parser.add_argument(
        "--rendezvous-address",
        metavar="ADDR",
        help=(
            "IPv4 address every rank connects to for distributed rendezvous, "
            "the value passed to torch as --master-addr; each node is checked "
            "for a route to it. A config file entry overrides this flag."
        ),
    )
    parser.add_argument(
        "--socket-interface",
        action="append",
        metavar="NODE=NAME[,NAME]",
        help=(
            "interface names a node gives to NCCL_SOCKET_IFNAME and "
            "GLOO_SOCKET_IFNAME, checked for existence and an IPv4 address on "
            "that node; applies to --node inputs and is repeatable. Use the "
            "per-node socket_interfaces key with --config."
        ),
    )
    parser.add_argument(
        "--ssh-timeout", type=int, default=20, help="SSH command timeout in seconds"
    )
    parser.add_argument(
        "--apply", action="store_true", help="execute the verified idempotent repair plan"
    )
    parser.add_argument(
        "--emit-unit",
        metavar="DIR",
        help="write a fail-closed systemd service and program for each node",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="ping every observed fabric address from every reachable node",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit the complete report as JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        specs, interfaces, timeout, rendezvous_address = _load_inputs(args)
    except ConfigurationError as exc:
        print(f"INVALID RING DOCTOR INPUT: {exc}", file=sys.stderr)
        return 2
    runner = SshRunner(timeout_seconds=timeout)
    observations, topology, plans, findings = _probe_by_name(
        specs, runner, interfaces, rendezvous_address
    )
    apply_findings: list[Finding] = []
    apply_executed = False
    enough = len(specs) == 4 and all(
        observation.reachable for observation in observations.values()
    ) and topology.valid_cycle
    if args.apply:
        if enough:
            apply_executed = True
            apply_findings = apply_plans(observations, plans, runner)
            observations, topology, plans, findings = _probe_by_name(
                specs, runner, interfaces, rendezvous_address
            )
            enough = len(specs) == 4 and all(
                observation.reachable for observation in observations.values()
            ) and topology.valid_cycle
        else:
            apply_findings.append(
                Finding(
                    "error",
                    "apply-withheld",
                    None,
                    "No repair was applied because the complete four-node cycle was not observed.",
                    topology.reason,
                )
            )
    unit_files: list[str] = []
    if args.emit_unit:
        if enough:
            try:
                unit_files = emit_units(Path(args.emit_unit), observations, plans)
            except OSError as exc:
                apply_findings.append(
                    Finding(
                        "error",
                        "unit-write-failed",
                        None,
                        "The systemd files could not be written.",
                        str(exc),
                    )
                )
        else:
            apply_findings.append(
                Finding(
                    "error",
                    "unit-withheld",
                    None,
                    "No systemd files were written because the complete four-node cycle was not observed.",
                    topology.reason,
                )
            )
    findings.extend(apply_findings)
    verification = (
        verify_reachability(specs, observations, runner) if args.verify else None
    )
    report = {
        "schema": "sparkring-ring-doctor/v1",
        "rendezvous_address": (
            str(rendezvous_address) if rendezvous_address else None
        ),
        "nodes": {
            name: observation.to_dict()
            for name, observation in observations.items()
        },
        "topology": topology.to_dict(),
        "findings": [finding.to_dict() for finding in findings],
        "plans": {name: plan.to_dict() for name, plan in plans.items()},
        "apply": {
            "requested": args.apply,
            "executed": apply_executed,
            "failures": [
                finding.to_dict()
                for finding in apply_findings
                if finding.code == "apply-failed"
            ],
        },
        "unit_files": unit_files,
        "verification": verification,
        "discovery_sufficient": enough,
    }
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(render_text(report))
    if not enough:
        return 2
    if any(finding.severity != "info" for finding in findings):
        return 1
    if verification and (verification["failed"] or verification["unknown"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
