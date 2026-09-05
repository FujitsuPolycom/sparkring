#!/usr/bin/env python3
"""Plan and orchestrate a four-rank ConnectX-7 hardware diagonal.

Status: research-only. Local planning and validation are implemented. Remote
operations are restricted to one configured helper on each of four topology
hosts; this module contains no Docker or model command.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


TOPOLOGY_SCHEMA = "sparkring-cx7-hardware-diagonal-fabric/v1"
PLAN_SCHEMA = "sparkring-cx7-hardware-diagonal-plan/v1"
RECEIPT_SCHEMA = "sparkring-cx7-hardware-diagonal-receipt/v1"
HARDWARE_GATE_SCHEMA = "sparkring-cx7-ethertype-hardware-gate/v1"
APPLY_MANIFEST_SCHEMA = "sparkring-cx7-hardware-diagonal-apply-manifest/v1"
CLEANUP_MANIFEST_SCHEMA = "sparkring-cx7-hardware-diagonal-cleanup-manifest/v1"
AUTHORIZATION_TOKEN = "APPLY_TP4_HARDWARE_DIAGONALS"
RANK_COUNT = 4
MAX_RUNTIME_SECONDS = 7200
ROCE_SOURCE_PORT_BASE = 49152
ROCE_SOURCE_PORT_MASK = 0x3FFF
FLOW_LABEL_LIMIT = ROCE_SOURCE_PORT_MASK
FILTER_ID_BASE = 49152
ETH_P_IP = 0x0800
ETH_P_DIAGONAL = 0x88B5

_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_NETDEV = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,14}\Z")
_RDMA_DEVICE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")
_MAC = re.compile(r"(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\Z")


class FabricError(ValueError):
    """The fabric request is incomplete, unsafe, or internally inconsistent."""


@dataclasses.dataclass(frozen=True)
class Port:
    """One host-facing function for one physical cycle direction."""

    direction: str
    function: int
    netdev: str
    rdma_device: str
    ipv4: str
    mac: str
    peer_rank: int
    peer_direction: str
    peer_function: int


@dataclasses.dataclass(frozen=True)
class Rank:
    """One TP rank and its explicitly connected fabric functions."""

    rank: int
    ssh_alias: str
    management_netdev: str
    ports: tuple[Port, ...]

    def port(self, direction: str, function: int) -> Port:
        for port in self.ports:
            if port.direction == direction and port.function == function:
                return port
        raise FabricError(
            f"rank {self.rank} lacks {direction} Socket-Direct function {function}"
        )


@dataclasses.dataclass(frozen=True)
class FabricTopology:
    """Validated four-rank input and its byte identity."""

    source_path: Path
    sha256: str
    group_id: int
    socket_direct_functions: int
    flow_label_base: int
    shared_diagonal_flow_label: bool
    endpoint_route_strategy: str
    tc_rule_identity_overrides: tuple["TcRuleIdentityOverride", ...]
    standard_ether_type: int
    marked_ether_type: int
    standard_udp_destination_port: int
    roce_gid_index: int
    bounded_runtime_seconds: int
    expected_ethernet_mtu: int
    expected_roce_mtu: int
    host_helper_path: str
    remote_state_root: str
    ranks: tuple[Rank, ...]

    def rank(self, value: int) -> Rank:
        return self.ranks[value]


@dataclasses.dataclass(frozen=True)
class FabricPath:
    """One directed direct or two-link logical path."""

    name: str
    kind: str
    source_rank: int
    destination_rank: int
    direction: str
    function: int
    intermediate_rank: int | None
    source_netdev: str
    source_rdma_device: str
    source_ipv4: str
    destination_ipv4: str
    next_hop_mac: str
    final_destination_mac: str


@dataclasses.dataclass(frozen=True)
class EndpointRoute:
    """One source host route and permanent next-hop neighbor entry."""

    path_name: str
    source_rank: int
    source_netdev: str
    source_ipv4: str
    destination_ipv4: str
    next_hop_mac: str
    gateway_ipv4: str | None
    permanent_final_neighbor: bool


@dataclasses.dataclass(frozen=True)
class SourceMarker:
    """Packet identity assigned to a diagonal path's RDMA-TX marker.

    The native source-port helper matches the reserved UDP source port on
    a function, not every IP/QPN field represented by this descriptor.
    """

    path_name: str
    source_rank: int
    rdma_device: str
    source_ipv4: str
    destination_ipv4: str
    flow_label: int
    udp_source_port: int
    match_ether_type: int
    marked_ether_type: int
    match_udp_destination_port: int


@dataclasses.dataclass(frozen=True)
class TcRestoreRule:
    """One middle-rank EtherType restore, MAC rewrite, and redirect rule."""

    name: str
    path_name: str
    source_rank: int
    destination_rank: int
    intermediate_rank: int
    direction: str
    function: int
    ingress_netdev: str
    egress_netdev: str
    preference: int
    handle: int
    match_ethernet_source: str
    match_ethernet_destination: str
    match_ether_type: int
    restore_ether_type: int
    rewrite_ethernet_destination: str


@dataclasses.dataclass(frozen=True)
class TcRuleIdentityOverride:
    """Exact tc preference and handle assigned to one semantic diagonal path."""

    path_name: str
    preference: int
    handle: int


@dataclasses.dataclass(frozen=True)
class FabricPlan:
    """Complete deterministic plan for one topology document."""

    topology_sha256: str
    group_id: int
    socket_direct_functions: int
    shared_diagonal_flow_label: bool
    endpoint_route_strategy: str
    bounded_runtime_seconds: int
    expected_ethernet_mtu: int
    expected_roce_mtu: int
    roce_gid_index: int
    host_helper_path: str
    remote_state_root: str
    hardware_gate_status: str
    apply_permitted: bool
    ranks: tuple[Rank, ...]
    paths: tuple[FabricPath, ...]
    routes: tuple[EndpointRoute, ...]
    markers: tuple[SourceMarker, ...]
    tc_rules: tuple[TcRestoreRule, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return the stable JSON-compatible plan representation."""

        return {
            "schema": PLAN_SCHEMA,
            "status": "research-only",
            "topology_sha256": self.topology_sha256,
            "group_id": self.group_id,
            "socket_direct_functions": self.socket_direct_functions,
            "shared_diagonal_flow_label": self.shared_diagonal_flow_label,
            "endpoint_route_strategy": self.endpoint_route_strategy,
            "bounded_runtime_seconds": self.bounded_runtime_seconds,
            "expected_ethernet_mtu": self.expected_ethernet_mtu,
            "expected_roce_mtu": self.expected_roce_mtu,
            "roce_gid_index": self.roce_gid_index,
            "host_helper_path": self.host_helper_path,
            "remote_state_root": self.remote_state_root,
            "hardware_gate_status": self.hardware_gate_status,
            "apply_permitted": self.apply_permitted,
            "negative_controls": {
                "udp_port_mark_and_restore": {
                    "status": "unsupported",
                    "executable": False,
                    "reason": "post-ICRC inner-header modification",
                }
            },
            "ranks": [dataclasses.asdict(item) for item in self.ranks],
            "paths": [dataclasses.asdict(item) for item in self.paths],
            "routes": [dataclasses.asdict(item) for item in self.routes],
            "markers": [dataclasses.asdict(item) for item in self.markers],
            "tc_rules": [dataclasses.asdict(item) for item in self.tc_rules],
        }

    @property
    def sha256(self) -> str:
        """Hash the canonical plan bytes."""

        return hashlib.sha256(_canonical_bytes(self.to_dict())).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise FabricError(f"{field} must be a JSON object")
    return value


def _only_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    unknown = set(value) - expected
    missing = expected - set(value)
    if unknown:
        raise FabricError(
            f"{field} contains unsupported fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise FabricError(f"{field} omits fields: {', '.join(sorted(missing))}")


def _required_and_optional_keys(
    value: Mapping[str, object],
    required: set[str],
    optional: set[str],
    field: str,
) -> None:
    unknown = set(value) - required - optional
    missing = required - set(value)
    if unknown:
        raise FabricError(
            f"{field} contains unsupported fields: {', '.join(sorted(unknown))}"
        )
    if missing:
        raise FabricError(f"{field} omits fields: {', '.join(sorted(missing))}")


def _integer(value: object, field: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise FabricError(f"{field} must be an integer")
    if not minimum <= value <= maximum:
        raise FabricError(f"{field} must be in [{minimum}, {maximum}]")
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise FabricError(f"{field} must be a boolean")
    return value


def _string(value: object, field: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise FabricError(f"{field} has an invalid value")
    return value


def _absolute_path(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise FabricError(f"{field} must be an absolute POSIX path")
    path = PurePosixPath(value)
    if ".." in path.parts or str(path) != value or any(ch.isspace() for ch in value):
        raise FabricError(f"{field} must be a normalized absolute POSIX path")
    return value


def _reject_unresolved(value: object, field: str = "topology") -> None:
    if isinstance(value, str) and "REPLACE" in value.upper():
        raise FabricError(f"{field} contains an unresolved value")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _reject_unresolved(item, f"{field}[{index}]")
    if isinstance(value, dict):
        for key, item in value.items():
            _reject_unresolved(item, f"{field}.{key}")


def _parse_port(
    value: object,
    *,
    rank: int,
    direction: str,
    functions: int,
    field: str,
) -> Port:
    item = _mapping(value, field)
    _only_keys(
        item,
        {
            "function",
            "netdev",
            "rdma_device",
            "ipv4_cidr",
            "mac",
            "peer_rank",
            "peer_direction",
            "peer_function",
        },
        field,
    )
    function = _integer(item["function"], f"{field}.function", 0, functions - 1)
    try:
        interface = ipaddress.ip_interface(item["ipv4_cidr"])
    except (TypeError, ValueError):
        raise FabricError(f"{field}.ipv4_cidr must be an IPv4 /32") from None
    if interface.version != 4 or interface.network.prefixlen != 32:
        raise FabricError(f"{field}.ipv4_cidr must be an IPv4 /32")
    if (
        interface.ip.is_unspecified
        or interface.ip.is_multicast
        or interface.ip.is_loopback
        or int(interface.ip) == 0xFFFFFFFF
    ):
        raise FabricError(f"{field}.ipv4_cidr must identify a unicast endpoint")
    mac = _string(item["mac"], f"{field}.mac", _MAC).lower()
    first = int(mac.split(":", 1)[0], 16)
    if first & 1 or mac == "00:00:00:00:00:00":
        raise FabricError(f"{field}.mac must be a nonzero unicast address")
    return Port(
        direction=direction,
        function=function,
        netdev=_string(item["netdev"], f"{field}.netdev", _NETDEV),
        rdma_device=_string(
            item["rdma_device"], f"{field}.rdma_device", _RDMA_DEVICE
        ),
        ipv4=str(interface.ip),
        mac=mac,
        peer_rank=_integer(item["peer_rank"], f"{field}.peer_rank", 0, 3),
        peer_direction=_string(
            item["peer_direction"], f"{field}.peer_direction", _NAME
        ),
        peer_function=_integer(
            item["peer_function"], f"{field}.peer_function", 0, functions - 1
        ),
    )


def load_topology(path: Path) -> FabricTopology:
    """Load and validate one exact four-rank topology document."""

    if path.is_symlink():
        raise FabricError("topology path must not be a symbolic link")
    try:
        metadata = path.stat()
        raw = path.read_bytes()
    except OSError as error:
        raise FabricError(f"topology cannot be read: {error}") from None
    if not stat.S_ISREG(metadata.st_mode):
        raise FabricError("topology must be a regular file")
    try:
        document = json.loads(raw)
    except json.JSONDecodeError as error:
        raise FabricError(f"topology is not valid JSON: {error}") from None
    root = _mapping(document, "topology")
    _reject_unresolved(root)
    _required_and_optional_keys(
        root,
        {
            "schema",
            "status",
            "group_id",
            "socket_direct_functions",
            "flow_label_base",
            "shared_diagonal_flow_label",
            "standard_ether_type",
            "marked_ether_type",
            "standard_udp_destination_port",
            "roce_gid_index",
            "bounded_runtime_seconds",
            "expected_ethernet_mtu",
            "expected_roce_mtu",
            "orchestration",
            "ranks",
        },
        {"endpoint_route_strategy", "tc_rule_identity_overrides"},
        "topology",
    )
    if root["schema"] != TOPOLOGY_SCHEMA:
        raise FabricError(f"topology.schema must be {TOPOLOGY_SCHEMA!r}")
    if root["status"] != "research-only":
        raise FabricError("topology.status must be 'research-only'")
    functions = _integer(
        root["socket_direct_functions"], "topology.socket_direct_functions", 1, 2
    )
    values = root["ranks"]
    if not isinstance(values, list) or len(values) != RANK_COUNT:
        raise FabricError("topology.ranks must contain exactly ranks 0, 1, 2, 3")
    ranks: list[Rank] = []
    for index, value in enumerate(values):
        field = f"topology.ranks[{index}]"
        item = _mapping(value, field)
        _only_keys(item, {"rank", "ssh_alias", "management_netdev", "ports"}, field)
        rank_id = _integer(item["rank"], f"{field}.rank", 0, 3)
        ports_object = _mapping(item["ports"], f"{field}.ports")
        _only_keys(
            ports_object,
            {"clockwise", "counter_clockwise"},
            f"{field}.ports",
        )
        ports: list[Port] = []
        for direction in ("clockwise", "counter_clockwise"):
            entries = ports_object[direction]
            if not isinstance(entries, list) or len(entries) != functions:
                raise FabricError(
                    f"{field}.ports.{direction} must contain {functions} functions"
                )
            ports.extend(
                _parse_port(
                    entry,
                    rank=rank_id,
                    direction=direction,
                    functions=functions,
                    field=f"{field}.ports.{direction}[{port_index}]",
                )
                for port_index, entry in enumerate(entries)
            )
        ranks.append(
            Rank(
                rank=rank_id,
                ssh_alias=_string(item["ssh_alias"], f"{field}.ssh_alias", _NAME),
                management_netdev=_string(
                    item["management_netdev"],
                    f"{field}.management_netdev",
                    _NETDEV,
                ),
                ports=tuple(ports),
            )
        )
    ranks.sort(key=lambda item: item.rank)
    if [item.rank for item in ranks] != list(range(RANK_COUNT)):
        raise FabricError("topology ranks must be exactly 0, 1, 2, 3")
    orchestration = _mapping(root["orchestration"], "topology.orchestration")
    _only_keys(
        orchestration,
        {"host_helper_path", "remote_state_root"},
        "topology.orchestration",
    )
    endpoint_route_strategy = root.get(
        "endpoint_route_strategy", "scope_link_permanent_final_neighbor"
    )
    if endpoint_route_strategy not in {
        "scope_link_permanent_final_neighbor",
        "adjacent_gateway",
    }:
        raise FabricError(
            "topology.endpoint_route_strategy must be "
            "'scope_link_permanent_final_neighbor' or 'adjacent_gateway'"
        )
    override_values = _mapping(
        root.get("tc_rule_identity_overrides", {}),
        "topology.tc_rule_identity_overrides",
    )
    overrides: list[TcRuleIdentityOverride] = []
    for path_name, value in override_values.items():
        name = _string(path_name, "topology.tc_rule_identity_overrides path", _NAME)
        item = _mapping(value, f"topology.tc_rule_identity_overrides.{name}")
        _only_keys(
            item,
            {"preference", "handle"},
            f"topology.tc_rule_identity_overrides.{name}",
        )
        overrides.append(
            TcRuleIdentityOverride(
                path_name=name,
                preference=_integer(
                    item["preference"],
                    f"topology.tc_rule_identity_overrides.{name}.preference",
                    1,
                    0xFFFF,
                ),
                handle=_integer(
                    item["handle"],
                    f"topology.tc_rule_identity_overrides.{name}.handle",
                    1,
                    0xFFFFFFFF,
                ),
            )
        )
    overrides.sort(key=lambda item: item.path_name)
    topology = FabricTopology(
        source_path=path.resolve(),
        sha256=hashlib.sha256(raw).hexdigest(),
        group_id=_integer(root["group_id"], "topology.group_id", 1, 0xFFFFFFFF),
        socket_direct_functions=functions,
        flow_label_base=_integer(
            root["flow_label_base"], "topology.flow_label_base", 1, FLOW_LABEL_LIMIT
        ),
        shared_diagonal_flow_label=(
            _boolean(
                root["shared_diagonal_flow_label"],
                "topology.shared_diagonal_flow_label",
            )
            if "shared_diagonal_flow_label" in root
            else False
        ),
        endpoint_route_strategy=str(endpoint_route_strategy),
        tc_rule_identity_overrides=tuple(overrides),
        standard_ether_type=_integer(
            root["standard_ether_type"],
            "topology.standard_ether_type",
            0x0600,
            0xFFFF,
        ),
        marked_ether_type=_integer(
            root["marked_ether_type"],
            "topology.marked_ether_type",
            0x0600,
            0xFFFF,
        ),
        standard_udp_destination_port=_integer(
            root["standard_udp_destination_port"],
            "topology.standard_udp_destination_port",
            1,
            65535,
        ),
        roce_gid_index=_integer(
            root["roce_gid_index"], "topology.roce_gid_index", 0, 255
        ),
        bounded_runtime_seconds=_integer(
            root["bounded_runtime_seconds"],
            "topology.bounded_runtime_seconds",
            1,
            MAX_RUNTIME_SECONDS,
        ),
        expected_ethernet_mtu=_integer(
            root["expected_ethernet_mtu"],
            "topology.expected_ethernet_mtu",
            1500,
            65535,
        ),
        expected_roce_mtu=_integer(
            root["expected_roce_mtu"], "topology.expected_roce_mtu", 256, 4096
        ),
        host_helper_path=_absolute_path(
            orchestration["host_helper_path"],
            "topology.orchestration.host_helper_path",
        ),
        remote_state_root=_absolute_path(
            orchestration["remote_state_root"],
            "topology.orchestration.remote_state_root",
        ),
        ranks=tuple(ranks),
    )
    _validate_topology(topology)
    return topology


def _opposite_direction(direction: str) -> str:
    return "counter_clockwise" if direction == "clockwise" else "clockwise"


def _direction_step(direction: str) -> int:
    return 1 if direction == "clockwise" else -1


def _validate_topology(topology: FabricTopology) -> None:
    if topology.standard_ether_type != ETH_P_IP:
        raise FabricError("standard EtherType must be IPv4 0x0800")
    if topology.marked_ether_type != ETH_P_DIAGONAL:
        raise FabricError("marked EtherType must be reserved value 0x88b5")
    aliases = [rank.ssh_alias for rank in topology.ranks]
    if len(set(aliases)) != len(aliases):
        raise FabricError("topology SSH aliases must be unique")
    addresses: list[str] = []
    macs: list[str] = []
    for rank in topology.ranks:
        netdevs = [port.netdev for port in rank.ports]
        devices = [port.rdma_device for port in rank.ports]
        if rank.management_netdev in netdevs:
            raise FabricError(f"rank {rank.rank} reuses its management netdev")
        if len(set(netdevs)) != len(netdevs) or len(set(devices)) != len(devices):
            raise FabricError(f"rank {rank.rank} repeats a fabric netdev or RDMA device")
        addresses.extend(port.ipv4 for port in rank.ports)
        macs.extend(port.mac for port in rank.ports)
        for port in rank.ports:
            expected_peer = (rank.rank + _direction_step(port.direction)) % 4
            expected_direction = _opposite_direction(port.direction)
            if (
                port.peer_rank != expected_peer
                or port.peer_direction != expected_direction
                or port.peer_function != port.function
            ):
                raise FabricError(
                    f"rank {rank.rank} {port.direction} function {port.function} "
                    "does not identify its reciprocal cycle endpoint"
                )
            peer = topology.rank(port.peer_rank).port(
                port.peer_direction, port.peer_function
            )
            if (
                peer.peer_rank != rank.rank
                or peer.peer_direction != port.direction
                or peer.peer_function != port.function
            ):
                raise FabricError("physical cycle contains a nonreciprocal link")
    if len(set(addresses)) != len(addresses):
        raise FabricError("fabric endpoint IPv4 addresses must be globally unique")
    if len(set(macs)) != len(macs):
        raise FabricError("fabric endpoint MAC addresses must be globally unique")
    reciprocal_channel_count = (
        RANK_COUNT * 2 * topology.socket_direct_functions // 2
    )
    if (
        not topology.shared_diagonal_flow_label
        and topology.flow_label_base + reciprocal_channel_count - 1
        > FLOW_LABEL_LIMIT
    ):
        raise FabricError("reserved flow-label range would alias a RoCE UDP source port")


def _flow_label_to_udp_source_port(flow_label: int) -> int:
    return ROCE_SOURCE_PORT_BASE | (flow_label & ROCE_SOURCE_PORT_MASK)


def build_plan(topology: FabricTopology) -> FabricPlan:
    """Generate every direct route, diagonal marker, and middle restore rule."""

    paths: list[FabricPath] = []
    routes: list[EndpointRoute] = []
    markers: list[SourceMarker] = []
    rules: list[TcRestoreRule] = []
    channel_flow_labels: dict[tuple[int, int, int, int], int] = {}
    identity_overrides = {
        item.path_name: item for item in topology.tc_rule_identity_overrides
    }
    used_identity_overrides: set[str] = set()
    rule_ordinal = 0
    for source_rank in range(RANK_COUNT):
        source = topology.rank(source_rank)
        for direction in ("clockwise", "counter_clockwise"):
            step = _direction_step(direction)
            neighbor_rank = (source_rank + step) % RANK_COUNT
            opposite_rank = (source_rank + 2) % RANK_COUNT
            middle = topology.rank(neighbor_rank)
            destination = topology.rank(opposite_rank)
            incoming_direction = _opposite_direction(direction)
            for function in range(topology.socket_direct_functions):
                source_port = source.port(direction, function)
                direct_destination = topology.rank(neighbor_rank).port(
                    incoming_direction, function
                )
                direct_name = (
                    f"rank{source_rank}-to-rank{neighbor_rank}-direct-"
                    f"{direction}-function{function}"
                )
                direct = FabricPath(
                    name=direct_name,
                    kind="direct",
                    source_rank=source_rank,
                    destination_rank=neighbor_rank,
                    direction=direction,
                    function=function,
                    intermediate_rank=None,
                    source_netdev=source_port.netdev,
                    source_rdma_device=source_port.rdma_device,
                    source_ipv4=source_port.ipv4,
                    destination_ipv4=direct_destination.ipv4,
                    next_hop_mac=direct_destination.mac,
                    final_destination_mac=direct_destination.mac,
                )
                paths.append(direct)
                routes.append(_route(topology, direct))

                middle_ingress = middle.port(incoming_direction, function)
                middle_egress = middle.port(direction, function)
                final_port = destination.port(incoming_direction, function)
                diagonal_name = (
                    f"rank{source_rank}-to-rank{opposite_rank}-diagonal-"
                    f"{direction}-via-rank{neighbor_rank}-function{function}"
                )
                diagonal = FabricPath(
                    name=diagonal_name,
                    kind="diagonal",
                    source_rank=source_rank,
                    destination_rank=opposite_rank,
                    direction=direction,
                    function=function,
                    intermediate_rank=neighbor_rank,
                    source_netdev=source_port.netdev,
                    source_rdma_device=source_port.rdma_device,
                    source_ipv4=source_port.ipv4,
                    destination_ipv4=final_port.ipv4,
                    next_hop_mac=middle_ingress.mac,
                    final_destination_mac=final_port.mac,
                )
                paths.append(diagonal)
                routes.append(_route(topology, diagonal))
                channel_key = (
                    min(source_rank, opposite_rank),
                    max(source_rank, opposite_rank),
                    neighbor_rank,
                    function,
                )
                flow_label = channel_flow_labels.setdefault(
                    channel_key,
                    topology.flow_label_base
                    if topology.shared_diagonal_flow_label
                    else topology.flow_label_base + len(channel_flow_labels),
                )
                udp_source_port = _flow_label_to_udp_source_port(flow_label)
                markers.append(
                    SourceMarker(
                        path_name=diagonal.name,
                        source_rank=source_rank,
                        rdma_device=source_port.rdma_device,
                        source_ipv4=source_port.ipv4,
                        destination_ipv4=final_port.ipv4,
                        flow_label=flow_label,
                        udp_source_port=udp_source_port,
                        match_ether_type=topology.standard_ether_type,
                        marked_ether_type=topology.marked_ether_type,
                        match_udp_destination_port=(
                            topology.standard_udp_destination_port
                        ),
                    )
                )
                identity = FILTER_ID_BASE + rule_ordinal
                override = identity_overrides.get(diagonal.name)
                if override is not None:
                    preference = override.preference
                    handle = override.handle
                    used_identity_overrides.add(diagonal.name)
                else:
                    preference = identity
                    handle = identity
                rules.append(
                    TcRestoreRule(
                        name=f"restore-{diagonal.name}",
                        path_name=diagonal.name,
                        source_rank=source_rank,
                        destination_rank=opposite_rank,
                        intermediate_rank=neighbor_rank,
                        direction=direction,
                        function=function,
                        ingress_netdev=middle_ingress.netdev,
                        egress_netdev=middle_egress.netdev,
                        preference=preference,
                        handle=handle,
                        match_ethernet_source=source_port.mac,
                        match_ethernet_destination=middle_ingress.mac,
                        match_ether_type=topology.marked_ether_type,
                        restore_ether_type=topology.standard_ether_type,
                        rewrite_ethernet_destination=final_port.mac,
                    )
                )
                rule_ordinal += 1
    unused_overrides = set(identity_overrides) - used_identity_overrides
    if unused_overrides:
        raise FabricError(
            "topology.tc_rule_identity_overrides contains unknown diagonal paths: "
            + ", ".join(sorted(unused_overrides))
        )
    plan = FabricPlan(
        topology_sha256=topology.sha256,
        group_id=topology.group_id,
        socket_direct_functions=topology.socket_direct_functions,
        shared_diagonal_flow_label=topology.shared_diagonal_flow_label,
        endpoint_route_strategy=topology.endpoint_route_strategy,
        bounded_runtime_seconds=topology.bounded_runtime_seconds,
        expected_ethernet_mtu=topology.expected_ethernet_mtu,
        expected_roce_mtu=topology.expected_roce_mtu,
        roce_gid_index=topology.roce_gid_index,
        host_helper_path=topology.host_helper_path,
        remote_state_root=topology.remote_state_root,
        hardware_gate_status="unproven",
        apply_permitted=False,
        ranks=topology.ranks,
        paths=tuple(paths),
        routes=tuple(routes),
        markers=tuple(markers),
        tc_rules=tuple(rules),
    )
    _validate_plan(plan)
    return plan


def _route(topology: FabricTopology, path: FabricPath) -> EndpointRoute:
    source_port = topology.rank(path.source_rank).port(path.direction, path.function)
    peer_port = topology.rank(source_port.peer_rank).port(
        source_port.peer_direction, source_port.peer_function
    )
    adjacent_gateway = topology.endpoint_route_strategy == "adjacent_gateway"
    return EndpointRoute(
        path_name=path.name,
        source_rank=path.source_rank,
        source_netdev=path.source_netdev,
        source_ipv4=path.source_ipv4,
        destination_ipv4=path.destination_ipv4,
        next_hop_mac=path.next_hop_mac,
        gateway_ipv4=peer_port.ipv4 if adjacent_gateway else None,
        permanent_final_neighbor=not adjacent_gateway,
    )


def _validate_plan(plan: FabricPlan) -> None:
    for route in plan.routes:
        if plan.endpoint_route_strategy == "adjacent_gateway":
            valid_route = (
                route.gateway_ipv4 is not None
                and route.permanent_final_neighbor is False
            )
        else:
            valid_route = (
                plan.endpoint_route_strategy
                == "scope_link_permanent_final_neighbor"
                and route.gateway_ipv4 is None
                and route.permanent_final_neighbor is True
            )
        if not valid_route:
            raise FabricError(
                f"route {route.path_name} does not match endpoint route strategy "
                f"{plan.endpoint_route_strategy!r}"
            )
    label_groups: dict[int, list[SourceMarker]] = {}
    for marker in plan.markers:
        label_groups.setdefault(marker.flow_label, []).append(marker)
    marker_identities = [
        (marker.source_rank, marker.rdma_device, marker.udp_source_port)
        for marker in plan.markers
    ]
    label_ports = {
        label: {marker.udp_source_port for marker in markers}
        for label, markers in label_groups.items()
    }
    identities = [
        (rule.intermediate_rank, rule.ingress_netdev, rule.preference, rule.handle)
        for rule in plan.tc_rules
    ]
    if plan.shared_diagonal_flow_label:
        labels_valid = (
            len(label_groups) == 1
            and len(next(iter(label_groups.values()))) == len(plan.markers)
            and len(next(iter(label_ports.values()))) == 1
        )
    else:
        labels_valid = (
            all(len(markers) == 2 for markers in label_groups.values())
            and all(len(ports) == 1 for ports in label_ports.values())
            and len({next(iter(ports)) for ports in label_ports.values()})
            == len(label_ports)
        )
    if not labels_valid or len(marker_identities) != len(set(marker_identities)):
        raise FabricError(
            "diagonal marker labels and UDP source ports must match the selected "
            "shared or per-channel policy and remain unique on each source HCA"
        )
    if len(identities) != len(set(identities)):
        raise FabricError("intermediate filter identities must be unique")
    for rule in plan.tc_rules:
        reverse = [
            item
            for item in plan.tc_rules
            if item.source_rank == rule.destination_rank
            and item.destination_rank == rule.source_rank
            and item.intermediate_rank == rule.intermediate_rank
            and item.function == rule.function
        ]
        if len(reverse) != 1:
            raise FabricError(f"diagonal rule {rule.name} has no reciprocal path")
        peer = reverse[0]
        marker = next(
            (item for item in plan.markers if item.path_name == rule.path_name),
            None,
        )
        peer_marker = next(
            (item for item in plan.markers if item.path_name == peer.path_name),
            None,
        )
        if (
            peer.ingress_netdev != rule.egress_netdev
            or peer.egress_netdev != rule.ingress_netdev
            or peer.match_ether_type != rule.match_ether_type
            or peer.restore_ether_type != rule.restore_ether_type
            or peer.match_ethernet_source != rule.rewrite_ethernet_destination
            or peer.rewrite_ethernet_destination != rule.match_ethernet_source
            or marker is None
            or peer_marker is None
            or peer_marker.flow_label != marker.flow_label
            or peer_marker.udp_source_port != marker.udp_source_port
        ):
            raise FabricError(f"diagonal rule {rule.name} is nonreciprocal")


def build_rocenante_plan(complete: FabricPlan) -> FabricPlan:
    """Select the 24 origin QPs used by a balanced TP4 RoCEnante mesh."""

    if complete.socket_direct_functions != 2:
        raise FabricError("RoCEnante selection requires two Socket-Direct functions")
    direct = tuple(path for path in complete.paths if path.kind == "direct")
    candidates = tuple(path for path in complete.paths if path.kind == "diagonal")
    if len(direct) != 16 or len(candidates) != 16:
        raise FabricError("RoCEnante selection requires 16 direct and 16 diagonal paths")

    def selected_function(path: FabricPath) -> int:
        pair = frozenset((path.source_rank, path.destination_rank))
        if pair not in {frozenset((0, 2)), frozenset((1, 3))}:
            raise FabricError(f"path {path.name} is not a TP4 opposite-rank path")
        lower = min(pair)
        if path.intermediate_rank == (lower + 1) % RANK_COUNT:
            return 0
        if path.intermediate_rank == (lower - 1) % RANK_COUNT:
            return 1
        raise FabricError(f"path {path.name} has an invalid opposite-rank route")

    diagonal = tuple(
        path for path in candidates if path.function == selected_function(path)
    )
    selected_names = {path.name for path in diagonal}
    selected = dataclasses.replace(
        complete,
        paths=tuple(
            path
            for path in complete.paths
            if path.kind == "direct" or path.name in selected_names
        ),
        routes=tuple(
            route for route in complete.routes if route.path_name in selected_names
        ),
        markers=tuple(
            marker for marker in complete.markers if marker.path_name in selected_names
        ),
        tc_rules=tuple(
            rule for rule in complete.tc_rules if rule.path_name in selected_names
        ),
        hardware_gate_status="unproven",
        apply_permitted=False,
    )
    _validate_plan(selected)
    rocenante_inventory(selected)
    return selected


def rocenante_inventory(plan: FabricPlan) -> dict[str, object]:
    """Return QP counts and physical-edge load for a selected TP4 mesh."""

    direct = [path for path in plan.paths if path.kind == "direct"]
    diagonal = [path for path in plan.paths if path.kind == "diagonal"]
    per_rank = {
        str(rank): sum(path.source_rank == rank for path in plan.paths)
        for rank in range(RANK_COUNT)
    }
    per_rank_function = {
        f"rank{rank}/function{function}": sum(
            path.source_rank == rank and path.function == function
            for path in plan.paths
        )
        for rank in range(RANK_COUNT)
        for function in (0, 1)
    }
    edge_load: dict[str, int] = {
        f"{left}-{right}/function{function}": 0
        for left, right in ((0, 1), (1, 2), (2, 3), (0, 3))
        for function in (0, 1)
    }
    for path in diagonal:
        if path.intermediate_rank is None:
            raise FabricError(f"diagonal path {path.name} has no intermediate rank")
        for first, second in (
            (path.source_rank, path.intermediate_rank),
            (path.intermediate_rank, path.destination_rank),
        ):
            left, right = sorted((first, second))
            key = f"{left}-{right}/function{path.function}"
            edge_load[key] = edge_load.get(key, 0) + 1
    expected_edges = {
        "0-1/function0": 2,
        "0-1/function1": 2,
        "0-3/function0": 0,
        "0-3/function1": 4,
        "1-2/function0": 4,
        "1-2/function1": 0,
        "2-3/function0": 2,
        "2-3/function1": 2,
    }
    if (
        len(direct) != 16
        or len(diagonal) != 8
        or any(value != 6 for value in per_rank.values())
        or any(value != 3 for value in per_rank_function.values())
        or edge_load != expected_edges
    ):
        raise FabricError(
            "RoCEnante selection must contain six origin QPs per rank and "
            "three origin QPs per local function with reciprocal opposite paths"
        )
    return {
        "status": "research-only",
        "topology_sha256": plan.topology_sha256,
        "plan_sha256": plan.sha256,
        "direct_origin_qps": len(direct),
        "forwarded_origin_qps": len(diagonal),
        "total_origin_qps": len(plan.paths),
        "origin_qps_per_rank": per_rank,
        "origin_qps_per_rank_and_function": per_rank_function,
        "opposite_route_function_assignment": {
            "0-2/via1": 0,
            "0-2/via3": 1,
            "1-3/via2": 0,
            "1-3/via0": 1,
        },
        "diagonal_edge_function_traversals": edge_load,
    }


def rocenante_native_path_arguments(plan: FabricPlan) -> dict[str, tuple[str, ...]]:
    """Return six native benchmark path specifications for every rank."""

    result: dict[str, tuple[str, ...]] = {}
    for rank in range(RANK_COUNT):
        local = sorted(
            (path for path in plan.paths if path.source_rank == rank),
            key=lambda path: (path.destination_rank, path.function),
        )
        if len(local) != 6:
            raise FabricError(f"rank {rank} must own six RoCEnante origin QPs")
        specifications: list[str] = []
        for path in local:
            reverse = [
                item
                for item in plan.paths
                if item.source_rank == path.destination_rank
                and item.destination_rank == path.source_rank
                and item.function == path.function
                and item.intermediate_rank == path.intermediate_rank
            ]
            if len(reverse) != 1 or reverse[0].direction == path.direction:
                raise FabricError(
                    f"path {path.name} does not have one reciprocal native endpoint"
                )
            hops = 1 if path.kind == "direct" else 2
            specifications.append(
                f"{path.destination_rank},{path.function},"
                f"{path.source_rdma_device},{plan.roce_gid_index},{hops}"
            )
        result[str(rank)] = tuple(specifications)
    return result


def route_command(route: EndpointRoute, *, add: bool) -> list[str]:
    """Return one exact `/32` route add or delete command."""

    command = [
        "ip",
        "route",
        "add" if add else "del",
        f"{route.destination_ipv4}/32",
    ]
    if route.gateway_ipv4 is not None:
        command.extend(["via", route.gateway_ipv4])
    command.extend(
        ["dev", route.source_netdev, "src", route.source_ipv4]
    )
    if route.gateway_ipv4 is None:
        command.extend(["scope", "link"])
    return command


def neighbor_command(route: EndpointRoute, *, add: bool) -> list[str]:
    """Return one exact permanent-neighbor add or delete command."""

    if not route.permanent_final_neighbor:
        raise FabricError(
            f"route {route.path_name} uses an adjacent gateway and has no "
            "permanent final-destination neighbor"
        )
    command = ["ip", "neigh", "add" if add else "del", route.destination_ipv4]
    if add:
        command.extend(["lladdr", route.next_hop_mac, "nud", "permanent"])
    command.extend(["dev", route.source_netdev])
    return command


def marker_command(
    plan: FabricPlan, marker: SourceMarker, *, apply: bool
) -> list[str]:
    """Return one QP-scoped EtherType-marker helper invocation."""

    command = [
        plan.host_helper_path,
        "marker",
        "apply" if apply else "cleanup",
        "--state-root",
        plan.remote_state_root,
        "--path-name",
        marker.path_name,
        "--device",
        marker.rdma_device,
        "--source-ip",
        marker.source_ipv4,
        "--destination-ip",
        marker.destination_ipv4,
        "--flow-label",
        str(marker.flow_label),
        "--udp-source-port",
        str(marker.udp_source_port),
        "--match-ether-type",
        f"0x{marker.match_ether_type:04x}",
        "--marked-ether-type",
        f"0x{marker.marked_ether_type:04x}",
        "--match-udp-destination-port",
        str(marker.match_udp_destination_port),
    ]
    if apply:
        command.extend(["--run-seconds", str(plan.bounded_runtime_seconds)])
    else:
        command.append("--if-present")
    return command


def tc_rule_command(rule: TcRestoreRule, *, add: bool) -> list[str]:
    """Return one hardware-only EtherType restore rule or exact deletion."""

    identity = [
        "tc",
        "filter",
        "add" if add else "delete",
        "dev",
        rule.ingress_netdev,
        "ingress",
        "protocol",
        f"0x{rule.match_ether_type:04x}",
        "pref",
        str(rule.preference),
        "handle",
        str(rule.handle),
    ]
    if not add:
        return identity + ["flower"]
    return identity + [
        "flower",
        "skip_sw",
        "src_mac",
        rule.match_ethernet_source,
        "dst_mac",
        rule.match_ethernet_destination,
        "action",
        "pedit",
        "ex",
        "munge",
        "eth",
        "type",
        "set",
        f"0x{rule.restore_ether_type:04x}",
        "pipe",
        "action",
        "pedit",
        "ex",
        "munge",
        "eth",
        "dst",
        "set",
        rule.rewrite_ethernet_destination,
        "pipe",
        "action",
        "mirred",
        "egress",
        "redirect",
        "dev",
        rule.egress_netdev,
    ]


def require_hardware_gate(plan: FabricPlan, receipt: object) -> None:
    """Reject apply unless an exact EtherType-path qualification receipt passes."""

    value = _mapping(receipt, "hardware gate receipt")
    required = {
        "schema",
        "status",
        "topology_sha256",
        "plan_sha256",
        "source_ethertype_rewrite_in_hw",
        "intermediate_ethertype_restore_in_hw",
        "remote_payload_byte_match",
        "rx_icrc_encapsulated_delta",
        "cleanup_verified",
    }
    _only_keys(value, required, "hardware gate receipt")
    boolean_fields = (
        "source_ethertype_rewrite_in_hw",
        "intermediate_ethertype_restore_in_hw",
        "remote_payload_byte_match",
        "cleanup_verified",
    )
    passed = (
        value["schema"] == HARDWARE_GATE_SCHEMA
        and value["status"] == "qualified"
        and value["topology_sha256"] == plan.topology_sha256
        and value["plan_sha256"] == plan.sha256
        and all(value[field] is True for field in boolean_fields)
        and type(value["rx_icrc_encapsulated_delta"]) is int
        and value["rx_icrc_encapsulated_delta"] == 0
    )
    if not passed:
        raise FabricError(
            "hardware gate must qualify exact in-hardware rewrite and restore, "
            "remote payload bytes, zero ICRC errors, and cleanup"
        )


def _command(plan: FabricPlan, rank: int, argv: list[str]) -> dict[str, object]:
    return {
        "rank": rank,
        "ssh_alias": plan.ranks[rank].ssh_alias,
        "argv": argv,
    }


def _mtu_command(plan: FabricPlan, rank: Rank, operation: str) -> list[str]:
    command = [
        plan.host_helper_path,
        "mtu",
        operation,
        "--state-root",
        plan.remote_state_root,
        "--rank",
        str(rank.rank),
    ]
    if operation in {"snapshot", "verify"}:
        command.extend(
            [
                "--expected-ethernet-mtu",
                str(plan.expected_ethernet_mtu),
                "--expected-roce-mtu",
                str(plan.expected_roce_mtu),
            ]
        )
        for port in rank.ports:
            command.extend(
                ["--netdev", port.netdev, "--rdma-device", port.rdma_device]
            )
    if operation == "restore":
        command.append("--if-present")
    return command


def _qdisc_command(
    plan: FabricPlan, rank: int, netdev: str, operation: str
) -> list[str]:
    command = [
        plan.host_helper_path,
        "qdisc",
        operation,
        "--state-root",
        plan.remote_state_root,
        "--rank",
        str(rank),
        "--netdev",
        netdev,
        "--kind",
        "clsact",
    ]
    if operation == "cleanup":
        command.append("--if-owned-and-empty")
    return command


def _qdisc_identities(plan: FabricPlan) -> list[tuple[int, str]]:
    return list(
        dict.fromkeys(
            (rule.intermediate_rank, rule.ingress_netdev) for rule in plan.tc_rules
        )
    )


def _phase(name: str, commands: list[dict[str, object]]) -> dict[str, object]:
    return {"name": name, "commands": commands}


def build_cleanup_manifest(plan: FabricPlan) -> dict[str, object]:
    """Build exact reverse-order cleanup without authorizing any apply action."""

    marker_deletes = [
        _command(plan, marker.source_rank, marker_command(plan, marker, apply=False))
        for marker in reversed(plan.markers)
    ]
    rule_deletes = [
        _command(plan, rule.intermediate_rank, tc_rule_command(rule, add=False))
        for rule in reversed(plan.tc_rules)
    ]
    qdisc_deletes = [
        _command(plan, rank, _qdisc_command(plan, rank, netdev, "cleanup"))
        for rank, netdev in reversed(_qdisc_identities(plan))
    ]
    neighbor_deletes = [
        _command(plan, route.source_rank, neighbor_command(route, add=False))
        for route in reversed(plan.routes)
        if route.permanent_final_neighbor
    ]
    route_deletes = [
        _command(plan, route.source_rank, route_command(route, add=False))
        for route in reversed(plan.routes)
    ]
    mtu_restore = [
        _command(plan, rank.rank, _mtu_command(plan, rank, "restore"))
        for rank in reversed(plan.ranks)
    ]
    mtu_verify = [
        _command(plan, rank.rank, _mtu_command(plan, rank, "verify"))
        for rank in plan.ranks
    ]
    return {
        "schema": CLEANUP_MANIFEST_SCHEMA,
        "status": "research-only",
        "topology_sha256": plan.topology_sha256,
        "plan_sha256": plan.sha256,
        "automatic_on_exit": True,
        "automatic_on_apply_failure": True,
        "idempotent": True,
        "absent_owned_object_is_success": True,
        "mtu_snapshot_restore_required": True,
        "phases": [
            _phase("source_markers", marker_deletes),
            _phase("intermediate_rules", rule_deletes),
            _phase("intermediate_qdiscs", qdisc_deletes),
            _phase("endpoint_neighbors", neighbor_deletes),
            _phase("endpoint_routes", route_deletes),
            _phase("mtu_restore", mtu_restore),
            _phase("mtu_verify", mtu_verify),
        ],
    }


def build_apply_manifest(
    plan: FabricPlan,
    hardware_gate_receipt: object,
    *,
    authorization_token: str,
) -> dict[str, object]:
    """Build a non-executing all-rank manifest after the hardware gate passes."""

    if authorization_token != AUTHORIZATION_TOKEN:
        raise FabricError("apply requires the exact all-four authorization token")
    if tuple(rank.rank for rank in plan.ranks) != tuple(range(RANK_COUNT)):
        raise FabricError("apply requires exactly ranks 0, 1, 2, 3")
    require_hardware_gate(plan, hardware_gate_receipt)

    route_adds: list[dict[str, object]] = []
    neighbor_adds: list[dict[str, object]] = []
    for route in plan.routes:
        route_adds.append(
            _command(plan, route.source_rank, route_command(route, add=True))
        )
        if route.permanent_final_neighbor:
            neighbor_adds.append(
                _command(plan, route.source_rank, neighbor_command(route, add=True))
            )

    rule_adds = [
        _command(plan, rule.intermediate_rank, tc_rule_command(rule, add=True))
        for rule in plan.tc_rules
    ]
    qdisc_adds = [
        _command(plan, rank, _qdisc_command(plan, rank, netdev, "ensure"))
        for rank, netdev in _qdisc_identities(plan)
    ]
    marker_adds = [
        _command(plan, marker.source_rank, marker_command(plan, marker, apply=True))
        for marker in plan.markers
    ]
    mtu_snapshot = [
        _command(plan, rank.rank, _mtu_command(plan, rank, "snapshot"))
        for rank in plan.ranks
    ]
    mtu_verify = [
        _command(plan, rank.rank, _mtu_command(plan, rank, "verify"))
        for rank in plan.ranks
    ]
    apply_phases = [
        _phase("mtu_snapshot", mtu_snapshot),
        _phase("endpoint_routes", route_adds + neighbor_adds),
        _phase("intermediate_qdiscs", qdisc_adds),
        _phase("intermediate_rules", rule_adds),
        _phase("source_markers", marker_adds),
        _phase("mtu_verify", mtu_verify),
    ]
    cleanup = build_cleanup_manifest(plan)
    return {
        "schema": APPLY_MANIFEST_SCHEMA,
        "status": "research-only",
        "topology_sha256": plan.topology_sha256,
        "plan_sha256": plan.sha256,
        "hardware_gate_receipt_sha256": hashlib.sha256(
            _canonical_bytes(hardware_gate_receipt)
        ).hexdigest(),
        "authorized_ranks": list(range(RANK_COUNT)),
        "bounded_runtime_seconds": plan.bounded_runtime_seconds,
        "receipt_schema": RECEIPT_SCHEMA,
        "apply_phases": apply_phases,
        "cleanup": cleanup,
    }
