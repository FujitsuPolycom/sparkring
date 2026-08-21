#!/usr/bin/env python3
"""SparkRing site configuration: loader and fail-closed validator.

A "site" is one four-node ring: the ranks, the four 200GbE cables that join
them, the control-channel rendezvous addresses, the pinned runtime, and the
paths and artifacts every rank must agree on.  ``scripts/config/site.yaml``
(copied from ``exl3-r7-site.example.yaml``) is the single source of truth for the
public tooling; ``scripts/preflight.py`` is driven entirely by it.

Design rules:

* **Fail closed.**  Anything unrecognised, missing, out of range, or internally
  inconsistent raises :class:`SiteConfigError` naming the exact field.  There is
  no "best effort" mode and no silent defaulting of anything that describes
  hardware.
* **No third-party dependencies** beyond PyYAML.  If PyYAML is missing the
  error says so plainly instead of surfacing as an ImportError traceback.
* **Structure, not just types.**  The four ring edges must form a single closed
  4-cycle through all four ranks, each edge must be claimed exactly once by
  each of its two endpoints, and every address must live in its own edge's /24.
  A mis-cabled or half-edited config fails here rather than at collective init.

Usage::

    python scripts/sparkring_site.py scripts/config/exl3-r7-site.example.yaml
    python -m sparkring_site scripts/config/site.yaml --json

Exit status is 0 when the file validates, 1 when it does not.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # PyYAML is the only third-party dependency, and it is optional at import
    import yaml as _yaml
except Exception as _exc:  # pragma: no cover - exercised only without PyYAML
    _yaml = None
    _YAML_IMPORT_ERROR: Exception | None = _exc
else:
    _YAML_IMPORT_ERROR = None


SCHEMA_ID = "sparkring-site/v1"
SCHEMA_VERSION = 1

# --------------------------------------------------------------------------
# Accepted ranges.  These are deliberately wide enough for hardware that is
# not a DGX Spark, and narrow enough that a typo (mtu: 90000, tp: 400) fails.
# --------------------------------------------------------------------------
RANK_IDS = (0, 1, 2, 3)
RING_PORTS_PER_RANK = 2
EDGE_PREFIX_LENGTH = 24

MTU_RANGE = (1280, 9216)
LINK_SPEED_RANGE = (1000, 1_600_000)          # Mb/s
RDMA_PORT_RANGE = (1, 8)
GID_INDEX_RANGE = (0, 255)
TP_RANGE = (1, 64)
DCP_RANGE = (1, 64)
MAX_MODEL_LEN_RANGE = (256, 8_388_608)
KV_BYTES_RANGE = (1 << 20, 1 << 44)
MAX_NUM_SEQS_RANGE = (1, 8192)
MTP_TOKENS_RANGE = (0, 16)
PORT_RANGE = (1, 65535)
SSH_TIMEOUT_RANGE = (1, 3600)
MIN_FREE_BYTES_RANGE = (0, 1 << 60)

MTP_MODES = ("off", "static", "adaptive")

# --------------------------------------------------------------------------
# Character classes.  Every value that is later interpolated into a remote
# shell command is restricted here, so command construction in preflight.py
# cannot be turned into injection by an edited config.
# --------------------------------------------------------------------------
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_REVISION_RE = re.compile(r"^[0-9a-f]{40}$")
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$")
_IMAGE_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")
_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/+@:-]*[A-Za-z0-9._+@:-]$")
_LOCAL_PATH_RE = re.compile(r"^[A-Za-z0-9._/\\:+@-]+$")

# Reserved ranges used by the shipped example.  Seeing one of these in a real
# site file means the operator has not finished filling the template in.
_DOCUMENTATION_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in (
        "192.0.2.0/24",      # RFC 5737 TEST-NET-1
        "198.51.100.0/24",   # RFC 5737 TEST-NET-2
        "203.0.113.0/24",    # RFC 5737 TEST-NET-3
        "198.18.0.0/15",     # RFC 2544 benchmarking
        "233.252.0.0/24",    # RFC 5771 MCAST-TEST-NET
    )
)


class SiteConfigError(ValueError):
    """A site configuration is invalid.  ``field`` names the offending key."""

    def __init__(self, field_path: str, message: str) -> None:
        self.field = field_path
        self.reason = message
        super().__init__(f"{field_path}: {message}")


# ==========================================================================
# Primitive validators
# ==========================================================================


def _mapping(value: Any, where: str) -> dict:
    if not isinstance(value, Mapping):
        raise SiteConfigError(
            where, f"expected a mapping, got {type(value).__name__}"
        )
    for key in value:
        if not isinstance(key, str):
            raise SiteConfigError(where, f"non-string key {key!r}")
    return dict(value)


def _sequence(value: Any, where: str) -> list:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise SiteConfigError(
            where, f"expected a list, got {type(value).__name__}"
        )
    return list(value)


def _keys(data: Mapping[str, Any], where: str,
          required: Iterable[str], optional: Iterable[str] = ()) -> None:
    required = tuple(required)
    optional = tuple(optional)
    missing = sorted(key for key in required if key not in data)
    if missing:
        raise SiteConfigError(
            where, "missing required key(s): " + ", ".join(missing)
        )
    allowed = set(required) | set(optional)
    unknown = sorted(key for key in data if key not in allowed)
    if unknown:
        raise SiteConfigError(
            where,
            "unknown key(s): " + ", ".join(unknown)
            + " (allowed: " + ", ".join(sorted(allowed)) + ")",
        )


def _string(data: Mapping[str, Any], where: str, key: str,
            pattern: re.Pattern[str] | None = None,
            what: str = "value") -> str:
    raw = data[key]
    if not isinstance(raw, str):
        raise SiteConfigError(
            f"{where}.{key}",
            f"expected a string, got {type(raw).__name__}",
        )
    if not raw.strip():
        raise SiteConfigError(f"{where}.{key}", "must not be empty")
    if raw != raw.strip():
        raise SiteConfigError(
            f"{where}.{key}", f"has leading/trailing whitespace: {raw!r}"
        )
    if pattern is not None and not pattern.match(raw):
        raise SiteConfigError(
            f"{where}.{key}",
            f"invalid {what} {raw!r}; must match {pattern.pattern}",
        )
    return raw


def _integer(data: Mapping[str, Any], where: str, key: str,
             bounds: tuple[int, int]) -> int:
    raw = data[key]
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise SiteConfigError(
            f"{where}.{key}",
            f"expected an integer, got {type(raw).__name__}",
        )
    low, high = bounds
    if not low <= raw <= high:
        raise SiteConfigError(
            f"{where}.{key}", f"{raw} is out of range [{low}, {high}]"
        )
    return raw


def _boolean(data: Mapping[str, Any], where: str, key: str) -> bool:
    raw = data[key]
    if not isinstance(raw, bool):
        raise SiteConfigError(
            f"{where}.{key}",
            f"expected true or false, got {type(raw).__name__}",
        )
    return raw


def _ipv4(data: Mapping[str, Any], where: str, key: str
          ) -> ipaddress.IPv4Address:
    raw = data[key]
    if not isinstance(raw, str):
        raise SiteConfigError(
            f"{where}.{key}",
            f"expected a dotted-quad IPv4 string, got {type(raw).__name__}",
        )
    try:
        address = ipaddress.IPv4Address(raw.strip())
    except ValueError as exc:
        raise SiteConfigError(
            f"{where}.{key}", f"not a valid IPv4 address: {raw!r} ({exc})"
        ) from None
    if address.is_loopback or address.is_multicast or address.is_unspecified:
        raise SiteConfigError(
            f"{where}.{key}",
            f"{address} is loopback/multicast/unspecified and cannot be used",
        )
    return address


def _ipv4_network(data: Mapping[str, Any], where: str, key: str
                  ) -> ipaddress.IPv4Network:
    raw = data[key]
    if not isinstance(raw, str):
        raise SiteConfigError(
            f"{where}.{key}",
            f"expected a CIDR string, got {type(raw).__name__}",
        )
    try:
        network = ipaddress.IPv4Network(raw.strip(), strict=True)
    except ValueError as exc:
        raise SiteConfigError(
            f"{where}.{key}",
            f"not a valid IPv4 network: {raw!r} ({exc})",
        ) from None
    if network.prefixlen != EDGE_PREFIX_LENGTH:
        raise SiteConfigError(
            f"{where}.{key}",
            f"ring edge subnets must be /{EDGE_PREFIX_LENGTH}, "
            f"got /{network.prefixlen}",
        )
    return network


def _sha256(data: Mapping[str, Any], where: str, key: str) -> str:
    raw = data[key]
    if not isinstance(raw, str):
        raise SiteConfigError(
            f"{where}.{key}",
            f"expected a 64-character sha256 hex string, "
            f"got {type(raw).__name__}",
        )
    if not _SHA256_RE.match(raw):
        raise SiteConfigError(
            f"{where}.{key}",
            f"must be exactly 64 lowercase hex characters, got {raw!r}",
        )
    return raw


def ipv4_mapped_gid(address: str | ipaddress.IPv4Address) -> str:
    """Return the RoCEv2 IPv4-mapped GID for a dotted quad.

    ``192.0.2.10`` -> ``0000:0000:0000:0000:0000:ffff:c000:020a``.  This is the
    exact string the kernel writes into
    ``/sys/class/infiniband/<dev>/ports/<n>/gids/<index>`` for an IPv4 RoCEv2
    GID, which is what makes it comparable byte-for-byte in preflight.
    """
    parsed = ipaddress.IPv4Address(str(address))
    a, b, c, d = (int(part) for part in str(parsed).split("."))
    return (
        "0000:0000:0000:0000:0000:ffff:"
        f"{a:02x}{b:02x}:{c:02x}{d:02x}"
    )


def is_documentation_address(address: str | ipaddress.IPv4Address) -> bool:
    """True when ``address`` is from a reserved documentation/benchmark range."""
    try:
        parsed = ipaddress.IPv4Address(str(address))
    except ValueError:
        return False
    return any(parsed in network for network in _DOCUMENTATION_NETWORKS)


def _looks_like_placeholder_hex(value: str) -> bool:
    """True for hex strings made of a single repeated digit (aaaa..., 0000...)."""
    return len(value) >= 8 and len(set(value)) == 1


# ==========================================================================
# Dataclasses
# ==========================================================================


@dataclass(frozen=True)
class Edge:
    id: str
    subnet: ipaddress.IPv4Network
    endpoints: tuple[int, int]

    def other(self, rank_id: int) -> int:
        first, second = self.endpoints
        if rank_id == first:
            return second
        if rank_id == second:
            return first
        raise KeyError(f"rank {rank_id} is not an endpoint of edge {self.id}")


@dataclass
class RingPort:
    edge: str
    interface: str
    address: ipaddress.IPv4Address
    rdma_device: str
    rdma_port: int
    roce_gid_index: int
    # Filled during cross-validation once both endpoints of the edge are known.
    peer_rank: int = -1
    peer_address: ipaddress.IPv4Address | None = None
    prefix_length: int = EDGE_PREFIX_LENGTH

    @property
    def cidr(self) -> str:
        return f"{self.address}/{self.prefix_length}"

    @property
    def rdma_key(self) -> str:
        return f"{self.rdma_device}:{self.rdma_port}"

    @property
    def expected_gid(self) -> str:
        return ipv4_mapped_gid(self.address)


@dataclass(frozen=True)
class TransportPeer:
    rank: int
    address: ipaddress.IPv4Address


@dataclass(frozen=True)
class Management:
    interface: str
    address: ipaddress.IPv4Address


@dataclass
class Rank:
    id: int
    ssh_target: str
    management: Management
    ring_ports: tuple[RingPort, ...]
    transport_peers: tuple[TransportPeer, ...]

    @property
    def neighbour_ranks(self) -> tuple[int, ...]:
        return tuple(sorted(port.peer_rank for port in self.ring_ports))


@dataclass(frozen=True)
class Topology:
    mtu: int
    link_speed_mbps: int
    edges: tuple[Edge, ...]

    @property
    def jumbo_payload_bytes(self) -> int:
        """ICMP payload that exactly fills ``mtu`` (20B IP + 8B ICMP header)."""
        return self.mtu - 28

    def by_id(self, edge_id: str) -> Edge:
        for edge in self.edges:
            if edge.id == edge_id:
                return edge
        raise KeyError(edge_id)


@dataclass(frozen=True)
class Runtime:
    container_image: str
    container_image_digest: str
    model_path: str
    model_repo: str
    model_revision: str
    checkpoint_sha256: str


@dataclass(frozen=True)
class Serving:
    tensor_parallel_size: int
    decode_context_parallel_size: int
    mtp_mode: str
    mtp_tokens: int
    max_model_len: int
    kv_cache_bytes_per_rank: int
    max_num_seqs: int
    master_rank: int
    api_port: int
    master_port: int


@dataclass(frozen=True)
class Paths:
    jit_cache_dir: str
    context_cache_dir: str
    evidence_dir: str
    jit_cache_min_free_bytes: int
    context_cache_min_free_bytes: int

    def remote_space_targets(self) -> tuple[tuple[str, str, int], ...]:
        """``(label, remote path, minimum free bytes)`` triples for preflight."""
        return (
            ("jit_cache", self.jit_cache_dir, self.jit_cache_min_free_bytes),
            ("context_cache", self.context_cache_dir,
             self.context_cache_min_free_bytes),
        )


@dataclass(frozen=True)
class Artifact:
    name: str
    path: str
    sha256: str
    executable: bool


@dataclass(frozen=True)
class PreflightOptions:
    ssh_timeout_seconds: int
    required_free_ports: tuple[int, ...]


@dataclass
class SiteConfig:
    schema_version: int
    name: str
    description: str
    topology: Topology
    ranks: tuple[Rank, ...]
    runtime: Runtime
    serving: Serving
    paths: Paths
    artifacts: tuple[Artifact, ...]
    preflight: PreflightOptions
    source: str | None = None

    def rank(self, rank_id: int) -> Rank:
        for entry in self.ranks:
            if entry.id == rank_id:
                return entry
        raise KeyError(rank_id)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA_ID,
            "schema_version": self.schema_version,
            "source": self.source,
            "site": {"name": self.name, "description": self.description},
            "topology": {
                "mtu": self.topology.mtu,
                "link_speed_mbps": self.topology.link_speed_mbps,
                "edges": [
                    {
                        "id": edge.id,
                        "subnet": str(edge.subnet),
                        "endpoints": list(edge.endpoints),
                    }
                    for edge in self.topology.edges
                ],
            },
            "ranks": [
                {
                    "id": rank.id,
                    "ssh_target": rank.ssh_target,
                    "management": {
                        "interface": rank.management.interface,
                        "address": str(rank.management.address),
                    },
                    "ring_ports": [
                        {
                            "edge": port.edge,
                            "interface": port.interface,
                            "address": str(port.address),
                            "cidr": port.cidr,
                            "rdma_device": port.rdma_device,
                            "rdma_port": port.rdma_port,
                            "roce_gid_index": port.roce_gid_index,
                            "expected_gid": port.expected_gid,
                            "peer_rank": port.peer_rank,
                            "peer_address": str(port.peer_address),
                        }
                        for port in rank.ring_ports
                    ],
                    "transport_peers": [
                        {"rank": peer.rank, "address": str(peer.address)}
                        for peer in rank.transport_peers
                    ],
                }
                for rank in self.ranks
            ],
            "runtime": {
                "container_image": self.runtime.container_image,
                "container_image_digest": self.runtime.container_image_digest,
                "model_path": self.runtime.model_path,
                "model_repo": self.runtime.model_repo,
                "model_revision": self.runtime.model_revision,
                "checkpoint_sha256": self.runtime.checkpoint_sha256,
            },
            "serving": {
                "tensor_parallel_size": self.serving.tensor_parallel_size,
                "decode_context_parallel_size":
                    self.serving.decode_context_parallel_size,
                "mtp_mode": self.serving.mtp_mode,
                "mtp_tokens": self.serving.mtp_tokens,
                "max_model_len": self.serving.max_model_len,
                "kv_cache_bytes_per_rank":
                    self.serving.kv_cache_bytes_per_rank,
                "max_num_seqs": self.serving.max_num_seqs,
                "master_rank": self.serving.master_rank,
                "api_port": self.serving.api_port,
                "master_port": self.serving.master_port,
            },
            "paths": {
                "jit_cache_dir": self.paths.jit_cache_dir,
                "context_cache_dir": self.paths.context_cache_dir,
                "evidence_dir": self.paths.evidence_dir,
                "min_free_bytes": {
                    "jit_cache": self.paths.jit_cache_min_free_bytes,
                    "context_cache":
                        self.paths.context_cache_min_free_bytes,
                },
            },
            "artifacts": [
                {
                    "name": artifact.name,
                    "path": artifact.path,
                    "sha256": artifact.sha256,
                    "executable": artifact.executable,
                }
                for artifact in self.artifacts
            ],
            "preflight": {
                "ssh_timeout_seconds": self.preflight.ssh_timeout_seconds,
                "required_free_ports":
                    list(self.preflight.required_free_ports),
            },
        }

    # -- reporting ---------------------------------------------------------

    def placeholder_warnings(self) -> list[str]:
        """Fields that still look like the shipped example, not a real site.

        These never fail validation - the example file has to validate - but
        preflight surfaces them so a half-filled config is obvious.
        """
        warnings: list[str] = []
        for rank in self.ranks:
            if is_documentation_address(rank.management.address):
                warnings.append(
                    f"ranks[{rank.id}].management.address "
                    f"{rank.management.address} is a reserved "
                    "documentation/benchmark address"
                )
            for index, port in enumerate(rank.ring_ports):
                if is_documentation_address(port.address):
                    warnings.append(
                        f"ranks[{rank.id}].ring_ports[{index}].address "
                        f"{port.address} is a reserved "
                        "documentation/benchmark address"
                    )
        if _looks_like_placeholder_hex(self.runtime.checkpoint_sha256):
            warnings.append(
                "runtime.checkpoint_sha256 is a repeated-digit placeholder"
            )
        if _looks_like_placeholder_hex(self.runtime.model_revision):
            warnings.append(
                "runtime.model_revision is a repeated-digit placeholder"
            )
        digest_hex = self.runtime.container_image_digest.split(":", 1)[1]
        if _looks_like_placeholder_hex(digest_hex):
            warnings.append(
                "runtime.container_image_digest is a repeated-digit placeholder"
            )
        if "REPLACE" in self.runtime.container_image.upper():
            warnings.append(
                "runtime.container_image still contains 'REPLACE'"
            )
        for index, artifact in enumerate(self.artifacts):
            if _looks_like_placeholder_hex(artifact.sha256):
                warnings.append(
                    f"artifacts[{index}] ({artifact.name}).sha256 is a "
                    "repeated-digit placeholder"
                )
        return warnings

    def summary_lines(self) -> list[str]:
        serving = self.serving
        kv_gib = serving.kv_cache_bytes_per_rank / float(1 << 30)
        lines = [
            f"site        : {self.name} (schema {self.schema_version})",
            f"description : {self.description}",
            f"source      : {self.source or '<in-memory>'}",
            "",
            f"topology    : closed 4-cycle, mtu={self.topology.mtu}, "
            f"link={self.topology.link_speed_mbps} Mb/s, "
            f"jumbo payload={self.topology.jumbo_payload_bytes}B",
        ]
        for edge in self.topology.edges:
            first, second = edge.endpoints
            port_a = self._port_for(first, edge.id)
            port_b = self._port_for(second, edge.id)
            lines.append(
                f"  {edge.id:<10} {str(edge.subnet):<18} "
                f"rank{first} {port_a.address} ({port_a.interface}/"
                f"{port_a.rdma_key} gid{port_a.roce_gid_index})"
                f"  <->  rank{second} {port_b.address} ({port_b.interface}/"
                f"{port_b.rdma_key} gid{port_b.roce_gid_index})"
            )
        lines.append("")
        lines.append("ranks       :")
        for rank in self.ranks:
            peers = ", ".join(
                f"rank{peer.rank}@{peer.address}"
                for peer in rank.transport_peers
            )
            lines.append(
                f"  rank{rank.id}  {rank.ssh_target:<28} "
                f"mgmt {rank.management.interface}="
                f"{rank.management.address}  peers: {peers}"
            )
        lines.extend([
            "",
            f"runtime     : {self.runtime.container_image}",
            f"              digest {self.runtime.container_image_digest}",
            f"              model  {self.runtime.model_repo}"
            f"@{self.runtime.model_revision} -> {self.runtime.model_path}",
            f"              checkpoint sha256 "
            f"{self.runtime.checkpoint_sha256}",
            f"serving     : tp={serving.tensor_parallel_size} "
            f"dcp={serving.decode_context_parallel_size} "
            f"mtp={serving.mtp_mode}/{serving.mtp_tokens} "
            f"max_model_len={serving.max_model_len} "
            f"kv={kv_gib:.1f} GiB/rank "
            f"max_num_seqs={serving.max_num_seqs}",
            f"              master rank{serving.master_rank} "
            f"api:{serving.api_port} rendezvous:{serving.master_port}",
            f"paths       : jit={self.paths.jit_cache_dir} "
            f"ctx={self.paths.context_cache_dir} "
            f"evidence={self.paths.evidence_dir}",
            f"artifacts   : {len(self.artifacts)} pinned "
            f"({', '.join(a.name for a in self.artifacts)})",
            f"preflight   : ssh timeout "
            f"{self.preflight.ssh_timeout_seconds}s, "
            f"required-free ports "
            f"{list(self.preflight.required_free_ports) or 'none'}",
        ])
        return lines

    def _port_for(self, rank_id: int, edge_id: str) -> RingPort:
        for port in self.rank(rank_id).ring_ports:
            if port.edge == edge_id:
                return port
        raise KeyError(f"rank {rank_id} has no port on edge {edge_id}")


# ==========================================================================
# Section validators
# ==========================================================================


def _validate_topology(raw: Any) -> Topology:
    data = _mapping(raw, "topology")
    _keys(data, "topology", ("mtu", "link_speed_mbps", "edges"))
    mtu = _integer(data, "topology", "mtu", MTU_RANGE)
    speed = _integer(data, "topology", "link_speed_mbps", LINK_SPEED_RANGE)

    edges_raw = _sequence(data["edges"], "topology.edges")
    if len(edges_raw) != len(RANK_IDS):
        raise SiteConfigError(
            "topology.edges",
            f"a four-node ring has exactly {len(RANK_IDS)} edges, "
            f"got {len(edges_raw)}",
        )

    edges: list[Edge] = []
    seen_ids: dict[str, int] = {}
    seen_subnets: dict[str, str] = {}
    seen_pairs: dict[frozenset[int], str] = {}
    for index, entry in enumerate(edges_raw):
        where = f"topology.edges[{index}]"
        item = _mapping(entry, where)
        _keys(item, where, ("id", "subnet", "endpoints"))
        edge_id = _string(item, where, "id", _NAME_RE, "edge id")
        if edge_id in seen_ids:
            raise SiteConfigError(
                f"{where}.id",
                f"duplicate edge id {edge_id!r} (also topology.edges"
                f"[{seen_ids[edge_id]}])",
            )
        seen_ids[edge_id] = index

        subnet = _ipv4_network(item, where, "subnet")
        if str(subnet) in seen_subnets:
            raise SiteConfigError(
                f"{where}.subnet",
                f"subnet {subnet} is already used by edge "
                f"{seen_subnets[str(subnet)]!r}; every ring edge needs its "
                "own distinct /24",
            )
        for existing in edges:
            if subnet.overlaps(existing.subnet):
                raise SiteConfigError(
                    f"{where}.subnet",
                    f"subnet {subnet} overlaps edge {existing.id!r} "
                    f"subnet {existing.subnet}",
                )
        seen_subnets[str(subnet)] = edge_id

        endpoints_raw = _sequence(item["endpoints"], f"{where}.endpoints")
        if len(endpoints_raw) != 2:
            raise SiteConfigError(
                f"{where}.endpoints",
                f"an edge joins exactly 2 ranks, got {len(endpoints_raw)}",
            )
        endpoints: list[int] = []
        for position, value in enumerate(endpoints_raw):
            if isinstance(value, bool) or not isinstance(value, int):
                raise SiteConfigError(
                    f"{where}.endpoints[{position}]",
                    f"expected a rank id integer, got {type(value).__name__}",
                )
            if value not in RANK_IDS:
                raise SiteConfigError(
                    f"{where}.endpoints[{position}]",
                    f"rank id {value} is not one of {list(RANK_IDS)}",
                )
            endpoints.append(value)
        if endpoints[0] == endpoints[1]:
            raise SiteConfigError(
                f"{where}.endpoints",
                f"self-loop: rank {endpoints[0]} cannot be both endpoints",
            )
        pair = frozenset(endpoints)
        if pair in seen_pairs:
            raise SiteConfigError(
                f"{where}.endpoints",
                f"ranks {sorted(pair)} are already joined by edge "
                f"{seen_pairs[pair]!r}; a ring has no parallel edges",
            )
        seen_pairs[pair] = edge_id
        edges.append(
            Edge(id=edge_id, subnet=subnet,
                 endpoints=(endpoints[0], endpoints[1]))
        )

    _require_single_cycle(edges)
    return Topology(mtu=mtu, link_speed_mbps=speed, edges=tuple(edges))


def _require_single_cycle(edges: Sequence[Edge]) -> None:
    """Reject any edge set that is not one closed cycle through all ranks."""
    degree: dict[int, list[Edge]] = {rank_id: [] for rank_id in RANK_IDS}
    for edge in edges:
        for endpoint in edge.endpoints:
            degree[endpoint].append(edge)

    for rank_id in RANK_IDS:
        incident = degree[rank_id]
        if len(incident) != RING_PORTS_PER_RANK:
            raise SiteConfigError(
                "topology.edges",
                f"rank {rank_id} is on {len(incident)} edge(s) "
                f"({', '.join(edge.id for edge in incident) or 'none'}); "
                f"every rank in a ring must be on exactly "
                f"{RING_PORTS_PER_RANK}",
            )

    start = RANK_IDS[0]
    current = start
    used: set[str] = set()
    visited = [start]
    for _ in range(len(edges)):
        candidates = [
            edge for edge in degree[current] if edge.id not in used
        ]
        if not candidates:
            break
        edge = candidates[0]
        used.add(edge.id)
        current = edge.other(current)
        visited.append(current)

    closed = current == start and len(used) == len(edges)
    covered = set(visited) == set(RANK_IDS)
    if not (closed and covered):
        raise SiteConfigError(
            "topology.edges",
            "edges do not form a single closed ring through all four ranks; "
            f"walking from rank {start} visited {visited} using "
            f"{sorted(used)}. Check the endpoint pairs - a ring is "
            "0-1, 1-2, 2-3, 3-0 (in any labelling), not two disjoint pairs.",
        )


def _validate_ring_port(raw: Any, where: str, topology: Topology) -> RingPort:
    item = _mapping(raw, where)
    _keys(
        item, where,
        ("edge", "interface", "address", "rdma_device", "rdma_port",
         "roce_gid_index"),
    )
    edge_id = _string(item, where, "edge", _NAME_RE, "edge id")
    try:
        edge = topology.by_id(edge_id)
    except KeyError:
        raise SiteConfigError(
            f"{where}.edge",
            f"unknown edge {edge_id!r}; declared edges are "
            + ", ".join(repr(entry.id) for entry in topology.edges),
        ) from None
    interface = _string(
        item, where, "interface", _NAME_RE, "interface name"
    )
    address = _ipv4(item, where, "address")
    rdma_device = _string(
        item, where, "rdma_device", _NAME_RE, "RDMA device name"
    )
    rdma_port = _integer(item, where, "rdma_port", RDMA_PORT_RANGE)
    gid_index = _integer(item, where, "roce_gid_index", GID_INDEX_RANGE)
    return RingPort(
        edge=edge.id,
        interface=interface,
        address=address,
        rdma_device=rdma_device,
        rdma_port=rdma_port,
        roce_gid_index=gid_index,
        prefix_length=edge.subnet.prefixlen,
    )


def _validate_ranks(raw: Any, topology: Topology) -> tuple[Rank, ...]:
    entries = _sequence(raw, "ranks")
    if len(entries) != len(RANK_IDS):
        raise SiteConfigError(
            "ranks",
            f"expected exactly {len(RANK_IDS)} ranks, got {len(entries)}",
        )

    ranks: list[Rank] = []
    seen_ids: dict[int, int] = {}
    for index, entry in enumerate(entries):
        where = f"ranks[{index}]"
        item = _mapping(entry, where)
        _keys(
            item, where,
            ("id", "ssh_target", "management", "ring_ports",
             "transport_peers"),
        )
        raw_id = item["id"]
        if isinstance(raw_id, bool) or not isinstance(raw_id, int):
            raise SiteConfigError(
                f"{where}.id",
                f"expected an integer rank id, got {type(raw_id).__name__}",
            )
        if raw_id not in RANK_IDS:
            raise SiteConfigError(
                f"{where}.id",
                f"rank id {raw_id} is not one of {list(RANK_IDS)}",
            )
        if raw_id in seen_ids:
            raise SiteConfigError(
                f"{where}.id",
                f"duplicate rank id {raw_id} (also ranks[{seen_ids[raw_id]}])",
            )
        seen_ids[raw_id] = index

        ssh_target = _string(
            item, where, "ssh_target", _SSH_TARGET_RE, "ssh target"
        )

        mgmt_where = f"{where}.management"
        mgmt = _mapping(item["management"], mgmt_where)
        _keys(mgmt, mgmt_where, ("interface", "address"))
        management = Management(
            interface=_string(
                mgmt, mgmt_where, "interface", _NAME_RE, "interface name"
            ),
            address=_ipv4(mgmt, mgmt_where, "address"),
        )

        ports_raw = _sequence(item["ring_ports"], f"{where}.ring_ports")
        if len(ports_raw) != RING_PORTS_PER_RANK:
            raise SiteConfigError(
                f"{where}.ring_ports",
                f"each rank has exactly {RING_PORTS_PER_RANK} ring ports, "
                f"got {len(ports_raw)}",
            )
        ports = tuple(
            _validate_ring_port(
                port_raw, f"{where}.ring_ports[{position}]", topology
            )
            for position, port_raw in enumerate(ports_raw)
        )
        if ports[0].edge == ports[1].edge:
            raise SiteConfigError(
                f"{where}.ring_ports",
                f"both ports claim edge {ports[0].edge!r}; a rank sits on two "
                "different edges",
            )
        if ports[0].interface == ports[1].interface:
            raise SiteConfigError(
                f"{where}.ring_ports",
                f"both ports use interface {ports[0].interface!r}; the two "
                "ring ports are distinct netdevs",
            )
        for position, port in enumerate(ports):
            if port.interface == management.interface:
                raise SiteConfigError(
                    f"{where}.ring_ports[{position}].interface",
                    f"interface {port.interface!r} is also the management "
                    "interface; ring traffic must not share the management "
                    "NIC",
                )
        if ports[0].rdma_key == ports[1].rdma_key:
            raise SiteConfigError(
                f"{where}.ring_ports",
                f"both ports map to RDMA {ports[0].rdma_key!r}; give each "
                "port its own device (mlx5_0/mlx5_1) or its own rdma_port",
            )

        peers_raw = _sequence(
            item["transport_peers"], f"{where}.transport_peers"
        )
        if len(peers_raw) != RING_PORTS_PER_RANK:
            raise SiteConfigError(
                f"{where}.transport_peers",
                f"each rank has exactly {RING_PORTS_PER_RANK} control-channel "
                f"peers (its two ring neighbours), got {len(peers_raw)}",
            )
        peers: list[TransportPeer] = []
        for position, peer_raw in enumerate(peers_raw):
            peer_where = f"{where}.transport_peers[{position}]"
            peer_item = _mapping(peer_raw, peer_where)
            _keys(peer_item, peer_where, ("rank", "address"))
            peer_rank = peer_item["rank"]
            if isinstance(peer_rank, bool) or not isinstance(peer_rank, int):
                raise SiteConfigError(
                    f"{peer_where}.rank",
                    f"expected an integer rank id, "
                    f"got {type(peer_rank).__name__}",
                )
            if peer_rank not in RANK_IDS:
                raise SiteConfigError(
                    f"{peer_where}.rank",
                    f"rank id {peer_rank} is not one of {list(RANK_IDS)}",
                )
            if peer_rank == raw_id:
                raise SiteConfigError(
                    f"{peer_where}.rank",
                    f"rank {raw_id} cannot be its own control-channel peer",
                )
            peers.append(
                TransportPeer(
                    rank=peer_rank,
                    address=_ipv4(peer_item, peer_where, "address"),
                )
            )
        if peers[0].rank == peers[1].rank:
            raise SiteConfigError(
                f"{where}.transport_peers",
                f"duplicate peer rank {peers[0].rank}",
            )
        if peers[0].address == peers[1].address:
            raise SiteConfigError(
                f"{where}.transport_peers",
                f"duplicate peer address {peers[0].address}",
            )

        ranks.append(
            Rank(
                id=raw_id,
                ssh_target=ssh_target,
                management=management,
                ring_ports=ports,
                transport_peers=tuple(peers),
            )
        )

    ranks.sort(key=lambda entry: entry.id)
    missing = sorted(set(RANK_IDS) - set(seen_ids))
    if missing:
        raise SiteConfigError(
            "ranks", "missing rank id(s): " + ", ".join(str(x) for x in missing)
        )
    return tuple(ranks)


def _cross_validate(topology: Topology, ranks: Sequence[Rank]) -> None:
    """Join ranks to edges and enforce every structural invariant."""
    # 1. every declared edge is claimed exactly once by each of its endpoints
    claims: dict[str, list[tuple[int, int, RingPort]]] = {
        edge.id: [] for edge in topology.edges
    }
    for rank in ranks:
        for position, port in enumerate(rank.ring_ports):
            claims[port.edge].append((rank.id, position, port))

    for edge in topology.edges:
        entries = claims[edge.id]
        if len(entries) != 2:
            owners = ", ".join(
                f"ranks[{rank_id}].ring_ports[{position}]"
                for rank_id, position, _ in entries
            )
            raise SiteConfigError(
                f"topology.edges[{edge.id}]",
                f"edge {edge.id!r} is claimed by {len(entries)} ring port(s) "
                f"({owners or 'none'}); every edge must be claimed exactly "
                "twice, once per endpoint",
            )
        claimed_ranks = {entries[0][0], entries[1][0]}
        declared = set(edge.endpoints)
        if claimed_ranks != declared:
            raise SiteConfigError(
                f"topology.edges[{edge.id}]",
                f"edge {edge.id!r} declares endpoints {sorted(declared)} but "
                f"is claimed by ranks {sorted(claimed_ranks)}",
            )

        # 2. addresses live in the edge subnet, are usable hosts, and differ
        for rank_id, position, port in entries:
            where = f"ranks[{rank_id}].ring_ports[{position}].address"
            if port.address not in edge.subnet:
                raise SiteConfigError(
                    where,
                    f"{port.address} is not inside edge {edge.id!r} subnet "
                    f"{edge.subnet}",
                )
            if port.address == edge.subnet.network_address:
                raise SiteConfigError(
                    where,
                    f"{port.address} is the network address of "
                    f"{edge.subnet} and cannot be assigned to a port",
                )
            if port.address == edge.subnet.broadcast_address:
                raise SiteConfigError(
                    where,
                    f"{port.address} is the broadcast address of "
                    f"{edge.subnet} and cannot be assigned to a port",
                )
        first, second = entries
        if first[2].address == second[2].address:
            raise SiteConfigError(
                f"ranks[{second[0]}].ring_ports[{second[1]}].address",
                f"both ends of edge {edge.id!r} use {second[2].address}; the "
                "two endpoints of a cable need different addresses",
            )

        # 3. resolve peers for the jumbo-ping / GID checks
        first[2].peer_rank = second[0]
        first[2].peer_address = second[2].address
        second[2].peer_rank = first[0]
        second[2].peer_address = first[2].address

    # 4. global address uniqueness across ring + management
    seen: dict[str, str] = {}
    for rank in ranks:
        mgmt_field = f"ranks[{rank.id}].management.address"
        key = str(rank.management.address)
        if key in seen:
            raise SiteConfigError(
                mgmt_field,
                f"address {key} is already used by {seen[key]}",
            )
        seen[key] = mgmt_field
        for position, port in enumerate(rank.ring_ports):
            port_field = f"ranks[{rank.id}].ring_ports[{position}].address"
            key = str(port.address)
            if key in seen:
                raise SiteConfigError(
                    port_field,
                    f"address {key} is already used by {seen[key]}",
                )
            seen[key] = port_field

    # 5. management must not sit inside a ring subnet
    for rank in ranks:
        for edge in topology.edges:
            if rank.management.address in edge.subnet:
                raise SiteConfigError(
                    f"ranks[{rank.id}].management.address",
                    f"{rank.management.address} is inside ring edge "
                    f"{edge.id!r} subnet {edge.subnet}; the management "
                    "network must be separate from every ring subnet",
                )

    # 6. ssh targets unique
    targets: dict[str, int] = {}
    for rank in ranks:
        if rank.ssh_target in targets:
            raise SiteConfigError(
                f"ranks[{rank.id}].ssh_target",
                f"ssh target {rank.ssh_target!r} is already used by rank "
                f"{targets[rank.ssh_target]}",
            )
        targets[rank.ssh_target] = rank.id

    # 7. control-channel peers must be exactly the ring neighbours
    mgmt_owner = {
        str(rank.management.address): rank.id for rank in ranks
    }
    for rank in ranks:
        neighbours = set(rank.neighbour_ranks)
        declared = {peer.rank for peer in rank.transport_peers}
        if declared != neighbours:
            raise SiteConfigError(
                f"ranks[{rank.id}].transport_peers",
                f"peers name ranks {sorted(declared)} but rank {rank.id}'s "
                f"ring neighbours are {sorted(neighbours)}; the control "
                "channel must rendezvous with the ranks it is cabled to",
            )
        for position, peer in enumerate(rank.transport_peers):
            peer_where = f"ranks[{rank.id}].transport_peers[{position}]"
            key = str(peer.address)
            if key == str(rank.management.address):
                raise SiteConfigError(
                    f"{peer_where}.address",
                    f"{key} is this rank's own management address",
                )
            if any(key == str(port.address) for port in rank.ring_ports):
                raise SiteConfigError(
                    f"{peer_where}.address",
                    f"{key} is one of this rank's own ring addresses",
                )
            owner = mgmt_owner.get(key)
            if owner is not None and owner != peer.rank:
                raise SiteConfigError(
                    f"{peer_where}.address",
                    f"{key} is the management address of rank {owner}, but "
                    f"this peer entry names rank {peer.rank}; a stale peer "
                    "table like this points the control channel at the wrong "
                    "node",
                )


def _validate_runtime(raw: Any) -> Runtime:
    data = _mapping(raw, "runtime")
    _keys(
        data, "runtime",
        ("container_image", "container_image_digest", "model_path",
         "model_repo", "model_revision", "checkpoint_sha256"),
    )
    image = _string(
        data, "runtime", "container_image", _IMAGE_REF_RE,
        "container image reference",
    )
    digest = data["container_image_digest"]
    if not isinstance(digest, str) or not _DIGEST_RE.match(digest):
        raise SiteConfigError(
            "runtime.container_image_digest",
            "must be 'sha256:' followed by exactly 64 lowercase hex "
            f"characters, got {digest!r}",
        )
    model_path = _string(
        data, "runtime", "model_path", _REMOTE_PATH_RE, "absolute path"
    )
    model_repo = _string(data, "runtime", "model_repo")
    revision = data["model_revision"]
    if not isinstance(revision, str) or not _REVISION_RE.match(revision):
        raise SiteConfigError(
            "runtime.model_revision",
            "must be an immutable 40-character lowercase hex revision "
            f"(never a branch or tag), got {revision!r}",
        )
    return Runtime(
        container_image=image,
        container_image_digest=digest,
        model_path=model_path,
        model_repo=model_repo,
        model_revision=revision,
        checkpoint_sha256=_sha256(data, "runtime", "checkpoint_sha256"),
    )


def _validate_serving(raw: Any) -> Serving:
    data = _mapping(raw, "serving")
    _keys(
        data, "serving",
        ("tensor_parallel_size", "decode_context_parallel_size", "mtp_mode",
         "mtp_tokens", "max_model_len", "kv_cache_bytes_per_rank",
         "max_num_seqs", "master_rank", "api_port", "master_port"),
    )
    tp = _integer(data, "serving", "tensor_parallel_size", TP_RANGE)
    dcp = _integer(data, "serving", "decode_context_parallel_size", DCP_RANGE)
    if dcp > tp:
        raise SiteConfigError(
            "serving.decode_context_parallel_size",
            f"{dcp} exceeds tensor_parallel_size {tp}",
        )
    if tp % dcp != 0:
        raise SiteConfigError(
            "serving.decode_context_parallel_size",
            f"tensor_parallel_size {tp} is not divisible by {dcp}",
        )
    mode = _string(data, "serving", "mtp_mode")
    if mode not in MTP_MODES:
        raise SiteConfigError(
            "serving.mtp_mode",
            f"{mode!r} is not one of {list(MTP_MODES)}",
        )
    tokens = _integer(data, "serving", "mtp_tokens", MTP_TOKENS_RANGE)
    if mode == "off" and tokens != 0:
        raise SiteConfigError(
            "serving.mtp_tokens",
            f"must be 0 when mtp_mode is 'off', got {tokens}",
        )
    if mode != "off" and tokens < 1:
        raise SiteConfigError(
            "serving.mtp_tokens",
            f"must be at least 1 when mtp_mode is {mode!r}, got {tokens}",
        )
    master_rank = _integer(data, "serving", "master_rank", (0, max(RANK_IDS)))
    api_port = _integer(data, "serving", "api_port", PORT_RANGE)
    master_port = _integer(data, "serving", "master_port", PORT_RANGE)
    if api_port == master_port:
        raise SiteConfigError(
            "serving.master_port",
            f"must differ from serving.api_port ({api_port})",
        )
    return Serving(
        tensor_parallel_size=tp,
        decode_context_parallel_size=dcp,
        mtp_mode=mode,
        mtp_tokens=tokens,
        max_model_len=_integer(
            data, "serving", "max_model_len", MAX_MODEL_LEN_RANGE
        ),
        kv_cache_bytes_per_rank=_integer(
            data, "serving", "kv_cache_bytes_per_rank", KV_BYTES_RANGE
        ),
        max_num_seqs=_integer(
            data, "serving", "max_num_seqs", MAX_NUM_SEQS_RANGE
        ),
        master_rank=master_rank,
        api_port=api_port,
        master_port=master_port,
    )


def _validate_paths(raw: Any) -> Paths:
    data = _mapping(raw, "paths")
    _keys(
        data, "paths",
        ("jit_cache_dir", "context_cache_dir", "evidence_dir",
         "min_free_bytes"),
    )
    jit = _string(
        data, "paths", "jit_cache_dir", _REMOTE_PATH_RE, "absolute path"
    )
    ctx = _string(
        data, "paths", "context_cache_dir", _REMOTE_PATH_RE, "absolute path"
    )
    if jit == ctx:
        raise SiteConfigError(
            "paths.context_cache_dir",
            f"must differ from paths.jit_cache_dir ({jit})",
        )
    evidence = _string(
        data, "paths", "evidence_dir", _LOCAL_PATH_RE, "local path"
    )
    min_where = "paths.min_free_bytes"
    min_free = _mapping(data["min_free_bytes"], min_where)
    _keys(min_free, min_where, ("jit_cache", "context_cache"))
    return Paths(
        jit_cache_dir=jit,
        context_cache_dir=ctx,
        evidence_dir=evidence,
        jit_cache_min_free_bytes=_integer(
            min_free, min_where, "jit_cache", MIN_FREE_BYTES_RANGE
        ),
        context_cache_min_free_bytes=_integer(
            min_free, min_where, "context_cache", MIN_FREE_BYTES_RANGE
        ),
    )


def _validate_artifacts(raw: Any) -> tuple[Artifact, ...]:
    entries = _sequence(raw, "artifacts")
    # Digest-pinned images may bake every native dependency into the image and
    # verify it from runtime-manifest.json before vLLM starts. In that lane the
    # image digest is the host-side artifact contract, so an empty loose-file
    # list is valid. Host-mounted deployments should still enumerate every
    # mounted binary here.
    artifacts: list[Artifact] = []
    seen_paths: dict[str, int] = {}
    seen_names: dict[str, int] = {}
    for index, entry in enumerate(entries):
        where = f"artifacts[{index}]"
        item = _mapping(entry, where)
        _keys(item, where, ("name", "path", "sha256", "executable"))
        name = _string(item, where, "name", _NAME_RE, "artifact name")
        if name in seen_names:
            raise SiteConfigError(
                f"{where}.name",
                f"duplicate artifact name {name!r} (also artifacts"
                f"[{seen_names[name]}])",
            )
        seen_names[name] = index
        path = _string(item, where, "path", _REMOTE_PATH_RE, "absolute path")
        if path in seen_paths:
            raise SiteConfigError(
                f"{where}.path",
                f"duplicate artifact path {path!r} (also artifacts"
                f"[{seen_paths[path]}])",
            )
        seen_paths[path] = index
        artifacts.append(
            Artifact(
                name=name,
                path=path,
                sha256=_sha256(item, where, "sha256"),
                executable=_boolean(item, where, "executable"),
            )
        )
    return tuple(artifacts)


def _validate_preflight(raw: Any) -> PreflightOptions:
    data = _mapping(raw, "preflight")
    _keys(data, "preflight", ("ssh_timeout_seconds", "required_free_ports"))
    timeout = _integer(
        data, "preflight", "ssh_timeout_seconds", SSH_TIMEOUT_RANGE
    )
    ports_raw = _sequence(
        data["required_free_ports"], "preflight.required_free_ports"
    )
    ports: list[int] = []
    for index, value in enumerate(ports_raw):
        where = f"preflight.required_free_ports[{index}]"
        if isinstance(value, bool) or not isinstance(value, int):
            raise SiteConfigError(
                where,
                f"expected a port number, got {type(value).__name__}",
            )
        low, high = PORT_RANGE
        if not low <= value <= high:
            raise SiteConfigError(
                where, f"{value} is out of range [{low}, {high}]"
            )
        if value in ports:
            raise SiteConfigError(where, f"duplicate port {value}")
        ports.append(value)
    return PreflightOptions(
        ssh_timeout_seconds=timeout, required_free_ports=tuple(ports)
    )


# ==========================================================================
# Entry points
# ==========================================================================


def validate_site(document: Any, source: str | None = None) -> SiteConfig:
    """Validate an already-parsed site document and return a :class:`SiteConfig`.

    Raises :class:`SiteConfigError` on the first problem found, naming the
    exact field.
    """
    data = _mapping(document, "<root>")
    _keys(
        data, "<root>",
        ("schema_version", "site", "topology", "ranks", "runtime", "serving",
         "paths", "artifacts", "preflight"),
    )
    version = data["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise SiteConfigError(
            "schema_version",
            f"expected integer {SCHEMA_VERSION}, got {type(version).__name__}",
        )
    if version != SCHEMA_VERSION:
        raise SiteConfigError(
            "schema_version",
            f"unsupported schema version {version}; this tool understands "
            f"version {SCHEMA_VERSION}",
        )

    site = _mapping(data["site"], "site")
    _keys(site, "site", ("name", "description"))
    name = _string(site, "site", "name")
    description = _string(site, "site", "description")

    topology = _validate_topology(data["topology"])
    ranks = _validate_ranks(data["ranks"], topology)
    _cross_validate(topology, ranks)

    runtime = _validate_runtime(data["runtime"])
    serving = _validate_serving(data["serving"])
    paths = _validate_paths(data["paths"])
    artifacts = _validate_artifacts(data["artifacts"])
    preflight = _validate_preflight(data["preflight"])

    if serving.master_rank not in {rank.id for rank in ranks}:
        raise SiteConfigError(
            "serving.master_rank",
            f"rank {serving.master_rank} is not defined in ranks",
        )

    return SiteConfig(
        schema_version=version,
        name=name,
        description=description,
        topology=topology,
        ranks=ranks,
        runtime=runtime,
        serving=serving,
        paths=paths,
        artifacts=artifacts,
        preflight=preflight,
        source=source,
    )


def parse_site_yaml(text: str, source: str | None = None) -> SiteConfig:
    """Parse YAML text and validate it."""
    if _yaml is None:
        raise SiteConfigError(
            "<pyyaml>",
            "PyYAML is required to read site configuration files but is not "
            "installed. Install it with 'python -m pip install PyYAML' "
            f"(import failed with: {_YAML_IMPORT_ERROR})",
        )
    try:
        document = _yaml.safe_load(text)
    except Exception as exc:  # yaml.YAMLError and friends
        raise SiteConfigError(
            source or "<yaml>", f"could not parse YAML: {exc}"
        ) from None
    if document is None:
        raise SiteConfigError(source or "<yaml>", "file is empty")
    return validate_site(document, source=source)


def load_site(path: str | Path) -> SiteConfig:
    """Load and validate a site configuration file.

    This is the public API used by ``preflight.py`` and by operators.
    """
    resolved = Path(path)
    try:
        text = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SiteConfigError(
            str(resolved),
                "site configuration file not found; copy "
                "scripts/config/exl3-r7-site.example.yaml and fill it in",
        ) from None
    except OSError as exc:
        raise SiteConfigError(
            str(resolved), f"could not read site configuration: {exc}"
        ) from None
    return parse_site_yaml(text, source=str(resolved))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sparkring_site",
        description=(
            "Validate a SparkRing site configuration and print a summary. "
            "Exit status is 0 only when the file is fully valid."
        ),
    )
    parser.add_argument(
        "site", nargs="?", default="scripts/config/site.yaml",
        help="path to the site YAML (default: scripts/config/site.yaml)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the normalised configuration as JSON instead of a summary",
    )
    parser.add_argument(
        "--strict-placeholders", action="store_true",
        help="treat leftover example placeholders as failures",
    )
    args = parser.parse_args(argv)

    try:
        site = load_site(args.site)
    except SiteConfigError as exc:
        print(f"INVALID {args.site}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    warnings = site.placeholder_warnings()
    if args.json:
        payload = site.to_dict()
        payload["placeholder_warnings"] = warnings
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for line in site.summary_lines():
            print(line)
        if warnings:
            print("")
            print(f"placeholder warnings ({len(warnings)}):")
            for warning in warnings:
                print(f"  [WARN] {warning}")
        print("")
        print(f"OK: {args.site} is a valid SparkRing site configuration")

    if warnings and args.strict_placeholders:
        print(
            "FAIL: --strict-placeholders and the config still contains "
            f"{len(warnings)} example placeholder(s)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
