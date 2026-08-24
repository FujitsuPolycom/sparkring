#!/usr/bin/env python3
"""Bootstrap a blank set of DGX Sparks into a Ring Doctor inventory.

Passwords are handled only by the system ``ssh-copy-id`` interaction. This
module never accepts, reads, stores, logs, or forwards a password itself.
"""

from __future__ import annotations

import dataclasses
import base64
import getpass
import ipaddress
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, Sequence

import yaml

try:
    from .sparkring_cluster import ClusterConfig, validate_cluster, write_cluster
    from .sparkring_site import SUPPORTED_RING_SIZES
except ImportError:  # Direct execution
    from sparkring_cluster import ClusterConfig, validate_cluster, write_cluster
    from sparkring_site import SUPPORTED_RING_SIZES

TARGET_RE = re.compile(r"^[A-Za-z0-9._-]+@([0-9]{1,3}(?:\.[0-9]{1,3}){3})$")
PUBLIC_KEY_RE = re.compile(r"^ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]+)?$")
DEFAULT_FABRIC_SUPERNET = "198.18.0.0/21"
DEFAULT_INTERFACES = ("enp1s0f0np0", "enp1s0f1np1")
DEFAULT_RDMA_DEVICES = ("rocep1s0f0", "rocep1s0f1")
PROBE_ADDR_MARKER = "__SPARKRING_ADDR__"
PROBE_RDMA_MARKER = "__SPARKRING_RDMA__"


class BootstrapError(RuntimeError):
    """Bootstrap cannot safely continue."""


@dataclasses.dataclass(frozen=True)
class NodeFacts:
    rank: int
    target: str
    hostname: str
    management_interface: str
    management_address: ipaddress.IPv4Address
    fabric_interfaces: tuple[str, str]
    rdma_devices: tuple[str, str]


def parse_target(value: str) -> tuple[str, ipaddress.IPv4Address]:
    match = TARGET_RE.fullmatch(value)
    if not match:
        raise BootstrapError(
            f"target {value!r} must have the form username@IPv4-address"
        )
    try:
        address = ipaddress.ip_address(match.group(1))
    except ValueError as exc:
        raise BootstrapError(f"target {value!r} has an invalid IPv4 address") from exc
    if not isinstance(address, ipaddress.IPv4Address):
        raise BootstrapError("only IPv4 management addresses are supported")
    return value.split("@", 1)[0], address


def _run(
    argv: Sequence[str],
    *,
    input_text: str | None = None,
    capture: bool = True,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        input=input_text,
        capture_output=capture,
        text=True,
        check=check,
    )


def ensure_local_key() -> Path:
    ssh_dir = Path.home() / ".ssh"
    private_key = ssh_dir / "id_ed25519"
    public_key = ssh_dir / "id_ed25519.pub"
    if private_key.exists() and public_key.exists():
        return public_key
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    completed = _run(
        [
            "ssh-keygen",
            "-q",
            "-t",
            "ed25519",
            "-N",
            "",
            "-C",
            f"{getpass.getuser()}@{os.uname().nodename}-sparkring",
            "-f",
            str(private_key),
        ]
    )
    if completed.returncode:
        raise BootstrapError(completed.stderr.strip() or "ssh-keygen failed")
    return public_key


def authorize_local_key(
    public_key: Path,
    *,
    ssh_dir: Path | None = None,
) -> Path:
    """Idempotently authorize the bootstrap key for head self-SSH."""
    directory = ssh_dir or (Path.home() / ".ssh")
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    directory.chmod(0o700)
    authorized = directory / "authorized_keys"
    key = public_key.read_text(encoding="utf-8").strip()
    if not PUBLIC_KEY_RE.fullmatch(key):
        raise BootstrapError(f"unexpected public key format in {public_key}")
    existing = authorized.read_text(encoding="utf-8") if authorized.exists() else ""
    lines = [line.strip() for line in existing.splitlines() if line.strip()]
    if key not in lines:
        if authorized.exists():
            backup = directory / (
                "authorized_keys.before-sparkring-"
                + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
            )
            backup.write_bytes(authorized.read_bytes())
            backup.chmod(0o600)
        lines.append(key)
        authorized.write_text("\n".join(lines) + "\n", encoding="utf-8")
    authorized.chmod(0o600)
    return authorized


def _known_host_exists(host: str) -> bool:
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    if not known_hosts.exists():
        return False
    result = _run(["ssh-keygen", "-F", host, "-f", str(known_hosts)])
    return result.returncode == 0


def trust_host(
    target: str,
    *,
    confirm: Callable[[str], str] = input,
) -> None:
    _user, address = parse_target(target)
    host = str(address)
    if _known_host_exists(host):
        return
    scan = _run(["ssh-keyscan", "-T", "5", "-t", "ed25519", host])
    lines = [
        line for line in scan.stdout.splitlines()
        if line and not line.startswith("#")
    ]
    if scan.returncode or len(lines) != 1:
        raise BootstrapError(f"could not obtain one Ed25519 host key from {host}")
    fingerprint = _run(
        ["ssh-keygen", "-lf", "-"], input_text=lines[0] + "\n"
    )
    if fingerprint.returncode:
        raise BootstrapError(f"could not fingerprint the host key for {host}")
    answer = confirm(
        f"Trust {target} host key?\n  {fingerprint.stdout.strip()}\nType yes: "
    )
    if answer.strip().lower() != "yes":
        raise BootstrapError(f"host key for {target} was not accepted")
    known_hosts = Path.home() / ".ssh" / "known_hosts"
    known_hosts.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with known_hosts.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(lines[0] + "\n")
    known_hosts.chmod(0o600)


def enroll_target(target: str, public_key: Path) -> None:
    trust_host(target)
    probe = _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, "true"]
    )
    if probe.returncode == 0:
        return
    print(f"Password required once to enroll {target}.")
    result = _run(
        ["ssh-copy-id", "-i", str(public_key), target], capture=False
    )
    if result.returncode:
        raise BootstrapError(f"ssh-copy-id failed for {target}")
    verified = _run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", target, "true"]
    )
    if verified.returncode:
        raise BootstrapError(
            f"key enrollment for {target} did not produce non-interactive SSH: "
            f"{verified.stderr.strip()}"
        )


def _probe_command() -> str:
    return (
        "hostname\n"
        f"printf '%s\\n' {PROBE_ADDR_MARKER}\n"
        "ip -j -4 addr show\n"
        f"printf '%s\\n' {PROBE_RDMA_MARKER}\n"
        "ibdev2netdev 2>/dev/null || true\n"
    )


def _parse_probe(
    rank: int,
    target: str,
    management_address: ipaddress.IPv4Address,
    output: str,
) -> NodeFacts:
    try:
        hostname_text, remainder = output.split(PROBE_ADDR_MARKER, 1)
        address_text, rdma_text = remainder.split(PROBE_RDMA_MARKER, 1)
        rows = json.loads(address_text.strip())
    except (ValueError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"rank {rank} returned malformed inventory output") from exc
    hostname = hostname_text.strip().splitlines()[0]
    management_interface = None
    observed_interfaces: set[str] = set()
    for row in rows:
        name = row.get("ifname")
        if isinstance(name, str):
            observed_interfaces.add(name)
        for item in row.get("addr_info", []):
            if item.get("family") == "inet" and item.get("local") == str(
                management_address
            ):
                management_interface = name
    if not management_interface:
        raise BootstrapError(
            f"rank {rank} does not hold management address {management_address}"
        )
    missing = [name for name in DEFAULT_INTERFACES if name not in observed_interfaces]
    if missing:
        raise BootstrapError(
            f"rank {rank} is missing expected ConnectX-7 interface(s): "
            + ", ".join(missing)
        )
    mappings = rdma_text.strip()
    for interface, device in zip(DEFAULT_INTERFACES, DEFAULT_RDMA_DEVICES):
        expected = f"{device} port 1 ==> {interface}"
        if expected not in mappings:
            raise BootstrapError(
                f"rank {rank} did not report expected RDMA mapping {expected!r}"
            )
    if management_interface in DEFAULT_INTERFACES:
        raise BootstrapError(
            f"rank {rank} management interface overlaps the fabric"
        )
    return NodeFacts(
        rank=rank,
        target=target,
        hostname=hostname,
        management_interface=management_interface,
        management_address=management_address,
        fabric_interfaces=DEFAULT_INTERFACES,
        rdma_devices=DEFAULT_RDMA_DEVICES,
    )


def probe_local(rank: int, target: str) -> NodeFacts:
    _user, address = parse_target(target)
    result = _run(["sh", "-c", _probe_command()])
    if result.returncode:
        raise BootstrapError(result.stderr.strip() or "local inventory probe failed")
    return _parse_probe(rank, target, address, result.stdout)


def probe_remote(rank: int, target: str) -> NodeFacts:
    _user, address = parse_target(target)
    result = _run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            target,
            _probe_command(),
        ]
    )
    if result.returncode:
        raise BootstrapError(
            f"could not inventory rank {rank} ({target}): "
            f"{result.stderr.strip()}"
        )
    return _parse_probe(rank, target, address, result.stdout)


def _edge_subnets(supernet: str, count: int) -> list[ipaddress.IPv4Network]:
    try:
        network = ipaddress.ip_network(supernet, strict=True)
    except ValueError as exc:
        raise BootstrapError(f"invalid fabric supernet {supernet!r}: {exc}") from exc
    if not isinstance(network, ipaddress.IPv4Network) or network.prefixlen > 24:
        raise BootstrapError("fabric supernet must be an IPv4 network at least /24")
    subnets = list(network.subnets(new_prefix=24))
    if len(subnets) < count:
        raise BootstrapError(
            f"fabric supernet {network} has {len(subnets)} /24s; {count} are required"
        )
    return subnets[:count]


def build_cluster_document(
    facts: Sequence[NodeFacts],
    *,
    name: str,
    fabric_supernet: str,
    mtu: int = 9000,
    link_speed_mbps: int = 200000,
) -> dict:
    size = len(facts)
    if size not in SUPPORTED_RING_SIZES:
        raise BootstrapError(f"ring size must be one of {SUPPORTED_RING_SIZES}")
    ordered = sorted(facts, key=lambda item: item.rank)
    if [item.rank for item in ordered] != list(range(size)):
        raise BootstrapError("facts must contain contiguous ranks starting at zero")
    management_addresses = {item.management_address for item in ordered}
    if len(management_addresses) != size:
        raise BootstrapError("management addresses must be unique")
    subnets = _edge_subnets(fabric_supernet, size)
    if any(
        address in subnet
        for address in management_addresses
        for subnet in subnets
    ):
        raise BootstrapError("fabric supernet overlaps a management address")
    edges = [
        {
            "id": f"r{rank}-r{(rank + 1) % size}",
            "subnet": str(subnets[rank]),
            "endpoints": [rank, (rank + 1) % size],
        }
        for rank in range(size)
    ]
    ranks = []
    for fact in ordered:
        rank = fact.rank
        previous = (rank - 1) % size
        following = (rank + 1) % size
        incoming = subnets[previous]
        outgoing = subnets[rank]
        ranks.append(
            {
                "id": rank,
                "ssh_target": fact.target,
                "management": {
                    "interface": fact.management_interface,
                    "address": str(fact.management_address),
                },
                "ring_ports": [
                    {
                        "edge": f"r{rank}-r{following}",
                        "interface": fact.fabric_interfaces[0],
                        "address": str(list(outgoing.hosts())[9]),
                        "rdma_device": fact.rdma_devices[0],
                        "rdma_port": 1,
                        "roce_gid_index": 3,
                    },
                    {
                        "edge": f"r{previous}-r{rank}",
                        "interface": fact.fabric_interfaces[1],
                        "address": str(list(incoming.hosts())[10]),
                        "rdma_device": fact.rdma_devices[1],
                        "rdma_port": 1,
                        "roce_gid_index": 3,
                    },
                ],
                "transport_peers": [
                    {
                        "rank": previous,
                        "address": str(ordered[previous].management_address),
                    },
                    {
                        "rank": following,
                        "address": str(ordered[following].management_address),
                    },
                ],
            }
        )
    return {
        "schema_version": 1,
        "cluster": {
            "name": name,
            "description": (
                f"Bootstrapped {size}-Spark direct ConnectX-7 ring."
            ),
        },
        "topology": {
            "mtu": mtu,
            "link_speed_mbps": link_speed_mbps,
            "edges": edges,
        },
        "ranks": ranks,
        "preflight": {
            "ssh_timeout_seconds": 45,
            "required_free_ports": [],
        },
    }


def initialise_cluster(
    *,
    size: int,
    head_target: str,
    worker_targets: Sequence[str],
    name: str,
    fabric_supernet: str,
    output: str | Path,
    enroll: bool = True,
) -> ClusterConfig:
    if size not in SUPPORTED_RING_SIZES:
        raise BootstrapError(f"ring size must be one of {SUPPORTED_RING_SIZES}")
    if len(worker_targets) != size - 1:
        raise BootstrapError(
            f"ring size {size} requires exactly {size - 1} worker targets"
        )
    targets = [head_target, *worker_targets]
    for target in targets:
        parse_target(target)
    if enroll:
        public_key = ensure_local_key()
        authorize_local_key(public_key)
        trust_host(head_target)
        head_access = _run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                head_target,
                "true",
            ]
        )
        if head_access.returncode:
            raise BootstrapError(
                "head self-SSH did not work after local key authorization: "
                + head_access.stderr.strip()
            )
        for target in worker_targets:
            enroll_target(target, public_key)
    facts = [probe_local(0, head_target)]
    facts.extend(
        probe_remote(rank, target)
        for rank, target in enumerate(worker_targets, start=1)
    )
    document = build_cluster_document(
        facts, name=name, fabric_supernet=fabric_supernet
    )
    config = validate_cluster(document, source=str(output))
    write_cluster(config, output)
    return config


def render_rank_netplan(cluster: ClusterConfig, rank_id: int) -> str:
    """Render a fabric-only netplan; management is intentionally absent."""
    rank = cluster.rank(rank_id)
    document = {
        "network": {
            "version": 2,
            "renderer": "NetworkManager",
            "ethernets": {
                port.interface: {
                    "addresses": [port.cidr],
                    "dhcp4": False,
                    "dhcp6": False,
                    "mtu": cluster.topology.mtu,
                    "optional": True,
                }
                for port in rank.ring_ports
            },
        }
    }
    return yaml.safe_dump(document, sort_keys=False)


def _network_apply_script(cluster: ClusterConfig, rank_id: int) -> str:
    rank = cluster.rank(rank_id)
    target = "/etc/netplan/40-sparkring-fabric.yaml"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    backup = f"{target}.before-sparkring-{timestamp}"
    management_cidr_fragment = f" {rank.management.address}/"
    return "\n".join(
        [
            "set -eu",
            f"target={target}",
            f"backup={backup}",
            "had_previous=0",
            'if test -f "$target"; then cp -a "$target" "$backup"; had_previous=1; fi',
            'rollback() { if test "$had_previous" -eq 1; then '
            'cp -a "$backup" "$target"; else rm -f "$target"; fi; }',
            "install -m 600 /tmp/sparkring-fabric.yaml \"$target\"",
            "if ! netplan generate; then",
            "  rollback",
            "  exit 1",
            "fi",
            "if ! netplan apply; then",
            "  rollback",
            "  netplan generate",
            "  netplan apply || true",
            "  echo 'netplan apply failed; prior fabric netplan restored' >&2",
            "  exit 1",
            "fi",
            f"if ! ip -4 -o addr show dev {rank.management.interface} | "
            f"grep -F -q -- '{management_cidr_fragment}'; then",
            "  rollback",
            "  netplan generate",
            "  netplan apply",
            "  echo 'management invariant failed; fabric netplan rolled back' >&2",
            "  exit 91",
            "fi",
            "rm -f /tmp/sparkring-fabric.yaml",
        ]
    )


def apply_fabric_network(cluster: ClusterConfig) -> None:
    """Install fabric-only netplan on each rank with management rollback."""
    for rank in cluster.ranks:
        content = render_rank_netplan(cluster, rank.id)
        encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
        stage_command = (
            f"printf '%s' {encoded} | base64 -d > /tmp/sparkring-fabric.yaml && "
            "chmod 600 /tmp/sparkring-fabric.yaml"
        )
        script = _network_apply_script(cluster, rank.id)
        if rank.id == 0:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", delete=False, suffix=".yaml"
            ) as handle:
                handle.write(content)
                staged = Path(handle.name)
            try:
                copy = _run(
                    ["sudo", "install", "-m", "600", str(staged), "/tmp/sparkring-fabric.yaml"],
                    capture=False,
                )
                if copy.returncode:
                    raise BootstrapError("could not stage rank 0 fabric netplan")
                result = _run(["sudo", "sh", "-c", script], capture=False)
            finally:
                staged.unlink(missing_ok=True)
        else:
            stage = _run(
                ["ssh", "-o", "BatchMode=yes", rank.ssh_target, stage_command]
            )
            if stage.returncode:
                raise BootstrapError(
                    f"could not stage rank {rank.id} fabric netplan: "
                    f"{stage.stderr.strip()}"
                )
            encoded_script = base64.b64encode(script.encode("utf-8")).decode("ascii")
            remote_apply = (
                f"printf '%s' {encoded_script} | base64 -d > "
                "/tmp/sparkring-apply-fabric.sh && "
                "chmod 700 /tmp/sparkring-apply-fabric.sh && "
                "sudo /tmp/sparkring-apply-fabric.sh; "
                "rc=$?; rm -f /tmp/sparkring-apply-fabric.sh; exit $rc"
            )
            result = _run(
                ["ssh", "-t", rank.ssh_target, remote_apply], capture=False
            )
        if result.returncode:
            raise BootstrapError(
                f"rank {rank.id} fabric configuration failed; remaining ranks "
                "were not changed"
            )
        verify = probe_local(rank.id, rank.ssh_target) if rank.id == 0 else probe_remote(
            rank.id, rank.ssh_target
        )
        expected = {
            port.interface: port.address for port in cluster.rank(rank.id).ring_ports
        }
        if verify.management_address != rank.management.address:
            raise BootstrapError(
                f"rank {rank.id} management address changed after netplan apply"
            )
        # The detailed address and MTU verdict is intentionally left to Doctor.
        if set(expected) != set(verify.fabric_interfaces):
            raise BootstrapError(f"rank {rank.id} fabric interfaces changed unexpectedly")
