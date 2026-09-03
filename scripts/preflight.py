#!/usr/bin/env python3
"""SparkRing public preflight: a read-only, fail-closed cluster checker.

Everything this tool knows about your cluster comes from the site
configuration (``scripts/config/site.yaml``, see
``exl3-r7-site.example.yaml``). It
opens one ssh session per rank, runs a single read-only probe script, and
turns the result into a table of pass/fail checks plus a machine-readable
evidence file.

Why it is worth running before every launch
-------------------------------------------
The expensive failures on a small multi-node ring are almost never dramatic.
They are a NIC that quietly stopped holding its address after a reboot, a
cable that trained down to half speed, a GID table that shifted after a driver
upgrade, one rank still running last week's binary, or a jumbo path that
silently fragments.  Each of those surfaces hours later as an opaque hang.
Every check below turns one of those into an immediate, attributable failure
with a stable check id.

Read-only by construction
-------------------------
This tool never mutates remote state.  It does not start, stop, pull or remove
containers, does not change interfaces or routes, and writes nothing on the
nodes.  Every command is passed through :func:`assert_read_only` before it can
reach ssh, so a future edit that introduces a mutating command fails loudly in
this process instead of quietly on the cluster.  The only thing written
anywhere is the local evidence JSON.

Usage::

    python scripts/preflight.py --site scripts/config/site.yaml
    python scripts/preflight.py --site scripts/config/site.yaml --scope fabric
    python scripts/preflight.py --site scripts/config/site.yaml --json out.json
    python scripts/preflight.py --site scripts/config/site.yaml --print-plan
    python scripts/preflight.py --list-checks

Exit status is 0 only when every check passes on every rank.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sparkring_site import (  # noqa: E402  (local import after sys.path fix)
    Rank,
    RingPort,
    SiteConfig,
    SiteConfigError,
    ipv4_mapped_gid,
    load_site,
)
from sparkring_cluster import ClusterConfig  # noqa: E402
from sparkring_diagnostics import (  # noqa: E402
    CheckStatus,
    DiagnosticCheck,
    build_receipt,
)

EVIDENCE_SCHEMA = "sparkring-preflight/v1"
PROBE_SENTINEL = "SPARKRING-PREFLIGHT-READY"
PREFLIGHT_SCOPES = ("full", "fabric")
FabricConfiguration = SiteConfig | ClusterConfig

# --------------------------------------------------------------------------
# Stable check ids.  These are contract: they appear in evidence JSON and are
# safe to alert on.  Add new ids, never renumber or repurpose existing ones.
# --------------------------------------------------------------------------
CHECK_DESCRIPTIONS: dict[str, str] = {
    "SSH.REACHABLE":
        "rank answers over ssh with BatchMode (key auth, no prompt)",
    "MGMT.ADDRESS_PRESENT":
        "the configured management address is held by some interface",
    "MGMT.INTERFACE_MATCH":
        "it is held by exactly the configured management interface",
    "RING.LINK_UP":
        "ring interface exists and its operstate is up",
    "RING.MTU":
        "ring interface MTU equals topology.mtu",
    "RING.LINK_SPEED":
        "ring interface negotiated topology.link_speed_mbps",
    "RING.ADDRESS":
        "ring interface holds exactly the configured address, nothing else",
    "RING.RDMA_PORT_ACTIVE":
        "backing RDMA device exists and its port state is ACTIVE",
    "RING.RDMA_LINK_LAYER":
        "that RDMA port's link layer is Ethernet (RoCE, not InfiniBand)",
    "RING.ROCE_GID":
        "GID at the configured index is the RoCEv2 IPv4-mapped ring address",
    "RING.JUMBO_PING":
        "don't-fragment ping fills the MTU across the edge to the far end",
    "PEER.CONTROL_CHANNEL":
        "control-channel rendezvous peer answers ICMP from this rank",
    "ARTIFACT.PRESENT":
        "pinned artifact exists on this rank",
    "ARTIFACT.SHA256":
        "pinned artifact hashes to the configured sha256",
    "ARTIFACT.EXECUTABLE":
        "pinned artifact marked executable carries the +x bit",
    "PORT.FREE":
        "a port required to be free has no listener",
    "DISK.PATH_PRESENT":
        "cache/JIT directory exists on this rank",
    "DISK.FREE":
        "cache/JIT directory has at least the configured free space",
    "HOST.MEMORY_AVAILABLE":
        "the rank has the configured available RAM before model launch",
    "HOST.MEMORY_CONTIGUITY":
        "the Normal memory zone has enough configured high-order free blocks",
    "IMAGE.PRESENT":
        "the configured container image exists in the local image store",
    "IMAGE.DIGEST":
        "that image matches the configured digest (image ID or RepoDigest)",
}
CHECK_IDS: tuple[str, ...] = tuple(CHECK_DESCRIPTIONS)


class ReadOnlyViolation(RuntimeError):
    """A command that could mutate remote state was about to be executed."""


# --------------------------------------------------------------------------
# Read-only guard
# --------------------------------------------------------------------------

_ALLOWED_REDIRECTIONS = (
    "2>/dev/null",
    "1>/dev/null",
    "&>/dev/null",
    ">/dev/null",
    "2>&1",
)

_MUTATING_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"\brm\b", "rm"),
    (r"\brmdir\b", "rmdir"),
    (r"\bmkdir\b", "mkdir"),
    (r"\bmv\b", "mv"),
    (r"\bcp\b", "cp"),
    (r"\bdd\b", "dd"),
    (r"\btee\b", "tee"),
    (r"\btouch\b", "touch"),
    (r"\btruncate\b", "truncate"),
    (r"\bchmod\b", "chmod"),
    (r"\bchown\b", "chown"),
    (r"\bln\b", "ln"),
    (r"\bmkfs\b", "mkfs"),
    (r"\bmount\b", "mount"),
    (r"\bumount\b", "umount"),
    (r"\bsed\b[^|\n]*\s-i\b", "sed -i"),
    (r"\bsystemctl\b", "systemctl"),
    (r"\bservice\b", "service"),
    (r"\bmodprobe\b", "modprobe"),
    (r"\brmmod\b", "rmmod"),
    (r"\binsmod\b", "insmod"),
    (r"\bsysctl\b\s+-w\b", "sysctl -w"),
    (r"\bip\s+link\s+set\b", "ip link set"),
    (r"\bip\s+(addr|address)\s+(add|del|flush|change)\b", "ip addr mutation"),
    (r"\bip\s+route\s+(add|del|change|replace|flush)\b", "ip route mutation"),
    (r"\bethtool\s+-[sSGKCLN]\b", "ethtool set"),
    (r"\bkill\b", "kill"),
    (r"\bkillall\b", "killall"),
    (r"\bpkill\b", "pkill"),
    (r"\breboot\b|\bshutdown\b|\bhalt\b|\bpoweroff\b", "reboot/shutdown"),
    (r"\b(apt|apt-get|yum|dnf|zypper|pacman)\b", "package manager"),
    (r"\bpip\b|\bpip3\b", "pip"),
    (r"\bnvidia-smi\b[^|\n]*\s-(pl|lgc|ac|rgc|rac|pm)\b", "nvidia-smi set"),
    (r"/dev/tcp/", "/dev/tcp socket write"),
    (r"\bcrontab\b", "crontab"),
    (r"\bnmcli\b", "nmcli"),
    (r"\bnetplan\b", "netplan"),
    (
        r"\bdocker\s+(?!(?:image\s+inspect|image\s+ls|inspect|images|ps|"
        r"version|info)\b)",
        "docker subcommand that may mutate state",
    ),
)

_COMPILED_MUTATING = tuple(
    (re.compile(pattern), label) for pattern, label in _MUTATING_PATTERNS
)


def assert_read_only(command: str) -> None:
    """Raise :class:`ReadOnlyViolation` unless ``command`` is read-only.

    This is a guard rail, not a sandbox: it exists so that a well-meaning edit
    to :func:`build_probe_script` cannot start mutating production nodes
    without someone noticing.  Redirection to anywhere other than /dev/null is
    treated as a write.
    """
    for pattern, label in _COMPILED_MUTATING:
        match = pattern.search(command)
        if match:
            raise ReadOnlyViolation(
                f"refusing to run a command containing {label!r}: "
                f"...{command[max(0, match.start() - 40):match.end() + 40]}..."
            )
    stripped = command
    for allowed in _ALLOWED_REDIRECTIONS:
        stripped = stripped.replace(allowed, " ")
    if ">" in stripped:
        index = stripped.index(">")
        raise ReadOnlyViolation(
            "refusing to run a command that redirects to a file: "
            f"...{stripped[max(0, index - 40):index + 40]}..."
        )


# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    rank: int
    subject: str
    passed: bool
    detail: str

    def to_dict(self) -> dict:
        return {
            "check_id": self.check_id,
            "rank": self.rank,
            "subject": self.subject,
            "passed": self.passed,
            "detail": self.detail,
        }

    def to_diagnostic_check(self) -> DiagnosticCheck:
        """Represent a preflight result in the shared diagnostic receipt."""
        return DiagnosticCheck(
            check_id=self.check_id,
            status=CheckStatus.PASS if self.passed else CheckStatus.FAIL,
            scope=("fabric" if self.check_id.startswith("RING.") else "host"),
            subject=self.subject,
            summary=CHECK_DESCRIPTIONS[self.check_id],
            evidence=self.detail,
            node=f"rank{self.rank}",
            source="preflight",
        )


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    stdout: str = ""
    stderr: str = ""
    detail: str = ""


class Runner(Protocol):
    """Anything that can execute a read-only command on a rank."""

    def run(self, target: str, command: str) -> CommandResult:
        ...


class SshRunner:
    """Default runner: one ssh invocation per rank, key auth, no prompts."""

    def __init__(self, timeout_seconds: int, connect_timeout: int = 8) -> None:
        self.timeout_seconds = timeout_seconds
        self.connect_timeout = connect_timeout

    def run(self, target: str, command: str) -> CommandResult:
        assert_read_only(command)
        argv = [
            "ssh",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "BatchMode=yes",
            target,
            command,
        ]
        try:
            completed = subprocess.run(
                argv,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(
                ok=False,
                detail=f"ssh timed out after {self.timeout_seconds}s",
            )
        except OSError as exc:
            return CommandResult(ok=False, detail=f"ssh could not run: {exc}")
        return CommandResult(
            ok=completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            detail=(
                "" if completed.returncode == 0
                else f"ssh exited {completed.returncode}: "
                f"{completed.stderr.strip()[:200]}"
            ),
        )


# --------------------------------------------------------------------------
# Probe script construction
# --------------------------------------------------------------------------


def _shell_quote(value: str) -> str:
    """Single-quote a value already restricted by the site validator.

    ``sparkring_site`` rejects quotes, whitespace and shell metacharacters in
    every field interpolated here, so plain single quoting is sufficient and
    the assertion documents the dependency.
    """
    if "'" in value or any(character.isspace() for character in value):
        raise ReadOnlyViolation(
            f"value {value!r} is not safe to interpolate into a shell command"
        )
    return f"'{value}'"


def _validate_scope(scope: str) -> None:
    if scope not in PREFLIGHT_SCOPES:
        raise ValueError(
            f"unsupported preflight scope {scope!r}; "
            f"expected one of {PREFLIGHT_SCOPES}"
        )


def _validate_configuration_scope(
    configuration: FabricConfiguration, scope: str
) -> None:
    _validate_scope(scope)
    if isinstance(configuration, ClusterConfig) and scope != "fabric":
        raise ValueError(
            "a cluster inventory supports fabric preflight only; use a "
            "deployment site for runtime, image, disk, and port checks"
        )


def build_probe_script(
    site: FabricConfiguration,
    rank: Rank,
    scope: str = "full",
) -> str:
    """Build the single read-only shell script executed on ``rank``.

    Output is a stream of ``KEY field field ...`` records, parsed by
    :func:`parse_probe_output`.  Every value is guarded so that a missing sysfs
    file becomes the literal ``-`` rather than a shifted field.
    """
    _validate_configuration_scope(site, scope)
    lines: list[str] = [
        "set -u",
        # v <path>: print the file's contents, or '-' if absent/empty.
        "v() { _val=$(cat \"$1\" 2>/dev/null); "
        "[ -n \"$_val\" ] || _val='-'; printf '%s' \"$_val\"; }",
        f'echo "{PROBE_SENTINEL}"',
    ]

    for port in rank.ring_ports:
        interface = port.interface
        lines.append(
            f'echo "IF {interface} '
            f"$(v /sys/class/net/{interface}/operstate) "
            f"$(v /sys/class/net/{interface}/mtu) "
            f'$(v /sys/class/net/{interface}/speed)"'
        )

    # Full IPv4 address table: used both for RING.ADDRESS (exact match) and for
    # MGMT.* (which interface really holds the management address).
    lines.append("ip -o -4 addr 2>/dev/null | sed 's/^/IPROW /'")

    for port in rank.ring_ports:
        base = f"/sys/class/infiniband/{port.rdma_device}/ports/{port.rdma_port}"
        key = port.rdma_key
        gid_key = f"{key}:{port.roce_gid_index}"
        lines.append(f'echo "RDMA_STATE {key} $(v {base}/state)"')
        lines.append(f'echo "RDMA_LINK {key} $(v {base}/link_layer)"')
        lines.append(
            f'echo "GID {gid_key} $(v {base}/gids/{port.roce_gid_index})"'
        )
        lines.append(
            f'echo "GID_TYPE {gid_key} '
            f'$(v {base}/gid_attrs/types/{port.roce_gid_index})"'
        )

    payload = site.topology.jumbo_payload_bytes
    for port in rank.ring_ports:
        lines.append(
            f"if ping -c 2 -W 2 -M do -s {payload} {port.peer_address} "
            f'>/dev/null 2>&1; then echo "PING {port.edge} ok"; '
            f'else echo "PING {port.edge} fail"; fi'
        )

    for index, peer in enumerate(rank.transport_peers):
        lines.append(
            f"if ping -c 1 -W 2 {peer.address} >/dev/null 2>&1; "
            f'then echo "PEER {index} ok"; else echo "PEER {index} fail"; fi'
        )

    if scope == "full":
        for index, artifact in enumerate(site.artifacts):
            quoted = _shell_quote(artifact.path)
            lines.append(
                f"_h=$(sha256sum -- {quoted} 2>/dev/null | cut -d' ' -f1); "
                "[ -n \"$_h\" ] || _h='-'"
            )
            lines.append(f"_p=absent; [ -f {quoted} ] && _p=present")
            lines.append(f"_x=noexec; [ -x {quoted} ] && _x=exec")
            lines.append(f'echo "ART {index} $_h $_p $_x"')

        if site.preflight.required_free_ports:
            lines.append("ss -ltnH 2>/dev/null | sed 's/^/SSROW /'")

        for index, (_label, path, _minimum) in enumerate(
            site.paths.remote_space_targets()
        ):
            quoted = _shell_quote(path)
            lines.append(f"_d=no; [ -d {quoted} ] && _d=yes")
            lines.append(f'echo "DIR {index} $_d"')
            lines.append(f'echo "DFPATH {index}"')
            lines.append(
                f"df -Pk {quoted} 2>/dev/null | sed 's/^/DFROW /'"
            )

        if site.preflight.memory is not None:
            lines.extend([
                'echo "MEM_PAGE_SIZE $(getconf PAGESIZE 2>/dev/null || '
                "printf '-')\"",
                "awk '$1 == \"MemAvailable:\" "
                "{print \"MEM_AVAILABLE_KIB\", $2}' "
                "/proc/meminfo 2>/dev/null",
                "awk '{printf \"BUDDY %s\", $4; "
                "for (i = 5; i <= NF; i++) printf \" %s\", $i; "
                "printf \"\\n\"}' /proc/buddyinfo 2>/dev/null",
                "awk '$1 == \"compact_stall\" || "
                "$1 == \"compact_fail\" || $1 == \"compact_success\" "
                "{print \"VMSTAT\", $1, $2}' /proc/vmstat 2>/dev/null",
            ])

        image = _shell_quote(site.runtime.container_image)
        lines.append(
            f"_iid=$(docker image inspect {image} "
            "--format '{{.Id}}' 2>/dev/null); [ -n \"$_iid\" ] || _iid='-'"
        )
        lines.append('echo "IMAGE_ID $_iid"')
        lines.append(
            f"_rd=$(docker image inspect {image} "
            "--format '{{join .RepoDigests \",\"}}' 2>/dev/null); "
            "[ -n \"$_rd\" ] || _rd='-'"
        )
        lines.append('echo "IMAGE_REPODIGESTS $_rd"')

    script = "\n".join(lines) + "\n"
    assert_read_only(script)
    return script


# --------------------------------------------------------------------------
# Probe output parsing
# --------------------------------------------------------------------------


@dataclass
class ProbeState:
    ready: bool = False
    interfaces: dict[str, dict[str, str]] = field(default_factory=dict)
    addresses: dict[str, set[str]] = field(default_factory=dict)
    rdma_state: dict[str, str] = field(default_factory=dict)
    rdma_link: dict[str, str] = field(default_factory=dict)
    gids: dict[str, str] = field(default_factory=dict)
    gid_types: dict[str, str] = field(default_factory=dict)
    pings: dict[str, str] = field(default_factory=dict)
    peers: dict[int, str] = field(default_factory=dict)
    artifacts: dict[int, dict[str, str]] = field(default_factory=dict)
    listening_ports: set[int] = field(default_factory=set)
    directories: dict[int, str] = field(default_factory=dict)
    available_kib: dict[int, int] = field(default_factory=dict)
    memory_page_size: int | None = None
    memory_available_kib: int | None = None
    buddy_orders: dict[str, list[int]] = field(default_factory=dict)
    vmstat: dict[str, int] = field(default_factory=dict)
    image_id: str = "-"
    image_repo_digests: tuple[str, ...] = ()

    def interfaces_holding(self, address: str) -> list[str]:
        prefix = f"{address}/"
        return sorted(
            name for name, cidrs in self.addresses.items()
            if any(cidr.startswith(prefix) for cidr in cidrs)
        )


def _port_of(socket_address: str) -> int | None:
    if ":" not in socket_address:
        return None
    tail = socket_address.rsplit(":", 1)[1]
    try:
        return int(tail)
    except ValueError:
        return None


def parse_probe_output(text: str) -> ProbeState:
    """Turn the probe script's record stream into a :class:`ProbeState`."""
    state = ProbeState()
    current_df: int | None = None
    for raw_line in text.splitlines():
        tokens = raw_line.split()
        if not tokens:
            continue
        key = tokens[0]
        if key == PROBE_SENTINEL:
            state.ready = True
        elif key == "IF" and len(tokens) >= 5:
            state.interfaces[tokens[1]] = {
                "operstate": tokens[2],
                "mtu": tokens[3],
                "speed": tokens[4],
            }
        elif key == "IPROW" and len(tokens) >= 5:
            interface = tokens[2].rstrip(":")
            if "inet" in tokens:
                position = tokens.index("inet")
                if position + 1 < len(tokens):
                    state.addresses.setdefault(interface, set()).add(
                        tokens[position + 1]
                    )
        elif key == "RDMA_STATE" and len(tokens) >= 3:
            state.rdma_state[tokens[1]] = " ".join(tokens[2:])
        elif key == "RDMA_LINK" and len(tokens) >= 3:
            state.rdma_link[tokens[1]] = " ".join(tokens[2:])
        elif key == "GID" and len(tokens) >= 3:
            state.gids[tokens[1]] = tokens[2]
        elif key == "GID_TYPE" and len(tokens) >= 3:
            state.gid_types[tokens[1]] = " ".join(tokens[2:])
        elif key == "PING" and len(tokens) >= 3:
            state.pings[tokens[1]] = tokens[2]
        elif key == "PEER" and len(tokens) >= 3:
            state.peers[int(tokens[1])] = tokens[2]
        elif key == "ART" and len(tokens) >= 5:
            state.artifacts[int(tokens[1])] = {
                "sha256": tokens[2],
                "presence": tokens[3],
                "mode": tokens[4],
            }
        elif key == "SSROW" and len(tokens) >= 5:
            port = _port_of(tokens[4])
            if port is not None:
                state.listening_ports.add(port)
        elif key == "DIR" and len(tokens) >= 3:
            state.directories[int(tokens[1])] = tokens[2]
        elif key == "DFPATH" and len(tokens) >= 2:
            current_df = int(tokens[1])
        elif key == "DFROW" and current_df is not None:
            fields = tokens[1:]
            if fields and fields[0] == "Filesystem":
                continue
            for index in range(len(fields) - 2):
                triple = fields[index:index + 3]
                if all(item.isdigit() for item in triple):
                    state.available_kib[current_df] = int(triple[2])
                    break
        elif key == "MEM_PAGE_SIZE" and len(tokens) >= 2:
            if tokens[1].isdigit():
                state.memory_page_size = int(tokens[1])
        elif key == "MEM_AVAILABLE_KIB" and len(tokens) >= 2:
            if tokens[1].isdigit():
                state.memory_available_kib = int(tokens[1])
        elif key == "BUDDY" and len(tokens) >= 3:
            counts = tokens[2:]
            if all(item.isdigit() for item in counts):
                zone = tokens[1]
                observed = [int(item) for item in counts]
                aggregate = state.buddy_orders.setdefault(zone, [])
                if len(aggregate) < len(observed):
                    aggregate.extend([0] * (len(observed) - len(aggregate)))
                for index, count in enumerate(observed):
                    aggregate[index] += count
        elif key == "VMSTAT" and len(tokens) >= 3:
            if tokens[2].isdigit():
                state.vmstat[tokens[1]] = int(tokens[2])
        elif key == "IMAGE_ID" and len(tokens) >= 2:
            state.image_id = tokens[1]
        elif key == "IMAGE_REPODIGESTS" and len(tokens) >= 2:
            raw = tokens[1]
            state.image_repo_digests = (
                () if raw == "-"
                else tuple(item for item in raw.split(",") if item)
            )
    return state


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------


def _normalise_digest(value: str) -> str:
    """``repo/name@sha256:abcd`` -> ``sha256:abcd``; otherwise unchanged."""
    return value.rsplit("@", 1)[-1].strip()


def _equivalent_buddy_blocks(
    state: ProbeState, block_bytes: int, zone: str = "Normal"
) -> tuple[int, int] | None:
    """Return ``(target_order, equivalent_blocks)`` for one buddy zone.

    Larger free blocks are expressed as the number of target-sized blocks they
    can provide. The calculation uses the rank's reported base page size, so it
    remains correct for both 4 KiB and 64 KiB kernels.
    """
    page_size = state.memory_page_size
    counts = state.buddy_orders.get(zone)
    if page_size is None or counts is None or page_size <= 0:
        return None
    if block_bytes % page_size:
        return None
    pages = block_bytes // page_size
    if pages <= 0 or pages & (pages - 1):
        return None
    target_order = pages.bit_length() - 1
    if target_order >= len(counts):
        return target_order, 0
    equivalent = sum(
        count << (order - target_order)
        for order, count in enumerate(counts[target_order:], target_order)
    )
    return target_order, equivalent


def evaluate_rank(
    site: FabricConfiguration,
    rank: Rank,
    state: ProbeState,
    scope: str = "full",
) -> list[CheckResult]:
    """Turn one rank's probe state into check results.  Pure: no I/O."""
    _validate_configuration_scope(site, scope)
    results: list[CheckResult] = []

    def record(check_id: str, subject: str, passed: bool, detail: str) -> None:
        results.append(
            CheckResult(
                check_id=check_id, rank=rank.id, subject=subject,
                passed=passed, detail=detail,
            )
        )

    record(
        "SSH.REACHABLE", rank.ssh_target, state.ready,
        "probe ran and answered" if state.ready
        else "probe produced no readiness marker",
    )

    mgmt_address = str(rank.management.address)
    holders = state.interfaces_holding(mgmt_address)
    record(
        "MGMT.ADDRESS_PRESENT", mgmt_address, bool(holders),
        f"held by {', '.join(holders)}" if holders
        else "NOT held by any interface on this rank",
    )
    record(
        "MGMT.INTERFACE_MATCH", mgmt_address,
        holders == [rank.management.interface],
        f"observed on {holders or ['none']}, configured "
        f"{rank.management.interface!r}",
    )

    for port in rank.ring_ports:
        results.extend(_evaluate_ring_port(site, rank, port, state))

    for index, peer in enumerate(rank.transport_peers):
        outcome = state.peers.get(index, "missing")
        record(
            "PEER.CONTROL_CHANNEL", f"rank{peer.rank}@{peer.address}",
            outcome == "ok",
            f"icmp reachability {outcome} (control-channel rendezvous for "
            f"the native TP/vocab/DCP sessions)",
        )

    if scope == "fabric":
        return results

    memory = site.preflight.memory
    if memory is not None:
        available = (
            None if state.memory_available_kib is None
            else state.memory_available_kib * 1024
        )
        record(
            "HOST.MEMORY_AVAILABLE",
            "host RAM before model launch",
            available is not None
            and available >= memory.minimum_available_bytes,
            (
                "MemAvailable unavailable; "
                f"want >= {memory.minimum_available_bytes} bytes"
                if available is None else
                f"MemAvailable={available} bytes "
                f"({available / (1 << 30):.1f} GiB), want >= "
                f"{memory.minimum_available_bytes} bytes "
                f"({memory.minimum_available_bytes / (1 << 30):.1f} GiB)"
            ),
        )
        contiguous = _equivalent_buddy_blocks(
            state, memory.contiguous_block_bytes
        )
        compact_stall = state.vmstat.get("compact_stall", "unavailable")
        compact_fail = state.vmstat.get("compact_fail", "unavailable")
        compact_success = state.vmstat.get("compact_success", "unavailable")
        if contiguous is None:
            detail = (
                "Normal-zone buddy information or page size unavailable; "
                f"compact_stall={compact_stall} compact_fail={compact_fail} "
                f"compact_success={compact_success}"
            )
            contiguous_passed = False
        else:
            target_order, equivalent = contiguous
            block_mib = memory.contiguous_block_bytes // (1 << 20)
            detail = (
                f"page_size={state.memory_page_size} target_order={target_order} "
                f"equivalent_{block_mib}MiB_blocks={equivalent}, want >= "
                f"{memory.minimum_contiguous_blocks}; "
                f"compact_stall={compact_stall} compact_fail={compact_fail} "
                f"compact_success={compact_success}"
            )
            contiguous_passed = equivalent >= memory.minimum_contiguous_blocks
        record(
            "HOST.MEMORY_CONTIGUITY",
            "Normal-zone contiguous memory before model launch",
            contiguous_passed,
            detail,
        )

    for index, artifact in enumerate(site.artifacts):
        observed = state.artifacts.get(index, {})
        presence = observed.get("presence", "missing")
        record(
            "ARTIFACT.PRESENT", f"{artifact.name} ({artifact.path})",
            presence == "present", f"presence={presence}",
        )
        actual = observed.get("sha256", "-")
        record(
            "ARTIFACT.SHA256", f"{artifact.name} ({artifact.path})",
            actual == artifact.sha256,
            f"sha256={actual} (want {artifact.sha256})",
        )
        if artifact.executable:
            mode = observed.get("mode", "missing")
            record(
                "ARTIFACT.EXECUTABLE", f"{artifact.name} ({artifact.path})",
                mode == "exec", f"mode={mode} (want exec)",
            )

    for port_number in site.preflight.required_free_ports:
        free = port_number not in state.listening_ports
        record(
            "PORT.FREE", f"tcp/{port_number}", free,
            "no listener" if free
            else "a process is already listening on this port",
        )

    for index, (label, path, minimum) in enumerate(
        site.paths.remote_space_targets()
    ):
        exists = state.directories.get(index, "no") == "yes"
        record(
            "DISK.PATH_PRESENT", f"{label} ({path})", exists,
            "directory present" if exists else "directory missing",
        )
        available_kib = state.available_kib.get(index)
        if available_kib is None:
            record(
                "DISK.FREE", f"{label} ({path})", False,
                f"could not read free space (want >= {minimum} bytes)",
            )
        else:
            available = available_kib * 1024
            record(
                "DISK.FREE", f"{label} ({path})", available >= minimum,
                f"free={available} bytes ({available / (1 << 30):.1f} GiB), "
                f"want >= {minimum} bytes "
                f"({minimum / (1 << 30):.1f} GiB)",
            )

    image = site.runtime.container_image
    present = state.image_id not in ("-", "")
    record("IMAGE.PRESENT", image, present,
           f"image id {state.image_id}" if present
           else "image not found in the local store")
    wanted = site.runtime.container_image_digest
    observed_digests = {state.image_id} | {
        _normalise_digest(entry) for entry in state.image_repo_digests
    }
    record(
        "IMAGE.DIGEST", image, wanted in observed_digests,
        f"want {wanted}; observed id={state.image_id} repo_digests="
        f"{list(state.image_repo_digests) or 'none'}",
    )
    return results


def _evaluate_ring_port(site: FabricConfiguration, rank: Rank, port: RingPort,
                        state: ProbeState) -> list[CheckResult]:
    results: list[CheckResult] = []
    subject = f"{port.interface} [{port.edge}]"

    def record(check_id: str, item: str, passed: bool, detail: str) -> None:
        results.append(
            CheckResult(
                check_id=check_id, rank=rank.id, subject=item,
                passed=passed, detail=detail,
            )
        )

    fields = state.interfaces.get(port.interface, {})
    operstate = fields.get("operstate", "missing")
    record(
        "RING.LINK_UP", subject, operstate == "up",
        f"operstate={operstate}",
    )
    mtu = fields.get("mtu", "-")
    record(
        "RING.MTU", subject, mtu == str(site.topology.mtu),
        f"mtu={mtu} (want {site.topology.mtu})",
    )
    speed = fields.get("speed", "-")
    record(
        "RING.LINK_SPEED", subject,
        speed == str(site.topology.link_speed_mbps),
        f"speed={speed} Mb/s (want {site.topology.link_speed_mbps})",
    )
    observed = sorted(state.addresses.get(port.interface, set()))
    record(
        "RING.ADDRESS", subject, observed == [port.cidr],
        f"addresses={observed or ['none']} (want exactly [{port.cidr!r}])",
    )

    rdma_state = state.rdma_state.get(port.rdma_key, "missing")
    record(
        "RING.RDMA_PORT_ACTIVE", port.rdma_key, "ACTIVE" in rdma_state,
        f"port state={rdma_state}",
    )
    link_layer = state.rdma_link.get(port.rdma_key, "missing")
    record(
        "RING.RDMA_LINK_LAYER", port.rdma_key,
        link_layer.strip().lower() == "ethernet",
        f"link_layer={link_layer} (want Ethernet for RoCE)",
    )

    gid_key = f"{port.rdma_key}:{port.roce_gid_index}"
    wanted_gid = ipv4_mapped_gid(port.address)
    observed_gid = state.gids.get(gid_key, "-")
    observed_type = state.gid_types.get(gid_key, "-")
    type_ok = observed_type.replace(" ", "").lower() == "rocev2"
    record(
        "RING.ROCE_GID", f"{gid_key} (gid index {port.roce_gid_index})",
        observed_gid == wanted_gid and type_ok,
        f"gid={observed_gid} type={observed_type!r} "
        f"(want {wanted_gid} 'RoCE v2')",
    )

    outcome = state.pings.get(port.edge, "missing")
    record(
        "RING.JUMBO_PING",
        f"{port.interface} -> {port.peer_address} (rank{port.peer_rank})",
        outcome == "ok",
        f"ping -M do -s {site.topology.jumbo_payload_bytes} => {outcome}",
    )
    return results


def unreachable_results(rank: Rank, detail: str) -> list[CheckResult]:
    """The single result produced when a rank cannot be probed at all."""
    return [
        CheckResult(
            check_id="SSH.REACHABLE", rank=rank.id, subject=rank.ssh_target,
            passed=False, detail=detail or "rank unreachable over ssh",
        )
    ]


def check_rank(
    site: FabricConfiguration,
    rank: Rank,
    runner: Runner,
    scope: str = "full",
) -> list[CheckResult]:
    """Probe one rank and evaluate it.  Never mutates remote state."""
    _validate_configuration_scope(site, scope)
    script = build_probe_script(site, rank, scope=scope)
    outcome = runner.run(rank.ssh_target, script)
    state = parse_probe_output(outcome.stdout)
    if not state.ready:
        detail = outcome.detail or (
            "ssh succeeded but the probe produced no output"
            if outcome.ok else "rank unreachable over ssh"
        )
        return unreachable_results(rank, detail)
    return evaluate_rank(site, rank, state, scope=scope)


def run_preflight(site: FabricConfiguration, runner: Runner,
                  ranks: Sequence[int] | None = None,
                  max_workers: int = 4,
                  scope: str = "full") -> list[CheckResult]:
    """Probe every selected rank in parallel and return all check results."""
    _validate_configuration_scope(site, scope)
    selected = [
        rank for rank in site.ranks
        if ranks is None or rank.id in set(ranks)
    ]
    if not selected:
        return []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, min(max_workers, len(selected)))
    ) as pool:
        batches = list(
            pool.map(
                lambda rank: check_rank(site, rank, runner, scope=scope),
                selected,
            )
        )
    results: list[CheckResult] = []
    for batch in batches:
        results.extend(batch)
    return results


# --------------------------------------------------------------------------
# Aggregation, evidence and rendering
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Summary:
    total: int
    passed: int
    failed: int
    failed_check_ids: tuple[str, ...]
    failed_ranks: tuple[int, ...]

    @property
    def ok(self) -> bool:
        return self.failed == 0 and self.total > 0


def summarise(results: Sequence[CheckResult]) -> Summary:
    failures = [result for result in results if not result.passed]
    return Summary(
        total=len(results),
        passed=len(results) - len(failures),
        failed=len(failures),
        failed_check_ids=tuple(
            sorted({result.check_id for result in failures})
        ),
        failed_ranks=tuple(sorted({result.rank for result in failures})),
    )


def exit_code_for(results: Sequence[CheckResult]) -> int:
    """0 only when at least one check ran and every check passed."""
    return 0 if summarise(results).ok else 1


def build_evidence(site: SiteConfig, results: Sequence[CheckResult],
                   generated_at: str | None = None,
                   warnings: Sequence[str] | None = None,
                   scope: str = "full") -> dict:
    """The machine-readable evidence document written alongside the table."""
    _validate_scope(scope)
    summary = summarise(results)
    if warnings is None:
        warnings = site.placeholder_warnings()
    receipt_generated_at = generated_at or time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
    )
    by_rank: dict[int, list[CheckResult]] = {}
    for result in results:
        by_rank.setdefault(result.rank, []).append(result)
    return {
        "schema": EVIDENCE_SCHEMA,
        "generated_at": receipt_generated_at,
        "read_only": True,
        "scope": scope,
        "passed": summary.ok,
        "site": {
            "name": site.name,
            "schema_version": site.schema_version,
            "source": site.source,
        },
        "totals": {
            "checks": summary.total,
            "passed": summary.passed,
            "failed": summary.failed,
        },
        "failed_check_ids": list(summary.failed_check_ids),
        "failed_ranks": list(summary.failed_ranks),
        "placeholder_warnings": list(warnings),
        "known_check_ids": list(CHECK_IDS),
        "diagnostics": build_receipt(
            [result.to_diagnostic_check() for result in results],
            generated_at=receipt_generated_at,
            source="preflight",
        ),
        "ranks": [
            {
                "rank": rank.id,
                "ssh_target": rank.ssh_target,
                "checks": [
                    result.to_dict() for result in by_rank.get(rank.id, [])
                ],
            }
            for rank in site.ranks
            if rank.id in by_rank
        ],
    }


def render_text(site: SiteConfig, results: Sequence[CheckResult],
                warnings: Sequence[str] | None = None,
                scope: str = "full") -> str:
    """Human-readable pass/fail table."""
    _validate_scope(scope)
    if warnings is None:
        warnings = site.placeholder_warnings()
    by_rank: dict[int, list[CheckResult]] = {}
    for result in results:
        by_rank.setdefault(result.rank, []).append(result)

    lines = [
        f"SparkRing preflight (read-only, scope={scope}) - site {site.name!r} "
        f"from {site.source or '<in-memory>'}",
    ]
    for rank in site.ranks:
        entries = by_rank.get(rank.id)
        if entries is None:
            continue
        lines.append("")
        lines.append(f"-- rank {rank.id} ({rank.ssh_target}) --")
        for result in entries:
            mark = "PASS" if result.passed else "FAIL"
            lines.append(
                f"  [{mark}] {result.check_id} {result.subject}: "
                f"{result.detail}"
            )

    summary = summarise(results)
    lines.append("")
    if warnings:
        lines.append(
            f"placeholder warnings ({len(warnings)}) - this site config still "
            "contains example values:"
        )
        for warning in warnings:
            lines.append(f"  [WARN] {warning}")
        lines.append("")
    lines.append(
        f"checks: {summary.passed}/{summary.total} passed, "
        f"{summary.failed} failed"
    )
    if summary.failed:
        lines.append(
            "failing check ids: " + ", ".join(summary.failed_check_ids)
        )
        lines.append(
            "failing ranks: "
            + ", ".join(str(value) for value in summary.failed_ranks)
        )
    lines.append("preflight: " + ("PASS" if summary.ok else "FAIL"))
    return "\n".join(lines)


def default_evidence_path(site: SiteConfig,
                          timestamp: str | None = None) -> Path:
    stamp = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return Path(site.paths.evidence_dir) / f"preflight-{stamp}.json"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="preflight",
        description=(
            "Read-only SparkRing cluster preflight driven by a site "
            "configuration. Never mutates remote state."
        ),
    )
    parser.add_argument(
        "--site", default="scripts/config/site.yaml",
        help="path to the site YAML (default: scripts/config/site.yaml)",
    )
    parser.add_argument(
        "--json", default=None,
        help="write the evidence JSON here (default: "
             "<paths.evidence_dir>/preflight-<utc>.json)",
    )
    parser.add_argument(
        "--rank", action="append", type=int, default=None, metavar="N",
        help="only check this rank; repeatable",
    )
    parser.add_argument(
        "--scope",
        choices=PREFLIGHT_SCOPES,
        default="full",
        help=(
            "full deployment preflight (default), or early fabric-only "
            "SSH/management/ring/RDMA/peer checks that do not require a "
            "built image or installed artifacts"
        ),
    )
    parser.add_argument(
        "--print-plan", action="store_true",
        help="print the read-only probe script for each rank and exit "
             "without contacting anything",
    )
    parser.add_argument(
        "--list-checks", action="store_true",
        help="print every stable check id with a one-line description",
    )
    parser.add_argument(
        "--strict-placeholders", action="store_true",
        help="fail when the site config still contains example placeholders",
    )
    parser.add_argument(
        "--no-evidence", action="store_true",
        help="do not write an evidence file",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.list_checks:
        for check_id in CHECK_IDS:
            print(f"{check_id:<24} {CHECK_DESCRIPTIONS[check_id]}")
        return 0

    try:
        site = load_site(args.site)
    except SiteConfigError as exc:
        print(f"INVALID SITE CONFIG {args.site}", file=sys.stderr)
        print(f"  {exc}", file=sys.stderr)
        return 1

    if args.print_plan:
        for rank in site.ranks:
            if args.rank is not None and rank.id not in set(args.rank):
                continue
            print(f"# ---- rank {rank.id} ({rank.ssh_target}) ----")
            print(build_probe_script(site, rank, scope=args.scope))
        return 0

    runner = SshRunner(timeout_seconds=site.preflight.ssh_timeout_seconds)
    results = run_preflight(
        site,
        runner,
        ranks=args.rank,
        scope=args.scope,
    )
    warnings = site.placeholder_warnings()
    print(render_text(site, results, warnings, scope=args.scope))

    if not args.no_evidence:
        evidence = build_evidence(
            site,
            results,
            warnings=warnings,
            scope=args.scope,
        )
        target = Path(args.json) if args.json else default_evidence_path(site)
        try:
            if target.parent and str(target.parent):
                os.makedirs(target.parent, exist_ok=True)
            target.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"evidence={target}")
        except OSError as exc:
            print(f"WARNING: could not write evidence to {target}: {exc}",
                  file=sys.stderr)

    status = exit_code_for(results)
    if warnings and args.strict_placeholders:
        print(
            "FAIL: --strict-placeholders and the site config still contains "
            f"{len(warnings)} example placeholder(s)",
            file=sys.stderr,
        )
        status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
