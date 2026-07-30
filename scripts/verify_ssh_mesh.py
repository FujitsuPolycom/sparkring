#!/usr/bin/env python3
"""Verify and optionally repair SparkRing's SSH management/fanout paths.

Read-only by default.  ``--fix`` may create an Ed25519 user key on a source
rank, append only its public key to the destination's ``authorized_keys``, and
enrol the destination host key for the address used by the selected scope.
Existing private keys never leave their source rank and passwords are never
accepted.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from sparkring_site import SiteConfig, SiteConfigError, load_site


SSH_KEY_RE = re.compile(
    r"^(ssh-ed25519|ecdsa-sha2-nistp(?:256|384|521)|ssh-rsa) "
    r"([A-Za-z0-9+/]+={0,3})(?:\s+.*)?$"
)


@dataclass(frozen=True)
class Hop:
    source_rank: int
    destination_rank: int
    destination_address: str
    edge: str
    kind: str = "direct"


@dataclass
class Result:
    kind: str
    source_rank: int | None
    destination_rank: int
    destination: str
    edge: str | None
    status: str
    detail: str
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return self.status == "ok"


def split_ssh_target(target: str) -> tuple[str, str]:
    user, separator, host = target.partition("@")
    if not separator or not user or not host:
        raise ValueError(f"invalid SSH target: {target!r}")
    return user, host


def fanout_hops(site: SiteConfig) -> list[Hop]:
    """Return the redundant four-hop rank0 fanout tree for a four-node ring."""
    master = site.serving.master_rank
    master_rank = site.rank(master)
    neighbours = sorted(master_rank.neighbour_ranks)
    remaining = sorted(
        rank.id for rank in site.ranks
        if rank.id != master and rank.id not in neighbours
    )
    if len(neighbours) != 2 or len(remaining) != 1:
        raise ValueError("fanout derivation requires one master, two neighbours, "
                         "and one opposite rank")
    opposite = remaining[0]
    pairs = [(master, neighbours[0]), (master, neighbours[1]),
             (neighbours[0], opposite), (neighbours[1], opposite)]
    return [_hop(site, source, destination) for source, destination in pairs]


def image_fanout_hops(site: SiteConfig) -> list[Hop]:
    """Return the exact three-hop tree used for image archive payloads.

    Rank 0 sends to both direct neighbours.  The lower-ID neighbour then
    relays to the one opposite rank.  Every payload hop stays on a configured
    direct-ring edge; management addresses are used only to orchestrate it.
    """
    master = 0
    neighbours = sorted(site.rank(master).neighbour_ranks)
    remaining = sorted(
        rank.id for rank in site.ranks
        if rank.id != master and rank.id not in neighbours
    )
    if len(neighbours) != 2 or len(remaining) != 1:
        raise ValueError(
            "image fanout derivation requires rank0, two neighbours, "
            "and one opposite rank"
        )
    opposite = remaining[0]
    relay = neighbours[0]
    pairs = [
        (master, neighbours[0]),
        (master, neighbours[1]),
        (relay, opposite),
    ]
    return [_hop(site, source, destination) for source, destination in pairs]


def all_adjacent_hops(site: SiteConfig) -> list[Hop]:
    pairs: list[tuple[int, int]] = []
    for edge in site.topology.edges:
        left, right = edge.endpoints
        pairs.extend(((left, right), (right, left)))
    return [_hop(site, source, destination) for source, destination in pairs]


def bootstrap_hops(site: SiteConfig) -> list[Hop]:
    """Return rank0 management hops required by ``bootstrap_nf3.py``.

    The public bootstrap runs natively on physical rank 0 and sends images and
    remote commands directly from there to every follower. It also invokes
    this verifier locally, so rank 0 must trust its own configured management
    identity. These hops therefore use every rank's management ``ssh_target``
    host, not direct-ring addresses used by the optional no-registry relay
    tree.
    """
    # bootstrap_nf3.py is intentionally executed on physical rank 0 and fans
    # from there, independently of the API/master-rank setting.
    master = 0
    hops: list[Hop] = []
    for rank in sorted(site.ranks, key=lambda item: item.id):
        _, host = split_ssh_target(rank.ssh_target)
        hops.append(Hop(master, rank.id, host, "management", "bootstrap"))
    return hops


def _hop(site: SiteConfig, source: int, destination: int) -> Hop:
    source_rank = site.rank(source)
    for port in source_rank.ring_ports:
        if port.peer_rank == destination and port.peer_address is not None:
            return Hop(source, destination, str(port.peer_address), port.edge)
    raise ValueError(f"rank{source} has no direct ring link to rank{destination}")


def classify_failure(stderr: str, returncode: int) -> tuple[str, str]:
    text = stderr.strip()
    lowered = text.lower()
    if "host key verification failed" in lowered or (
        "no ed25519 host key is known" in lowered
        or "remote host identification has changed" in lowered
    ):
        return "host_key", text or "destination host key is not trusted"
    if "permission denied" in lowered or "no supported authentication" in lowered:
        return "authorization", text or "public-key authorization failed"
    if "could not resolve hostname" in lowered:
        return "name_resolution", text
    if any(token in lowered for token in (
        "connection timed out", "operation timed out", "no route to host",
        "network is unreachable", "connection refused",
    )):
        return "unreachable", text
    return "ssh_error", text or f"ssh exited {returncode}"


class Ssh:
    def __init__(self, timeout: int):
        self.timeout = timeout

    def run(self, target: str, command: str) -> subprocess.CompletedProcess[str]:
        argv = [
            "ssh", "-o", "BatchMode=yes",
            "-o", f"ConnectTimeout={self.timeout}",
            "-o", "StrictHostKeyChecking=yes",
            target, command,
        ]
        try:
            return subprocess.run(
                argv, text=True, capture_output=True,
                timeout=self.timeout + 5, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return subprocess.CompletedProcess(argv, 255, "", str(exc))


def _probe_management(ssh: Ssh, rank_id: int, target: str) -> Result:
    completed = ssh.run(target, "true")
    if completed.returncode == 0:
        return Result("management", None, rank_id, target, None, "ok",
                      "BatchMode SSH succeeded")
    status, detail = classify_failure(completed.stderr, completed.returncode)
    return Result("management", None, rank_id, target, None, status, detail)


def _direct_target(site: SiteConfig, hop: Hop) -> str:
    user, _ = split_ssh_target(site.rank(hop.destination_rank).ssh_target)
    return f"{user}@{hop.destination_address}"


def _probe_hop(ssh: Ssh, site: SiteConfig, hop: Hop) -> Result:
    source = site.rank(hop.source_rank)
    destination = _direct_target(site, hop)
    nested = (
        f"ssh -o BatchMode=yes -o ConnectTimeout={ssh.timeout} "
        f"-o StrictHostKeyChecking=yes {shlex.quote(destination)} true"
    )
    completed = ssh.run(source.ssh_target, nested)
    if completed.returncode == 0:
        return Result(hop.kind, hop.source_rank, hop.destination_rank,
                      destination, hop.edge, "ok", "BatchMode SSH succeeded")
    status, detail = classify_failure(completed.stderr, completed.returncode)
    return Result(hop.kind, hop.source_rank, hop.destination_rank,
                  destination, hop.edge, status, detail)


def _validated_public_key(value: str, label: str) -> tuple[str, str]:
    line = value.strip()
    match = SSH_KEY_RE.fullmatch(line)
    if not match:
        raise RuntimeError(f"{label} returned an invalid SSH public key")
    return line, f"{match.group(1)} {match.group(2)}"


def _repair_hop(ssh: Ssh, site: SiteConfig, hop: Hop) -> tuple[bool, str]:
    source = site.rank(hop.source_rank)
    destination = site.rank(hop.destination_rank)

    source_key_command = (
        "set -eu; umask 077; mkdir -p \"$HOME/.ssh\"; "
        "if [ ! -f \"$HOME/.ssh/id_ed25519\" ]; then "
        "ssh-keygen -q -t ed25519 -N '' -f \"$HOME/.ssh/id_ed25519\"; fi; "
        "cat \"$HOME/.ssh/id_ed25519.pub\""
    )
    source_key_result = ssh.run(source.ssh_target, source_key_command)
    if source_key_result.returncode != 0:
        return False, "could not create/read source public key: " + (
            source_key_result.stderr.strip() or "remote command failed"
        )
    try:
        source_key, _ = _validated_public_key(
            source_key_result.stdout, "source rank"
        )
    except RuntimeError as exc:
        return False, str(exc)

    encoded_source_key = base64.b64encode(source_key.encode()).decode()
    install_key_command = (
        "set -eu; umask 077; mkdir -p \"$HOME/.ssh\"; "
        "touch \"$HOME/.ssh/authorized_keys\"; chmod 600 \"$HOME/.ssh/authorized_keys\"; "
        f"k=$(printf %s {shlex.quote(encoded_source_key)} | base64 -d); "
        "grep -qxF \"$k\" \"$HOME/.ssh/authorized_keys\" || "
        "printf '%s\\n' \"$k\" >> \"$HOME/.ssh/authorized_keys\""
    )
    install_result = ssh.run(destination.ssh_target, install_key_command)
    if install_result.returncode != 0:
        return False, "could not authorize source public key: " + (
            install_result.stderr.strip() or "remote command failed"
        )

    host_key_result = ssh.run(
        destination.ssh_target,
        "set -eu; "
        "for f in /etc/ssh/ssh_host_ed25519_key.pub "
        "/etc/ssh/ssh_host_ecdsa_key.pub /etc/ssh/ssh_host_rsa_key.pub; do "
        "if [ -r \"$f\" ]; then cat \"$f\"; exit 0; fi; done; exit 1",
    )
    if host_key_result.returncode != 0:
        return False, "could not read destination host public key"
    try:
        _, bare_host_key = _validated_public_key(
            host_key_result.stdout, "destination rank"
        )
    except RuntimeError as exc:
        return False, str(exc)

    known_host_line = f"{hop.destination_address} {bare_host_key}"
    encoded_host_key = base64.b64encode(known_host_line.encode()).decode()
    install_host_command = (
        "set -eu; umask 077; mkdir -p \"$HOME/.ssh\"; "
        "touch \"$HOME/.ssh/known_hosts\"; chmod 600 \"$HOME/.ssh/known_hosts\"; "
        f"k=$(printf %s {shlex.quote(encoded_host_key)} | base64 -d); "
        "ssh-keygen -R " + shlex.quote(hop.destination_address) +
        " -f \"$HOME/.ssh/known_hosts\" >/dev/null 2>&1 || true; "
        "printf '%s\\n' \"$k\" >> \"$HOME/.ssh/known_hosts\""
    )
    install_host_result = ssh.run(source.ssh_target, install_host_command)
    if install_host_result.returncode != 0:
        return False, "could not enrol authenticated destination host key: " + (
            install_host_result.stderr.strip() or "remote command failed"
        )
    return True, (
        "installed source public key and authenticated "
        f"{hop.edge} host key"
    )


def verify(site: SiteConfig, scope: str, fix: bool,
           ssh: Ssh | None = None) -> list[Result]:
    runner = ssh or Ssh(site.preflight.ssh_timeout_seconds)
    results = [
        _probe_management(runner, rank.id, rank.ssh_target)
        for rank in sorted(site.ranks, key=lambda item: item.id)
    ]
    management_ok = {result.destination_rank: result.ok for result in results}
    if scope == "bootstrap":
        hops = bootstrap_hops(site)
    elif scope == "image-fanout":
        hops = image_fanout_hops(site)
    elif scope == "fanout":
        hops = fanout_hops(site)
    elif scope == "all-adjacent":
        hops = all_adjacent_hops(site)
    else:
        raise ValueError(f"unsupported SSH verification scope: {scope!r}")
    for hop in hops:
        result = _probe_hop(runner, site, hop)
        if fix and not result.ok:
            if not (management_ok[hop.source_rank]
                    and management_ok[hop.destination_rank]):
                result.detail += "; repair refused: management SSH is not healthy"
            elif result.status not in {"host_key", "authorization"}:
                result.detail += "; repair refused: this is not a trust/auth failure"
            else:
                repaired, detail = _repair_hop(runner, site, hop)
                if repaired:
                    after = _probe_hop(runner, site, hop)
                    after.repaired = after.ok
                    after.detail = detail + "; " + after.detail
                    result = after
                else:
                    result.detail += "; repair failed: " + detail
        results.append(result)
    return results


def _format_result(result: Result) -> str:
    marker = "PASS" if result.ok else "FAIL"
    repaired = " repaired" if result.repaired else ""
    if result.kind == "management":
        path = f"controller -> rank{result.destination_rank}"
    elif result.kind == "bootstrap":
        path = (
            f"rank{result.source_rank} -> rank{result.destination_rank} "
            f"(management, {result.destination})"
        )
    else:
        path = (
            f"rank{result.source_rank} -> rank{result.destination_rank} "
            f"({result.edge}, {result.destination})"
        )
    return f"[{marker}] {path}: {result.status}{repaired} — {result.detail}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True, type=Path,
                        help="validated SparkRing site.yaml")
    parser.add_argument(
        "--scope",
        choices=("bootstrap", "image-fanout", "fanout", "all-adjacent"),
        default="bootstrap",
        help=(
            "verify rank0-to-all-follower management paths used by the public "
            "bootstrap, including rank0 self-trust (default), the exact "
            "three-hop image fanout tree, the redundant direct-ring relay "
            "tree, or all 8 "
            "direct-ring directions"
        ),
    )
    parser.add_argument("--fix", action="store_true",
                        help="repair selected SSH keys/trust, then reverify")
    parser.add_argument("--json", action="store_true",
                        help="emit machine-readable results")
    args = parser.parse_args(argv)
    try:
        site = load_site(args.site)
        results = verify(site, args.scope, args.fix)
    except (SiteConfigError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps({
            "schema": "sparkring.ssh-mesh.v1",
            "site": site.name,
            "scope": args.scope,
            "fix_requested": args.fix,
            "ready": all(result.ok for result in results),
            "results": [asdict(result) | {"ok": result.ok}
                        for result in results],
        }, indent=2))
    else:
        print(f"SparkRing SSH mesh: site={site.name} scope={args.scope}")
        for result in results:
            print(_format_result(result))
        passed = sum(result.ok for result in results)
        print(f"\n{passed}/{len(results)} paths ready")
    return 0 if all(result.ok for result in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
