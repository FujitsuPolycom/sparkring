#!/usr/bin/env python3
"""Operator command for bootstrapping and diagnosing SparkRing clusters."""

from __future__ import annotations

import argparse
import getpass
import sys
from pathlib import Path
from typing import Sequence

try:
    from . import ring_doctor
    from . import spark_doctor
    from .sparkring_bootstrap import (
        DEFAULT_FABRIC_SUPERNET,
        BootstrapError,
        apply_fabric_network,
        initialise_cluster,
        render_rank_netplan,
    )
    from .sparkring_cluster import ClusterConfigError, load_cluster
    from .sparkring_site import SUPPORTED_RING_SIZES
except ImportError:  # Direct execution
    import ring_doctor
    import spark_doctor
    from sparkring_bootstrap import (
        DEFAULT_FABRIC_SUPERNET,
        BootstrapError,
        apply_fabric_network,
        initialise_cluster,
        render_rank_netplan,
    )
    from sparkring_cluster import ClusterConfigError, load_cluster
    from sparkring_site import SUPPORTED_RING_SIZES

DEFAULT_CLUSTER_PATH = Path.home() / ".config" / "sparkring" / "cluster.yaml"


def _confirm(prompt: str, assume_yes: bool) -> None:
    if assume_yes:
        return
    if input(prompt + " Type yes: ").strip().lower() != "yes":
        raise BootstrapError("operation cancelled")


def _target(prompt: str, default_user: str | None = None) -> str:
    suffix = f" [{default_user}@IP]" if default_user else " [username@IP]"
    value = input(prompt + suffix + ": ").strip()
    if "@" not in value and default_user:
        value = f"{default_user}@{value}"
    return value


def _cluster_init(args: argparse.Namespace) -> int:
    default_user = getpass.getuser()
    head = args.head or _target("Head/rank-0 management address", default_user)
    workers = list(args.node or [])
    while len(workers) < args.size - 1:
        rank = len(workers) + 1
        workers.append(_target(f"Rank {rank} management address", default_user))
    if len(workers) != args.size - 1:
        raise BootstrapError(
            f"size {args.size} needs {args.size - 1} --node values"
        )
    output = Path(args.output).expanduser()
    print("\nSparkRing cluster initialization plan")
    print(f"  size             : {args.size}")
    print(f"  head             : {head}")
    for rank, target in enumerate(workers, start=1):
        print(f"  rank {rank:<10}: {target}")
    print(f"  fabric supernet  : {args.fabric_supernet}")
    print(f"  output           : {output}")
    print(
        "  SSH enrollment   : "
        + ("skipped" if args.skip_enroll else "password once per untrusted worker")
    )
    _confirm("Continue?", args.yes)
    cluster = initialise_cluster(
        size=args.size,
        head_target=head,
        worker_targets=workers,
        name=args.name,
        fabric_supernet=args.fabric_supernet,
        output=output,
        enroll=not args.skip_enroll,
    )
    print("\n".join(cluster.summary_lines()))
    print(f"\nWrote {output}")
    print("Next:")
    print(f"  sparkring doctor --cluster {output}")
    print(
        "The first Doctor run is read-only. On blank fabric interfaces it will "
        "report the addresses/routes that still need configuration."
    )
    return 0


def _cluster_show(args: argparse.Namespace) -> int:
    cluster = load_cluster(Path(args.cluster).expanduser())
    print("\n".join(cluster.summary_lines()))
    return 0


def _cluster_configure(args: argparse.Namespace) -> int:
    path = Path(args.cluster).expanduser()
    cluster = load_cluster(path)
    print("SparkRing fabric network plan")
    print("Management interfaces are absent from every generated netplan.")
    for rank in cluster.ranks:
        print(f"\n--- rank {rank.id} ({rank.ssh_target}) ---")
        print(render_rank_netplan(cluster, rank.id).rstrip())
    if not args.apply:
        print("\nPlan only; no node was changed. Add --apply after review.")
        return 0
    _confirm(
        "Install these fabric-only netplans with per-rank management rollback?",
        args.yes,
    )
    apply_fabric_network(cluster)
    print("\nFabric addresses installed. Next run the read-only diagnosis:")
    print(f"  sparkring doctor --cluster {path} --verify")
    return 0


def _doctor(args: argparse.Namespace, remainder: Sequence[str]) -> int:
    cluster = str(Path(args.cluster).expanduser())
    doctor_args = ["--cluster", cluster, *remainder]
    return ring_doctor.main(doctor_args)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sparkring")
    subcommands = parser.add_subparsers(dest="command", required=True)

    cluster = subcommands.add_parser("cluster", help="cluster bootstrap and inventory")
    cluster_commands = cluster.add_subparsers(dest="cluster_command", required=True)
    init = cluster_commands.add_parser("init", help="bootstrap a blank 4/6-Spark ring")
    init.add_argument("--size", type=int, choices=SUPPORTED_RING_SIZES, required=True)
    init.add_argument("--head", metavar="USER@IP")
    init.add_argument("--node", action="append", metavar="USER@IP")
    init.add_argument("--name", default="sparkring")
    init.add_argument("--fabric-supernet", default=DEFAULT_FABRIC_SUPERNET)
    init.add_argument("--output", default=str(DEFAULT_CLUSTER_PATH))
    init.add_argument(
        "--skip-enroll",
        action="store_true",
        help="require SSH keys to be configured already",
    )
    init.add_argument("--yes", action="store_true", help="accept the printed plan")

    show = cluster_commands.add_parser("show", help="validate and show an inventory")
    show.add_argument("--cluster", default=str(DEFAULT_CLUSTER_PATH))

    configure = cluster_commands.add_parser(
        "configure", help="plan or apply fabric-only netplan"
    )
    configure.add_argument("--cluster", default=str(DEFAULT_CLUSTER_PATH))
    configure.add_argument("--apply", action="store_true")
    configure.add_argument("--yes", action="store_true", help="accept the printed plan")

    doctor = subcommands.add_parser("doctor", help="run Ring Doctor")
    doctor.add_argument("--cluster", default=str(DEFAULT_CLUSTER_PATH))

    host = subcommands.add_parser("host", help="diagnose one blank DGX Spark")
    host_commands = host.add_subparsers(dest="host_command", required=True)
    host_check = host_commands.add_parser("check", help="run read-only host checks")
    host_check.add_argument("--json", action="store_true")
    host_check.add_argument("--require-telemetry-disabled", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw = list(argv if argv is not None else sys.argv[1:])
    parser = _parser()
    if raw and raw[0] == "doctor":
        known, remainder = parser.parse_known_args(raw)
    else:
        known = parser.parse_args(raw)
        remainder = []
    try:
        if known.command == "cluster" and known.cluster_command == "init":
            return _cluster_init(known)
        if known.command == "cluster" and known.cluster_command == "show":
            return _cluster_show(known)
        if known.command == "cluster" and known.cluster_command == "configure":
            return _cluster_configure(known)
        if known.command == "doctor":
            return _doctor(known, remainder)
        if known.command == "host" and known.host_command == "check":
            host_args = []
            if known.json:
                host_args.append("--json")
            if known.require_telemetry_disabled:
                host_args.append("--require-telemetry-disabled")
            return spark_doctor.main(host_args)
    except (BootstrapError, ClusterConfigError) as exc:
        print(f"SPARKRING ERROR: {exc}", file=sys.stderr)
        return 2
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
