#!/usr/bin/env python3
"""One-command lifecycle control for the DeepSeek-V4-Flash four-Spark cycle.

Orchestrates the existing per-rank launcher (``deepseek_v4_cycle_serve.sh``)
from one node over SSH. Ranks and their SSH targets come from a SparkRing
cluster inventory YAML (the ``scripts/sparkring_cluster.py`` schema); this
tool contains no site-specific host names, addresses, or accounts.

Why it exists
  The DeepSeek cycle quickstart launches one rank per host by hand and in a
  fixed order (workers first, then rank 0). For a deployment with a stable
  cluster inventory that is repetitive and easy to get wrong. This controller
  turns it into one command per lifecycle action, with the same ordering and
  the same per-rank launcher.

Commands
  start
    Start worker ranks first (rank 1, 2, ..., N-1), then rank 0 (the head),
    then poll the head API until it answers or a timeout elapses. Each rank
    is launched over SSH with ``nohup`` so the SSH session returns. A rank
    whose container is already running is skipped, matching the launcher's
    own "container already exists" guard.
  stop
    Remove the serving container on every rank (workers first, then head).
    Idempotent: ranks with no container are reported and skipped.
  status
    Print one line per rank with container state, plus the head API HTTP
    status. Read-only.

Placeholder policy
  Host facts are read from ``--cluster`` only. ``--repo`` is the path to the
  SparkRing checkout on every rank (the parent of ``scripts/`` and of the
  ``rank-<N>.env`` files). ``--api-port`` is the rank-0 API port from the
  cycle environment template. Nothing else is required.

Operational notes
  - The container name matches ``deepseek_v4_cycle_serve.sh``:
    ``deepseek-v4-flash-r<NODE_RANK>``.
  - The per-rank environment file is ``rank-<N>.env`` in ``--repo``.
  - Use ``--dry-run`` to print every SSH command without executing it.
  - A failed rank aborts ``start`` so a half-started cluster is not left
    behind; run ``stop`` and re-run ``start`` after fixing the failure.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

try:
    from .sparkring_cluster import load_cluster
except ImportError:  # Direct execution: python scripts/deepseek_v4_cycle_ctl.py
    from sparkring_cluster import load_cluster

DEFAULT_SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
DEFAULT_CONTAINER_PREFIX = "deepseek-v4-flash-r"
LAUNCH_REL = "scripts/deepseek_v4_cycle_serve.sh"
ENV_REL = "rank-{rank}.env"  # per-rank env file inside --repo
API_PATH = "/v1/models"


def _run_ssh(
    ssh_target: str,
    command: str,
    *,
    timeout: float = 120.0,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one remote command over SSH. Raises on SSH failure only."""
    argv = ["ssh", *DEFAULT_SSH_OPTS, ssh_target, command]
    kwargs: dict = {"text": True, "timeout": timeout}
    if capture:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    return subprocess.run(argv, check=False, **kwargs)


def _rank_env_file(repo: str, rank: int) -> str:
    return f"{repo}/{ENV_REL.format(rank=rank)}"


def _launch_command(repo: str, rank: int, log_path: str) -> str:
    launcher = f"{repo}/{LAUNCH_REL}"
    env_file = _rank_env_file(repo, rank)
    return (
        f"cd {repo} && nohup {launcher} --run {env_file} "
        f">{log_path} 2>&1 </dev/null &"
    )


def _container_name(prefix: str, rank: int) -> str:
    return f"{prefix}{rank}"


def _container_running(ssh_target: str, name: str) -> bool:
    probe = (
        f"docker ps --format '{{{{.Names}}}}' | grep -qx '{name}'"
    )
    result = _run_ssh(ssh_target, probe, timeout=30.0)
    return result.returncode == 0


def _head_api_ready(ssh_target: str, port: int) -> bool:
    probe = f"curl -fsS -m 5 -o /dev/null http://127.0.0.1:{port}{API_PATH}"
    result = _run_ssh(ssh_target, probe, timeout=20.0)
    return result.returncode == 0


def start_ranks(
    ranks: Sequence,
    repo: str,
    container_prefix: str,
    log_dir: str,
    *,
    dry_run: bool = False,
    wait_container: int = 12,
    wait_api_minutes: int = 40,
    api_port: int = 8888,
) -> int:
    """Start workers first, then the head, then wait for the API."""
    ordered = sorted(ranks, key=lambda r: (r.id == 0, r.id))  # head (id 0) last
    for rank in ordered:
        name = _container_name(container_prefix, rank.id)
        log_path = f"{log_dir}/ctl-start-rank{rank.id}.log"
        command = _launch_command(repo, rank.id, log_path)
        if dry_run:
            print(f"[start] rank{rank.id} ({rank.ssh_target}) container={name}")
            print(f"  [dry-run] ssh {rank.ssh_target} {command}")
            continue
        print(f"[start] rank{rank.id} ({rank.ssh_target}) container={name}")
        if _container_running(rank.ssh_target, name):
            print("  already running; skip")
            continue
        print(f"  launch: {command}")
        launched = _run_ssh(rank.ssh_target, command, timeout=60.0)
        if launched.returncode != 0:
            print(f"  FAILED to launch rank{rank.id} (ssh rc={launched.returncode})")
            return 1
        up = False
        for _ in range(wait_container):
            time.sleep(2)
            if _container_running(rank.ssh_target, name):
                up = True
                break
        if not up:
            print(
                f"  FAILED: container {name} did not appear; "
                f"see remote {log_path}"
            )
            return 1
        print(f"  container {name} is up")
    if dry_run:
        print("[start] dry-run complete; nothing started")
        return 0
    if wait_api_minutes <= 0:
        print("[start] API wait disabled; ranks launched")
        return 0
    head = next(r for r in ranks if r.id == 0)
    print(f"[start] waiting for head API on {head.ssh_target}:{api_port} "
          f"(up to {wait_api_minutes} min)")
    deadline = time.monotonic() + wait_api_minutes * 60
    while time.monotonic() < deadline:
        if _head_api_ready(head.ssh_target, api_port):
            print("[start] API ready")
            return 0
        time.sleep(15)
    print(f"[start] TIMEOUT: API not ready after {wait_api_minutes} minutes")
    return 1


def stop_ranks(
    ranks: Sequence,
    container_prefix: str,
    *,
    dry_run: bool = False,
) -> int:
    """Remove serving containers, workers first, then the head."""
    ordered = sorted(ranks, key=lambda r: (r.id == 0, r.id))
    for rank in ordered:
        name = _container_name(container_prefix, rank.id)
        command = f"docker rm -f {name} 2>/dev/null || true"
        if dry_run:
            print(f"[stop] [dry-run] ssh {rank.ssh_target} {command}")
            continue
        _run_ssh(rank.ssh_target, command, timeout=60.0)
        gone = not _container_running(rank.ssh_target, name)
        status = "stopped" if gone else "STILL RUNNING"
        print(f"[stop] rank{rank.id} ({rank.ssh_target}) {name}: {status}")
    return 0


def status_ranks(
    ranks: Sequence,
    container_prefix: str,
    api_port: int,
    *,
    dry_run: bool = False,
) -> int:
    """Print one status line per rank and the head API health."""
    for rank in sorted(ranks, key=lambda r: r.id):
        name = _container_name(container_prefix, rank.id)
        if dry_run:
            print(
                f"[status] rank{rank.id} ({rank.ssh_target}) {name}: "
                f"(dry-run, not probed)"
            )
            continue
        running = _container_running(rank.ssh_target, name)
        state = "UP" if running else "down"
        print(f"[status] rank{rank.id} ({rank.ssh_target}) {name}: {state}")
    head = next(r for r in ranks if r.id == 0)
    if dry_run:
        return 0
    if _head_api_ready(head.ssh_target, api_port):
        print(f"[status] head API http://127.0.0.1:{api_port}{API_PATH}: 200 OK")
    else:
        print(f"[status] head API http://127.0.0.1:{api_port}{API_PATH}: not reachable")
        return 1
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "One-command start/stop/status for the DeepSeek four-Spark cycle, "
            "driven by a SparkRing cluster inventory over SSH."
        )
    )
    parser.add_argument("action", choices=("start", "stop", "status"))
    parser.add_argument(
        "--cluster", required=True, type=Path,
        help="path to the cluster inventory YAML (sparkring_cluster schema)",
    )
    parser.add_argument(
        "--repo", required=True,
        help="path to the SparkRing checkout on every rank "
             "(parent of scripts/ and rank-<N>.env)",
    )
    parser.add_argument(
        "--container-prefix", default=DEFAULT_CONTAINER_PREFIX,
        help=f"container name prefix (default: {DEFAULT_CONTAINER_PREFIX})",
    )
    parser.add_argument(
        "--log-dir", default="/tmp",
        help="remote directory for per-rank launch logs (default: /tmp)",
    )
    parser.add_argument("--api-port", type=int, default=8888,
                        help="rank-0 API port (default: 8888)")
    parser.add_argument("--wait-api-minutes", type=int, default=40,
                        help="head API wait budget for start (default: 40)")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the SSH commands without executing them")
    args = parser.parse_args(argv)

    if not args.cluster.is_file():
        print(f"error: cluster inventory not found: {args.cluster}", file=sys.stderr)
        return 2
    cluster = load_cluster(args.cluster)
    ranks = tuple(cluster.ranks)
    if not ranks or not any(r.id == 0 for r in ranks):
        print("error: cluster inventory has no rank 0 (the head)", file=sys.stderr)
        return 2

    if args.action == "start":
        return start_ranks(
            ranks, args.repo, args.container_prefix, args.log_dir,
            dry_run=args.dry_run,
            wait_api_minutes=args.wait_api_minutes,
            api_port=args.api_port,
        )
    if args.action == "stop":
        return stop_ranks(ranks, args.container_prefix, dry_run=args.dry_run)
    return status_ranks(
        ranks, args.container_prefix, args.api_port, dry_run=args.dry_run
    )


if __name__ == "__main__":
    raise SystemExit(main())
