#!/usr/bin/env python3
"""Reclaim and compact host memory before a large GB10 model launch.

The default mode prints the exact per-rank commands without contacting a host.
Execution requires an explicit confirmation token. Each remote command refuses
to proceed while a configured serving or rendezvous port is listening, asks
the kernel to release clean page-cache pages, requests memory compaction, and
then returns. The tool runs the ordinary read-only preflight afterward and
reports whether every configured memory threshold recovered.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Sequence

import preflight
from preflight import CheckResult
from sparkring_site import SiteConfigError, load_site


CONFIRMATION = "PREPARE_GB10_LAUNCH_MEMORY"
PLAN_SCHEMA = "sparkring-launch-memory-plan/v1"
RECEIPT_SCHEMA = "sparkring-launch-memory-recovery/v1"
MEMORY_CHECK_IDS = (
    "HOST.MEMORY_AVAILABLE",
    "HOST.MEMORY_CONTIGUITY",
)


class PrepareMemoryError(RuntimeError):
    """The requested host-memory preparation could not be verified."""


def _memory_snapshot_function() -> str:
    return "\n".join(
        (
            "memory_snapshot() {",
            "  _label=$1",
            "  printf '%s\\n' \"${_label}\"",
            "  printf 'MEM_PAGE_SIZE '; getconf PAGESIZE 2>/dev/null || printf '-\\n'",
            "  awk '$1 == \"MemAvailable:\" {print \"MEM_AVAILABLE_KIB\", $2}' /proc/meminfo",
            "  awk '{printf \"BUDDY %s\", $4; for (i = 5; i <= NF; i++) "
            "printf \" %s\", $i; printf \"\\n\"}' /proc/buddyinfo",
            "  awk '$1 == \"compact_stall\" || $1 == \"compact_fail\" || "
            "$1 == \"compact_success\" {print \"VMSTAT\", $1, $2}' /proc/vmstat",
            "}",
        )
    )


def _port_guard_lines(required_free_ports: Sequence[int]) -> list[str]:
    ports = " ".join(str(port) for port in required_free_ports)
    if not ports:
        return []
    return [
        f"for _port in {ports}; do",
        "  if ss -ltnH \"sport = :${_port}\" 2>/dev/null | grep -q .; then",
        "    printf 'refusing memory preparation: tcp/%s has a listener\\n' "
        '"${_port}" >&2',
        "    exit 73",
        "  fi",
        "done",
    ]


def precheck_command(required_free_ports: Sequence[int]) -> tuple[str, ...]:
    """Return a non-mutating readiness check used on every rank first."""
    lines = ["set -euo pipefail"]
    lines.extend(_port_guard_lines(required_free_ports))
    lines.append("sudo -n true")
    return ("bash", "-lc", "\n".join(lines))


def remote_command(required_free_ports: Sequence[int]) -> tuple[str, ...]:
    """Return one guarded remote shell command for a rank."""
    lines = [
        "set -euo pipefail",
        _memory_snapshot_function(),
    ]
    lines.extend(_port_guard_lines(required_free_ports))
    lines.extend(
        (
            "sudo -n true",
            "memory_snapshot MEMORY_BEFORE",
            "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; "
            "echo 1 > /proc/sys/vm/compact_memory'",
            "sleep 2",
            "memory_snapshot MEMORY_AFTER",
        )
    )
    return ("bash", "-lc", "\n".join(lines))


def _require_memory_config(site: Any) -> Any:
    memory = getattr(site.preflight, "memory", None)
    if memory is None:
        raise PrepareMemoryError(
            "preflight.memory thresholds are required before host mutation"
        )
    return memory


def plan_document(site: Any) -> dict[str, Any]:
    """Describe every remote mutation without executing it."""
    memory = _require_memory_config(site)
    command = remote_command(site.preflight.required_free_ports)
    return {
        "schema": PLAN_SCHEMA,
        "safety": ["MUTATES HOST"],
        "confirmation": CONFIRMATION,
        "thresholds": {
            "minimum_available_bytes": memory.minimum_available_bytes,
            "contiguous_block_bytes": memory.contiguous_block_bytes,
            "minimum_contiguous_blocks": memory.minimum_contiguous_blocks,
        },
        "actions": [
            {
                "rank": rank.id,
                "ssh_target": rank.ssh_target,
                "precheck": list(
                    precheck_command(site.preflight.required_free_ports)
                ),
                "command": list(command),
            }
            for rank in site.ranks
        ],
    }


def _prepare_one(
    rank: Any,
    required_free_ports: Sequence[int],
    timeout_seconds: int,
) -> dict[str, Any]:
    remote = shlex.join(remote_command(required_free_ports))
    completed = subprocess.run(
        ("ssh", rank.ssh_target, remote),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PrepareMemoryError(
            f"rank {rank.id} memory preparation failed: {detail}"
        )
    return {
        "rank": rank.id,
        "ssh_target": rank.ssh_target,
        "stdout": completed.stdout.strip(),
    }


def _precheck_one(
    rank: Any,
    required_free_ports: Sequence[int],
    timeout_seconds: int,
) -> dict[str, Any]:
    remote = shlex.join(precheck_command(required_free_ports))
    completed = subprocess.run(
        ("ssh", rank.ssh_target, remote),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PrepareMemoryError(
            f"rank {rank.id} precheck failed before any host mutation: {detail}"
        )
    return {"rank": rank.id, "ssh_target": rank.ssh_target}


def build_receipt(
    site: Any,
    actions: Sequence[dict[str, Any]],
    checks: Sequence[CheckResult],
) -> dict[str, Any]:
    """Combine mutation output with the read-only post-action verification."""
    _require_memory_config(site)
    memory_checks = [
        check for check in checks if check.check_id in MEMORY_CHECK_IDS
    ]
    observed = {(check.rank, check.check_id) for check in memory_checks}
    expected = {
        (rank.id, check_id)
        for rank in site.ranks
        for check_id in MEMORY_CHECK_IDS
    }
    passed = observed == expected and all(check.passed for check in memory_checks)
    return {
        "schema": RECEIPT_SCHEMA,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "recovered" if passed else "reboot-required",
        "passed": passed,
        "safety": ["MUTATES HOST"],
        "actions": sorted(actions, key=lambda action: action["rank"]),
        "verification": [
            check.to_dict()
            for check in sorted(
                memory_checks, key=lambda check: (check.rank, check.check_id)
            )
        ],
        "recommended_action": (
            "Proceed with the model launch."
            if passed else
            "Reboot every rank whose memory check failed, then rerun preflight."
        ),
    }


def prepare_cluster(site: Any, timeout_seconds: int) -> dict[str, Any]:
    """Prepare ranks concurrently, then verify them through read-only preflight."""
    _require_memory_config(site)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(site.ranks)) as pool:
        prechecks = [
            pool.submit(
                _precheck_one,
                rank,
                site.preflight.required_free_ports,
                timeout_seconds,
            )
            for rank in site.ranks
        ]
        for future in concurrent.futures.as_completed(prechecks):
            future.result()

    actions: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(site.ranks)) as pool:
        futures = [
            pool.submit(
                _prepare_one,
                rank,
                site.preflight.required_free_ports,
                timeout_seconds,
            )
            for rank in site.ranks
        ]
        for future in concurrent.futures.as_completed(futures):
            actions.append(future.result())

    checks = preflight.run_preflight(
        site,
        preflight.SshRunner(timeout_seconds),
        scope="full",
    )
    return build_receipt(site, actions, checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--site", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    args = parser.parse_args()

    try:
        site = load_site(args.site)
        if not args.execute:
            print(json.dumps(plan_document(site), indent=2, sort_keys=True))
            return 0
        if args.confirmation != CONFIRMATION:
            parser.error(f"execute requires --confirmation {CONFIRMATION}")
        receipt = prepare_cluster(site, args.timeout_seconds)
    except (
        OSError,
        PrepareMemoryError,
        SiteConfigError,
        subprocess.TimeoutExpired,
    ) as exc:
        parser.error(str(exc))

    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0 if receipt["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
