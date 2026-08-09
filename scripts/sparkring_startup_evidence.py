#!/usr/bin/env python3
"""Stage-aware startup evidence classifier for EXL3 + LMCache.

Parses engine and LMCache-server container logs and Docker state to classify
startup evidence into three categories:

- **recoverable**: bounded legacy mixed-Trellis RM allocation retries where
  the container stayed running and eventually reached readiness.  The
  ``CUDA out of memory`` string can appear in this context because the
  PyTorch expandable-segments allocator retries with a smaller pool; the
  retry succeeds and startup continues.
- **fatal**: Xid GPU errors, OOM-killed containers, unexpected restarts, SSH
  connection failures, or fabric/RoCE initialisation failures.  These are
  never ignored.
- **clean**: no concerning signatures at all.

The classifier is **purely diagnostic**: it reads evidence that has already
been captured (log files, ``docker inspect`` JSON) and returns a structured
JSON report.  It does not contact the cluster, start or stop anything, or
modify any host.  It is OFFLINE when given local files and READ-ONLY REMOTE
when given SSH targets.

Usage::

    # Offline: classify from captured log files
    python scripts/sparkring_startup_evidence.py \\
        --engine-log engine-r0.log --engine-log engine-r1.log \\
        --server-log server-r0.log --server-log server-r1.log \\
        classify

    # Offline: inspect a single log snippet
    python scripts/sparkring_startup_evidence.py \\
        --engine-log engine-r0.log classify

The script intentionally does **not** globally ignore ``CUDA out of memory``.
Instead, it distinguishes a recoverable retry (the allocator retries and
startup continues) from a fatal OOM (the container died or the process
exited).  The distinction is based on whether the container is still running
and whether a subsequent readiness or model-load line appears after the OOM
message.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-startup-evidence/v1"

# Exit codes
EXIT_OK = 0
EXIT_FATAL = 2
EXIT_CONFIG_ERROR = 3

# ---------------------------------------------------------------------------
# Signature patterns
# ---------------------------------------------------------------------------

# Fatal: Xid errors (NVIDIA Xid 13/31/43/48/62/63/74/79/119/120/121 etc.)
_XID = re.compile(r"Xid\s+\d+|NVRM:\s+Xid", re.IGNORECASE)

# Fatal: process killed by OOM (OOMKilled, or explicit kill)
_FATAL_OOM = re.compile(
    r"OOMKilled|"
    r"OutOfMemoryError\s*\[|"
    r"Killed process|"
    r"out of memory\s*\(\d+\)|"
    r"MemoryError",
    re.IGNORECASE,
)

# Fatal: container restart (unexpected, not an operator-initiated restart)
_RESTART = re.compile(
    r"container\s+\S+\s+restarted|"
    r"Container\s+\S+\s+is\s+restarting|"
    r"RestartCount.*[1-9]\d*",
    re.IGNORECASE,
)

# Fatal: SSH / connection failures during startup
_SSH_FAIL = re.compile(
r"ssh:\s+(connect|connection).*\b(refused|timed out|failed|reset)|"
    r"Lost connection|"
    r"ssh_dispatch_run_fatal",
    re.IGNORECASE,
)

# Fatal: fabric / RoCE / NCCL bootstrap failures
_FABRIC_FAIL = re.compile(
    r"NCCL.*\b(FAILED|FATAL|abort)\b|"
    r"RoCE.*\b(failed|error)\b|"
    r"IB.*\b(no\s+path|unreachable|link\s+down)\b|"
    r"transport\s+probe\s+failed|"
    r"bootstrap\s+failed",
    re.IGNORECASE,
)

# Fatal: CUDA driver / runtime init failure
_DRIVER_FAIL = re.compile(
    r"cuda(Init|Driver|SetDevice).*\b(failed|error)\b|"
    r"all\s+CUDA-capable\s+devices\s+are\s+busy|"
    r"no\s+CUDA-capable\s+device\s+detected|"
    r"CUDA\s+error\s+:\s+no\s+kernel\s+image",
    re.IGNORECASE,
)

# Recoverable: PyTorch RM allocation retry (expandable segments)
# The allocator logs "CUDA out of memory" then retries with a smaller pool.
# This is the bounded legacy mixed-Trellis RM allocation retry.
_RM_RETRY = re.compile(
    r"CUDA out of memory.*(?:retry|expandable|segment|attempting|"
    r"free|reserved|allocated)\d*|"
    r"PYTORCH_CUDA_ALLOC_CONF.*retry|"
    r"expandable_segments:True.*retry",
    re.IGNORECASE,
)

# Recoverable: allocator printed OOM but then continued
_ALLOC_CONTINUE = re.compile(
    r"CUDA out of memory.*(?:Trying|Allocating|Retrying|Reserving|"
    r"after\s+freeing)",
    re.IGNORECASE,
)

# Neutral: lines indicating successful progress after a potential OOM
_PROGRESS = re.compile(
    r"Model loaded|"
    r"KV cache allocated|"
    r"Capturing CUDA graphs|"
    r"Engine is ready|"
    r"Uvicorn running|"
    r"API server started|"
    r"Worker ready|"
    r"init engine|"
    r"Memory profiling done|"
    r"KV allocation complete",
    re.IGNORECASE,
)

# Neutral: LMCache server ready
_LMCACHE_READY = re.compile(
    r"LMCache.*server.*started|"
    r"storage_manager.*initialized|"
    r"Listening on",
    re.IGNORECASE,
)

ALL_FATAL_PATTERNS = [
    ("xid", _XID),
    ("fatal_oom", _FATAL_OOM),
    ("unexpected_restart", _RESTART),
    ("ssh_failure", _SSH_FAIL),
    ("fabric_failure", _FABRIC_FAIL),
    ("driver_failure", _DRIVER_FAIL),
]

RECOVERABLE_PATTERNS = [
    ("rm_allocation_retry", _RM_RETRY),
    ("alloc_retry_continue", _ALLOC_CONTINUE),
]


class ConfigError(ValueError):
    """The operator supplied an invalid argument."""


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_log(text: str, *, container_running: bool = True) -> dict[str, Any]:
    """Classify a single container log.

    Returns a structured dict with:
    - ``verdict``: ``clean``, ``recoverable``, or ``fatal``
    - ``fatal_signatures``: list of {pattern, line, line_number}
    - ``recoverable_signatures``: same shape
    - ``progress_after_oom``: whether a progress line appeared after any OOM
    - ``line_count``: total lines parsed
    - ``sha256``: hash of the log for provenance
    """
    lines = text.splitlines()
    fatal_hits: list[dict[str, Any]] = []
    recoverable_hits: list[dict[str, Any]] = []
    oom_line_numbers: list[int] = []
    progress_line_numbers: list[int] = []

    for index, line in enumerate(lines, start=1):
        for label, pattern in ALL_FATAL_PATTERNS:
            if pattern.search(line):
                fatal_hits.append({
                    "pattern": label,
                    "line_number": index,
                    "line": line.strip()[:500],
                })
        for label, pattern in RECOVERABLE_PATTERNS:
            if pattern.search(line):
                recoverable_hits.append({
                    "pattern": label,
                    "line_number": index,
                    "line": line.strip()[:500],
                })
        # Track ALL lines mentioning "CUDA out of memory" for progress
        # detection — including RM retry lines that already matched
        # recoverable patterns.
        if "CUDA out of memory" in line.upper() or "out of memory" in line.lower():
            oom_line_numbers.append(index)
        if _PROGRESS.search(line) or _LMCACHE_READY.search(line):
            progress_line_numbers.append(index)

    # Determine if bare OOM messages are recoverable: a progress line
    # appears after the OOM, and the container is still running.
    progress_after_oom = any(
        any(prog > oom for prog in progress_line_numbers)
        for oom in oom_line_numbers
    )

    # If the container is NOT running, any OOM is fatal, not recoverable.
    if not container_running:
        for oom in oom_line_numbers:
            fatal_hits.append({
                "pattern": "oom_container_dead",
                "line_number": oom,
                "line": lines[oom - 1].strip()[:500] if oom <= len(lines) else "",
            })
        oom_line_numbers = []

    # Bare OOM with progress after it and container running = recoverable
    for oom in oom_line_numbers:
        if progress_after_oom:
            recoverable_hits.append({
                "pattern": "bare_oom_with_progress",
                "line_number": oom,
                "line": lines[oom - 1].strip()[:500] if oom <= len(lines) else "",
            })
        else:
            # Bare OOM without progress: treat as fatal (conservative)
            fatal_hits.append({
                "pattern": "bare_oom_no_progress",
                "line_number": oom,
                "line": lines[oom - 1].strip()[:500] if oom <= len(lines) else "",
            })

    if fatal_hits:
        verdict = "fatal"
    elif recoverable_hits:
        verdict = "recoverable"
    else:
        verdict = "clean"

    return {
        "verdict": verdict,
        "fatal_signatures": fatal_hits,
        "recoverable_signatures": recoverable_hits,
        "progress_after_oom": progress_after_oom,
        "line_count": len(lines),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def classify_rank(
    rank: int,
    engine_log: str | None = None,
    server_log: str | None = None,
    *,
    engine_running: bool = True,
    server_running: bool = True,
) -> dict[str, Any]:
    """Classify startup evidence for one rank.

    Each of engine_log and server_log is optional. The overall rank verdict
    is the worst of the two (fatal > recoverable > clean).
    """
    engine_result = None
    server_result = None
    if engine_log is not None:
        engine_result = classify_log(
            engine_log, container_running=engine_running,
        )
    if server_log is not None:
        server_result = classify_log(
            server_log, container_running=server_running,
        )

    verdicts = [
        r["verdict"] for r in (engine_result, server_result) if r is not None
    ]
    if "fatal" in verdicts:
        rank_verdict = "fatal"
    elif "recoverable" in verdicts:
        rank_verdict = "recoverable"
    else:
        rank_verdict = "clean"

    return {
        "rank": rank,
        "verdict": rank_verdict,
        "engine": engine_result,
        "server": server_result,
    }


def aggregate_report(rank_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-rank evidence into a single report."""
    verdicts = [r["verdict"] for r in rank_reports]
    if "fatal" in verdicts:
        overall = "fatal"
    elif "recoverable" in verdicts:
        overall = "recoverable"
    else:
        overall = "clean"

    fatal_count = sum(
        1 for r in rank_reports
        for component in ("engine", "server")
        if r.get(component) and r[component]["fatal_signatures"]
    )
    recoverable_count = sum(
        1 for r in rank_reports
        for component in ("engine", "server")
        if r.get(component) and r[component]["recoverable_signatures"]
    )

    return {
        "schema": SCHEMA,
        "verdict": overall,
        "rank_count": len(rank_reports),
        "ranks_fatal": sorted(
            r["rank"] for r in rank_reports if r["verdict"] == "fatal"
        ),
        "ranks_recoverable": sorted(
            r["rank"] for r in rank_reports if r["verdict"] == "recoverable"
        ),
        "ranks_clean": sorted(
            r["rank"] for r in rank_reports if r["verdict"] == "clean"
        ),
        "fatal_signature_count": fatal_count,
        "recoverable_signature_count": recoverable_count,
        "ranks": rank_reports,
        "evidence_scope": (
            "Diagnostic log classification from captured evidence; "
            "does not contact or mutate the cluster"
        ),
        "classification_note": (
            "Bounded RM allocation retries (CUDA out of memory followed by "
            "progress) are classified recoverable when the container stayed "
            "running. Xid, OOMKilled, unexpected restarts, SSH failures, "
            "fabric failures, and bare OOM without subsequent progress are "
            "fatal. NVIDIA errors are never globally ignored."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--engine-log",
        action="append",
        dest="engine_logs",
        default=[],
        help=(
            "path to a captured engine container log; repeat once per rank "
            "(order = rank 0, 1, 2, 3)"
        ),
    )
    parser.add_argument(
        "--server-log",
        action="append",
        dest="server_logs",
        default=[],
        help=(
            "path to a captured LMCache server container log; repeat once per rank"
        ),
    )
    parser.add_argument(
        "--engine-dead",
        action="append_const",
        const=True,
        dest="engine_dead",
        default=[],
        help=(
            "mark the corresponding engine container as not-running; "
            "repeat to mark multiple ranks"
        ),
    )
    parser.add_argument(
        "--server-dead",
        action="append_const",
        const=True,
        dest="server_dead",
        default=[],
        help="mark the corresponding LMCache server container as not-running",
    )
    parser.add_argument("command", choices=("classify",))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.engine_logs and not args.server_logs:
        parser.error("at least one --engine-log or --server-log is required")

    max_ranks = max(len(args.engine_logs), len(args.server_logs))
    if max_ranks == 0:
        parser.error("no logs provided")

    rank_reports: list[dict[str, Any]] = []
    for rank in range(max_ranks):
        engine_log = None
        server_log = None
        if rank < len(args.engine_logs):
            path = Path(args.engine_logs[rank])
            if not path.is_file():
                print(
                    f"sparkring-startup-evidence: engine log not found: {path}",
                    file=sys.stderr,
                )
                return EXIT_CONFIG_ERROR
            engine_log = path.read_text(encoding="utf-8", errors="replace")
        if rank < len(args.server_logs):
            path = Path(args.server_logs[rank])
            if not path.is_file():
                print(
                    f"sparkring-startup-evidence: server log not found: {path}",
                    file=sys.stderr,
                )
                return EXIT_CONFIG_ERROR
            server_log = path.read_text(encoding="utf-8", errors="replace")

        engine_running = rank >= len(args.engine_dead)
        server_running = rank >= len(args.server_dead)

        rank_reports.append(
            classify_rank(
                rank,
                engine_log=engine_log,
                server_log=server_log,
                engine_running=engine_running,
                server_running=server_running,
            )
        )

    report = aggregate_report(rank_reports)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "fatal":
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
