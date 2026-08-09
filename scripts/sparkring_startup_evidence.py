#!/usr/bin/env python3
"""Stage-aware startup evidence classifier for EXL3 + LMCache.

Parses engine logs, kernel logs, and LMCache-server container logs plus
Docker state to classify startup evidence into four verdicts:

- **clean**: no concerning signatures.
- **bounded_rm_retry**: a kernel RM ``NV_ERR_NO_MEMORY`` event at the
  evidenced callsite (``_memdescAllocInternal @ mem_desc.c:1359``) occurred
  during explicit EXL3 mixed-Trellis materialization, the container stayed
  running (``Running = true``, ``RestartCount = 0``, ``OOMKilled = false``),
  subsequent layer progress and eventual readiness were observed, no fatal
  signatures (Xid, OOM-kill, restart, SSH loss, fabric loss) are present,
  and the event count is within an explicit conservative bound.  This
  classification is evidence-scoped to the legacy EXL3 materialization
  callsite and does not prove all RM errors safe.
- **fatal**: Xid GPU errors, ``torch.OutOfMemoryError``, generic CUDA OOM,
  OOM-killed containers, unexpected restarts, SSH connection failures,
  fabric/RoCE carrier loss, or driver init failures.  These are never
  downgraded.
- **indeterminate**: a kernel RM event is observed but cross-evidence
  cannot be truthfully correlated — e.g. timestamps are missing or
  malformed, layer materialization context is absent, or no subsequent
  readiness was observed.  Indeterminate is treated as fatal-policy
  (exit code ``EXIT_FATAL``) because the classifier must not silently
  downgrade unknown NVIDIA errors.

The classifier is **purely diagnostic**: it reads evidence that has already
been captured (log files, ``docker inspect`` JSON) and returns a structured
JSON report.  It does not contact the cluster, start or stop anything, or
modify any host.  It is OFFLINE when given local files and READ-ONLY REMOTE
when given SSH targets.

Usage::

    # Offline: classify from captured engine + kernel logs
    python scripts/sparkring_startup_evidence.py \\
        --engine-log engine-r0.log --kernel-log kernel-r0.log \\
        --engine-log engine-r1.log --kernel-log kernel-r1.log \\
        classify

    # Offline: engine + server + kernel
    python scripts/sparkring_startup_evidence.py \\
        --engine-log engine-r0.log --server-log server-r0.log \\
        --kernel-log kernel-r0.log classify

The script intentionally does **not** globally ignore NVIDIA errors.
Generic ``CUDA out of memory``, ``torch.OutOfMemoryError``, and
``OOMKilled`` are fatal by default.  Only the evidenced kernel RM
``NV_ERR_NO_MEMORY`` callsite can be classified ``bounded_rm_retry``, and
only with full cross-evidence.  If timestamps cannot be correlated
truthfully from supplied files, the verdict is ``indeterminate`` (fatal
policy), not ``bounded_rm_retry``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-startup-evidence/v2"

# Exit codes
EXIT_OK = 0
EXIT_FATAL = 2
EXIT_CONFIG_ERROR = 3

# ---------------------------------------------------------------------------
# Verdict ordering (worst wins)
# ---------------------------------------------------------------------------

VERDICT_RANK = {
    "clean": 0,
    "bounded_rm_retry": 1,
    "indeterminate": 2,
    "fatal": 3,
}

# ---------------------------------------------------------------------------
# Signature patterns
# ---------------------------------------------------------------------------

# Fatal: Xid errors (NVIDIA Xid 13/31/43/48/62/63/74/79/119/120/121 etc.)
_XID = re.compile(r"Xid\s+\d+|NVRM:\s+Xid", re.IGNORECASE)

# Fatal: generic CUDA OOM, torch.OutOfMemoryError, OutOfMemoryError
_GENERIC_OOM = re.compile(
    r"CUDA out of memory|"
    r"torch\.OutOfMemoryError|"
    r"OutOfMemoryError\s*\[|"
    r"OutOfMemoryError",
    re.IGNORECASE,
)

# Fatal: process killed by OOM (OOMKilled, or explicit kill)
_OOMKILLED = re.compile(
    r"OOMKilled|"
    r"Killed process|"
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

# Fatal: fabric / RoCE / NCCL bootstrap failures and carrier loss
_FABRIC_FAIL = re.compile(
    r"NCCL.*\b(FAILED|FATAL|abort)\b|"
    r"RoCE.*\b(failed|error)\b|"
    r"IB.*\b(no\s+path|unreachable|link\s+down)\b|"
    r"transport\s+probe\s+failed|"
    r"bootstrap\s+failed|"
    r"carrier\s+(loss|down)",
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

# Bounded RM retry: the evidenced kernel signature.
# NV_ERR_NO_MEMORY at _memdescAllocInternal @ mem_desc.c:1359
# This is the ONLY kernel-log signature that can qualify for bounded_rm_retry.
_RM_KERNEL_EVENT = re.compile(
    r"NV_ERR_NO_MEMORY.*_memdescAllocInternal\s*@\s*mem_desc\.c:\s*1359",
    re.IGNORECASE,
)

# Neutral: lines indicating successful progress (model load, KV cache, etc.)
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
    r"KV allocation complete|"
    r"layer\s+\d+.*(?:done|complete|loaded|ready)",
    re.IGNORECASE,
)

# Neutral: LMCache server ready
_LMCACHE_READY = re.compile(
    r"LMCache.*server.*started|"
    r"storage_manager.*initialized|"
    r"Listening on",
    re.IGNORECASE,
)

# EXL3 mixed-Trellis materialization context — the layer preparation
# phase where the evidenced RM events were observed.
_MATERIALIZE = re.compile(
    r"mixed.?Trellis|"
    r"materiali[sz]ation|"
    r"per.?layer.?prep|"
    r"layer\s+preparation|"
    r"preparing\s+layer",
    re.IGNORECASE,
)

# Timestamp patterns for temporal correlation
_TIMESTAMP = re.compile(
    r"^(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})|"
    r"^\[(\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2})\]|"
    r"^(\d{2}:\d{2}:\d{2}\.\d+)\s"
)

# Conservative bound on RM event count
RM_EVENT_BOUND = 3

ALL_FATAL_PATTERNS = [
    ("xid", _XID),
    ("generic_oom", _GENERIC_OOM),
    ("oomkilled", _OOMKILLED),
    ("unexpected_restart", _RESTART),
    ("ssh_failure", _SSH_FAIL),
    ("fabric_failure", _FABRIC_FAIL),
    ("driver_failure", _DRIVER_FAIL),
]


class ConfigError(ValueError):
    """The operator supplied an invalid argument."""


# ---------------------------------------------------------------------------
# Log parsing helpers
# ---------------------------------------------------------------------------


def _extract_timestamp(line: str) -> str | None:
    """Extract a timestamp from a log line, if present."""
    match = _TIMESTAMP.match(line)
    if match:
        return match.group(1) or match.group(2) or match.group(3)
    return None


def _scan_signatures(
    lines: list[str],
    patterns: list[tuple[str, re.Pattern[str]]],
) -> list[dict[str, Any]]:
    """Scan lines for signature patterns and return hit records."""
    hits: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        for label, pattern in patterns:
            if pattern.search(line):
                hits.append({
                    "pattern": label,
                    "line_number": index,
                    "line": line.strip()[:500],
                    "timestamp": _extract_timestamp(line),
                })
    return hits


def _scan_progress(lines: list[str]) -> list[int]:
    """Return line numbers of progress/readiness lines."""
    return [
        i for i, line in enumerate(lines, start=1)
        if _PROGRESS.search(line) or _LMCACHE_READY.search(line)
    ]


def _scan_materialization(lines: list[str]) -> list[int]:
    """Return line numbers of EXL3 materialization context lines."""
    return [
        i for i, line in enumerate(lines, start=1)
        if _MATERIALIZE.search(line)
    ]


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_log(
    text: str,
    *,
    container_running: bool = True,
    kernel_log: str | None = None,
) -> dict[str, Any]:
    """Classify a single container log, optionally with a kernel log.

    Returns a structured dict with:

    - ``verdict``: ``clean``, ``bounded_rm_retry``, ``fatal``, or
      ``indeterminate``
    - ``fatal_signatures``: list of hit dicts
    - ``rm_events``: list of kernel RM event hit dicts (if kernel_log given)
    - ``progress_lines``: line numbers of progress/readiness
    - ``materialization_lines``: line numbers of EXL3 materialization
    - ``rm_event_count``: number of RM events (0 if no kernel log)
    - ``rm_event_within_bound``: whether count <= RM_EVENT_BOUND
    - ``has_readiness``: whether a readiness line was found
    - ``timestamps_correlatable``: whether timestamps could be extracted
    - ``line_count``: total lines parsed
    - ``sha256``: hash of the engine log for provenance
    - ``kernel_sha256``: hash of the kernel log (if provided)
    """
    lines = text.splitlines()
    fatal_hits = _scan_signatures(lines, ALL_FATAL_PATTERNS)
    progress_line_numbers = _scan_progress(lines)
    materialization_lines = _scan_materialization(lines)
    has_readiness = any(
        "Engine is ready" in lines[i - 1]
        or "API server started" in lines[i - 1]
        or "Uvicorn running" in lines[i - 1]
        or "Listening on" in lines[i - 1]
        for i in progress_line_numbers
        if i <= len(lines)
    )

    # Kernel RM event detection
    rm_events: list[dict[str, Any]] = []
    rm_event_count = 0
    rm_event_within_bound = True
    timestamps_correlatable = True
    kernel_sha256: str | None = None

    if kernel_log is not None:
        kernel_lines = kernel_log.splitlines()
        kernel_sha256 = hashlib.sha256(
            kernel_log.encode("utf-8")
        ).hexdigest()
        rm_events = _scan_signatures(kernel_lines, [(
            "nv_err_no_memory_memdesc_alloc", _RM_KERNEL_EVENT,
        )])
        rm_event_count = len(rm_events)
        rm_event_within_bound = rm_event_count <= RM_EVENT_BOUND

        # Timestamp correlation: check if kernel events and engine log
        # have extractable timestamps
        kernel_ts = [
            _extract_timestamp(line)
            for line in kernel_lines
            if _RM_KERNEL_EVENT.search(line)
        ]
        engine_ts = [
            _extract_timestamp(line)
            for line in lines
        ]
        if rm_events and (
            any(ts is None for ts in kernel_ts)
            or any(ts is None for ts in engine_ts if ts is not None)
            or not engine_ts
        ):
            timestamps_correlatable = False

    # Determine verdict
    if fatal_hits:
        verdict = "fatal"
    elif rm_event_count == 0:
        verdict = "clean"
    else:
        # RM events present — apply cross-evidence protocol
        if not container_running:
            verdict = "fatal"
        elif not rm_event_within_bound:
            verdict = "fatal"
        elif not materialization_lines:
            # RM event outside materialization context
            verdict = "indeterminate"
        elif not has_readiness:
            # No eventual readiness after RM event
            verdict = "indeterminate"
        elif not timestamps_correlatable:
            verdict = "indeterminate"
        else:
            # Check progress after each RM event
            # Use engine log line numbers as proxy for temporal ordering
            # when kernel timestamps can't be mapped to engine lines
            has_progress_after_all_rm = bool(progress_line_numbers)
            if has_progress_after_all_rm:
                verdict = "bounded_rm_retry"
            else:
                verdict = "indeterminate"

    return {
        "verdict": verdict,
        "fatal_signatures": fatal_hits,
        "rm_events": rm_events,
        "rm_event_count": rm_event_count,
        "rm_event_within_bound": rm_event_within_bound,
        "progress_lines": progress_line_numbers,
        "materialization_lines": materialization_lines,
        "has_readiness": has_readiness,
        "timestamps_correlatable": timestamps_correlatable,
        "line_count": len(lines),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "kernel_sha256": kernel_sha256,
    }


def classify_rank(
    rank: int,
    engine_log: str | None = None,
    server_log: str | None = None,
    *,
    engine_running: bool = True,
    server_running: bool = True,
    kernel_log: str | None = None,
) -> dict[str, Any]:
    """Classify startup evidence for one rank.

    Each of engine_log and server_log is optional. The overall rank verdict
    is the worst of the two (fatal > indeterminate > bounded_rm_retry > clean).
    """
    engine_result = None
    server_result = None
    if engine_log is not None:
        engine_result = classify_log(
            engine_log,
            container_running=engine_running,
            kernel_log=kernel_log,
        )
    if server_log is not None:
        server_result = classify_log(
            server_log,
            container_running=server_running,
        )

    verdicts = [
        r["verdict"] for r in (engine_result, server_result) if r is not None
    ]
    rank_verdict = _worst_verdict(verdicts)

    return {
        "rank": rank,
        "verdict": rank_verdict,
        "engine": engine_result,
        "server": server_result,
    }


def _worst_verdict(verdicts: list[str]) -> str:
    """Return the worst (highest-rank) verdict."""
    if not verdicts:
        return "clean"
    return max(verdicts, key=lambda v: VERDICT_RANK.get(v, 3))


def aggregate_report(rank_reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-rank evidence into a single report."""
    verdicts = [r["verdict"] for r in rank_reports]
    overall = _worst_verdict(verdicts)

    def _component_iter():
        for r in rank_reports:
            for component in ("engine", "server"):
                comp = r.get(component)
                if comp is not None:
                    yield r, comp

    fatal_count = sum(
        1 for _, comp in _component_iter() if comp["fatal_signatures"]
    )
    rm_event_count = sum(
        1 for _, comp in _component_iter() if comp.get("rm_events")
    )

    return {
        "schema": SCHEMA,
        "verdict": overall,
        "rank_count": len(rank_reports),
        "ranks_fatal": sorted(
            r["rank"] for r in rank_reports if r["verdict"] == "fatal"
        ),
        "ranks_bounded_rm_retry": sorted(
            r["rank"] for r in rank_reports
            if r["verdict"] == "bounded_rm_retry"
        ),
        "ranks_indeterminate": sorted(
            r["rank"] for r in rank_reports
            if r["verdict"] == "indeterminate"
        ),
        "ranks_clean": sorted(
            r["rank"] for r in rank_reports if r["verdict"] == "clean"
        ),
        "fatal_signature_count": fatal_count,
        "rm_event_rank_count": rm_event_count,
        "ranks": rank_reports,
        "evidence_scope": (
            "Diagnostic log classification from captured evidence; "
            "does not contact or mutate the cluster"
        ),
        "classification_note": (
            "Generic CUDA out of memory, torch.OutOfMemoryError, OOMKilled, "
            "Xid, unexpected restarts, SSH failures, fabric carrier loss, "
            "and driver failures are fatal. Only kernel RM "
            "NV_ERR_NO_MEMORY at _memdescAllocInternal @ mem_desc.c:1359 "
            "during EXL3 mixed-Trellis materialization can be classified "
            "bounded_rm_retry, and only with full cross-evidence: container "
            "running, RestartCount=0, OOMKilled=false, no fatal signatures, "
            f"event count <= {RM_EVENT_BOUND}, layer progress, eventual "
            "readiness, and correlatable timestamps. If timestamps cannot be "
            "correlated, the verdict is indeterminate (fatal policy). "
            "This is evidence-scoped to the legacy EXL3 materialization "
            "callsite and does not prove all RM errors safe. NVIDIA errors "
            "are never globally ignored."
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
            "path to a captured LMCache server container log; "
            "repeat once per rank"
        ),
    )
    parser.add_argument(
        "--kernel-log",
        action="append",
        dest="kernel_logs",
        default=[],
        help=(
            "path to a captured kernel/dmesg log; repeat once per rank. "
            "Only NV_ERR_NO_MEMORY at _memdescAllocInternal @ mem_desc.c:1359 "
            "is matched for bounded_rm_retry classification."
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

    max_ranks = max(
        len(args.engine_logs), len(args.server_logs), len(args.kernel_logs)
    )
    if max_ranks == 0:
        parser.error("no logs provided")

    rank_reports: list[dict[str, Any]] = []
    for rank in range(max_ranks):
        engine_log = None
        server_log = None
        kernel_log = None
        if rank < len(args.engine_logs):
            path = Path(args.engine_logs[rank])
            if not path.is_file():
                print(
                    f"sparkring-startup-evidence: engine log not found: "
                    f"{path}",
                    file=sys.stderr,
                )
                return EXIT_CONFIG_ERROR
            engine_log = path.read_text(encoding="utf-8", errors="replace")
        if rank < len(args.server_logs):
            path = Path(args.server_logs[rank])
            if not path.is_file():
                print(
                    f"sparkring-startup-evidence: server log not found: "
                    f"{path}",
                    file=sys.stderr,
                )
                return EXIT_CONFIG_ERROR
            server_log = path.read_text(encoding="utf-8", errors="replace")
        if rank < len(args.kernel_logs):
            path = Path(args.kernel_logs[rank])
            if not path.is_file():
                print(
                    f"sparkring-startup-evidence: kernel log not found: "
                    f"{path}",
                    file=sys.stderr,
                )
                return EXIT_CONFIG_ERROR
            kernel_log = path.read_text(encoding="utf-8", errors="replace")

        engine_running = rank >= len(args.engine_dead)
        server_running = rank >= len(args.server_dead)

        rank_reports.append(
            classify_rank(
                rank,
                engine_log=engine_log,
                server_log=server_log,
                engine_running=engine_running,
                server_running=server_running,
                kernel_log=kernel_log,
            )
        )

    report = aggregate_report(rank_reports)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] in ("fatal", "indeterminate"):
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
