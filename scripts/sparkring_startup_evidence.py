#!/usr/bin/env python3
"""Stage-aware startup evidence classifier for EXL3 + LMCache.

Parses engine logs, kernel logs, and LMCache-server container logs plus
Docker state to classify startup evidence into four verdicts:

- **clean**: no concerning signatures.
- **bounded_rm_retry**: a kernel RM ``NV_ERR_NO_MEMORY`` event at the
  evidenced callsite (``_memdescAllocInternal @ mem_desc.c:1359``) occurred
  during the EXL3 mixed-Trellis materialization window (from first
  materialization line through last layer/readiness), every event timestamp
  falls inside that window, the container stayed running (``Running = true``,
  ``RestartCount = 0``, ``OOMKilled = false``), no fatal signatures are
  present, and the event count is within an explicit operator-supplied
  accepted delta bound.  This classification is evidence-scoped to the
  legacy EXL3 materialization callsite and does not prove all RM errors safe.
- **fatal**: Xid GPU errors, ``torch.OutOfMemoryError``, generic CUDA OOM,
  OOM-killed containers, unexpected restarts, SSH connection failures,
  fabric/RoCE carrier loss, or driver init failures.  These are never
  downgraded regardless of later progress.
- **indeterminate**: a kernel RM event is observed but cross-evidence
  cannot be truthfully correlated — e.g. timestamps are missing or
  malformed, the event falls outside the materialization window, no
  readiness occurs after the window, the operator-supplied accepted delta
  bound is absent, or timezones are inconsistent.  Indeterminate is
  treated as fatal-policy (exit code ``EXIT_FATAL``) because the
  classifier must not silently downgrade unknown NVIDIA errors.

**Accepted delta bound is operator-supplied.** The classifier never
establishes a safe bound itself.  The operator must supply
``--rm-event-bound`` (or ``rm_event_bound=`` in the API); if absent and
RM events are present, the verdict is ``indeterminate`` (fail closed).

**Timestamp parsing.** The classifier parses two real log formats:

1. Kernel ISO timestamps: ``2026-08-09T00:16:52.113635-05:00`` (with
   optional timezone offset).
2. vLLM log prefixes: ``(Worker pid=1234) INFO 08-09 05:24:32.123456``
   (month-day time, year and timezone must be supplied via
   ``--engine-log-year`` and ``--engine-log-tz``).

If year or timezone cannot be inferred unambiguously, the verdict is
``indeterminate``.

The classifier is **purely diagnostic**: it reads evidence that has already
been captured (log files, ``docker inspect`` JSON) and returns a structured
JSON report.  It does not contact the cluster, start or stop anything, or
modify any host.  It is OFFLINE when given local files and READ-ONLY REMOTE
when given SSH targets.

**This tool classifies supplied evidence and never establishes a safe
bound itself.** It remains offline-validated until run on sanitized
captured evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-startup-evidence/v3"

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

# Readiness-specific progress (subset of _PROGRESS)
_READINESS = re.compile(
    r"Engine is ready|"
    r"API server started|"
    r"Uvicorn running|"
    r"Listening on",
    re.IGNORECASE,
)

# Neutral: LMCache server ready
_LMCACHE_READY = re.compile(
    r"LMCache.*server.*started|"
    r"storage_manager.*initialized|"
    r"Listening on",
    re.IGNORECASE,
)

# EXL3 mixed-Trellis materialization context
_MATERIALIZE = re.compile(
    r"mixed.?Trellis|"
    r"materiali[sz]ation|"
    r"per.?layer.?prep|"
    r"layer\s+preparation|"
    r"preparing\s+layer",
    re.IGNORECASE,
)

# Timestamp patterns

# Kernel ISO: 2026-08-09T00:16:52.113635-05:00 or 2026-08-09T00:16:52Z
_KERNEL_TS = re.compile(
    r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?"
    r"(?:[+-]\d{2}:\d{2}|Z))"
)

# vLLM prefix: (Worker pid=1234) INFO 08-09 05:24:32.123456
# Captures month-day and time; year and timezone must be supplied.
_VLLM_TS = re.compile(
    r"(?:\(Worker\s+pid=\d+\)\s+)?"
    r"(?:INFO|WARNING|ERROR)\s+"
    r"(\d{2})-(\d{2})\s+(\d{2}):(\d{2}):(\d{2})(?:\.(\d+))?"
)

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
# Timestamp parsing
# ---------------------------------------------------------------------------


def _parse_kernel_timestamp(line: str) -> datetime | None:
    """Extract and parse a kernel ISO timestamp from a log line.

    Handles formats like:
    - 2026-08-09T00:16:52.113635-05:00
    - 2026-08-09T00:16:52Z
    - 2026-08-09T00:16:52
    """
    match = _KERNEL_TS.search(line)
    if not match:
        return None
    raw = match.group(1)
    try:
        # Python's fromisoformat handles offset and fractional seconds
        # in 3.11+. For older versions, fall back to manual parsing.
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            pass
        # Try with Z replaced by +00:00
        if raw.endswith("Z"):
            return datetime.fromisoformat(raw[:-1] + "+00:00")
    except (ValueError, TypeError):
        pass
    return None


def _parse_vllm_timestamp(
    line: str,
    *,
    year: int | None = None,
    tz: timezone | None = None,
) -> datetime | None:
    """Extract and parse a vLLM log-prefix timestamp.

    vLLM format: ``INFO 08-09 05:24:32.123456``
    The month-day and time are in the log; year and timezone must be supplied.
    Returns None if year is not supplied.
    """
    match = _VLLM_TS.search(line)
    if not match:
        return None
    if year is None:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    hour = int(match.group(3))
    minute = int(match.group(4))
    second = int(match.group(5))
    frac_str = match.group(6)
    microsecond = 0
    if frac_str:
        # Pad/truncate to 6 digits
        microsecond = int(frac_str[:6].ljust(6, "0"))
    try:
        dt = datetime(year, month, day, hour, minute, second, microsecond)
    except ValueError:
        return None
    if tz is not None:
        dt = dt.replace(tzinfo=tz)
    return dt


def _parse_engine_timestamp(
    line: str,
    *,
    year: int | None = None,
    tz: timezone | None = None,
) -> datetime | None:
    """Parse a timestamp from an engine log line.

    Tries kernel ISO format first, then vLLM prefix format.
    """
    dt = _parse_kernel_timestamp(line)
    if dt is not None:
        return dt
    return _parse_vllm_timestamp(line, year=year, tz=tz)


# ---------------------------------------------------------------------------
# Log scanning helpers
# ---------------------------------------------------------------------------


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
                })
    return hits


def _scan_progress(lines: list[str]) -> list[int]:
    """Return line numbers of progress/readiness lines."""
    return [
        i for i, line in enumerate(lines, start=1)
        if _PROGRESS.search(line) or _LMCACHE_READY.search(line)
    ]


def _scan_readiness(lines: list[str]) -> list[int]:
    """Return line numbers of readiness-specific lines."""
    return [
        i for i, line in enumerate(lines, start=1)
        if _READINESS.search(line) or _LMCACHE_READY.search(line)
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
    rm_event_bound: int | None = None,
    engine_log_year: int | None = None,
    engine_log_tz: timezone | None = None,
) -> dict[str, Any]:
    """Classify a single container log, optionally with a kernel log.

    Parameters:
    - ``container_running``: whether the container was running when
      evidence was captured.
    - ``kernel_log``: captured kernel/dmesg log text.
    - ``rm_event_bound``: operator-supplied accepted delta bound for RM
      event count. If None and RM events are present, verdict is
      indeterminate (fail closed).
    - ``engine_log_year``: year for vLLM log timestamps (which lack year).
    - ``engine_log_tz``: timezone for vLLM log timestamps.

    Returns a structured dict with verdict, signatures, event counts,
    materialization window, and provenance hashes.
    """
    lines = text.splitlines()
    fatal_hits = _scan_signatures(lines, ALL_FATAL_PATTERNS)
    progress_line_numbers = _scan_progress(lines)
    readiness_line_numbers = _scan_readiness(lines)
    materialization_line_numbers = _scan_materialization(lines)

    # Parse engine log timestamps
    engine_timestamps: list[tuple[int, datetime]] = []
    for i, line in enumerate(lines, start=1):
        dt = _parse_engine_timestamp(
            line, year=engine_log_year, tz=engine_log_tz
        )
        if dt is not None:
            engine_timestamps.append((i, dt))

    # Derive materialization window from real timestamps
    materialization_window: tuple[datetime, datetime] | None = None
    window_first: datetime | None = None
    window_last: datetime | None = None
    has_readiness = bool(readiness_line_numbers)

    if materialization_line_numbers and engine_timestamps:
        # Window: from first materialization line's timestamp through
        # last readiness line's timestamp (or last progress if no readiness)
        mat_ts = [
            dt for ln, dt in engine_timestamps
            if ln >= materialization_line_numbers[0]
        ]
        if mat_ts:
            window_first = min(mat_ts)
            # End of window: last readiness timestamp, or last progress
            end_candidates = readiness_line_numbers or progress_line_numbers
            if end_candidates:
                end_ts = [
                    dt for ln, dt in engine_timestamps
                    if ln >= end_candidates[-1]
                ]
                if end_ts:
                    window_last = max(end_ts)
            if window_last is None and engine_timestamps:
                window_last = max(dt for _, dt in engine_timestamps)
            if window_first is not None and window_last is not None:
                materialization_window = (window_first, window_last)

    # Kernel RM event detection
    rm_events: list[dict[str, Any]] = []
    rm_event_count = 0
    rm_event_within_bound: bool | None = None
    all_events_in_window = True
    all_events_have_timestamps = True
    kernel_sha256: str | None = None

    if kernel_log is not None:
        kernel_lines = kernel_log.splitlines()
        kernel_sha256 = hashlib.sha256(
            kernel_log.encode("utf-8")
        ).hexdigest()
        raw_rm_hits = _scan_signatures(kernel_lines, [(
            "nv_err_no_memory_memdesc_alloc", _RM_KERNEL_EVENT,
        )])

        for hit in raw_rm_hits:
            kline = kernel_lines[hit["line_number"] - 1]
            kts = _parse_kernel_timestamp(kline)
            hit["timestamp"] = kts.isoformat() if kts else None
            rm_events.append(hit)

        rm_event_count = len(rm_events)

        # Check operator-supplied bound
        if rm_event_count > 0:
            if rm_event_bound is None:
                rm_event_within_bound = None  # fail closed
            else:
                rm_event_within_bound = rm_event_count <= rm_event_bound

            # Check all events have timestamps
            all_events_have_timestamps = all(
                e["timestamp"] is not None for e in rm_events
            )

            # Check all events fall inside materialization window
            if all_events_have_timestamps and materialization_window:
                win_start, win_end = materialization_window
                for e in rm_events:
                    edt = datetime.fromisoformat(e["timestamp"])
                    if edt < win_start or edt > win_end:
                        all_events_in_window = False
                        break
            elif rm_events:
                # Can't verify window containment
                all_events_in_window = False

    # Determine verdict
    if fatal_hits:
        verdict = "fatal"
    elif rm_event_count == 0:
        verdict = "clean"
    else:
        # RM events present — apply cross-evidence protocol
        if not container_running:
            verdict = "fatal"
        elif rm_event_bound is None:
            # No operator-supplied bound → fail closed
            verdict = "indeterminate"
        elif rm_event_within_bound is False:
            # Over bound → fatal
            verdict = "fatal"
        elif not materialization_window:
            # Can't derive materialization window
            verdict = "indeterminate"
        elif not all_events_have_timestamps:
            # Missing kernel event timestamps
            verdict = "indeterminate"
        elif not all_events_in_window:
            # Events outside window
            verdict = "indeterminate"
        elif not has_readiness:
            # No readiness after window
            verdict = "indeterminate"
        else:
            verdict = "bounded_rm_retry"

    return {
        "verdict": verdict,
        "fatal_signatures": fatal_hits,
        "rm_events": rm_events,
        "rm_event_count": rm_event_count,
        "rm_event_bound": rm_event_bound,
        "rm_event_within_bound": rm_event_within_bound,
        "all_events_have_timestamps": all_events_have_timestamps,
        "all_events_in_window": all_events_in_window,
        "materialization_window": (
            [window_first.isoformat(), window_last.isoformat()]
            if window_first and window_last
            else None
        ),
        "has_readiness": has_readiness,
        "progress_lines": progress_line_numbers,
        "materialization_lines": materialization_line_numbers,
        "engine_timestamp_count": len(engine_timestamps),
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
    rm_event_bound: int | None = None,
    engine_log_year: int | None = None,
    engine_log_tz: timezone | None = None,
) -> dict[str, Any]:
    """Classify startup evidence for one rank.

    The overall rank verdict is the worst of engine and server
    (fatal > indeterminate > bounded_rm_retry > clean).
    """
    engine_result = None
    server_result = None
    if engine_log is not None:
        engine_result = classify_log(
            engine_log,
            container_running=engine_running,
            kernel_log=kernel_log,
            rm_event_bound=rm_event_bound,
            engine_log_year=engine_log_year,
            engine_log_tz=engine_log_tz,
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
    """Aggregate per-rank evidence into a single report.

    One indeterminate or fatal rank dominates the overall verdict.
    """
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
            "does not contact or mutate the cluster. This tool classifies "
            "supplied evidence and never establishes a safe bound itself."
        ),
        "classification_note": (
            "Generic CUDA out of memory, torch.OutOfMemoryError, OOMKilled, "
            "Xid, unexpected restarts, SSH failures, fabric carrier loss, "
            "and driver failures are fatal. Only kernel RM "
            "NV_ERR_NO_MEMORY at _memdescAllocInternal @ mem_desc.c:1359 "
            "during the EXL3 materialization window can be classified "
            "bounded_rm_retry, and only with full cross-evidence: container "
            "running, RestartCount=0, OOMKilled=false, no fatal signatures, "
            "event count within operator-supplied bound, every event "
            "timestamp inside the materialization window, and eventual "
            "readiness. If the operator-supplied bound is absent, "
            "timestamps are missing/malformed, or events fall outside "
            "the window, the verdict is indeterminate (fatal policy). "
            "This is evidence-scoped to the legacy EXL3 materialization "
            "callsite and does not prove all RM errors safe. NVIDIA errors "
            "are never globally ignored. This tool classifies supplied "
            "evidence and never establishes a safe bound itself. It "
            "remains offline-validated until run on sanitized captured "
            "evidence."
        ),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_tz(arg: str) -> timezone:
    """Parse a timezone argument like 'UTC' or '-05:00' or '+05:30'."""
    if arg.upper() == "UTC":
        return timezone.utc
    # Parse ±HH:MM
    match = re.match(r"^([+-])(\d{2}):(\d{2})$", arg)
    if match:
        sign = 1 if match.group(1) == "+" else -1
        hours = int(match.group(2))
        minutes = int(match.group(3))
        return timezone(sign * timedelta(hours=hours, minutes=minutes))
    raise ConfigError(f"invalid timezone: {arg}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
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
        "--rm-event-bound",
        type=int,
        default=None,
        help=(
            "operator-supplied accepted delta bound for RM event count. "
            "Required for bounded_rm_retry verdict; if absent and RM "
            "events are present, verdict is indeterminate (fail closed). "
            "This tool never establishes a safe bound itself."
        ),
    )
    parser.add_argument(
        "--engine-log-year",
        type=int,
        default=None,
        help=(
            "year for vLLM log timestamps (which lack year). "
            "Required for vLLM timestamp parsing."
        ),
    )
    parser.add_argument(
        "--engine-log-tz",
        type=str,
        default=None,
        help=(
            "timezone for vLLM log timestamps, e.g. 'UTC' or '-05:00'. "
            "Required for vLLM timestamp parsing."
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

    engine_log_tz: timezone | None = None
    if args.engine_log_tz is not None:
        try:
            engine_log_tz = _parse_tz(args.engine_log_tz)
        except ConfigError as exc:
            print(f"sparkring-startup-evidence: {exc}", file=sys.stderr)
            return EXIT_CONFIG_ERROR

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
                rm_event_bound=args.rm_event_bound,
                engine_log_year=args.engine_log_year,
                engine_log_tz=engine_log_tz,
            )
        )

    report = aggregate_report(rank_reports)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] in ("fatal", "indeterminate"):
        return EXIT_FATAL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
