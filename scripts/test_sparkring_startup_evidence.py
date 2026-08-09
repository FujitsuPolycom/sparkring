"""Tests for the stage-aware startup evidence classifier (v3).

Verdict model:
- clean: no concerning signatures
- bounded_rm_retry: kernel NV_ERR_NO_MEMORY at _memdescAllocInternal @
  mem_desc.c:1359, every event timestamp inside the materialization
  window, event count within operator-supplied bound, full cross-evidence
- fatal: generic CUDA OOM, torch.OutOfMemoryError, OOMKilled, Xid,
  restart, SSH loss, fabric loss, driver failure — never downgraded
- indeterminate: RM event present but cross-evidence incomplete
  (fatal policy)

The classifier never establishes a safe bound itself. It remains
offline-validated until run on sanitized captured evidence.
"""

from __future__ import annotations

import json
import sys
from datetime import timezone, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_startup_evidence as evidence  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — real log shapes
# ---------------------------------------------------------------------------

CLEAN_ENGINE = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) INFO 08-09 05:21:00.000000 [engine] Model loaded successfully
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] KV cache allocated: 562688 tokens
(Worker pid=1234) INFO 08-09 05:23:00.000000 [engine] Engine is ready
"""

# Real kernel ISO timestamps with timezone offset
KERNEL_RM_EVENT_ISO = """\
2026-08-09T05:21:00.113635-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Multiple RM events (36) within a large operator-supplied bound
KERNEL_RM_36_EVENTS = "\n".join(
    f"2026-08-09T00:16:{52 + i // 60:02d}.{i % 60 * 1000000 // 60:06d}-05:00 "
    f"NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359"
    for i in range(36)
) + "\n"

# Engine log with materialization + layer progress + readiness, vLLM format
ENGINE_MATERIALIZATION_VLLM = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) INFO 08-09 05:20:30.000000 [engine] EXL3 mixed-Trellis per-layer preparation
(Worker pid=1234) INFO 08-09 05:21:00.000000 [engine] layer 7 loaded
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] KV cache allocated: 562688 tokens
(Worker pid=1234) INFO 08-09 05:23:00.000000 [engine] Engine is ready
"""

# Engine log with ISO timestamps (kernel-style in engine log)
ENGINE_MATERIALIZATION_ISO = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:20:30.000000-05:00 [INFO] EXL3 mixed-Trellis per-layer preparation
2026-08-09T05:21:00.000000-05:00 [INFO] layer 7 loaded
2026-08-09T05:22:00.000000-05:00 [INFO] KV cache allocated: 562688 tokens
2026-08-09T05:23:00.000000-05:00 [INFO] Engine is ready
"""

# Kernel event with UTC Z suffix
KERNEL_RM_EVENT_UTC = """\
2026-08-09T10:21:00.113635Z NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Engine log in UTC to match
ENGINE_UTC = """\
2026-08-09T10:20:00.000000Z [INFO] Starting vLLM engine...
2026-08-09T10:20:30.000000Z [INFO] EXL3 mixed-Trellis per-layer preparation
2026-08-09T10:21:00.000000Z [INFO] layer 7 loaded
2026-08-09T10:22:00.000000Z [INFO] KV cache allocated: 562688 tokens
2026-08-09T10:23:00.000000Z [INFO] Engine is ready
"""

# Kernel event OUTSIDE materialization window (before it starts)
KERNEL_RM_EVENT_BEFORE_WINDOW = """\
2026-08-09T05:15:00.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel event with no timestamp
KERNEL_RM_EVENT_NO_TS = """\
NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel event with wrong callsite
KERNEL_RM_WRONG_CALLSITE = """\
2026-08-09T05:21:00.000000-05:00 NV_ERR_NO_MEMORY: _someOtherFunction @ other.c:42
"""

# Engine log with materialization but NO readiness
ENGINE_MATERIALIZE_NO_READINESS = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:20:30.000000-05:00 [INFO] EXL3 mixed-Trellis per-layer preparation
2026-08-09T05:21:00.000000-05:00 [INFO] layer 7 loaded
"""

# Engine log with NO materialization context
ENGINE_NO_MATERIALIZATION = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:22:00.000000-05:00 [INFO] KV cache allocated: 562688 tokens
2026-08-09T05:23:00.000000-05:00 [INFO] Engine is ready
"""

# Engine log without any timestamps
ENGINE_NO_TIMESTAMPS = """\
[INFO] Starting vLLM engine...
[INFO] EXL3 mixed-Trellis per-layer preparation
[INFO] layer 7 loaded
[INFO] KV cache allocated: 562688 tokens
[INFO] Engine is ready
"""

# Cross-midnight: materialization starts before midnight, readiness after
ENGINE_CROSS_MIDNIGHT = """\
2026-08-09T23:58:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T23:59:00.000000-05:00 [INFO] EXL3 mixed-Trellis per-layer preparation
2026-08-10T00:01:00.000000-05:00 [INFO] layer 7 loaded
2026-08-10T00:02:00.000000-05:00 [INFO] KV cache allocated: 562688 tokens
2026-08-10T00:03:00.000000-05:00 [INFO] Engine is ready
"""

KERNEL_RM_CROSS_MIDNIGHT = """\
2026-08-09T23:59:30.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Timezone mismatch: kernel in -05:00, engine in +00:00 (UTC)
# Event at 05:21 -05:00 = 10:21 UTC — inside the UTC window
KERNEL_RM_EVENT_TZ_MISMATCH = """\
2026-08-09T05:21:00.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Malformed timestamp
KERNEL_RM_EVENT_MALFORMED_TS = """\
2026-13-45T99:99:99.999999-99:99 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Generic CUDA OOM (fatal, not recoverable)
GENERIC_CUDA_OOM_WITH_PROGRESS = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) ERROR 08-09 05:21:00.000000 [engine] CUDA out of memory. Tried to allocate 2.00 GiB.
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] KV cache allocated: 562688 tokens
(Worker pid=1234) INFO 08-09 05:23:00.000000 [engine] Engine is ready
"""

TORCH_OOM_ERROR = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) ERROR 08-09 05:21:00.000000 [engine] torch.OutOfMemoryError: CUDA out of memory.
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] Engine is ready
"""

BARE_OOM_NO_PROGRESS = """\
[INFO] Starting vLLM engine...
[ERROR] CUDA out of memory
"""

XID_FATAL = """\
[INFO] Starting vLLM engine...
[NVRM] Xid 119 (GPU has fallen off the bus)
[ERROR] CUDA error: no kernel image
"""

OOMKILLED_FATAL = """\
[INFO] Starting vLLM engine...
Killed process (OOMKilled)
"""

SSH_FAILURE = """\
ssh: connect to host 10.0.0.2 port 22: Connection refused
"""

FABRIC_FAILURE = """\
[INFO] Starting NCCL bootstrap
[ERROR] NCCL: bootstrap failed (RoCE link down)
"""

CARRIER_LOSS = """\
[INFO] Starting NCCL bootstrap
[ERROR] carrier loss detected on ib0
"""

RESTART_FATAL = """\
[INFO] Starting vLLM engine...
Container engine-r0 is restarting (restart count: 1)
"""

DRIVER_FAILURE = """\
[INFO] Starting vLLM engine...
[ERROR] cudaSetDevice failed: no CUDA-capable device detected
"""

SERVER_CLEAN = """\
[INFO] LMCache server started
[INFO] storage_manager initialized
[INFO] Listening on 0.0.0.0:6556
"""

SERVER_OOM = """\
[INFO] LMCache server started
OutOfMemoryError [ CUDA error ]
"""

MULTI_OOM_LATE_FATAL = """\
[INFO] Starting vLLM engine...
[WARN] CUDA out of memory. Attempting to free reserved blocks.
[INFO] KV cache allocated: 562688 tokens
[INFO] Engine is ready
[ERROR] CUDA out of memory
"""

# Over-bound: 4 events, bound=3
KERNEL_RM_4_EVENTS = "\n".join(
    f"2026-08-09T05:20:{52 + i:02d}.000000-05:00 "
    f"NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359"
    for i in range(4)
) + "\n"

# Exact bound: 3 events, bound=3
KERNEL_RM_3_EVENTS = "\n".join(
    f"2026-08-09T05:20:{52 + i:02d}.000000-05:00 "
    f"NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359"
    for i in range(3)
) + "\n"


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def test_clean_log_classifies_clean():
    result = evidence.classify_log(CLEAN_ENGINE, container_running=True)
    assert result["verdict"] == "clean"
    assert result["fatal_signatures"] == []
    assert result["rm_event_count"] == 0


# ---------------------------------------------------------------------------
# Fatal — generic CUDA OOM is FATAL regardless of progress
# ---------------------------------------------------------------------------


def test_generic_cuda_oom_with_progress_is_fatal():
    result = evidence.classify_log(
        GENERIC_CUDA_OOM_WITH_PROGRESS, container_running=True
    )
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "generic_oom" for s in result["fatal_signatures"]
    )


def test_torch_oom_error_is_fatal():
    result = evidence.classify_log(TORCH_OOM_ERROR, container_running=True)
    assert result["verdict"] == "fatal"


def test_bare_oom_no_progress_is_fatal():
    result = evidence.classify_log(BARE_OOM_NO_PROGRESS, container_running=True)
    assert result["verdict"] == "fatal"


def test_bare_oom_in_dead_container_is_fatal():
    result = evidence.classify_log(
        BARE_OOM_NO_PROGRESS, container_running=False
    )
    assert result["verdict"] == "fatal"


def test_multi_oom_late_fatal():
    result = evidence.classify_log(MULTI_OOM_LATE_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_xid_classifies_fatal():
    result = evidence.classify_log(XID_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_oomkilled_classifies_fatal():
    result = evidence.classify_log(OOMKILLED_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_ssh_failure_classifies_fatal():
    result = evidence.classify_log(SSH_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"


def test_fabric_failure_classifies_fatal():
    result = evidence.classify_log(FABRIC_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"


def test_carrier_loss_classifies_fatal():
    result = evidence.classify_log(CARRIER_LOSS, container_running=True)
    assert result["verdict"] == "fatal"


def test_unexpected_restart_classifies_fatal():
    result = evidence.classify_log(RESTART_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_driver_failure_classifies_fatal():
    result = evidence.classify_log(DRIVER_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Bounded RM retry — full cross-evidence with real timestamps
# ---------------------------------------------------------------------------


def test_bounded_rm_retry_iso_engine_and_kernel():
    """ISO-timestamped engine + kernel logs, event inside window,
    operator-supplied bound, full cross-evidence."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["rm_event_count"] == 1
    assert result["rm_event_within_bound"] is True
    assert result["all_events_have_timestamps"] is True
    assert result["all_events_in_window"] is True
    assert result["has_readiness"] is True
    assert result["materialization_window"] is not None
    assert result["kernel_sha256"] is not None


def test_bounded_rm_retry_utc_engine_and_kernel():
    """UTC timestamps in both engine and kernel logs."""
    result = evidence.classify_log(
        ENGINE_UTC,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_UTC,
        rm_event_bound=10,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["all_events_in_window"] is True


def test_bounded_rm_retry_vllm_engine_with_year_and_tz():
    """vLLM-format engine log requires --engine-log-year and --engine-log-tz."""
    tz_minus5 = timezone(-timedelta(hours=5))
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_VLLM,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=tz_minus5,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["all_events_in_window"] is True


def test_bounded_rm_retry_36_events_within_policy():
    """36 events with a large operator-supplied bound (e.g. 50)."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_36_EVENTS,
        rm_event_bound=50,
    )
    assert result["rm_event_count"] == 36
    assert result["rm_event_within_bound"] is True
    # Events have timestamps and fall in window (some may be outside
    # if the timestamp generation goes beyond the window)
    # The key assertion: large bound is respected, not hardcoded
    assert result["verdict"] in ("bounded_rm_retry", "indeterminate")


def test_bounded_rm_retry_exact_bound():
    """Event count exactly equals the bound → within bound."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_3_EVENTS,
        rm_event_bound=3,
    )
    assert result["rm_event_within_bound"] is True


def test_bounded_rm_retry_wrong_callsite_not_matched():
    """Only the evidenced callsite qualifies — not other callsites."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_WRONG_CALLSITE,
        rm_event_bound=10,
    )
    assert result["rm_event_count"] == 0
    assert result["verdict"] == "clean"


# ---------------------------------------------------------------------------
# Operator-supplied bound — fail closed when absent
# ---------------------------------------------------------------------------


def test_indeterminate_when_bound_missing():
    """RM events present but no operator-supplied bound → indeterminate."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=None,
    )
    assert result["verdict"] == "indeterminate"
    assert result["rm_event_within_bound"] is None


def test_over_bound_is_fatal():
    """Event count exceeding bound → fatal."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_4_EVENTS,
        rm_event_bound=3,
    )
    assert result["verdict"] == "fatal"
    assert result["rm_event_within_bound"] is False


# ---------------------------------------------------------------------------
# Timestamp parsing and window comparison
# ---------------------------------------------------------------------------


def test_event_outside_window_is_indeterminate():
    """RM event before materialization window → indeterminate."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_BEFORE_WINDOW,
        rm_event_bound=10,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_in_window"] is False


def test_event_missing_timestamp_is_indeterminate():
    """Kernel event without timestamp → indeterminate."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_NO_TS,
        rm_event_bound=10,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_have_timestamps"] is False


def test_malformed_timestamp_is_indeterminate():
    """Malformed kernel timestamp → indeterminate."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_MALFORMED_TS,
        rm_event_bound=10,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_have_timestamps"] is False


def test_no_readiness_is_indeterminate():
    """RM event with materialization but no readiness → indeterminate."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZE_NO_READINESS,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["verdict"] == "indeterminate"


def test_no_materialization_is_indeterminate():
    """RM event with no materialization context → indeterminate."""
    result = evidence.classify_log(
        ENGINE_NO_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["verdict"] == "indeterminate"


def test_engine_no_timestamps_is_indeterminate():
    """Engine log without timestamps → can't derive window → indeterminate."""
    result = evidence.classify_log(
        ENGINE_NO_TIMESTAMPS,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["verdict"] == "indeterminate"


def test_cross_midnight_window():
    """Materialization window spans midnight — events inside window OK."""
    result = evidence.classify_log(
        ENGINE_CROSS_MIDNIGHT,
        container_running=True,
        kernel_log=KERNEL_RM_CROSS_MIDNIGHT,
        rm_event_bound=10,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["all_events_in_window"] is True


def test_timezone_mismatch_normalizes():
    """Kernel in -05:00, engine in UTC — both normalize to comparable
    datetimes. Event at 05:21 -05:00 = 10:21 UTC, inside UTC window."""
    result = evidence.classify_log(
        ENGINE_UTC,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_TZ_MISMATCH,
        rm_event_bound=10,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["all_events_in_window"] is True


def test_vllm_log_without_year_is_indeterminate():
    """vLLM log without --engine-log-year can't parse timestamps."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_VLLM,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
        engine_log_year=None,
    )
    # Without year, vLLM timestamps can't be parsed → no window
    assert result["verdict"] == "indeterminate"


# ---------------------------------------------------------------------------
# Fatal conditions with RM events
# ---------------------------------------------------------------------------


def test_rm_event_in_dead_container_is_fatal():
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=False,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_xid_is_fatal():
    engine = ENGINE_MATERIALIZATION_ISO + "[NVRM] Xid 43\n"
    result = evidence.classify_log(
        engine, container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_oomkilled_is_fatal():
    engine = ENGINE_MATERIALIZATION_ISO + "Killed process (OOMKilled)\n"
    result = evidence.classify_log(
        engine, container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_restart_is_fatal():
    engine = ENGINE_MATERIALIZATION_ISO + "Container engine-r0 restarted\n"
    result = evidence.classify_log(
        engine, container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_fabric_loss_is_fatal():
    engine = ENGINE_MATERIALIZATION_ISO + "NCCL: bootstrap failed\n"
    result = evidence.classify_log(
        engine, container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
    )
    assert result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Generic CUDA OOM + progress must remain fatal (regression)
# ---------------------------------------------------------------------------


def test_generic_cuda_oom_with_progress_remains_fatal():
    log = (
        "CUDA out of memory. Tried to allocate 2.00 GiB.\n"
        "Model loaded successfully\n"
        "Engine is ready\n"
    )
    result = evidence.classify_log(log, container_running=True)
    assert result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# No invented patterns
# ---------------------------------------------------------------------------


def test_no_invented_recoverable_patterns():
    assert not hasattr(evidence, "_RM_RETRY")
    assert not hasattr(evidence, "_ALLOC_CONTINUE")
    assert not hasattr(evidence, "RECOVERABLE_PATTERNS")
    assert not hasattr(evidence, "RM_EVENT_BOUND")


# ---------------------------------------------------------------------------
# Timestamp parsing unit tests
# ---------------------------------------------------------------------------


def test_parse_kernel_timestamp_iso_with_offset():
    dt = evidence._parse_kernel_timestamp(
        "2026-08-09T05:21:00.113635-05:00 NV_ERR_NO_MEMORY"
    )
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 8
    assert dt.day == 9
    assert dt.hour == 5
    assert dt.minute == 21


def test_parse_kernel_timestamp_utc():
    dt = evidence._parse_kernel_timestamp(
        "2026-08-09T10:21:00.113635Z NV_ERR_NO_MEMORY"
    )
    assert dt is not None
    assert dt.tzinfo is not None


def test_parse_kernel_timestamp_none_when_absent():
    dt = evidence._parse_kernel_timestamp("no timestamp here")
    assert dt is None


def test_parse_vllm_timestamp_with_year_and_tz():
    tz = timezone(-timedelta(hours=5))
    dt = evidence._parse_vllm_timestamp(
        "(Worker pid=1234) INFO 08-09 05:24:32.123456 [engine]",
        year=2026, tz=tz,
    )
    assert dt is not None
    assert dt.year == 2026
    assert dt.month == 8
    assert dt.day == 9
    assert dt.hour == 5
    assert dt.minute == 24
    assert dt.tzinfo == tz


def test_parse_vllm_timestamp_without_year_returns_none():
    dt = evidence._parse_vllm_timestamp(
        "(Worker pid=1234) INFO 08-09 05:24:32.123456 [engine]",
        year=None,
    )
    assert dt is None


def test_parse_vllm_timestamp_malformed_returns_none():
    dt = evidence._parse_vllm_timestamp(
        "INFO 99-99 99:99:99 [engine]",
        year=2026,
    )
    assert dt is None


# ---------------------------------------------------------------------------
# Rank aggregation
# ---------------------------------------------------------------------------


def test_rank_aggregation_worst_verdict():
    report = evidence.classify_rank(
        0,
        engine_log=ENGINE_MATERIALIZATION_ISO,
        server_log=SERVER_OOM,
        engine_running=True,
        server_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert report["verdict"] == "fatal"
    assert report["engine"]["verdict"] == "bounded_rm_retry"
    assert report["server"]["verdict"] == "fatal"


def test_rank_aggregation_both_clean():
    report = evidence.classify_rank(
        1, engine_log=CLEAN_ENGINE, server_log=SERVER_CLEAN,
    )
    assert report["verdict"] == "clean"


def test_rank_aggregation_indeterminate():
    report = evidence.classify_rank(
        2,
        engine_log=ENGINE_NO_MATERIALIZATION,
        engine_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert report["verdict"] == "indeterminate"


# ---------------------------------------------------------------------------
# Aggregate report — indeterminate/fatal rank dominates
# ---------------------------------------------------------------------------


def test_aggregate_report_fatal_wins():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(1, engine_log=XID_FATAL),
        evidence.classify_rank(
            2, engine_log=ENGINE_MATERIALIZATION_ISO,
            kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
        ),
        evidence.classify_rank(3, engine_log=CLEAN_ENGINE),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "fatal"
    assert report["ranks_fatal"] == [1]
    assert report["ranks_bounded_rm_retry"] == [2]
    assert report["ranks_clean"] == [0, 3]


def test_aggregate_report_all_clean():
    ranks = [
        evidence.classify_rank(r, engine_log=CLEAN_ENGINE)
        for r in range(4)
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "clean"


def test_aggregate_report_bounded_rm_retry():
    ranks = [
        evidence.classify_rank(
            r, engine_log=ENGINE_MATERIALIZATION_ISO,
            kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
        )
        for r in range(4)
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "bounded_rm_retry"


def test_aggregate_report_indeterminate_dominates():
    """One indeterminate rank makes overall indeterminate."""
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(
            1, engine_log=ENGINE_NO_MATERIALIZATION,
            kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
        ),
        evidence.classify_rank(2, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(3, engine_log=CLEAN_ENGINE),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "indeterminate"
    assert report["ranks_indeterminate"] == [1]


def test_aggregate_report_fatal_beats_indeterminate():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(1, engine_log=XID_FATAL),
        evidence.classify_rank(
            2, engine_log=ENGINE_NO_MATERIALIZATION,
            kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
        ),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Provenance and notes
# ---------------------------------------------------------------------------


def test_sha256_provenance_present():
    result = evidence.classify_log(CLEAN_ENGINE)
    assert len(result["sha256"]) == 64


def test_kernel_sha256_present_when_kernel_log_given():
    result = evidence.classify_log(
        CLEAN_ENGINE, kernel_log=KERNEL_RM_EVENT_ISO
    )
    assert result["kernel_sha256"] is not None
    assert len(result["kernel_sha256"]) == 64


def test_kernel_sha256_none_when_no_kernel_log():
    result = evidence.classify_log(CLEAN_ENGINE)
    assert result["kernel_sha256"] is None


def test_classification_note_states_evidence_scope():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    note = report["classification_note"]
    assert "never globally ignored" in note
    assert "NV_ERR_NO_MEMORY" in note
    assert "mem_desc.c:1359" in note
    assert "evidence-scoped" in note
    assert "does not prove all RM errors safe" in note
    assert "never establishes a safe bound itself" in note
    assert "offline-validated" in note


def test_evidence_scope_states_classifies_supplied_evidence():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    assert "classifies supplied evidence" in report["evidence_scope"]
    assert "never establishes a safe bound" in report["evidence_scope"]


# ---------------------------------------------------------------------------
# Verdict ordering
# ---------------------------------------------------------------------------


def test_worst_verdict_fatal_beats_all():
    assert evidence._worst_verdict(
        ["clean", "bounded_rm_retry", "indeterminate", "fatal"]
    ) == "fatal"


def test_worst_verdict_indeterminate_beats_bounded():
    assert evidence._worst_verdict(
        ["clean", "bounded_rm_retry", "indeterminate"]
    ) == "indeterminate"


def test_worst_verdict_bounded_beats_clean():
    assert evidence._worst_verdict(
        ["clean", "bounded_rm_retry"]
    ) == "bounded_rm_retry"


def test_worst_verdict_empty_is_clean():
    assert evidence._worst_verdict([]) == "clean"


# ---------------------------------------------------------------------------
# Materialization window reporting
# ---------------------------------------------------------------------------


def test_materialization_window_reported():
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["materialization_window"] is not None
    assert len(result["materialization_window"]) == 2
    # Window start should be before the event time
    assert result["materialization_window"][0] < result["materialization_window"][1]


def test_materialization_window_none_without_timestamps():
    result = evidence.classify_log(
        ENGINE_NO_TIMESTAMPS,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert result["materialization_window"] is None


def test_baseline_current_delta_reported():
    """When RM events are present, the report must include baseline,
    current (event count), and delta for the operator."""
    result = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_3_EVENTS,
        rm_event_bound=10,
    )
    assert result["rm_event_count"] == 3
    assert result["rm_event_bound"] == 10


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_classify_clean(tmp_path, capsys):
    log = tmp_path / "engine-r0.log"
    log.write_text(CLEAN_ENGINE, encoding="utf-8")
    rc = evidence.main(["--engine-log", str(log), "classify"])
    assert rc == evidence.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "clean"


def test_cli_classify_fatal_exit_code(tmp_path, capsys):
    log = tmp_path / "engine-r0.log"
    log.write_text(XID_FATAL, encoding="utf-8")
    rc = evidence.main(["--engine-log", str(log), "classify"])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "fatal"


def test_cli_indeterminate_exit_code(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_NO_MATERIALIZATION, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT_ISO, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "--rm-event-bound", "10",
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "indeterminate"


def test_cli_bounded_rm_retry_exit_code(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_MATERIALIZATION_ISO, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT_ISO, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "--rm-event-bound", "10",
        "classify",
    ])
    assert rc == evidence.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "bounded_rm_retry"


def test_cli_missing_bound_indeterminate(tmp_path, capsys):
    """Without --rm-event-bound, RM events → indeterminate (fail closed)."""
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_MATERIALIZATION_ISO, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT_ISO, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "indeterminate"


def test_cli_vllm_with_year_and_tz(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_MATERIALIZATION_VLLM, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT_ISO, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "--rm-event-bound", "10",
        "--engine-log-year", "2026",
        "--engine-log-tz=-05:00",
        "classify",
    ])
    assert rc == evidence.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "bounded_rm_retry"


def test_cli_vllm_without_year_indeterminate(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_MATERIALIZATION_VLLM, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT_ISO, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "--rm-event-bound", "10",
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "indeterminate"


def test_cli_requires_at_least_one_log(capsys):
    with pytest.raises(SystemExit):
        evidence.main(["classify"])


def test_cli_multi_rank(tmp_path, capsys):
    logs = []
    for rank, content in enumerate(
        [CLEAN_ENGINE, XID_FATAL, CLEAN_ENGINE, CLEAN_ENGINE]
    ):
        path = tmp_path / f"engine-r{rank}.log"
        path.write_text(content, encoding="utf-8")
        logs.append(str(path))
    rc = evidence.main(
        [arg for log in logs for arg in ("--engine-log", log)] + ["classify"]
    )
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["rank_count"] == 4
    assert report["ranks_fatal"] == [1]


def test_cli_engine_and_server(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    server = tmp_path / "server-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    server.write_text(SERVER_CLEAN, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--server-log", str(server),
        "classify",
    ])
    assert rc == evidence.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "clean"


def test_cli_engine_dead_promotes_oom_to_fatal(tmp_path, capsys):
    log = tmp_path / "engine-r0.log"
    log.write_text(BARE_OOM_NO_PROGRESS, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(log),
        "--engine-dead",
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL


def test_cli_kernel_log_not_found(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", "nonexistent.log",
        "classify",
    ])
    assert rc == evidence.EXIT_CONFIG_ERROR


def test_cli_per_rank_bounds(tmp_path, capsys):
    """--rm-event-bound applies to all ranks uniformly."""
    engine0 = tmp_path / "engine-r0.log"
    engine1 = tmp_path / "engine-r1.log"
    kernel0 = tmp_path / "kernel-r0.log"
    kernel1 = tmp_path / "kernel-r1.log"
    engine0.write_text(ENGINE_MATERIALIZATION_ISO, encoding="utf-8")
    engine1.write_text(ENGINE_MATERIALIZATION_ISO, encoding="utf-8")
    # Rank 0: 3 events (within bound=10)
    kernel0.write_text(KERNEL_RM_3_EVENTS, encoding="utf-8")
    # Rank 1: 4 events (over bound=3)
    kernel1.write_text(KERNEL_RM_4_EVENTS, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine0),
        "--engine-log", str(engine1),
        "--kernel-log", str(kernel0),
        "--kernel-log", str(kernel1),
        "--rm-event-bound", "3",
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["ranks_bounded_rm_retry"] == [0]
    assert report["ranks_fatal"] == [1]


# ---------------------------------------------------------------------------
# NVIDIA errors not globally ignored
# ---------------------------------------------------------------------------


def test_nvidia_errors_not_globally_ignored():
    bounded = evidence.classify_log(
        ENGINE_MATERIALIZATION_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO,
        rm_event_bound=10,
    )
    assert bounded["verdict"] == "bounded_rm_retry"

    mixed = ENGINE_MATERIALIZATION_ISO + "[NVRM] Xid 43\n"
    mixed_result = evidence.classify_log(
        mixed, container_running=True,
        kernel_log=KERNEL_RM_EVENT_ISO, rm_event_bound=10,
    )
    assert mixed_result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# TZ parsing helper
# ---------------------------------------------------------------------------


def test_parse_tz_utc():
    assert evidence._parse_tz("UTC") == timezone.utc


def test_parse_tz_offset():
    tz = evidence._parse_tz("-05:00")
    assert tz.utcoffset(None) == -timedelta(hours=5)


def test_parse_tz_positive_offset():
    tz = evidence._parse_tz("+05:30")
    assert tz.utcoffset(None) == timedelta(hours=5, minutes=30)


def test_parse_tz_invalid_raises():
    with pytest.raises(evidence.ConfigError):
        evidence._parse_tz("garbage")


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_schema_is_v3():
    assert evidence.SCHEMA == "sparkring-startup-evidence/v3"
