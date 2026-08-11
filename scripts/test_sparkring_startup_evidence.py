"""Tests for the stage-aware startup evidence classifier (v1).

Verdict model:
- clean: no concerning signatures
- bounded_rm_retry: kernel NV_ERR_NO_MEMORY at _memdescAllocInternal @
  mem_desc.c:1359, every event timestamp inside the per-layer
  materialization window, post-materialization success milestone after
  window, cluster/API readiness supplied, event count within
  operator-supplied bound, full cross-evidence
- fatal: generic CUDA OOM, torch.OutOfMemoryError, OOMKilled, Xid,
  restart, SSH loss, fabric loss, driver failure — never downgraded
- indeterminate: RM event present but cross-evidence incomplete
  (fatal policy)

The classifier never establishes a safe bound itself. window_event_count
is a count, not a delta. It remains offline-validated until run on
sanitized captured evidence.
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

# Per-layer materialization lines — the evidenced shape
ENGINE_PER_LAYER_MATERIALIZATION = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) INFO 08-09 05:20:10.000000 [engine] EXL3 mixed Trellis model.layers.0.mlp.experts
(Worker pid=1234) INFO 08-09 05:20:30.000000 [engine] EXL3 mixed Trellis model.layers.1.mlp.experts
(Worker pid=1234) INFO 08-09 05:21:00.000000 [engine] Graph capturing finished
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] Engine is ready
"""

# ISO timestamp version
ENGINE_PER_LAYER_ISO = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:20:10.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T05:20:30.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.1.mlp.experts
2026-08-09T05:21:00.000000-05:00 [INFO] Graph capturing finished
2026-08-09T05:22:00.000000-05:00 [INFO] Engine is ready
"""

# Headless rank (1-3) — no API strings, has graph-finished milestone
ENGINE_HEADLESS = """\
(Worker_TP1_DCP1) INFO 08-09 05:20:00.000000 Starting vLLM engine...
(Worker_TP1_DCP1) INFO 08-09 05:20:10.000000 EXL3 mixed Trellis model.layers.0.mlp.experts
(Worker_TP1_DCP1) INFO 08-09 05:20:30.000000 EXL3 mixed Trellis model.layers.1.mlp.experts
(Worker_TP1_DCP1) INFO 08-09 05:21:00.000000 Graph capturing finished
"""

# Headless rank with Kernel JIT monitor milestone
ENGINE_HEADLESS_JIT = """\
(Worker_TP2_DCP2) INFO 08-09 05:20:00.000000 Starting vLLM engine...
(Worker_TP2_DCP2) INFO 08-09 05:20:10.000000 EXL3 mixed Trellis model.layers.0.mlp.experts
(Worker_TP2_DCP2) INFO 08-09 05:20:30.000000 EXL3 mixed Trellis model.layers.1.mlp.experts
(Worker_TP2_DCP2) INFO 08-09 05:21:00.000000 Kernel JIT monitor activated
"""

# Rank 0 with API readiness
ENGINE_RANK0_API = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) INFO 08-09 05:20:10.000000 [engine] EXL3 mixed Trellis model.layers.0.mlp.experts
(Worker pid=1234) INFO 08-09 05:20:30.000000 [engine] EXL3 mixed Trellis model.layers.1.mlp.experts
(Worker pid=1234) INFO 08-09 05:21:00.000000 [engine] Graph capturing finished
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] Engine is ready
(Worker pid=1234) INFO 08-09 05:22:30.000000 [engine] API server started
"""

# "mixed Trellis runtime planned" — must NOT extend the window
ENGINE_NON_PER_LAYER = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:20:10.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T05:20:30.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.1.mlp.experts
2026-08-09T05:25:00.000000-05:00 [INFO] mixed Trellis runtime planned
2026-08-09T05:26:00.000000-05:00 [INFO] Graph capturing finished
"""

# Kernel RM event inside window
KERNEL_RM_IN_WINDOW = """\
2026-08-09T05:20:15.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel RM event between last layer and graph capture (after window end)
KERNEL_RM_AFTER_WINDOW = """\
2026-08-09T05:20:45.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel RM event before materialization starts
KERNEL_RM_BEFORE_WINDOW = """\
2026-08-09T05:15:00.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel RM event with no timestamp
KERNEL_RM_NO_TS = """\
NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel RM event with wrong callsite
KERNEL_RM_WRONG_CALLSITE = """\
2026-08-09T05:20:15.000000-05:00 NV_ERR_NO_MEMORY: _someOtherFunction @ other.c:42
"""

# Multiple events within bound
KERNEL_RM_3_EVENTS = "\n".join(
    f"2026-08-09T05:20:{15 + i * 5:02d}.000000-05:00 "
    f"NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359"
    for i in range(3)
) + "\n"

# 36 events within a large bound
KERNEL_RM_36_EVENTS = "\n".join(
    f"2026-08-09T05:20:{15 + i:02d}.{i * 1000:06d}-05:00 "
    f"NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359"
    for i in range(36)
) + "\n"

# UTC versions
ENGINE_UTC = """\
2026-08-09T10:20:00.000000Z [INFO] Starting vLLM engine...
2026-08-09T10:20:10.000000Z [INFO] EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T10:20:30.000000Z [INFO] EXL3 mixed Trellis model.layers.1.mlp.experts
2026-08-09T10:21:00.000000Z [INFO] Graph capturing finished
2026-08-09T10:22:00.000000Z [INFO] Engine is ready
"""

KERNEL_RM_UTC = """\
2026-08-09T10:20:15.000000Z NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Cross-midnight
ENGINE_CROSS_MIDNIGHT = """\
2026-08-09T23:58:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T23:59:00.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-10T00:00:30.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.1.mlp.experts
2026-08-10T00:01:00.000000-05:00 [INFO] Graph capturing finished
2026-08-10T00:02:00.000000-05:00 [INFO] Engine is ready
"""

KERNEL_RM_CROSS_MIDNIGHT = """\
2026-08-09T23:59:30.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# TZ mismatch: kernel -05:00, engine UTC
KERNEL_RM_TZ_MISMATCH = """\
2026-08-09T05:20:15.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Malformed timestamp
KERNEL_RM_MALFORMED = """\
2026-13-45T99:99:99.999999-99:99 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Unrelated earlier boot RM line (before engine even starts)
KERNEL_RM_EARLY_BOOT = """\
2026-08-09T05:00:00.000000-05:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Materialization with NO post-materialization success milestone
ENGINE_NO_POST_MAT_SUCCESS = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:20:10.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T05:20:30.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.1.mlp.experts
"""

# Readiness BEFORE last materialization line
ENGINE_READINESS_BEFORE_LAST_LAYER = """\
2026-08-09T05:20:00.000000-05:00 [INFO] Starting vLLM engine...
2026-08-09T05:20:10.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T05:20:15.000000-05:00 [INFO] Graph capturing finished
2026-08-09T05:20:30.000000-05:00 [INFO] EXL3 mixed Trellis model.layers.1.mlp.experts
"""

# Fatal fixtures
GENERIC_CUDA_OOM = """\
(Worker pid=1234) INFO 08-09 05:20:00.000000 [engine] Starting vLLM engine...
(Worker pid=1234) ERROR 08-09 05:21:00.000000 [engine] CUDA out of memory. Tried to allocate 2.00 GiB.
(Worker pid=1234) INFO 08-09 05:22:00.000000 [engine] Engine is ready
"""

XID_FATAL = """\
[INFO] Starting vLLM engine...
[NVRM] Xid 119 (GPU has fallen off the bus)
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
[ERROR] carrier loss detected on ib0
"""

RESTART_FATAL = """\
[INFO] Starting vLLM engine...
Container engine-r0 is restarting (restart count: 1)
"""

DRIVER_FAILURE = """\
[ERROR] cudaSetDevice failed: no CUDA-capable device detected
"""

BARE_OOM = """\
[INFO] Starting vLLM engine...
[ERROR] CUDA out of memory
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


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def test_clean_log_classifies_clean():
    result = evidence.classify_log(CLEAN_ENGINE, container_running=True)
    assert result["verdict"] == "clean"


def test_stopped_container_is_fatal_without_log_signature():
    result = evidence.classify_log(CLEAN_ENGINE, container_running=False)
    assert result["verdict"] == "fatal"
    assert result["container_running"] is False
    assert result["fatal_signatures"] == []
    assert result["window_event_count"] == 0


# ---------------------------------------------------------------------------
# Fatal — generic CUDA OOM is FATAL regardless of progress
# ---------------------------------------------------------------------------


def test_generic_cuda_oom_with_progress_is_fatal():
    result = evidence.classify_log(GENERIC_CUDA_OOM, container_running=True)
    assert result["verdict"] == "fatal"


def test_bare_oom_is_fatal():
    result = evidence.classify_log(BARE_OOM, container_running=True)
    assert result["verdict"] == "fatal"


def test_bare_oom_dead_container_fatal():
    result = evidence.classify_log(BARE_OOM, container_running=False)
    assert result["verdict"] == "fatal"


def test_xid_fatal():
    result = evidence.classify_log(XID_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_oomkilled_fatal():
    result = evidence.classify_log(OOMKILLED_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_oomkilled_false_is_not_a_fatal_signature():
    result = evidence.classify_log('State: {"OOMKilled": false}')
    assert result["verdict"] == "clean"
    assert result["fatal_signatures"] == []


def test_ssh_failure_fatal():
    result = evidence.classify_log(SSH_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"


def test_fabric_failure_fatal():
    result = evidence.classify_log(FABRIC_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"


def test_carrier_loss_fatal():
    result = evidence.classify_log(CARRIER_LOSS, container_running=True)
    assert result["verdict"] == "fatal"


def test_restart_fatal():
    result = evidence.classify_log(RESTART_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


def test_driver_failure_fatal():
    result = evidence.classify_log(DRIVER_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Bounded RM retry — full cross-evidence
# ---------------------------------------------------------------------------


def test_bounded_rm_retry_iso_with_cluster_ready():
    """Full cross-evidence: per-layer window, event in window,
    post-materialization success, cluster ready, bound supplied."""
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["window_event_count"] == 1
    assert result["all_events_in_window"] is True
    assert result["readiness_after_window"] is True
    assert result["materialization_window"] is not None


def test_bounded_rm_retry_utc():
    result = evidence.classify_log(
        ENGINE_UTC,
        container_running=True,
        kernel_log=KERNEL_RM_UTC,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"


def test_bounded_rm_retry_vllm_with_year_tz():
    tz = timezone(-timedelta(hours=5))
    result = evidence.classify_log(
        ENGINE_PER_LAYER_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=tz,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"


def test_bounded_rm_retry_36_events():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_36_EVENTS,
        rm_event_bound=50,
        cluster_ready=True,
    )
    assert result["window_event_count"] == 36
    assert result["rm_event_within_bound"] is True
    assert result["verdict"] in ("bounded_rm_retry", "indeterminate")


def test_wrong_rm_callsite_is_indeterminate():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_WRONG_CALLSITE,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["window_event_count"] == 0
    assert result["verdict"] == "indeterminate"
    assert len(result["unknown_rm_events"]) == 1


def test_bounded_rm_retry_exact_bound():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_3_EVENTS,
        rm_event_bound=3,
        cluster_ready=True,
    )
    assert result["rm_event_within_bound"] is True


# ---------------------------------------------------------------------------
# cluster_ready required — LMCache readiness alone does not qualify
# ---------------------------------------------------------------------------


def test_bounded_rm_retry_requires_cluster_ready():
    """Without --cluster-ready, bounded_rm_retry cannot be reached."""
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=False,
    )
    assert result["verdict"] == "indeterminate"


def test_lmcache_server_readiness_alone_does_not_qualify():
    """LMCache server readiness must not classify a rank as bounded."""
    result = evidence.classify_log(
        SERVER_CLEAN,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=False,
    )
    # Server log with LMCache ready but no cluster_ready fact
    assert result["verdict"] != "bounded_rm_retry"


# ---------------------------------------------------------------------------
# Operator-supplied bound — fail closed when absent
# ---------------------------------------------------------------------------


def test_indeterminate_when_bound_missing():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=None,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["rm_event_within_bound"] is None


def test_over_bound_is_fatal():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_3_EVENTS,
        rm_event_bound=2,
        cluster_ready=True,
    )
    assert result["verdict"] == "fatal"
    assert result["rm_event_within_bound"] is False


# ---------------------------------------------------------------------------
# Window — per-layer materialization only
# ---------------------------------------------------------------------------


def test_non_per_layer_line_does_not_extend_window():
    """'mixed Trellis runtime planned' must NOT be a materialization line."""
    result = evidence.classify_log(
        ENGINE_NON_PER_LAYER,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    # The window must be from layers.0 through layers.1, not extended
    # by "mixed Trellis runtime planned" at 05:25
    window = result["materialization_window"]
    assert window is not None
    # Window end should be 05:20:30 (last model.layers line), not 05:25
    # Window end should be before "mixed Trellis runtime planned"
    # Both are ISO strings with tz; string comparison works for same tz
    assert "10:20:30" in window[1]  # UTC normalized from 05:20:30-05:00


def test_rm_after_window_is_indeterminate():
    """RM event between last layer and graph capture (after window end)
    must be indeterminate."""
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_AFTER_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_in_window"] is False


def test_rm_before_window_is_indeterminate():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_BEFORE_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_in_window"] is False


def test_rm_early_boot_is_indeterminate():
    """An unrelated RM event from earlier boot (before engine start)
    must be indeterminate."""
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_EARLY_BOOT,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_in_window"] is False


# ---------------------------------------------------------------------------
# Timestamp issues
# ---------------------------------------------------------------------------


def test_event_missing_timestamp_indeterminate():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_NO_TS,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"


def test_malformed_timestamp_indeterminate():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_MALFORMED,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["all_events_have_timestamps"] is False


def test_year_without_tz_no_crash_indeterminate():
    """Year supplied without timezone must not crash; must be indeterminate."""
    result = evidence.classify_log(
        ENGINE_PER_LAYER_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=None,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["year_without_tz"] is True

# Engine log with a materialization line that has a malformed vLLM timestamp
# (month 13, day 45). The regex matches but datetime() raises ValueError.
# The classifier must catch it and set timestamp_parse_error=True, producing
# indeterminate — not silently drop the timestamp and proceed.
ENGINE_MALFORMED_VLLM_TS_MAT = """\
(Worker pid=1234) INFO 13-45 05:20:15 EXL3 mixed Trellis model.layers.0.mlp.experts
(Worker pid=1234) INFO 08-09 05:20:25 EXL3 mixed Trellis model.layers.1.mlp.experts
(Worker pid=1234) INFO 08-09 05:20:35 Graph capturing finished
"""


def test_malformed_vllm_ts_on_mat_line_indeterminate():
    """A malformed vLLM timestamp on a materialization line must set
    timestamp_parse_error and produce indeterminate, not silently drop
    the timestamp."""
    from datetime import timezone, timedelta
    result = evidence.classify_log(
        ENGINE_MALFORMED_VLLM_TS_MAT,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=timezone(-timedelta(hours=5)),
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["timestamp_parse_error"] is True


def test_cross_midnight_window():
    result = evidence.classify_log(
        ENGINE_CROSS_MIDNIGHT,
        container_running=True,
        kernel_log=KERNEL_RM_CROSS_MIDNIGHT,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"


def test_tz_mismatch_normalizes():
    """Kernel -05:00, engine UTC — both normalize to UTC for comparison."""
    result = evidence.classify_log(
        ENGINE_UTC,
        container_running=True,
        kernel_log=KERNEL_RM_TZ_MISMATCH,
        rm_event_bound=10,
        cluster_ready=True,
    )
    # Event at 05:20:15-05:00 = 10:20:15 UTC, inside UTC window
    assert result["verdict"] == "bounded_rm_retry"


# ---------------------------------------------------------------------------
# Post-materialization success milestones
# ---------------------------------------------------------------------------


def test_no_post_materialization_success_indeterminate():
    """RM event with per-layer materialization but no success milestone
    after window → indeterminate."""
    result = evidence.classify_log(
        ENGINE_NO_POST_MAT_SUCCESS,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"


def test_readiness_before_last_layer_indeterminate():
    """Readiness milestone before the last materialization line → not
    after window → indeterminate."""
    result = evidence.classify_log(
        ENGINE_READINESS_BEFORE_LAST_LAYER,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"


# ---------------------------------------------------------------------------
# Headless rank success milestones
# ---------------------------------------------------------------------------


def test_headless_graph_finished_success():
    """Headless rank (1-3) with 'Graph capturing finished' milestone
    and cluster_ready → bounded_rm_retry."""
    tz = timezone(-timedelta(hours=5))
    result = evidence.classify_log(
        ENGINE_HEADLESS,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=tz,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["post_materialization_success"] is not None


def test_headless_jit_monitor_success():
    """Headless rank with 'Kernel JIT monitor activated' milestone."""
    tz = timezone(-timedelta(hours=5))
    result = evidence.classify_log(
        ENGINE_HEADLESS_JIT,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=tz,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"


def test_rank0_api_readiness_success():
    """Rank 0 with API server started → engine API readiness qualifies."""
    tz = timezone(-timedelta(hours=5))
    result = evidence.classify_log(
        ENGINE_RANK0_API,
        container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        engine_log_year=2026,
        engine_log_tz=tz,
        cluster_ready=True,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["engine_api_ready"] is not None


# ---------------------------------------------------------------------------
# RM events with fatal conditions
# ---------------------------------------------------------------------------


def test_rm_event_in_dead_container_fatal():
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=False,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_xid_fatal():
    engine = ENGINE_PER_LAYER_ISO + "[NVRM] Xid 43\n"
    result = evidence.classify_log(
        engine, container_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10, cluster_ready=True,
    )
    assert result["verdict"] == "fatal"


def test_kernel_xid_is_fatal_even_when_engine_log_is_clean():
    result = evidence.classify_log(
        CLEAN_ENGINE,
        kernel_log="2026-08-09T05:20:15-05:00 NVRM: Xid 79\n",
    )
    assert result["verdict"] == "fatal"
    assert any(
        hit["pattern"] == "xid" and hit["source"] == "kernel_log"
        for hit in result["fatal_signatures"]
    )


def test_unrecognized_kernel_rm_event_is_indeterminate():
    result = evidence.classify_log(
        CLEAN_ENGINE,
        kernel_log=(
            "2026-08-09T05:20:15-05:00 NV_ERR_NO_MEMORY "
            "at another_callsite.c:1\n"
        ),
    )
    assert result["verdict"] == "indeterminate"
    assert result["unknown_rm_events"][0]["source"] == "kernel_log"


def test_restart_count_zero_does_not_match_later_numeric_field():
    result = evidence.classify_log(
        '{"RestartCount":0,"ExitCode":1,"OOMKilled":false}'
    )
    assert result["verdict"] == "clean"


def test_torch_oom_is_counted_once():
    result = evidence.classify_log("torch.OutOfMemoryError: allocation failed")
    assert result["verdict"] == "fatal"
    assert len(result["fatal_signatures"]) == 1


def test_reordered_materialization_timestamps_are_indeterminate():
    engine = """\
2026-08-09T05:20:30-05:00 EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T05:20:10-05:00 EXL3 mixed Trellis model.layers.1.mlp.experts
2026-08-09T05:21:00-05:00 Graph capturing finished
"""
    result = evidence.classify_log(
        engine,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["materialization_timestamps_monotonic"] is False


def test_success_line_before_materialization_window_does_not_qualify():
    engine = """\
2026-08-09T05:21:00-05:00 Graph capturing finished
2026-08-09T05:20:10-05:00 EXL3 mixed Trellis model.layers.0.mlp.experts
2026-08-09T05:20:30-05:00 EXL3 mixed Trellis model.layers.1.mlp.experts
"""
    result = evidence.classify_log(
        engine,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert result["verdict"] == "indeterminate"
    assert result["post_materialization_success"] is None


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


def test_parse_kernel_timestamp_iso():
    dt = evidence._parse_kernel_timestamp(
        "2026-08-09T05:21:00.113635-05:00 NV_ERR_NO_MEMORY"
    )
    assert dt is not None
    assert dt.year == 2026
    assert dt.tzinfo is not None


def test_parse_kernel_timestamp_utc():
    dt = evidence._parse_kernel_timestamp(
        "2026-08-09T10:21:00.113635Z NV_ERR_NO_MEMORY"
    )
    assert dt is not None
    assert dt.tzinfo == timezone.utc


def test_parse_kernel_timestamp_none():
    assert evidence._parse_kernel_timestamp("no ts") is None


def test_parse_vllm_timestamp_with_year_tz():
    tz = timezone(-timedelta(hours=5))
    dt = evidence._parse_vllm_timestamp(
        "(Worker pid=1234) INFO 08-09 05:24:32.123456 [engine]",
        year=2026, tz=tz,
    )
    assert dt is not None
    assert dt.year == 2026
    assert dt.tzinfo == timezone.utc  # normalized to UTC


def test_parse_vllm_timestamp_no_year_none():
    assert evidence._parse_vllm_timestamp(
        "INFO 08-09 05:24:32 [engine]", year=None,
    ) is None


def test_parse_vllm_timestamp_no_tz_none():
    """Year supplied without tz → None (fail closed, no crash)."""
    assert evidence._parse_vllm_timestamp(
        "INFO 08-09 05:24:32 [engine]", year=2026, tz=None,
    ) is None


def test_parse_vllm_timestamp_worker_tp_prefix():
    """Real Worker_TP0_DCP0 prefix must be parsed."""
    tz = timezone(-timedelta(hours=5))
    dt = evidence._parse_vllm_timestamp(
        "(Worker_TP0_DCP0) INFO 08-09 05:24:32.123456 Starting",
        year=2026, tz=tz,
    )
    assert dt is not None
    assert dt.year == 2026


def test_parse_vllm_malformed_raises():
    """Malformed vLLM timestamps that match the regex but have invalid
    date/time values must raise ValueError (caller catches and flags
    timestamp_parse_error for indeterminate verdict), not silently
    return None."""
    with pytest.raises(ValueError):
        evidence._parse_vllm_timestamp(
            "INFO 99-99 99:99:99", year=2026,
            tz=timezone.utc,
        )


# ---------------------------------------------------------------------------
# Rank aggregation
# ---------------------------------------------------------------------------


def test_rank_aggregation_worst_verdict():
    report = evidence.classify_rank(
        0,
        engine_log=ENGINE_PER_LAYER_ISO,
        server_log=SERVER_OOM,
        engine_running=True,
        server_running=True,
        kernel_log=KERNEL_RM_IN_WINDOW,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert report["verdict"] == "fatal"
    assert report["engine"]["verdict"] == "bounded_rm_retry"
    assert report["server"]["verdict"] == "fatal"


def test_rank_aggregation_both_clean():
    report = evidence.classify_rank(
        1, engine_log=CLEAN_ENGINE, server_log=SERVER_CLEAN,
    )
    assert report["verdict"] == "clean"


def test_rank_without_engine_evidence_is_indeterminate():
    report = evidence.classify_rank(1, server_log=SERVER_CLEAN)
    assert report["verdict"] == "indeterminate"
    assert report["missing_engine_evidence"] is True


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def test_aggregate_report_fatal_wins():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(1, engine_log=XID_FATAL),
        evidence.classify_rank(
            2, engine_log=ENGINE_PER_LAYER_ISO,
            kernel_log=KERNEL_RM_IN_WINDOW,
            rm_event_bound=10, cluster_ready=True,
        ),
        evidence.classify_rank(3, engine_log=CLEAN_ENGINE),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "fatal"
    assert report["ranks_fatal"] == [1]
    assert report["ranks_bounded_rm_retry"] == [2]


def test_aggregate_report_all_clean():
    ranks = [
        evidence.classify_rank(r, engine_log=CLEAN_ENGINE)
        for r in range(4)
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "clean"


def test_fatal_signature_count_counts_signatures_not_components():
    rank = evidence.classify_rank(
        0,
        engine_log="NVRM: Xid 43\nCUDA out of memory\n",
    )
    report = evidence.aggregate_report([rank])
    assert report["fatal_signature_count"] == 2


def test_aggregate_report_indeterminate_dominates():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(
            1, engine_log=ENGINE_PER_LAYER_ISO,
            kernel_log=KERNEL_RM_IN_WINDOW,
            rm_event_bound=10, cluster_ready=False,
        ),
        evidence.classify_rank(2, engine_log=CLEAN_ENGINE),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "indeterminate"
    assert report["ranks_indeterminate"] == [1]


def test_aggregate_report_exposes_timestamp_parse_error():
    """The aggregate report must expose which ranks had timestamp_parse_error."""
    from datetime import timezone, timedelta
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(
            1,
            engine_log=ENGINE_MALFORMED_VLLM_TS_MAT,
            kernel_log=KERNEL_RM_IN_WINDOW,
            rm_event_bound=10,
            engine_log_year=2026,
            engine_log_tz=timezone(-timedelta(hours=5)),
            cluster_ready=True,
        ),
        evidence.classify_rank(2, engine_log=CLEAN_ENGINE),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "indeterminate"
    assert 1 in report["ranks_with_timestamp_parse_error"]
    assert 0 not in report["ranks_with_timestamp_parse_error"]
    # The per-rank engine component must also expose the flag
    rank1 = next(r for r in report["ranks"] if r["rank"] == 1)
    assert rank1["engine"]["timestamp_parse_error"] is True


def test_aggregate_report_timestamp_parse_error_deduplicated():
    """If both engine and server components have timestamp_parse_error,
    the rank must appear only once in ranks_with_timestamp_parse_error."""
    from datetime import timezone, timedelta
    # Create a rank where both engine and server logs have malformed timestamps
    ranks = [
        evidence.classify_rank(
            0,
            engine_log=ENGINE_MALFORMED_VLLM_TS_MAT,
            server_log=ENGINE_MALFORMED_VLLM_TS_MAT,
            kernel_log=KERNEL_RM_IN_WINDOW,
            rm_event_bound=10,
            engine_log_year=2026,
            engine_log_tz=timezone(-timedelta(hours=5)),
            cluster_ready=True,
        ),
    ]
    report = evidence.aggregate_report(ranks)
    # Rank 0 should appear exactly once, not twice
    assert report["ranks_with_timestamp_parse_error"].count(0) == 1


def test_aggregate_report_fatal_beats_indeterminate():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(1, engine_log=XID_FATAL),
        evidence.classify_rank(
            2, engine_log=ENGINE_PER_LAYER_ISO,
            kernel_log=KERNEL_RM_IN_WINDOW,
            rm_event_bound=10, cluster_ready=False,
        ),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Provenance and notes
# ---------------------------------------------------------------------------


def test_sha256_provenance():
    assert len(evidence.classify_log(CLEAN_ENGINE)["sha256"]) == 64


def test_kernel_sha256_present():
    result = evidence.classify_log(
        CLEAN_ENGINE, kernel_log=KERNEL_RM_IN_WINDOW
    )
    assert result["kernel_sha256"] is not None
    assert len(result["kernel_sha256"]) == 64


def test_kernel_sha256_none():
    assert evidence.classify_log(CLEAN_ENGINE)["kernel_sha256"] is None


def test_classification_note_states_evidence_scope():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    note = report["classification_note"]
    assert "per-layer materialization window" in note
    assert "NV_ERR_NO_MEMORY" in note
    assert "mem_desc.c:1359" in note
    assert "LMCache server readiness alone" in note
    assert "window_event_count is a count, not a delta" in note
    assert "offline-validated" in note
    assert "never establishes a safe bound" in note


def test_evidence_scope_states_count_not_delta():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    assert "not a delta" in report["evidence_scope"]


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


def test_worst_verdict_empty_is_clean():
    assert evidence._worst_verdict([]) == "clean"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_inspect(
    tmp_path: Path,
    name: str,
    *,
    running: bool = True,
    restart_count: int = 0,
    oom_killed: bool = False,
    container_id: str | None = None,
    container_name: str | None = None,
) -> Path:
    path = tmp_path / f"{name}-inspect.json"
    path.write_text(json.dumps([{
        "Id": container_id or f"sha256:{name}",
        "Name": container_name or f"/{name}",
        "State": {"Running": running, "OOMKilled": oom_killed},
        "RestartCount": restart_count,
    }]), encoding="utf-8")
    return path


def _engine_cli_args(
    tmp_path: Path,
    logs: list[Path],
    *,
    states: dict[int, dict[str, object]] | None = None,
) -> list[str]:
    all_logs = list(logs)
    for rank in range(len(all_logs), evidence.EXPECTED_RANKS):
        path = tmp_path / f"engine-r{rank}-padding.log"
        path.write_text(CLEAN_ENGINE, encoding="utf-8")
        all_logs.append(path)

    args: list[str] = []
    for rank, log in enumerate(all_logs):
        args.extend(("--engine-log", str(log)))
        state = (states or {}).get(rank, {})
        inspect = _write_inspect(tmp_path, f"engine-r{rank}", **state)
        args.extend(("--engine-inspect", str(inspect)))
        args.extend(("--engine-container-name", f"engine-r{rank}"))
    return args


def _kernel_cli_args(tmp_path: Path, logs: list[Path]) -> list[str]:
    all_logs = list(logs)
    for rank in range(len(all_logs), evidence.EXPECTED_RANKS):
        path = tmp_path / f"kernel-r{rank}-padding.log"
        path.write_text("", encoding="utf-8")
        all_logs.append(path)
    return [arg for log in all_logs for arg in ("--kernel-log", str(log))]


def test_cli_clean(tmp_path, capsys):
    log = tmp_path / "engine-r0.log"
    log.write_text(CLEAN_ENGINE, encoding="utf-8")
    rc = evidence.main(_engine_cli_args(tmp_path, [log]) + ["classify"])
    assert rc == evidence.EXIT_OK


def test_cli_requires_complete_default_four_rank_evidence(tmp_path):
    log = tmp_path / "engine-r0.log"
    log.write_text(CLEAN_ENGINE, encoding="utf-8")
    inspect = _write_inspect(tmp_path, "engine-r0")
    with pytest.raises(SystemExit):
        evidence.main([
            "--engine-log", str(log),
            "--engine-inspect", str(inspect),
            "--engine-container-name", "engine-r0",
            "classify",
        ])


def test_cli_requires_engine_inspect(tmp_path):
    args: list[str] = []
    for rank in range(evidence.EXPECTED_RANKS):
        log = tmp_path / f"engine-r{rank}.log"
        log.write_text(CLEAN_ENGINE, encoding="utf-8")
        args.extend(("--engine-log", str(log)))
        args.extend(("--engine-container-name", f"engine-r{rank}"))
    with pytest.raises(SystemExit):
        evidence.main(args + ["classify"])


def test_cli_rejects_empty_engine_log(tmp_path):
    log = tmp_path / "engine-r0.log"
    log.write_text("", encoding="utf-8")
    rc = evidence.main(_engine_cli_args(tmp_path, [log]) + ["classify"])
    assert rc == evidence.EXIT_CONFIG_ERROR


def test_cli_rejects_unaligned_rank_logs(tmp_path):
    engine = tmp_path / "engine-r0.log"
    kernel0 = tmp_path / "kernel-r0.log"
    kernel1 = tmp_path / "kernel-r1.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    kernel0.write_text("", encoding="utf-8")
    kernel1.write_text("", encoding="utf-8")
    with pytest.raises(SystemExit):
        evidence.main(_engine_cli_args(tmp_path, [engine]) + [
            "--kernel-log", str(kernel0),
            "--kernel-log", str(kernel1),
            "classify",
        ])


def test_cli_inspect_restart_is_fatal(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    rc = evidence.main(_engine_cli_args(
        tmp_path, [engine], states={0: {"restart_count": 1}}
    ) + ["classify"])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["ranks_fatal"] == [0]
    assert report["ranks"][0]["engine"]["restart_count"] == 1


def test_cli_inspect_oom_killed_is_fatal(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    rc = evidence.main(_engine_cli_args(
        tmp_path, [engine], states={0: {"oom_killed": True}}
    ) + ["classify"])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["ranks"][0]["engine"]["oom_killed"] is True


def test_cli_rejects_inspect_name_at_wrong_rank(tmp_path):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    args = _engine_cli_args(tmp_path, [engine])
    expected_name_index = args.index("engine-r0")
    args[expected_name_index] = "wrong-engine-r0"
    rc = evidence.main(args + ["classify"])
    assert rc == evidence.EXIT_CONFIG_ERROR


def test_cli_rejects_engine_identity_reused_as_server(tmp_path):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    args = _engine_cli_args(tmp_path, [engine])
    for rank in range(evidence.EXPECTED_RANKS):
        server_log = tmp_path / f"server-r{rank}.log"
        server_log.write_text(SERVER_CLEAN, encoding="utf-8")
        server_inspect = _write_inspect(
            tmp_path,
            f"server-r{rank}",
            container_id=(
                "sha256:engine-r0" if rank == 0 else f"sha256:server-r{rank}"
            ),
        )
        args.extend(("--server-log", str(server_log)))
        args.extend(("--server-inspect", str(server_inspect)))
        args.extend(("--server-container-name", f"server-r{rank}"))
    rc = evidence.main(args + ["classify"])
    assert rc == evidence.EXIT_CONFIG_ERROR


def test_inspect_requires_container_identity():
    with pytest.raises(evidence.ConfigError):
        evidence._parse_container_inspect(json.dumps([{
            "State": {"Running": True, "OOMKilled": False},
            "RestartCount": 0,
        }]))


def test_cli_rejects_negative_event_bound(tmp_path):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    with pytest.raises(SystemExit):
        evidence.main(_engine_cli_args(tmp_path, [engine]) + [
            "--rm-event-bound", "-1",
            "classify",
        ])


def test_parse_tz_rejects_out_of_range_offset():
    with pytest.raises(evidence.ConfigError):
        evidence._parse_tz("+24:00")


def test_cli_fatal(tmp_path, capsys):
    log = tmp_path / "engine-r0.log"
    log.write_text(XID_FATAL, encoding="utf-8")
    rc = evidence.main(_engine_cli_args(tmp_path, [log]) + ["classify"])
    assert rc == evidence.EXIT_FATAL


def test_cli_bounded_rm_retry(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_PER_LAYER_ISO, encoding="utf-8")
    kernel.write_text(KERNEL_RM_IN_WINDOW, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine])
        + _kernel_cli_args(tmp_path, [kernel])
        + [
        "--rm-event-bound", "10",
        "--cluster-ready",
        "classify",
        ]
    )
    assert rc == evidence.EXIT_OK


def test_cli_indeterminate_no_cluster_ready(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_PER_LAYER_ISO, encoding="utf-8")
    kernel.write_text(KERNEL_RM_IN_WINDOW, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine])
        + _kernel_cli_args(tmp_path, [kernel])
        + ["--rm-event-bound", "10", "classify"]
    )
    assert rc == evidence.EXIT_FATAL


def test_cli_missing_bound_indeterminate(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_PER_LAYER_ISO, encoding="utf-8")
    kernel.write_text(KERNEL_RM_IN_WINDOW, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine])
        + _kernel_cli_args(tmp_path, [kernel])
        + ["--cluster-ready", "classify"]
    )
    assert rc == evidence.EXIT_FATAL


def test_cli_vllm_year_tz(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_PER_LAYER_MATERIALIZATION, encoding="utf-8")
    kernel.write_text(KERNEL_RM_IN_WINDOW, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine])
        + _kernel_cli_args(tmp_path, [kernel])
        + [
        "--rm-event-bound", "10",
        "--cluster-ready",
        "--engine-log-year", "2026",
        "--engine-log-tz=-05:00",
        "classify",
        ]
    )
    assert rc == evidence.EXIT_OK


def test_cli_year_without_tz_indeterminate(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_PER_LAYER_MATERIALIZATION, encoding="utf-8")
    kernel.write_text(KERNEL_RM_IN_WINDOW, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine])
        + _kernel_cli_args(tmp_path, [kernel])
        + [
        "--rm-event-bound", "10",
        "--cluster-ready",
        "--engine-log-year", "2026",
        "classify",
        ]
    )
    assert rc == evidence.EXIT_FATAL


def test_cli_requires_log(capsys):
    with pytest.raises(SystemExit):
        evidence.main(["classify"])


def test_cli_multi_rank(tmp_path, capsys):
    logs = []
    for rank, content in enumerate(
        [CLEAN_ENGINE, XID_FATAL, CLEAN_ENGINE, CLEAN_ENGINE]
    ):
        path = tmp_path / f"engine-r{rank}.log"
        path.write_text(content, encoding="utf-8")
        logs.append(path)
    rc = evidence.main(_engine_cli_args(tmp_path, logs) + ["classify"])
    assert rc == evidence.EXIT_FATAL


def test_cli_kernel_not_found(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    engine.write_text(CLEAN_ENGINE, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine])
        + _kernel_cli_args(tmp_path, [Path("nonexistent.log")])
        + ["classify"]
    )
    assert rc == evidence.EXIT_CONFIG_ERROR


def test_cli_per_rank_bounds(tmp_path, capsys):
    engine0 = tmp_path / "engine-r0.log"
    engine1 = tmp_path / "engine-r1.log"
    kernel0 = tmp_path / "kernel-r0.log"
    kernel1 = tmp_path / "kernel-r1.log"
    engine0.write_text(ENGINE_PER_LAYER_ISO, encoding="utf-8")
    engine1.write_text(ENGINE_PER_LAYER_ISO, encoding="utf-8")
    kernel0.write_text(KERNEL_RM_3_EVENTS, encoding="utf-8")
    kernel1.write_text(KERNEL_RM_36_EVENTS, encoding="utf-8")
    rc = evidence.main(
        _engine_cli_args(tmp_path, [engine0, engine1])
        + _kernel_cli_args(tmp_path, [kernel0, kernel1])
        + [
        "--rm-event-bound", "3",
        "--cluster-ready",
        "classify",
        ]
    )
    assert rc == evidence.EXIT_FATAL


# ---------------------------------------------------------------------------
# TZ parsing
# ---------------------------------------------------------------------------


def test_parse_tz_utc():
    assert evidence._parse_tz("UTC") == timezone.utc


def test_parse_tz_offset():
    tz = evidence._parse_tz("-05:00")
    assert tz.utcoffset(None) == -timedelta(hours=5)


def test_parse_tz_invalid():
    with pytest.raises(evidence.ConfigError):
        evidence._parse_tz("garbage")


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_is_v1():
    assert evidence.SCHEMA == "sparkring-startup-evidence/v1"


# ---------------------------------------------------------------------------
# window_event_count is a count, not a delta
# ---------------------------------------------------------------------------


def test_report_uses_window_event_count_not_rm_event_count():
    """The report field must be 'window_event_count', not 'rm_event_count'
    or 'delta'."""
    result = evidence.classify_log(
        ENGINE_PER_LAYER_ISO,
        container_running=True,
        kernel_log=KERNEL_RM_3_EVENTS,
        rm_event_bound=10,
        cluster_ready=True,
    )
    assert "window_event_count" in result
    assert "rm_event_count" not in result
