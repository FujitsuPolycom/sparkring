"""Tests for the stage-aware startup evidence classifier.

Verdict model (v2):
- clean: no concerning signatures
- bounded_rm_retry: kernel NV_ERR_NO_MEMORY at _memdescAllocInternal @
  mem_desc.c:1359 during EXL3 materialization, with full cross-evidence
- fatal: generic CUDA OOM, torch.OutOfMemoryError, OOMKilled, Xid,
  restart, SSH loss, fabric loss, driver failure
- indeterminate: RM event present but cross-evidence incomplete
  (fatal policy)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_startup_evidence as evidence  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

CLEAN_ENGINE = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[INFO] Model loaded successfully
[INFO] KV cache allocated: 562688 tokens
[INFO] Capturing CUDA graphs...
[INFO] Engine is ready
"""

# Real PyTorch OOM format — MUST be fatal, not recoverable
GENERIC_CUDA_OOM_WITH_PROGRESS = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[ERROR] CUDA out of memory. Tried to allocate 2.00 GiB.
[INFO] KV cache allocated: 562688 tokens
[INFO] Engine is ready
"""

TORCH_OOM_ERROR = """\
[INFO] Starting vLLM engine...
[ERROR] torch.OutOfMemoryError: CUDA out of memory.
[INFO] Engine is ready
"""

BARE_OOM_NO_PROGRESS = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[ERROR] CUDA out of memory
"""

# The exact evidenced kernel RM signature
KERNEL_RM_EVENT = """\
2025-01-15T03:42:11.123456+00:00 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Multiple RM events within bound
KERNEL_RM_EVENTS_WITHIN_BOUND = """\
2025-01-15T03:42:11.123 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
2025-01-15T03:42:12.456 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
2025-01-15T03:42:13.789 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# RM events exceeding bound
KERNEL_RM_EVENTS_OVER_BOUND = """\
2025-01-15T03:42:11.123 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
2025-01-15T03:42:12.456 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
2025-01-15T03:42:13.789 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
2025-01-15T03:42:14.012 NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Kernel log with no timestamps
KERNEL_RM_EVENT_NO_TIMESTAMP = """\
NV_ERR_NO_MEMORY: _memdescAllocInternal @ mem_desc.c:1359
"""

# Engine log with materialization context + layer progress + readiness
ENGINE_WITH_MATERIALIZATION = """\
2025-01-15T03:42:00.000 [INFO] Starting vLLM engine...
2025-01-15T03:42:05.000 [INFO] EXL3 mixed-Trellis per-layer preparation
2025-01-15T03:42:10.000 [INFO] layer 7 loaded
2025-01-15T03:42:14.000 [INFO] KV cache allocated: 562688 tokens
2025-01-15T03:42:15.000 [INFO] Engine is ready
"""

# Engine log with materialization but NO readiness
ENGINE_MATERIALIZE_NO_READINESS = """\
2025-01-15T03:42:00.000 [INFO] Starting vLLM engine...
2025-01-15T03:42:05.000 [INFO] EXL3 mixed-Trellis per-layer preparation
2025-01-15T03:42:10.000 [INFO] layer 7 loaded
"""

# Engine log with NO materialization context
ENGINE_NO_MATERIALIZATION = """\
2025-01-15T03:42:00.000 [INFO] Starting vLLM engine...
2025-01-15T03:42:10.000 [INFO] KV cache allocated: 562688 tokens
2025-01-15T03:42:15.000 [INFO] Engine is ready
"""

# Engine log without timestamps
ENGINE_NO_TIMESTAMPS = """\
[INFO] Starting vLLM engine...
[INFO] EXL3 mixed-Trellis per-layer preparation
[INFO] layer 7 loaded
[INFO] KV cache allocated: 562688 tokens
[INFO] Engine is ready
"""

XID_FATAL = """\
[INFO] Starting vLLM engine...
[NVRM] Xid 119 (GPU has fallen off the bus)
[ERROR] CUDA error: no kernel image
"""

OOMKILLED_FATAL = """\
[INFO] Starting vLLM engine...
[INFO] Loading model
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

# Kernel RM event with a different callsite — not the evidenced one
KERNEL_RM_WRONG_CALLSITE = """\
2025-01-15T03:42:11.123 NV_ERR_NO_MEMORY: _someOtherFunction @ other.c:42
"""


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------


def test_clean_log_classifies_clean():
    result = evidence.classify_log(CLEAN_ENGINE, container_running=True)
    assert result["verdict"] == "clean"
    assert result["fatal_signatures"] == []
    assert result["rm_event_count"] == 0


# ---------------------------------------------------------------------------
# Fatal — generic CUDA OOM is now FATAL, not recoverable
# ---------------------------------------------------------------------------


def test_generic_cuda_oom_with_progress_is_fatal():
    """Generic 'CUDA out of memory' must be fatal even with progress.
    No repository evidence authorizes calling a bare CUDA OOM recoverable."""
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
    assert any(
        s["pattern"] == "generic_oom" for s in result["fatal_signatures"]
    )


def test_bare_oom_no_progress_is_fatal():
    result = evidence.classify_log(BARE_OOM_NO_PROGRESS, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "generic_oom" for s in result["fatal_signatures"]
    )


def test_bare_oom_in_dead_container_is_fatal():
    result = evidence.classify_log(
        BARE_OOM_NO_PROGRESS, container_running=False
    )
    assert result["verdict"] == "fatal"


def test_multi_oom_late_fatal():
    """A late OOM after progress must still be fatal."""
    result = evidence.classify_log(MULTI_OOM_LATE_FATAL, container_running=True)
    assert result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Fatal — Xid, OOMKilled, SSH, fabric, restart, driver
# ---------------------------------------------------------------------------


def test_xid_classifies_fatal():
    result = evidence.classify_log(XID_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(s["pattern"] == "xid" for s in result["fatal_signatures"])


def test_oomkilled_classifies_fatal():
    result = evidence.classify_log(OOMKILLED_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "oomkilled" for s in result["fatal_signatures"]
    )


def test_ssh_failure_classifies_fatal():
    result = evidence.classify_log(SSH_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "ssh_failure" for s in result["fatal_signatures"]
    )


def test_fabric_failure_classifies_fatal():
    result = evidence.classify_log(FABRIC_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "fabric_failure" for s in result["fatal_signatures"]
    )


def test_carrier_loss_classifies_fatal():
    result = evidence.classify_log(CARRIER_LOSS, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "fabric_failure" for s in result["fatal_signatures"]
    )


def test_unexpected_restart_classifies_fatal():
    result = evidence.classify_log(RESTART_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "unexpected_restart"
        for s in result["fatal_signatures"]
    )


def test_driver_failure_classifies_fatal():
    result = evidence.classify_log(DRIVER_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "driver_failure"
        for s in result["fatal_signatures"]
    )


# ---------------------------------------------------------------------------
# Bounded RM retry — the evidenced kernel signature with cross-evidence
# ---------------------------------------------------------------------------


def test_bounded_rm_retry_with_full_cross_evidence():
    """RM event + materialization + layer progress + readiness + container
    running + timestamps + within bound = bounded_rm_retry."""
    result = evidence.classify_log(
        ENGINE_WITH_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["rm_event_count"] == 1
    assert result["rm_event_within_bound"] is True
    assert result["has_readiness"] is True
    assert result["timestamps_correlatable"] is True
    assert len(result["materialization_lines"]) >= 1
    assert result["kernel_sha256"] is not None


def test_bounded_rm_retry_multiple_events_within_bound():
    result = evidence.classify_log(
        ENGINE_WITH_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_EVENTS_WITHIN_BOUND,
    )
    assert result["verdict"] == "bounded_rm_retry"
    assert result["rm_event_count"] == 3
    assert result["rm_event_within_bound"] is True


def test_bounded_rm_retry_exact_evidenced_callsite():
    """Only NV_ERR_NO_MEMORY at _memdescAllocInternal @ mem_desc.c:1359
    qualifies — not other callsites."""
    result = evidence.classify_log(
        ENGINE_WITH_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_WRONG_CALLSITE,
    )
    assert result["rm_event_count"] == 0
    assert result["verdict"] == "clean"


# ---------------------------------------------------------------------------
# Indeterminate — RM event but cross-evidence incomplete
# ---------------------------------------------------------------------------


def test_indeterminate_when_rm_event_outside_materialization():
    result = evidence.classify_log(
        ENGINE_NO_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "indeterminate"


def test_indeterminate_when_no_readiness():
    result = evidence.classify_log(
        ENGINE_MATERIALIZE_NO_READINESS,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "indeterminate"


def test_indeterminate_when_timestamps_missing():
    """If timestamps cannot be correlated, verdict is indeterminate."""
    result = evidence.classify_log(
        ENGINE_NO_TIMESTAMPS,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT_NO_TIMESTAMP,
    )
    assert result["verdict"] == "indeterminate"


def test_indeterminate_when_no_progress_after_rm():
    """RM event with materialization but no progress lines at all."""
    engine = (
        "2025-01-15T03:42:00.000 [INFO] Starting vLLM engine...\n"
        "2025-01-15T03:42:05.000 [INFO] EXL3 mixed-Trellis materialization\n"
    )
    result = evidence.classify_log(
        engine,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "indeterminate"


# ---------------------------------------------------------------------------
# Fatal — RM events with disqualifying conditions
# ---------------------------------------------------------------------------


def test_rm_event_over_bound_is_fatal():
    """RM event count exceeding the conservative bound is fatal."""
    result = evidence.classify_log(
        ENGINE_WITH_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_EVENTS_OVER_BOUND,
    )
    assert result["verdict"] == "fatal"
    assert result["rm_event_within_bound"] is False


def test_rm_event_in_dead_container_is_fatal():
    result = evidence.classify_log(
        ENGINE_WITH_MATERIALIZATION,
        container_running=False,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_xid_is_fatal():
    """Xid alongside RM event — Xid wins (fatal)."""
    engine_with_xid = (
        ENGINE_WITH_MATERIALIZATION
        + "[NVRM] Xid 43 (GPU reset)\n"
    )
    result = evidence.classify_log(
        engine_with_xid,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_oomkilled_is_fatal():
    engine_with_oomkill = (
        ENGINE_WITH_MATERIALIZATION
        + "Killed process (OOMKilled)\n"
    )
    result = evidence.classify_log(
        engine_with_oomkill,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_restart_is_fatal():
    engine_with_restart = (
        ENGINE_WITH_MATERIALIZATION
        + "Container engine-r0 is restarting (restart count: 1)\n"
    )
    result = evidence.classify_log(
        engine_with_restart,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "fatal"


def test_rm_event_with_fabric_loss_is_fatal():
    engine_with_fabric = (
        ENGINE_WITH_MATERIALIZATION
        + "[ERROR] NCCL: bootstrap failed (RoCE link down)\n"
    )
    result = evidence.classify_log(
        engine_with_fabric,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert result["verdict"] == "fatal"


# ---------------------------------------------------------------------------
# Generic CUDA OOM + progress must remain fatal (regression)
# ---------------------------------------------------------------------------


def test_generic_cuda_oom_with_progress_remains_fatal():
    """Regression: generic CUDA OOM followed by progress must NOT be
    downgraded to bounded_rm_retry. No invented recoverable signature."""
    log = (
        "CUDA out of memory. Tried to allocate 2.00 GiB.\n"
        "Model loaded successfully\n"
        "Engine is ready\n"
    )
    result = evidence.classify_log(log, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "generic_oom" for s in result["fatal_signatures"]
    )


# ---------------------------------------------------------------------------
# No invented patterns
# ---------------------------------------------------------------------------


def test_no_invented_recoverable_patterns():
    """No invented allocator-internal retry signatures must exist."""
    assert not hasattr(evidence, "_RM_RETRY")
    assert not hasattr(evidence, "_ALLOC_CONTINUE")
    assert not hasattr(evidence, "RECOVERABLE_PATTERNS")


# ---------------------------------------------------------------------------
# Rank aggregation
# ---------------------------------------------------------------------------


def test_rank_aggregation_worst_verdict():
    report = evidence.classify_rank(
        0,
        engine_log=ENGINE_WITH_MATERIALIZATION,
        server_log=SERVER_OOM,
        engine_running=True,
        server_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert report["verdict"] == "fatal"
    assert report["engine"]["verdict"] == "bounded_rm_retry"
    assert report["server"]["verdict"] == "fatal"


def test_rank_aggregation_both_clean():
    report = evidence.classify_rank(
        1,
        engine_log=CLEAN_ENGINE,
        server_log=SERVER_CLEAN,
    )
    assert report["verdict"] == "clean"


def test_rank_aggregation_indeterminate():
    report = evidence.classify_rank(
        2,
        engine_log=ENGINE_NO_MATERIALIZATION,
        engine_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert report["verdict"] == "indeterminate"


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


def test_aggregate_report_fatal_wins():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(1, engine_log=XID_FATAL),
        evidence.classify_rank(
            2, engine_log=ENGINE_WITH_MATERIALIZATION,
            kernel_log=KERNEL_RM_EVENT,
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
    assert report["ranks_clean"] == [0, 1, 2, 3]


def test_aggregate_report_bounded_rm_retry():
    ranks = [
        evidence.classify_rank(
            r,
            engine_log=ENGINE_WITH_MATERIALIZATION,
            kernel_log=KERNEL_RM_EVENT,
        )
        for r in range(4)
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "bounded_rm_retry"
    assert report["ranks_bounded_rm_retry"] == [0, 1, 2, 3]
    assert report["ranks_fatal"] == []


def test_aggregate_report_indeterminate():
    ranks = [
        evidence.classify_rank(
            0,
            engine_log=ENGINE_NO_MATERIALIZATION,
            kernel_log=KERNEL_RM_EVENT,
        ),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "indeterminate"
    assert report["ranks_indeterminate"] == [0]


# ---------------------------------------------------------------------------
# Provenance and notes
# ---------------------------------------------------------------------------


def test_sha256_provenance_present():
    result = evidence.classify_log(CLEAN_ENGINE)
    assert len(result["sha256"]) == 64


def test_kernel_sha256_present_when_kernel_log_given():
    result = evidence.classify_log(
        CLEAN_ENGINE, kernel_log=KERNEL_RM_EVENT
    )
    assert result["kernel_sha256"] is not None
    assert len(result["kernel_sha256"]) == 64


def test_kernel_sha256_none_when_no_kernel_log():
    result = evidence.classify_log(CLEAN_ENGINE)
    assert result["kernel_sha256"] is None


def test_classification_note_present():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    assert "never globally ignored" in report["classification_note"]
    assert "NV_ERR_NO_MEMORY" in report["classification_note"]
    assert "mem_desc.c:1359" in report["classification_note"]


def test_classification_note_states_evidence_scope():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    assert "evidence-scoped" in report["classification_note"]
    assert "does not prove all RM errors safe" in report["classification_note"]


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
    """Indeterminate must use fatal exit code (fatal policy)."""
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_NO_MATERIALIZATION, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "indeterminate"


def test_cli_bounded_rm_retry_exit_code(tmp_path, capsys):
    engine = tmp_path / "engine-r0.log"
    kernel = tmp_path / "kernel-r0.log"
    engine.write_text(ENGINE_WITH_MATERIALIZATION, encoding="utf-8")
    kernel.write_text(KERNEL_RM_EVENT, encoding="utf-8")
    rc = evidence.main([
        "--engine-log", str(engine),
        "--kernel-log", str(kernel),
        "classify",
    ])
    assert rc == evidence.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "bounded_rm_retry"


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
    assert report["ranks"][0]["server"]["verdict"] == "clean"


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


# ---------------------------------------------------------------------------
# NVIDIA errors not globally ignored
# ---------------------------------------------------------------------------


def test_nvidia_errors_not_globally_ignored():
    """The classifier must distinguish evidenced RM events from fatal
    NVIDIA errors like Xid."""
    # Bounded RM event with full cross-evidence
    bounded = evidence.classify_log(
        ENGINE_WITH_MATERIALIZATION,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert bounded["verdict"] == "bounded_rm_retry"

    # Xid must be fatal even with RM event and progress
    mixed = (
        ENGINE_WITH_MATERIALIZATION
        + "[NVRM] Xid 43\n"
    )
    mixed_result = evidence.classify_log(
        mixed,
        container_running=True,
        kernel_log=KERNEL_RM_EVENT,
    )
    assert mixed_result["verdict"] == "fatal"


def test_rm_event_bound_constant():
    """The conservative bound must be a small explicit number."""
    assert isinstance(evidence.RM_EVENT_BOUND, int)
    assert evidence.RM_EVENT_BOUND >= 1
    assert evidence.RM_EVENT_BOUND <= 10
