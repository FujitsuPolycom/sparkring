"""Tests for the stage-aware startup evidence classifier."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_startup_evidence as evidence  # noqa: E402


CLEAN_ENGINE = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[INFO] Model loaded successfully
[INFO] KV cache allocated: 562688 tokens
[INFO] Capturing CUDA graphs...
[INFO] Engine is ready
"""

OOM_WITH_PROGRESS = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[ERROR] CUDA out of memory. Tried to allocate 2.00 GiB.
[INFO] KV cache allocated: 562688 tokens
[INFO] Engine is ready
"""

BARE_OOM_NO_PROGRESS = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[ERROR] CUDA out of memory
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

RESTART_FATAL = """\
[INFO] Starting vLLM engine...
Container engine-r0 is restarting (restart count: 1)
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


def test_clean_log_classifies_clean():
    result = evidence.classify_log(CLEAN_ENGINE, container_running=True)
    assert result["verdict"] == "clean"
    assert result["fatal_signatures"] == []
    assert result["recoverable_signatures"] == []


def test_oom_with_progress_classifies_recoverable():
    """A CUDA OOM line followed by a progress line while the container
    is running must classify as recoverable — no invented signature."""
    result = evidence.classify_log(OOM_WITH_PROGRESS, container_running=True)
    assert result["verdict"] == "recoverable"
    assert len(result["recoverable_signatures"]) >= 1
    assert result["fatal_signatures"] == []
    assert result["progress_after_oom"] is True

def test_bare_oom_no_progress_classifies_fatal():
    result = evidence.classify_log(BARE_OOM_NO_PROGRESS, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "bare_oom_no_progress" for s in result["fatal_signatures"]
    )


def test_bare_oom_with_progress_classifies_recoverable():
    log = BARE_OOM_NO_PROGRESS + "[INFO] KV cache allocated\n[INFO] Engine is ready\n"
    result = evidence.classify_log(log, container_running=True)
    assert result["verdict"] == "recoverable"
    assert result["progress_after_oom"] is True


def test_bare_oom_in_dead_container_is_fatal():
    result = evidence.classify_log(BARE_OOM_NO_PROGRESS, container_running=False)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "oom_container_dead" for s in result["fatal_signatures"]
    )


def test_xid_classifies_fatal():
    result = evidence.classify_log(XID_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(s["pattern"] == "xid" for s in result["fatal_signatures"])


def test_oomkilled_classifies_fatal():
    result = evidence.classify_log(OOMKILLED_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "fatal_oom" for s in result["fatal_signatures"]
    )


def test_ssh_failure_classifies_fatal():
    result = evidence.classify_log(SSH_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(s["pattern"] == "ssh_failure" for s in result["fatal_signatures"])


def test_fabric_failure_classifies_fatal():
    result = evidence.classify_log(FABRIC_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "fabric_failure" for s in result["fatal_signatures"]
    )


DRIVER_FAILURE = """\
[INFO] Starting vLLM engine...
[ERROR] all CUDA-capable devices are busy or unavailable
[ERROR] no CUDA-capable device detected
"""


def test_driver_failure_classifies_fatal():
    result = evidence.classify_log(DRIVER_FAILURE, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "driver_failure" for s in result["fatal_signatures"]
    )


MULTI_OOM_LATE_FATAL = """\
[INFO] Starting vLLM engine...
[WARN] CUDA out of memory. Attempting to free reserved blocks.
[INFO] KV cache allocated: 562688 tokens
[INFO] Engine is ready
[ERROR] CUDA out of memory
"""


def test_multi_oom_late_fatal_without_progress():
    """An OOM after the last progress line must be fatal, even if
    an earlier OOM had progress after it."""
    result = evidence.classify_log(MULTI_OOM_LATE_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    # The early OOM (line 2) should be recoverable
    recov = [s for s in result["recoverable_signatures"] if s["line_number"] == 2]
    assert len(recov) >= 1
    # The late OOM (line 5) should be fatal — no progress after it
    fatal = [s for s in result["fatal_signatures"] if s["line_number"] == 5]
    assert len(fatal) >= 1
    assert fatal[0]["pattern"] == "bare_oom_no_progress"


def test_unexpected_restart_classifies_fatal():
    result = evidence.classify_log(RESTART_FATAL, container_running=True)
    assert result["verdict"] == "fatal"
    assert any(
        s["pattern"] == "unexpected_restart" for s in result["fatal_signatures"]
    )

def test_cli_missing_file_config_error(tmp_path):
    rc = evidence.main([
        "--engine-log", str(tmp_path / "nonexistent.log"), "classify",
    ])
    assert rc == evidence.EXIT_CONFIG_ERROR

def test_rank_aggregation_worst_verdict():
    report = evidence.classify_rank(
        0,
        engine_log=OOM_WITH_PROGRESS,
        server_log=SERVER_OOM,
        engine_running=True,
        server_running=True,
    )
    assert report["verdict"] == "fatal"
    assert report["engine"]["verdict"] == "recoverable"
    assert report["server"]["verdict"] == "fatal"


def test_rank_aggregation_both_clean():
    report = evidence.classify_rank(
        1,
        engine_log=CLEAN_ENGINE,
        server_log=SERVER_CLEAN,
    )
    assert report["verdict"] == "clean"


def test_aggregate_report_fatal_wins():
    ranks = [
        evidence.classify_rank(0, engine_log=CLEAN_ENGINE),
        evidence.classify_rank(1, engine_log=XID_FATAL),
        evidence.classify_rank(2, engine_log=OOM_WITH_PROGRESS),
        evidence.classify_rank(3, engine_log=CLEAN_ENGINE),
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "fatal"
    assert report["ranks_fatal"] == [1]
    assert report["ranks_recoverable"] == [2]
    assert report["ranks_clean"] == [0, 3]


def test_aggregate_report_all_clean():
    ranks = [
        evidence.classify_rank(r, engine_log=CLEAN_ENGINE)
        for r in range(4)
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "clean"
    assert report["ranks_fatal"] == []
    assert report["ranks_clean"] == [0, 1, 2, 3]


def test_aggregate_report_recoverable():
    ranks = [
        evidence.classify_rank(r, engine_log=OOM_WITH_PROGRESS)
        for r in range(4)
    ]
    report = evidence.aggregate_report(ranks)
    assert report["verdict"] == "recoverable"
    assert report["ranks_recoverable"] == [0, 1, 2, 3]
    assert report["ranks_fatal"] == []


def test_sha256_provenance_present():
    result = evidence.classify_log(CLEAN_ENGINE)
    assert len(result["sha256"]) == 64


def test_classification_note_present():
    ranks = [evidence.classify_rank(0, engine_log=CLEAN_ENGINE)]
    report = evidence.aggregate_report(ranks)
    assert "never globally ignored" in report["classification_note"]


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


def test_cli_requires_at_least_one_log(capsys):
    with pytest.raises(SystemExit):
        evidence.main(["classify"])



def test_cli_multi_rank(tmp_path, capsys):
    logs = []
    for rank, content in enumerate(
        [CLEAN_ENGINE, OOM_WITH_PROGRESS, XID_FATAL, CLEAN_ENGINE]
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
    assert report["ranks_fatal"] == [2]
    assert report["ranks_recoverable"] == [1]


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
    assert report["ranks"][0]["engine"]["verdict"] == "clean"
    assert report["ranks"][0]["server"]["verdict"] == "clean"


def test_cli_engine_dead_promotes_oom_to_fatal(tmp_path, capsys):
    log = tmp_path / "engine-r0.log"
    log.write_text(
        "[INFO] Starting\n[ERROR] CUDA out of memory\n",
        encoding="utf-8",
    )
    rc = evidence.main([
        "--engine-log", str(log),
        "--engine-dead",
        "classify",
    ])
    assert rc == evidence.EXIT_FATAL


def test_nvidia_errors_not_globally_ignored():
    """The classifier must not ignore all NVIDIA errors; it must
    distinguish a recoverable OOM (with progress) from fatal Xid/driver errors."""
    # A bare OOM followed by progress is recoverable
    oom_progress = (
        "CUDA out of memory. Tried to allocate 2.00 GiB.\n"
        "KV cache allocated: 562688 tokens\n"
        "Engine is ready\n"
    )
    result = evidence.classify_log(oom_progress, container_running=True)
    assert result["verdict"] == "recoverable"

    # A log with Xid should be fatal even if progress also appears
    mixed = (
        "CUDA out of memory. Tried to allocate 2.00 GiB.\n"
        "NVRM: Xid 43\n"
        "KV cache allocated\n"
    )
    result = evidence.classify_log(mixed, container_running=True)
    assert result["verdict"] == "fatal"


def test_no_invented_recoverable_patterns():
    """No invented allocator-internal retry signatures must exist.
    The classifier must not use patterns like expandable_segments:True
    or 'Trying' that don't match real PyTorch/vLLM log output."""
    # The EXL3 profile sets expandable_segments:False, so a pattern
    # matching expandable_segments:True can never fire on real logs.
    assert not hasattr(evidence, "_RM_RETRY")
    assert not hasattr(evidence, "_ALLOC_CONTINUE")
    assert not hasattr(evidence, "RECOVERABLE_PATTERNS")


def test_recoverable_uses_oom_plus_progress_only():
    """The only recoverable signal is: OOM line + progress line + container alive.
    No invented signature pattern should be matched."""
    log = (
        "CUDA out of memory. Tried to allocate 2.00 GiB.\n"
        "Model loaded successfully\n"
    )
    result = evidence.classify_log(log, container_running=True)
    assert result["verdict"] == "recoverable"
    sigs = result["recoverable_signatures"]
    assert len(sigs) == 1
    assert sigs[0]["pattern"] == "bare_oom_with_progress"


def test_real_pytorch_oom_format_classified():
    """Real PyTorch OOM format: 'CUDA out of memory. Tried to allocate
    X GiB. The device has Y GiB free...' must be detected as OOM."""
    log = (
        "CUDA out of memory. Tried to allocate 2.00 GiB. "
        "The device has 0.50 GiB free in total.\n"
        "Model loaded successfully\n"
    )
    result = evidence.classify_log(log, container_running=True)
    assert result["verdict"] == "recoverable"
    assert result["progress_after_oom"] is True
