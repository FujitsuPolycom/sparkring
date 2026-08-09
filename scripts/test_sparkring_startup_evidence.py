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

RM_RETRY_ENGINE = """\
[INFO] Starting vLLM engine...
[INFO] Loading model from /models/glm52
[WARN] CUDA out of memory. Attempting to free reserved blocks.
[WARN] PYTORCH_CUDA_ALLOC_CONF expandable_segments:True retry
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


def test_rm_retry_with_progress_classifies_recoverable():
    result = evidence.classify_log(RM_RETRY_ENGINE, container_running=True)
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
        engine_log=RM_RETRY_ENGINE,
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
        evidence.classify_rank(2, engine_log=RM_RETRY_ENGINE),
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
        evidence.classify_rank(r, engine_log=RM_RETRY_ENGINE)
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
        [CLEAN_ENGINE, RM_RETRY_ENGINE, XID_FATAL, CLEAN_ENGINE]
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
    distinguish recoverable RM retries from fatal Xid/OOM/driver errors."""
    # A log with only RM retry patterns should be recoverable
    retry_only = (
        "CUDA out of memory. Attempting to free reserved blocks.\n"
        "KV cache allocated: 562688 tokens\n"
        "Engine is ready\n"
    )
    result = evidence.classify_log(retry_only, container_running=True)
    assert result["verdict"] == "recoverable"

    # A log with Xid should be fatal even if an RM retry also appears
    mixed = (
        "CUDA out of memory. Attempting to free reserved blocks.\n"
        "NVRM: Xid 43\n"
        "KV cache allocated\n"
    )
    result = evidence.classify_log(mixed, container_running=True)
    assert result["verdict"] == "fatal"
