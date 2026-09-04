"""Offline contracts for explicit GB10 launch-memory recovery."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import prepare_launch_memory as prepare
from preflight import CheckResult


def _site():
    return SimpleNamespace(
        ranks=(
            SimpleNamespace(id=0, ssh_target="operator@rank0.example.net"),
            SimpleNamespace(id=1, ssh_target="operator@rank1.example.net"),
        ),
        preflight=SimpleNamespace(
            required_free_ports=(8015, 29775),
            memory=SimpleNamespace(
                minimum_available_bytes=96 * (1 << 30),
                contiguous_block_bytes=32 * (1 << 20),
                minimum_contiguous_blocks=200,
            ),
        ),
    )


def test_plan_names_mutation_confirmation_and_every_rank() -> None:
    plan = prepare.plan_document(_site())

    assert plan["safety"] == ["MUTATES HOST"]
    assert plan["confirmation"] == prepare.CONFIRMATION
    assert [action["rank"] for action in plan["actions"]] == [0, 1]
    assert "drop_caches" in plan["actions"][0]["command"][-1]
    assert "compact_memory" in plan["actions"][0]["command"][-1]


def test_remote_command_refuses_active_serving_ports_before_mutation() -> None:
    script = prepare.remote_command((8015, 29775))[-1]

    assert script.index("ss -ltnH") < script.index("drop_caches")
    assert "sync" in script
    assert "sudo -n" in script
    assert "MEMORY_BEFORE" in script
    assert "MEMORY_AFTER" in script


def test_plan_requires_configured_memory_thresholds() -> None:
    site = _site()
    site.preflight.memory = None

    with pytest.raises(prepare.PrepareMemoryError, match="preflight.memory"):
        prepare.plan_document(site)


def test_prepare_one_preserves_remote_shell_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = []

    def fake_run(arguments, **_kwargs):
        observed.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "ok", "")

    monkeypatch.setattr(prepare.subprocess, "run", fake_run)
    result = prepare._prepare_one(_site().ranks[0], (8015,), 30)

    assert result["rank"] == 0
    assert result["stdout"] == "ok"
    assert observed[0][:2] == ("ssh", "operator@rank0.example.net")
    assert observed[0][2].startswith("bash -lc ")


def test_receipt_requires_both_memory_checks_on_every_rank() -> None:
    actions = [{"rank": 0, "stdout": "ok"}, {"rank": 1, "stdout": "ok"}]
    checks = [
        CheckResult(check_id=check_id, rank=rank, subject="memory", passed=passed,
                    detail="fixture")
        for rank in (0, 1)
        for check_id, passed in (
            ("HOST.MEMORY_AVAILABLE", True),
            ("HOST.MEMORY_CONTIGUITY", rank == 0),
        )
    ]

    receipt = prepare.build_receipt(_site(), actions, checks)

    assert receipt["status"] == "reboot-required"
    assert receipt["passed"] is False
    assert receipt["recommended_action"] == (
        "Reboot every rank whose memory check failed, then rerun preflight."
    )


def test_prepare_cluster_verifies_recovery_with_read_only_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = _site()
    prepared = []

    def fake_precheck(rank, _ports, _timeout):
        return {"rank": rank.id}

    def fake_prepare(rank, _ports, _timeout):
        prepared.append(rank.id)
        return {"rank": rank.id, "stdout": "MEMORY_AFTER"}

    checks = [
        CheckResult(check_id=check_id, rank=rank, subject="memory", passed=True,
                    detail="fixture")
        for rank in (0, 1)
        for check_id in prepare.MEMORY_CHECK_IDS
    ]
    observed = {}

    class FakeReadOnlyRunner:
        def __init__(self, timeout):
            observed["timeout"] = timeout

    def fake_preflight(received_site, runner, scope):
        observed.update(site=received_site, runner=runner, scope=scope)
        return checks

    monkeypatch.setattr(prepare, "_precheck_one", fake_precheck)
    monkeypatch.setattr(prepare, "_prepare_one", fake_prepare)
    monkeypatch.setattr(prepare.preflight, "SshRunner", FakeReadOnlyRunner)
    monkeypatch.setattr(prepare.preflight, "run_preflight", fake_preflight)

    receipt = prepare.prepare_cluster(site, 30)

    assert sorted(prepared) == [0, 1]
    assert observed["site"] is site
    assert observed["scope"] == "full"
    assert observed["timeout"] == 30
    assert receipt["status"] == "recovered"


def test_prepare_cluster_checks_every_rank_before_any_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    site = _site()
    prepared = []

    def fake_precheck(rank, _ports, _timeout):
        if rank.id == 1:
            raise prepare.PrepareMemoryError("rank 1 precheck failed")
        return {"rank": rank.id}

    def fake_prepare(rank, _ports, _timeout):
        prepared.append(rank.id)
        return {"rank": rank.id, "stdout": "unexpected"}

    monkeypatch.setattr(prepare, "_precheck_one", fake_precheck)
    monkeypatch.setattr(prepare, "_prepare_one", fake_prepare)

    with pytest.raises(prepare.PrepareMemoryError, match="rank 1 precheck"):
        prepare.prepare_cluster(site, 30)

    assert prepared == []
