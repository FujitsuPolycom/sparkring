import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.deploy_engine import execute_plan, validate_plan, plan_digest
from scripts.deploy_runtime import (
    build_runtime_plan,
    verify_test_receipt,
    probe_readiness,
)
from scripts.deploy_suite import create_spec
from scripts.test_deploy_suite import inventory


def prepared():
    result = {
        "schema": "sparkring-deploy-preparation/v1",
        "epoch": "1" * 32,
        "spec": create_spec(inventory(), "test-mesh", "/srv/sparkring/test-mesh"),
        "network_plan": {"requires_hardware_validation": True},
        "source": {"files": {}},
        "controller_launch": str(Path(".private/deploy-fixture/launch").resolve()),
        "controller_source": str(Path(".private/deploy-fixture/source").resolve()),
    }
    result["network_verification"] = {
        "ready": True,
        "spec_sha256": plan_digest(result["spec"]),
    }
    return result


@pytest.mark.parametrize(
    "action",
    [
        "create",
        "install",
        "up",
        "start",
        "stop",
        "down",
        "recover",
        "status",
        "logs",
        "ready",
        "native-check",
    ],
)
def test_managed_plans_reuse_canonical_commands(action):
    plan = build_runtime_plan(prepared(), action)
    validate_plan(plan)
    text = json.dumps(plan)
    assert "lil run" not in text
    assert plan["phases"]
    if action == "create":
        assert "SPARKRING_CREATE_ONLY=1" in text
        assert "launch-rank.sh" in text
        assert all(
            a["risk"] != "starts-model" for p in plan["phases"] for a in p["actions"]
        )
    if action == "install":
        assert "managed_install.py" in text
    if action == "recover":
        ids = [p["id"] for p in plan["phases"]]
        assert (
            ids.index("all-rank-stop-barrier")
            < ids.index("stop-mesh")
            < ids.index("cleanup")
            < ids.index("start-mesh-supervisors")
        )
    if action == "start":
        assert plan["phases"][0]["id"] == "require-mesh-ready"


def test_model_permission_precedes_any_quiesce(tmp_path):
    calls = []

    def run(*args):
        calls.append(args)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    plan = build_runtime_plan(prepared(), "stop")
    with pytest.raises(ValueError, match="model-action"):
        execute_plan(plan, tmp_path / "r.json", plan["sha256"], runner=run)
    assert calls == []


def test_install_requires_shared_epoch():
    p = prepared()
    del p["epoch"]
    with pytest.raises(ValueError, match="epoch"):
        build_runtime_plan(p, "install")


def test_recover_uses_installed_tools_without_staged_source():
    p = prepared()
    del p["source"]
    plan = build_runtime_plan(p, "recover")
    assert "verify-host" not in json.dumps(plan)
    assert "start-mesh-supervisors" in json.dumps(plan)
    assert "starts-model" not in json.dumps(plan)


def test_duplicate_hosts_rejected():
    p = prepared()
    p["spec"]["hosts"][1]["host"] = p["spec"]["hosts"][0]["host"]
    with pytest.raises(ValueError, match="distinct"):
        build_runtime_plan(p, "stop")


def test_readiness_receipt_requires_success(tmp_path):
    output = tmp_path / "ready.json"
    output.write_text(
        json.dumps({"schema": "sparkring-managed-model-readiness/v1", "ready": False})
    )
    with pytest.raises(ValueError, match="did not pass"):
        verify_test_receipt(output, "ready")
    output.write_text(
        json.dumps({"schema": "sparkring-managed-model-readiness/v1", "ready": True})
    )
    assert verify_test_receipt(output, "ready") == {"passed": True}


def test_native_receipt_is_directory_with_all_rank_results(tmp_path):
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "schema": "sparkring-mtp3-native-plan/v1",
                "cells": [{"rows": 4}, {"rows": 20}],
            }
        )
    )
    results = [{"returncode": 0, "stdout": "EVIDENCE_JSON {}"}] * 4
    (tmp_path / "q4.json").write_text(json.dumps(results))
    with pytest.raises(FileNotFoundError):
        verify_test_receipt(tmp_path, "native-check")
    (tmp_path / "q20.json").write_text(json.dumps(results[:3]))
    with pytest.raises(ValueError, match="incomplete"):
        verify_test_receipt(tmp_path, "native-check")
    (tmp_path / "q20.json").write_text(json.dumps(results))
    assert verify_test_receipt(tmp_path, "native-check") == {"passed": True}


def test_recovery_quiesce_verification_accepts_absent_boot_state(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state_dir": str(tmp_path / "runtime")}))
    plan = build_runtime_plan(prepared(), "recover")
    verify = plan["phases"][0]["actions"][0]["verify"]["argv"]
    code = verify[verify.index("-c") + 1]
    result = subprocess.run(
        [sys.executable, "-c", code, str(config)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_unit_stop_does_not_block_pinned_container_stop(tmp_path):
    running = {f"spark-r{i}": True for i in range(4)}
    calls = []

    def run(host, argv, timeout):
        calls.append((host, argv))
        if "stop-model" in argv:
            running[host] = False
        return {
            "returncode": int("model-stopped" in argv and running[host]),
            "stdout": "",
            "stderr": "",
        }

    plan = build_runtime_plan(prepared(), "stop")
    execute_plan(
        plan,
        tmp_path / "stop.json",
        plan["sha256"],
        runner=run,
        allow_model_actions=True,
    )
    assert not any(running.values())


def test_ready_plan_reprobes_on_resume_instead_of_reading_historical_receipt(tmp_path):
    plan = build_runtime_plan(prepared(), "ready")
    calls = []

    def run(host, argv, timeout):
        calls.append(argv)
        return {"returncode": 0, "stdout": "", "stderr": ""}

    path = tmp_path / "ready.json"
    execute_plan(plan, path, plan["sha256"], runner=run)
    original = plan["phases"][0]["actions"][0]["argv"]
    calls.clear()
    execute_plan(plan, path, plan["sha256"], runner=run, resume=True)
    assert calls == [original]


def test_readiness_probe_preserves_each_run_and_observes_changed_health(tmp_path):
    responses = iter([True, False])
    calls = []

    def wait(plan, timeout):
        calls.append((plan, timeout))
        return {
            "schema": "sparkring-managed-model-readiness/v1",
            "ready": next(responses),
        }

    first = probe_readiness(
        tmp_path / "launch",
        tmp_path / "results",
        wait=wait,
        load=lambda path: {"path": str(path)},
    )
    second = probe_readiness(
        tmp_path / "launch",
        tmp_path / "results",
        wait=wait,
        load=lambda path: {"path": str(path)},
    )
    assert len(calls) == 2 and first["ready"] is True and second["ready"] is False
    assert first["receipt"] != second["receipt"]
    assert json.loads(Path(first["receipt"]).read_text())["ready"] is True
    assert json.loads(Path(second["receipt"]).read_text())["ready"] is False


def test_quiesce_verification_rejects_active_or_wrong_generation_intent(tmp_path):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"state_dir": str(tmp_path)}))
    (tmp_path / "status.json").write_text(json.dumps({"generation": "a"}))
    code = build_runtime_plan(prepared(), "stop")["phases"][0]["actions"][0]["verify"][
        "argv"
    ][-2]
    for active, generation, succeeds in (
        (True, "a", False),
        (False, "b", False),
        (False, "a", True),
    ):
        (tmp_path / "model-intent.json").write_text(
            json.dumps({"active": active, "generation": generation})
        )
        result = subprocess.run(
            [sys.executable, "-c", code, str(config)], capture_output=True, text=True
        )
        assert (result.returncode == 0) is succeeds


def test_down_stops_mesh_without_restarting_supervisors():
    p = prepared()
    del p["source"]
    phases = [phase["id"] for phase in build_runtime_plan(p, "down")["phases"]]
    assert phases[-3:] == ["stop-mesh", "cleanup", "reset-units"]
    assert "start-mesh-supervisors" not in phases


def test_memory_preparation_is_explicit_and_ordered_before_start():
    p = prepared()
    assert not any(
        phase["id"].startswith("memory-")
        for phase in build_runtime_plan(p, "start")["phases"]
    )
    p["lifecycle_capabilities"] = ["memory-idle", "memory-prepare", "memory-check"]
    plan = build_runtime_plan(p, "start")
    assert [phase["id"] for phase in plan["phases"]] == [
        "all-rank-start-stop-barrier",
        "memory-idle",
        "memory-prepare",
        "memory-check",
        "require-mesh-ready",
        "start-managed-models",
    ]
    assert all(
        action["risk"] == "mutates-host" for action in plan["phases"][2]["actions"]
    )
    assert all(
        "memory-check" in action["verify"]["argv"]
        for action in plan["phases"][2]["actions"]
    )


def test_partial_memory_capabilities_rejected():
    p = prepared()
    p["lifecycle_capabilities"] = ["memory-prepare"]
    with pytest.raises(ValueError, match="all three"):
        build_runtime_plan(p, "start")


def test_memory_idle_failure_prevents_preparation_and_model_start(tmp_path):
    p = prepared()
    p["lifecycle_capabilities"] = ["memory-idle", "memory-prepare", "memory-check"]
    plan = build_runtime_plan(p, "start")
    calls = []

    def run(host, argv, timeout):
        calls.append(argv)
        return {
            "returncode": int("memory-idle" in argv and host == "spark-r2"),
            "stdout": '{"verified":true}',
            "stderr": "",
        }

    with pytest.raises(RuntimeError, match="memory-idle"):
        execute_plan(
            plan,
            tmp_path / "start.json",
            plan["sha256"],
            runner=run,
            allow_model_actions=True,
        )
    assert not any("memory-prepare" in argv for argv in calls)
    assert not any("systemctl" in argv and "start" in argv for argv in calls)


@pytest.mark.parametrize(
    "state,expected",
    [("inactive", True), ("failed", True), ("active", False), ("deactivating", False)],
)
def test_unit_stop_verifier_checks_unit_state_not_container_state(
    monkeypatch, state, expected
):
    from types import SimpleNamespace

    plan = build_runtime_plan(prepared(), "stop")
    code = plan["phases"][1]["actions"][0]["verify"]["argv"][-1]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=f"LoadState=loaded\nActiveState={state}\n"
        ),
    )
    if expected:
        exec(code, {})
    else:
        with pytest.raises(AssertionError):
            exec(code, {})
