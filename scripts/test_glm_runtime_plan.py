"""Managed GLM creation selects a backend without bypassing lifecycle phases."""
import json
from pathlib import Path

import pytest

from scripts import deploy_selection
from scripts.deploy_engine import plan_digest, validate_plan
from scripts.deploy_runtime import build_runtime_plan
from scripts.test_deploy_runtime import prepared


@pytest.mark.parametrize("backend", ["docker", "compose"])
def test_registered_candidate_uses_structured_creation(monkeypatch, backend):
    preparation = prepared()
    public = deploy_selection.selection(preparation["spec"], Path(__file__).resolve().parents[1] / "runtime/glm53-spark-mtp3-mesh")
    public["receipt"] = {"schema": "sparkring-candidate-image-receipt/v1"}
    monkeypatch.setattr(deploy_selection, "selection", lambda *a: public)
    preparation["network_verification"]["spec_sha256"] = plan_digest(preparation["spec"])
    plan = build_runtime_plan(preparation, "create", container_backend=backend)
    validate_plan(plan)
    assert [phase["id"] for phase in plan["phases"]] == ["plan-stopped-containers", "create-stopped-containers"]
    commands = [action["argv"] for phase in plan["phases"] for action in phase["actions"]]
    text = json.dumps(commands)
    assert "runtime/common/glm_launch.py" in text
    assert "SPARKRING_CREATE_ONLY=1" not in text
    arguments = [json.loads(command[-1]) for command in commands]
    assert all("--backend" in command and command[command.index("--backend")+1] == backend for command in arguments)
    assert all(action["risk"] != "starts-model" for phase in plan["phases"] for action in phase["actions"])


def test_compose_selection_cannot_replace_managed_start():
    with pytest.raises(ValueError, match="managed start/stop remain unchanged"):
        build_runtime_plan(prepared(), "start", container_backend="compose")


def test_legacy_images_keep_their_existing_creation_path():
    plan = build_runtime_plan(prepared(), "create")
    assert "launch-rank.sh" in json.dumps(plan)
    with pytest.raises(ValueError, match="explicit R35 or candidate"):
        build_runtime_plan(prepared(), "create", container_backend="compose")


def test_prepared_cache_diagnostics_keep_trusted_legacy_docker_creation(monkeypatch):
    preparation = prepared()
    preparation["spec"]["site"].update(
        runtime_profile="tp4-dcp1-sparkcache",
        cache_diagnostics={"namespace": "isolated-diagnostic", "access_mode": "restore-only", "trace_reuse": 1},
    )
    public = deploy_selection.selection(prepared()["spec"], Path(__file__).resolve().parents[1] / "runtime/glm53-spark-mtp3-mesh")
    public["receipt"] = {"schema": "sparkring-candidate-image-receipt/v1"}
    monkeypatch.setattr(deploy_selection, "selection", lambda *a: public)
    preparation["network_verification"]["spec_sha256"] = plan_digest(preparation["spec"])
    plan = build_runtime_plan(preparation, "create")
    validate_plan(plan)
    assert [phase["id"] for phase in plan["phases"]] == ["create-stopped-containers"]
    for action in plan["phases"][0]["actions"]:
        assert action["argv"][:3] == ["env", "SPARKRING_CREATE_ONLY=1", "bash"]
        assert "launch-rank.sh" in action["argv"][3]
        assert action["checks"][0]["json"] == {"verified": True}
        assert plan_digest(preparation) in " ".join(action["checks"][0]["argv"])
    with pytest.raises(ValueError, match="Compose creation does not support cache diagnostic sites"):
        build_runtime_plan(preparation, "create", container_backend="compose")


def test_prepared_nccl_trace_keeps_structured_creation(monkeypatch):
    preparation = prepared()
    preparation["spec"]["site"].update(runtime_profile="tp4-dcp1", nccl_debug="INFO")
    public = deploy_selection.selection(prepared()["spec"], Path(__file__).resolve().parents[1] / "runtime/glm53-spark-mtp3-mesh")
    public["receipt"] = {"schema": "sparkring-r35-image-receipt/v1"}
    monkeypatch.setattr(deploy_selection, "selection", lambda *a: public)
    preparation["network_verification"]["spec_sha256"] = plan_digest(preparation["spec"])
    plan = build_runtime_plan(preparation, "create", container_backend="compose")
    assert [phase["id"] for phase in plan["phases"]] == ["plan-stopped-containers", "create-stopped-containers"]
    assert "runtime/common/glm_launch.py" in json.dumps(plan)
