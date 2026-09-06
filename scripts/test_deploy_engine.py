import json

import pytest

from scripts.deploy_engine import execute_plan, seal_plan


def plan():
    return seal_plan(
        {
            "schema": "sparkring-deploy-plan/v1",
            "phases": [
                {
                    "id": "prepare",
                    "actions": [
                        {
                            "host": f"node{i}",
                            "risk": "mutates-host",
                            "argv": ["prepare"],
                            "checks": [{"argv": ["preflight"]}],
                            "verify": {"argv": ["verify"], "stdout": "ready"},
                        }
                        for i in range(4)
                    ],
                },
                {
                    "id": "install",
                    "actions": [
                        {
                            "host": "node0",
                            "risk": "mutates-host",
                            "argv": ["install"],
                            "verify": {"argv": ["verify"], "stdout": "ready"},
                        }
                    ],
                },
            ],
        }
    )


def success(host, argv, timeout):
    return {
        "returncode": 0,
        "stdout": "ready" if argv == ["verify"] else "",
        "stderr": "",
        "uncertain": False,
    }


def test_barriers_and_resume(tmp_path):
    calls = []

    def run(host, argv, timeout):
        calls.append((host, argv[0]))
        return success(host, argv, timeout)

    p = plan()
    receipt = tmp_path / "run.json"
    assert execute_plan(p, receipt, p["sha256"], runner=run)["complete"]
    first_prepare = next(i for i, c in enumerate(calls) if c[1] == "prepare")
    assert all(c[1] == "preflight" for c in calls[:first_prepare])
    assert (
        sum(
            c[1] == "verify"
            for c in calls[: next(i for i, c in enumerate(calls) if c[1] == "install")]
        )
        == 4
    )
    calls.clear()
    execute_plan(p, receipt, p["sha256"], runner=run, resume=True)
    assert all(c[1] == "verify" for c in calls)


def test_failed_preflight_never_mutates(tmp_path):
    calls = []

    def run(host, argv, timeout):
        calls.append(argv[0])
        return {
            **success(host, argv, timeout),
            "returncode": 1 if host == "node2" else 0,
        }

    p = plan()
    with pytest.raises(RuntimeError, match="preflight"):
        execute_plan(p, tmp_path / "r.json", p["sha256"], runner=run)
    assert set(calls) == {"preflight"}


def test_uncertain_change_blocks_resume_and_later_phases(tmp_path):
    calls = []

    def run(host, argv, timeout):
        calls.append(argv[0])
        return (
            {**success(host, argv, timeout), "returncode": 124, "uncertain": True}
            if host == "node1" and argv == ["prepare"]
            else success(host, argv, timeout)
        )

    p = plan()
    path = tmp_path / "r.json"
    with pytest.raises(RuntimeError, match="phase failed"):
        execute_plan(p, path, p["sha256"], runner=run)
    assert "install" not in calls
    assert (
        json.loads(path.read_text())["actions"]["prepare:node1"]["state"] == "uncertain"
    )
    with pytest.raises(ValueError, match="uncertain"):
        execute_plan(p, path, p["sha256"], runner=success, resume=True)


def test_drift_and_explicit_model_authorization(tmp_path):
    p = plan()
    p["phases"][0]["actions"][0]["argv"] = ["changed"]
    with pytest.raises(ValueError, match="Plan changed"):
        execute_plan(p, tmp_path / "r.json", p["sha256"], runner=success)
    p = plan()
    p["phases"][0]["actions"][0]["risk"] = "starts-model"
    p = seal_plan(p)
    with pytest.raises(ValueError, match="model-action"):
        execute_plan(p, tmp_path / "m.json", p["sha256"], runner=success)


def test_locked_receipt_cannot_execute(tmp_path):
    p = plan()
    path = tmp_path / "r.json"
    (tmp_path / "r.json.lock").write_text("operator")
    with pytest.raises(ValueError, match="locked"):
        execute_plan(p, path, p["sha256"], runner=success)


def test_resume_drift_records_failed_verification_and_revokes_complete(tmp_path):
    p = plan()
    receipt = tmp_path / "run.json"
    execute_plan(p, receipt, p["sha256"], runner=success)

    def changed(*_):
        return {"returncode": 1, "stdout": "", "stderr": "host changed"}

    with pytest.raises(RuntimeError, match="no longer verifies"):
        execute_plan(p, receipt, p["sha256"], runner=changed, resume=True)
    result = json.loads(receipt.read_text())
    assert result["complete"] is False
    assert result["failure"]["stage"] == "resume_verification"
