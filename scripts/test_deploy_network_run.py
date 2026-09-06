"""Network execution uses fake probes and commands; no host networking is changed."""

import copy
import json
import os
from types import SimpleNamespace

import pytest

from scripts.deploy_engine import execute_plan, validate_plan
from scripts.deploy_network_run import _guard, _network_local, build_network_plan
from scripts.test_deploy_runtime import prepared
from scripts.test_deploy_suite import configured_inventory


def changed():
    preparation = prepared()
    inventory = configured_inventory(preparation["spec"])
    host = inventory["hosts"]["spark-r0"]
    host["interfaces"][1]["hw_tc_offload"] = False
    return preparation, inventory


def test_plan_noop_and_configuration_are_offline():
    p = prepared()
    plan = build_network_plan(p, configured_inventory(p["spec"]))
    validate_plan(plan)
    assert len(plan["phases"]) == 1
    p, facts = changed()
    plan = build_network_plan(p, facts)
    validate_plan(plan)
    assert len(plan["phases"][-1]["actions"]) == 1
    assert plan["phases"][-1]["actions"][0]["risk"] == "mutates-host"
    assert "archive-network-config" in json.dumps(plan)


def test_reload_is_one_host_and_requires_explicit_permission(tmp_path):
    p, facts = changed()
    for f in facts["hosts"].values():
        for r in f["rdma"]:
            r["devlink"]["parameters"]["hairpin_num_queues"]["value"] = 0
    plan = build_network_plan(p, facts)
    actions = plan["phases"][-1]["actions"]
    assert len(actions) == 1 and actions[0]["risk"] == "driver-reload"
    calls = []
    with pytest.raises(ValueError, match="driver-reload"):
        execute_plan(
            plan,
            tmp_path / "receipt.json",
            plan["sha256"],
            runner=lambda *a: calls.append(a),
        )
    assert calls == []


@pytest.mark.parametrize("blocker", ["container", "rdma", "unknown"])
def test_network_plan_rejects_active_or_unknown_users(blocker):
    p, facts = changed()
    row = facts["hosts"]["spark-r0"]
    if blocker == "container":
        row["docker"]["containers"] = [{"state": "running"}]
    else:
        row["network"]["rdma_resources"] = (
            None if blocker == "unknown" else [{"pid": 123}]
        )
    with pytest.raises(ValueError):
        build_network_plan(p, facts)


def test_remote_check_detects_inventory_drift_before_commands(tmp_path):
    p, inv = changed()
    host = p["spec"]["hosts"][0]
    host["backup_dir"] = str(tmp_path / "backup")
    facts = inv["hosts"][host["host"]]
    payload = {
        "host": host,
        "request": {},
        "guard": copy.deepcopy(_guard(facts, host)),
        "commands": [{"id": "change"}],
    }
    facts["management"]["interface"] = "wrong"
    with pytest.raises(ValueError, match="changed since"):
        _network_local(payload, "check", collect=lambda _: facts)
    assert not (tmp_path / "backup").exists()


def test_remote_journal_captures_backups_and_stops_at_reload(tmp_path, monkeypatch):
    monkeypatch.setattr(os, "geteuid", lambda: 0, raising=False)
    p, inv = changed()
    host = p["spec"]["hosts"][0]
    host["backup_dir"] = str(tmp_path / "backup")
    facts = inv["hosts"][host["host"]]
    payload = {
        "host": host,
        "request": {},
        "guard": _guard(facts, host),
        "identity": "f" * 64,
        "rediscover": True,
        "commands": [
            {
                "id": "addresses",
                "argv": ["fake-read"],
                "risk": "read-only",
                "stdout_path": str(tmp_path / "backup/addresses.json"),
            },
            {
                "id": "reload",
                "argv": ["fake-reload"],
                "risk": "driver-reload",
                "stop_after": True,
            },
            {"id": "must-not-run", "argv": ["fake-bad"], "risk": "mutates-host"},
        ],
    }
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="[]", stderr="")

    assert _network_local(payload, "apply", collect=lambda _: facts, run=run)[
        "rediscover"
    ]
    assert calls == [["fake-read"], ["fake-reload"]]
    assert json.loads((tmp_path / "backup/execution.json").read_text())["complete"]
    assert _network_local(payload, "verify", collect=lambda _: facts)["complete"]
    facts["interfaces"][1]["mtu"] = 1200
    with pytest.raises(ValueError, match="changed after"):
        _network_local(payload, "verify", collect=lambda _: facts)


def test_remote_source_compiles_without_local_imports():
    p, inv = changed()
    plan = build_network_plan(p, inv)
    for phase in plan["phases"]:
        for action in phase["actions"]:
            compile(action["argv"][-1], "remote-network", "exec")
