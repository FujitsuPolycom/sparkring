"""Network execution uses fake probes and commands; no host networking is changed."""

import ast
import copy
import json
import os
from types import SimpleNamespace

import pytest

from scripts.deploy_engine import execute_plan, validate_plan
from scripts.deploy_network_run import _guard, _network_local, build_network_plan, remote_command
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
    assert plan["driver_pending"] == []
    p, facts = changed()
    plan = build_network_plan(p, facts)
    validate_plan(plan)
    assert len(plan["phases"][-1]["actions"]) == 1
    assert plan["phases"][-1]["actions"][0]["risk"] == "mutates-host"
    assert "archive-network-config" in json.dumps(plan)
    # The only change is hardware TC offload: a driver step on a host whose
    # NetworkManager settings already match.
    host = plan["network"]["hosts"][0]
    assert host["action"] == "none" and host["driver_action"] == "apply"
    assert "hw-tc-offload" in plan["phases"][-1]["actions"][0]["argv"][-1]
    assert plan["driver_pending"] == ["spark-r0"] and not plan["driver_reload"]


def reload_needed(facts):
    for row in facts["hosts"].values():
        for function in row["rdma"]:
            function["devlink"]["parameters"]["hairpin_queue_size"]["value"] = 1024
    return facts


def remote_definitions(argv):
    """Define the functions of a remote helper command in an empty namespace."""
    source = argv[-1]
    namespace = {"__name__": "remote-network"}
    exec(compile(source[: source.rindex("\nimport json\n")], "remote-network", "exec"), namespace)
    return namespace


def remote_payload(argv):
    """Return the payload literal of print(json.dumps(_network_local(payload, operation)))."""
    call = ast.parse(argv[-1]).body[-1].value.args[0].args[0]
    return ast.literal_eval(call.args[0])


def test_deferred_driver_plan_executes_only_networkmanager_changes():
    p = prepared()
    facts = reload_needed(configured_inventory(p["spec"]))
    plan = build_network_plan(p, facts, defer_driver=True)
    validate_plan(plan)
    assert plan["driver_reload"] is False
    assert plan["driver_pending"] == ["spark-r0", "spark-r1", "spark-r2", "spark-r3"]
    assert [phase["id"] for phase in plan["phases"]] == ["check-network-inventory"]
    # A host with NetworkManager changes still gets them, without driver steps.
    port = p["spec"]["hosts"][1]["data_interfaces"][0]
    interface = next(i for i in facts["hosts"]["spark-r1"]["interfaces"] if i["name"] == port["netdev"])
    interface["network_manager"]["autoconnect"] = False
    port["replace_connection_uuid"] = interface["network_manager"]["connection_uuid"]
    plan = build_network_plan(p, facts, defer_driver=True)
    actions = plan["phases"][-1]["actions"]
    assert [(a["host"], a["risk"]) for a in actions] == [("spark-r1", "mutates-host")]
    commands = remote_payload(actions[0]["argv"])["commands"]
    assert any(c["argv"][2:5] == ["nmcli", "connection", "add"] for c in commands)
    # Backups still record devlink parameters, read-only; nothing changes them.
    assert not any(
        c["risk"] != "read-only" and ("devlink" in c["argv"] or "ethtool" in c["argv"])
        for c in commands
    )
    # Without deferral the same inventory plans one driver reload instead.
    actions = build_network_plan(p, facts)["phases"][-1]["actions"]
    assert [(a["host"], a["risk"]) for a in actions] == [("spark-r0", "driver-reload")]


def test_driver_only_drift_is_not_blocked_by_gpu_users_when_deferred():
    p = prepared()
    facts = reload_needed(configured_inventory(p["spec"]))
    facts["hosts"]["spark-r0"]["gpu"] = {"compute_processes": [4242]}
    facts["hosts"]["spark-r0"]["network"]["rdma_resources"] = [{"pid": 4242, "comm": "python3"}]
    plan = build_network_plan(p, facts, defer_driver=True)
    assert "spark-r0" in plan["driver_pending"]
    assert not plan["network"]["hosts"][0]["apply_permitted"]
    with pytest.raises(ValueError, match="GPU processes"):
        build_network_plan(p, facts)


def test_driver_restart_refused_when_only_management_path_is_the_administration_network():
    p = prepared()
    facts = reload_needed(configured_inventory(p["spec"]))
    p["spec"]["hosts"][2]["management_netdev"] = "sr-control"
    management = facts["hosts"]["spark-r2"]["management"]
    management["interface"] = "sr-control"
    management["route_to_controller"]["dev"] = "sr-control"
    with pytest.raises(ValueError, match="spark-r2: a driver reload would interrupt its only management path"):
        build_network_plan(p, facts)
    # Offload needs no restart and stays executable on such a host.
    for function in facts["hosts"]["spark-r2"]["rdma"]:
        function["devlink"]["parameters"]["hairpin_queue_size"]["value"] = 8192
    facts["hosts"]["spark-r2"]["interfaces"][1]["hw_tc_offload"] = False
    for name in ("spark-r0", "spark-r1", "spark-r3"):
        for function in facts["hosts"][name]["rdma"]:
            function["devlink"]["parameters"]["hairpin_queue_size"]["value"] = 8192
    plan = build_network_plan(p, facts)
    assert [a["host"] for a in plan["phases"][-1]["actions"]] == ["spark-r2"]
    assert plan["phases"][-1]["actions"][0]["risk"] == "mutates-host"
    # Deferred plans leave the restart to the node service.
    reload_needed(facts)
    assert build_network_plan(p, facts, defer_driver=True)["driver_pending"]


def test_unknown_hairpin_state_is_refused_only_by_driver_execution():
    p = prepared()
    facts = configured_inventory(p["spec"])
    del facts["hosts"]["spark-r3"]["rdma"][0]["devlink"]["reload"]
    with pytest.raises(ValueError, match="spark-r3: devlink reload statistics are unavailable"):
        build_network_plan(p, facts)
    plan = build_network_plan(p, facts, defer_driver=True)
    assert plan["driver_pending"] == ["spark-r3"]
    assert plan["network"]["hosts"][3]["driver_action"] == "unknown"


def test_guard_ignores_reload_counter_but_not_parameters():
    p = prepared()
    facts = configured_inventory(p["spec"])["hosts"]["spark-r0"]
    host = p["spec"]["hosts"][0]
    before = copy.deepcopy(_guard(facts, host))
    for function in facts["rdma"]:
        function["devlink"]["reload"] = {"driver_reinit": 7, "failed": False}
    assert _guard(facts, host) == before
    del facts["rdma"][0]["devlink"]["reload"]
    assert _guard(facts, host) == before
    facts["rdma"][0]["devlink"]["parameters"]["hairpin_queue_size"]["value"] = 1024
    assert _guard(facts, host) != before
    # The guard leaves the inventory it reads unchanged.
    assert facts["rdma"][1]["devlink"]["reload"]["driver_reinit"] == 7


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


def test_remote_guard_runs_without_repository_modules():
    p, inv = changed()
    host = p["spec"]["hosts"][0]
    facts = inv["hosts"][host["host"]]
    remote = remote_definitions(remote_command({"host": host}, "check"))
    assert remote["_guard"](facts, host) == _guard(facts, host)
    facts["rdma"][0]["devlink"]["reload"]["driver_reinit"] = 9
    assert remote["_guard"](facts, host) == _guard(facts, host)


def test_network_plan_allows_gpu_less_helpers_and_kernel_queue_pairs():
    p, facts = changed()
    row = facts["hosts"]["spark-r0"]
    row["docker"]["containers"] = [{"name": "netadm", "state": "running"}]
    row["gpu"] = {**row.get("gpu", {}), "compute_processes": []}
    row["network"]["rdma_resources"] = [{"ifname": "rocep1s0f0", "comm": "ib_core", "type": "GSI"}]
    assert build_network_plan(p, facts)["phases"]
    row["gpu"]["compute_processes"] = [4242]
    with pytest.raises(ValueError):
        build_network_plan(p, facts)
