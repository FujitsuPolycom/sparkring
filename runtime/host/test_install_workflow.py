"""Public CLI acceptance with host/SSH boundaries simulated, not the coordinator."""
import json

import pytest

from runtime.host import controller, install_workflow as flow, node, rollout
from runtime.host.install_errors import NeedsInput
from runtime.host.test_fabric_ssh import cluster
from scripts import sparkring

PROFILE = "qwen38-flash-next-tp2"


@pytest.fixture
def machine(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = cluster(2)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(flow, "require_head", lambda *_: value["plan"]["nodes"][0]["node_id"])
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    monkeypatch.setattr(flow.distribution, "identity", lambda _: "a" * 40)
    monkeypatch.setattr(flow.distribution, "bundle", lambda root, dest: dest.write_bytes(b"retained source"))
    monkeypatch.setattr(flow.discovery, "ssh", lambda *a: json.dumps({"model_path": "/srv/models/cached"}))
    events = []
    previous = controller.STATE / "deployments" / "previous"
    previous.mkdir(parents=True)
    node.save(controller.STATE, "active.json", {"path": str(previous)})
    monkeypatch.setattr(flow, "check_workloads", lambda *a: events.append("check-workloads"))
    monkeypatch.setattr(flow.retained_source, "checkout", lambda *a: events.append("rollback-source"))
    class Transport:
        def __init__(self, cluster, directory):
            self.hosts = cluster["plan"]["spec"]["hosts"]
        def verify(self):
            events.append("verify-fabric")
            return {"transport": "fiber-ssh", "caller_relay": False}
    monkeypatch.setattr(flow.fabric_ssh, "Transport", Transport)
    class Assets:
        def __init__(self, *a):
            pass
        def sync_packages(self):
            events.append("update-workers")
        def images(self, card):
            events.append("fill-missing-image")
        def runner(self, directory, previous=None):
            def run(host, argv, timeout):
                events.append("prepare:" + argv[1])
                return {"returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False}
            return run
    monkeypatch.setattr(flow.install_assets, "Assets", Assets)
    def operation(path, action, **kwargs):
        events.append(("previous" if path == previous else "candidate") + ":" + action)
        return {"verified": True}
    monkeypatch.setattr(flow.retained_source, "apply", operation)
    return events, previous, Assets, operation


def command(*extra):
    return sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", *extra])


def test_documented_command_updates_prepares_switches_and_emits_only_json(machine, capsys):
    events, previous, _, _ = machine
    assert command() == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "complete" and result["transfer"]["caller_relay"] is False
    assert events.index("update-workers") < events.index("fill-missing-image") < events.index("previous:down")
    assert events.index("prepare:model-check") < events.index("previous:down") < events.index("candidate:up") < events.index("candidate:verify")
    assert "Progress:" in out.err and "Model ready:" in out.err
    assert rollout.active(controller.STATE) != previous


def test_storage_failure_keeps_current_model_and_returns_actionable_json(machine, monkeypatch, capsys):
    events, previous, assets, _ = machine
    def full(*args):
        raise NeedsInput("Node 1 needs storage", field="storage", details={"rank": 1})
    monkeypatch.setattr(assets, "images", full)
    assert command() == 3
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "needs_input" and result["field"] == "storage"
    assert not any(e.endswith(":down") for e in events)
    assert rollout.active(controller.STATE) == previous


def test_failed_start_recovers_previous_using_the_same_public_command(machine, monkeypatch, capsys):
    events, previous, _, operation = machine
    def fail(path, action, **kwargs):
        result = operation(path, action)
        if path != previous and action == "verify":
            raise ValueError("injected API failure")
        return result
    monkeypatch.setattr(flow.retained_source, "apply", fail)
    assert command() == 2
    result = json.loads(capsys.readouterr().out)
    assert result["transaction"]["state"] == "failed-recovered"
    assert events[-4:] == ["candidate:down", "previous:down", "previous:up", "previous:verify"]
    assert rollout.active(controller.STATE) == previous


def test_plan_never_updates_workers_or_stops_model(machine, capsys):
    events, previous, _, _ = machine
    assert command("--plan") == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "planned" and result["replaces"] == str(previous)
    assert events == []


def test_noninteractive_missing_approval_is_a_result_not_a_prompt(machine, capsys):
    assert sparkring.main(["install", "--profile", PROFILE, "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["field"] == "approval"


def test_tp4_command_adopts_the_discovered_mesh_without_network_changes(machine, monkeypatch, capsys):
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    def remote(target, argv):
        if "native-mesh" in argv:
            rank = int(argv[-1])
            return json.dumps({"mesh": {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json",
                                                      "site_sha256": "c" * 64, "plan_sha256": "d" * 64},
                                      "host_ip": f"192.0.2.{110 + rank}", "interface": "eth0", "unit": "sparkring-mesh.service"}})
        return json.dumps({"model_path": "/srv/models/cached"})
    monkeypatch.setattr(flow.discovery, "ssh", remote)
    assert sparkring.main(["install", "--profile", "qwen38-flash-next-qad-tp4", "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    from runtime.common import installer
    lock = installer.load(result["deployment"])
    assert result["nodes"] == 4 and "native_mesh" not in lock["site_input"]
    assert [r["host_ip"] for r in lock["site"]["ranks"]] == [f"192.0.2.{110 + rank}" for rank in range(4)]


def test_wrong_node_is_refused_before_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(flow.sys, "platform", "linux")
    monkeypatch.setattr(flow.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(flow.distribution, "installed", lambda _: True)
    monkeypatch.setattr(node, "read", lambda *a: {"node_id": "wrong"})
    with pytest.raises(ValueError, match="not the enrolled Node A"):
        flow.require_head(cluster(2))


def test_glm_mesh_conflict_returns_input_before_updates_or_model_stop(machine, monkeypatch, capsys):
    events, previous, _, _ = machine
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    def remote(target, argv):
        if "assets" in argv:
            return json.dumps({"model_path": "/srv/models/cached"})
        return json.dumps({"available": False, "occupied": ["/etc/sparkring/managed-mesh"]})
    monkeypatch.setattr(flow.discovery, "ssh", remote)
    assert sparkring.main(["install", "--profile", "glm53-flash-spark-tp4-dcp1-sparkcache", "--yes", "--json"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "fabric" and result["state"] == "needs_input"
    assert not events and rollout.active(controller.STATE) == previous
