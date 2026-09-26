"""Public CLI acceptance with host/SSH boundaries simulated, not the coordinator."""
import base64
import copy
import fnmatch
import json
import os
from pathlib import Path
import sys
import threading
import uuid

import pytest

from runtime.common import distribution, installer
from runtime.host import (control, controller, hairpin, hairpin_ring, install_assets, install_workflow as flow, node,
                          rollout, single_uplink, test_hairpin, topology)
from runtime.host.install_errors import NeedsInput
from runtime.host.test_appliance import nodes
from runtime.host.test_fabric_ssh import cluster as fabric_cluster
from runtime.host.test_hairpin_ring import OLDER, Ring, document, kept, needing
from scripts import hairpin_setting, installer_runner, sparkring

PROFILE = "qwen38-flash-next-tp2"
TP4 = "qwen38-flash-next-qad-tp4"


def cluster(size):
    """A recorded cluster; four-Spark inspect documents carry kept ConnectX hairpin status."""
    value = fabric_cluster(size)
    kept(value["plan"])
    return value


# The asset preparer whose runner the installer uses, captured before the
# machine fixture replaces it with a simulation.
ASSETS = install_assets.Assets


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
    monkeypatch.setattr(flow, "check_workloads", lambda *a, **k: events.append("check-workloads"))
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
        def runner(self, directory, previous=None, images=None):
            def run(host, argv, timeout):
                if argv[1] in ("model", "image"):
                    images.result()
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


def prepared_runner(monkeypatch, *, images, models):
    """Install through the real asset runner; SSH actions succeed and asset transfers are simulated."""
    monkeypatch.setattr(installer_runner.Runner, "_call",
                        lambda self, target, argv, timeout: {"returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False})

    class Assets(ASSETS):
        def sync_packages(self):
            return {"updated": []}
    Assets.images, Assets.models = images, models
    monkeypatch.setattr(flow.install_assets, "Assets", Assets)


def test_checkpoint_phase_waits_for_pending_image_distribution(machine, monkeypatch, capsys):
    image = {"present": False}
    checkpoint_phase, checkpoint_work = threading.Event(), threading.Event()
    call = installer_runner.Runner.__call__
    def observed(self, target, argv, timeout):
        if argv[1] == "model":
            checkpoint_phase.set()
        return call(self, target, argv, timeout)
    monkeypatch.setattr(installer_runner.Runner, "__call__", observed)
    def images(self, card):
        assert checkpoint_phase.wait(10), "the checkpoint phase did not start while image distribution was pending"
        # The distribution is still pending here, so checkpoint work must not start.
        assert not checkpoint_work.wait(1), "checkpoint work started before image distribution finished"
        image["present"] = True
    def models(self, lock, runner, previous=None):
        checkpoint_work.set()
        # A checkpoint download or repair runs the serving image with --pull never.
        if not image["present"]:
            raise RuntimeError("docker: Error response from daemon: No such image: " + lock["selection"]["image_id"])
        return {"donor_rank": 0}
    prepared_runner(monkeypatch, images=images, models=models)
    assert command() == 0
    assert json.loads(capsys.readouterr().out)["state"] == "complete"


@pytest.mark.parametrize("failure", ["storage", "download", "interrupt"])
def test_failed_or_interrupted_preparation_is_repeated_by_the_same_command(machine, monkeypatch, capsys, failure):
    events, previous, _, _ = machine
    calls = {"images": 0, "models": 0}
    def images(self, card):
        calls["images"] += 1
        if failure == "storage" and calls["images"] == 1:
            raise NeedsInput("Node 1: insufficient image-import space.", field="storage", details={"rank": 1})
    def models(self, lock, runner, previous=None):
        calls["models"] += 1
        if calls["models"] == 1 and failure == "download":
            raise RuntimeError("Hugging Face download failed: 503")
        if calls["models"] == 1 and failure == "interrupt":
            raise KeyboardInterrupt
        return {"donor_rank": 0}
    prepared_runner(monkeypatch, images=images, models=models)
    if failure == "interrupt":
        with pytest.raises(KeyboardInterrupt):
            command()
        capsys.readouterr()
    else:
        assert command() == (3 if failure == "storage" else 2)
        out = capsys.readouterr()
        first = json.loads(out.out)
        if failure == "storage":
            assert first["field"] == "storage" and calls["models"] == 0
        else:
            # Both ranks' checkpoint actions report one failed preparation.
            assert first["state"] == "failed" and calls["models"] == 1
            assert "Checkpoint preparation failed: Hugging Face download failed: 503" in out.err
    assert rollout.active(controller.STATE) == previous and not any(e.endswith(":down") for e in events)
    candidate = next(p for p in (controller.STATE / "deployments").iterdir() if p != previous)
    incomplete = installer.read(candidate / "state.json")
    assert incomplete["operation"] == "prepare" and not incomplete["complete"]

    assert command() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "complete" and Path(result["deployment"]) == candidate
    assert rollout.active(controller.STATE) == candidate
    state = installer.read(candidate / "state.json")
    assert state == {**state, "generation": incomplete["generation"] + 1, "operation": "prepare", "complete": True}
    # The incomplete receipt stays for inspection; the repeat used its own.
    assert (candidate / incomplete["receipt"]).exists() and state["receipt"] != incomplete["receipt"]


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


def test_early_error_does_not_report_a_previous_transaction(machine, monkeypatch, capsys):
    node.save(controller.STATE, 'transaction.json', {'state': 'complete', 'candidate': 'previous-run', 'complete': True})
    def fail(*args):
        raise ValueError('Current discovery failed')
    monkeypatch.setattr(flow, 'refresh_cluster', fail)
    assert command() == 2
    result = json.loads(capsys.readouterr().out)
    assert result['message'] == 'Current discovery failed' and 'transaction' not in result


@pytest.mark.parametrize("profile", ["qwen38-flash-next-qad-tp4", "glm53-flash-nvfp4-spark-tp4", "mimo-v26-flash-rl-tp4"])
def test_tp4_command_adopts_the_discovered_mesh_without_network_changes(machine, monkeypatch, capsys, profile):
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
    assert sparkring.main(["install", "--profile", profile, "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    from runtime.common import installer
    lock = installer.load(result["deployment"])
    assert result["nodes"] == 4 and "native_mesh" not in lock["site_input"]
    assert [r["host_ip"] for r in lock["site"]["ranks"]] == [f"192.0.2.{110 + rank}" for rank in range(4)]
    # Every installer profile runs on the shared image without an explicit lock.
    assert lock["image_runtime"] == installer.installer_image.default_lock()
    assert result["image_id"] == lock["image_runtime"]["image_id"] and lock["backend"] == "compose"


def test_wrong_node_is_refused_before_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(flow.sys, "platform", "linux")
    monkeypatch.setattr(flow.os, "geteuid", lambda: 0, raising=False)
    monkeypatch.setattr(flow.distribution, "installed", lambda _: True)
    monkeypatch.setattr(node, "read", lambda *a: {"node_id": "wrong"})
    with pytest.raises(ValueError, match="not the enrolled Node A"):
        flow.require_head(cluster(2))


def test_profiles_outside_the_shared_image_are_refused_before_host_changes(machine, monkeypatch, capsys):
    events, previous, _, _ = machine
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    def remote(target, argv):
        if "discover_contract(" in argv[-1]:
            return json.dumps({"model_path": "/srv/models/cached"})
        if "native-mesh" in argv:
            return json.dumps({"mesh": None})
        return json.dumps({"available": False, "occupied": ["/etc/sparkring/managed-mesh"]})
    monkeypatch.setattr(flow.discovery, "ssh", remote)
    assert sparkring.main(["install", "--profile", "glm53-flash-spark-tp4-dcp1-sparkcache", "--yes", "--json"]) == 2
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "failed" and "own guide" in result["message"]
    assert not events and rollout.active(controller.STATE) == previous


def test_missing_sudo_returns_access_input_with_fix_before_any_change(monkeypatch):
    value = cluster(2)
    calls = []
    worker = value["plan"]["spec"]["hosts"][1]["host"]
    def invoke(host, argv):
        calls.append((host, argv))
        if host == worker:
            raise RuntimeError(host + ": sudo: a password is required")
        return ""
    from runtime.host.install_errors import NeedsInput
    with pytest.raises(NeedsInput) as caught:
        flow.check_access(value, invoke=invoke)
    document = caught.value.document()
    assert document["field"] == "access" and document["state"] == "needs_input"
    assert [row["rank"] for row in document["details"]["hosts"]] == [1]
    assert "NOPASSWD" in document["details"]["hosts"][0]["fix"] and "visudo" in document["details"]["hosts"][0]["fix"]
    assert all(argv == ["sudo", "-n", "true"] for _, argv in calls)


@pytest.mark.parametrize("approved", [False, True])
def test_unrelated_gpu_containers_are_stopped_only_with_approval(monkeypatch, tmp_path, approved):
    from scripts import installer_runner
    lock = {"id": "new", "site": {"ranks": [{"rank": 0, "host": "root@a"}]}}
    monkeypatch.setattr(flow.installer, "load", lambda _: lock)
    state = {"running": True, "stopped": []}

    def ssh(host, argv, timeout=None):
        if argv[:2] == ["docker", "stop"]:
            state["stopped"].append(argv[-1])
            state["running"] = False
            return ""
        containers = [{"name": "qwen-manual-r0", "labels": {}}] if state["running"] else []
        return json.dumps({"gpu_containers": containers})
    monkeypatch.setattr(installer_runner, "ssh", ssh)
    monkeypatch.setattr(installer_runner, "check_facts", lambda facts, row: None)

    def check(facts, permitted, rank, managed_prepared=False):
        if facts["gpu_containers"]:
            raise ValueError("busy")
    monkeypatch.setattr(installer_runner, "check_workloads", check)
    approvals = []
    stop = (lambda host, names: approvals.append((host, names))) if approved else None
    from runtime.host.install_errors import NeedsInput
    if approved:
        flow.check_workloads(tmp_path, None, stop=stop)
        assert approvals == [("root@a", ["qwen-manual-r0"])] and state["stopped"] == ["qwen-manual-r0"]
    else:
        with pytest.raises(NeedsInput) as caught:
            flow.check_workloads(tmp_path, None, stop=stop)
        assert caught.value.document()["details"]["containers"] == ["qwen-manual-r0"] and not state["stopped"]


def four(monkeypatch):
    """Record a four-Spark ring whose Sparks each run an adoptable native mesh; returns the cluster."""
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
    return value


def simulate(monkeypatch, value):
    """Serve hairpin commands from simulated Sparks; the real ring procedure runs against them."""
    ring = Ring(value["plan"])
    monkeypatch.setattr(hairpin_ring, "remote", ring.invoke)
    monkeypatch.setattr(hairpin_ring, "local", ring.run_local)
    monkeypatch.setattr(hairpin_ring, "POLL", 0)
    monkeypatch.setattr(hairpin_ring, "DISPATCH_RETRY", 0)
    return ring


def test_ring_that_needs_restarts_applies_the_setting_before_the_model_transaction(machine, monkeypatch, capsys):
    events, previous, _, _ = machine
    value = four(monkeypatch)
    needing(value["plan"], [3])
    received = []

    def ensure(plan, **options):
        events.append("ensure")
        received.append(options)
        options["update_workers"]()
        options["record"].update(state="complete", path="/var/lib/sparkring/controller/hairpin/1/hairpin.json",
                                 ranks=[{"rank": 3, "after": "kept", "error": None, "outcome": "applied"}])
        return kept(copy.deepcopy(plan))
    monkeypatch.setattr(flow.hairpin_ring, "ensure", ensure)
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "complete" and events.count("ensure") == 1
    assert events.index("verify-fabric") < events.index("ensure") < events.index("update-workers")
    assert events.index("update-workers") < events.index("fill-missing-image") < events.index("previous:down")
    assert received[0]["approved"] is True and received[0]["update_workers"].__name__ == "sync_packages"
    steps = result["steps"]
    assert steps.index("update-workers") + 1 == steps.index("apply-hairpin-setting")
    assert result["hairpin"]["required"] and result["hairpin"]["receipt"].endswith("hairpin.json")
    assert result["hairpin"]["ranks"][3]["before"] == "restart" and result["hairpin"]["ranks"][3]["after"] == "kept"
    assert hairpin_ring.COMPLETE in out.err


def test_ring_that_needs_restarts_without_yes_asks_for_approval(machine, monkeypatch, capsys):
    events, _, _, _ = machine
    value = four(monkeypatch)
    needing(value["plan"], [3])
    monkeypatch.setattr(flow.hairpin_ring, "ensure", lambda *a, **k: pytest.fail("ensure ran without approval"))
    assert sparkring.main(["install", "--profile", TP4, "--json"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "approval"
    assert result["message"].startswith("This installation also applies the ConnectX hairpin setting: each function "
                                        "of rank 3 that needs it restarts its driver once")
    assert events == []


def test_mixed_revisions_plan_without_changes_then_update_and_apply(machine, monkeypatch, capsys):
    events, _, assets, _ = machine
    value = four(monkeypatch)
    del value["plan"]["nodes"][2]["hairpin"]
    value["plan"]["nodes"][2]["revision"] = OLDER
    ring = simulate(monkeypatch, value)
    assert sparkring.main(["install", "--profile", TP4, "--plan", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["hairpin"]["ranks"][2]["message"] == ("Rank 2 (spark2) runs SparkRing bbbbbbbbbbbb; its hairpin state "
                                                        "is read after the update to aaaaaaaaaaaa.")
    assert events == [] and ring.calls == []

    def sync_packages(self):
        events.append("update-workers")
        ring.update()
    monkeypatch.setattr(assets, "sync_packages", sync_packages)
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "complete"
    # The workers were updated before the ring was read again; nothing needed a restart.
    assert ring.calls[0] == ("update-workers",) and not ring.changing()
    assert ring.commands("lldpcli", "update")


def test_busy_ring_needs_input_without_approval_or_restart(machine, monkeypatch, capsys):
    events, previous, _, _ = machine
    value = four(monkeypatch)
    needing(value["plan"], [3])
    ring = simulate(monkeypatch, value)
    ring.sparks[1].busy = [{"kind": "unit", "unit": "sparkring-mesh.service", "active_state": "active",
                            "processes": [], "detail": "sparkring-mesh.service is active"}]
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "driver" and "1 Spark is in use" in result["message"]
    assert result["details"]["receipt"].endswith("hairpin.json")
    assert not ring.changing()
    assert not any(e.endswith(":down") for e in events) and rollout.active(controller.STATE) == previous


def test_plan_lists_the_hairpin_step_and_changes_nothing(machine, monkeypatch, capsys):
    events, _, _, _ = machine
    value = four(monkeypatch)
    needing(value["plan"], [3])
    monkeypatch.setattr(flow.hairpin_ring, "ensure", lambda *a, **k: pytest.fail("--plan ran the hairpin step"))
    assert sparkring.main(["install", "--profile", TP4, "--plan", "--json"]) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["steps"][:3] == ["verify-fabric", "update-workers", "apply-hairpin-setting"]
    assert result["hairpin"]["required"] and [row["before"] for row in result["hairpin"]["ranks"]] == ["kept"] * 3 + ["restart"]
    assert "rank 3 spark3: restart 4 functions (1024 -> 8192)" in out.err
    assert events == []


def test_pair_never_runs_the_hairpin_step_and_accepts_the_driver_flag(machine, monkeypatch, capsys):
    monkeypatch.setattr(flow.hairpin_ring, "ensure", lambda *a, **k: pytest.fail("a pair ran the hairpin step"))
    assert command("--allow-driver-reload") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "complete" and "hairpin" not in result
    assert "apply-hairpin-setting" not in result["steps"]
    assert "--allow-driver-reload: " + controller.ALLOW_DRIVER_RELOAD in out.err


def test_first_installation_approval_lists_the_hairpin_scope(machine, monkeypatch, capsys):
    (controller.STATE / "cluster.json").unlink()
    assert sparkring.main(["install", "--profile", PROFILE, "--json"]) == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "approval"
    scope = result["details"]["scope"]
    assert "  - on a four-Spark ring: apply the ConnectX hairpin setting that four-Spark" in scope
    # Terminal mode prints the same lines under the message.
    assert sparkring.main(["install", "--profile", PROFILE]) == 3
    assert "    about 8 seconds, about 30 seconds per Spark and about 3 minutes for the" in capsys.readouterr().err


# End to end: sparkring install applies the ConnectX hairpin setting on a
# simulated four-Spark ring. Each Spark runs the real node code
# (runtime/host/hairpin.py) against test_hairpin's simulated devlink, ethtool,
# NetworkManager, WireGuard and sysfs; setup, the ring procedure and the
# installation run unchanged. Model preparation and serving are simulated as in
# the tests above.

# Breadth-first position of each rank in the administration tree, which orders
# the control addresses: Node A, its two cable neighbors, then rank 2 behind rank 1.
TREE_POSITION = {0: 0, 1: 1, 3: 2, 2: 3}
# The administration tunnels of each rank: the fabric function that carries a
# tunnel, and whether its peer is the parent (towards Node A) or a child.
TUNNELS = {0: {"cw_primary": ("child", 1), "ccw_primary": ("child", 3)},
           1: {"cw_primary": ("child", 2), "ccw_primary": ("parent", 0)},
           2: {"ccw_primary": ("parent", 1)},
           3: {"cw_primary": ("parent", 0)}}
# The ranks that a child tunnel reaches.
BEHIND = {1: [1, 2], 2: [2], 3: [3]}


def control_address(rank):
    return f"192.0.2.{10 + TREE_POSITION[rank]}"


def administered_nodes():
    """Inspect documents of a four-Spark ring whose workers SparkRing reaches only over sr-control."""
    found = nodes(4)
    for rank, current in enumerate(found):
        facts = current["facts"]
        facts["ssh_target"] = "root@" + control_address(rank)
        facts["management"].update(address=control_address(rank), interface=control.INTERFACE)
        facts["management"]["route_to_controller"]["dev"] = control.INTERFACE
    return found


class RingSpark(test_hairpin.Spark):
    """One Spark of the simulated ring, with its SparkRing package installed.

    Its four ConnectX functions carry the identity that the ring's inventory
    records and start at the probe default: hairpin_queue_size 1024 and no
    driver restart since boot. Its systemd runs the real ``hairpin.apply``
    when sparkring-hairpin.service is restarted: at once for a blocking
    restart, and before the next status read for ``--no-block``, as a run
    that ends between two polls of the ring.
    """

    def __init__(self, root, rank, document, events):
        self.root, self.rank, self.events = Path(root), rank, events
        self.time = 100.0
        self.calls, self.lines, self.requests = [], [], []
        self.booting, self.pending, self.runs = False, False, 0
        self.invocation = ""
        self.enabled, self.units = set(), {}
        self.tc, self.qp, self.pd, self.gpu = {}, [], [], ""
        self.reload, self.on_reload = {}, None
        self.drop_addresses, self.ping, self.unreachable, self.stats = False, True, set(), True
        self.rdma_down = {}
        self.hairpin_unit = {"ActiveState": "inactive", "Result": "success"}
        facts = document["facts"]
        interfaces = {row["name"]: row for row in facts["interfaces"]}
        self.functions = {}
        for index, role in enumerate(hairpin.ROLES):
            rdma = next(row for row in facts["rdma"] if row["device"] == topology.DEVICES[role])
            interface = interfaces[rdma["netdev"]]
            self.functions[rdma["pci_address"]] = {
                "role": role, "netdev": rdma["netdev"], "rdma": rdma["device"], "mac": interface["mac"].lower(),
                "values": {"hairpin_queue_size": 1024, "hairpin_num_queues": 4}, "pending": {}, "counter": 0,
                "failed": False, "offload": "on", "ipv4": interface["ipv4"][0].split("/")[0],
                "link_local": f"fe80::{rank + 1}:{index + 1}", "addressed": True,
                "connection": interface["network_manager"]["connection_uuid"], "autoconnect": "yes",
                "bound": rdma["netdev"]}
            self.sysfs(rdma["pci_address"])
        self.write(hairpin.NODE, {"schema": "sparkring-node/v1", "node_id": document["node_id"]})
        for name, text in (("proc/sys/kernel/random/boot_id", str(uuid.UUID(int=0x100 + rank))),
                           ("proc/sys/kernel/osrelease", "6.17.0-1029-nvidia"),
                           ("proc/cmdline", "BOOT_IMAGE=/vmlinuz ro quiet")):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_text(text + "\n")

    def role(self, role):
        return next(function for function in self.functions.values() if function["role"] == role)

    def administer(self, ring):
        """Record this Spark's administration tunnels in /etc/sparkring/control.json."""
        peers, links = [], []
        for role, (relation, other) in TUNNELS[self.rank].items():
            local = self.role(role)
            remote = ring[other].role("ccw_primary" if role == "cw_primary" else "cw_primary")
            allowed = ["0.0.0.0/0"] if relation == "parent" else [control_address(r) + "/32" for r in BEHIND[other]]
            peers.append({"id": f"spark{other}", "key": base64.b64encode(bytes([other + 1]) * 32).decode(),
                          "allowed_ips": allowed, "endpoint": f"[{remote['link_local']}%{local['netdev']}]:51871",
                          "netdev": local["netdev"]})
            links.append({"netdev": local["netdev"], "mac": local["mac"], "address": local["link_local"]})
        self.write(hairpin.CONTROL, {"schema": "sparkring-control/v1", "id": f"spark{self.rank}",
                                     "address": control_address(self.rank), "head": self.rank == 0,
                                     "head_address": control_address(0), "subnet": "192.0.2.8/29",
                                     "share_uplink": True, "peers": peers, "links": links,
                                     "uplink": "enP7s7" if self.rank == 0 else None})

    def host(self, **overrides):
        return super().host(**{"hostname": f"spark{self.rank}", **overrides})

    def run(self, argv, *, timeout=None, **options):
        if list(argv) == ["lldpcli", "update"]:
            self.calls.append(list(argv))
            return test_hairpin.ok()
        return super().run(argv, timeout=timeout, **options)

    def systemctl(self, args):
        if args[0] == "restart":
            assert args[-1] == hairpin.UNIT, args
            return self.start(blocking="--no-block" not in args)
        if args[0] == "list-unit-files":
            patterns = [word for word in args[1:] if not word.startswith("-")]
            return test_hairpin.ok("".join(f"{name} {'enabled' if name in self.enabled else 'disabled'} enabled\n"
                                           for name in sorted(self.units)
                                           if any(fnmatch.fnmatchcase(name, p) for p in patterns)))
        return super().systemctl(args)

    def start(self, *, blocking):
        # The unit's ConditionPathExists=/etc/sparkring/hairpin.json.
        assert self.exists(hairpin.APPROVAL), "sparkring-hairpin.service started without an approval"
        self.runs += 1
        self.invocation = f"{self.rank + 1:x}{self.runs:031x}"
        self.hairpin_unit = {"ActiveState": "activating", "Result": "success"}
        self.events.append(("start", self.rank))
        if not blocking:
            self.pending = True
            return test_hairpin.ok()
        return test_hairpin.ok() if self.finish() == 0 else test_hairpin.fail("Job for sparkring-hairpin.service failed")

    def finish(self):
        self.pending = False
        code = hairpin.apply(host=self.host())
        self.hairpin_unit = ({"ActiveState": "active", "Result": "success"} if code == 0
                             else {"ActiveState": "failed", "Result": "exit-code"})
        self.events.append(("finish", self.rank))
        return code

    def node(self, arguments):
        """``sparkring node hairpin <arguments>`` as root on this Spark."""
        if self.pending:
            self.finish()
        if arguments[0] == "status":
            return json.dumps(hairpin.status(busy="--busy" in arguments, host=self.host()))
        if arguments[0] == "start":
            return json.dumps(hairpin.start(arguments[arguments.index("--after") + 1], host=self.host()))
        if arguments == ["resume"]:
            return json.dumps(hairpin.resume(host=self.host()))
        assert arguments == ["approve"], arguments
        return json.dumps(hairpin.approve(host=self.host()))

    def inspect(self, document):
        """``sparkring node inspect``: the recorded facts with this Spark's devlink state."""
        current = copy.deepcopy(document)
        facts = current["facts"]
        interfaces = {row["name"]: row for row in facts["interfaces"]}
        for row in facts["rdma"]:
            function = self.functions[row["pci_address"]]
            for name in hairpin_setting.PARAMETERS:
                row["devlink"]["parameters"][name]["value"] = function["pending"].get(name, function["values"][name])
            row["devlink"]["reload"] = {"driver_reinit": function["counter"], "failed": function["failed"]}
            interfaces[row["netdev"]].update(hw_tc_offload=function["offload"] == "on",
                                             hw_tc_offload_fixed=function["offload"] == "off [fixed]")
        current["hairpin"] = hairpin.status(facts=facts, host=self.host())
        return current


class SimulatedRing:
    """Four RingSparks behind Node A's SSH, local-command and inspection boundaries."""

    def __init__(self, root, monkeypatch):
        self.documents = administered_nodes()
        self.events = []
        self.sparks = [RingSpark(root / f"spark{rank}", rank, document, self.events)
                       for rank, document in enumerate(self.documents)]
        for spark in self.sparks:
            spark.administer(self.sparks)
        self.ranks = {document["facts"]["ssh_target"]: rank for rank, document in enumerate(self.documents)}
        self.setup_commands = []
        monkeypatch.setattr(hairpin_ring, "remote", self.remote)
        monkeypatch.setattr(hairpin_ring, "local", self.local)
        monkeypatch.setattr(hairpin_ring, "POLL", 0)
        monkeypatch.setattr(hairpin_ring, "DISPATCH_RETRY", 0)
        monkeypatch.setattr(controller, "collect", self.collect)
        monkeypatch.setattr(flow.discovery, "ssh", self.ssh)
        # Status documents report the same package revision as the inspect documents.
        monkeypatch.setattr(distribution, "installed", lambda root, verify=True: {"revision": "a" * 40})

    def command(self, rank, argv):
        if argv[:3] == [hairpin_ring.SPARKRING, "node", "hairpin"]:
            return self.sparks[rank].node(argv[3:])
        result = self.sparks[rank].run(argv, timeout=30)
        if result.returncode:
            raise RuntimeError(f"rank {rank}: {' '.join(argv)}: {result.stderr}")
        return result.stdout

    def local(self, argv, *, timeout):
        return self.command(0, list(argv))

    def remote(self, host, argv, *, timeout):
        assert argv[:2] == ["sudo", "-n"], argv
        return self.command(self.ranks[host], list(argv[2:]))

    def collect(self, targets):
        return [self.sparks[self.ranks[target]].inspect(self.documents[self.ranks[target]]) for target in targets]

    def ssh(self, target, argv, **_):
        """The installation's other SSH commands: access checks, cached checkpoints and the running mesh."""
        if "native-mesh" in argv:
            rank = int(argv[-1])
            return json.dumps({"mesh": {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json",
                                                      "site_sha256": "c" * 64, "plan_sha256": "d" * 64},
                                        "host_ip": f"192.0.2.{110 + rank}", "interface": "eth0",
                                        "unit": "sparkring-mesh.service"}})
        return json.dumps({"model_path": "/srv/models/cached"})

    def setup_invoke(self, host, argv, *, data=None, **_):
        """Setup's commands after the network steps: node verify, configure, workspace and pings."""
        self.setup_commands.append((self.ranks[host], tuple(argv)))
        if argv[3:5] == ["node", "configure"]:
            self.sparks[self.ranks[host]].write(hairpin.FABRIC, json.loads(data))
        return "{}"

    def changes(self):
        return [spark.changes() for spark in self.sparks]


def first_installation(ring, monkeypatch):
    """Node A with enrolled workers and no ring record: sparkring install runs setup first."""
    head = ring.documents[0]["node_id"]
    for name in ("cluster.json", "active.json"):
        (controller.STATE / name).unlink()
    node.save(controller.STATE, "enrolled.json", {"targets": [d["facts"]["ssh_target"] for d in ring.documents],
                                                  "api_address": control_address(0)}, mode=0o600)
    (controller.STATE / "controller_ed25519").write_text("private\n")
    (controller.STATE / "controller_ed25519.pub").write_text(
        "ssh-ed25519 " + base64.b64encode(b"B" * 32).decode() + " sparkring-controller\n")
    monkeypatch.setattr(single_uplink.os, "geteuid", lambda: 0, raising=False)
    read = node.read
    monkeypatch.setattr(node, "read", lambda root, name: {"node_id": head}
                        if (str(root), name) == ("/", "/etc/sparkring/node.json") else read(root, name))
    apply = controller.apply

    def configured(plan, directory, **options):
        return apply(plan, directory, run=lambda *a: {"returncode": 0, "stdout": '{"checked":true}', "stderr": ""},
                     invoke=ring.setup_invoke, inspect_nodes=ring.collect, **options)
    monkeypatch.setattr(controller, "apply", configured)
    return "setups/*/hairpin.json"


def installed_ring(ring):
    """A ring recorded before its Sparks approved the setting, rebooted since: every function at 1024."""
    targets = [document["facts"]["ssh_target"] for document in ring.documents]
    plan = topology.build_spec(ring.collect(targets), ring.documents[0]["node_id"], preserve_control=True)
    node.save(controller.STATE, "cluster.json", {"schema": "sparkring-appliance-cluster/v1", "name": "sparkring",
                                                 "plan": plan, "api_address": control_address(0)}, mode=0o600)
    for rank, spark in enumerate(ring.sparks):
        spark.write(hairpin.FABRIC, topology.persistent_config(plan, rank))
    return "hairpin/*/hairpin.json"


@pytest.mark.skipif(os.name != "posix", reason="the node's hairpin lock and file modes need POSIX")
@pytest.mark.parametrize("installed", [False, True], ids=["first-installation", "installed-ring"])
def test_install_applies_the_hairpin_setting_to_a_ring_at_1024_and_repeats_as_a_no_op(
        machine, monkeypatch, tmp_path, capsys, installed):
    ring = SimulatedRing(tmp_path / "ring", monkeypatch)
    receipts = installed_ring(ring) if installed else first_installation(ring, monkeypatch)
    monkeypatch.setattr(flow.sys.stdin, "isatty", lambda: True)
    asked = []

    def answer(prompt):
        asked.append(prompt)
        sys.stderr.write("<question>\n")
        return "y"
    monkeypatch.setattr("builtins.input", answer)
    assert sparkring.main(["install", "--profile", TP4]) == 0
    err = capsys.readouterr().err
    approval = err[:err.index("<question>")]

    # One question, and the text before it lists the driver restarts.
    if installed:
        assert asked == ["Apply the ConnectX hairpin setting and this installation? [y/N]: "]
        for rank in range(4):
            assert f"  rank {rank} spark{rank}: restart 4 functions (1024 -> 8192)\n" in approval
        assert "Each restart takes one function's link down for about 8 seconds" in approval
        assert hairpin_ring.NOTICE_SINGLE_UPLINK in approval
        assert hairpin_ring.COMPLETE in err
    else:
        assert asked == ["Proceed? [Y/n]: "]
        assert "\n".join(single_uplink.HAIRPIN_SCOPE) in approval

    # Node A first, then one worker at a time in the administration tree's order.
    assert ring.events == [(kind, rank) for rank in (0, 1, 3, 2) for kind in ("start", "finish")]
    # On each Spark, functions without an administration tunnel restart first,
    # and the one that carries its own path towards Node A restarts last.
    expected = {0: ["cw_secondary", "ccw_secondary", "cw_primary", "ccw_primary"],
                1: ["cw_secondary", "ccw_secondary", "cw_primary", "ccw_primary"],
                2: ["cw_primary", "cw_secondary", "ccw_secondary", "ccw_primary"],
                3: ["cw_secondary", "ccw_primary", "ccw_secondary", "cw_primary"]}
    for spark in ring.sparks:
        assert [spark.functions[pci]["role"] for pci in spark.restarted()] == expected[spark.rank]
        # Armed once, by its own run, after all four restarts succeeded.
        enables = [index for index, argv in enumerate(spark.calls) if argv == ["systemctl", "enable", hairpin.UNIT]]
        reloads = [index for index, argv in enumerate(spark.calls) if argv[:3] == ["devlink", "dev", "reload"]]
        assert len(enables) == 1 and enables[0] > max(reloads) and spark.enabled == {hairpin.UNIT}
        state = spark.read(hairpin.STATE)
        assert state["in_effect"] and state["armed"] and state["error"] is None and state["mode"] == "live"
        assert [record["state"] for record in spark.records()] == ["applied"] * 4
        assert all(f["values"] == hairpin_setting.PARAMETERS and f["counter"] == 1 for f in spark.functions.values())
    [path] = controller.STATE.glob(receipts)
    receipt = json.loads(path.read_text())
    assert receipt["state"] == "complete" and receipt["reinspected"] is True
    assert [(entry["outcome"], entry["after"]) for entry in receipt["ranks"]] == [("applied", "kept")] * 4
    if not installed:
        # Setup records the plan that the hairpin step returned.
        record = installer.read(controller.STATE / "cluster.json")
        assert all(current["hairpin"]["in_effect"] and current["hairpin"]["armed"] for current in record["plan"]["nodes"])
        [setup] = controller.STATE.glob("setups/*/setup.json")
        assert {"hairpin": "complete"} in json.loads(setup.read_text())["steps"]

    # A second installation finds every Spark kept and changes no ConnectX function.
    changes, events = ring.changes(), list(ring.events)
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["state"] == "complete" and "apply-hairpin-setting" not in result["steps"]
    assert result["hairpin"]["required"] is False
    assert [row["before"] for row in result["hairpin"]["ranks"]] == ["kept"] * 4
    assert ring.changes() == changes and ring.events == events


MESH = {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json", "site_sha256": "c" * 64,
                      "plan_sha256": "d" * 64}, "interface": "eth0", "unit": "sparkring-mesh.service"}


def test_install_after_a_reboot_names_the_hairpin_setting_in_the_mesh_refusal(machine, monkeypatch, capsys):
    events, _, _, _ = machine
    value = four(monkeypatch)
    needing(value["plan"], [1, 2, 3])

    def remote(target, argv):
        if "native-mesh" in argv:
            # Only Node A's mesh runs: the start check refused the workers' meshes.
            rank = int(argv[-1])
            return json.dumps({"mesh": {**MESH, "host_ip": "192.0.2.110"} if rank == 0 else None})
        return json.dumps({"model_path": "/srv/models/cached"})
    monkeypatch.setattr(flow.discovery, "ssh", remote)
    monkeypatch.setattr(flow.hairpin_ring, "ensure", lambda *a, **k: pytest.fail("the hairpin step ran"))
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 2
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["message"] == (
        "Only part of the native mesh is active. Use a reviewed --fresh-mesh replacement after inspection. The "
        "ConnectX hairpin setting is not in effect on ranks 1-3, so their mesh services cannot start; run sudo "
        "sparkring hairpin on Node A first.")
    # The hairpin listing comes before the refusal.
    assert out.err.index("  rank 1 spark1: restart 4 functions") < out.err.index("Only part of the native mesh")
    assert events == []


def test_unknown_statistics_stop_the_installation_before_its_question(machine, monkeypatch, capsys):
    events, _, _, _ = machine
    value = four(monkeypatch)
    value["plan"]["nodes"][2]["hairpin"] = document(value["plan"], 2, [hairpin_setting.UNKNOWN]
                                                    + [hairpin_setting.IN_EFFECT] * 3)
    monkeypatch.setattr(flow.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("asked before refusing unknown statistics"))
    monkeypatch.setattr(flow.hairpin_ring, "ensure", lambda *a, **k: pytest.fail("the hairpin step ran"))
    assert sparkring.main(["install", "--profile", TP4]) == 3
    assert "Rank 2 (spark2): devlink reload statistics are unavailable" in capsys.readouterr().err
    assert events == []


def test_an_approval_without_restarts_stops_when_an_update_reveals_one(machine, monkeypatch, capsys):
    events, _, assets, _ = machine
    value = four(monkeypatch)
    # Node A in effect but not recorded; the workers run an older SparkRing and their rows show the setting.
    value["plan"]["nodes"][0]["hairpin"] = document(value["plan"], 0, approved=False, armed=False)
    for rank in (1, 2, 3):
        del value["plan"]["nodes"][rank]["hairpin"]
        value["plan"]["nodes"][rank]["revision"] = OLDER
    ring = simulate(monkeypatch, value)

    def sync_packages(self):
        events.append("update-workers")
        ring.update()
        # After its update, rank 3 turns out to need restarts.
        ring.sparks[3].status = document(value["plan"], 3, hairpin_setting.DEFAULT, approved=False, armed=False)
    monkeypatch.setattr(assets, "sync_packages", sync_packages)
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 3
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["field"] == "approval"
    assert result["message"].startswith("SparkRing was updated on ranks 1-3. Rank 3 then needs ConnectX driver "
                                        "restarts, which the approval did not cover")
    # The listing that --yes approved announced no restart.
    assert "No driver restarts now." in out.err and "Each restart takes" not in out.err.split(result["message"])[0]
    assert not ring.starts() and not ring.commands("approve") and not ring.commands("--busy")
    assert not any(e.endswith(":down") for e in events)


@pytest.mark.skipif(os.name != "posix", reason="the node's hairpin lock and file modes need POSIX")
@pytest.mark.parametrize("operation", ["install", "hairpin"])
def test_refused_mesh_units_start_only_after_every_spark_has_the_setting(machine, monkeypatch, tmp_path, capsys,
                                                                         operation):
    ring = SimulatedRing(tmp_path / "ring", monkeypatch)
    installed_ring(ring)
    monkeypatch.setattr(flow, "require_head", lambda *a, **k: ring.documents[0]["node_id"])
    order = []
    for spark in ring.sparks:
        def run(argv, *, timeout=None, _spark=spark, _original=spark.run, **options):
            words = list(argv)
            if words[:3] == ["devlink", "dev", "reload"]:
                order.append((_spark.rank, "reload"))
            if words[:3] == ["systemctl", "--no-block", "start"] and words[3].endswith("mesh.service"):
                order.append((_spark.rank, "start " + words[3]))
            return _original(argv, timeout=timeout, **options)
        spark.run = run
    head = ring.sparks[0]
    # Node A's enabled mesh unit was refused by its start check earlier in this boot.
    head.write(hairpin.BLOCKED + "/sparkring-mesh.service", {"unit": "sparkring-mesh.service"})
    head.enabled.add("sparkring-mesh.service")
    head.units["sparkring-mesh.service"] = {"ActiveState": "failed"}
    argv = ["install", "--profile", TP4] if operation == "install" else ["hairpin"]
    assert sparkring.main([*argv, "--yes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "complete"
    reloads = [entry for entry in order if entry[1] == "reload"]
    assert len(reloads) == 16
    if operation == "install":
        # The model installation that follows owns the mesh.
        assert len(order) == 16 and head.exists(hairpin.BLOCKED + "/sparkring-mesh.service")
    else:
        assert order[-1] == (0, "start sparkring-mesh.service") and len(order) == 17
        assert not head.exists(hairpin.BLOCKED + "/sparkring-mesh.service")


@pytest.mark.skipif(os.name != "posix", reason="the node's hairpin lock and file modes need POSIX")
def test_first_installation_with_yes_prints_the_approved_scope(machine, monkeypatch, tmp_path, capsys):
    ring = SimulatedRing(tmp_path / "ring", monkeypatch)
    first_installation(ring, monkeypatch)
    monkeypatch.setattr(flow.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt: pytest.fail("--yes asked a question"))
    assert sparkring.main(["install", "--profile", TP4, "--yes"]) == 0
    err = capsys.readouterr().err
    assert "Approved with --yes:\n" in err and "\n".join(single_uplink.HAIRPIN_SCOPE) in err
    assert all(spark.enabled == {hairpin.UNIT} for spark in ring.sparks)


def test_install_yes_help_names_the_connectx_restarts(capsys):
    with pytest.raises(SystemExit):
        flow.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert ("--yes approve the displayed setup, the listed ConnectX driver restarts on an idle ring, and model "
            "replacement; SSH trust is still required") in help_text
