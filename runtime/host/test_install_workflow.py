"""Public CLI acceptance with host/SSH boundaries simulated, not the coordinator.

Node A's SSH to the Sparks is simulated by ``Sparks``: checkpoint surveys
answer with survey documents built from the pins and options that each probe
carries, image checks find the serving image, and native-mesh inspection
reports no mesh unless a test sets one. The checkpoint end-to-end tests instead
run the real survey probe, plan, rank operations and fabric copy on two
simulated Sparks (``SimulatedSparks``); the ConnectX hairpin end-to-end tests
run the real node code on a simulated four-Spark ring (``SimulatedRing``).
"""
import ast
import base64
import builtins
import copy
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import uuid

import pytest

from runtime.common import distribution, installer
from runtime.host import checkpoint_place as place
from runtime.host import (checkpoint_plan, control, controller, hairpin, hairpin_ring, install_assets,
                          install_workflow as flow, node, rollout, single_uplink, test_hairpin, topology)
from runtime.host.install_errors import NeedsInput
from runtime.host.test_appliance import nodes
from runtime.host.test_fabric_ssh import cluster as fabric_cluster
from runtime.host.test_hairpin_ring import OLDER, Ring, document, kept, needing
from scripts import hairpin_setting, installer_host, installer_runner, sparkring
# The audit-hook fixtures refuse writes under protected trees; the end-to-end test requests them by name.
from scripts.test_installer_adopt import audit_hook, audited, only_links_changed, tree_state  # noqa: F401

PROFILE = "qwen38-flash-next-tp2"
TP4 = "qwen38-flash-next-qad-tp4"
# The command that repeats the tests' deployment request, as messages suggest it.
REPEAT = "sudo sparkring install --profile " + PROFILE


def cluster(size):
    """A recorded cluster; four-Spark inspect documents carry kept ConnectX hairpin status."""
    value = fabric_cluster(size)
    kept(value["plan"])
    return value


# The asset preparer whose runner the installer uses, captured before the
# machine fixture replaces it with a simulation.
ASSETS = install_assets.Assets
GIB = 1024 ** 3
REVISION = "60215d26cf5e42c2db6128774032d57fc62678da"
DIRECTORY = "/srv/sparkring/test/checkpoints/local-inference-lab--Qwen3.8-Flash-Next-NVFP4/" + REVISION
# The owner's copy, a Hugging Face download folder on the root filesystem of every Spark.
FOLDER = "/var/tmp/models/Qwen3.8-Flash-Next-NVFP4-QAD/" + REVISION
SHARD = "model-00002-of-00041.safetensors"
DEVICE, MOUNT = 66306, 29
OK = {"returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False}


def probe_inputs(source):
    """The pins and options a survey probe carries, read from its final call line."""
    printed = ast.parse(source.rstrip().splitlines()[-1]).body[0].value
    call = printed.args[0].args[0]
    return tuple(json.loads(argument.args[0].value) for argument in call.args)


def required(pins):
    return {name: entry for name, entry in sorted(pins["files"].items()) if name not in pins["optional"]}


def copy_candidate(pins, path=FOLDER, *, without=(), differs=(), device=DEVICE, mount_id=MOUNT, **extra):
    """A folder holding the pinned files, identified by SparkRing's earlier checksums."""
    files = {}
    for number, (name, entry) in enumerate(required(pins).items()):
        if name in without:
            continue
        files[name] = {"state": "differs" if name in differs else "match",
                       "evidence": "hashed" if name in differs else "recorded", "size": entry["size"],
                       "source": path + "/" + name, "kind": "file", "identity": [device, 1000 + number],
                       "owner": "code", "mode": 0o644, "mount_id": mount_id}
    counts = {state: sum(1 for value in files.values() if value["state"] == state)
              for state in ("match", "differs", "size-only", "missing", "incomplete")}
    return {"path": path, "layout": "local-dir", "found_by": ["folder"], "commit": pins["revision"], "branches": [],
            "home": None, "sparkring": False, "mount_id": mount_id, "device": device, "rotational": False,
            "files": files, "counts": counts, **extra}


def survey_document(pins, options, host, *, candidates=(), free=300 * GIB, named=None):
    return {"schema": "sparkring-checkpoint-survey/v1", "host": "spark-" + host.rsplit(".", 1)[1],
            "repository": pins["repository"], "revision": pins["revision"], "operator": options["operator"],
            "docker": {"userns": False, "driver": "overlay2", "root": "/var/lib/docker", "device": DEVICE},
            "owned": {"path": options["owned"], "resolved": options["owned"], "state": "absent",
                      "probe_path": "/srv/sparkring", "mount_point": "/", "mount_id": MOUNT, "device": DEVICE,
                      "cache_device": DEVICE, "fstype": "ext4", "free_bytes": free, "files": {}},
            "search": {"complete": True, "stopped": None, "passes": 1, "seconds": 1.2, "entries": 81234,
                       "unvisited": {}, "skipped_mounts": [], "large_directories": 0, "unreadable": 0, "errors": []},
            "candidates": list(candidates), "not_used": [],
            "named": named if named is not None else [{"path": path, "resolved": None, "state": "absent"}
                                                      for path in options["named"]]}


class Sparks:
    """Simulated SSH from Node A to the enrolled Sparks.

    ``survey(host, pins, options)`` answers a checkpoint survey; it returns a
    document, text, or an exception to raise. By default every Spark holds the
    owner's complete copy on the checkpoint directory's filesystem.
    ``surveys`` records ``(host, options)`` of every survey.
    """

    def __init__(self):
        self.surveys = []
        self.guard = threading.Lock()
        self.barrier = None
        self.mesh = lambda rank: None
        self.survey = lambda host, pins, options: survey_document(pins, options, host,
                                                                  candidates=[copy_candidate(pins)])

    def __call__(self, host, argv, *, data=None, timeout=None):
        if argv == flow.SURVEY_COMMAND:
            assert timeout == flow.SURVEY_TIMEOUT
            pins, options = probe_inputs(data)
            with self.guard:
                self.surveys.append((host, options))
            if self.barrier is not None:
                self.barrier.wait()
            answer = self.survey(host, pins, options)
            if isinstance(answer, BaseException):
                raise answer
            return answer if isinstance(answer, str) else json.dumps(answer)
        if argv[:3] == ["sudo", "-n", "docker"] and "inspect" in argv:
            return argv[-1] + "\n"
        if "native-mesh" in argv:
            return json.dumps({"mesh": self.mesh(int(argv[-1]))})
        return ""

    def options(self, host):
        return [options for name, options in self.surveys if name == host]


@pytest.fixture
def sparks(monkeypatch):
    value = Sparks()
    monkeypatch.setattr(flow.discovery, "ssh", value)
    monkeypatch.setenv("SUDO_USER", "code")
    return value


@pytest.fixture
def machine(tmp_path, monkeypatch, sparks):
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = cluster(2)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(flow, "require_head", lambda *_: value["plan"]["nodes"][0]["node_id"])
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    monkeypatch.setattr(flow.distribution, "identity", lambda _: "a" * 40)
    monkeypatch.setattr(flow.distribution, "bundle", lambda root, dest: dest.write_bytes(b"retained source"))
    events = []
    previous = controller.STATE / "deployments" / "previous"
    previous.mkdir(parents=True)
    node.save(controller.STATE, "active.json", {"path": str(previous)})
    monkeypatch.setattr(flow, "check_workloads", lambda *a, **k: events.append("check-workloads"))
    monkeypatch.setattr(flow.retained_source, "checkout", lambda *a: events.append("rollback-source"))
    # An installed deployment serves on every Spark unless a test says otherwise.
    monkeypatch.setattr(flow, "serving", lambda directory: True)
    class Transport:
        def __init__(self, cluster, directory):
            self.hosts = cluster["plan"]["spec"]["hosts"]
        def verify(self):
            events.append("verify-fabric")
            return {"transport": "fiber-ssh", "caller_relay": False}
    monkeypatch.setattr(flow.fabric_ssh, "Transport", Transport)
    class Assets:
        # When set, replaces the model action: called with the runner and the
        # approved plan, it returns that action's result.
        on_model = None
        prepared = {}
        def __init__(self, *a):
            pass
        def sync_packages(self):
            events.append("update-workers")
        def images(self, card):
            events.append("fill-missing-image")
        def runner(self, directory, previous=None, images=None, plan=None, receipts=None):
            Assets.prepared.update(plan=plan, receipts=receipts)
            class Run:
                needs_input = None
                def __call__(self, host, argv, timeout):
                    if argv[1] == "model" and Assets.on_model is not None:
                        return Assets.on_model(self, plan)
                    if argv[1] in ("model", "image"):
                        images.result()
                    events.append("prepare:" + argv[1])
                    return dict(OK)
            return Run()
    monkeypatch.setattr(flow.install_assets, "Assets", Assets)
    def operation(path, action, **kwargs):
        events.append(("previous" if path == previous else "candidate") + ":" + action)
        return {"verified": True}
    monkeypatch.setattr(flow.retained_source, "apply", operation)
    return events, previous, Assets, operation


def command(*extra):
    return sparkring.main(["install", "--profile", PROFILE, "--yes", "--json", *extra])


def test_documented_command_updates_prepares_switches_and_emits_only_json(machine, capsys):
    events, previous, assets, _ = machine
    assert command() == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["state"] == "complete" and result["transfer"]["caller_relay"] is False
    # The plan approved by --yes bounds the checkpoint preparation.
    assert result["checkpoint"]["approval"] == "command-line" and result["checkpoint"]["hub_files"] == []
    assert assets.prepared["plan"]["approval"] == "command-line" and assets.prepared["receipts"] == {0: [], 1: []}
    assert events.index("update-workers") < events.index("fill-missing-image") < events.index("previous:down")
    assert events.index("prepare:model-check") < events.index("previous:down") < events.index("candidate:up") < events.index("candidate:verify")
    assert "Progress:" in out.err and "Model ready:" in out.err
    assert rollout.active(controller.STATE) != previous


def test_installing_the_active_model_again_restarts_it_when_it_does_not_serve(machine, monkeypatch, capsys):
    events, _, _, _ = machine
    assert command() == 0
    capsys.readouterr()
    del events[:]
    monkeypatch.setattr(flow, "serving", lambda directory: events.append("serving") or False)
    assert command() == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["state"] == "complete"
    assert events[events.index("serving"):] == ["serving", "candidate:down", "candidate:up", "candidate:verify"]
    assert "The installed model does not serve on every Spark" in out.err


class RankChecks:
    """``Runner.remote`` of an installed deployment; the listed (rank, operation) checks fail."""

    def __init__(self, count, backend="compose", failing=()):
        self.lock = {"backend": backend, "site": {"ranks": [{"rank": rank} for rank in range(count)]}}
        self.failing, self.calls = set(failing), []

    def remote(self, number, operation):
        self.calls.append((number, operation))
        if (number, operation) in self.failing:
            raise RuntimeError("Rank is not running")
        return {"ok": True}


def test_an_installed_model_serves_when_every_rank_runs_and_passes_its_ring_check():
    ring = RankChecks(4)
    assert flow.serving(None, runner=ring)
    assert ring.calls == [(rank, operation) for rank in range(4) for operation in ("running", "ring-check")]
    assert not flow.serving(None, runner=RankChecks(4, failing={(2, "ring-check")}))
    pair = RankChecks(2, failing={(1, "running")})
    assert not flow.serving(None, runner=pair) and pair.calls == [(0, "running"), (0, "gid-check"), (1, "running")]
    managed = RankChecks(4, backend="glm-managed", failing={(0, "running")})
    assert flow.serving(None, runner=managed) and managed.calls == []


@pytest.mark.parametrize("size", [2, 4])
def test_a_moved_roce_gid_passes_the_refresh_for_the_installation_to_repair(monkeypatch, size):
    value = cluster(size)
    found = copy.deepcopy(value["plan"]["nodes"])
    port = value["plan"]["spec"]["hosts"][1]["data_interfaces"][0]
    next(r for r in found[1]["facts"]["rdma"] if r["device"] == port["rdma_device"])["gid"] = "0000:" * 7 + "0000"
    monkeypatch.setattr(controller, "collect", lambda _: found)
    monkeypatch.setattr(flow, "require_head", lambda *_: None)
    refreshed = flow.refresh_cluster(value)["plan"]["spec"]["hosts"]
    assert [h["node_id"] for h in refreshed] == [h["node_id"] for h in value["plan"]["spec"]["hosts"]]


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
    def models(self, lock, runner, previous=None, plan=None, receipts=None):
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
    def models(self, lock, runner, previous=None, plan=None, receipts=None):
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
            # Both ranks' checkpoint actions report one failed preparation, and
            # the result names its cause rather than only the failed phase.
            assert first["state"] == "failed" and calls["models"] == 1
            assert first["message"] == "Checkpoint preparation failed: Hugging Face download failed: 503"
            assert out.err.rstrip().splitlines()[-1] == "Error: Checkpoint preparation failed: Hugging Face download failed: 503"
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
def test_tp4_command_adopts_the_discovered_mesh_without_network_changes(machine, sparks, monkeypatch, capsys, profile):
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    sparks.mesh = lambda rank: {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json",
                                              "site_sha256": "c" * 64, "plan_sha256": "d" * 64},
                              "host_ip": f"192.0.2.{110 + rank}", "interface": "eth0", "unit": "sparkring-mesh.service"}
    assert sparkring.main(["install", "--profile", profile, "--yes", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    from runtime.common import installer
    lock = installer.load(result["deployment"])
    assert result["nodes"] == 4 and "native_mesh" not in lock["site_input"]
    assert [r["host_ip"] for r in lock["site"]["ranks"]] == [f"192.0.2.{110 + rank}" for rank in range(4)]
    # The adopted mesh records no model roots; each rank's row names the cluster's checkpoint directory.
    checkpoint = installer.checkpoint_directory("test", installer.setup.selection(profile))
    assert [(row["model"], row["reuse_verified_model"]) for row in lock["site"]["ranks"]] == [(checkpoint, False)] * 4
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
    def remote(target, argv, **kwargs):
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


# Checkpoint survey, plan and approval.

def output_lines(text):
    return [line for line in text.splitlines()]


def line_index(lines, prefix):
    return next(index for index, line in enumerate(lines) if line.startswith(prefix))


def saved(result):
    return installer.read(Path(result["deployment"]) / flow.PLAN_FILE)


def holding(changes):
    """A survey answer: the owner's copy on every Spark, with per-host changes to copy_candidate."""
    def answer(host, pins, options):
        change = changes.get(host, {})
        if change is None:
            return survey_document(pins, options, host)
        return survey_document(pins, options, host, candidates=[copy_candidate(pins, **change)])
    return answer


def test_install_surveys_every_node_in_parallel_and_prints_the_plan_after_the_header(machine, sparks, capsys):
    # Each survey waits until both are running; a sequential survey breaks the barrier.
    sparks.barrier = threading.Barrier(2, timeout=5)
    assert command("--plan") == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert all(node["search"] == {"complete": True, "seconds": 1.2, "not_searched": [], "unvisited": {}}
               for node in result["checkpoint"]["nodes"])
    for host in ("root@192.0.2.10", "root@192.0.2.11"):
        [options] = sparks.options(host)
        assert options["owned"] == DIRECTORY and options["operator"] == "code"
        assert options["named"] == [] and options["ignore_local"] is False and options["root"] == "/"
        assert options["cache"] == "/srv/sparkring/test/cache"
    lines = output_lines(out.err)
    assert (line_index(lines, "Looking for existing copies of local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at "
                              "60215d26cf5e on 2 Sparks (up to 20 s each; up to 75 s on a Spark whose search "
                              "needs a second pass).")
            < line_index(lines, "Install qwen38-flash-next-tp2 on 2 Sparks.")
            < line_index(lines, "Update workers and prepare assets; then replace the current model.")
            < line_index(lines, "Checkpoint local-inference-lab/Qwen3.8-Flash-Next-NVFP4 at 60215d26cf5e: 53 files, 102.6 GiB")
            < line_index(lines, "Node 0 spark-10 -> " + DIRECTORY)
            < line_index(lines, "    hard-link 41 weight files (no copy, no extra space); copy 12 other files (56.1 MB)")
            < line_index(lines, "Node 1 spark-11: as Node 0 (300 GiB free)"))
    # The bottom line of the plan stays directly above the next message or prompt.
    assert lines[line_index(lines, "Plan saved.") - 1] == "Nothing is downloaded."
    # The suggested command repeats the request, so it installs the plan just saved.
    assert lines[line_index(lines, "Plan saved.")] == f"Plan saved. Install it with {REPEAT} --yes."
    assert result["checkpoint"]["command"] == REPEAT and result["checkpoint"]["reviewed"] is True
    node0 = result["checkpoint"]["nodes"][0]
    assert node0["mode"] == "owned" and node0["path"] == DIRECTORY and node0["required_bytes"] == 34448927703
    assert node0["bytes"]["link"] == 110131860580 and node0["bytes"]["copy"] == 56080100


def test_an_unlisted_checkpoint_changes_nothing(machine, sparks, capsys):
    assert command("--checkpoint", "main") == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "checkpoint_name"
    assert "qad-step-4000, qad-step5500-ple1000" in result["message"] and not sparks.surveys


def test_naming_the_default_checkpoint_installs_the_same_deployment(machine, capsys):
    assert command() == 0
    first = json.loads(capsys.readouterr().out)
    for name in ("qad-step5500-ple1000", "qad-step-5500"):
        assert command("--checkpoint", name) == 0
        second = json.loads(capsys.readouterr().out)
        assert second["deployment"] == first["deployment"] and second["replaces"] is None
        assert second["checkpoint"]["command"] == REPEAT


def test_survey_runs_on_every_install_and_rewrites_the_saved_plan(machine, sparks, capsys):
    assert command("--plan") == 0
    first = json.loads(capsys.readouterr().out)
    plan = saved(first)
    assert plan["reviewed"] is True and plan["nodes"][1]["files"]["config.json"]["action"] == "copy"
    # The deployment exists now; the next run surveys again and replaces the saved plan.
    sparks.survey = holding({"root@192.0.2.11": {"without": ("config.json",)}})
    assert command("--plan") == 0
    second = json.loads(capsys.readouterr().out)
    assert second["deployment"] == first["deployment"] and len(sparks.surveys) == 4
    plan = saved(second)
    assert plan["reviewed"] is True
    assert plan["nodes"][1]["files"]["config.json"] == {"action": "receive", "size": 29820, "from": 0,
                                                        "transport": "fabric"}


@pytest.mark.parametrize("change", ["equal", "download", "writes"])
def test_reviewed_plan_bounds_a_later_yes(machine, sparks, capsys, change):
    events, previous, assets, _ = machine
    assert command("--plan") == 0
    reviewed = saved(json.loads(capsys.readouterr().out))
    if change == "download":
        # Both copies lost the same shard, so it must come from huggingface.co.
        sparks.survey = holding({host: {"without": (SHARD,)} for host in ("root@192.0.2.10", "root@192.0.2.11")})
    elif change == "writes":
        # Node 1's copy is gone: it would receive 102.6 GiB over the fabric.
        sparks.survey = holding({"root@192.0.2.11": None})
    code = command()
    out = capsys.readouterr()
    result = json.loads(out.out)
    if change == "equal":
        assert code == 0 and result["state"] == "complete" and result["checkpoint"]["approval"] == "reviewed-plan"
        # The reviewed plan stays the bound for a repeated --yes.
        assert saved(result) == reviewed and assets.prepared["plan"]["approval"] == "reviewed-plan"
        assert assets.prepared["plan"]["hub_files"] == reviewed["hub_files"]
        return
    assert code == 3 and result["field"] == "checkpoint"
    assert result["message"].startswith("The checkpoint plan differs from the plan reviewed with --plan: ")
    assert result["message"].endswith(f"Nothing was changed. Review the plan again with {REPEAT} --plan, then "
                                      f"repeat {REPEAT} --yes.")
    if change == "download":
        assert ("Node 0 spark-10 would download 3.96 GiB from huggingface.co (" + SHARD + "), which the reviewed "
                "plan took from " + FOLDER + ", where that file is absent") in result["message"]
    else:
        assert "Node 1 spark-11 would write 102.62 GiB instead of 56.1 MB" in result["message"]
    assert not events and rollout.active(controller.STATE) == previous
    # The reviewed plan stays the bound; the refused plan is kept beside it.
    deployment = next(p for p in (controller.STATE / "deployments").iterdir() if p != previous)
    assert saved({"deployment": str(deployment)}) == reviewed
    refused = installer.read(deployment / flow.REFUSED_FILE)
    assert refused["reviewed"] is False and refused["hub_files"] == ([SHARD] if change == "download" else [])
    # Repeating --yes without a review is refused again.
    assert command() == 3
    assert json.loads(capsys.readouterr().out)["field"] == "checkpoint" and not events
    # A review of the new plan replaces the bound, and the next --yes proceeds within it.
    assert command("--plan") == 0
    capsys.readouterr()
    assert saved({"deployment": str(deployment)})["reviewed"] is True
    assert not (deployment / flow.REFUSED_FILE).exists()
    assert command() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint"]["approval"] == "reviewed-plan"
    assert assets.prepared["plan"]["hub_files"] == ([SHARD] if change == "download" else [])


@pytest.fixture
def first_installation(machine, monkeypatch):
    """No cluster record yet: setup runs first, and its approval lists the installation."""
    from runtime.host import single_uplink
    value = installer.read(controller.STATE / "cluster.json")
    (controller.STATE / "cluster.json").unlink()
    runs = []

    def setup(options, follow):
        runs.append(options)
        node.save(controller.STATE, "cluster.json", value)
        return 0
    monkeypatch.setattr(single_uplink, "main", setup)
    return runs


@pytest.mark.parametrize("approval", ["terminal", "no-terminal", "command-line"])
def test_first_installation_asks_before_attention_items(machine, sparks, first_installation, monkeypatch, capsys,
                                                        approval):
    events, previous, assets, _ = machine
    # No Spark holds a copy: Node 0 downloads 102.6 GiB, more than setup's approval covers.
    sparks.survey = holding({host: None for host in ("root@192.0.2.10", "root@192.0.2.11")})
    prompts = []
    if approval == "no-terminal":
        # Setup itself needs a terminal or --yes, so this case is the approval step alone.
        fresh = checkpoint_plan.plan(installer.checkpoint_pins(installer.setup.selection(PROFILE)),
                                     [{"error": "timeout"}, {"error": "timeout"}],
                                     [{"host": "root@192.0.2.10", "model": DIRECTORY},
                                      {"host": "root@192.0.2.11", "model": DIRECTORY}])
        with pytest.raises(NeedsInput) as caught:
            flow.approve(fresh, None, command_line=False, setup_only=True, interactive=False, request={})
        assert caught.value.field == "checkpoint"
        assert str(caught.value).startswith("The checkpoint plan downloads 102.6 GiB from huggingface.co on Node 0 "
                                            "root@192.0.2.10, the search on Node 0 root@192.0.2.10 failed (timeout), "
                                            "and the search on Node 1 root@192.0.2.11 failed (timeout).")
        assert str(caught.value).endswith("Setup's approval did not show this plan. Review it with sudo sparkring "
                                          "install --plan, or approve it with sudo sparkring install --yes. Nothing "
                                          "has been changed.")
        return
    if approval == "terminal":
        class Terminal:
            def isatty(self):
                return True
        monkeypatch.setattr(flow.sys, "stdin", Terminal())

        def answer(prompt):
            prompts.append((prompt, output_lines(capsys.readouterr().err)))
            return "y"
        monkeypatch.setattr(builtins, "input", answer)
        assert sparkring.main(["install", "--profile", PROFILE]) == 0
        assert first_installation == [[]]
        [(prompt, printed)] = prompts
        assert prompt == "Proceed with this checkpoint plan? [y/N]: "
        assert [line for line in printed if line][-1] == "Downloads 102.6 GiB from huggingface.co on Node 0."
        assert assets.prepared["plan"]["approval"] == "prompt"
    else:
        monkeypatch.setattr(builtins, "input", lambda prompt: pytest.fail("prompted: " + prompt))
        assert command() == 0
        assert first_installation == [["--yes"]]
        result = json.loads(capsys.readouterr().out)
        assert result["checkpoint"]["approval"] == "command-line" and result["checkpoint"]["hub_bytes"] > 98 * GIB
        assert assets.prepared["plan"]["hub_files"] == result["checkpoint"]["hub_files"]
    assert rollout.active(controller.STATE) != previous


def test_first_installation_without_attention_items_is_approved_by_setup(machine, sparks, first_installation,
                                                                          monkeypatch, capsys):
    events, previous, assets, _ = machine

    class Terminal:
        def isatty(self):
            return True
    monkeypatch.setattr(flow.sys, "stdin", Terminal())
    monkeypatch.setattr(builtins, "input", lambda prompt: pytest.fail("prompted: " + prompt))
    assert sparkring.main(["install", "--profile", PROFILE]) == 0
    assert assets.prepared["plan"]["approval"] == "setup"


def named_copies(host, pins, options):
    """Named copies: /data/qwen on the checkpoint directory's filesystem, /mnt/usb/qwen on a USB disk."""
    candidates, named = [], []
    for path in options["named"]:
        other = path.startswith("/mnt/usb")
        candidates.append(copy_candidate(pins, path, device=2049 if other else DEVICE, mount_id=40 if other else MOUNT,
                                         found_by=["named"], named_as=path, exact=True, extra=[], symlinks=[]))
        named.append({"path": path, "resolved": path, "state": "exact", "extra": [], "symlinks": []})
    return survey_document(pins, options, host, candidates=candidates, named=named)


def test_named_paths_parse_per_node_and_enter_the_request(machine, sparks, capsys):
    events, previous, _, _ = machine
    sparks.survey = named_copies
    assert command("--plan", "--model-path", "/data/qwen", "--model-path", "1=/mnt/usb/qwen") == 0
    result = json.loads(capsys.readouterr().out)
    # A path for every Spark, and one for Node 1 only.
    assert [options["named"] for options in sparks.options("root@192.0.2.10")] == [["/data/qwen"]]
    assert [options["named"] for options in sparks.options("root@192.0.2.11")] == [["/data/qwen", "/mnt/usb/qwen"]]
    lock = installer.load(result["deployment"])
    # Node 0 links from /data/qwen into SparkRing's directory; Node 1 serves the
    # exact copy on its other filesystem in place.
    assert [(row["model"], row["reuse_verified_model"]) for row in lock["site"]["ranks"]] == [
        (DIRECTORY, False), ("/mnt/usb/qwen", True)]
    assert [node["mode"] for node in result["checkpoint"]["nodes"]] == ["owned", "in-place"]
    assert result["checkpoint"]["nodes"][0]["sources"][0]["path"] == "/data/qwen"
    # --ignore-local-copies narrows the search but keeps the deployment request.
    assert command("--plan", "--model-path", "/data/qwen", "--model-path", "1=/mnt/usb/qwen",
                   "--ignore-local-copies") == 0
    again = json.loads(capsys.readouterr().out)
    assert again["deployment"] == result["deployment"] and sparks.surveys[-1][1]["ignore_local"] is True
    # Other named paths are another deployment.
    assert command("--plan", "--model-path", "/data/qwen") == 0
    assert json.loads(capsys.readouterr().out)["deployment"] != result["deployment"]
    for value, message in (("2=/data/qwen", "--model-path names Node 2, which this cluster does not have"),
                           ("data/qwen", "--model-path needs an absolute path: 'data/qwen'")):
        count = len(sparks.surveys)
        assert command("--plan", "--model-path", value) == 3
        refused = json.loads(capsys.readouterr().out)
        assert refused["field"] == "model_path" and refused["message"] == message + ". Nothing has been changed."
        assert len(sparks.surveys) == count
    assert not events


def adoption(plan, differs):
    """``model-adopt`` results of every Spark when ``differs`` turned out to be another checkpoint's files."""
    results = []
    for spark in plan["nodes"]:
        local = [name for name, entry in spark["files"].items() if entry["action"] in ("present", "link", "copy")]
        results.append({"complete": False, "verified": {name: [] for name in local if name not in differs},
                        "missing": sorted(differs), "differs": [{"name": name, "source": FOLDER + "/" + name}
                                                                for name in differs],
                        "linked": 35, "copied": 12, "bytes_written": 56497920, "refreshed": []})
    return results


@pytest.mark.parametrize("runner", ["simulated", "prepared"])
def test_unplanned_download_returns_needs_input_and_keeps_the_model_running(machine, monkeypatch, capsys, runner):
    events, previous, assets, _ = machine

    def guard(plan):
        # Hashing at adoption found another checkpoint's shard on both Sparks.
        items = checkpoint_plan.unplanned(plan, adoption(plan, [SHARD]))
        return NeedsInput(checkpoint_plan.unplanned_message(items, plan["command"]), field="checkpoint",
                          details={"items": items})
    if runner == "simulated":
        def model(runner, plan):
            runner.needs_input = guard(plan)
            return {"returncode": 1, "stdout": "", "stderr": str(runner.needs_input), "uncertain": False}
        assets.on_model = model
    else:
        # The asset runner the installer uses keeps the request with its type.
        def models(self, lock, runner, previous=None, plan=None, receipts=None):
            raise guard(plan)
        prepared_runner(monkeypatch, images=lambda self, card: None, models=models)
    assert command() == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "checkpoint" and result["details"]["items"][0]["names"] == [SHARD]
    assert result["message"].startswith("Node 0 spark-10: 1 file in " + FOLDER + " is not the pinned model's")
    assert result["message"].endswith("Nothing was downloaded and the running model was not changed. Review the "
                                      f"resulting plan with {REPEAT} --plan.")
    assert rollout.active(controller.STATE) == previous and not any(e.endswith(":down") for e in events)


@pytest.mark.parametrize("field, shown", [("checkpoint", False), ("access", True)])
def test_terminal_prints_needs_input_details_only_when_the_message_lacks_them(machine, monkeypatch, capsys, field,
                                                                                shown):
    details = {"items": [{"rank": 0, "path": "/data/copy", "names": "model-00002-of-00036.safetensors"}]}

    def execute(args):
        raise NeedsInput("The message names every file and path.", field=field, details=details)
    monkeypatch.setattr(flow, "execute", execute)
    assert sparkring.main(["install", "--profile", PROFILE]) == 3
    err = capsys.readouterr().err
    assert err.count("The message names every file and path.") == 1
    assert ("items:" in err and "/data/copy" in err) is shown
    assert sparkring.main(["install", "--profile", PROFILE, "--json"]) == 3
    assert json.loads(capsys.readouterr().out)["details"] == details


def test_needs_input_is_not_masked_by_an_image_failure(machine, monkeypatch, capsys):
    events, previous, assets, _ = machine
    released = threading.Event()

    def images(self, card):
        released.wait(10)
        raise RuntimeError("Node 1: relay pull failed: registry unavailable")
    monkeypatch.setattr(assets, "images", images)

    def model(runner, plan):
        runner.needs_input = NeedsInput("Checkpoint request", field="checkpoint")
        released.set()
        return {"returncode": 1, "stdout": "", "stderr": "Checkpoint request", "uncertain": False}
    assets.on_model = model
    assert command() == 3
    result = json.loads(capsys.readouterr().out)
    assert result["field"] == "checkpoint" and result["message"] == "Checkpoint request"
    assert result["details"]["image_error"] == "Node 1: relay pull failed: registry unavailable"
    assert rollout.active(controller.STATE) == previous


@pytest.mark.parametrize("failure", ["ssh", "timeout", "output"])
def test_failed_survey_on_one_node_is_reported_not_fatal(machine, sparks, capsys, failure):
    default = sparks.survey

    def answer(host, pins, options):
        if host != "root@192.0.2.11":
            return default(host, pins, options)
        if failure == "ssh":
            return RuntimeError(host + ": Traceback (most recent call last):\n  ...\nPermissionError: [Errno 13] denied")
        if failure == "timeout":
            return subprocess.TimeoutExpired(["ssh"], 150)
        return json.dumps({**default(host, pins, options), "revision": "f" * 40})
    sparks.survey = answer
    assert command() == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    error = {"ssh": "PermissionError: [Errno 13] denied", "timeout": "Command '['ssh']' timed out after 150 seconds",
             "output": "invalid survey output"}[failure]
    assert result["state"] == "complete"
    assert result["checkpoint"]["nodes"][1]["search"] == {"complete": False, "seconds": None, "not_searched": [],
                                                          "unvisited": {}, "error": error}
    # Node 1 is planned as holding no copy and receives the checkpoint from Node 0.
    assert result["checkpoint"]["nodes"][1]["bytes"]["receive"] == result["checkpoint"]["nodes"][0]["bytes"]["link"] + \
        result["checkpoint"]["nodes"][0]["bytes"]["copy"]
    assert f"The search on root@192.0.2.11 failed: {error}." in out.err
    assert f"Node 1 root@192.0.2.11: search failed: {error}" in out.err


def test_tp4_mesh_model_roots_are_the_cluster_checkpoint_directory(machine, sparks, monkeypatch, capsys):
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    (controller.STATE / "active.json").unlink()
    assert sparkring.main(["install", "--profile", "qwen38-flash-next-qad-tp4", "--plan", "--json"]) == 0
    result = json.loads(capsys.readouterr().out)
    lock = installer.load(result["deployment"])
    # No active mesh: the installer creates one whose ranks mount SparkRing's directory.
    assert lock["site_input"]["native_mesh"]["site"]["model_roots"] == [DIRECTORY] * 4
    assert [row["model"] for row in lock["site"]["ranks"]] == [DIRECTORY] * 4
    assert len(sparks.surveys) == 4 and result["checkpoint"]["hub_files"] == []


def test_tp4_mesh_model_roots_name_a_copy_served_in_place(machine, sparks, monkeypatch, capsys):
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    (controller.STATE / "active.json").unlink()
    sparks.survey = named_copies
    assert sparkring.main(["install", "--profile", "qwen38-flash-next-qad-tp4", "--plan", "--json",
                           "--model-path", "2=/mnt/usb/qwen"]) == 0
    lock = installer.load(json.loads(capsys.readouterr().out)["deployment"])
    # Node 2's exact copy on another filesystem is served in place, so the created mesh mounts it on that rank.
    assert lock["site_input"]["native_mesh"]["site"]["model_roots"] == [DIRECTORY, DIRECTORY, "/mnt/usb/qwen", DIRECTORY]
    assert [row["reuse_verified_model"] for row in lock["site"]["ranks"]] == [False, False, True, False]


def test_replanning_after_a_linked_file_changed(machine, sparks, capsys):
    events, previous, assets, _ = machine
    assert command("--plan") == 0
    capsys.readouterr()
    # A linked shard was rewritten in both copies after the review.
    sparks.survey = holding({host: {"differs": (SHARD,)} for host in ("root@192.0.2.10", "root@192.0.2.11")})
    assert command() == 3
    assert json.loads(capsys.readouterr().out)["field"] == "checkpoint" and not events
    # A new review shows the download, and the next --yes applies it.
    assert command("--plan") == 0
    out = capsys.readouterr()
    lines = output_lines(out.err)
    assert lines[line_index(lines, "Plan saved.") - 1] == "Downloads 4.0 GiB from huggingface.co on Node 0."
    assert "        52 of 53 files identified by SparkRing's earlier checksums; " + SHARD + \
        " differs from the pinned revision" in lines
    assert command() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["checkpoint"]["approval"] == "reviewed-plan" and result["checkpoint"]["hub_files"] == [SHARD]
    assert assets.prepared["plan"]["hub_files"] == [SHARD]


def deployments_besides(previous):
    return sorted(path.name for path in (controller.STATE / "deployments").iterdir() if path != previous)


def test_a_yes_run_stopped_by_a_plan_problem_keeps_the_reviewed_plan(machine, sparks, capsys):
    events, previous, assets, _ = machine
    assert command("--plan") == 0
    reviewed = saved(json.loads(capsys.readouterr().out))
    deployment = controller.STATE / "deployments" / deployments_besides(previous)[0]
    # Node 1 is short of space during a --yes run: a plan problem, not a change of the plan.
    sparks.survey = lambda host, pins, options: survey_document(
        pins, options, host, candidates=[copy_candidate(pins)], free=GIB if host.endswith(".11") else 300 * GIB)
    assert command() == 3
    stopped = json.loads(capsys.readouterr().out)
    assert stopped["field"] == "storage" and not events
    assert saved({"deployment": str(deployment)}) == reviewed
    assert installer.read(deployment / flow.REFUSED_FILE)["problems"][0]["field"] == "storage"
    # Space is freed, but both copies lost a shard meanwhile: the reviewed plan still bounds --yes.
    sparks.survey = holding({host: {"without": (SHARD,)} for host in ("root@192.0.2.10", "root@192.0.2.11")})
    assert command() == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "checkpoint" and SHARD in refused["message"] and not events
    assert saved({"deployment": str(deployment)}) == reviewed


def test_a_plan_with_problems_records_no_deployment(machine, sparks, capsys):
    events, previous, _, _ = machine
    before = deployments_besides(previous)
    assert command("--plan", "--model-path", "/data/typo") == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["message"] == "--model-path /data/typo exists on no Spark. Nothing has been changed."
    # No deployment directory lists the mistyped path as a user of SparkRing's directory.
    assert deployments_besides(previous) == before and not events

    # Node 1 names a copy on another filesystem that holds the main branch's config.json and has too
    # little space to copy it. The message says how to complete the copy and which command to repeat.
    usb = "/mnt/usb/qwen"
    state = {"exact": False}

    def answer(host, pins, options):
        if usb not in options["named"]:
            return survey_document(pins, options, host, candidates=[copy_candidate(pins)])
        exact = state["exact"]
        candidate = copy_candidate(pins, usb, device=2049, mount_id=40, found_by=["named"], named_as=usb,
                                   differs=() if exact else ("config.json",), exact=exact, extra=[], symlinks=[],
                                   branches=[] if exact else ["main"])
        return survey_document(pins, options, host, candidates=[candidate], free=40 * GIB, named=[
            {"path": usb, "resolved": usb, "state": "exact" if exact else "differs", "extra": [], "symlinks": []}])
    sparks.survey = answer
    assert command("--plan", "--model-path", "1=" + usb) == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "storage" and "holds the main branch's config.json" in refused["message"]
    assert refused["message"].endswith(f"Then repeat {REPEAT} --model-path 1={usb} --plan. The running model has not "
                                       "been stopped.")
    assert deployments_besides(previous) == before
    # Once the operator completed the copy, the same command plans the deployment afresh and serves it in place.
    state["exact"] = True
    assert command("--plan", "--model-path", "1=" + usb) == 0
    result = json.loads(capsys.readouterr().out)
    lock = installer.load(result["deployment"])
    assert [(row["model"], row["reuse_verified_model"]) for row in lock["site"]["ranks"]] == [
        (DIRECTORY, False), (usb, True)]
    assert result["checkpoint"]["nodes"][1]["in_place"] == "exact" and not events


def test_suggested_commands_repeat_the_deployment_request(machine, sparks, capsys):
    events, previous, _, _ = machine
    sparks.survey = named_copies
    options = ("--model-path", "1=/mnt/usb/qwen", "--cache-path", "/mnt/fast/cache")
    assert command("--plan", *options) == 0
    out = capsys.readouterr()
    result = json.loads(out.out)
    repeat = f"{REPEAT} --model-path 1=/mnt/usb/qwen --cache-path /mnt/fast/cache"
    assert result["checkpoint"]["command"] == repeat
    assert f"Plan saved. Install it with {repeat} --yes." in output_lines(out.err)
    # The survey measures the named cache's filesystem, which the plan counts the cache allowance on.
    assert {options["cache"] for _, options in sparks.surveys} == {"/mnt/fast/cache"}
    # Without --yes and without a terminal, the request names both ways forward.
    assert sparkring.main(["install", "--profile", PROFILE, "--json", *options]) == 3
    refused = json.loads(capsys.readouterr().out)
    assert refused["field"] == "approval" and refused["message"] == (
        f"Approve this installation with {repeat} --yes, or review its plan first with {repeat} --plan. Nothing has "
        "been changed.")
    assert not events


def test_cancelling_the_checkpoint_prompt_keeps_setup_and_names_the_command(machine, sparks, first_installation,
                                                                          monkeypatch, capsys):
    events, previous, assets, _ = machine
    # No Spark holds a copy: the 102.6 GiB download needs its own answer after setup's approval.
    sparks.survey = holding({host: None for host in ("root@192.0.2.10", "root@192.0.2.11")})

    class Terminal:
        def isatty(self):
            return True
    monkeypatch.setattr(flow.sys, "stdin", Terminal())
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda prompt: prompts.append(prompt) or "")
    assert sparkring.main(["install", "--profile", PROFILE]) == 2
    assert prompts == ["Proceed with this checkpoint plan? [y/N]: "] and first_installation == [[]]
    assert ("Error: Cancelled before any checkpoint or model change; setup is complete. Repeat "
            f"{REPEAT} to review the checkpoint plan again.") in output_lines(capsys.readouterr().err)
    assert (controller.STATE / "cluster.json").exists() and not events
    # Answering y approves the plan it printed, which then bounds later --yes runs as a reviewed plan.
    monkeypatch.setattr(builtins, "input", lambda prompt: "y")
    assert sparkring.main(["install", "--profile", PROFILE]) == 0
    assert assets.prepared["plan"]["approval"] == "prompt"
    [deployment] = deployments_besides(previous)
    assert saved({"deployment": str(controller.STATE / "deployments" / deployment)})["reviewed"] is True


def test_retained_deployments_are_named_and_their_receipts_passed_for_refresh(machine, capsys):
    events, previous, assets, _ = machine
    # A retained deployment that serves the owner's copy in place on both Sparks.
    retained = controller.STATE / "deployments" / "qwen38-flash-next-tp2-iold"
    node.save(retained, "deployment.lock.json", {"id": "id-old", "site": {
        "workspace": "/srv/sparkring/test/qwen38-flash-next-tp2-iold",
        "ranks": [{"host": "root@192.0.2.10", "model": FOLDER}, {"host": "root@192.0.2.11", "model": FOLDER}]}})
    # A lock without an ID cannot bind its receipt to its deployment, so it is neither named nor passed.
    unnamed = controller.STATE / "deployments" / "qwen38-flash-next-tp2-iunnamed"
    node.save(unnamed, "deployment.lock.json", {"site": {
        "workspace": "/srv/sparkring/test/qwen38-flash-next-tp2-iunnamed",
        "ranks": [{"host": "root@192.0.2.10", "model": FOLDER}]}})
    assert command() == 0
    out = capsys.readouterr()
    assert ("Also updates the recorded file identities of the retained deployments that serve the linked folder: "
            "qwen38-flash-next-tp2-iold") in out.err
    # Each receipt carries the deployment its workspace's owner record must name.
    receipt = {"path": "/srv/sparkring/test/qwen38-flash-next-tp2-iold/installer/model.json",
               "deployment": "id-old"}
    assert assets.prepared["receipts"] == {0: [receipt], 1: [receipt]}
    assert json.loads(out.out)["checkpoint"]["refreshed_receipts"] == 1


# End to end: survey, plan, adoption, download and fabric copy on two simulated Sparks.

LINUX = pytest.mark.skipif(not sys.platform.startswith("linux"), reason=(
    "the end-to-end checkpoint installation needs Linux: hard links through /proc/self/fd, flock, POSIX "
    "symlinks and loopback addresses standing in for the fabric"))
MAIN_COMMIT = "7c4f1bc1a2d6847e0cbc01ac6b823f00251de8dd"
CACHE = "/home/code/.cache/huggingface"
MAIN_CACHE = CACHE + "/hub/models--local-inference-lab--Qwen3.8-Flash-Next-NVFP4"
# Host paths that a simulated Spark keeps below its own root directory.
HOST_PATHS = ("/srv/", "/var/lib/sparkring/", "/home/", "/var/tmp/")
FAKE_DOCKER = r'''#!{python}
"""Stand-in for one simulated Spark's docker CLI.

It answers the checkpoint survey's queries, reports whether the serving image
is present, and emulates the pinned image's hf_hub_download into the run's one
bind mount. Every call is appended to the shared event log.
"""
import json, os, sys
spark = json.load(open(os.environ["FAKE_DOCKER"]))
args = sys.argv[1:]
if args[:2] == ["--context", "default"]:
    args = args[2:]
with open(spark["log"], "a") as log:
    log.write(json.dumps({"rank": spark["rank"], "docker": args}) + "\n")
present = json.load(open(spark["images"]))
if args[:1] in (["ps"], ["volume"]):
    sys.exit(0)
if args[:1] == ["info"]:
    print(json.dumps({"Driver": "overlay2", "DockerRootDir": "/var/lib/docker",
                      "SecurityOptions": ["name=seccomp,profile=builtin"]}))
    sys.exit(0)
if args[:2] == ["container", "inspect"]:
    sys.exit(1)
if args[:2] == ["image", "inspect"]:
    if args[-1] in present:
        print(json.dumps([{"Id": args[-1]}]))
        sys.exit(0)
    print("Error: No such image: " + args[-1], file=sys.stderr)
    sys.exit(1)
if args[:1] == ["run"]:
    if args[args.index("--pull") + 1] != "never" or args[args.index("--entrypoint") + 2] not in present:
        print("Unable to find image locally", file=sys.stderr)
        sys.exit(125)
    mount = dict(item.split("=", 1) for item in args[args.index("--mount") + 1].split(","))
    repository, revision, *names = args[args.index("-c") + 2:]
    for name in names:
        path = os.path.join(mount["src"], name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".incomplete", "wb") as output:
            output.write(bytes.fromhex(spark["hub"][name]))
        os.replace(path + ".incomplete", path)
        record = os.path.join(mount["src"], ".cache", "huggingface", "download", name + ".metadata")
        os.makedirs(os.path.dirname(record), exist_ok=True)
        with open(record, "w") as output:
            output.write(revision + "\n" + "etag\n1700000000.0\n")
    sys.exit(0)
print("unexpected docker call: " + " ".join(args), file=sys.stderr)
sys.exit(2)
'''


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git_blob(value):
    return hashlib.sha1(b"blob %d\0" % len(value) + value).hexdigest()


def small_checkpoint(real):
    """Small contents under every file name of the pinned revision, and their pin manifest.

    The manifest keeps the revision's names, index, weights, optional files and
    LFS flags; each file holds a few kilobytes.
    """
    data = {}
    for name in sorted(real["files"]):
        seed = hashlib.sha256(name.encode()).digest()
        data[name] = seed * (64 if name in real["weights"] else 4) + name.encode()
    data[real["index"]] = json.dumps({"metadata": {}, "weight_map": {
        f"model.layers.{number}.weight": name for number, name in enumerate(real["weights"])}}).encode()
    files = {}
    for name, value in data.items():
        files[name] = {"size": len(value), "sha256": digest(value), "git_blob": git_blob(value)}
        if real["files"][name].get("lfs"):
            files[name]["lfs"] = True
    pins = {**{key: real[key] for key in ("schema", "repository", "revision", "index", "weights", "optional")},
            "files": files}
    return data, pins


class SimulatedSparks:
    """Two Sparks with their own root directories, docker and SSH, on this machine.

    Node A's SSH to a Spark runs the survey probe it receives, unchanged except
    for its ``root`` option, against that Spark's root directory. Rank
    operations (``installer_runner.ssh``) run in this process, one at a time, on
    that Spark's view: host paths below ``/srv/``, ``/var/lib/sparkring/``,
    ``/home/`` and ``/var/tmp/`` in the deployment lock and in the operation's
    input are paths below the Spark's root, and so are the placement module's
    mount table and record directory. The fabric receiver and sender run as
    subprocesses on the same views, over loopback addresses that stand in for
    each fabric function's address. ``hub`` holds what huggingface.co serves.
    """

    HOSTS = ("root@192.0.2.10", "root@192.0.2.11")
    NAMES = ("spark-10", "spark-11")

    def __init__(self, base, pins, hub):
        self.base, self.pins = base, pins
        self.roots = [base / name for name in self.NAMES]
        self.log = base / "events.jsonl"
        self.bin = base / "bin"
        self.guard = threading.Lock()
        self.translated = set()
        self.operations = []
        self.blobs = {}
        self.bin.mkdir(parents=True)
        program = self.bin / "docker"
        program.write_text(FAKE_DOCKER.replace("{python}", sys.executable))
        program.chmod(0o755)
        uid = os.getuid() if os.getuid() >= 1000 else 1000
        for rank, root in enumerate(self.roots):
            for directory in ("etc", "proc/self", "root", "home/code", "srv/sparkring/test", "var/tmp"):
                (root / directory).mkdir(parents=True, exist_ok=True)
            (root / "etc/passwd").write_text(f"root:x:0:0::/root:/bin/bash\ncode:x:{uid}:{uid}::/home/code:/bin/bash\n")
            (root / "proc/self/mountinfo").write_text("21 1 259:2 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n")
            state = base / "docker" / self.NAMES[rank]
            state.mkdir(parents=True)
            (state / "images.json").write_text("[]")
            (state / "config.json").write_text(json.dumps({
                "rank": rank, "log": str(self.log), "images": str(state / "images.json"),
                "hub": {name: value.hex() for name, value in hub.items()}}))
        self.log.touch()

    def path(self, rank, host_path):
        return self.roots[rank] / host_path.lstrip("/")

    def cache(self, rank, repository, contents, commit):
        """A Hugging Face cache repository folder: content-addressed blobs, snapshot links and ``refs/main``."""
        folder = self.path(rank, repository)
        for name, value in contents.items():
            blob = folder / "blobs" / (digest(value) if self.pins["files"][name].get("lfs") else git_blob(value))
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes(value)
            self.blobs[rank, name] = blob
            link = folder / "snapshots" / commit / name
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(os.path.relpath(blob, link.parent))
        (folder / "refs").mkdir()
        (folder / "refs/main").write_text(commit)
        return folder

    def download_folder(self, rank, folder, contents):
        """An ``hf download --local-dir`` folder of the pinned revision with the client's download records."""
        folder = self.path(rank, folder)
        for name, value in contents.items():
            target = folder / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(value)
            pin = self.pins["files"][name]
            record = folder / ".cache/huggingface/download" / (name + ".metadata")
            record.parent.mkdir(parents=True, exist_ok=True)
            record.write_text(f"{self.pins['revision']}\n{pin['sha256'] if pin.get('lfs') else pin['git_blob']}\n"
                              f"{target.stat().st_mtime + 1}\n")
        return folder

    def rank(self, host):
        return self.HOSTS.index(host)

    def docker(self, rank):
        return self.base / "docker" / self.NAMES[rank] / "config.json"

    def event(self, **value):
        with open(self.log, "a") as log:
            log.write(json.dumps(value) + "\n")

    def events(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def images(self, rank):
        return json.loads((self.base / "docker" / self.NAMES[rank] / "images.json").read_text())

    def load_image(self, rank, image):
        (self.base / "docker" / self.NAMES[rank] / "images.json").write_text(json.dumps([image]))
        self.event(rank=rank, image_present=image)

    def view(self, rank, value):
        """``value`` with every host path below the simulated Spark's root."""
        if isinstance(value, dict):
            return {key: self.view(rank, item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.view(rank, item) for item in value]
        if isinstance(value, str) and value.startswith(HOST_PATHS):
            return str(self.roots[rank]) + value
        return value

    def program(self, rank, code):
        """Shipped Python source run on ``rank``'s view: its checkpoint paths, mount table and records."""
        for prefix in ("/srv/sparkring/", "/var/lib/sparkring/", "/proc/self/mountinfo"):
            code = code.replace(prefix, str(self.roots[rank]) + prefix)
        return code

    def administration(self, host, argv, *, data=None, timeout=None):
        """``discovery.ssh`` from Node A: access checks, surveys and image checks."""
        rank = self.rank(host)
        if argv == ["sudo", "-n", "true"]:
            return ""
        if argv == flow.SURVEY_COMMAND:
            pins, options = probe_inputs(data)
            options = {**options, "root": str(self.roots[rank])}
            source = (data.rstrip().rsplit("\n", 1)[0] + "\nprint(json.dumps(survey(json.loads("
                      + repr(json.dumps(pins)) + "), json.loads(" + repr(json.dumps(options)) + "))))\n")
            done = subprocess.run([sys.executable, "-I", "-B", "-"], input=source, capture_output=True, text=True,
                                  timeout=timeout, env={**os.environ, "FAKE_DOCKER": str(self.docker(rank))})
            if done.returncode:
                raise RuntimeError(f"{host}: {done.stderr.strip()}")
            document = json.loads(done.stdout)
            document["host"] = self.NAMES[rank]
            return json.dumps(document)
        if argv[:7] == ["sudo", "-n", "docker", "--context", "default", "image", "inspect"]:
            if argv[-1] in self.images(rank):
                return argv[-1] + "\n"
            raise RuntimeError(f"{host}: Error: No such image: {argv[-1]}")
        raise AssertionError(f"unexpected SSH command on {host}: {argv}")

    def operation(self, target, argv, *, data=None, timeout=7200):
        """``installer_runner.ssh`` for rank operations: ``perform`` on the Spark's view."""
        rank = self.rank(target)
        assert argv[:5] == ["python3", "-I", "-B", "-c", installer_runner.HOST], argv[:4]
        payload, operation, number = argv[6], argv[7], int(argv[8])
        assert number == rank
        lock = self.view(rank, json.loads(base64.b64decode(payload)))
        text = data.decode() if isinstance(data, bytes) else data or ""
        stdin = json.dumps(self.view(rank, json.loads(text))) if text else ""
        root = self.roots[rank]
        with self.guard:
            saved = (installer_host.CHECKPOINTS, place.RECORDS, place.MOUNTINFO, sys.stdin,
                     os.environ.get("FAKE_DOCKER"))
            installer_host.CHECKPOINTS = root / "var/lib/sparkring/checkpoints"
            place.RECORDS = str(root / "var/lib/sparkring/checkpoints/files")
            place.MOUNTINFO = str(root / "proc/self/mountinfo")
            sys.stdin = io.StringIO(stdin)
            os.environ["FAKE_DOCKER"] = str(self.docker(rank))
            self.translated.add(id(lock))
            try:
                result = installer_host.perform(operation, lock, number)
            except Exception as error:
                raise RuntimeError(f"{target}: Traceback (most recent call last):\n"
                                   f"{type(error).__name__}: {error}") from error
            finally:
                self.translated.discard(id(lock))
                installer_host.CHECKPOINTS, place.RECORDS, place.MOUNTINFO, sys.stdin = saved[:4]
                if saved[4] is None:
                    os.environ.pop("FAKE_DOCKER", None)
                else:
                    os.environ["FAKE_DOCKER"] = saved[4]
            self.operations.append((rank, operation))
        return json.dumps(result).replace(str(root), "")

    def transport(self, cluster, directory):
        """The bulk transport of ``sparkring install``, running each program on its Spark's view."""
        simulation = self

        class Transport:
            mode = "fiber-ssh"

            def __init__(self):
                self.hosts = copy.deepcopy(cluster["plan"]["spec"]["hosts"])
                for host in self.hosts:
                    for port in host["data_interfaces"]:
                        port["address"] = port["address"].replace("198.18.", "127.18.", 1)

            def verify(self):
                return {"transport": "fiber-ssh", "caller_relay": False}

            def command(self, rank, argv):
                if rank:
                    assert argv[:2] == ["sudo", "-n"]
                    argv = argv[2:]
                assert argv[:3] == ["python3", "-I", "-c"], argv[:3]
                simulation.event(rank=rank, program="fabric-receiver" if "def receive_checkpoint" in argv[3]
                                 else "fabric-sender")
                return [sys.executable, "-I", "-c", simulation.program(rank, argv[3])]

            def argv(self, rank):
                raise AssertionError("the checkpoint copy fell back to rsync")
        return Transport()


@pytest.fixture
def simulated(machine, monkeypatch, tmp_path):
    """A first installation of the Qwen TP2 profile on two ``SimulatedSparks`` with a small checkpoint.

    Node A's SSH, the rank operations' SSH, docker and the fabric transport are
    simulated; the survey probe, the plan, approval, the prepared runner, the
    rank operations and the fabric receiver and sender are the real code. Image
    distribution loads the serving image on both Sparks. Preparation actions
    other than the checkpoint's report success.
    """
    (controller.STATE / "active.json").unlink()
    card = installer.setup.selection(PROFILE)
    real = installer.checkpoint_pins(card)
    data, pins = small_checkpoint(real)
    required = sorted(name for name in pins["files"] if name not in pins["optional"])
    simulation = SimulatedSparks(tmp_path / "sparks", pins, {"config.json": data["config.json"]})

    def checkpoint_pins(value, **kwargs):
        assert (value["model_repository"], value["model_revision"]) == (real["repository"], real["revision"])
        return copy.deepcopy(pins)
    contract, validate = installer.checkpoint_contract, installer.validate
    monkeypatch.setattr(installer, "checkpoint_pins", checkpoint_pins)
    monkeypatch.setattr(installer, "checkpoint_contract", lambda value: {
        **contract(value), "config_sha256": pins["files"]["config.json"]["sha256"],
        "index_sha256": pins["files"][pins["index"]]["sha256"]})
    sums = tmp_path / "SHA256SUMS"
    sums.write_text("".join(f"{pins['files'][name]['sha256']}  {name}\n" for name in required))
    monkeypatch.setattr(installer_host, "checksum_manifest", lambda profile, revision=None: sums)
    # Rank operations validate the lock they receive; a Spark's view of the validated lock is accepted.
    monkeypatch.setattr(installer, "validate",
                        lambda lock: lock if id(lock) in simulation.translated else validate(lock))
    monkeypatch.setattr(checkpoint_plan, "storage_policy",
                        lambda root=None: {"cache_bytes": 1 << 20, "image_bytes": 1 << 20})
    monkeypatch.setenv("PATH", str(simulation.bin) + os.pathsep + os.environ["PATH"])
    monkeypatch.setattr(flow.discovery, "ssh", simulation.administration)
    monkeypatch.setattr(installer_runner, "ssh", simulation.operation)
    monkeypatch.setattr(flow.fabric_ssh, "Transport", simulation.transport)
    rank_call = installer_runner.Runner._call

    def call(self, target, argv, timeout):
        # Checkpoint actions run the rank operations; the other preparation actions are simulated.
        if argv[1] in ("model", "model-check"):
            return rank_call(self, target, argv, timeout)
        if argv[1] == "source":
            workspace = simulation.path(int(argv[2]), self.lock["site"]["workspace"])
            workspace.mkdir(parents=True, exist_ok=True)
            (workspace / ".installer-owner.json").write_text(json.dumps({"deployment": self.lock["id"]}))
        return dict(OK)
    monkeypatch.setattr(installer_runner.Runner, "_call", call)

    class Assets(ASSETS):
        def sync_packages(self):
            return {"updated": []}

        def images(self, card):
            for rank in range(2):
                simulation.load_image(rank, card["image_id"])
            return {"image_id": card["image_id"]}
    monkeypatch.setattr(flow.install_assets, "Assets", Assets)
    weights = sorted(pins["weights"])
    return SimpleNamespace(sparks=simulation, data=data, pins=pins, required=required, weights=weights,
                           others=[name for name in required if name not in weights and name != "config.json"],
                           size={name: pins["files"][name]["size"] for name in required})


def install_simulated(request, capsys, *protected):
    """Run ``sudo sparkring install --yes --json`` while the audit hook refuses changes under ``protected``."""
    protect = request.getfixturevalue("audited")
    with protect(*protected):
        code = sparkring.main(["install", "--profile", PROFILE, "--yes", "--json"])
    out = capsys.readouterr()
    assert code == 0, out.err[-3000:]
    return json.loads(out.out), output_lines(out.err)


def journal_origins(directory):
    state = directory.parent / ("." + directory.name + ".sparkring")
    return {name: entry["origin"] for name, entry in json.loads((state / "journal.json").read_text())["files"].items()}


@LINUX
def test_node_a_links_a_main_cache_downloads_config_once_and_streams_to_an_empty_worker(
        simulated, request, capsys):
    sparks, pins, required, weights, others, size = (simulated.sparks, simulated.pins, simulated.required,
                                                     simulated.weights, simulated.others, simulated.size)
    # Node A's Hugging Face cache of branch main: only config.json and README.md differ from the pinned revision.
    main = {**simulated.data, "config.json": b'{"model_type": "qwen3_8_main_branch"}\n', "README.md": b"# main\n"}
    sparks.cache(0, MAIN_CACHE, main, MAIN_COMMIT)
    cache = sparks.path(0, CACHE)
    before = tree_state(cache)
    result, lines = install_simulated(request, capsys, cache)
    assert result["state"] == "complete" and not result["replaces"]

    # The plan links the shards from the cache, copies the other files it holds,
    # and downloads only config.json, whose pinned blob the cache lacks.
    assert "Node 0 spark-10 -> " + DIRECTORY in lines and "    from " + MAIN_CACHE in lines
    assert any(line.startswith("        Hugging Face cache, snapshot 7c4f1bc1a2d6 (branch main), in code's home "
                               "(the operator's)") for line in lines)
    assert "        52 of 53 files identified by their blob names; config.json differs from the pinned revision" in lines
    assert any(line.startswith("    hard-link 41 weight files (no copy, no extra space); copy 11 other files (")
               for line in lines)
    assert any(line.startswith("Node 1 spark-11: no copy found") for line in lines)
    assert any(line.startswith("Downloads ") and line.endswith(" from huggingface.co on Node 0.") for line in lines)
    checkpoint = result["checkpoint"]
    assert checkpoint["approval"] == "command-line" and checkpoint["hub_files"] == ["config.json"]
    assert checkpoint["nodes"][0]["bytes"] == {"present": 0, "link": sum(size[n] for n in weights),
                                               "copy": sum(size[n] for n in others), "pool": 0, "receive": 0,
                                               "hub": size["config.json"]}
    assert checkpoint["nodes"][1]["bytes"]["receive"] == sum(size.values())
    assert {name: entry["action"] for name, entry in saved(result)["nodes"][0]["files"].items()} == {
        **{name: "link" for name in weights}, **{name: "copy" for name in others}, "config.json": "hub"}

    # Node 0 adopted and downloaded; then it streamed every file to the worker over the fabric.
    outcome = checkpoint["result"]
    assert outcome["donor_rank"] == 0 and outcome["downloaded"] == ["config.json"] and outcome["pooled"] == []
    assert outcome["received"] == [{"source": 0, "target": 1, "names": required, "level": 0, "transport": "fabric"}]
    assert sorted(sparks.operations) == sorted([
        (0, "model-adopt"), (1, "model-adopt"), (0, "model-fetch"), (1, "model-transfer-prepare"),
        (1, "model-transfer-complete"), (0, "model"), (0, "model-check"), (1, "model"), (1, "model-check")])
    first, second = (sparks.path(rank, DIRECTORY) for rank in range(2))
    for directory in (first, second):
        assert sorted(os.listdir(directory)) == required
        assert all(digest((directory / name).read_bytes()) == pins["files"][name]["sha256"] for name in required)
        state = directory.parent / ("." + directory.name + ".sparkring")
        assert sorted(os.listdir(state)) == ["journal.json", "lock", "owner.json"]
    assert all(os.stat(first / name).st_ino == os.stat(sparks.blobs[0, name]).st_ino for name in weights)
    assert all(os.stat(first / name).st_nlink == 1 for name in [*others, "config.json"])
    assert all(os.stat(second / name).st_nlink == 1 for name in required)
    assert journal_origins(first) == {**{name: "link" for name in weights}, **{name: "copy" for name in others},
                                      "config.json": "hub"}
    assert journal_origins(second) == {name: "fabric" for name in required}
    lock = installer.load(result["deployment"])
    receipts = [json.loads(sparks.path(rank, lock["site"]["workspace"]).joinpath("installer/model.json").read_text())
                for rank in range(2)]
    assert [receipt["origin"] for receipt in receipts] == ["adopted-local-copy", "verified-fabric-copy"]
    assert receipts[0]["fetched"] == ["config.json"]

    # config.json was downloaded once, on Node A, into its fetch staging directory
    # only, after the serving image was present there.
    log = sparks.events()
    runs = [(index, entry) for index, entry in enumerate(log) if entry.get("docker", [None])[0] == "run"]
    assert len(runs) == 1
    index, run = runs[0]
    argv = run["docker"]
    assert run["rank"] == 0 and argv[-3:] == [pins["repository"], pins["revision"], "config.json"]
    assert argv[argv.index("--pull") + 1] == "never" and argv[argv.index("--runtime") + 1] == "runc"
    assert argv.count("--mount") == 1 and "-v" not in argv and "--volume" not in argv
    fetch = first.parent / ("." + first.name + ".sparkring") / "fetch"
    assert argv[argv.index("--mount") + 1] == f"type=bind,src={fetch},dst=/fetch"
    assert log.index({"rank": 0, "image_present": lock["selection"]["image_id"]}) < index
    assert [entry["program"] for entry in log if "program" in entry] == ["fabric-receiver", "fabric-sender"]

    # The cache was only read: its weight blobs gained a hard link, nothing else changed.
    linked = [os.path.relpath(sparks.blobs[0, name], cache) for name in weights]
    assert only_links_changed(before, tree_state(cache), linked)


@LINUX
def test_each_node_links_its_own_download_folder_and_nothing_is_downloaded_or_copied_between_nodes(
        simulated, request, capsys):
    sparks, required, weights, others = simulated.sparks, simulated.required, simulated.weights, simulated.others
    # The owner's hf download --local-dir folder of the pinned revision on both Sparks.
    folders = [sparks.download_folder(rank, FOLDER, simulated.data) for rank in range(2)]
    before = [tree_state(folder) for folder in folders]
    result, lines = install_simulated(request, capsys, *folders)
    assert result["state"] == "complete"
    assert any(line.startswith("        Hugging Face download folder, commit 60215d26cf5e (the pinned revision)")
               for line in lines)
    assert "        53 of 53 files identified by their download records" in lines
    assert any(line.startswith("    hard-link 41 weight files (no copy, no extra space); copy 12 other files (")
               for line in lines)
    assert any(line.startswith("Node 1 spark-11: as Node 0") for line in lines)
    assert "Nothing is downloaded." in lines
    checkpoint = result["checkpoint"]
    assert checkpoint["hub_files"] == [] and checkpoint["result"]["received"] == []
    assert checkpoint["result"]["complete_after_adoption"] == [0, 1]
    assert sorted(sparks.operations) == sorted([(rank, operation) for rank in range(2)
                                                for operation in ("model-adopt", "model", "model-check")])
    assert not [entry for entry in sparks.events() if entry.get("docker", [None])[0] == "run" or "program" in entry]
    for rank, folder in enumerate(folders):
        directory = sparks.path(rank, DIRECTORY)
        assert sorted(os.listdir(directory)) == required
        assert all(os.stat(directory / name).st_ino == os.stat(folder / name).st_ino for name in weights)
        assert all(os.stat(directory / name).st_nlink == 1 for name in [*others, "config.json"])
        assert journal_origins(directory) == {**{name: "link" for name in weights},
                                              **{name: "copy" for name in [*others, "config.json"]}}
        assert only_links_changed(before[rank], tree_state(folder), weights)


# A native mesh that SparkRing can adopt, as native-mesh inspection reports it without its host address.
MESH = {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json", "site_sha256": "c" * 64,
                      "plan_sha256": "d" * 64}, "interface": "eth0", "unit": "sparkring-mesh.service"}


def running_meshes(rank):
    return {**MESH, "host_ip": f"192.0.2.{110 + rank}"}


def four(monkeypatch):
    """Record a four-Spark ring whose Sparks each run an adoptable native mesh; returns the cluster.

    SSH answers as ``Sparks`` does: every Spark holds the owner's copy of the checkpoint.
    """
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    remote = Sparks()
    remote.mesh = running_meshes
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
        # The installation's other SSH commands: access checks, checkpoint
        # surveys, image checks and the running mesh of every Spark.
        self.administration = Sparks()
        self.administration.mesh = running_meshes
        monkeypatch.setattr(hairpin_ring, "remote", self.remote)
        monkeypatch.setattr(hairpin_ring, "local", self.local)
        monkeypatch.setattr(hairpin_ring, "POLL", 0)
        monkeypatch.setattr(hairpin_ring, "DISPATCH_RETRY", 0)
        monkeypatch.setattr(controller, "collect", self.collect)
        monkeypatch.setattr(flow.discovery, "ssh", self.administration)
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

    def setup_invoke(self, host, argv, *, data=None, **_):
        """Setup's commands after the network steps: node verify, configure, workspace and pings."""
        self.setup_commands.append((self.ranks[host], tuple(argv)))
        if argv[3:5] == ["node", "configure"]:
            self.sparks[self.ranks[host]].write(hairpin.FABRIC, json.loads(data))
        return "{}"

    def changes(self):
        return [spark.changes() for spark in self.sparks]


def enrolled_ring(ring, monkeypatch):
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
    receipts = installed_ring(ring) if installed else enrolled_ring(ring, monkeypatch)
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


def test_install_after_a_reboot_names_the_hairpin_setting_in_the_mesh_refusal(machine, monkeypatch, capsys):
    events, _, _, _ = machine
    value = four(monkeypatch)
    needing(value["plan"], [1, 2, 3])
    # Only Node A's mesh runs: the start check refused the workers' meshes.
    flow.discovery.ssh.mesh = lambda rank: running_meshes(0) if rank == 0 else None
    monkeypatch.setattr(flow.hairpin_ring, "ensure", lambda *a, **k: pytest.fail("the hairpin step ran"))
    assert sparkring.main(["install", "--profile", TP4, "--yes", "--json"]) == 2
    out = capsys.readouterr()
    result = json.loads(out.out)
    assert result["message"] == (
        "Only part of the native mesh is enabled or running. Use a reviewed --fresh-mesh replacement after "
        "inspection. The ConnectX hairpin setting is not in effect on ranks 1-3, so their mesh services cannot "
        "start; run sudo sparkring hairpin on Node A first.")
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
    enrolled_ring(ring, monkeypatch)
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
    assert ("--yes approve the displayed setup, checkpoint plan, the listed ConnectX driver restarts on an idle "
            "ring, and model replacement; SSH trust is still required") in help_text
