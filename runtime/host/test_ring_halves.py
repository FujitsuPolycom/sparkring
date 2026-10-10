"""Two models on one four-Spark ring: slots, switching rules, mesh mode and endpoints.

The public ``sparkring install`` command runs against a simulated four-Spark
ring (``test_install_workflow``'s ``machine`` and ``sparks`` fixtures). Each
deployment's memory settle, start and stop is recorded by profile and
placement; a start or stop leaves the state a completed operation leaves, so
the switching rules see which models run.
"""
import json
from pathlib import Path

import pytest

from runtime.common import installer
from runtime.host import controller, install_workflow as flow, node, placement, recovery, rollout
from runtime.host import test_recovery
from runtime.host.test_install_workflow import cluster, machine, sparks  # noqa: F401 (fixtures)
from scripts import sparkring

QWEN = "qwen38-flash-next-tp2"
GLM = "glm53-flash-nvfp4-spark-tp2"
TP4 = "qwen38-flash-next-qad-tp4"
MESH = {"reference": {"site_path": "/etc/sparkring/managed-mesh/site.json", "site_sha256": "c" * 64,
                      "plan_sha256": "d" * 64}, "interface": "eth0", "unit": "sparkring-mesh.service"}


def label(path):
    lock = installer.read(Path(path) / "deployment.lock.json")
    where = placement.from_lock(lock)
    return lock["selection"]["profile"] + ("@" + "".join(map(str, where)) if where else "")


class Ring:
    """The simulated ring's record of model operations and of the installer's other steps."""

    def __init__(self, events):
        self.events = events
        self.workloads = []
        self.fail = None


@pytest.fixture
def ring(machine, sparks, monkeypatch):  # noqa: F811 (fixture argument)
    events, previous, _, _ = machine
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    monkeypatch.setattr(flow, "require_head", lambda *_: value["plan"]["nodes"][0]["node_id"])
    (controller.STATE / "active.json").unlink()
    previous.rmdir()
    sparks.mesh = lambda rank: {**MESH, "host_ip": f"192.0.2.{110 + rank}"}
    simulated = Ring(events)

    def operation(path, action, **kwargs):
        name = label(path)
        events.append(f"{name}:{action}")
        if simulated.fail == (name, action):
            raise ValueError("injected failure")
        if action in ("up", "down"):
            state = installer.read(Path(path) / "state.json") if (Path(path) / "state.json").exists() else {}
            node.save(path, "state.json", {"generation": state.get("generation", 0) + 1, "operation": action,
                                           "complete": True})
        return {"verified": True}
    monkeypatch.setattr(flow.retained_source, "apply", operation)
    # The page-cache drop and memory settle before each start (runtime.host.memory_settle), by deployment.
    monkeypatch.setattr(flow, "settle_memory", lambda path, **kwargs: events.append(f"{label(path)}:settle-memory")
                        or [])
    monkeypatch.setattr(flow, "park_ring", lambda cluster, **kwargs: events.append("park-ring"))
    monkeypatch.setattr(flow, "check_workloads", lambda *a, **k: simulated.workloads.append(k.get("others")))

    class Transport:
        def __init__(self, cluster, directory):
            self.hosts = cluster["plan"]["spec"]["hosts"]

        def verify(self):
            return {"transport": "fiber-ssh", "caller_relay": False}

        def view(self, ranks):
            events.append(("view", tuple(ranks)))
            return self
    monkeypatch.setattr(flow.fabric_ssh, "Transport", Transport)
    return simulated


def install(*extra):
    return sparkring.main(["install", "--yes", "--json", *extra])


def result(capsys):
    return json.loads(capsys.readouterr().out)


def ops(ring):
    """The model operations since the last call, as ``profile[@half]:operation``, and the mesh parks."""
    found = [event for event in ring.events if isinstance(event, str) and (":" in event and not event.startswith(
        ("prepare:", "check-workloads")) or event == "park-ring")]
    ring.events.clear()
    return found


def recorded():
    """The deployment each slot of the ring records: the whole ring's and each half's."""
    return {slot: (label(path) if path else None) for slot, path in
            ((slot, placement.recorded(controller.STATE, slot)) for slot in (None, *placement.HALVES))}


def test_a_ring_switches_from_one_model_to_two_and_back(ring, capsys):
    assert install("--profile", TP4) == 0
    tp4 = result(capsys)
    assert tp4["nodes"] == 4 and tp4["stops"] == [] and "placement" not in tp4
    ops(ring)

    # Half (0, 1): the running four-Spark model stops, the ring's mesh parks, the half's memory settles and the
    # half starts.
    assert install("--profile", QWEN, "--on", "0,1") == 0
    first = result(capsys)
    assert first["placement"] == [0, 1] and first["nodes"] == 2 and first["replaces"] is None
    assert [(row["profile"], row["placement"]) for row in first["stops"]] == [(TP4, None)]
    assert ops(ring) == [f"{TP4}:down", "park-ring", f"{QWEN}@01:settle-memory", f"{QWEN}@01:up", f"{QWEN}@01:verify"]
    assert ring.workloads[-1] == [Path(tp4["deployment"]).resolve()]
    assert recorded() == {None: None, (0, 1): QWEN + "@01", (2, 3): None}
    switch = placement.journal(controller.STATE, (0, 1))
    assert switch["state"] == "complete" and switch["displaced"] == [str(Path(tp4["deployment"]).resolve())]
    assert first["commands"]["stop"] == "sudo sparkring down --on 0,1 --execute"
    assert first["commands"]["switch_back"] == f"sudo sparkring install --profile {TP4}"
    assert first["checkpoint"]["command"] == f"sudo sparkring install --profile {QWEN} --on 0,1"

    # Half (2, 3): the other half keeps running.
    assert install("--profile", GLM, "--on", "2,3", "--checkpoint", "nvfp4-qad") == 0
    second = result(capsys)
    assert second["stops"] == [] and second["replaces"] is None
    assert ops(ring) == ["park-ring", f"{GLM}@23:settle-memory", f"{GLM}@23:up", f"{GLM}@23:verify"]
    assert recorded() == {None: None, (0, 1): QWEN + "@01", (2, 3): GLM + "@23"}
    assert second["commands"]["switch_back"] is None
    lock = installer.read(Path(second["deployment"]) / "deployment.lock.json")
    assert [row["host"] for row in lock["site"]["ranks"]] == ["root@192.0.2.12", "root@192.0.2.13"]
    assert lock["selection"]["target_variant"] == "nvfp4-qad"

    # Back to the four-Spark model: both halves stop, and the ring's own step serves the mesh again.
    assert install("--profile", TP4) == 0
    back = result(capsys)
    assert back["deployment"] == tp4["deployment"]
    assert sorted((row["profile"], tuple(row["placement"])) for row in back["stops"]) == [(GLM, (2, 3)), (QWEN, (0, 1))]
    assert ops(ring) == [f"{QWEN}@01:down", f"{GLM}@23:down", f"{TP4}:settle-memory", f"{TP4}:up", f"{TP4}:verify"]
    assert recorded() == {None: TP4, (0, 1): None, (2, 3): None}
    assert rollout.active(controller.STATE) == Path(tp4["deployment"]).resolve()
    assert " ; " in back["commands"]["switch_back"] and "--on 0,1" in back["commands"]["switch_back"]
    # The four-Spark model's own start serves the parked mesh again on every Spark; a half's start parks it.
    phases = [phase["id"] for phase in installer.operation_plan(
        installer.read(Path(back["deployment"]) / "deployment.lock.json"), "up")["phases"]]
    assert phases.index("ring-stop") + 1 == phases.index("ring-serve") < phases.index("start-api")
    assert "ring-park" not in phases


def test_installing_on_a_serving_half_replaces_only_that_half(ring, capsys):
    assert install("--profile", QWEN, "--on", "0,1") == 0
    first = result(capsys)
    assert install("--profile", GLM, "--on", "2,3") == 0
    capsys.readouterr()
    ops(ring)
    assert install("--profile", GLM, "--on", "0,1") == 0
    replaced = result(capsys)
    assert Path(replaced["replaces"]).resolve() == Path(first["deployment"]).resolve()
    assert [row["profile"] for row in replaced["stops"]] == [QWEN]
    assert ops(ring) == [f"{QWEN}@01:down", "park-ring", f"{GLM}@01:settle-memory", f"{GLM}@01:up",
                         f"{GLM}@01:verify"]
    assert recorded() == {None: None, (0, 1): GLM + "@01", (2, 3): GLM + "@23"}
    assert replaced["commands"]["switch_back"] == f"sudo sparkring install --profile {QWEN} --on 0,1"


def test_a_failed_half_restarts_the_four_spark_model_it_stopped(ring, capsys):
    assert install("--profile", TP4) == 0
    tp4 = result(capsys)
    ops(ring)
    ring.fail = (QWEN + "@23", "verify")
    assert install("--profile", QWEN, "--on", "2,3") == 2
    failed = result(capsys)
    assert failed["transaction"]["state"] == "failed-recovered"
    # The recovery start of the four-Spark model settles memory on every Spark first, like any start.
    assert ops(ring) == [f"{TP4}:down", "park-ring", f"{QWEN}@23:settle-memory", f"{QWEN}@23:up", f"{QWEN}@23:verify",
                         f"{QWEN}@23:down", f"{TP4}:down", f"{TP4}:settle-memory", f"{TP4}:up", f"{TP4}:verify"]
    assert recorded() == {None: TP4, (0, 1): None, (2, 3): None}
    assert rollout.active(controller.STATE) == Path(tp4["deployment"]).resolve()


def test_a_two_spark_profile_goes_on_the_one_free_half(ring, capsys):
    assert install("--profile", QWEN, "--on", "2,3") == 0
    capsys.readouterr()
    assert install("--profile", GLM) == 0
    out = capsys.readouterr()
    chosen = json.loads(out.out)
    assert chosen["placement"] == [0, 1]
    assert f"{GLM} uses two Sparks: it goes on Sparks 0 and 1 (--on 0,1), the half that serves no model." in out.err


def test_a_two_spark_profile_without_a_free_half_names_the_choices(ring, capsys):
    assert install("--profile", TP4) == 0
    capsys.readouterr()
    assert install("--profile", QWEN) == 3
    refused = result(capsys)
    assert refused["field"] == "placement" and "Choose a half with --on 0,1 or --on 2,3" in refused["message"]
    # The four-Spark model runs on both halves' Sparks, so neither is free.
    assert refused["details"]["lines"] == [f"--on 0,1: Sparks 0 and 1, serving {TP4}",
                                           f"--on 2,3: Sparks 2 and 3, serving {TP4}"]
    assert install("--profile", QWEN, "--on", "0,1") == 0 and install("--profile", GLM, "--on", "2,3") == 0
    capsys.readouterr()
    assert install("--profile", QWEN) == 3
    lines = result(capsys)["details"]["lines"]
    assert lines == [f"--on 0,1: Sparks 0 and 1, serving {QWEN}", f"--on 2,3: Sparks 2 and 3, serving {GLM}"]


@pytest.mark.parametrize("args, message", [
    (["--profile", TP4, "--on", "0,1"], f"{TP4} uses all four Sparks; --on applies to two-Spark profiles"),
    (["--profile", QWEN, "--on", "1,2"], "--on 1,2 is not a half of the ring; use --on 0,1 or --on 2,3"),
])
def test_on_is_refused_for_a_four_spark_profile_or_another_pair_of_sparks(ring, capsys, args, message):
    assert install(*args) == 3
    refused = result(capsys)
    assert refused["field"] == "placement" and refused["message"].startswith(message)
    assert ops(ring) == []


def test_on_is_refused_on_a_pair(machine, capsys):  # noqa: F811 (fixture argument)
    assert install("--profile", QWEN, "--on", "0,1") == 3
    refused = result(capsys)
    assert refused["field"] == "placement" and "this cluster is a pair" in refused["message"]


def test_stopping_a_model_outside_the_requested_half_is_asked(ring, capsys, monkeypatch):
    assert install("--profile", TP4) == 0
    capsys.readouterr()
    ops(ring)
    questions = []

    def confirm(prompt, yes=False, *, default=False):
        questions.append(prompt)
        raise ValueError("Cancelled; no further changes")
    monkeypatch.setattr(flow.controller, "confirm", confirm)
    monkeypatch.setattr(flow.sys.stdin, "isatty", lambda: True)
    assert sparkring.main(["install", "--profile", QWEN, "--on", "0,1"]) == 2
    assert questions == [f"Stop {TP4} on all four Sparks and apply this installation?"]
    assert ops(ring) == [] and recorded()[None] == TP4


def test_the_plan_lists_every_model_the_switch_stops(ring, capsys):
    assert install("--profile", QWEN, "--on", "0,1") == 0 and install("--profile", GLM, "--on", "2,3") == 0
    capsys.readouterr()
    ops(ring)
    assert sparkring.main(["install", "--profile", TP4, "--plan", "--json"]) == 0
    out = capsys.readouterr()
    planned = json.loads(out.out)
    assert planned["state"] == "planned" and [row["profile"] for row in planned["stops"]] == [QWEN, GLM]
    assert f"It stops {QWEN} on Sparks 0 and 1." in out.err and f"It stops {GLM} on Sparks 2 and 3." in out.err
    assert ops(ring) == []


def test_half_two_three_serves_on_spark_twos_lan_address(ring, capsys, monkeypatch):
    value = installer.read(controller.STATE / "cluster.json")
    value["api_address"] = "198.51.100.10"
    # The installation reads each Spark's inventory again; Spark 2's default route leaves on its LAN port.
    lan = next(row for row in value["plan"]["nodes"][2]["facts"]["interfaces"] if row["name"] == "enP7s7")
    lan["ipv4"] = ["198.51.100.12/24"]
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    assert install("--profile", QWEN, "--on", "2,3") == 0
    half = result(capsys)
    assert half["api_url"] == "http://198.51.100.12:8000/v1"
    assert install("--profile", GLM, "--on", "0,1") == 0
    assert result(capsys)["api_url"] == "http://198.51.100.10:8000/v1"


def test_a_half_moves_its_checkpoint_between_its_own_sparks(ring, capsys):
    assert install("--profile", QWEN, "--on", "2,3") == 0
    capsys.readouterr()
    assert ("view", (2, 3)) in ring.events


def test_parking_the_ring_runs_on_every_spark():
    value = cluster(4)
    calls = []

    def invoke(host, argv):
        calls.append((host, argv))
        return json.dumps({"ok": True, "parked": []})
    flow.park_ring(value, invoke=invoke)
    assert sorted(calls) == [(f"root@192.0.2.{10 + rank}", ["sudo", "-n", "/usr/bin/sparkring", "node", "mesh-park"])
                             for rank in range(4)]
    assert flow.park_ring(cluster(2), invoke=invoke) == [] and len(calls) == 4


# sparkring up, down, status and automatic recovery of two halves, with each
# deployment's real operation plan run against simulated Sparks.

class RingSparks(test_recovery.Sparks):
    """The simulated Sparks of test_recovery, with each container state kept per Spark so that two halves do not mix."""

    def __init__(self, hosts):
        super().__init__(hosts)
        self.running = {host: False for host in hosts}
        self.exit_codes = {host: 0 for host in hosts}
        self.parks = []

    def rank_operation(self, runner, target, argv, timeout):
        operation, rank = argv[1], int(argv[2])
        self.operations.append((operation, target))
        if operation == "status":
            return {"returncode": 0, "stderr": "", "uncertain": False, "stdout": json.dumps(self.state(target, rank))}
        if operation == "start":
            self.running[target] = True
        elif operation == "stop":
            self.running[target] = False
        elif operation == "running" and not self.running[target]:
            return {"returncode": 1, "stdout": "", "stderr": "Rank is not running", "uncertain": False}
        return {"returncode": 0, "stdout": "ok", "stderr": "", "uncertain": False}

    def state(self, host, rank):
        running = self.running[host]
        value = {"schema": "sparkring-model-observation/v1", "rank": rank, "present": True, "running": running,
                 "health": "healthy" if running and rank == 0 else None, "container_name": f"sr-test-r{rank}"}
        if not running:
            value.update(exit_code=self.exit_codes[host], finished_at="2026-09-30T17:02:11.5Z")
        return value

    def ssh(self, host, argv, **kwargs):
        if argv[-1] == "mesh-park":
            self.parks.append(host)
            return json.dumps({"ok": True, "parked": []})
        if recovery.CONTAINER in argv:
            rank = int(argv[-3].rsplit("-r", 1)[1])
            return json.dumps({**self.state(host, rank), "name": argv[-3], "owned": True})
        return super().ssh(host, argv, **kwargs)


@pytest.fixture
def halves(tmp_path, monkeypatch):
    """Qwen TP2 on Sparks 0 and 1 and GLM TP2 on Sparks 2 and 3, each started by sparkring up --on."""
    from runtime.common import distribution
    from runtime.host import discovery
    from scripts import installer_runner
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = cluster(4)
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(distribution, "identity", lambda _: "a" * 40)
    monkeypatch.setattr(distribution, "bundle", lambda root, dest: dest.write_bytes(b"retained source"))
    monkeypatch.setattr(controller, "_hairpin_problem", lambda: pytest.fail("a half needs no hairpin check"))
    simulated = RingSparks([host["host"] for host in value["plan"]["spec"]["hosts"]])
    monkeypatch.setattr(discovery, "ssh", simulated.ssh)
    monkeypatch.setattr(installer_runner.Runner, "_call", lambda runner, target, argv, timeout:
                        simulated.rank_operation(runner, target, argv, timeout))
    assert controller.lifecycle(["up", QWEN, "--on", "0,1", "--execute"]) == 0
    assert controller.lifecycle(["up", GLM, "--on", "2,3", "--execute"]) == 0
    simulated.first = (controller.STATE / "deployments" / (QWEN + "-on-0-1")).resolve()
    simulated.second = (controller.STATE / "deployments" / (GLM + "-on-2-3")).resolve()
    for directory in (simulated.first, simulated.second):
        recovery.started(directory)
    return simulated


def test_up_on_a_half_parks_the_ring_and_starts_on_its_own_sparks(halves):
    hosts = halves.hosts
    assert sorted(halves.parks) == sorted(hosts * 2)
    assert all(halves.running.values())
    for operation in ("ring-park", "gid-serve", "start"):
        assert sorted(host for name, host in halves.operations if name == operation) == sorted(hosts)
    assert recorded() == {None: None, (0, 1): QWEN + "@01", (2, 3): GLM + "@23"}
    lock = installer.read(halves.second / "deployment.lock.json")
    assert [row["host_ip"] for row in lock["site"]["ranks"]] == ["198.18.3.1", "198.18.3.2"]


def test_status_lists_each_half_with_its_own_api(halves, capsys):
    assert controller.lifecycle(["status"]) == 0
    out = capsys.readouterr().out
    first, second = out.index("Sparks 0 and 1:"), out.index("Sparks 2 and 3:")
    assert first < out.index("http://192.0.2.10:8000/v1") < second < out.index("http://192.0.2.12:8000/v1")
    assert out.count("Automatic recovery: on") == 2
    assert controller.lifecycle(["status", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [(row["placement"], row["deployment"]["api_url"]) for row in document["slots"]] == [
        ([0, 1], "http://192.0.2.10:8000/v1"), ([2, 3], "http://192.0.2.12:8000/v1")]
    assert controller.lifecycle(["status", "--on", "2,3", "--json"]) == 0
    only = json.loads(capsys.readouterr().out)
    assert [row["placement"] for row in only["slots"]] == [[2, 3]]


def test_down_without_a_half_names_the_choices_and_with_one_stops_only_that_half(halves, capsys):
    with pytest.raises(ValueError, match="more than one placement") as refused:
        controller.lifecycle(["down", "--execute"])
    assert refused.value.details["lines"] == [f"Sparks 0 and 1: {QWEN} (started); name it with --on 0,1",
                                              f"Sparks 2 and 3: {GLM} (started); name it with --on 2,3"]
    halves.operations.clear()
    assert controller.lifecycle(["down", "--on", "2,3", "--execute"]) == 0
    assert {host for name, host in halves.operations if name == "stop"} == set(halves.hosts[2:])
    assert [halves.running[host] for host in halves.hosts] == [True, True, False, False]
    assert recorded()[(2, 3)] == GLM + "@23"


def test_a_four_spark_model_does_not_start_while_a_half_serves(halves):
    with pytest.raises(ValueError, match=f"{QWEN} runs on Sparks 0 and 1. Stop it first: "
                                         "sudo sparkring down --on 0,1 --execute"):
        controller.lifecycle(["up", TP4, "--execute"])


def test_recovery_restarts_only_the_half_that_stopped(halves):
    clock = test_recovery.Clock()
    assert recovery.check(now=clock, api=lambda url: test_recovery.SERVING, tunnel={})["state"] == "serving"
    halves.running[halves.hosts[3]] = False
    halves.operations.clear()
    first = recovery.check(now=clock, api=lambda url: test_recovery.SERVING, tunnel={})
    assert [(Path(row["deployment"]).name, row["state"]) for row in first["deployments"]] == [
        (halves.first.name, "serving"), (halves.second.name, "confirming")]
    clock.advance(60)
    second = recovery.check(now=clock, api=lambda url: test_recovery.SERVING, tunnel={})
    assert second["state"] == "recovered"
    touched = {host for name, host in halves.operations if name in ("stop", "start", "gid-serve", "ring-park")}
    assert touched == set(halves.hosts[2:]) and all(halves.running.values())


def test_recover_status_and_off_cover_each_half(halves, capsys):
    assert recovery.main([]) == 0
    out = capsys.readouterr().out
    assert out.count("Deployment: ") == 2 and str(halves.second) in out
    assert recovery.main(["off"]) == 0
    capsys.readouterr()
    assert not recovery.record_of(recovery.load(), halves.first)["enabled"]
    assert not recovery.record_of(recovery.load(), halves.second)["enabled"]


def test_retention_keeps_each_halfs_active_and_rollback_deployments(ring, capsys):
    from runtime.host import checkpoints
    assert install("--profile", TP4) == 0
    tp4 = Path(result(capsys)["deployment"]).resolve()
    assert install("--profile", QWEN, "--on", "0,1") == 0
    first = Path(result(capsys)["deployment"]).resolve()
    # The four-Spark model the half stopped is that switch's rollback target.
    roles = {Path(path).name: role for path, role in checkpoints.roles(controller.STATE).items()}
    assert roles == {first.name: "active", tp4.name: "rollback"}
    assert install("--profile", GLM, "--on", "0,1") == 0
    second = Path(result(capsys)["deployment"]).resolve()
    roles = {Path(path).name: role for path, role in checkpoints.roles(controller.STATE).items()}
    assert roles == {second.name: "active", first.name: "rollback"}
    assert install("--profile", QWEN, "--on", "2,3") == 0
    third = Path(result(capsys)["deployment"]).resolve()
    roles = {Path(path).name: role for path, role in checkpoints.roles(controller.STATE).items()}
    assert roles == {second.name: "active", first.name: "rollback", third.name: "active"}
