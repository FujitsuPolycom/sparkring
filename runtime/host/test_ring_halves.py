"""Two models on one four-Spark ring: slots, switching rules, mesh mode and endpoints.

The public ``sparkring install`` command runs against a simulated four-Spark
ring (``test_install_workflow``'s ``machine`` and ``sparks`` fixtures). Each
deployment's start and stop is recorded by profile and placement and leaves
the state a completed operation leaves, so the switching rules see which
models run.
"""
import json
from pathlib import Path

import pytest

from runtime.common import installer
from runtime.host import controller, install_workflow as flow, node, placement, rollout
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
    return {slot: (label(path) if path else None) for slot, path in
            ((slot, placement.recorded(controller.STATE, slot)) for slot in placement.slots(4))}


def test_a_ring_switches_from_one_model_to_two_and_back(ring, capsys):
    assert install("--profile", TP4) == 0
    tp4 = result(capsys)
    assert tp4["nodes"] == 4 and tp4["stops"] == [] and "placement" not in tp4
    ops(ring)

    # Half (0, 1): the running four-Spark model stops, the ring's mesh parks, the half starts.
    assert install("--profile", QWEN, "--on", "0,1") == 0
    first = result(capsys)
    assert first["placement"] == [0, 1] and first["nodes"] == 2 and first["replaces"] is None
    assert [(row["profile"], row["placement"]) for row in first["stops"]] == [(TP4, None)]
    assert ops(ring) == [f"{TP4}:down", "park-ring", f"{QWEN}@01:up", f"{QWEN}@01:verify"]
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
    assert ops(ring) == ["park-ring", f"{GLM}@23:up", f"{GLM}@23:verify"]
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
    assert ops(ring) == [f"{QWEN}@01:down", f"{GLM}@23:down", f"{TP4}:up", f"{TP4}:verify"]
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
    assert ops(ring) == [f"{QWEN}@01:down", "park-ring", f"{GLM}@01:up", f"{GLM}@01:verify"]
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
    assert ops(ring) == [f"{TP4}:down", "park-ring", f"{QWEN}@23:up", f"{QWEN}@23:verify", f"{QWEN}@23:down",
                         f"{TP4}:down", f"{TP4}:up", f"{TP4}:verify"]
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
    assert refused["details"]["lines"] == ["--on 0,1: Sparks 0 and 1, free", "--on 2,3: Sparks 2 and 3, free"]
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
