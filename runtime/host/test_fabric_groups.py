"""Models on arcs of an eight-Spark ring: ``--on`` arcs, groups side by side, a group on every Spark and status.

``sparkring install`` runs against a simulated eight-Spark ring recorded with
its fabric document and relay table (``test_install_workflow``'s ``machine``
and ``sparks`` fixtures) and an image lock v3 whose image carries the SIRCL
layer and lists the installer profiles of eight Sparks. Each deployment's
start and stop is recorded by profile and placement and leaves the state a
completed operation leaves, so the switching rules see which models run.
"""
import json
from pathlib import Path

import pytest

from runtime.common import fabric_document, fabric_layout, installer, installer_image
from runtime.common.test_image_lock import sircl_block, sircl_lock
from runtime.host import controller, fabric, install_workflow as flow, node, placement, relays, topology
from runtime.host.test_fabric_layouts import configured, sparks as layout_sparks
from runtime.host.test_hairpin_ring import document as hairpin_document
from runtime.host.test_install_workflow import machine, sparks  # noqa: F401  (pytest fixtures)
from scripts import sparkring

QWEN_TP2 = "qwen38-flash-next-tp2"
GLM_TP2 = "glm53-flash-nvfp4-spark-tp2"
QWEN_TP4 = "qwen38-flash-next-qad-tp4"
DEEPSEEK_TP4 = "deepseek-v41-flash-tp4"
GLM_TP8 = "glm53-flash-csf-tp8"
GLM_FULL = "glm53-nvfp4-tp8"
MARKER = {"binary": relays.MARKER_BINARY, "sha256": "ab" * 32}
CSF_BUILD = "sparkring-kraken-beta-20261007-bc9ea774"


def ring_cluster(size=8):
    """A recorded cluster of ``size`` Sparks cabled as one cycle, every relaying Spark with its hairpin kept."""
    layout = fabric_layout.layout("cycle", size)
    found = layout_sparks(layout)
    blank = topology.build_spec(found, found[0]["node_id"], name="test")
    nodes = configured(blank)
    plan = topology.build_spec(nodes, nodes[0]["node_id"], name="test")
    for rank, current in enumerate(plan["nodes"]):
        current["hairpin"] = hairpin_document(plan, rank, revision=current.get("revision"))
    return {"name": "test", "plan": plan}


def label(path):
    lock = installer.read(Path(path) / "deployment.lock.json")
    where = placement.from_lock(lock)
    return lock["selection"]["profile"] + ("@" + "-".join(map(str, where)) if where else "")


def image_lock_file(tmp_path, *, pins=("lil-image-aba309e4610c", "sparkring-kraken-beta-20261007-bc9ea774"),
                    plugins=True):
    """A v3 lock listing every installer profile, the eight-Spark ones included; its image's vLLM matches the
    pinned builds ``pins``, by default the image's and the build that reads the CSF checkpoint, and, with
    ``plugins``, the image carries the GLM-5.3 plugin layer (runtime/images/derive_glm53_plugins.py)."""
    from runtime.images import derive_glm53_plugins
    profiles = sorted({*installer_image.default_lock()["profiles"], *installer_image.SIRCL_ONLY})
    path = tmp_path / ("sircl-image-" + "-".join(pins) + ("-plugins" if plugins else "") + ".json")
    added = {"vllm_plugins": dict(derive_glm53_plugins.PLUGINS)} if plugins else {}
    path.write_text(json.dumps(sircl_lock(profiles=profiles, sircl=dict(sircl_block(), vllm_pins=sorted(pins)),
                                          **added)))
    return path


class Ring:
    """The simulated ring's record of model operations."""

    def __init__(self, events, lock):
        self.events = events
        self.lock = lock
        self.fail = None


@pytest.fixture
def ring8(machine, sparks, monkeypatch, tmp_path):  # noqa: F811 (fixture argument)
    events, previous, _, _ = machine
    value = ring_cluster()
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    monkeypatch.setattr(flow, "require_head", lambda *_: value["plan"]["nodes"][0]["node_id"])
    (controller.STATE / "active.json").unlink()
    previous.rmdir()
    document, _ = fabric.prepare(value["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(document))
    simulated = Ring(events, image_lock_file(tmp_path))

    def operation(path, action, **kwargs):
        name = label(path)
        events.append(f"{name}:{action}")
        if simulated.fail == (name, action):
            raise ValueError("injected failure")
        if action in ("up", "down"):
            state = installer.read(Path(path) / "state.json") if (Path(path) / "state.json").exists() else {}
            node.save(path, "state.json", {"generation": state.get("generation", 0) + 1, "operation": action,
                                           "complete": True})
        if action == "transport":
            return {"verdict": "as-expected", "nccl_observed": "absent", "problems": []}
        return {"verified": True}
    monkeypatch.setattr(flow.retained_source, "apply", operation)
    monkeypatch.setattr(flow, "park_ring", lambda cluster, **kwargs: events.append("park-ring"))
    monkeypatch.setattr(flow, "check_workloads", lambda *a, **k: None)

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


def install(ring, *extra, lock=None):
    return sparkring.main(["install", "--yes", "--json", "--image-lock", str(lock or ring.lock), *extra])


def result(capsys):
    return json.loads(capsys.readouterr().out)


def ops(ring):
    """The model operations since the last call, as ``profile[@positions]:operation``."""
    found = [event for event in ring.events if isinstance(event, str) and ":" in event
             and not event.startswith(("prepare:", "check-workloads"))]
    ring.events.clear()
    return [event for event in found if not event.endswith(":transport")]


def lock_of(outcome):
    return installer.read(Path(outcome["deployment"]) / "deployment.lock.json")


def recorded():
    return {slot: label(path) for slot, path in placement.actives(controller.STATE).items()}


def test_two_four_spark_groups_serve_side_by_side_on_their_own_arcs(ring8, capsys):
    assert install(ring8, "--profile", QWEN_TP4, "--on", "0-3") == 0
    first = result(capsys)
    assert first["placement"] == [0, 1, 2, 3] and first["stops"] == []
    assert first["group"] == {"shape": "path-4", "positions": [0, 1, 2, 3], "api_position": 0}
    assert install(ring8, "--profile", DEEPSEEK_TP4, "--on", "4-7") == 0
    out = capsys.readouterr()
    second = json.loads(out.out)
    assert second["group"] == {"shape": "path-4", "positions": [4, 5, 6, 7], "api_position": 4}
    # The second group stops nothing: the two arcs share no Spark.
    assert second["stops"] == [] and second["replaces"] is None
    assert "Placement: path-4 at positions 4, 5, 6, 7; its API is on Spark 4." in out.err
    assert recorded() == {(0, 1, 2, 3): QWEN_TP4 + "@0-1-2-3", (4, 5, 6, 7): DEEPSEEK_TP4 + "@4-5-6-7"}
    # Each group's checkpoint moves between its own Sparks.
    assert ("view", (0, 1, 2, 3)) in ring8.events and ("view", (4, 5, 6, 7)) in ring8.events
    assert ops(ring8) == [f"{QWEN_TP4}@0-1-2-3:up", f"{QWEN_TP4}@0-1-2-3:verify",
                          f"{DEEPSEEK_TP4}@4-5-6-7:up", f"{DEEPSEEK_TP4}@4-5-6-7:verify"]
    lock = lock_of(second)
    hosts = installer.read(controller.STATE / "cluster.json")["plan"]["spec"]["hosts"]
    # The group's ranks bootstrap over their management addresses and check the fabric's relay table.
    assert [row["host"] for row in lock["site"]["ranks"]] == [hosts[position]["host"] for position in (4, 5, 6, 7)]
    assert [row["host_ip"] for row in lock["site"]["ranks"]] == [hosts[position]["management_address"]
                                                                  for position in (4, 5, 6, 7)]
    assert all(relays.is_reference(row["fabric"]) for row in lock["site"]["ranks"])
    assert lock["transport"]["group"]["positions"] == [4, 5, 6, 7] and lock["transport"]["group"]["name"] == "path-4"
    phases = [phase["id"] for phase in installer.operation_plan(lock, "up")["phases"]]
    assert phases.index("ring-stop") + 1 == phases.index("ring-serve") < phases.index("start-api")
    # Its checkpoint moves along the line of its own Sparks, whose ends share no cable.
    assert installer.read(Path(second["deployment"]) / flow.PLAN_FILE)["line"] is True
    assert second["commands"]["stop"] == "sudo sparkring down --on 4-7 --execute"
    assert f"--profile {DEEPSEEK_TP4} --on 4-7 --image-lock " in second["checkpoint"]["command"]


def test_four_pairs_and_a_group_of_four_beside_two_pairs(ring8, capsys):
    for arc, profile in (("0,1", QWEN_TP2), ("2,3", GLM_TP2), ("4,5", QWEN_TP2), ("6,7", GLM_TP2)):
        assert install(ring8, "--profile", profile, "--on", arc) == 0
        assert result(capsys)["stops"] == []
    assert sorted(recorded()) == [(0, 1), (2, 3), (4, 5), (6, 7)]
    lock = lock_of({"deployment": str(placement.recorded(controller.STATE, (4, 5)))})
    hosts = installer.read(controller.STATE / "cluster.json")["plan"]["spec"]["hosts"]
    # A pair serves on the cable it shares: the first Spark's port 0 faces the second's port 1.
    assert [row["host_ip"] for row in lock["site"]["ranks"]] == [
        next(port["address"] for port in hosts[4]["data_interfaces"] if port["role"] == "cw_primary").split("/")[0],
        next(port["address"] for port in hosts[5]["data_interfaces"] if port["role"] == "ccw_primary").split("/")[0]]
    assert lock["transport"]["group"]["name"] == "pair"
    ops(ring8)
    # A group of four on positions 0-3 stops the two pairs it overlaps and leaves the others serving.
    assert install(ring8, "--profile", QWEN_TP4, "--on", "0-3") == 0
    replaced = result(capsys)
    assert sorted(tuple(row["placement"]) for row in replaced["stops"]) == [(0, 1), (2, 3)]
    assert ops(ring8) == [f"{QWEN_TP2}@0-1:down", f"{GLM_TP2}@2-3:down", f"{QWEN_TP4}@0-1-2-3:up",
                          f"{QWEN_TP4}@0-1-2-3:verify"]
    assert recorded() == {(0, 1, 2, 3): QWEN_TP4 + "@0-1-2-3", (4, 5): QWEN_TP2 + "@4-5", (6, 7): GLM_TP2 + "@6-7"}


def test_a_group_across_the_cable_to_node_a_serves_its_api_on_its_first_spark(ring8, capsys, monkeypatch):
    value = installer.read(controller.STATE / "cluster.json")
    value["api_address"] = "198.51.100.10"
    # The installation reads each Spark's inventory again; Spark 6's default route leaves on its LAN port.
    lan = next(row for row in value["plan"]["nodes"][6]["facts"]["interfaces"] if row["name"] == "enP7s7")
    lan["ipv4"] = ["198.51.100.16/24"]
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    assert install(ring8, "--profile", QWEN_TP4, "--on", "6-1") == 0
    out = capsys.readouterr()
    wrapped = json.loads(out.out)
    assert wrapped["placement"] == [6, 7, 0, 1] and wrapped["group"]["api_position"] == 6
    assert wrapped["api_url"] == "http://198.51.100.16:8015/v1"
    assert "Install qwen38-flash-next-qad-tp4 on Sparks 6, 7, 0 and 1" in out.err
    assert lock_of(wrapped)["transport"]["group"]["positions"] == [6, 7, 0, 1]
    # The same arc named by its positions is the same deployment.
    assert install(ring8, "--profile", QWEN_TP4, "--on", "6,7,0,1") == 0
    assert result(capsys)["deployment"] == wrapped["deployment"]


def test_a_four_spark_profile_without_on_takes_the_one_free_half_of_the_ring(ring8, capsys):
    assert install(ring8, "--profile", QWEN_TP4) == 3
    refused = result(capsys)
    assert refused["field"] == "placement" and "Choose a half with --on 0-3 or --on 4-7" in refused["message"]
    assert refused["details"]["lines"] == ["--on 0-3: Sparks 0-3, free", "--on 4-7: Sparks 4-7, free"]
    assert install(ring8, "--profile", QWEN_TP4, "--on", "0-3") == 0
    capsys.readouterr()
    assert install(ring8, "--profile", DEEPSEEK_TP4) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["placement"] == [4, 5, 6, 7]
    assert (f"{DEEPSEEK_TP4} uses four Sparks: it goes on Sparks 4-7 (--on 4-7), the half that serves no model."
            in out.err)


def test_a_model_on_every_spark_stops_every_group_and_runs_tensor_parallel_eight(ring8, capsys):
    assert install(ring8, "--profile", QWEN_TP4, "--on", "0-3") == 0
    assert install(ring8, "--profile", DEEPSEEK_TP4, "--on", "4-7") == 0
    capsys.readouterr()
    ops(ring8)
    assert install(ring8, "--profile", GLM_TP8) == 0
    whole = result(capsys)
    assert "placement" not in whole and whole["group"] == {"shape": "cycle-8", "positions": list(range(8)),
                                                          "api_position": 0}
    assert sorted(row["profile"] for row in whole["stops"]) == [DEEPSEEK_TP4, QWEN_TP4]
    assert ops(ring8) == [f"{QWEN_TP4}@0-1-2-3:down", f"{DEEPSEEK_TP4}@4-5-6-7:down", f"{GLM_TP8}:up",
                          f"{GLM_TP8}:verify"]
    assert recorded() == {None: GLM_TP8}
    lock = lock_of(whole)
    assert lock["transport"]["group"]["name"] == "cycle-8" and lock["transport"]["nccl"] == "never"
    specs = installer.specifications(lock)
    assert len(specs) == 8 and {spec.environment["SIRCL_GROUPS"] for spec in specs} == {"tp"}
    assert all(spec.environment["SIRCL_RANK_POSITIONS"] == "0,1,2,3,4,5,6,7" for spec in specs)
    command = list(specs[0].command)
    assert command[command.index("--tensor-parallel-size") + 1] == "8"


def test_glm53_at_tp8_gives_each_decode_context_parallel_group_its_own_session(ring8, capsys):
    assert install(ring8, "--profile", GLM_FULL) == 0
    lock = lock_of(result(capsys))
    specs = installer.specifications(lock)
    assert {spec.environment["SIRCL_GROUPS"] for spec in specs} == {"tp,dcp"}
    command = list(specs[0].command)
    assert command[command.index("--decode-context-parallel-size") + 1] == "4"
    assert command[command.index("--quantization") + 1] == "modelopt_fp4"
    assert {spec.environment["SIRCL_FUSED_NORM"] for spec in specs} == {"1"}


def test_glm53_at_tp8_needs_an_image_that_carries_its_vllm_plugins(ring8, capsys, tmp_path):
    plain = image_lock_file(tmp_path, plugins=False)
    assert install(ring8, "--profile", GLM_FULL, lock=plain) == 3
    refused = result(capsys)
    assert "glm53-nvfp4-tp8 loads the vLLM plugins glm_dsa_indexer_split, glm53full_speedups" in refused["message"]
    assert ops(ring8) == []


def test_the_csf_checkpoint_needs_an_image_whose_vllm_reads_it(ring8, capsys, tmp_path):
    plain = image_lock_file(tmp_path, pins=("lil-image-aba309e4610c",))
    assert install(ring8, "--profile", GLM_TP8, lock=plain) == 3
    refused = result(capsys)
    assert CSF_BUILD in refused["message"]
    assert ops(ring8) == []
    assert install(ring8, "--profile", GLM_TP8) == 0
    lock = lock_of(result(capsys))
    assert lock["selection"]["model_repository"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"
    command = list(installer.specifications(lock, only_rank=0)[0].command)
    assert command[command.index("--quantization") + 1] == "nvfp4_csf"
    assert command[command.index("--load-format") + 1] == "nvfp4_csf"
    assert command[command.index("--served-model-name") + 1] == "GLM-5.3-Flash-CSF-TP8"


def test_the_glm_pair_installs_the_csf_checkpoint_without_a_name_on_an_image_that_reads_it(ring8, capsys, tmp_path):
    assert install(ring8, "--profile", GLM_TP2, "--on", "0,1") == 0
    preferred = result(capsys)
    lock = lock_of(preferred)
    assert lock["selection"]["target_variant"] == "csf"
    assert lock["selection"]["model_repository"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"
    command = list(installer.specifications(lock, only_rank=0)[0].command)
    assert command[command.index("--quantization") + 1] == "nvfp4_csf"
    assert command[command.index("--served-model-name") + 1] == "GLM-5.3-Flash-CSF-TP2"
    # The saved plan repeats the selection by name, and naming it requests the same deployment.
    assert "--checkpoint csf" in preferred["checkpoint"]["command"]
    assert install(ring8, "--profile", GLM_TP2, "--on", "0,1", "--checkpoint", "csf") == 0
    assert result(capsys)["deployment"] == preferred["deployment"]
    # NVFP4-Spark stays selectable on the same image, as its own deployment.
    assert install(ring8, "--profile", GLM_TP2, "--on", "0,1", "--checkpoint", "nvfp4-spark") == 0
    spark = result(capsys)
    assert spark["deployment"] != preferred["deployment"]
    assert lock_of(spark)["selection"]["model_repository"] == "local-inference-lab/GLM-5.3-Flash-NVFP4-Spark"
    command = list(installer.specifications(lock_of(spark), only_rank=0)[0].command)
    assert command[command.index("--quantization") + 1] == "modelopt_mixed"
    # An image whose vLLM cannot read it installs NVFP4-Spark without a name, the same request as naming
    # nvfp4-spark there, and refuses the name csf before any Spark changes.
    plain = image_lock_file(tmp_path, pins=("lil-image-aba309e4610c",))
    assert install(ring8, "--profile", GLM_TP2, "--on", "2,3", lock=plain) == 0
    other = result(capsys)
    assert lock_of(other)["selection"]["target_variant"] == "nvfp4-spark"
    assert "--checkpoint" not in other["checkpoint"]["command"]
    assert install(ring8, "--profile", GLM_TP2, "--on", "2,3", "--checkpoint", "nvfp4-spark", lock=plain) == 0
    assert result(capsys)["deployment"] == other["deployment"]
    ops(ring8)
    assert install(ring8, "--profile", GLM_TP2, "--on", "2,3", "--checkpoint", "csf", lock=plain) == 3
    refused = result(capsys)
    assert refused["field"] == "checkpoint_name" and CSF_BUILD in refused["message"]
    assert ops(ring8) == []


def test_an_arc_whose_relaying_spark_lacks_the_hairpin_setting_is_refused(ring8, capsys, monkeypatch):
    value = installer.read(controller.STATE / "cluster.json")
    plan = value["plan"]
    plan["nodes"][5]["hairpin"] = hairpin_document(plan, 5, approved=False, armed=False,
                                                   revision=plan["nodes"][5].get("revision"))
    node.save(controller.STATE, "cluster.json", value)
    monkeypatch.setattr(controller, "collect", lambda _: value["plan"]["nodes"])
    assert install(ring8, "--profile", QWEN_TP2, "--on", "4,5") == 0
    capsys.readouterr()
    assert install(ring8, "--profile", QWEN_TP4, "--on", "4-7") == 3
    refused = result(capsys)
    assert refused["field"] == "hairpin" and refused["message"].startswith("Spark 5 (")
    assert "relay the lanes of Sparks 4-7" in refused["message"]


@pytest.mark.parametrize("args, field, message", [
    (["--profile", QWEN_TP4, "--on", "0-5"], "placement", f"{QWEN_TP4} serves four Sparks; Sparks 0-5 are six"),
    (["--profile", QWEN_TP4, "--on", "0,2,3,4"], "placement", "--on 0,2,3,4 names no consecutive Sparks of this "
                                                             "cycle-8"),
    (["--profile", QWEN_TP2, "--on", "3-2"], "placement", "--on 3-2 names every Spark from position 3 on"),
])
def test_an_arc_that_is_not_the_profiles_group_is_refused_before_any_change(ring8, capsys, args, field, message):
    assert install(ring8, *args) == 3
    refused = result(capsys)
    assert refused["field"] == field and refused["message"].startswith(message)
    assert ops(ring8) == []


def test_without_sircl_an_arc_of_a_larger_ring_is_refused(ring8, capsys):
    assert sparkring.main(["install", "--yes", "--json", "--profile", QWEN_TP4, "--on", "4-7"]) == 3
    refused = result(capsys)
    assert refused["field"] == "placement"
    assert refused["message"].startswith("The prepared transport runs a pair, a four-Spark ring and the ring's halves; "
                                         "Sparks 4-7 of this cycle-8 need SIRCL ring sessions")
    assert "carries no SIRCL layer" in refused["message"]


def test_a_profile_of_more_sparks_than_the_fabric_names_the_profiles_that_fit(machine, capsys):  # noqa: F811
    assert sparkring.main(["install", "--yes", "--json", "--profile", GLM_TP8]) == 3
    refused = result(capsys)
    assert refused["field"] == "placement"
    assert refused["message"].startswith(f"{GLM_TP8} serves eight Sparks and this pair has two. Installer profiles "
                                         "that fit it: ")
    fit = refused["message"].split("that fit it: ", 1)[1]
    assert QWEN_TP2 in fit and QWEN_TP4 not in fit and GLM_TP8 not in fit


# sparkring up, status and down of groups on the ring, each deployment's real
# operation plan run against simulated Sparks.

def up_ring(tmp_path, monkeypatch):
    """A recorded eight-Spark ring whose Sparks the real operation plans of sparkring up run against."""
    from runtime.common import distribution
    from runtime.host import discovery
    from runtime.host.test_ring_halves import RingSparks
    from scripts import installer_runner
    monkeypatch.setenv("SPARKRING_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(controller, "STATE", tmp_path / "state")
    value = ring_cluster()
    node.save(controller.STATE, "cluster.json", value)
    document, _ = fabric.prepare(value["plan"], cluster="test", marker=MARKER)
    (controller.STATE / "fabric.json").write_text(fabric_document.encoded(document))
    monkeypatch.setattr(distribution, "identity", lambda _: "a" * 40)
    monkeypatch.setattr(distribution, "bundle", lambda root, dest: dest.write_bytes(b"retained source"))
    simulated = RingSparks([host["host"] for host in value["plan"]["spec"]["hosts"]])
    monkeypatch.setattr(discovery, "ssh", simulated.ssh)
    monkeypatch.setattr(installer_runner.Runner, "_call", lambda runner, target, argv, timeout:
                        simulated.rank_operation(runner, target, argv, timeout))
    # Up checks the hairpin setting of the Sparks that relay each group's lanes.
    simulated.hairpin = []
    monkeypatch.setattr(controller, "_hairpin_problem", lambda placement=None: simulated.hairpin.append(placement))
    simulated.lock = image_lock_file(tmp_path)
    return simulated


@pytest.fixture
def groups(tmp_path, monkeypatch):
    """Qwen TP4 on positions 0-3 and DeepSeek TP4 on 4-7, each started by sparkring up --on."""
    simulated = up_ring(tmp_path, monkeypatch)
    for profile, arc in ((QWEN_TP4, "0-3"), (DEEPSEEK_TP4, "4-7")):
        assert controller.lifecycle(["up", profile, "--on", arc, "--image-lock", str(simulated.lock), "--execute"]) == 0
    return simulated


def test_up_creates_the_glm_pair_with_the_checkpoint_its_image_reads(tmp_path, monkeypatch):
    simulated = up_ring(tmp_path, monkeypatch)
    plain = image_lock_file(tmp_path, pins=("lil-image-aba309e4610c",))
    expected = {(0, 1): (simulated.lock, "csf", "GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD"),
                (2, 3): (plain, "nvfp4-spark", "GLM-5.3-Flash-NVFP4-Spark")}
    for arc, (lock, variant, name) in expected.items():
        on = ",".join(map(str, arc))
        assert controller.lifecycle(["up", GLM_TP2, "--on", on, "--image-lock", str(lock), "--plan"]) == 0
        directory = controller.deployment_directory(GLM_TP2, placement.instance_label(arc))
        value = installer.read(directory / "deployment.lock.json")
        assert value["selection"]["target_variant"] == variant
        assert value["selection"]["model_repository"] == "local-inference-lab/" + name
        # Every rank uses the cluster's checkpoint directory of that checkpoint's revision.
        assert {row["model"] for row in value["site"]["ranks"]} == {
            installer.checkpoint_directory(installer.read(controller.STATE / "cluster.json"), value["selection"])}


def test_status_prints_one_block_per_group_with_its_own_api(groups, capsys):
    assert groups.hairpin == [(0, 1, 2, 3), (4, 5, 6, 7)]
    capsys.readouterr()
    assert controller.lifecycle(["status"]) == 0
    out = capsys.readouterr().out
    first, second = out.index("Sparks 0-3:"), out.index("Sparks 4-7:")
    assert first < out.index("Group: path-4 at positions 0, 1, 2, 3; API on Spark 0") < second
    assert out.index("Group: path-4 at positions 4, 5, 6, 7; API on Spark 4") > second
    assert controller.lifecycle(["status", "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert [(row["placement"], row["group"]["shape"], row["group"]["api_position"]) for row in document["slots"]] == [
        ([0, 1, 2, 3], "path-4", 0), ([4, 5, 6, 7], "path-4", 4)]
    assert controller.lifecycle(["status", "--on", "4-7", "--json"]) == 0
    assert [row["placement"] for row in json.loads(capsys.readouterr().out)["slots"]] == [[4, 5, 6, 7]]


def test_down_names_each_group_and_stops_only_the_one_named(groups, capsys):
    with pytest.raises(ValueError, match="more than one placement. Name one with --on 0-3, --on 4-7") as refused:
        controller.lifecycle(["down", "--execute"])
    assert refused.value.details["lines"] == [f"Sparks 0-3: {QWEN_TP4} (started); name it with --on 0-3",
                                              f"Sparks 4-7: {DEEPSEEK_TP4} (started); name it with --on 4-7"]
    groups.operations.clear()
    assert controller.lifecycle(["down", "--on", "4-7", "--execute"]) == 0
    assert {host for name, host in groups.operations if name == "stop"} == set(groups.hosts[4:])
    with pytest.raises(ValueError, match=f"{QWEN_TP4} runs on Sparks 0-3. Stop it first: "
                                         "sudo sparkring down --on 0-3 --execute"):
        controller.lifecycle(["up", QWEN_TP2, "--on", "2,3", "--image-lock", str(groups.lock), "--execute"])


def test_a_group_with_the_end_of_a_line_checks_only_that_sparks_cabled_functions():
    from runtime.common import image_lock, transport
    layout = fabric_layout.layout("path", 5)
    found = layout_sparks(layout)
    blank = topology.build_spec(found, found[0]["node_id"], name="test")
    plan = topology.build_spec(configured(blank), found[0]["node_id"], name="test")
    cluster = {"name": "test", "plan": plan}
    document, _ = fabric.prepare(plan, cluster="test", marker=MARKER)
    reference = {"site_path": fabric_document.HOST_PATH, "site_sha256": "1" * 64, "plan_sha256": "2" * 64}
    site = controller.model_site(cluster, QWEN_TP4, "iline", (0, 1, 2, 3), fabric=reference)
    # Node A ends the line: only its port 0 is cabled.
    assert site["hosts"][0]["hcas"] == ["rocep1s0f0", "roceP2p1s0f0"]
    assert all("hcas" not in row for row in site["hosts"][1:])
    for row in site["hosts"]:
        row["model"] = "/srv/sparkring/test/checkpoints/model"
    image = sircl_lock()
    section = transport.section(image, document, [0, 1, 2, 3], nccl="never", tuning=transport.load_tuning())
    lock = installer.make_lock(QWEN_TP4, site, "1" * 40, "2" * 64, image_runtime=image_lock.v2_view(image),
                               transport=section)
    assert [len(row["hcas"]) for row in lock["site"]["ranks"]] == [2, 4, 4, 4]
    assert section["group"]["name"] == "path-4" and section["devices"][0] == ["rocep1s0f0", "roceP2p1s0f0"]
    # The container's prepared-transport device variables keep the four functions; SIRCL names the devices.
    specs = installer.specifications(lock)
    assert specs[0].environment["B12X_ROCE_HCA"] == ",".join(installer.RING_HCAS)
