"""The Install Builder's data, page and engine (docs/operations/compose-builder.md)."""
import json
import os
import shutil
import subprocess

import pytest

from runtime.common import installer_image
from scripts import generate_compose_builder
from scripts.compose_builder import export, verify

NODE = shutil.which("node")
# Random valid sites per profile checkpoint; CI's compose-builder job raises it.
CASES = int(os.environ.get("SPARKRING_COMPOSE_BUILDER_CASES", "3"))
COMMIT = "0" * 40


@pytest.fixture(scope="module")
def data():
    return export.export(commit=COMMIT)


@pytest.fixture
def node():
    if NODE is None:
        if os.environ.get("SPARKRING_REQUIRE_NODE") == "1":
            pytest.fail("Node.js is required by this CI job")
        pytest.skip("Node.js unavailable")
    return NODE


def test_data_lists_every_offered_profile_and_checkpoint(data):
    assert [profile["id"] for profile in data["profiles"]] == export.profile_ids()
    assert not export.EXCLUDED & {profile["id"] for profile in data["profiles"]}
    assert data["ref"] == COMMIT and data["tag"] is None
    for profile in data["profiles"]:
        assert profile["checkpoints"][0]["default"]
        assert sum(checkpoint["default"] for checkpoint in profile["checkpoints"]) == 1
        for checkpoint in profile["checkpoints"]:
            assert set(checkpoint["variants"]) == {"off", "on"}
            assert all(len(variant["ranks"]) == profile["nodes"] for variant in checkpoint["variants"].values())
    qwen = next(profile for profile in data["profiles"] if profile["id"] == "qwen38-flash-next-tp2")
    names = {checkpoint["name"]: checkpoint for checkpoint in qwen["checkpoints"]}
    assert names["qad-step5500-ple1000"]["aliases"] == ["qad-step-5500"]
    assert names["qad-step-4000"]["changes"] == ["VLLM_MXFP8_LM_HEAD=1", "speculative moe_backend: b12x"]
    assert names["qad-step5500-mxfp8-attention"]["derived"]
    assert {profile["id"]: profile["model_name"] for profile in data["profiles"]}["glm53-flash-nvfp4-spark-tp4"] == "GLM-5.3-Flash"


def test_profiles_and_checkpoints_carry_their_status_and_purpose(data):
    from runtime.common import profiles
    for profile in data["profiles"]:
        assert profile["status"] == profiles.load(profile["id"])[0]["status"] and profile["purpose"]
        for checkpoint in profile["checkpoints"]:
            assert checkpoint["status"] in export.STATUSES
            if checkpoint["default"]:
                # The default checkpoint is what the profile's own status describes.
                assert checkpoint["status"] == profile["status"] and checkpoint["evidence"] is None
            else:
                assert checkpoint["purpose"] and checkpoint["evidence"]
            assert checkpoint["name"] is None or checkpoint["purpose"]
    by_id = {profile["id"]: profile for profile in data["profiles"]}
    assert by_id["swift15-qwen38-flash-next-tp4"]["status"] == "research-only"
    for profile_id in (TP2, "qwen38-flash-next-qad-tp4"):
        named = {c["name"]: c for c in by_id[profile_id]["checkpoints"]}
        assert named["qad-step5500-ple1000"]["purpose"] == "Default QAD checkpoint"
        assert named["qad-step5500-mxfp8-attention"]["purpose"] == "MXFP8 attention, derived during install"
        assert (named["jmni-qad5500-hybrid"]["status"], named["jmni-qad5500-hybrid"]["purpose"]) == (
            "research-only", "Third-party hybrid by JMNI Labs, research-only")


def test_labels_refuse_a_missing_label_status_or_evidence(tmp_path, monkeypatch):
    (tmp_path / "profiles").mkdir()
    (tmp_path / "record.md").write_text("evidence")
    path = tmp_path / "profiles" / "labels.json"

    def write(checkpoints):
        path.write_text(json.dumps({"schema": "sparkring-profile-labels/v1",
                                    "profiles": {"p": {"purpose": "A model", "checkpoints": checkpoints}}}))

    write({"main": {"purpose": "Default"}, "other": {"purpose": "Other", "status": "implemented", "evidence": "record.md#x"}})
    assert export.labels(tmp_path)["profiles"]["p"]["purpose"] == "A model"
    for checkpoints in ({"other": {"purpose": "Other", "status": "tested", "evidence": "record.md"}},
                        {"other": {"purpose": "Other", "status": "implemented", "evidence": "missing.md"}},
                        {"other": {"purpose": "Other", "status": "implemented"}},
                        {"other": {"purpose": ""}}):
        write(checkpoints)
        with pytest.raises(ValueError, match="profiles/labels.json"):
            export.labels(tmp_path)
    # A profile the file does not label, or a checkpoint it omits, stops the export.
    monkeypatch.setattr(export, "labels", lambda: {"schema": "sparkring-profile-labels/v1", "profiles": {}})
    with pytest.raises(ValueError, match="must label mimo-v26-flash-mopd-tp2"):
        export.profile_data("mimo-v26-flash-mopd-tp2")


def test_features_record_the_command_options_the_source_defines(data, tmp_path):
    # This checkout's `sparkring install` defines --on.
    assert data["features"]["ring_halves"] is True
    assert data["features"] == export.features()
    assert all(data["features"][name] for name in ("api_port", "api_bind", "api_address"))
    host = tmp_path / "runtime" / "host"
    host.mkdir(parents=True)
    none = {"ring_halves": False, "cable_check": False, "api_address": False, "api_port": False, "api_bind": False}
    assert export.features(tmp_path) == none
    (host / "install_workflow.py").write_text('def main():\n    parser.add_argument("--on", metavar="RANKS")\n')
    # A help text that names an option does not define it.
    (host / "cabling.py").write_text('def main():\n    parser.add_argument("--json", help="unlike --bandwidth")\n')
    assert export.features(tmp_path) == {**none, "ring_halves": True}
    (host / "cabling.py").write_text('def main():\n    parser.add_argument("--bandwidth", action="store_true")\n')
    assert export.features(tmp_path) == {**none, "ring_halves": True, "cable_check": True}
    # The API endpoint's settings are keys of the source's SETTINGS; a value that names one is not.
    common = tmp_path / "runtime" / "common"
    common.mkdir(parents=True)
    (common / "serving.py").write_text('SETTINGS = {"api_port": ("--port", None, 1, 1024, "api_bind")}\n')
    assert export.features(tmp_path) == {**none, "ring_halves": True, "cable_check": True, "api_port": True}
    (host / "install_workflow.py").write_text('def main():\n    parser.add_argument("--api-address")\n')
    assert export.features(tmp_path)["api_address"] is True


def test_image_catalog_lists_the_default_first_with_options_install_accepts(data):
    rows = data["images"]
    assert rows[0]["default"] and rows[0]["file"] is None and sum(row["default"] for row in rows) == 1
    for row in rows:
        assert set(row["profiles"]) <= set(export.profile_ids())
        path = installer_image.lock_path(row["option"])
        assert path is None if row["default"] else path.parent.name == row["name"]
        assert row["default"] or row["file"] == f"images/{row['name']}.json"
    # The page offers the default image and the images a release published, each by its release tag.
    catalog = installer_image.catalog()
    released = {name for name in installer_image.release_tags().values()
                if set(installer_image.profiles_of(next(r for r in catalog if r["name"] == name)["lock"])) & set(export.profile_ids())}
    assert all(row["default"] or row["tags"] for row in rows) and {row["name"] for row in rows} >= released
    assert all(row["option"] in row["tags"] for row in rows if row["tags"])
    # A development image stays out of the page; --image still takes its short name.
    statusrows = "dev-20261001-statusrows-cuda1342-nccl2323-status034"
    assert statusrows not in {row["name"] for row in rows}
    assert export._image_option(next(r for r in catalog if r["name"] == statusrows), [r["name"] for r in catalog]) == "statusrows"


def test_save_cpu_is_offered_only_on_images_that_read_its_variable(data, node, monkeypatch):
    from runtime.common import serving
    older = "dev-20260928-plainstatus-cuda1342-nccl2323-status033"
    # The Builder and the installer read the same capability: the default image has the reader window.
    for profile in data["profiles"]:
        assert profile["image_capabilities"] == list(installer_image.capabilities(profile["image_release"])) == ["shm_reader_window"]
        assert next(r for r in profile["checkpoints"][0]["settings"] if r["name"] == "save_cpu")["needs"] == "shm_reader_window"
    monkeypatch.setattr(export, "profile_ids", lambda: [TP2])
    profile, = export.image_data(older)["profiles"]
    assert profile["image_capabilities"] == [] and all(set(c["variants"]) == {"off"} for c in profile["checkpoints"])
    found = run_engine(node, """
const { profile } = value, cp = E.checkpointOf(profile, null);
console.log(JSON.stringify({ check: E.servingCheck(profile, { save_cpu: true }, null), offered: E.offered(profile, cp.settings.find(r => r.switch)),
  read: E.readSelection(profile, cp, { save_cpu: true, max_images: 1 }, { mode: 'auto' }) }));""", {"profile": profile})
    with pytest.raises(ValueError) as refused:
        serving.check_image({"save_cpu": True}, older, installer_image.capabilities(older))
    # The engine refuses the switch with the installer's message, and the page's fields leave it out.
    assert found["check"] == {"ok": False, "error": str(refused.value)}
    assert found["offered"] is False and found["read"]["settings"] == {"max_images": 1}


def test_image_data_renders_profiles_on_that_image(monkeypatch, data):
    monkeypatch.setattr(export, "profile_ids", lambda: ["mimo-v26-flash-mopd-tp2"])
    row = next(row for row in data["images"] if "2026.09.5" in row["tags"])
    document = export.image_data(row["name"])
    profile, = document["profiles"]
    default = next(p for p in data["profiles"] if p["id"] == "mimo-v26-flash-mopd-tp2")
    assert document["image"] == row["name"] and profile["image_option"] == "2026.09.5"
    assert profile["image_release"] == row["name"] and profile["image_id"] != default["image_id"]
    assert default["image_option"] is None


def run_engine(node, script, value):
    """What ``script`` prints with ``E``, the engine, and ``value``, a JSON document read from stdin."""
    prelude = f"const E = require({json.dumps(str(export.HERE / 'engine.js'))});\n" \
              "const value = JSON.parse(require('fs').readFileSync(0, 'utf8'));\n"
    result = subprocess.run([node, "-e", prelude + script], input=json.dumps(value),
                            capture_output=True, text=True, encoding="utf-8", check=True)
    return json.loads(result.stdout)


def test_commands_select_a_non_default_image(data, node):
    profile = {**next(p for p in data["profiles"] if p["id"] == "mimo-v26-flash-mopd-tp2"), "image_option": "2026.09.5"}
    meta = {key: data[key] for key in ("repository", "tag", "commit", "ref", "since_tag", "commits_since")}
    install, render = run_engine(node, """
const { profile, meta } = value, checkpoint = E.checkpointOf(profile, null);
console.log(JSON.stringify([
  E.installCommand(profile, checkpoint, {}, meta, { form: 'installed', approval: 'ask' }),
  E.renderCommand(profile, checkpoint, {}, { name: 'ring' }),
]));""", {"profile": profile, "meta": meta})
    assert install == "sudo sparkring install --profile mimo-v26-flash-mopd-tp2 --image 2026.09.5"
    assert render == "sparkring compose render mimo-v26-flash-mopd-tp2 --site ring.site.yaml --output ring --image 2026.09.5"


def meta_of(data):
    return {key: data[key] for key in ("repository", "tag", "commit", "ref", "since_tag", "commits_since")}


TP2, TP4, GLM2, MIMO2 = "qwen38-flash-next-tp2", "glm53-flash-nvfp4-spark-tp4", "glm53-flash-nvfp4-spark-tp2", "mimo-v26-flash-mopd-tp2"
SCRIPT = f"curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/{COMMIT}/install.sh | bash -s -- --ref {COMMIT} "
MY_SPARKS = [{"host": "spark-a", "host_ip": "198.51.100.10"}, {"host": "spark-b", "host_ip": "198.51.100.11"},
             {"host": "spark-c", "host_ip": "198.51.100.12"}, {"host": "spark-d", "host_ip": ""}]


def packs(node, data, cases):
    """The command pack of each case: {layout, features, main, halves, opts}, a selection being
    {profile, checkpoint, settings} with a profile ID."""
    used = {selection["profile"] for case in cases for selection in [case["main"], *case.get("halves", [])]}
    # A selection's image_option stands for its profile rendered on that installer image.
    script = """
const byId = Object.fromEntries(value.profiles.map(p => [p.id, p]));
const sel = s => ({ profile: { ...byId[s.profile], image_option: s.image_option || null }, checkpoint: s.checkpoint || null,
  settings: s.settings || {}, endpoint: s.endpoint || null, problems: s.problems || [] });
console.log(JSON.stringify(value.cases.map(c => E.commandPack({ layout: c.layout, features: c.features, main: sel(c.main),
  halves: (c.halves || [c.main, c.main]).map(sel) }, value.meta, c.opts))));"""
    return run_engine(node, script, {"profiles": [p for p in data["profiles"] if p["id"] in used], "cases": cases,
                                     "meta": meta_of(data)})


def commands(pack, key=None):
    return [c["command"] for group in pack["groups"] if key in (None, group["key"]) for c in group["commands"]]


def endpoints(pack, key=None):
    return [c["endpoint"] for group in pack["groups"] if key in (None, group["key"]) for c in group["commands"]]


NEW = {"form": "script", "pin": True, "approval": "ask", "downloadLimit": "", "order": "auto"}
BOTH = {"ring_halves": True, "cable_check": False}


def test_layouts_follow_the_source_features(node):
    found = run_engine(node, "console.log(JSON.stringify([E.layouts(value[0]), E.layouts(value[1]), E.layouts(null), E.LAYOUT_SPARKS]))",
                       [{"ring_halves": True}, {"ring_halves": False}])
    assert found == [["pair", "ring", "halves"], ["pair", "ring"], ["pair", "ring"], {"pair": 2, "ring": 4, "halves": 4}]


def test_install_command_places_a_two_spark_profile_on_a_half(data, node):
    profile = next(p for p in data["profiles"] if p["id"] == TP2)
    install = run_engine(node, """
const { profile, meta } = value;
console.log(JSON.stringify(E.installCommand(profile, E.checkpointOf(profile, 'qad-step-4000'), {}, meta,
  { form: 'installed', approval: 'yes', on: '2,3' })));""", {"profile": profile, "meta": meta_of(data)})
    assert install == f"sudo sparkring install --profile {TP2} --on 2,3 --checkpoint qad-step-4000 --yes"


def test_pack_for_a_pair_installs_and_checks(data, node):
    plain, measured = packs(node, data, [
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2}, "opts": NEW},
        {"layout": "pair", "features": {**BOTH, "cable_check": True}, "main": {"profile": TP2}, "opts": NEW},
    ])
    assert [g["key"] for g in plain["groups"]] == ["install", "check"]
    install, = plain["groups"][0]["commands"]
    assert install["command"] == SCRIPT + f"--profile {TP2}"
    assert install["where"] == "On Node A, the Spark connected to your network, as a user with sudo"
    assert install["what"] == "Installs SparkRing and starts Qwen3.8-Flash-Next on both Sparks."
    assert install["endpoint"] == "http://NODE_A:8000/v1"
    assert commands(plain, "check") == ["sudo sparkring status"]
    assert commands(measured, "check") == ["sudo sparkring status", "sudo sparkring cabling --bandwidth"]
    # Where to run and connect: Node A runs the commands and serves; nothing runs on the other Spark.
    assert [(r["spark"], r["text"]) for r in plain["roles"]] == [
        ("Node A", "Run the commands here. The model answers at its address."), ("The other Spark", "Nothing to run on it.")]
    assert plain["run"] is None and plain["order"] is None
    assert plain["ready"] and "Replace NODE_A with Node A's address." in plain["notes"]


def test_pack_for_a_ring_serves_its_profile_port_and_switches_to_two_pairs(data, node):
    port = next(p for p in data["profiles"] if p["id"] == TP4)["checkpoints"][0]["port"]
    halves = [{"profile": TP2}, {"profile": GLM2, "checkpoint": "nvfp4-qad"}]
    ring, without = packs(node, data, [
        {"layout": "ring", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
        {"layout": "ring", "features": {"ring_halves": False}, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
    ])
    assert commands(ring, "install") == [SCRIPT + f"--profile {TP4}"]
    assert endpoints(ring, "install") == [f"http://NODE_A:{port}/v1"]
    assert [g["key"] for g in ring["groups"]] == ["install", "check", "switch"]
    switch = ring["groups"][-1]
    assert switch["title"] == "Switch to two models" and switch["optional"]
    assert [c["command"] for c in switch["commands"]] == [
        f"sudo sparkring install --profile {TP2} --on 0,1",
        f"sudo sparkring install --profile {GLM2} --on 2,3 --checkpoint nvfp4-qad"]
    assert [c["endpoint"] for c in switch["commands"]] == ["http://NODE_A:8000/v1", "http://SPARK_2:8000/v1"]
    assert "stops the model on all four" in switch["commands"][0]["what"]
    assert switch["commands"][1]["what"].endswith("Run it after the command above finishes.")
    assert [g["key"] for g in without["groups"]] == ["install", "check"]
    assert [(r["spark"], r["text"]) for r in ring["roles"]] == [
        ("Node A", "Run the commands here. The model answers at its address."), ("The other three", "Nothing to run on them.")]
    assert ring["run"] is None and ring["order"] is None


def test_pack_for_two_pairs_installs_each_half_then_switches_back(data, node):
    halves = [{"profile": TP2, "checkpoint": "qad-step-4000"}, {"profile": MIMO2}]
    mimo_port = next(p for p in data["profiles"] if p["id"] == MIMO2)["checkpoints"][0]["port"]
    auto, filled = packs(node, data, [
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves,
         "opts": {**NEW, "order": "fill", "sparks": MY_SPARKS}},
    ])
    # Install, then check the requested models; switching layouts is optional and comes last.
    assert [g["key"] for g in auto["groups"]] == ["install", "check", "switch"]
    assert [g.get("optional", False) for g in auto["groups"]] == [False, False, True]
    assert auto["groups"][2]["note"] == "Optional: switch layouts later. This stops the models above."
    assert commands(auto, "install") == [SCRIPT + f"--profile {TP2} --on 0,1 --checkpoint qad-step-4000",
                                         f"sudo sparkring install --profile {MIMO2} --on 2,3"]
    assert endpoints(auto, "install") == ["http://NODE_A:8000/v1", f"http://SPARK_2:{mimo_port}/v1"]
    first, second = auto["groups"][0]["commands"]
    assert first["what"] == ("Installs SparkRing on all four Sparks and starts Qwen3.8-Flash-Next (qad-step-4000) "
                             "on the first pair: Node A and Spark 1.")
    assert second["what"].startswith("Starts MiMo-V2.6-Flash-MOPD on the second pair: Spark 2 and Spark 3.")
    assert commands(auto, "switch") == [f"sudo sparkring install --profile {TP4}"]
    assert "stops both pairs' models" in auto["groups"][2]["commands"][0]["what"]
    assert auto["groups"][1]["commands"][0]["what"] == "Shows each pair's model separately."
    # Spark 2, two cables from Node A in the ring order, serves the second pair; SPARK_2 stands for its address.
    assert [(r["spark"], r["text"]) for r in auto["roles"]] == [
        ("First pair", "Node A and Spark 1. Answers at Node A's address."),
        ("Second pair", "Spark 2 and Spark 3. Answers at Spark 2's address.")]
    assert auto["run"] == "Run every command on Node A."
    assert auto["order"] == ("Node A is the Spark you run setup on. The cables number the others: Spark 1 is on Node A's "
                             "port 0, Spark 3 is on its port 1, and Spark 2 is the one with no cable to Node A.")
    assert "Replace NODE_A with Node A's address and SPARK_2 with Spark 2's address." in auto["notes"]
    assert "If Spark 2 has no network cable of its own, only Node A can reach the second pair's model." in auto["notes"]
    # Fill in my Sparks: the user's names say where to run, their addresses where each model answers.
    assert commands(filled) == commands(auto)
    assert {c["where"] for c in filled["groups"][0]["commands"][1:] + filled["groups"][1]["commands"]} == {
        "On spark-a (Node A), as a user with sudo"}
    assert endpoints(filled, "install") == ["http://198.51.100.10:8000/v1", f"http://198.51.100.12:{mimo_port}/v1"]
    assert endpoints(filled, "switch") == ["http://198.51.100.10:8015/v1"]
    assert filled["groups"][0]["commands"][0]["what"].endswith("on the first pair: spark-a and spark-b.")
    assert [(r["spark"], r["text"]) for r in filled["roles"]] == [
        ("First pair", "spark-a and spark-b. Answers at 198.51.100.10."),
        ("Second pair", "spark-c and spark-d. Answers at 198.51.100.12.")]
    assert filled["run"] == "Run every command on spark-a."
    assert not any("Replace" in note for note in filled["notes"])
    assert "If spark-c has no network cable of its own, only spark-a can reach the second pair's model." in filled["notes"]


def test_where_to_run_names_the_sparks_the_user_fills_in(data, node):
    pair, ring = packs(node, data, [
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2}, "opts": {**NEW, "order": "fill", "sparks": MY_SPARKS[:2]}},
        {"layout": "ring", "features": BOTH, "main": {"profile": TP4}, "opts": {**NEW, "order": "fill", "sparks": MY_SPARKS}},
    ])
    assert [(r["spark"], r["text"]) for r in pair["roles"]] == [
        ("spark-a (Node A)", "Run the commands here. The model answers at 198.51.100.10."), ("spark-b", "Nothing to run on it.")]
    assert [(r["spark"], r["text"]) for r in ring["roles"]] == [
        ("spark-a (Node A)", "Run the commands here. The model answers at 198.51.100.10."),
        ("spark-b, spark-c and spark-d", "Nothing to run on them.")]
    assert pair["order"] is ring["order"] is None


@pytest.mark.parametrize("opts, expected", [
    ({"form": "installed"}, [f"sudo sparkring install --profile {TP2} --on 0,1", f"sudo sparkring install --profile {GLM2} --on 2,3",
                             f"sudo sparkring install --profile {TP4}"]),
    ({"pin": False}, [f"curl -fsSL https://raw.githubusercontent.com/FujitsuPolycom/sparkring/main/install.sh | bash -s -- --profile {TP2} --on 0,1",
                      f"sudo sparkring install --profile {GLM2} --on 2,3", f"sudo sparkring install --profile {TP4}"]),
    ({"form": "installed", "approval": "plan", "downloadLimit": "850Mbit"},
     [f"sudo sparkring install --profile {TP2} --on 0,1 --download-limit 850Mbit --plan",
      f"sudo sparkring install --profile {GLM2} --on 2,3 --download-limit 850Mbit --plan",
      f"sudo sparkring install --profile {TP4} --download-limit 850Mbit --plan"]),
    ({"form": "installed", "approval": "yes"}, [f"sudo sparkring install --profile {TP2} --on 0,1 --yes",
                                                f"sudo sparkring install --profile {GLM2} --on 2,3 --yes",
                                                f"sudo sparkring install --profile {TP4} --yes"]),
    # Ask during install leaves out --yes, so that every installation asks.
    ({"form": "installed", "approval": "yes", "order": "ask"}, [f"sudo sparkring install --profile {TP2} --on 0,1",
                                                                f"sudo sparkring install --profile {GLM2} --on 2,3",
                                                                f"sudo sparkring install --profile {TP4}"]),
])
def test_pack_options_apply_to_every_install_command(data, node, opts, expected):
    pack, = packs(node, data, [{"layout": "halves", "features": BOTH, "main": {"profile": TP4},
                                "halves": [{"profile": TP2}, {"profile": GLM2}], "opts": {**NEW, **opts}}])
    assert commands(pack, "install") + commands(pack, "switch") == expected
    assert commands(pack, "check") == ["sudo sparkring status"]
    assert any("asks before it changes" in note for note in pack["notes"]) == (opts.get("order") == "ask")


def test_pack_carries_each_selection_settings_image_and_problems(data, node):
    halves = [{"profile": TP2, "settings": {"max_concurrency": 4, "save_cpu": True}},
              {"profile": MIMO2, "image_option": "2026.09.5"}]
    good, bad = packs(node, data, [
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": {**NEW, "form": "installed"}},
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "opts": {**NEW, "form": "installed"},
         "halves": [{"profile": TP2, "settings": {"max_concurrency": 0}}, {"profile": MIMO2}]},
    ])
    assert commands(good, "install") == [f"sudo sparkring install --profile {TP2} --on 0,1 --max-concurrency 4 --save-cpu",
                                         f"sudo sparkring install --profile {MIMO2} --on 2,3 --image 2026.09.5"]
    first = bad["groups"][0]["commands"][0]
    assert first["command"] == "" and first["endpoint"] is None and "--max-concurrency" in first["error"]
    assert not bad["ready"] and good["ready"]


def test_pack_carries_each_deployed_selection_s_api_endpoint(data, node):
    mimo_port = next(p for p in data["profiles"] if p["id"] == MIMO2)["checkpoints"][0]["port"]
    halves = [{"profile": TP2, "settings": {"api_port": 9100, "api_bind": "198.51.100.20"},
               "endpoint": {"address": "llm.example.net"}},
              {"profile": MIMO2, "endpoint": {"ask": True}}]
    pack, bad, refused = packs(node, data, [
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves,
         "opts": {**NEW, "form": "installed", "approval": "yes"}},
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2, "endpoint": {"address": "http://llm:8000"}}, "opts": NEW},
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2, "settings": {"api_port": 29638}}, "opts": NEW},
    ])
    assert commands(pack, "install") == [
        f"sudo sparkring install --profile {TP2} --on 0,1 --api-bind 198.51.100.20 --api-port 9100 --api-address "
        "llm.example.net --yes",
        # Ask during install leaves out --yes, so that the installer asks.
        f"sudo sparkring install --profile {MIMO2} --on 2,3"]
    assert endpoints(pack, "install") == ["http://llm.example.net:9100/v1", f"http://SPARK_2:{mimo_port}/v1"]
    # The switch command takes the automatic endpoint.
    assert commands(pack, "switch") == [f"sudo sparkring install --profile {TP4} --yes"]
    assert "Replace NODE_A with Node A's address and SPARK_2 with Spark 2's address." in pack["notes"]
    assert ("An install set to ask lists the Spark's addresses and asks which address and port the model uses."
            in pack["notes"])
    first, = bad["groups"][0]["commands"]
    assert first["command"] == "" and first["error"].startswith("--api-address takes a host name or an IP address")
    first, = refused["groups"][0]["commands"]
    assert first["error"] == "--api-port 29638 is the profile's --master-port. Choose another port."


def test_fields_keep_their_entries_apart_from_the_settings_they_give(data, node):
    profile = next(p for p in data["profiles"] if p["id"] == TP2)
    found = run_engine(node, """
const { profile } = value, cp = E.checkpointOf(profile, null), row = name => cp.settings.find(r => r.name === name);
console.log(JSON.stringify({
  kv: ['', '1.5', '24', ' 26 ', '27', '0', 26, 'abc'].map(text => E.readSetting(row('kv_cache_gib'), text)),
  port: ['80', '2222', '9100', '70000'].map(text => E.readSetting(row('api_port'), text)),
  bind: ['0.0.0.0', '198.51.100.20'].map(text => E.readSetting(row('api_bind'), text)),
  address: ['', 'http://llm:8000', 'llm.example.net'].map(E.readAddress),
  limit: ['', 'none', '850Mbit', 'fast', '500kbit'].map(E.readDownloadLimit),
  auto: E.readSelection(profile, cp, { kv_cache_gib: '1.5', max_concurrency: '4', api_port: '80', save_cpu: true },
                        { mode: 'auto', address: 'http://llm:8000' }),
  set: E.readSelection(profile, cp, { kv_cache_gib: 20, api_port: '80', api_bind: '198.51.100.20' },
                       { mode: 'set', address: 'http://llm:8000' }),
}));""", {"profile": profile})
    whole, smallest, largest = "Enter a whole number.", "At least 1.", "At most 26."
    # An empty field keeps the profile's value; a value above the ceiling or not a whole number has a problem.
    assert found["kv"] == [{"problem": ""}, {"problem": whole}, {"value": 24, "problem": ""}, {"value": 26, "problem": ""},
                           {"problem": largest}, {"problem": smallest}, {"value": 26, "problem": ""}, {"problem": whole}]
    assert found["port"] == [{"problem": "From 1024 to 65535."}, {"problem": "This is the port of SparkRing's administration SSH."},
                             {"value": 9100, "problem": ""}, {"problem": "From 1024 to 65535."}]
    assert found["bind"] == [{"problem": "An IPv4 address of the Spark, such as 192.0.2.10."},
                             {"value": "198.51.100.20", "problem": ""}]
    assert found["address"] == [{"value": None, "problem": ""}, {"problem": "A name or an address, without http:// or a port."},
                                {"value": "llm.example.net", "problem": ""}]
    assert [bool(x["problem"]) for x in found["limit"]] == [False, False, False, True, True]
    # The KV cache text 1.5 gives no setting, so nothing falls back to the profile's 24 GiB unnoticed;
    # the endpoint's fields count only while it is set on the page.
    assert found["auto"] == {"settings": {"max_concurrency": 4, "save_cpu": True}, "address": None,
                             "problems": [{"field": "kv_cache_gib", "message": whole}]}
    assert found["set"] == {"settings": {"kv_cache_gib": 20, "api_bind": "198.51.100.20"}, "address": None,
                            "problems": [{"field": "api_port", "message": "From 1024 to 65535."},
                                         {"field": "api_address", "message": "A name or an address, without http:// or a port."}]}


def test_a_pack_gives_no_command_while_a_field_has_a_problem(data, node):
    problem = [{"field": "kv_cache_gib", "message": "Enter a whole number."}]
    halves = [{"profile": TP2}, {"profile": MIMO2, "problems": problem}]
    pair, second, limit, sparks, fine = packs(node, data, [
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2, "problems": problem}, "opts": NEW},
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2}, "opts": {**NEW, "downloadLimit": "fast"}},
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2},
         "opts": {**NEW, "order": "fill", "sparks": [{"host": "spark-a", "host_ip": ""}, {"host": "", "host_ip": "300.1.1.1"}]}},
        {"layout": "pair", "features": BOTH, "main": {"profile": TP2}, "opts": {**NEW, "downloadLimit": "850Mbit"}},
    ])
    for pack in (pair, second, limit, sparks):
        assert not pack["ready"] and set(commands(pack)) == {""}
        assert all(c["error"] for group in pack["groups"] for c in group["commands"])
    assert pair["problems"] == [{**problem[0], "pair": None}]
    # A half's problem names its pair.
    assert second["problems"] == [{**problem[0], "pair": 1}]
    assert limit["problems"] == [{"field": "download_limit", "pair": None,
                                  "message": "Use none, or a rate of at least 1Mbit, such as 850Mbit or 2Gbit."}]
    assert sparks["problems"] == [{"field": "rank1-host_ip", "message": "An IPv4 address, such as 192.0.2.10.", "pair": None}]
    assert fine["ready"] and fine["problems"] == []
    assert commands(fine, "install") == [SCRIPT + f"--profile {TP2} --download-limit 850Mbit"]


def test_compose_readme_names_the_api_at_its_shown_address(data, node):
    profile = next(p for p in data["profiles"] if p["id"] == TP2)
    readme = run_engine(node, """
(async () => {
  const { profile, site, meta } = value, settings = { api_port: 9100, api_bind: '198.51.100.20' };
  const output = await E.render(profile, site, settings, null);
  const checkpoint = E.checkpointOf(profile, null);
  const { bytes } = E.archive(profile, checkpoint, site, settings, output, meta, new Date(2026, 0, 1), 'llm.example.net');
  const text = Buffer.from(bytes).toString('latin1');
  console.log(JSON.stringify(text.slice(text.indexOf('SparkRing Compose deployment'), text.indexOf('PK', text.indexOf('SparkRing Compose deployment')))));
})();""", {"profile": profile, "site": profile["example_site"], "meta": meta_of(data)})
    assert "API http://llm.example.net:9100/v1, model " in readme
    assert "5. curl http://llm.example.net:9100/v1/models lists " in readme
    assert "--api-bind 198.51.100.20 --api-port 9100 --api-address llm.example.net" in readme


def test_spark_problems_name_each_field_the_pack_cannot_use(node):
    ranks = [{"host": "spark-a", "host_ip": ""}, {"host": "", "host_ip": "198.51.100.11"}, {"host": "", "host_ip": ""},
             {"host": "spark-a", "host_ip": "300.1.1.1"}]
    problems = run_engine(node, "console.log(JSON.stringify(E.sparkProblems(value)))", ranks)
    assert [(p["rank"], p["key"], p["message"]) for p in problems] == [
        (2, "host", "Enter a name or an address."),
        (3, "host", "Node A has this name too."),
        (3, "host_ip", "An IPv4 address, such as 192.0.2.10.")]
    assert run_engine(node, "console.log(JSON.stringify(E.sparkProblems(value)))", MY_SPARKS) == []


def test_links_carry_the_layout_each_selection_and_the_install_options(data, node):
    by_id = {p["id"]: p for p in data["profiles"] if p["id"] in {TP2, TP4, GLM2}}
    choices = {"layout": "halves", "mode": "install", "image": None,
               "main": {"profile": TP4, "checkpoint": None, "settings": {}},
               "halves": [{"profile": TP2, "checkpoint": "qad-step-4000", "settings": {"max_concurrency": 4, "save_cpu": True}},
                          {"profile": GLM2, "checkpoint": None, "settings": {}}],
               "install": {"form": "installed", "source": "release", "approval": "yes", "order": "fill", "limit": "850Mbit"}}
    script = """
const { choices, byId, links } = value, search = E.linkQuery(choices);
console.log(JSON.stringify({ search, back: E.linkChoices(search, byId, { ring_halves: true }),
  others: links.map(([text, features]) => E.linkChoices(text, byId, features)) }));"""
    found = run_engine(node, script, {"choices": choices, "byId": by_id, "links": [
        [f"profile={TP2}&mode=compose&approval=plan", BOTH],             # a link without a layout
        [f"profile={TP4}&layout=halves&first={TP2}", {"ring_halves": False}],
        [f"profile={TP2}&layout=ring&first={TP4}&second=unknown&order=never", BOTH],
        ["profile=unknown", BOTH],
    ]})
    assert found["search"] == (f"profile={TP4}&layout=halves&mode=install"
                               f"&first={TP2}&first.checkpoint=qad-step-4000&first.max_concurrency=4&first.save_cpu=1"
                               f"&second={GLM2}&form=installed&source=release&approval=yes&order=fill&limit=850Mbit")
    assert found["back"] == choices
    old, unsupported, mismatched, unknown = found["others"]
    assert (old["layout"], old["mode"], old["install"]) == ("pair", "compose", {"approval": "plan", "limit": ""})
    assert unsupported["layout"] == "ring" and unsupported["halves"][0]["profile"] == TP2
    assert mismatched["layout"] == "pair" and mismatched["halves"] == [None, None] and "order" not in mismatched["install"]
    assert unknown is None


def test_links_carry_each_selection_s_api_endpoint(data, node):
    by_id = {p["id"]: p for p in data["profiles"] if p["id"] in {TP2, TP4, MIMO2}}
    choices = {"layout": "halves", "mode": "install", "image": None,
               "main": {"profile": TP4, "checkpoint": None, "settings": {}},
               "halves": [{"profile": TP2, "checkpoint": None, "settings": {"api_bind": "198.51.100.20", "api_port": 9100},
                           "endpoint": {"mode": "set", "address": "llm.example.net"}},
                          {"profile": MIMO2, "checkpoint": None, "settings": {}, "endpoint": {"mode": "ask", "address": ""}}],
               "install": {"form": "installed", "approval": "ask", "limit": ""}}
    found = run_engine(node, """
const { choices, byId } = value, search = E.linkQuery(choices);
console.log(JSON.stringify({ search, back: E.linkChoices(search, byId, { ring_halves: true }),
  odd: E.linkChoices('profile=""" + TP2 + """&endpoint=never&api_bind=0.1.2.3.4&api_port=x', byId, {}) }));""",
                       {"choices": choices, "byId": by_id})
    assert found["search"] == (f"profile={TP4}&layout=halves&mode=install&first={TP2}&first.api_bind=198.51.100.20"
                               "&first.api_port=9100&first.endpoint=set&first.api_address=llm.example.net"
                               f"&second={MIMO2}&second.endpoint=ask&form=installed&approval=ask")
    assert found["back"] == choices
    # A link that names no valid endpoint choice leaves it automatic.
    assert found["odd"]["main"] == {"profile": TP2, "checkpoint": None, "settings": {}}


GIB = 2 ** 30


def test_kv_measurement_prefers_the_checkpoint_then_the_profile():
    record = {"tokens": 1000, "kv_bytes_per_rank": 10 * GIB, "conditions": "c", "source": "s", "witness": "w", "kv_evidence": "e",
              "checkpoints": {"other": {"tokens": 400, "kv_bytes_per_rank": 5 * GIB, "conditions": "d"}}}
    # A measurement carries its record, conditions and KV-size evidence, None where it names none.
    assert export.kv_measurement(record, "default", "default") == {"tokens": 1000, "kv_bytes_per_rank": 10 * GIB, "checkpoint": "default",
                                                                    "source": "s", "conditions": "c", "kv_evidence": "e"}
    assert export.kv_measurement(record, "other", "default") == {"tokens": 400, "kv_bytes_per_rank": 5 * GIB, "checkpoint": "other",
                                                                  "source": None, "conditions": "d", "kv_evidence": None}
    # Another checkpoint of the same profile takes the profile's own measurement and names its checkpoint.
    assert export.kv_measurement({**record, "checkpoint": "older"}, "third", "default")["checkpoint"] == "older"
    assert export.kv_measurement({k: v for k, v in record.items() if k != "kv_bytes_per_rank"}, "default", "default") is None
    assert export.kv_measurement(None, "default", "default") is None


def test_checkpoints_carry_the_measurement_of_their_own_profile(data):
    capacity = {(p["id"], c["name"]): c["capacity"] for p in data["profiles"] for c in p["checkpoints"]}
    records = export.capacity_records()
    measured = lambda c: {key: c[key] for key in ("tokens", "kv_bytes_per_rank", "checkpoint")}  # noqa: E731
    assert measured(capacity[(GLM2, "nvfp4-spark")]) == {"tokens": 1530566, "kv_bytes_per_rank": 10 * GIB, "checkpoint": "nvfp4-spark"}
    assert measured(capacity[(GLM2, "nvfp4-qad")]) == {"tokens": 736274, "kv_bytes_per_rank": 5 * GIB, "checkpoint": "nvfp4-qad"}
    # A four-Spark profile never takes a two-Spark figure.
    assert measured(capacity[(TP4, "nvfp4-qad")]) == {"tokens": 6128169, "kv_bytes_per_rank": 40 * GIB, "checkpoint": "nvfp4-spark"}
    assert capacity[(TP4, "nvfp4-qad")]["source"] == records[TP4]["source"]
    # The Qwen estimate keeps where it comes from: step 4000 with SparkCache on the shared 2026.09.3 image.
    qwen = capacity[(TP2, "qad-step5500-ple1000")]
    assert qwen["checkpoint"] == "qad-step-4000" and qwen["source"] == "runtime/releases/shared-2026.09.3/correctness.json"
    assert "SparkRing 2026.09.3" in qwen["conditions"] and "sparkcache.json" in qwen["kv_evidence"]
    assert capacity[(MIMO2, None)] is None


def test_kv_estimate_scales_the_measured_pool_to_the_chosen_size(data, node):
    by_id = {p["id"]: p for p in data["profiles"]}
    cases = [(GLM2, None, {}), (GLM2, None, {"kv_cache_gib": 5}), (GLM2, "nvfp4-qad", {}), (TP2, None, {"context_length": 131072}),
             (MIMO2, None, {}), ("deepseek-v41-flash-tp4", None, {}), ("qwen38-flash-next-qad-tp4", "jmni-qad5500-hybrid", {})]
    found = run_engine(node, """
console.log(JSON.stringify(value.cases.map(([id, name, settings]) => {
  const checkpoint = E.checkpointOf(value.byId[id], name);
  return [E.kvEstimate(checkpoint, settings), E.kvText(checkpoint, settings)];
})));""", {"byId": {key: by_id[key] for key in {c[0] for c in cases}}, "cases": cases})
    ((glm, glm_text), (half, half_text), (qad, qad_text), (qwen, qwen_text), (mimo, mimo_text), (deepseek, deepseek_text),
     (jmni, jmni_text)) = found
    room = "room for about {} full-length requests by size; not a tested concurrency"
    glm_record = export.capacity_records()[GLM2]
    # 1,530,566 tokens at 10 GiB in a 1,048,576-token window.
    assert (glm["gib"], glm["tokens"], glm["requests"]) == (10, 1500000, 1.5)
    assert glm_text == {"gib": 10, "line": "About 1.5 million tokens · " + room.format("1.5"),
                        "basis": "Estimated from nvfp4-spark at 10 GiB", "source": glm_record["source"],
                        "conditions": glm_record["conditions"]}
    # Half the bytes hold half the tokens: 765,283, rounded to two figures.
    assert (half["tokens"], half["requests"]) == (770000, 0.7)
    assert half_text["line"] == "About 770,000 tokens · " + room.format("0.7")
    # nvfp4-qad has its own measurement and record: 736,274 tokens at 5 GiB in a 524,288-token window.
    assert (qad["gib"], qad["tokens"], qad["requests"], qad["measured_gib"]) == (5, 740000, 1.4, 5)
    assert (qad_text["basis"], qad_text["source"]) == (
        "Estimated from nvfp4-qad at 5 GiB", glm_record["checkpoints"]["nvfp4-qad"]["source"])
    assert qad_text["source"].startswith("performance/records/")
    # Qwen's measurement is of qad-step-4000 at 24 GiB, and the basis says so.
    assert (qwen["tokens"], qwen["requests"]) == (2900000, 22.0)
    assert qwen_text["basis"] == "Estimated from qad-step-4000 at 24 GiB"
    assert qwen_text["source"] == "runtime/releases/shared-2026.09.3/correctness.json"
    # The JMNI hybrid on four Sparks is sized by step 4000's four-Spark measurement.
    assert jmni_text["line"] == "About 3.1 million tokens · " + room.format("11.9")
    assert jmni_text["basis"] == "Estimated from qad-step-4000 at 24 GiB"
    assert mimo == {"gib": 12, "measured": False}
    assert mimo_text == {"gib": 12, "line": "Not measured for this model", "basis": "", "source": None, "conditions": None}
    assert deepseek is None and deepseek_text is None


def test_kv_estimate_follows_each_pair_and_its_link(data, node):
    by_id = {p["id"]: p for p in data["profiles"] if p["id"] in {TP4, TP2, GLM2}}
    found = run_engine(node, """
const { byId, search } = value, choices = E.linkChoices(search, byId, { ring_halves: true });
console.log(JSON.stringify({ choices, lines: choices.halves.map(s => {
  const check = E.servingCheck(byId[s.profile], s.settings, s.checkpoint);
  return E.kvText(check.checkpoint, check.settings).line;
}) }));""", {"byId": by_id, "search": f"profile={TP4}&layout=halves&first={GLM2}&first.kv_cache_gib=5&second={GLM2}&second.checkpoint=nvfp4-qad"})
    assert found["choices"]["halves"][0]["settings"] == {"kv_cache_gib": 5}
    assert found["lines"] == [
        "About 770,000 tokens · room for about 0.7 full-length requests by size; not a tested concurrency",
        "About 740,000 tokens · room for about 1.4 full-length requests by size; not a tested concurrency"]


def test_field_problems_name_each_field_the_generator_refuses(node):
    example = export.example_site("qwen38-flash-next-tp2")
    site = json.loads(json.dumps(example))
    site["name"] = "Ring"
    site["ranks"][1].update(host=site["ranks"][0]["host"], gid=300, cache=site["ranks"][1]["model"] + "/cache")
    script = "console.log(JSON.stringify(E.fieldProblems(value, 2)))"
    problems = run_engine(node, script, site)
    assert [(p["rank"], p["key"]) for p in problems] == [(None, "name"), (1, "host"), (1, "gid"), (1, "cache")]
    assert run_engine(node, script, example) == []


def test_tagged_data_pins_the_tag(monkeypatch):
    monkeypatch.setattr(export, "profile_ids", lambda: ["mimo-v26-flash-mopd-tp2"])
    data = export.export(tag="2026.10.1", commit=COMMIT)
    assert data["ref"] == "2026.10.1" and data["commit"] == COMMIT
    assert data["since_tag"] is None and data["commits_since"] is None


@pytest.mark.parametrize("description, expected", [
    ("2026.10.0-24-g66f04d51", ("2026.10.0", 24)),
    ("shared-2026.09.4-rc.4-0-gabc123", ("shared-2026.09.4-rc.4", 0)),
    ("66f04d51", (None, None)),
    (None, (None, None)),
])
def test_untagged_checkouts_are_named_by_the_release_before_them(description, expected):
    assert export.release_distance(description) == expected


def test_untagged_git_checkouts_record_the_release_before_them(monkeypatch):
    monkeypatch.setattr(export, "profile_ids", lambda: ["mimo-v26-flash-mopd-tp2"])
    monkeypatch.setattr(export, "source_identity", lambda: (None, COMMIT))
    monkeypatch.setattr(export, "_git", lambda *argv: "2026.10.0-24-g0000000")
    data = export.export()
    assert (data["ref"], data["since_tag"], data["commits_since"]) == (COMMIT, "2026.10.0", 24)


@pytest.mark.parametrize("line, message", [
    ("    OTHER: zqif0", "reached key OTHER"),
    ("    NOTE: zqhost0", "zqhost reached a container file"),
    ("    NOTE: zqunknown", "unrecognized sentinel"),
    ("    NCCL_IB_GID_INDEX: '3'", "without the GID sentinel"),
])
def test_template_check_refuses_values_the_engine_cannot_place(line, message):
    site = export.sentinel_site(export.example_site("qwen38-flash-next-tp2"))
    with pytest.raises(ValueError, match=message):
        export._check_template(line + "\n", site, 0, "f" * 64, export._yaml_key, "test")


def test_page_inlines_data_engine_and_verification(data):
    html = export.page(data, {"valid": 1, "invalid": 2, "stricter": 3, "archives": 4, "seed": 5})
    assert "__DATA__" not in html and "__ENGINE__" not in html
    embedded = html.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0]
    assert "<" not in embedded
    document = json.loads(embedded)
    assert document["profiles"] == data["profiles"] and document["verification"]["archives"] == 4
    assert document["features"] == data["features"]
    assert "module.exports" not in html


def test_engine_matches_compose_build(data, node, tmp_path):
    summary = verify.run(data, per_checkpoint=CASES, node=node, work=tmp_path)
    assert summary["failures"] == []
    assert summary["valid"] and summary["invalid"] and summary["stricter"] and summary["archives"]


def test_generator_writes_a_verified_page(node, tmp_path, monkeypatch):
    monkeypatch.setattr(export, "profile_ids", lambda: ["qwen38-flash-next-tp2", "mimo-v26-flash-mopd-tp2"])
    catalog = export.image_catalog
    monkeypatch.setattr(export, "image_catalog", lambda: [row for row in catalog() if row["default"] or "2026.09.5" in row["tags"]])
    output = tmp_path / "site"
    assert generate_compose_builder.main(["--output", str(output), "--commit", COMMIT, "--verify", "--cases", "1"]) == 0
    html = (output / "index.html").read_text(encoding="utf-8")
    embedded = json.loads(html.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0])
    assert embedded["verification"]["archives"] > 0
    other = next(row for row in embedded["images"] if not row["default"])
    image = json.loads((output / other["file"]).read_text(encoding="utf-8"))
    assert image["image"] == other["name"] and {p["id"] for p in image["profiles"]} == set(other["profiles"])


def test_a_new_installation_is_never_plan_only(data, node):
    # install.sh --plan stops on a Spark without the SparkRing package, so a new installation asks instead.
    halves = [{"profile": TP2}, {"profile": GLM2, "checkpoint": "nvfp4-qad"}]
    new, installed = packs(node, data, [
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": {**NEW, "approval": "plan"}},
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves,
         "opts": {**NEW, "form": "installed", "approval": "plan"}},
    ])
    assert commands(new, "install") == [SCRIPT + f"--profile {TP2} --on 0,1",
                                        f"sudo sparkring install --profile {GLM2} --on 2,3 --checkpoint nvfp4-qad"]
    assert not any("--plan" in c for c in commands(new))
    assert all(c.endswith("--plan") for c in commands(installed, "install") + commands(installed, "switch"))
