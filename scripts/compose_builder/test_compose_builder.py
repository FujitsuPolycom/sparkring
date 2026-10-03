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


def test_features_record_the_command_options_the_source_defines(data, tmp_path):
    # This checkout's `sparkring install` defines --on.
    assert data["features"]["ring_halves"] is True
    assert data["features"] == export.features()
    host = tmp_path / "runtime" / "host"
    host.mkdir(parents=True)
    assert export.features(tmp_path) == {"ring_halves": False, "cable_check": False}
    (host / "install_workflow.py").write_text('def main():\n    parser.add_argument("--on", metavar="RANKS")\n')
    # A help text that names an option does not define it.
    (host / "cabling.py").write_text('def main():\n    parser.add_argument("--json", help="unlike --bandwidth")\n')
    assert export.features(tmp_path) == {"ring_halves": True, "cable_check": False}
    (host / "cabling.py").write_text('def main():\n    parser.add_argument("--bandwidth", action="store_true")\n')
    assert export.features(tmp_path) == {"ring_halves": True, "cable_check": True}


def test_image_catalog_lists_the_default_first_with_options_install_accepts(data):
    rows = data["images"]
    assert rows[0]["default"] and rows[0]["file"] is None and sum(row["default"] for row in rows) == 1
    for row in rows:
        assert set(row["profiles"]) <= set(export.profile_ids())
        path = installer_image.lock_path(row["option"])
        assert path is None if row["default"] else path.parent.name == row["name"]
        assert row["default"] or row["file"] == f"images/{row['name']}.json"
    assert {row["name"]: row["option"] for row in rows}["dev-20261001-statusrows-cuda1342-nccl2323-status034"] == "statusrows"


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
MY_SPARKS = [{"host": "spark-a", "host_ip": "192.168.1.10"}, {"host": "spark-b", "host_ip": "192.168.1.11"},
             {"host": "spark-c", "host_ip": "192.168.1.12"}, {"host": "spark-d", "host_ip": ""}]


def packs(node, data, cases):
    """The command pack of each case: {layout, features, main, halves, opts}, a selection being
    {profile, checkpoint, settings} with a profile ID."""
    used = {selection["profile"] for case in cases for selection in [case["main"], *case.get("halves", [])]}
    # A selection's image_option stands for its profile rendered on that installer image.
    script = """
const byId = Object.fromEntries(value.profiles.map(p => [p.id, p]));
const sel = s => ({ profile: { ...byId[s.profile], image_option: s.image_option || null }, checkpoint: s.checkpoint || null,
  settings: s.settings || {} });
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
    assert install["where"] == "On Node A, the Spark on your network, as a user with sudo"
    assert install["what"] == "Installs SparkRing and starts Qwen3.8-Flash-Next on both Sparks."
    assert install["endpoint"] == "http://NODE_A:8000/v1"
    assert commands(plain, "check") == ["sudo sparkring status"]
    assert commands(measured, "check") == ["sudo sparkring status", "sudo sparkring cabling --bandwidth"]
    assert [r["spark"] for r in plain["roles"]] == ["Node A", "Spark 1"] and plain["order"] is None
    assert plain["ready"] and "Replace NODE_A with the address of Node A." in plain["notes"]


def test_pack_for_a_ring_serves_its_profile_port_and_switches_to_two_pairs(data, node):
    port = next(p for p in data["profiles"] if p["id"] == TP4)["checkpoints"][0]["port"]
    halves = [{"profile": TP2}, {"profile": GLM2, "checkpoint": "nvfp4-qad"}]
    ring, without = packs(node, data, [
        {"layout": "ring", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
        {"layout": "ring", "features": {"ring_halves": False}, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
    ])
    assert commands(ring, "install") == [SCRIPT + f"--profile {TP4}"]
    assert endpoints(ring, "install") == [f"http://NODE_A:{port}/v1"]
    switch = next(g for g in ring["groups"] if g["key"] == "switch")
    assert switch["title"] == "Switch to two models"
    assert [c["command"] for c in switch["commands"]] == [
        f"sudo sparkring install --profile {TP2} --on 0,1",
        f"sudo sparkring install --profile {GLM2} --on 2,3 --checkpoint nvfp4-qad"]
    assert [c["endpoint"] for c in switch["commands"]] == ["http://NODE_A:8000/v1", "http://SPARK_2:8000/v1"]
    assert "stops the model on all four" in switch["commands"][0]["what"]
    assert "once the one before has finished" in switch["commands"][1]["what"]
    assert [g["key"] for g in without["groups"]] == ["install", "check"]
    assert [r["spark"] for r in ring["roles"]] == ["Node A", "Spark 1", "Spark 2", "Spark 3"]
    assert ring["roles"][1]["text"] == "Cabled to Node A's p0 port." and "can't choose" in ring["order"]


def test_pack_for_two_pairs_installs_each_half_then_switches_back(data, node):
    halves = [{"profile": TP2, "checkpoint": "qad-step-4000"}, {"profile": MIMO2}]
    mimo_port = next(p for p in data["profiles"] if p["id"] == MIMO2)["checkpoints"][0]["port"]
    auto, filled = packs(node, data, [
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves, "opts": NEW},
        {"layout": "halves", "features": BOTH, "main": {"profile": TP4}, "halves": halves,
         "opts": {**NEW, "order": "fill", "sparks": MY_SPARKS}},
    ])
    assert [g["key"] for g in auto["groups"]] == ["install", "switch", "check"]
    assert commands(auto, "install") == [SCRIPT + f"--profile {TP2} --on 0,1 --checkpoint qad-step-4000",
                                         f"sudo sparkring install --profile {MIMO2} --on 2,3"]
    assert endpoints(auto, "install") == ["http://NODE_A:8000/v1", f"http://SPARK_2:{mimo_port}/v1"]
    first, second = auto["groups"][0]["commands"]
    assert first["what"] == "Installs SparkRing and starts Qwen3.8-Flash-Next (qad-step-4000) on the first pair: Node A and Spark 1."
    assert second["what"].startswith("Starts MiMo-V2.6-Flash-MOPD on the second pair: Spark 2 and Spark 3.")
    assert commands(auto, "switch") == [f"sudo sparkring install --profile {TP4}"]
    assert "stops both pairs' models" in auto["groups"][1]["commands"][0]["what"]
    assert auto["groups"][2]["commands"][0]["what"] == "Shows each pair's model separately."
    assert "Replace NODE_A and SPARK_2 with the addresses of Node A and Spark 2." in auto["notes"]
    # Fill in my Sparks: the user's names say where to run, their addresses where each model answers.
    assert commands(filled) == commands(auto)
    assert {c["where"] for c in filled["groups"][0]["commands"][1:] + filled["groups"][2]["commands"]} == {
        "On spark-a (Node A), as a user with sudo"}
    assert endpoints(filled, "install") == ["http://192.168.1.10:8000/v1", f"http://192.168.1.12:{mimo_port}/v1"]
    assert endpoints(filled, "switch") == ["http://192.168.1.10:8015/v1"]
    assert filled["groups"][0]["commands"][0]["what"].endswith("on the first pair: spark-a and spark-b.")
    assert [r["spark"] for r in filled["roles"]] == ["spark-a (Node A)", "spark-b (Spark 1)", "spark-c (Spark 2)", "spark-d (Spark 3)"]
    assert not any("Replace" in note for note in filled["notes"])


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


def test_spark_problems_name_each_field_the_pack_cannot_use(node):
    ranks = [{"host": "spark-a", "host_ip": ""}, {"host": "", "host_ip": "192.168.1.11"}, {"host": "", "host_ip": ""},
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
