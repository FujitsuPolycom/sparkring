"""The Compose builder's data, page and engine (docs/operations/compose-builder.md)."""
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
