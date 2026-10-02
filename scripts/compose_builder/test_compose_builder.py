"""The Compose builder's data, page and engine (docs/operations/compose-builder.md)."""
import json
import os
import shutil

import pytest

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


def test_tagged_data_pins_the_tag(monkeypatch):
    monkeypatch.setattr(export, "profile_ids", lambda: ["mimo-v26-flash-mopd-tp2"])
    data = export.export(tag="2026.10.1", commit=COMMIT)
    assert data["ref"] == "2026.10.1" and data["commit"] == COMMIT


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
    output = tmp_path / "site"
    assert generate_compose_builder.main(["--output", str(output), "--commit", COMMIT, "--verify", "--cases", "1"]) == 0
    html = (output / "index.html").read_text(encoding="utf-8")
    embedded = json.loads(html.split('<script type="application/json" id="data">', 1)[1].split("</script>", 1)[0])
    assert embedded["verification"]["archives"] > 0
