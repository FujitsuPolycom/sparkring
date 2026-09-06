import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("lil_plan", HERE / "plan.py")
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


@pytest.fixture
def inputs():
    return planner.read_json(HERE / "glm53.json"), planner.read_json(
        HERE / "site.example.json"
    )


def test_render_is_deterministic_with_real_pins(inputs):
    descriptor, site = inputs
    before = copy.deepcopy(inputs)
    result = planner.render(descriptor, site)
    assert result == planner.render(descriptor, site)
    assert inputs == before
    assert result["executable"] is False
    assert len(result["ranks"]) == 4
    assert len({r["mounts"][-1]["source"] for r in result["ranks"]}) == 4
    runtime = planner.read_json(planner.ROOT / descriptor["sources"]["runtime"]["path"])
    assert result["ranks"][0]["image"] == runtime["operator_image"]["reference"]
    assert result["native_identities"]["sircl"] == runtime["sircl"]["native_sha256"]
    assert (
        result["checkpoints"]["draft"]["repository"] == "incoai/GLM-5.3-Flash-DFlash2"
    )
    assert result["transport"]["configuration_resolved"] is False


def test_cache_disabled_omits_cache_mount_and_connector_plan(inputs):
    descriptor, site = inputs
    site["cache"] = {"enabled": False}
    result = planner.render(descriptor, site)
    for rank in result["ranks"]:
        assert rank["cache"] is None
        assert {m["role"] for m in rank["mounts"]} == {"target", "draft", "jit"}


def test_capacity_overrides_and_restore_only(inputs):
    descriptor, site = inputs
    site["settings"] = {
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "kv_cache_memory_bytes": 123456,
        "max_model_len": 65536,
        "port": 8000,
    }
    site["cache"]["access_mode"] = "restore-only"
    result = planner.render(descriptor, site)["ranks"][0]
    assert result["settings"]["kv_cache_memory_bytes"] == 123456
    assert result["settings"]["max_num_seqs"] == 8
    assert result["cache"]["access_mode"] == "restore-only"


@pytest.mark.parametrize(
    "field,value",
    [
        ("tp", 2),
        ("dcp", 2),
        ("speculator", "mtp"),
        ("speculative_tokens", 3),
        ("max_num_seqs", 17),
        ("port", 65536),
        ("max_num_batched_tokens", True),
        ("max_model_len", -1),
    ],
)
def test_unsupported_settings(inputs, field, value):
    descriptor, site = inputs
    site["settings"][field] = value
    with pytest.raises(ValueError):
        planner.render(descriptor, site)


@pytest.mark.parametrize(
    "bad",
    ["/", "relative/path", "/srv/../models", "/srv/models/", "/srv/x,y", "/srv/x\ny"],
)
def test_bad_mounts(inputs, bad):
    descriptor, site = inputs
    site["storage"]["target"] = bad
    with pytest.raises(ValueError):
        planner.render(descriptor, site)


def test_overlapping_mounts(inputs):
    descriptor, site = inputs
    site["storage"]["cache"] = site["storage"]["jit"] + "/cache"
    with pytest.raises(ValueError, match="overlap"):
        planner.render(descriptor, site)


@pytest.mark.parametrize("field", ["rank", "host"])
def test_duplicate_rank_or_host(inputs, field):
    descriptor, site = inputs
    site["nodes"][1][field] = site["nodes"][0][field]
    with pytest.raises(ValueError):
        planner.render(descriptor, site)


def test_source_drift(inputs):
    descriptor, site = inputs
    descriptor["sources"]["runtime"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source changed"):
        planner.render(descriptor, site)


def test_missing_namespace(inputs):
    descriptor, site = inputs
    del site["cache"]["namespace"]
    with pytest.raises(ValueError, match="namespace"):
        planner.render(descriptor, site)


def test_unknown_settings_rejected(inputs):
    descriptor, site = inputs
    site["settings"]["execute"] = True
    with pytest.raises(ValueError, match="unknown"):
        planner.render(descriptor, site)


def test_topology_changes_namespace(inputs):
    descriptor, site = inputs
    original = planner.render(descriptor, site)["ranks"][0]["cache"]["namespace"]
    # An independently reviewed descriptor can permit another topology. Merely
    # reusing a friendly namespace must not reuse its physical storage directory.
    descriptor["supported"]["dcp"].append(2)
    site["settings"]["dcp"] = 2
    changed = planner.render(descriptor, site)["ranks"][0]["cache"]["namespace"]
    assert changed != original


def test_disabled_cache_rejects_ignored_controls(inputs):
    descriptor, site = inputs
    site["cache"]["enabled"] = False
    with pytest.raises(ValueError, match="ignored"):
        planner.render(descriptor, site)


def test_mutable_image_rejected(inputs, monkeypatch):
    descriptor, site = inputs
    original = planner.checked_source

    def changed(relative, digest):
        data = original(relative, digest)
        if "operator_image" in data:
            data["operator_image"]["reference"] = "example/image:latest"
        return data

    monkeypatch.setattr(planner, "checked_source", changed)
    with pytest.raises(ValueError, match="immutable"):
        planner.render(descriptor, site)


def test_cli_success_and_invalid_input(tmp_path):
    command = [
        sys.executable,
        str(HERE / "plan.py"),
        "--descriptor",
        str(HERE / "glm53.json"),
        "--site",
        str(HERE / "site.example.json"),
    ]
    success = subprocess.run(command, capture_output=True, text=True)
    assert success.returncode == 0, success.stderr
    assert json.loads(success.stdout)["executable"] is False
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    command[-1] = str(bad)
    failure = subprocess.run(command, capture_output=True, text=True)
    assert failure.returncode == 2
    assert failure.stdout == ""
    assert "Invalid integration plan:" in failure.stderr
