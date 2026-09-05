"""Offline contracts for portable native and cache qualification commands."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location("qualification_" + name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


native = load("run_native")
cache = load("model_cache")
recall = load("recall_prompt")


def inputs(tmp_path):
    site = json.loads((HERE.parent / "site.example.json").read_text())
    site["topology_file"] = "fabric.json"
    (tmp_path / "site.json").write_text(json.dumps(site))
    (tmp_path / "fabric.json").write_bytes((HERE.parent / "fabric.example.json").read_bytes())
    _, topology, _ = native.profile.load_site(tmp_path / "site.json")
    bundle_sha = native.profile.PINS["canonical_bundle_manifest_sha256"]
    receipt = {"schema": "sparkring-mtp3-mesh-image-receipt/v1", "checks_passed": True,
               "platform": "linux/arm64", "parent_image_id": native.profile.IMAGE["operator_image"]["image_id"],
               "image_id": "sha256:" + "a" * 64, "image_reference": "sha256:" + "a" * 64,
               "bundle_manifest_sha256": bundle_sha, "source_receipt_sha256": "b" * 64,
               "inside_image": {"checks_passed": True, "bundle_manifest_sha256": bundle_sha,
                                "source_receipt_sha256": "b" * 64, "cuda_initialized": False, "model_loaded": False}}
    receipt_path = tmp_path / "image.json"
    receipt_path.write_text(json.dumps(receipt))
    plan = {"schema": "sparkring-mtp3-mesh-render/v1", "image": receipt,
            "image_receipt_sha256": native.profile.sha(receipt_path), "topology_sha256": topology.sha256,
            "files": {name: native.profile.sha(tmp_path / name) for name in ("site.json", "fabric.json")},
            "ranks": [{"rank": r.rank, "ssh_alias": r.ssh_alias, "management_netdev": r.management_netdev} for r in topology.ranks]}
    (tmp_path / "fabric-plan.json").write_text(json.dumps(plan))
    return receipt_path


def test_plan_uses_rendered_site_and_exact_image(tmp_path):
    receipt = inputs(tmp_path)
    plan = native.make_plan(tmp_path, receipt, [4, 20, 28, 64], 29960)
    assert len(plan["cells"]) == 4
    for cell in plan["cells"]:
        assert len(cell["ranks"]) == 4
        for rank in cell["ranks"]:
            argv = rank["argv"]
            assert "MASTER_ADDR=192.0.2.10" in argv
            assert "GLOO_SOCKET_IFNAME=enP7s7" in argv
            assert "sha256:" + "a" * 64 in argv
            assert argv[argv.index("--bytes") + 1] == str(cell["rows"] * 8192)
            assert rank["host"] == f"spark-r{rank['rank']}"
            assert "196608" in next(item for item in argv if item.startswith("B12X_ROCE_TWO_WAVE"))


@pytest.mark.parametrize("filename", ["site.json", "fabric.json", "image.json"])
def test_modified_input_rejected(tmp_path, filename):
    receipt = inputs(tmp_path)
    path = tmp_path / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="match"):
        native.make_plan(tmp_path, receipt, [4], 29960)


@pytest.mark.parametrize("rows,port", [([8192], 29960), ([4, 4], 29960), ([], 29960), ([4], 65535), ([4], 100)])
def test_native_workload_bounds(tmp_path, rows, port):
    receipt = inputs(tmp_path)
    with pytest.raises(ValueError):
        native.make_plan(tmp_path, receipt, rows, port)


def test_ram_and_external_cache_evidence_remains_distinct():
    values = cache.cache_metrics('vllm:prefix_cache_hits_total{model_name="x"} 8192\n'
                                'vllm:external_prefix_cache_hits_total{model_name="x"} 0\n')
    assert values['vllm:prefix_cache_hits_total{model_name="x"}'] == 8192
    assert values['vllm:external_prefix_cache_hits_total{model_name="x"}'] == 0
    assert cache.cache_metrics("external_hits NaN\nexternal_hits broken") == {}


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "2.1"])
def test_model_temperature_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        cache.temperature_value(value)


def test_model_check_defaults_to_temperature_one(monkeypatch, tmp_path, capsys):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Remember cobalt orchard lantern. Repeat that phrase.")
    output = tmp_path / "out"
    monkeypatch.setattr("sys.argv", ["model_cache", "--endpoint", "http://localhost", "--model", "model",
                                    "--kind", "semantic", "--prompt-file", str(prompt), "--expected-text",
                                    "cobalt orchard lantern", "--max-tokens", "512", "--output", str(output)])
    monkeypatch.setattr(cache, "fetch", lambda *args: pytest.fail("Unexpected network request"))
    cache.main()
    assert not output.exists()
    plan = json.loads(capsys.readouterr().out)
    assert plan["executed"] is False
    assert plan["temperature"] == 1.0 and plan["max_tokens"] == 512
    assert plan["expected_text"] == "cobalt orchard lantern"


def test_recall_prompt_has_exact_bytes_and_rejects_overwrite(tmp_path):
    output = tmp_path / "recall.txt"
    receipt = recall.write_prompt(output)
    assert hashlib.sha256(output.read_bytes()).hexdigest() == "3d2bc5228895566b1497e6f35f6c5aa051685f99438f27226134dfcfab15c277"
    assert len(output.read_bytes()) == 129455
    assert not output.read_bytes().endswith(b"\n")
    assert b"\r" not in output.read_bytes()
    assert receipt["requests_sent"] == 0
    assert receipt["expected_text"] == "cobalt orchard lantern"
    with pytest.raises(FileExistsError):
        recall.write_prompt(output)
