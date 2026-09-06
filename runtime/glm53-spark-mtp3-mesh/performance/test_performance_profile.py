"""Verify profile selection without host operations or GPU access."""

import importlib.util
import json
from pathlib import Path
import shutil
import sys

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


profile = load("performance_profile_contract", HERE.parent / "profile.py")
example = load("performance_profile_examples", HERE.parent / "make_example.py")


def test_performance_receipt_rejects_modified_identity(tmp_path):
    receipt = json.loads((HERE / "public-image.json").read_text())
    receipt["image_id"] = "sha256:" + "0" * 64
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="repository pin"):
        profile.load_image_receipt(path)


def test_renderer_uses_matched_image_native_and_namespace(tmp_path, monkeypatch):
    receipt = profile.load_image_receipt(HERE / "public-image.json")
    bundle = tmp_path / "bundle"
    shutil.copytree(HERE / "transport/bundle-source", bundle)
    native = bundle / "libspark_transport_capi.so"
    native.write_bytes(b"fixture")
    original = profile.sha

    def digest(path):
        if Path(path) == native:
            return "056243fad27d224b82e437925ffa2aed42037e6bd29f239f56076a832f6ca5cb"
        return original(path)

    monkeypatch.setattr(profile, "sha", digest)
    (tmp_path / "fabric.example.json").write_text(
        json.dumps(example.topology_example())
    )
    site = tmp_path / "site.json"
    site.write_text(json.dumps(example.site_example()))
    output = tmp_path / "launch"
    result = profile.render(site, bundle, output, HERE / "public-image.json")
    assert result["image"]["image_id"] == receipt["image_id"]
    assert result["bundle_manifest_sha256"] == receipt["bundle_manifest_sha256"]
    for rank in range(4):
        env = profile.defaults(output / f"rank{rank}.env")
        assert env["IMAGE_REF"] == receipt["image_reference"]
        assert (
            env["SPARKCACHE_PLACEMENT_LIBRARY_SHA256"]
            == receipt["native_placement_sha256"]
        )
        assert env["SPARKCACHE_CACHE_NAMESPACE"] == receipt["cache_namespace"]
        assert env["SPARKRING_WARMUP_TEMPERATURE"] == "1"


def test_performance_recipe_and_guide_use_same_contract():
    root = HERE.parents[2]
    recipe = json.loads(
        (root / "recipes/glm53-mtp3-cache-checkpoints-tp4.json").read_text()
    )
    receipt = json.loads((HERE / "public-image.json").read_text())
    assert recipe["runtime"]["image"] == receipt["image_reference"]
    assert recipe["runtime"]["image_id"] == receipt["image_id"]
    assert recipe["sparkcache"]["source_commit"] == receipt["sparkcache_commit"]
    assert recipe["sparkcache"]["periodic_full_capture_interval_tokens"] == 0
    guide = (root / recipe["runtime"]["quickstart"]).read_text()
    assert receipt["image_reference"] in guide and receipt["image_id"] in guide
