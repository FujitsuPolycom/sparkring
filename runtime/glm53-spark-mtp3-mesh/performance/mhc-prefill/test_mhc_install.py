"""Check source identity, all-before-write validation, and image composition."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "mhc_package_installer", HERE / "install.py"
)
INSTALL = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INSTALL)


def fixture(site):
    manifest, before = INSTALL.package(preimages=True)
    _, after = INSTALL.package()
    ownership = json.loads(
        (HERE.parent / "checkpoints/ownership-contract.json").read_bytes()
    )
    for name, source in before.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source)
    return manifest, before, after, ownership


def snapshot(site):
    return {
        p.relative_to(site).as_posix(): p.read_bytes()
        for p in site.rglob("*")
        if p.is_file()
    }


def test_installs_nine_exact_sources_and_preserves_ownership_requirements(tmp_path):
    manifest, before, after, ownership = fixture(tmp_path)
    original = copy.deepcopy(ownership)
    result = INSTALL.apply(tmp_path, ownership)
    assert len(before) == 8 and len(after) == 9
    assert result["manifest_sha256"] == INSTALL.MANIFEST_SHA256
    assert snapshot(tmp_path) == after
    for old, new in zip(original["files"], ownership["files"]):
        if old["path"] in after:
            assert new["sha256"] == manifest["files"][old["path"]]["after_sha256"]
            assert {k: v for k, v in old.items() if k != "sha256"} == {
                k: v for k, v in new.items() if k != "sha256"
            }
        else:
            assert new == old


@pytest.mark.parametrize(
    "corrupt",
    ["last_runtime", "ownership", "missing_dependency", "new_helper", "duplicate"],
)
def test_rejects_every_preimage_before_first_write(tmp_path, corrupt):
    manifest, before, after, ownership = fixture(tmp_path)
    checkpoint = "vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py"
    if corrupt == "last_runtime":
        (tmp_path / list(before)[-1]).write_bytes(b"unsupported")
    elif corrupt == "ownership":
        next(row for row in ownership["files"] if row["path"] == checkpoint)[
            "sha256"
        ] = "0" * 64
    elif corrupt == "missing_dependency":
        ownership["files"] = [
            row for row in ownership["files"] if row["path"] != checkpoint
        ]
    elif corrupt == "duplicate":
        ownership["files"].append(copy.deepcopy(ownership["files"][0]))
    else:
        helper = next(name for name in after if name not in before)
        (tmp_path / helper).write_bytes(b"preexisting helper")
    runtime = snapshot(tmp_path)
    original = copy.deepcopy(ownership)
    with pytest.raises(ValueError):
        INSTALL.apply(tmp_path, ownership)
    assert snapshot(tmp_path) == runtime
    assert ownership == original


@pytest.mark.parametrize("name", ["manifest.json", "source.tar.gz", "preimages.tar.gz"])
def test_altered_package_is_rejected(tmp_path, name):
    context = tmp_path / "package"
    shutil.copytree(HERE, context, ignore=shutil.ignore_patterns("__pycache__"))
    path = context / name
    path.write_bytes(path.read_bytes() + b"altered")
    with pytest.raises(ValueError, match="differs"):
        INSTALL.package(context, preimages=name == "preimages.tar.gz")


def test_package_matches_serving_source_manifest_and_review_diff():
    manifest, before = INSTALL.package(preimages=True)
    _, after = INSTALL.package()
    import difflib

    expected = []
    # The review diff exposes all changes, including the added helper.
    for name in manifest["files"]:
        expected.extend(
            difflib.unified_diff(
                before.get(name, b"").decode().splitlines(keepends=True),
                after[name].decode().splitlines(keepends=True),
                fromfile="a/" + name,
                tofile="b/" + name,
            )
        )
    actual = (HERE / "source.patch").read_text(encoding="utf-8")
    # Order is immaterial to source identity; compare each file's patch block.
    def blocks(text):
        return sorted("--- a/" + item for item in text.split("--- a/")[1:])
    assert blocks(actual) == blocks("".join(expected))
    assert all(
        hashlib.sha256(after[name]).hexdigest() == row["after_sha256"]
        for name, row in manifest["files"].items()
    )


def test_build_installs_after_continuation_and_keeps_feature_default_off():
    source = (HERE.parent / "install.py").read_text()
    assert source.index("continuation.apply(SITE, data)") < source.index(
        "mhc_prefill.apply(SITE, data)"
    )
    assert source.index("apply_attribution(scheduler)") < source.index(
        "mhc_prefill.apply(SITE, data)"
    )
    assert source.index("mhc_prefill.apply(SITE, data)") < source.index(
        "contract.write_text"
    )
    assert '"token_sharded_mhc_prefill": mhc_transform' in source
    assert '"mhc-prefill"' in (HERE.parent / "prepare.py").read_text()
    assert "ENV SPARK_MHC_PREFILL_SHARD=0" in (HERE.parent / "Dockerfile").read_text()


def test_model_and_checkpoint_preimages_match_maintained_source_packages():
    spec = importlib.util.spec_from_file_location(
        "mhc_preimage_origins", HERE / "verify_preimages.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assert len(result["verified_files"]) == 2
    assert result["docker_build_exercised"] is False
