"""Source, receipt and profile admission contracts of the CRC32 SparkCache layer."""

import copy
import json
from pathlib import Path
import shutil

import pytest

from runtime.common import image_lock, installer_image
from runtime.images import derived_layer as layer, prepare_sparkcache as cache


def sources():
    return json.loads(cache.SOURCES.read_text(encoding="utf-8"))


def test_pinned_source_inventory_and_native_contract():
    record = sources()
    assert layer.digest(cache.SOURCES.read_bytes()) == "3fbfb64315b630cae371f8582d9a6880eda05cdc1849ad115a233506f1b0baf1"
    assert record["source"]["commit"] == "8abbe9da67b8143831d21b43d0395c4f347a619a"
    assert record["source"]["tree"] == "1ee0294eccbf233d2c8f08d7a91bb43afea724e0"
    assert record["source"]["native_tree"] == "9d70fdafe087951277770181fcb4d081f9cc3343"
    for row in record["files"].values():
        assert layer.digest((cache.ROOT / row["source"]).read_bytes()) == row["sha256"]
    for profile in installer_image.SPARKCACHE_CRC32_VARIANTS:
        config = json.loads((cache.ROOT / "profiles" / profile / "config.json").read_text())
        args = config["vllm_args"]
        extra = json.loads(args[args.index("--kv-transfer-config") + 1])["kv_connector_extra_config"]
        for field in ("spark_cache_cuda_placement_library", "spark_cache_async_page_capture_library"):
            assert record["native_libraries"][extra[field]] == extra[field + "_sha256"]
    contract = record["files"][cache.PACKAGE + "runtime_patches/vllm-connector-jobs-gab86b70.json"]
    assert contract["sha256"] == "ab53c097d7ca408e995bac7c2bc915b64119881ed64b9cec0b3cc3b30ed2f990"


def test_parent_and_build_inputs_preserve_release_and_admission():
    record = sources()
    raw = (cache.ROOT / record["parent_lock"]).read_bytes()
    assert layer.digest(raw) == record["parent_lock_sha256"]
    parent = json.loads(raw)
    build = json.loads((cache.SOURCES.parent / "build-inputs.json").read_text())
    assert build["parent_image_id"] == parent["image_id"]
    assert parent["name"] == "dev-20261013-dcp2prefill-status038-sircl-tp6"
    assert set(build["profiles"]) == set(parent["profiles"]) | set(installer_image.SPARKCACHE_CRC32_VARIANTS)
    assert len(build["profiles"]) == 17
    assert not (cache.ROOT / build["output"]).exists()
    recipe = (cache.SOURCES.parent / "build.sh").read_text()
    for value in (build["parent_image_id"], build["tag"], build["name"], build["output"], ",".join(build["profiles"])):
        assert value in recipe
    assert recipe.index("/verify.py --vllm-root") < recipe.index("derived_layer.py build")
    for sha in record["native_libraries"].values():
        assert sha in recipe
    plan = {"parent_lock": parent, "payload_bytes": 123, "status_version": parent["status_version"],
            "receipts": {layer.BASE_RECEIPT: "1" * 64, layer.TOOLCHAIN_RECEIPT: "2" * 64}}
    derived = layer.derived_lock(plan, {"Id": "sha256:" + "3" * 64, "Size": 123}, build["name"], build["profiles"])
    for field in ("sircl", "libsircl", "line", "transports", "vllm_plugins", "tuning_defaults_sha256"):
        assert derived[field] == parent[field]
    for profile in installer_image.SPARKCACHE_CRC32_VARIANTS:
        image_lock.validate(derived, profile)
        with pytest.raises(ValueError, match="not admitted"):
            image_lock.validate(parent, profile)


@pytest.fixture
def prepared_inputs(tmp_path):
    """Synthetic receipts exercise preparation without claiming real image bytes."""
    repository = tmp_path / "repository"
    repository.mkdir()
    record = sources()
    for row in record["files"].values():
        dest = repository / row["source"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache.ROOT / row["source"], dest)
    target = cache.PACKAGE + "spark_context_cache_config.py"
    base = {"files": {target: "a" * 64, **record["native_libraries"]},
            "capabilities": {"runtime_status": {"version": "0.3.8"}}}
    parent = json.loads((cache.ROOT / record["parent_lock"]).read_text())
    fixture_sources = repository / "sources.json"
    base_path, toolchain_path = tmp_path / "base.json", tmp_path / "toolchain.json"

    def write_receipts():
        base_raw = layer.canonical_json(base)
        toolchain_raw = layer.canonical_json({"parent_receipt_sha256": layer.digest(base_raw)})
        base_path.write_bytes(base_raw)
        toolchain_path.write_bytes(toolchain_raw)
        parent["parent_receipt_sha256"] = layer.digest(base_raw)
        parent["toolchain_receipt_sha256"] = layer.digest(toolchain_raw)
        parent_path = repository / record["parent_lock"]
        parent_path.parent.mkdir(parents=True, exist_ok=True)
        parent_path.write_bytes(layer.canonical_json(parent))
        record["parent_lock_sha256"] = layer.digest(parent_path.read_bytes())
        fixture_sources.write_bytes(layer.canonical_json(record))

    write_receipts()
    return repository, record, base, base_path, toolchain_path, fixture_sources, write_receipts


def prepare_fixture(fixture, output):
    repository, _, _, base, toolchain, source, _ = fixture
    return cache.prepare(base, toolchain, output, repository=repository, sources=source)


def test_preparation_records_actual_inherited_hashes_and_source_commit(prepared_inputs, tmp_path):
    result = prepare_fixture(prepared_inputs, tmp_path / "context")
    output = Path(result["context"])
    record = prepared_inputs[1]
    provenance = json.loads((output / "files" / record["provenance"].lstrip("/")).read_text())
    target = cache.PACKAGE + "spark_context_cache_config.py"
    assert provenance["files"][target]["inherited_sha256"] == "a" * 64
    assert provenance["files"][target]["sha256"] == record["files"][target]["sha256"]
    descriptor = layer.descriptor(output / "descriptor.json")
    assert layer.digest((output / "descriptor.json").read_bytes()) == json.loads((output / "plan.json").read_text())["descriptor_sha256"]
    source = cache.PACKAGE + "runtime_patches/sparkring-source-8abbe9da67b8.json"
    assert descriptor["files"][source]["inherited_sha256"] is None
    installed = json.loads((output / "files" / source.lstrip("/")).read_text())
    assert installed["source"]["commit"] == sources()["source"]["commit"]
    rewritten = json.loads((output / "files" / layer.BASE_RECEIPT.lstrip("/")).read_text())
    for target, sha in record["native_libraries"].items():
        assert rewritten["files"][target] == sha
    assert (output / "Dockerfile").read_text() == layer.dockerfile()


@pytest.mark.parametrize("change,match", [
    ("native", "native library differs"), ("unknown_module", "unreviewed SparkCache module"),
    ("base", "receipt differs"), ("toolchain", "receipt differs"),
    ("source", "pinned"), ("lock", "pinned"),
])
def test_preparation_refuses_drift_before_writing_context(prepared_inputs, tmp_path, change, match):
    repository, record, base, base_path, toolchain_path, _, rewrite = prepared_inputs
    if change == "native":
        base["files"][next(iter(record["native_libraries"]))] = "0" * 64
        rewrite()
    elif change == "unknown_module":
        base["files"][cache.PACKAGE + "unreviewed.py"] = "0" * 64
        rewrite()
    elif change == "base":
        base_path.write_bytes(b"{}")
    elif change == "toolchain":
        toolchain_path.write_bytes(b"{}")
    elif change == "lock":
        (repository / record["parent_lock"]).write_bytes(b"{}")
    else:
        (repository / next(iter(record["files"].values()))["source"]).write_bytes(b"changed")
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match=match):
        prepare_fixture(prepared_inputs, output)
    assert not output.exists()


def test_native_pin_cannot_be_omitted():
    record = sources()
    base = {"files": copy.deepcopy(record["native_libraries"])}
    del base["files"][next(iter(record["native_libraries"]))]
    with pytest.raises(ValueError, match="native library differs"):
        cache.resolve_descriptor(record, base, cache.ROOT)
