"""Source, receipt and profile admission contracts of the CRC32 SparkCache layer."""

import copy
import json
from pathlib import Path
import shutil

import pytest

from runtime.common import image_lock, installer_image
from runtime.images import derived_layer as layer, prepare_sparkcache as cache, sparkcache_metrics


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
    assert layer.digest((cache.SOURCES.parent / "built-image.json").read_bytes()) == "1be5d2e5a07b5352b597fe60f290526167d2f2266a645b4c0ab15061cf758a9d"
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


@pytest.fixture(params=[cache.SOURCES, cache.SOURCES.parent.with_name("installer-sparkcache-4a17f1f5198e") / "sources.json", cache.SOURCES.parent.with_name("installer-sparkcache-3daf882e358b") / "sources.json"])
def prepared_inputs(tmp_path, request):
    """Synthetic receipts exercise preparation without claiming real image bytes."""
    repository = tmp_path / "repository"
    repository.mkdir()
    record = json.loads(request.param.read_text())
    for row in record["files"].values():
        dest = repository / row["source"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(cache.ROOT / row["source"], dest)
    target = cache.PACKAGE + "spark_context_cache_config.py"
    base = {"files": {target: "a" * 64, **record["native_libraries"]},
            "capabilities": {"runtime_status": {"version": "0.3.8"}}}
    if record.get("operator_logging"):
        raw = ("def report(self, log_fn):\n    if True:\n        if True:\n" + sparkcache_metrics.ORIGINAL).encode()
        (repository / "metrics.py").write_bytes(raw)
        base["files"][sparkcache_metrics.TARGET] = layer.digest(raw)
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
    return cache.prepare(base, toolchain, output, repository=repository, sources=source,
                         metrics_source=repository / "metrics.py")


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
    source = cache.PACKAGE + "runtime_patches/sparkring-source-" + record["source"]["commit"][:12] + ".json"
    assert descriptor["files"][source]["inherited_sha256"] is None
    installed = json.loads((output / "files" / source.lstrip("/")).read_text())
    assert installed["source"]["commit"] == record["source"]["commit"]
    rewritten = json.loads((output / "files" / layer.BASE_RECEIPT.lstrip("/")).read_text())
    for target, sha in record["native_libraries"].items():
        assert rewritten["files"][target] == sha
    assert (output / "Dockerfile").read_text() == layer.dockerfile()
    if record.get("operator_logging"):
        logger_path = output / "files" / sparkcache_metrics.TARGET.lstrip("/")
        assert sparkcache_metrics.REPLACEMENT in logger_path.read_text()
        assert rewritten["files"][sparkcache_metrics.TARGET] == layer.digest(logger_path.read_bytes())
        assert descriptor["files"][sparkcache_metrics.TARGET]["inherited_sha256"] == prepared_inputs[2]["files"][sparkcache_metrics.TARGET]



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


def test_qwen_layer_pins_runtime_and_admits_both_model_families_without_release_selection():
    directory = cache.SOURCES.parent.with_name("installer-sparkcache-3daf882e358b")
    record = json.loads((directory / "sources.json").read_text())
    build = json.loads((directory / "build-inputs.json").read_text())
    parent = json.loads((cache.ROOT / record["parent_lock"]).read_text())
    assert record["source"]["commit"] == "3daf882e358b0c0846f14b44122204a8774f2f66"
    assert record["source"]["native_tree"] == sources()["source"]["native_tree"]
    assert record["native_libraries"] == sources()["native_libraries"]
    assert set(build["profiles"]) == (set(parent["profiles"]) | set(installer_image.SPARKCACHE_CRC32_VARIANTS)
                                      | set(installer_image.QWEN_SPARKCACHE_VARIANTS))
    assert len(build["profiles"]) == 19
    assert (cache.ROOT / build["output"]).parent == directory
    assert layer.digest((cache.ROOT / build["output"]).read_bytes()) == "6a77a18bb6196a317b9f4f32bf5415b3f37d5b866d0eaa7febf699d6bdacb7a2"
    for row in record["files"].values():
        assert layer.digest((cache.ROOT / row["source"]).read_bytes()) == row["sha256"]
    recipe = (directory / "build.sh").read_text()
    assert "--sources " + (directory / "sources.json").relative_to(cache.ROOT).as_posix() in recipe
    for value in (build["name"], build["tag"], build["output"], ",".join(build["profiles"])):
        assert value in recipe
    assert recipe.index("/verify.py --vllm-root") < recipe.index("derived_layer.py build")
    for sha in record["native_libraries"].values():
        assert sha in recipe


def test_deepseek_layer_pins_runtime_deployment_tools_and_admission():
    directory = cache.SOURCES.parent.with_name("installer-sparkcache-4a17f1f5198e")
    record = json.loads((directory / "sources.json").read_text())
    assert layer.digest((directory / "sources.json").read_bytes()) == "3feeab40a3aa2bbba50c7534f7731a2ed6ec1c4fa217654345503fb5d5c00e94"
    assert record["source"]["commit"] == "4a17f1f5198eb2402bcd6669425c62857937d39f"
    assert record["source"]["tree"] == "aed4c52b9e3368bd59b0ff50ea0e0831ba596759"
    assert record["native_libraries"] == sources()["native_libraries"]
    build = json.loads((directory / "build-inputs.json").read_text())
    parent_raw = (cache.ROOT / record["parent_lock"]).read_bytes()
    assert layer.digest(parent_raw) == record["parent_lock_sha256"]
    parent = json.loads(parent_raw)
    assert set(build["profiles"]) == (set(parent["profiles"]) | set(installer_image.SPARKCACHE_CRC32_VARIANTS)
                                      | set(installer_image.QWEN_SPARKCACHE_VARIANTS)
                                      | set(installer_image.DEEPSEEK_SPARKCACHE_VARIANTS))
    assert len(build["profiles"]) == 20
    assert (cache.ROOT / build["output"]).parent == directory
    for row in record["files"].values():
        assert layer.digest((cache.ROOT / row["source"]).read_bytes()) == row["sha256"]
    output = cache.ROOT / build["output"]
    # An unbuilt composition has no record; a recorded build names its image.
    if output.exists():
        assert json.loads(output.read_text())["image_id"].startswith("sha256:")
    vendor = cache.ROOT / "integrations/sparkcache/4a17f1f5198e"
    deployment = json.loads((vendor / "deployment-files.json").read_text())
    assert layer.digest((vendor / "deployment-files.json").read_bytes()) == "1c7ab79afc9b060f6b1b3c6d2d157170dafb059fe855cac7b106414fa74d1cd9"
    assert deployment["source_commit"] == record["source"]["commit"]
    for path, sha in deployment["files"].items():
        assert layer.digest((vendor / path).read_bytes()) == sha
    recipe = (directory / "build.sh").read_text()
    for value in (build["parent_image_id"], build["tag"], build["name"], build["output"], ",".join(build["profiles"])):
        assert value in recipe
    assert recipe.index("/verify.py --vllm-root") < recipe.index("derived_layer.py build")


def test_combined_metrics_source_is_required_and_authenticated(prepared_inputs, tmp_path):
    repository, record, _, base, toolchain, source, _ = prepared_inputs
    if not record.get("operator_logging"):
        return
    with pytest.raises(ValueError, match="requires --metrics-source"):
        cache.prepare(base, toolchain, tmp_path / "missing", repository=repository, sources=source)
    (repository / "metrics.py").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Metrics logger differs"):
        prepare_fixture(prepared_inputs, tmp_path / "tampered")
    assert not (tmp_path / "tampered").exists()
