"""Published source selection binds registry identity without local test flags."""

import copy
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from runtime.common import compose, qwen_flash_next as adapter, source_candidate as source
from runtime.common.container_spec import docker_create
from runtime.common.test_compose import compose_cli as compose_cli
from scripts import sparkring_compose as coordinator

PROFILE = "qwen38-flash-next-qad-tp4"
IMAGE_ID = "sha256:" + "c" * 64
REFERENCE = "example.invalid/sparkring@sha256:" + "d" * 64


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def publication(tmp_path, monkeypatch):
    descriptor = tmp_path / "descriptor.json"
    descriptor.write_bytes(source.DESCRIPTOR.read_bytes())
    monkeypatch.setattr(source, "DESCRIPTOR", descriptor)
    value = {
        "schema": "sparkring-image-publication/v1", "image_id": IMAGE_ID,
        "image_reference": REFERENCE, "platform": "linux/arm64",
        "descriptor_sha256": file_hash(descriptor), "anonymous_pull_verified": True,
    }
    write_json(descriptor.with_name("publication.json"), value)
    return value


def release_record(publication):
    prefix = "runtime/images/compositions/" + source.IDENTITY + "/"
    return {
        "schema": "sparkring-release-selection/v1", "id": "fixture-source-registry",
        "selection": "registered-source-extension-image", "image": publication["image_reference"],
        "inputs": [
            {"path": prefix + name, "sha256": file_hash(source.DESCRIPTOR.with_name(name))}
            for name in ("publication.json", "descriptor.json")
        ],
    }


def test_publication_binds_digest_image_and_source_descriptor(publication):
    assert source.publication(image_id=IMAGE_ID) == publication
    assert source.release_publication(release_record(publication)) == publication


@pytest.mark.parametrize("change", [
    {"schema": "unreviewed"}, {"image_id": "sparkring:tag"}, {"image_reference": "sparkring:tag"},
    {"platform": "linux/amd64"}, {"descriptor_sha256": "0" * 64},
    {"anonymous_pull_verified": False}, {"anonymous_pull_verified": 1},
])
def test_unbound_publication_cannot_select_an_image(publication, change):
    publication.update(change)
    write_json(source.DESCRIPTOR.with_name("publication.json"), publication)
    with pytest.raises(ValueError, match="Source publication"):
        source.publication()


def test_source_publication_is_required_before_public_selection(publication):
    source.DESCRIPTOR.with_name("publication.json").unlink()
    with pytest.raises(ValueError, match="no registered publication"):
        source.publication()


def test_exact_published_configuration_id_is_required_before_docker_access(publication):
    def forbidden(*args, **kwargs):
        pytest.fail("Unregistered image must be rejected before Docker access")
    with pytest.raises(ValueError, match="registered source publication"):
        adapter.verify_image("sha256:" + "e" * 64, source_extension=source.IDENTITY, run=forbidden)


@pytest.mark.parametrize("mutation", [
    lambda record: record.update(image="example.invalid/sparkring@sha256:" + "e" * 64),
    lambda record: record.update(selection="registered-shared-feature-image"),
    lambda record: record["inputs"][0].update(sha256="0" * 64),
    lambda record: record["inputs"][1].update(sha256="0" * 64),
    lambda record: record["inputs"].pop(),
    lambda record: record["inputs"].append(copy.deepcopy(record["inputs"][0])),
    lambda record: record["inputs"][0].update(path="runtime/images/other/publication.json"),
])
def test_release_kind_and_input_hashes_cannot_drift(publication, mutation):
    record = release_record(publication)
    mutation(record)
    with pytest.raises(ValueError, match="Source release"):
        source.release_publication(record)


@pytest.fixture
def published_repository(tmp_path, monkeypatch):
    """Build a tiny fixture tree; no public records are written into the checkout."""
    original_root = compose.ROOT
    original_load = compose.profiles.load
    metadata = {identity: original_load(identity)[0] for identity in (PROFILE, PROFILE + "-sparkcache")}
    names = set(compose.source_inventory(PROFILE)) | set(compose.source_inventory(PROFILE + "-sparkcache"))
    names.add("runtime/common/source_candidate.py")
    prefix = Path("runtime/images/compositions") / source.IDENTITY
    names.update(path.relative_to(original_root).as_posix() for path in (original_root / prefix).glob("*.json"))
    for name in names:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original_root / name, path)
    monkeypatch.setattr(compose, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "ROOT", tmp_path)
    monkeypatch.setattr(adapter, "CONFIG_ROOT", tmp_path / "profiles/qwen38-flash-next-tp2")
    monkeypatch.setattr(adapter, "TP4_CONFIG", tmp_path / "profiles" / PROFILE / "config.json")
    monkeypatch.setattr(adapter, "TP4_CACHE_CONFIG", tmp_path / "profiles" / PROFILE / "sparkcache.json")
    monkeypatch.setattr(source, "DESCRIPTOR", tmp_path / prefix / "descriptor.json")
    value = {
        "schema": "sparkring-image-publication/v1", "image_id": IMAGE_ID,
        "image_reference": REFERENCE, "platform": "linux/arm64",
        "descriptor_sha256": file_hash(source.DESCRIPTOR), "anonymous_pull_verified": True,
    }
    write_json(source.DESCRIPTOR.with_name("publication.json"), value)
    release = release_record(value)
    for item in metadata.values():
        write_json(tmp_path / item["release"], release)
        path = tmp_path / item["configuration"]["path"]
        profile = adapter.read(path)
        profile["image_extension"] = source.IDENTITY
        profile["environment"].update(VLLM_QWEN3_8_HC_PREFILL_MODE="shard", VLLM_QWEN3_8_PREFILL_COALESCE="1")
        arguments = profile["vllm_args"]
        if "--kv-transfer-config" in arguments:
            index = arguments.index("--kv-transfer-config") + 1
            transfer = json.loads(arguments[index])
            transfer["kv_connector_extra_config"]["spark_cache_async_page_capture_lease_contract"] = source.LEASE_CONTRACT
            transfer["kv_connector_extra_config"]["spark_cache_root"] = "/cache/persistent/fixture-source"
            arguments[index] = json.dumps(transfer, separators=(",", ":"))
        write_json(path, profile)
    monkeypatch.setattr(compose.profiles, "load", lambda identity: (metadata[identity], release) if identity in metadata else original_load(identity))
    return compose.read_site(original_root / f"profiles/{PROFILE}/compose/site.example.yaml")


@pytest.mark.parametrize("profile_id", [PROFILE, PROFILE + "-sparkcache"])
def test_ordinary_render_selects_published_source_image_without_local_flags(published_repository, profile_id):
    specs, image = compose.specifications(profile_id, published_repository)
    manifest, files = compose.build(profile_id, published_repository)
    assert image == REFERENCE
    assert not compose.selection_options(manifest)
    prefix = f"runtime/images/compositions/{source.IDENTITY}/"
    assert prefix + "publication.json" in manifest["inputs"]
    assert prefix + "descriptor.json" in manifest["inputs"]
    assert "runtime/common/source_candidate.py" in manifest["inputs"]
    for rank, spec in enumerate(specs):
        assert spec.image_id == IMAGE_ID and spec.command[0] == source.ENTRYPOINT
        assert spec.environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "shard"
        assert spec.environment["VLLM_QWEN3_8_PREFILL_COALESCE"] == "1"
        assert spec.command[spec.command.index("--kv-cache-memory-bytes") + 1] == str(24 * 1024 ** 3)
        assert spec.command[spec.command.index("--master-port") + 1] == "29779"
        argv = docker_create(spec)
        assert argv[argv.index(IMAGE_ID) + 1:] == list(spec.command)
        assert IMAGE_ID in files[f"rank{rank}/container.json"]


def test_public_source_defaults_are_taken_from_profile_not_implicit_feature_activation(published_repository):
    profile = adapter.read(adapter.TP4_CONFIG)
    profile["environment"].update(VLLM_QWEN3_8_HC_PREFILL_MODE="off", VLLM_QWEN3_8_PREFILL_COALESCE="0")
    write_json(adapter.TP4_CONFIG, profile)
    specs, _ = compose.specifications(PROFILE, published_repository)
    assert specs[0].environment["VLLM_QWEN3_8_HC_PREFILL_MODE"] == "off"
    assert specs[0].environment["VLLM_QWEN3_8_PREFILL_COALESCE"] == "0"


def test_public_source_compose_equivalence(published_repository, compose_cli):
    specs, image = compose.specifications(PROFILE + "-sparkcache", published_repository)
    for spec in specs:
        compose.check_equivalence(spec, image, compose.compose_text(spec, image))


def test_source_release_default_cannot_be_replaced_with_local_image_id(published_repository):
    with pytest.raises(ValueError, match="cannot be overridden"):
        compose.specifications(PROFILE, published_repository, local_image_id="sha256:" + "a" * 64)


def test_source_cache_profile_cannot_reuse_the_parent_scheduler_contract(published_repository):
    profile = adapter.read(adapter.TP4_CACHE_CONFIG)
    args = profile["vllm_args"]
    index = args.index("--kv-transfer-config") + 1
    transfer = json.loads(args[index])
    transfer["kv_connector_extra_config"]["spark_cache_async_page_capture_lease_contract"] = "/opt/sparkring/contracts/vllm-connector-jobs-lil-r37.json"
    args[index] = json.dumps(transfer)
    write_json(adapter.TP4_CACHE_CONFIG, profile)
    with pytest.raises(ValueError, match="packaged lease contract"):
        compose.specifications(PROFILE + "-sparkcache", published_repository)


def test_ordinary_coordinator_selects_source_admission(published_repository, tmp_path, monkeypatch):
    manifest, files = compose.build(PROFILE + "-sparkcache", published_repository)
    stage = tmp_path / "staged"
    stage.mkdir()
    for name in ("compose.yaml", "container.json"):
        (stage / name).write_text(files["rank0/" + name], encoding="utf-8", newline="\n")
    (stage / "deployment.json").write_text(compose.encoded(manifest), encoding="utf-8", newline="\n")
    monkeypatch.setattr(coordinator, "ROOT", compose.ROOT)
    monkeypatch.setattr(coordinator, "stage_path", lambda *args: stage)
    observed = []
    monkeypatch.setattr(adapter, "verify_image", lambda image, **kwargs: observed.append((image, kwargs)) or {"verified": True})
    coordinator.host_operation("admit", {"manifest": manifest, "files": files, "rank": 0})
    assert observed[0][0] == IMAGE_ID
    assert observed[0][1]["source_extension"] == source.IDENTITY
    assert "local_source_extension" not in observed[0][1]
    assert observed[0][1]["feature_enabled"] is False
