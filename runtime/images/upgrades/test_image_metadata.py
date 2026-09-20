"""Labels-only configuration identity and immutable installed-metadata admission."""

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime.images.upgrades import image_metadata as module


@pytest.fixture
def golden():
    return json.loads(
        (
            Path(__file__).parent / "checks/image-metadata-labels-a75bd02f.json"
        ).read_bytes()
    )


@pytest.fixture
def metadata(golden):
    return golden["metadata"]


def test_labels_match_recorded_arm64_configuration(golden):
    assert (
        module.labels_for_parent(
            golden["parent_labels"], golden["parent_image_id"], golden["metadata"]
        )
        == golden["labels"]
    )


def test_label_transform_preserves_nonlabel_fields_and_input(metadata):
    original = {
        "architecture": "arm64",
        "created": "2026-09-18T00:00:00Z",
        "config": {
            "Labels": {"fixture": "value"},
            "Env": ["A=B"],
            "Entrypoint": ["serve"],
            "User": "operator",
        },
        "history": [{"created_by": "fixture"}],
        "rootfs": {"diff_ids": ["sha256:layer"]},
        "future_field": {"retain": True},
    }
    unchanged = copy.deepcopy(original)
    candidate = json.loads(
        module.child_configuration(original, "sha256:" + "a" * 64, metadata)
    )
    assert original == unchanged
    assert module.without_labels(original) == module.without_labels(candidate)
    assert (
        candidate["config"]["Labels"]["org.sparkring.status"]
        == "implemented; profile qualification pending"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://ghcr.io/v2/sparkring/native/blobs/uploads/",
        "http://localhost:19555/v2/sparkring/native/blobs/uploads/",
        "http://127.0.0.1:19556/v2/sparkring/native/blobs/uploads/",
        "http://user@127.0.0.1:19555/v2/sparkring/native/blobs/uploads/",
        "/v2/another/repository/blobs/uploads/",
        "/v2/sparkring/native/../../another/blobs/",
        "/v2/sparkring/native/%2e%2e/another/blobs/",
        "/v2/sparkring/native/blobs/uploads/#fragment",
    ],
)
def test_registry_cannot_escape_owned_endpoint(url):
    with pytest.raises(ValueError, match="owned loopback repository"):
        module.registry_url(url)


def test_registry_upload_location_and_query_stay_local():
    path = "/v2/sparkring/native/blobs/uploads/owned?digest=sha256%3Aabc"
    assert module.registry_url(path) == module.BASE + path


def test_registry_redirects_are_rejected():
    with pytest.raises(ValueError, match="redirects"):
        module.NoRedirects().redirect_request(
            None, None, 307, "", {}, "https://ghcr.io/"
        )


def test_checks_are_not_disabled_by_python_optimization():
    code = "from runtime.images.upgrades.image_metadata import require; require(False, 'closed')"
    result = subprocess.run([sys.executable, "-O", "-c", code], capture_output=True)
    assert result.returncode != 0 and b"ValueError: closed" in result.stderr


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    manifest = {
        "schema": "sparkring-release-source-inputs/v1",
        "release_candidate": "shared-fixture",
        "components": [
            {
                "id": name,
                "upstream_commit": value * 40,
                "accepted_snapshot_sha256": value * 64,
            }
            for name, value in (("vllm", "a"), ("b12x", "b"))
        ],
    }
    transport_raw = module.encoded({"schema": "fixture-transport/v1"})
    transport_hash = module.sha(transport_raw)[7:]
    catalog = {
        "features": {"fixture": {}},
        "transport_profiles": {"prepared": {"manifest_sha256": transport_hash}},
    }
    catalog_raw = module.encoded(catalog)
    source_index = {
        "version": "shared-fixture",
        "sources": manifest,
        "transport_profile": "prepared",
        "transport_manifest_sha256": transport_hash,
    }
    image_files = {
        "/opt/sparkring/features/capabilities.json": catalog_raw,
        "/opt/sparkring/releases/shared/shared-fixture.json": module.encoded(
            source_index
        ),
        "/opt/sparkring/licenses/components.md": b"Component licenses remain separate.\n",
        "/opt/sparkring/transports/prepared/manifest.json": transport_raw,
    }
    receipt = {
        "compiler": {"source_trees": {"vllm": "a" * 64, "b12x": "b" * 64}},
        "feature_update": {
            "catalog": "/opt/sparkring/features/capabilities.json",
            "catalog_sha256": module.sha(catalog_raw)[7:],
            "descriptor_sha256": "d" * 64,
            "transport_bundles": {
                "prepared": {
                    "manifest": "/opt/sparkring/transports/prepared/manifest.json",
                    "manifest_sha256": transport_hash,
                }
            },
        },
        "files": {path: module.sha(data)[7:] for path, data in image_files.items()},
    }
    original = {
        "architecture": "arm64",
        "os": "linux",
        "config": {"Labels": {"fixture": "parent"}, "Env": ["KEEP=1"]},
        "rootfs": {"type": "layers", "diff_ids": ["sha256:" + "1" * 64]},
        "history": [{"created_by": "fixture"}],
    }
    original_raw = module.encoded(original)
    parent = module.sha(original_raw)
    before = {
        "Id": parent,
        "Os": "linux",
        "Architecture": "arm64",
        "Config": copy.deepcopy(original["config"]),
        "RootFS": {"Type": "layers", "Layers": original["rootfs"]["diff_ids"]},
    }
    manifest_path = tmp_path / "sources.json"
    manifest_path.write_bytes(module.encoded(manifest))
    output = tmp_path / "metadata"
    state = {
        "calls": [],
        "requests": [],
        "receipt": receipt,
        "files": image_files,
        "upload": None,
        "before": before,
        "after_change": None,
        "original": original,
        "parent": parent,
        "manifest": manifest,
        "catalog": catalog,
        "source_index": source_index,
        "output": output,
    }

    def docker(*args):
        state["calls"].append(args)
        if args[:2] == ("image", "inspect"):
            if args[2] == parent:
                return module.encoded([before])
            after = copy.deepcopy(before)
            after["Id"] = module.sha(state["upload"])
            after["Config"] = json.loads(state["upload"])["config"]
            if state["after_change"]:
                state["after_change"](after)
            return module.encoded([after])
        if args[0] == "create":
            return ("f" * 64).encode()
        if args[0] == "cp":
            path = args[1].split(":", 1)[1]
            if path != module.RECEIPT and path not in image_files:
                raise subprocess.CalledProcessError(1, ["docker", *args])
            data = (
                module.encoded(state["receipt"])
                if path == module.RECEIPT
                else image_files[path]
            )
            Path(args[2]).write_bytes(data)
            return b""
        if args[0] in ("rm", "pull"):
            return b"ok"
        pytest.fail("Unexpected Docker operation: " + args[0])

    def request(path, *, method="GET", data=None, content_type=None):
        module.registry_url(path)
        state["requests"].append((method, path))
        if method == "GET" and "/manifests/" in path:
            return (
                200,
                module.encoded(
                    {
                        "config": {"digest": parent, "size": len(original_raw)},
                        "mediaType": "application/vnd.oci.image.manifest.v1+json",
                        "layers": [{"digest": "sha256:layer"}],
                    }
                ),
                {},
            )
        if method == "GET":
            return 200, original_raw, {}
        if method == "POST":
            return 202, b"", {"Location": "/v2/sparkring/native/blobs/uploads/owned"}
        if "/blobs/uploads/" in path:
            state["upload"] = data
        return 201, b"", {}

    monkeypatch.setattr(module, "docker", docker)
    monkeypatch.setattr(module, "request", request)
    state["run"] = lambda: module.main(
        [
            "--parent-image",
            parent,
            "--source-manifest",
            str(manifest_path),
            "--output",
            str(output),
        ]
    )
    return state


def test_operator_step_verifies_metadata_without_starting_container(runtime):
    runtime["run"]()
    proof = json.loads((runtime["output"] / "equivalence.json").read_bytes())
    assert proof["rootfs_identical"] and proof["raw_config_except_labels_identical"]
    assert (
        proof["runtime_config_except_labels_identical"]
        and not proof["external_publication"]
    )
    assert (
        proof["metadata_inputs"]["derived"]["components"]["b12x"][
            "accepted_snapshot_sha256"
        ]
        == "b" * 64
    )
    assert not any(
        call[0] in ("start", "run", "exec", "build", "push")
        for call in runtime["calls"]
    )
    assert ("rm", "f" * 64) in runtime["calls"]
    assert module.without_labels(
        json.loads(runtime["upload"])
    ) == module.without_labels(runtime["original"])


def test_metadata_staging_refuses_unpullable_parent_before_registry_writes(runtime):
    runtime["before"]["RootFS"]["Layers"] = [
        "sha256:" + f"{index:064x}"
        for index in range(module.MAX_PULLABLE_ROOTFS_LAYERS + 1)
    ]
    with pytest.raises(ValueError, match="flatten it before metadata staging"):
        runtime["run"]()
    assert not runtime["requests"]
    assert not any(call[0] == "create" for call in runtime["calls"])


@pytest.fixture
def versioned(runtime):
    path = "/opt/sparkring/licenses/shared/shared-fixture.md"
    payload = b"Version-specific component licenses.\n"
    digest = module.sha(payload)[7:]
    runtime["files"][path] = payload
    runtime["receipt"]["files"][path] = digest
    runtime["source_index"]["component_license_index"] = {
        "schema": "sparkring-versioned-component-license-index/v1",
        "sha256": digest,
    }
    index = "/opt/sparkring/releases/shared/shared-fixture.json"
    runtime["files"][index] = module.encoded(runtime["source_index"])
    runtime["receipt"]["files"][index] = module.sha(runtime["files"][index])[7:]
    runtime["license_path"] = path
    return runtime


def test_versioned_license_index_is_verified_and_labeled_without_legacy_changes(
    versioned,
):
    original_files = copy.deepcopy(versioned["files"])
    versioned["run"]()
    proof = json.loads((versioned["output"] / "equivalence.json").read_bytes())
    labels = json.loads(versioned["upload"])["config"]["Labels"]
    path = versioned["license_path"]
    assert labels["org.sparkring.component-license-index"] == path
    assert "org.opencontainers.image.licenses" not in labels
    assert proof["metadata_inputs"]["derived"]["component_license_index"] == {
        "path": path,
        "sha256": module.sha(original_files[path])[7:],
    }
    verified = proof["metadata_inputs"]["verified_installed_files"]
    assert path in verified
    assert "/opt/sparkring/licenses/components.md" not in verified
    assert versioned["files"] == original_files


def test_versioned_labels_replace_inherited_source_index_hash(versioned):
    metadata = module.metadata_inputs(
        versioned["receipt"],
        versioned["manifest"],
        versioned["catalog"],
        versioned["source_index"],
    )
    labels = module.labels_for_parent(
        {
            "org.sparkring.source-index-sha256": "e" * 64,
            "org.opencontainers.image.licenses": "Apache-2.0",
        },
        versioned["parent"],
        metadata,
    )
    index = "/opt/sparkring/releases/shared/shared-fixture.json"
    assert (
        labels["org.sparkring.source-index-sha256"]
        == versioned["receipt"]["files"][index]
    )
    assert "org.opencontainers.image.licenses" not in labels


@pytest.mark.parametrize(
    "binding",
    [
        None,
        {},
        {"schema": "unknown/v1", "sha256": "a" * 64},
        {"schema": "sparkring-versioned-component-license-index/v1"},
        {"schema": "sparkring-versioned-component-license-index/v1", "sha256": None},
        {"schema": "sparkring-versioned-component-license-index/v1", "sha256": "bad"},
        {
            "schema": "sparkring-versioned-component-license-index/v1",
            "sha256": "a" * 64,
        },
        {
            "schema": "sparkring-versioned-component-license-index/v1",
            "sha256": "a" * 64,
            "path": "/opt/sparkring/licenses/components.md",
        },
    ],
)
def test_invalid_versioned_binding_cannot_fall_back_to_legacy(versioned, binding):
    versioned["source_index"]["component_license_index"] = binding
    index = "/opt/sparkring/releases/shared/shared-fixture.json"
    versioned["files"][index] = module.encoded(versioned["source_index"])
    versioned["receipt"]["files"][index] = module.sha(versioned["files"][index])[7:]
    with pytest.raises(ValueError, match="[Cc]omponent license index"):
        versioned["run"]()
    assert not versioned["requests"]


@pytest.mark.parametrize("defect", ["bytes", "missing-file", "missing-receipt"])
def test_versioned_license_requires_installed_bytes_and_receipt(versioned, defect):
    path = versioned["license_path"]
    if defect == "bytes":
        versioned["files"][path] += b"tampered"
    elif defect == "missing-file":
        del versioned["files"][path]
    else:
        del versioned["receipt"]["files"][path]
    with pytest.raises((ValueError, subprocess.CalledProcessError)):
        versioned["run"]()
    assert not versioned["requests"]
    assert not any(
        call[0] == "cp" and call[1].endswith(":/opt/sparkring/licenses/components.md")
        for call in versioned["calls"]
    )


@pytest.mark.parametrize("release", ["../escape", ".hidden", "x" * 97])
def test_versioned_binding_rejects_release_names_outside_fresh_owner(
    versioned, release
):
    versioned["manifest"]["release_candidate"] = release
    versioned["source_index"]["version"] = release
    with pytest.raises(ValueError, match="release identifier"):
        module.metadata_inputs(
            versioned["receipt"],
            versioned["manifest"],
            versioned["catalog"],
            versioned["source_index"],
        )


def test_legacy_top_level_schema_keeps_fixed_license_convention(runtime):
    runtime["source_index"]["schema"] = "retained-legacy-schema/v1"
    index = "/opt/sparkring/releases/shared/shared-fixture.json"
    runtime["files"][index] = module.encoded(runtime["source_index"])
    runtime["receipt"]["files"][index] = module.sha(runtime["files"][index])[7:]
    runtime["run"]()
    labels = json.loads(runtime["upload"])["config"]["Labels"]
    assert (
        labels["org.sparkring.component-license-index"]
        == "/opt/sparkring/licenses/components.md"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/opt/sparkring/releases/shared/shared-fixture.json",
        "/opt/sparkring/transports/prepared/manifest.json",
    ],
)
def test_versioned_source_and_transport_bytes_are_bound_before_labeling(
    versioned, path
):
    versioned["files"][path] += b"tampered"
    with pytest.raises(ValueError, match="Installed metadata differs"):
        versioned["run"]()
    assert not versioned["requests"]


def test_changed_catalog_fails_before_any_registry_write(runtime):
    runtime["files"]["/opt/sparkring/features/capabilities.json"] += b" "
    with pytest.raises(ValueError, match="Installed metadata differs"):
        runtime["run"]()
    assert not runtime["requests"]
    assert ("rm", "f" * 64) in runtime["calls"]


@pytest.mark.parametrize("fixture_name", ["runtime", "versioned"])
def test_source_mismatch_fails_before_any_registry_write(request, fixture_name):
    runtime = request.getfixturevalue(fixture_name)
    runtime["receipt"]["compiler"]["source_trees"]["b12x"] = "0" * 64
    with pytest.raises(ValueError, match="Source snapshot differs"):
        runtime["run"]()
    assert not runtime["requests"]


@pytest.mark.parametrize(
    "change",
    [
        lambda after: after["RootFS"].update(Layers=["different"]),
        lambda after: after["Config"].update(Env=["CHANGED=1"]),
        lambda after: after.update(Architecture="amd64"),
    ],
)
def test_changed_runtime_cannot_produce_equivalence_proof(runtime, change):
    runtime["after_change"] = change
    with pytest.raises(ValueError):
        runtime["run"]()
    assert not (runtime["output"] / "equivalence.json").exists()


@pytest.mark.parametrize("fixture_name", ["runtime", "versioned"])
def test_transport_metadata_mismatch_is_rejected(request, fixture_name):
    runtime = request.getfixturevalue(fixture_name)
    runtime["catalog"]["transport_profiles"]["prepared"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="transport differs"):
        module.metadata_inputs(
            runtime["receipt"],
            runtime["manifest"],
            runtime["catalog"],
            runtime["source_index"],
        )
