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


def test_changed_catalog_fails_before_any_registry_write(runtime):
    runtime["files"]["/opt/sparkring/features/capabilities.json"] += b" "
    with pytest.raises(ValueError, match="Installed metadata differs"):
        runtime["run"]()
    assert not runtime["requests"]
    assert ("rm", "f" * 64) in runtime["calls"]


def test_source_mismatch_fails_before_any_registry_write(runtime):
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


def test_transport_metadata_mismatch_is_rejected(runtime):
    runtime["catalog"]["transport_profiles"]["prepared"]["manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="transport differs"):
        module.metadata_inputs(
            runtime["receipt"],
            runtime["manifest"],
            runtime["catalog"],
            runtime["source_index"],
        )
