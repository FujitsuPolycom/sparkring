"""Native image attestation and compiler-policy checks without a container."""

import json

import pytest

from . import build_native, native_install


def recipe():
    return dict(
        schema="sparkring-native-recipe/v1",
        architecture="12.1a",
        jobs=8,
        cpus=12,
        memory_bytes=80 * 1024**3,
        build_seconds=21600,
        torch_version="2.13.0",
        build_type="Release",
        network="bridge",
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("architecture", "8.0"),
        ("jobs", True),
        ("jobs", 21),
        ("cpus", 2),
        ("memory_bytes", 128 * 1024**3),
        ("build_seconds", 0),
        ("network", "host"),
        ("torch_version", None),
    ],
)
def test_native_recipe_rejects_unbounded_or_incompatible_settings(field, value):
    value_dict = recipe()
    value_dict[field] = value
    with pytest.raises(ValueError):
        build_native.validate_recipe(value_dict)


def test_explicit_native_recipe_is_admitted():
    assert build_native.validate_recipe(recipe()) == recipe()


@pytest.mark.parametrize("tamper", [False, True])
def test_install_source_binding_requires_matching_installed_bytes(
    tmp_path, monkeypatch, tamper
):
    root, site, context = tmp_path / "runtime", tmp_path / "site", tmp_path / "context"
    (root / "contracts").mkdir(parents=True)
    (root / "receipts").mkdir()
    (site / "vllm").mkdir(parents=True)
    context.mkdir()
    source = site / "vllm/module.py"
    source.write_text("VALUE = 1\n")
    contract = {
        "schema": "sparkring-vllm-kv-block-lease-contract/v1",
        "files": [{"path": "vllm/module.py", "sha256": native_install.sha(source)}],
    }
    proof = {
        "schema": "sparkring-binding-equivalence/v1",
        "candidate_tree_sha256": "b" * 64,
        "serving_qualified": False,
        "oracle": {
            "input_sha256": "a" * 64,
            "subject_sha256": "b" * 64,
            "variant": "candidate",
            "outcome": "passed",
            "assertions": 1,
            "skipped": 0,
        },
    }
    (context / "binding.json").write_text(json.dumps(contract))
    (context / "proof.json").write_text(json.dumps(proof))
    destination = (
        root / "contracts" / ("vllm-connector-jobs-source-" + "b" * 16 + ".json")
    )
    descriptor = {
        "input_sha256": "a" * 64,
        "source_binding": {
            "file": "binding.json",
            "proof_file": "proof.json",
            "sha256": native_install.sha(context / "binding.json"),
            "proof_sha256": native_install.sha(context / "proof.json"),
            "destination": str(destination),
        },
    }
    monkeypatch.setattr(native_install, "ROOT", root)
    monkeypatch.setattr(native_install, "SITE", site)
    if tamper:
        source.write_text("VALUE = 2\n")
        with pytest.raises(ValueError, match="Installed source differs"):
            native_install.install_source_binding(
                context, descriptor, {"source_trees": {"vllm": "b" * 64}}
            )
        assert not destination.exists()
    else:
        result = native_install.install_source_binding(
            context, descriptor, {"source_trees": {"vllm": "b" * 64}}
        )
        assert str(destination) in result and len(result) == 2


def test_versioned_boundary_selector_requires_owned_unchanged_identity(
    tmp_path, monkeypatch
):
    path = tmp_path / "contracts/boundary-cache-example.json"
    path.parent.mkdir()
    path.write_text("{}")
    digest = native_install.sha(path)
    parent = {
        "files": {str(path): digest},
        "cache_extension": {"boundary_runtime": {"path": str(path), "sha256": digest}},
    }
    monkeypatch.setattr(native_install, "ROOT", tmp_path)
    assert native_install.selected_boundary_identity(parent) == path
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match="differs"):
        native_install.selected_boundary_identity(parent)
    assert (
        native_install.selected_boundary_identity({})
        == tmp_path / "contracts/boundary-runtime.json"
    )


def test_native_image_checks_inventory_versions_and_feature_files(
    tmp_path, monkeypatch
):
    root = tmp_path / "sparkring"
    root.mkdir()
    library = root / "runtime.so"
    library.write_bytes(b"fixture bytes")
    manifest = root / "features/capabilities.json"
    manifest.parent.mkdir()
    feature = manifest.parent / "selector.py"
    feature.write_bytes(b"selected = True\n")
    manifest.write_text(
        json.dumps(
            dict(
                schema="sparkring-image-capabilities/v1",
                features={
                    "collectives": {
                        "files": {"selector.py": native_install.sha(feature)}
                    }
                },
            )
        )
    )
    receipt = root / "native.json"
    receipt.write_text(
        json.dumps(
            dict(
                schema="sparkring-native-installed/v1",
                input_sha256="a" * 64,
                compiler={"source_trees": {"vllm": "b" * 64, "b12x": "c" * 64}},
                files={
                    str(library): native_install.sha(library),
                    str(manifest): native_install.sha(manifest),
                },
                removed_files=[],
                versions={"torch": "2.13.0"},
            )
        )
    )
    monkeypatch.setattr(native_install, "ROOT", root)
    monkeypatch.setattr(native_install, "NATIVE", receipt)
    monkeypatch.setattr(native_install.metadata, "version", lambda name: "2.13.0")
    result = native_install.verify()
    assert result["serving_qualified"] is False and result["features"] == [
        "collectives"
    ]
    feature.write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload differs"):
        native_install.verify()
    feature.write_bytes(b"selected = True\n")
    monkeypatch.setattr(native_install.metadata, "version", lambda name: "2.14.0")
    with pytest.raises(ValueError, match="version differs"):
        native_install.verify()


def test_distribution_ownership_does_not_include_other_packages(tmp_path, monkeypatch):
    site = tmp_path / "site"
    monkeypatch.setattr(native_install, "SITE", site)
    assert native_install.package_owned(
        site / "vllm/kernel.py", ["vllm", "b12x"], [site / "vllm-1.dist-info"]
    )
    assert not native_install.package_owned(
        site / "torch/kernel.py", ["vllm", "b12x"], [site / "vllm-1.dist-info"]
    )
    assert not native_install.package_owned(
        tmp_path / "credentials", ["vllm", "b12x"], []
    )
