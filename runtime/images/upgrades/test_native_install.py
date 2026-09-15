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
