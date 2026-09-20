"""Retired hook preimages require a fully bound replacement feature catalog."""

import hashlib
import json

import pytest

from .image_gate import verify_bindings


def migration(root):
    def put(name, value):
        path = root / name.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(value).encode() if not isinstance(value, bytes) else value
        path.write_bytes(data)
        return hashlib.sha256(data).hexdigest()

    prefix = "/opt/sparkring/features/"
    source = "/opt/venv/lib/python3.12/site-packages/engine.py"
    source_hash = put(source, b"accepted source")
    put(
        prefix + "qwen-prefill/manifest.json",
        {
            "schema": "sparkring-qwen-prefill/v1",
            "image_source_preimages": {source: "0" * 64},
        },
    )
    manifest = prefix + "qwen4-prefill/manifest.json"
    digest = put(
        manifest,
        {
            "schema": "sparkring-qwen4-prefill/v1",
            "image_source_preimages": {source: source_hash},
        },
    )
    parent = "/opt/sparkring/receipts/features-parent-fixture.json"
    parent_digest = put(
        parent,
        {
            "schema": "sparkring-image-capabilities/v1",
            "features": {"qwen-prefill": {}},
        },
    )
    catalog = prefix + "capabilities.json"
    child = {
        "schema": "sparkring-image-capabilities/v1",
        "features": {
            "qwen4-prefill": {
                "directory": "qwen4-prefill",
                "files": {"qwen4-prefill/manifest.json": digest},
                "manifest_sha256": digest,
            }
        },
        "unsupported_features": {
            "qwen-prefill": {
                "reason": "Source module moved; use the bound Qwen4 implementation.",
                "replacement": "qwen4-prefill",
            }
        },
    }
    catalog_hash = put(catalog, child)
    receipt = {
        "feature_update": {
            "parent_catalog": parent,
            "parent_capabilities_sha256": parent_digest,
            "catalog": catalog,
            "catalog_sha256": catalog_hash,
            "assets": {catalog: {"sha256": catalog_hash}, manifest: {"sha256": digest}},
        },
        "active_contracts": [],
    }
    receipt_name = "/opt/sparkring/receipts/native-installed.json"
    put(receipt_name, receipt)
    return put, child, receipt, catalog, receipt_name, source


def test_bound_replacement_validates_its_source_instead_of_retired_manifest(tmp_path):
    put, _, _, _, _, source = migration(tmp_path)
    assert verify_bindings(tmp_path) == (1, [])
    put(source, b"drift")
    assert verify_bindings(tmp_path)[1] == [
        "Feature source preimage changed: " + source
    ]


@pytest.mark.parametrize("change", ["catalog", "asset", "parent", "receipt"])
def test_unbound_feature_migration_fails_closed(tmp_path, change):
    put, _, receipt, catalog, receipt_name, _ = migration(tmp_path)
    name = {
        "catalog": catalog,
        "asset": "/opt/sparkring/features/qwen4-prefill/manifest.json",
        "parent": receipt["feature_update"]["parent_catalog"],
        "receipt": receipt_name,
    }[change]
    if change == "receipt":
        receipt["feature_update"]["assets"] = {}
        put(name, receipt)
    else:
        put(name, b"drift")
    assert verify_bindings(tmp_path)[1]


def test_even_bound_catalog_cannot_silently_remove_parent_feature(tmp_path):
    put, child, receipt, catalog, receipt_name, _ = migration(tmp_path)
    child["unsupported_features"] = {}
    digest = put(catalog, child)
    receipt["feature_update"]["catalog_sha256"] = digest
    receipt["feature_update"]["assets"][catalog]["sha256"] = digest
    put(receipt_name, receipt)
    assert "replacement" in verify_bindings(tmp_path)[1][0]


def prepared_transport(root, *, source_payload=b"accepted transport source"):
    put, _, receipt, _, receipt_name, _ = migration(root)
    source = "/opt/venv/lib/python3.12/site-packages/b12x/preparation/session.py"
    source_hash = put(source, source_payload)
    profile = "tp2-rocenante-adaptive-prepared"
    manifest = "/opt/sparkring/transports/" + profile + "/manifest.json"
    preimages = {source: source_hash}
    manifest_hash = put(
        manifest,
        {
            "schema": "sparkring-transport-bundle/v1",
            "name": profile,
            "image_source_preimages": preimages,
        },
    )
    receipt["feature_update"]["assets"][manifest] = {"sha256": manifest_hash}
    receipt["feature_update"]["transport_bundles"] = {
        profile: {
            "manifest": manifest,
            "manifest_sha256": manifest_hash,
            "image_source_preimages": preimages,
        }
    }
    put(receipt_name, receipt)
    return put, receipt, receipt_name, profile, manifest, source


def test_prepared_transport_accepts_receipt_owned_installed_preimages(tmp_path):
    _, _, _, _, _, source = prepared_transport(tmp_path)
    assertions, failed = verify_bindings(tmp_path)
    assert not failed and assertions == 2
    assert source.endswith("b12x/preparation/session.py")


def test_prepared_transport_rejects_installed_source_drift(tmp_path):
    put, _, _, _, _, source = prepared_transport(tmp_path)
    put(source, b"drift")
    assert "Transport source preimage changed: " + source in verify_bindings(tmp_path)[1]


def test_prepared_transport_rejects_receipt_manifest_preimage_disagreement(tmp_path):
    put, receipt, receipt_name, profile, _, _ = prepared_transport(tmp_path)
    receipt["feature_update"]["transport_bundles"][profile][
        "image_source_preimages"
    ] = {"/opt/venv/lib/python3.12/site-packages/b12x/other.py": "0" * 64}
    put(receipt_name, receipt)
    assert "Transport source preimage record differs" in verify_bindings(tmp_path)[1][0]


def test_prepared_transport_rejects_manifest_without_source_preimages(tmp_path):
    put, receipt, receipt_name, profile, manifest, _ = prepared_transport(tmp_path)
    manifest_hash = put(
        manifest,
        {"schema": "sparkring-transport-bundle/v1", "name": profile},
    )
    receipt["feature_update"]["assets"][manifest]["sha256"] = manifest_hash
    binding = receipt["feature_update"]["transport_bundles"][profile]
    binding["manifest_sha256"] = manifest_hash
    binding.pop("image_source_preimages")
    put(receipt_name, receipt)
    assert "Prepared transport lacks source preimages" in verify_bindings(tmp_path)[1][0]
