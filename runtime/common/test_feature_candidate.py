"""Feature admission rejects self-consistent changes outside the trusted chain."""

import copy
import hashlib
import json

import pytest

from runtime.common import cache_candidate, candidate, feature_candidate as feature
from runtime.images import feature_extension


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def sha(value):
    return hashlib.sha256(value).hexdigest()


def evidence(raw):
    installed = candidate._read(raw)
    return {
        "schema": "sparkring-candidate-verification/v1",
        "receipt_sha256": sha(raw),
        "files_verified": len(installed["files"]),
        "source_components": copy.deepcopy(installed["components"]),
        "serving_qualified": False,
    }


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    components = {}
    for name in ("vllm", "b12x"):
        components[name] = {
            "base_commit": "a" * 40,
            "base_tree": "b" * 40,
            "tree": "c" * 40,
            "archive": name + "-" + "c" * 40 + ".tar.gz",
            "archive_sha256": "d" * 64,
            "patch": name + "-sparkring.patch",
            "patch_sha256": "e" * 64,
            "native_comparison": {
                "reference": "f" * 40,
                "compared_tree": "c" * 40,
                "paths": ["csrc"],
                "unchanged": True,
            },
        }
    lease = "/opt/sparkring/contracts/fixture.json"
    base_descriptor = {
        "schema": "sparkring-candidate-image/v1",
        "composition_id": "fixture",
        "parent_image_id": "sha256:" + "1" * 64,
        "components": components,
        "distribution_version": "1.0+fixture",
        "integration_contracts": {lease: {"file": "fixture.json", "sha256": "4" * 64}},
    }
    files = {path: sha(path.encode()) for path in candidate.NATIVE}
    python_file = cache_candidate.SITE + "sparkcache/fixture.py"
    files.update(
        {
            candidate.ENTRYPOINT: sha(b"entrypoint"),
            lease: "4" * 64,
            python_file: sha(b"base"),
        }
    )
    base = {
        **copy.deepcopy(base_descriptor),
        "schema": "sparkring-candidate-installed/v1",
        "files": files,
        "versions": {"vllm": "1.0+fixture", "other": "retained-version"},
        "removed_authored_files": ["/opt/removed.py"],
        "inherited_metadata": {"retain": ["every", "field"]},
    }
    base_raw = encoded(base)
    cache_contract = {
        "id": "fixture-cache64",
        "source": {"commit": "a" * 40},
        "parent": {"receipt_sha256": sha(base_raw)},
        "python_files": {"sparkcache/fixture.py": sha(b"cache64")},
        "native": {
            "destination": "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so",
            "sha256": sha(b"cache64 native"),
        },
    }
    cache_path = tmp_path / "cache-descriptor.json"
    cache_path.write_bytes(encoded(cache_contract))
    monkeypatch.setattr(cache_candidate, "DESCRIPTOR", cache_path)
    native = {path: files[path] for path in candidate.NATIVE}
    monkeypatch.setattr(candidate, "composition", lambda _: (base_descriptor, native))
    parent = copy.deepcopy(base)
    parent["files"][python_file] = cache_contract["python_files"][
        "sparkcache/fixture.py"
    ]
    parent["files"][cache_contract["native"]["destination"]] = cache_contract["native"][
        "sha256"
    ]
    parent["cache_extension"] = {
        "descriptor_sha256": sha(cache_path.read_bytes()),
        "id": cache_contract["id"],
        "source": copy.deepcopy(cache_contract["source"]),
        "native": copy.deepcopy(cache_contract["native"]),
    }
    payloads = {
        feature.FEATURE_ROOT + "fixture.py": b"VALUE = 1\n",
        cache_candidate.SITE + "sparkring_features.pth": b"import fixture\n",
    }
    installer = b"fixture installer\n"
    parent_raw = encoded(parent)
    contract = {
        "schema": "sparkring-feature-extension/v1",
        "id": "fixture-features",
        "parent": {"image_id": "sha256:" + "2" * 64, "receipt_sha256": sha(parent_raw)},
        "installer_sha256": sha(installer),
        "capabilities": ["qwen-prefill", "qwen-collectives"],
        "assets": {
            path: {"text": raw.decode(), "sha256": sha(raw)}
            for path, raw in payloads.items()
        },
    }
    path = tmp_path / "feature-descriptor.json"
    path.write_bytes(encoded(contract))
    monkeypatch.setattr(feature, "DESCRIPTOR", path)
    installed = feature_extension.expected_receipt(
        parent, contract, path.read_bytes(), parent_raw, payloads, installer
    )
    raw = encoded(installed)
    return {
        "image_id": "sha256:" + "3" * 64,
        "installed_bytes": raw,
        "parent_bytes": parent_raw,
        "base_bytes": base_raw,
        "verification": evidence(raw),
        "feature_verification": {
            "schema": "sparkring-feature-verification/v1",
            "descriptor_sha256": sha(path.read_bytes()),
            "capabilities": sorted(contract["capabilities"]),
            "files_verified": len(installed["files"]),
            "serving_qualified": False,
        },
    }


def mutate_child(inputs, mutation):
    installed = candidate._read(inputs["installed_bytes"])
    mutation(installed)
    raw = encoded(installed)
    inputs["installed_bytes"] = raw
    inputs["verification"] = evidence(raw)
    inputs["feature_verification"]["files_verified"] = len(installed["files"])


def test_installer_receipt_passes_full_candidate_and_cache_admission(inputs):
    before = copy.deepcopy(inputs)
    result = feature.validate(**inputs)
    assert result["image_id"] == inputs["image_id"]
    assert result["receipt_sha256"] == sha(inputs["installed_bytes"])
    assert result["serving_qualified"] is False
    assert (
        result["feature_extension"]
        == candidate._read(inputs["installed_bytes"])["feature_extension"]
    )
    assert inputs == before


@pytest.mark.parametrize(
    "path",
    [
        feature.FEATURE_ROOT + "fixture.py",
        cache_candidate.SITE + "sparkring_features.pth",
        feature.PARENT_RECEIPT,
        feature.INSTALLED_DESCRIPTOR,
        feature.INSTALLER,
        candidate.ENTRYPOINT,
        *sorted(candidate.NATIVE),
    ],
)
@pytest.mark.parametrize("change", ["omit", "alter"])
def test_rehashed_child_cannot_omit_or_alter_any_required_file(inputs, path, change):
    def mutate(installed):
        if change == "omit":
            del installed["files"][path]
        else:
            installed["files"][path] = "0" * 64

    mutate_child(inputs, mutate)
    with pytest.raises(ValueError, match="complete inherited and added inventory"):
        feature.validate(**inputs)


def test_rehashed_child_cannot_add_unlisted_file(inputs):
    mutate_child(
        inputs,
        lambda value: value["files"].update(
            {feature.FEATURE_ROOT + "extra.py": "0" * 64}
        ),
    )
    with pytest.raises(ValueError, match="complete inherited and added inventory"):
        feature.validate(**inputs)


@pytest.mark.parametrize(
    "field",
    [
        "schema",
        "composition_id",
        "parent_image_id",
        "components",
        "versions",
        "removed_authored_files",
        "integration_contracts",
        "cache_extension",
        "inherited_metadata",
    ],
)
def test_every_inherited_metadata_field_must_remain_exact(inputs, field):
    mutate_child(inputs, lambda value: value.update({field: "changed"}))
    with pytest.raises(ValueError, match="complete inherited and added inventory"):
        feature.validate(**inputs)


def test_child_cannot_add_unlisted_metadata(inputs):
    mutate_child(inputs, lambda value: value.update(unlisted_metadata=True))
    with pytest.raises(ValueError, match="complete inherited and added inventory"):
        feature.validate(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("descriptor_sha256", "0" * 64),
        ("id", "other-features"),
        ("parent_image_id", "sha256:" + "0" * 64),
        ("parent_receipt_sha256", "0" * 64),
        ("capabilities", ["unreviewed-feature"]),
    ],
)
def test_extension_identity_cannot_be_rehashed_into_trust(inputs, field, value):
    mutate_child(
        inputs, lambda record: record["feature_extension"].update({field: value})
    )
    with pytest.raises(ValueError, match="complete inherited and added inventory"):
        feature.validate(**inputs)


@pytest.mark.parametrize("receipt", ["parent_bytes", "base_bytes"])
def test_parent_chain_requires_exact_raw_bytes(inputs, receipt):
    inputs[receipt] += b"\n"
    with pytest.raises(ValueError, match="parent receipt"):
        feature.validate(**inputs)


def test_exact_trusted_descriptor_bytes_are_bound(inputs):
    feature.DESCRIPTOR.write_bytes(feature.DESCRIPTOR.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="complete inherited and added inventory"):
        feature.validate(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("descriptor_sha256", "0" * 64),
        ("capabilities", ["unreviewed-feature"]),
        ("files_verified", 0),
        ("files_verified", True),
        ("serving_qualified", True),
        ("serving_qualified", 0),
    ],
)
def test_feature_evidence_must_match_the_complete_unqualified_payload(
    inputs, field, value
):
    inputs["feature_verification"][field] = value
    with pytest.raises(ValueError, match="Feature verification"):
        feature.validate(**inputs)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema", "other"),
        ("receipt_sha256", "0" * 64),
        ("files_verified", 0),
        ("source_components", {}),
        ("serving_qualified", True),
    ],
)
def test_candidate_evidence_is_checked_without_projecting_away_errors(
    inputs, field, value
):
    inputs["verification"][field] = value
    with pytest.raises(ValueError):
        feature.validate(**inputs)


def test_cache_descriptor_remains_an_independent_trust_boundary(inputs):
    cache_candidate.DESCRIPTOR.write_bytes(
        cache_candidate.DESCRIPTOR.read_bytes() + b"\n"
    )
    with pytest.raises(ValueError, match="trusted descriptor"):
        feature.validate(**inputs)


@pytest.mark.parametrize("change", ["native", "source", "versions"])
def test_feature_parent_pin_cannot_override_trusted_cache_contract(inputs, change):
    parent = candidate._read(inputs["parent_bytes"])
    if change == "native":
        parent["files"]["/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so"] = (
            "0" * 64
        )
    elif change == "source":
        parent["cache_extension"]["source"] = {"commit": "0" * 40}
    else:
        parent["versions"]["other"] = "changed-version"
    parent_raw = encoded(parent)
    contract = feature.descriptor()
    contract["parent"]["receipt_sha256"] = sha(parent_raw)
    feature.DESCRIPTOR.write_bytes(encoded(contract))
    payloads = {
        path: asset["text"].encode() for path, asset in contract["assets"].items()
    }
    installed = feature_extension.expected_receipt(
        parent,
        contract,
        feature.DESCRIPTOR.read_bytes(),
        parent_raw,
        payloads,
        b"fixture installer\n",
    )
    inputs.update(parent_bytes=parent_raw, installed_bytes=encoded(installed))
    inputs["verification"] = evidence(inputs["installed_bytes"])
    inputs["feature_verification"]["descriptor_sha256"] = sha(
        feature.DESCRIPTOR.read_bytes()
    )
    with pytest.raises(ValueError, match="Cache extension"):
        feature.validate(**inputs)


@pytest.mark.parametrize("field", ["installed_bytes", "parent_bytes", "base_bytes"])
def test_receipts_must_preserve_bytes(inputs, field):
    inputs[field] = inputs[field].decode()
    with pytest.raises(ValueError, match="Raw child"):
        feature.validate(**inputs)


def test_child_identity_must_be_an_exact_image_id(inputs):
    inputs["image_id"] = "image:mutable"
    with pytest.raises(ValueError, match="Exact candidate image ID"):
        feature.validate(**inputs)
