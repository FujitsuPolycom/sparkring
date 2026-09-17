"""Source image admission preserves the entire trusted feature/cache ancestry."""

import copy
import json
import subprocess

import pytest

from runtime.common import candidate, feature_candidate, qwen_flash_next, source_candidate as source
from runtime.common.test_feature_candidate import encoded, inputs as feature_inputs, sha
from runtime.images import source_extension


@pytest.fixture
def inputs(feature_inputs, tmp_path, monkeypatch):
    parent_raw = feature_inputs["installed_bytes"]
    parent = candidate._read(parent_raw)
    changed = "vllm/fixture.py"
    # Use the inherited entrypoint as a complete feature-preservation check;
    # the source extension can only add or replace Python package files.
    payloads = {source.SITE + changed: b"VALUE = 3\n", source.LEASE_CONTRACT: b'{"schema":"test"}\n'}
    installer, patch = b"source fixture installer\n", b"fixture patch\n"
    contract = {
        "schema": "sparkring-source-extension/v1", "id": source.IDENTITY,
        "parent": {"image_id": feature_inputs["image_id"], "receipt_sha256": sha(parent_raw)},
        "installer_sha256": sha(installer), "provenance": {"source": "fixture"},
        "sources": {changed: {"sha256": sha(payloads[source.SITE + changed]), "parent_sha256": None}},
        "patch": {"source": "fixture.patch", "sha256": sha(patch)},
        "integration_contracts": {source.LEASE_CONTRACT: {"source": "fixture.json", "sha256": sha(payloads[source.LEASE_CONTRACT])}},
    }
    path = tmp_path / "source-descriptor.json"
    path.write_bytes(encoded(contract))
    monkeypatch.setattr(source, "DESCRIPTOR", path)
    child = source_extension.expected_receipt(parent, contract, path.read_bytes(), parent_raw,
                                              payloads, installer, patch)
    raw = encoded(child)
    return {
        "image_id": "sha256:" + "4" * 64, "installed_bytes": raw,
        "parent_bytes": parent_raw, "cache_parent_bytes": feature_inputs["parent_bytes"],
        "base_bytes": feature_inputs["base_bytes"],
        "verification": {"schema": "sparkring-source-verification/v1", "descriptor_sha256": sha(path.read_bytes()),
                         "receipt_sha256": sha(raw), "files_verified": len(child["files"]), "serving_qualified": False},
    }


def alter(inputs, mutation):
    child = candidate._read(inputs["installed_bytes"])
    mutation(child)
    raw = encoded(child)
    inputs["installed_bytes"] = raw
    inputs["verification"].update(receipt_sha256=sha(raw), files_verified=len(child["files"]))


def test_installer_receipt_passes_independent_complete_ancestry_check(inputs):
    original = copy.deepcopy(inputs)
    accepted = source.validate(**inputs)
    assert accepted["source_extension"]["id"] == source.IDENTITY
    assert accepted["image_id"] == inputs["image_id"]
    assert accepted["serving_qualified"] is False
    assert accepted["feature_extension"] == candidate._read(inputs["parent_bytes"])["feature_extension"]
    assert inputs == original


@pytest.mark.parametrize("path", [
    candidate.ENTRYPOINT, source.ENTRYPOINT, source.PATCH, source.PARENT_RECEIPT,
    source.INSTALLED_DESCRIPTOR, source.LEASE_CONTRACT,
    feature_candidate.PARENT_RECEIPT, feature_candidate.INSTALLER,
    *sorted(candidate.NATIVE), source.SITE + "vllm/fixture.py",
])
@pytest.mark.parametrize("operation", ["change", "remove"])
def test_self_consistent_untrusted_file_mutation_fails(inputs, path, operation):
    def change(child):
        if operation == "change":
            child["files"][path] = "0" * 64
        else:
            child["files"].pop(path)
    alter(inputs, change)
    with pytest.raises(ValueError, match="complete reviewed"):
        source.validate(**inputs)


@pytest.mark.parametrize("mutation", [
    lambda child: child.pop("feature_extension"),
    lambda child: child["feature_extension"].update(capabilities=["unreviewed"]),
    lambda child: child["versions"].update(vllm="different"),
    lambda child: child["source_extension"].update(qualification="Serving qualified"),
    lambda child: child["files"].update({source.SITE + "unreviewed.py": "a" * 64}),
])
def test_parent_metadata_and_unlisted_additions_are_preserved(inputs, mutation):
    alter(inputs, mutation)
    with pytest.raises(ValueError, match="complete reviewed"):
        source.validate(**inputs)


@pytest.mark.parametrize("field,value", [
    ("descriptor_sha256", "0" * 64), ("receipt_sha256", "0" * 64),
    ("files_verified", True), ("serving_qualified", True), ("serving_qualified", 0),
])
def test_verifier_output_must_bind_exact_child(inputs, field, value):
    inputs["verification"][field] = value
    with pytest.raises(ValueError, match="Source verification"):
        source.validate(**inputs)


def test_parent_raw_bytes_are_pinned(inputs):
    inputs["parent_bytes"] += b"\n"
    with pytest.raises(ValueError, match="parent receipt"):
        source.validate(**inputs)


def test_cache_ancestry_is_not_replaced_by_source_receipt_claim(inputs):
    inputs["cache_parent_bytes"] += b"\n"
    with pytest.raises(ValueError, match="parent receipt"):
        source.validate(**inputs)


def test_declared_replacement_requires_its_actual_parent_preimage(inputs):
    contract = source.descriptor()
    contract["sources"]["vllm/fixture.py"]["parent_sha256"] = "b" * 64
    source.DESCRIPTOR.write_bytes(encoded(contract))
    with pytest.raises(ValueError, match="preimage"):
        source.validate(**inputs)


@pytest.mark.parametrize("published", [False, True])
def test_live_admission_reads_receipts_and_source_verifier_from_inspected_image(inputs, monkeypatch, published):
    image = inputs["image_id"]
    base_image = qwen_flash_next.publication()["image_id"]
    reads = {
        (image, "/opt/sparkring/receipts/candidate-installed.json"): inputs["installed_bytes"],
        (image, source.PARENT_RECEIPT): inputs["parent_bytes"],
        (image, feature_candidate.PARENT_RECEIPT): inputs["cache_parent_bytes"],
        (base_image, "/opt/sparkring/receipts/candidate-installed.json"): inputs["base_bytes"],
    }
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[1:3] == ["image", "inspect"]:
            assert argv[-1] == image
            out = json.dumps([{"Id": image, "Os": "linux", "Architecture": "arm64",
                               "Config": {"Entrypoint": ["/opt/venv/bin/python", source.ENTRYPOINT]}}])
        else:
            assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
            assert "--gpus" not in argv and "--mount" not in argv
            if argv[-1] == "verify":
                assert argv[-2] == image
                out = json.dumps(inputs["verification"])
            else:
                out = reads[(argv[-2], argv[-1])]
        return subprocess.CompletedProcess(argv, 0, stdout=out)

    if published:
        source.DESCRIPTOR.with_name("publication.json").write_bytes(encoded({
            "schema": "sparkring-image-publication/v1", "image_id": image,
            "image_reference": "example.invalid/sparkring@sha256:" + "a" * 64,
            "platform": "linux/arm64", "anonymous_pull_verified": True,
            "descriptor_sha256": sha(source.DESCRIPTOR.read_bytes()),
        }))
    selection = {"source_extension" if published else "local_source_extension": source.IDENTITY}
    result = qwen_flash_next.verify_image(image, **selection, run=run)
    assert result["source_extension"]["id"] == source.IDENTITY
    assert len(calls) == 6
