"""A source extension may supersede exactly one verified parent cache binding."""

import copy
import json

import pytest

from .build_native import select_parent_binding
from .contracts import sha

BASE = "/opt/sparkring/contracts/vllm-connector-jobs-base.json"
SELECTED = "/opt/sparkring/contracts/vllm-connector-jobs-extension.json"
OTHER = "/opt/sparkring/contracts/transport.json"
DESCRIPTOR = "/opt/sparkring/receipts/source-extension.json"
SOURCE = "/opt/venv/lib/python3.12/site-packages/vllm/lease.py"


def fixture():
    contract = {
        "schema": "sparkring-vllm-kv-block-lease-contract/v1",
        "files": [{"path": "vllm/lease.py", "sha256": "c" * 64}],
        "semantic_review": {"base_contract_sha256": "a" * 64},
    }
    raw = json.dumps(contract).encode()
    extension = {
        "schema": "sparkring-source-extension/v1",
        "id": "cache-checkpoint-extension",
        "parent": {"image_id": "sha256:" + "1" * 64, "receipt_sha256": "2" * 64},
        "integration_contracts": {SELECTED: {"sha256": sha(raw)}},
        "provenance": {"role": "Checkpoint lease extension"},
    }
    extension_raw = json.dumps(extension).encode()
    parent = {
        "integration_contracts": {
            BASE: {"sha256": "a" * 64},
            OTHER: {"sha256": "b" * 64},
        },
        "files": {
            BASE: "a" * 64,
            OTHER: "b" * 64,
            SELECTED: sha(raw),
            SOURCE: "c" * 64,
            DESCRIPTOR: sha(extension_raw),
        },
        "source_extension": {
            "id": extension["id"],
            "descriptor_sha256": sha(extension_raw),
            "parent_image_id": extension["parent"]["image_id"],
            "parent_receipt_sha256": extension["parent"]["receipt_sha256"],
            "provenance": extension["provenance"],
        },
    }
    return parent, raw, extension_raw


def test_verified_extension_replaces_only_its_active_base():
    parent, raw, extension = fixture()
    unchanged = copy.deepcopy(parent)
    active, proof = select_parent_binding(parent, SELECTED, raw, extension)
    assert active == {SELECTED, OTHER}
    assert proof["superseded_contract"] == BASE
    assert proof["selected_contract"] == SELECTED
    assert parent == unchanged


@pytest.mark.parametrize(
    "defect",
    [
        "absent_extension",
        "descriptor_hash",
        "source_hash",
        "ambiguous_base",
        "base_hash",
        "unowned_selection",
        "descriptor_identity",
        "provenance",
    ],
)
def test_unproven_extension_binding_is_rejected(defect):
    parent, raw, extension = fixture()
    if defect == "absent_extension":
        del parent["source_extension"]
    elif defect == "descriptor_hash":
        parent["files"][DESCRIPTOR] = "d" * 64
    elif defect == "source_hash":
        parent["files"][SOURCE] = "d" * 64
    elif defect == "ambiguous_base":
        second = "/opt/sparkring/contracts/vllm-connector-jobs-second.json"
        parent["integration_contracts"][second] = {"sha256": "a" * 64}
        parent["files"][second] = "a" * 64
    elif defect == "base_hash":
        parent["files"][BASE] = "d" * 64
    elif defect == "unowned_selection":
        del parent["files"][SELECTED]
    elif defect == "descriptor_identity":
        parent["source_extension"]["id"] = "unrelated-extension"
    elif defect == "provenance":
        parent["source_extension"]["provenance"] = {}
    with pytest.raises(ValueError):
        select_parent_binding(parent, SELECTED, raw, extension)


def test_retired_contract_not_declared_by_extension_is_rejected():
    parent, raw, extension = fixture()
    value = json.loads(extension)
    value["integration_contracts"] = {}
    extension = json.dumps(value).encode()
    parent["files"][DESCRIPTOR] = sha(extension)
    parent["source_extension"]["descriptor_sha256"] = sha(extension)
    with pytest.raises(ValueError, match="declare"):
        select_parent_binding(parent, SELECTED, raw, extension)


def test_already_active_owned_binding_requires_no_extension():
    raw = b"{}"
    active, evidence = select_parent_binding(
        {"integration_contracts": {BASE: {}}, "files": {BASE: sha(raw)}}, BASE, raw
    )
    assert active == {BASE} and evidence is None
