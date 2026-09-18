"""Migration policy cannot hide dropped cache interfaces or untested source bytes."""

import copy
import json

import pytest

from .contract_migration import migrate
from .contracts import Refused, encoded, sha
from .sources import tree_digest
from .build_native import prepare_source_binding


def inputs(tmp_path):
    root = tmp_path / "source"
    (root / "vllm").mkdir(parents=True)
    data = b"class Lease:\n def retain(self): return True\n"
    (root / "vllm/new.py").write_bytes(data)
    parent = dict(
        schema="sparkring-vllm-kv-block-lease-contract/v1",
        connector_api="lease/v1",
        required_semantics=["retains-pages"],
        files=[dict(path="vllm/old.py", sha256="0" * 64)],
    )
    contract = dict(
        schema=parent["schema"],
        connector_api=parent["connector_api"],
        base_commit="a" * 40,
        required_semantics=["retains-pages"],
        files=[
            dict(
                path="vllm/new.py", sha256=sha(data), required_symbols=["Lease.retain"]
            )
        ],
    )
    manifest = dict(
        schema="sparkring-binding-migration-policy/v1",
        parent_contract_sha256=sha(encoded(parent)),
        contract=contract,
        source_mapping={"vllm/old.py": ["vllm/new.py"]},
        required_oracles={"vllm": ["lease"]},
        semantic_review={
            "retains-pages": dict(
                reason="Retain the independently checked owning reference.",
                oracles=["lease"],
            )
        },
    )
    tree = tree_digest(root)
    receipt = dict(
        schema="sparkring-upgrade-gate/v1",
        gate="lease",
        subject_sha256=tree,
        input_sha256="f" * 64,
        variant="candidate",
        outcome="passed",
        assertions=2,
        skipped=0,
    )
    records = {
        "vllm": dict(
            target_commit="a" * 40, candidate_tree_sha256=tree, oracles=[receipt]
        )
    }
    return parent, manifest, {"vllm": root}, records


def test_exact_reviewed_migration_has_no_serving_qualification(tmp_path):
    values = inputs(tmp_path)
    before = copy.deepcopy(values[1])
    contract, proof = migrate(*values, input_sha256="f" * 64)
    assert contract == before["contract"] and values[1] == before
    assert proof["serving_qualified"] is False
    assert proof["source_mapping"] == {"vllm/old.py": ["vllm/new.py"]}


@pytest.mark.parametrize(
    "defect",
    [
        "missing_file",
        "hash",
        "dropped_semantic",
        "unmapped",
        "wrong_tree",
        "failed_oracle",
        "skipped_oracle",
        "wrong_input",
        "unreviewed",
    ],
)
def test_migration_fails_closed(tmp_path, defect):
    parent, manifest, roots, records = inputs(tmp_path)
    if defect == "missing_file":
        manifest["contract"]["files"][0]["path"] = "vllm/missing.py"
    if defect == "hash":
        manifest["contract"]["files"][0]["sha256"] = "1" * 64
    if defect == "dropped_semantic":
        manifest["contract"]["required_semantics"] = []
    if defect == "unmapped":
        manifest["source_mapping"] = {}
    if defect == "wrong_tree":
        records["vllm"]["candidate_tree_sha256"] = "a" * 64
    if defect == "failed_oracle":
        records["vllm"]["oracles"][0]["outcome"] = "failed"
    if defect == "skipped_oracle":
        records["vllm"]["oracles"][0]["skipped"] = 1
    if defect == "wrong_input":
        records["vllm"]["oracles"][0]["input_sha256"] = "b" * 64
    if defect == "unreviewed":
        manifest["semantic_review"] = {}
    with pytest.raises(Refused):
        migrate(parent, manifest, roots, records, input_sha256="f" * 64)


def test_builder_keeps_migrated_binding_distinct_from_parent(tmp_path):
    parent_contract, manifest, roots, records = inputs(tmp_path)
    old = tmp_path / "parent-contract.json"
    old.write_text(json.dumps(parent_contract))
    policy_file = tmp_path / "migration.json"
    policy_file.write_text(json.dumps(manifest))
    old_installed = "/opt/sparkring/contracts/" + old.name
    parent = {
        "integration_contracts": {old_installed: {}},
        "files": {old_installed: sha(old.read_bytes())},
    }
    policy = {
        "_root": str(tmp_path),
        "foundation": {
            "source_binding": {
                "contract": old.name,
                "sha256": sha(old.read_bytes()),
                "migration": {
                    "path": policy_file.name,
                    "sha256": sha(policy_file.read_bytes()),
                },
            }
        },
    }
    context = tmp_path / "context"
    context.mkdir()
    selection, active = prepare_source_binding(
        policy, {"sources": records, "input_sha256": "f" * 64}, roots, parent, context
    )
    proof = json.loads((context / selection["proof_file"]).read_text())
    assert proof["schema"] == "sparkring-binding-migration/v1"
    assert active == [selection["destination"]] and old_installed not in active
    assert json.loads(old.read_text()) == parent_contract


def test_migration_requires_b12x_oracle_when_lease_names_b12x(tmp_path):
    parent, manifest, roots, records = inputs(tmp_path)
    root = tmp_path / "b12x-source"
    (root / "b12x").mkdir(parents=True)
    data = b"VERSION=1\n"
    (root / "b12x/contract.py").write_bytes(data)
    roots["b12x"] = root
    records["b12x"] = {"candidate_tree_sha256": tree_digest(root), "oracles": []}
    manifest["contract"]["files"].append(
        {"path": "b12x/contract.py", "sha256": sha(data)}
    )
    manifest["required_oracles"]["b12x"] = ["kernel-contract"]
    with pytest.raises(Refused, match="oracle"):
        migrate(parent, manifest, roots, records, input_sha256="f" * 64)


@pytest.mark.parametrize(
    "required",
    [
        ["compile_auxiliary"],
        ["Lease.retain.extra"],
        [".retain"],
        ["Lease."],
        "Lease.retain",
        [1],
        [""],
    ],
)
def test_migration_rejects_symbols_outside_installed_lease_schema(tmp_path, required):
    parent, manifest, roots, records = inputs(tmp_path)
    path = roots["vllm"] / "vllm/new.py"
    path.write_bytes(path.read_bytes() + b"def compile_auxiliary(): pass\n")
    row = manifest["contract"]["files"][0]
    row.update(sha256=sha(path.read_bytes()), required_symbols=required)
    tree = tree_digest(roots["vllm"])
    records["vllm"]["candidate_tree_sha256"] = tree
    records["vllm"]["oracles"][0]["subject_sha256"] = tree
    with pytest.raises(Refused, match="required.symbol|Class.member"):
        migrate(parent, manifest, roots, records, input_sha256="f" * 64)


@pytest.mark.parametrize(
    "member", ["retain", "field", "assigned", "Inner", "async_method"]
)
def test_migration_accepts_installed_lease_class_member_forms(tmp_path, member):
    parent, manifest, roots, records = inputs(tmp_path)
    path = roots["vllm"] / "vllm/new.py"
    path.write_text(
        "class Lease:\n def retain(self): pass\n field: int\n assigned = 1\n class Inner: pass\n async def async_method(self): pass\n"
    )
    manifest["contract"]["files"][0].update(
        sha256=sha(path.read_bytes()), required_symbols=["Lease." + member]
    )
    tree = tree_digest(roots["vllm"])
    records["vllm"]["candidate_tree_sha256"] = tree
    records["vllm"]["oracles"][0]["subject_sha256"] = tree
    migrate(parent, manifest, roots, records, input_sha256="f" * 64)
