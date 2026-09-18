"""Admit reviewed cache-interface migrations bound to exact sources and oracles.

Unlike syntax-equivalent rebinding, this path requires an operator-owned file
mapping and semantic review. It cannot infer serving qualification from hashes.
"""

from __future__ import annotations

import ast
from pathlib import Path

from .contract_rebind import symbol
from .contracts import encoded, oid, relative, require, sha
from .sources import tree_digest


def migrate(parent, manifest, roots, source_records, *, input_sha256):
    """Validate a reviewed migration without writing or changing its inputs."""
    require(
        manifest.get("schema") == "sparkring-binding-migration-policy/v1",
        "Unknown binding migration policy",
    )
    require(
        manifest.get("parent_contract_sha256") == sha(encoded(parent)),
        "Migration does not identify the protected parent contract",
    )
    contract = manifest.get("contract", {})
    require(
        contract.get("schema")
        == parent.get("schema")
        == "sparkring-vllm-kv-block-lease-contract/v1",
        "Migration changes the cache contract schema",
    )
    require(
        contract.get("connector_api") == parent.get("connector_api"),
        "Connector API migration requires a distinct connector implementation",
    )
    require(
        oid(contract.get("base_commit"))
        and contract["base_commit"] == source_records["vllm"]["target_commit"],
        "Migration target differs from accepted vLLM source",
    )
    required = set(parent.get("required_semantics", []))
    require(
        required and required <= set(contract.get("required_semantics", [])),
        "Migration drops required cache semantics",
    )
    review = manifest.get("semantic_review", {})
    require(
        set(review) == set(contract["required_semantics"]),
        "Every cache semantic requires an explicit reviewed disposition",
    )
    old = {item["path"] for item in parent["files"]}
    mapping = manifest.get("source_mapping", {})
    require(set(mapping) == old, "Migration omits a parent-owned source file")
    rows = contract.get("files")
    require(isinstance(rows, list) and rows, "Migration has no target source inventory")
    names = [row["path"] for row in rows]
    require(len(set(names)) == len(names), "Migration contains duplicate target paths")
    for targets in mapping.values():
        require(
            isinstance(targets, list) and targets and set(targets) <= set(names),
            "Parent source must map to inventoried target files",
        )
    used_components = set()
    for row in rows:
        name = row["path"]
        relative(name)
        component = name.split("/")[0]
        require(component in roots, "Migration refers to an unbuilt component")
        used_components.add(component)
        root = Path(roots[component]).resolve()
        file = root / name
        require(
            file.resolve().is_relative_to(root)
            and not file.is_symlink()
            and file.is_file(),
            "Migration source is missing or escapes its component",
        )
        data = file.read_bytes()
        require(sha(data) == row["sha256"], "Migration source bytes differ: " + name)
        if row.get("required_symbols"):
            tree = ast.parse(data)
            for identifier in row["required_symbols"]:
                symbol(tree, identifier)
    trees = {component: tree_digest(roots[component]) for component in used_components}
    evidence = {}
    for component in used_components:
        source = source_records[component]
        require(
            source.get("candidate_tree_sha256") == trees[component],
            "Migration accepted source tree differs",
        )
        for receipt in source.get("oracles", []):
            gate = receipt.get("gate")
            require(gate not in evidence, "Migration oracle ID is ambiguous")
            if (
                receipt.get("schema") == "sparkring-upgrade-gate/v1"
                and receipt.get("subject_sha256") == trees[component]
                and receipt.get("input_sha256") == input_sha256
                and receipt.get("variant") == "candidate"
                and receipt.get("outcome") == "passed"
                and type(receipt.get("assertions")) is int
                and receipt["assertions"] > 0
                and receipt.get("skipped") == 0
            ):
                evidence[gate] = {"component": component, "receipt": receipt}
    required_gates = manifest.get("required_oracles", {})
    require(
        set(required_gates) == used_components,
        "Each bound component requires its own source oracle",
    )
    for component, gates in required_gates.items():
        require(
            isinstance(gates, list)
            and gates
            and all(
                gate in evidence and evidence[gate]["component"] == component
                for gate in gates
            ),
            "Required migration oracle is missing or did not pass",
        )
    approved_gates = {gate for gates in required_gates.values() for gate in gates}
    for item in review.values():
        require(
            isinstance(item.get("reason"), str)
            and item["reason"].strip()
            and isinstance(item.get("oracles"), list)
            and item["oracles"]
            and set(item["oracles"]) <= approved_gates,
            "Semantic review lacks a reason or required oracle",
        )
    proof = {
        "schema": "sparkring-binding-migration/v1",
        "migration_policy_sha256": sha(encoded(manifest)),
        "parent_contract_sha256": sha(encoded(parent)),
        "contract_sha256": sha(encoded(contract)),
        "candidate_tree_sha256": trees["vllm"],
        "component_trees": trees,
        "input_sha256": input_sha256,
        "oracles": [evidence[gate] for gate in sorted(approved_gates)],
        "semantic_review": review,
        "source_mapping": mapping,
        "serving_qualified": False,
        "scope": "Reviewed exact-source migration plus bounded CPU oracles; GPU/cache recovery qualification is separate.",
    }
    return contract, proof
