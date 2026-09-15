"""Admit source-binding updates only with structural and independent test evidence.

Equal syntax for named interfaces is a bounded compatibility check, not a proof
of whole-program semantics. Cache restore/corruption tests remain mandatory
before serving user traffic with an emitted binding.
"""

from __future__ import annotations

import ast
import copy
from pathlib import Path

from .contracts import encoded, oid, require, sha
from .sources import tree_digest


def canonical(node):
    value = copy.deepcopy(node)
    for item in ast.walk(value):
        body = getattr(item, "body", None)
        if (
            isinstance(body, list)
            and body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            del body[0]
    return ast.dump(value, include_attributes=False)


def symbol(tree, name):
    scope = tree
    contexts = []
    for part in name.split("."):
        if isinstance(scope, ast.ClassDef):
            contexts.append(
                {
                    "bases": [canonical(n) for n in scope.bases],
                    "keywords": [canonical(n) for n in scope.keywords],
                    "decorators": [canonical(n) for n in scope.decorator_list],
                    "class_globals": globals_digest(scope),
                }
            )
        matches = []
        for node in getattr(scope, "body", []):
            key = getattr(node, "name", None)
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                key = node.target.id
            if key == part:
                matches.append(node)
        require(
            len(matches) == 1,
            "Required contract symbol is missing or ambiguous: " + name,
        )
        scope = matches[0]
    return repr(contexts) + canonical(scope)


def globals_digest(tree):
    statements = [
        node
        for node in tree.body
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        and not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )
    ]
    return sha(encoded([canonical(node) for node in statements]))


def rebind(
    contract, baseline_root, candidate_root, target_commit, oracle, *, input_sha256
):
    require(
        contract.get("schema") == "sparkring-vllm-kv-block-lease-contract/v1",
        "Unsupported cache binding schema",
    )
    require(oid(target_commit), "Binding requires a full target commit")
    candidate_root, baseline_root = (
        Path(candidate_root).resolve(),
        Path(baseline_root).resolve(),
    )
    target_tree = tree_digest(candidate_root)
    require(
        oracle.get("schema") == "sparkring-upgrade-gate/v1"
        and oracle.get("input_sha256") == input_sha256
        and oracle.get("outcome") == "passed"
        and oracle.get("variant") == "candidate"
        and oracle.get("subject_sha256") == target_tree
        and type(oracle.get("assertions")) is int
        and oracle["assertions"] > 0
        and oracle.get("skipped") == 0,
        "Rebinding requires a passing protected oracle for this exact source tree",
    )
    result = copy.deepcopy(contract)
    evidence = []
    require(
        isinstance(result.get("files"), list) and result["files"],
        "Binding has no source files",
    )
    for row in result["files"]:
        relative = row["path"]
        before = (baseline_root / relative).resolve()
        after = (candidate_root / relative).resolve()
        require(
            before.is_relative_to(baseline_root)
            and after.is_relative_to(candidate_root),
            "Binding source escapes its root",
        )
        previous = before.read_bytes()
        replacement = after.read_bytes()
        require(
            sha(previous) == row["sha256"],
            "Binding does not describe the baseline source: " + relative,
        )
        names = row.get("required_symbols", [])
        if not names:
            require(
                previous == replacement,
                "Byte-only contract file changed; explicit migration required: "
                + relative,
            )
            evidence.append(
                {
                    "path": relative,
                    "baseline_sha256": row["sha256"],
                    "candidate_sha256": row["sha256"],
                    "equivalent_symbols": [],
                    "byte_identical": True,
                }
            )
            continue
        old_tree, new_tree = ast.parse(previous), ast.parse(replacement)
        require(
            globals_digest(old_tree) == globals_digest(new_tree),
            "Module globals changed; explicit interface migration required: "
            + relative,
        )
        for name in names:
            require(
                symbol(old_tree, name) == symbol(new_tree, name),
                "Bound interface changed; explicit migration required: " + name,
            )
        evidence.append(
            {
                "path": relative,
                "baseline_sha256": row["sha256"],
                "candidate_sha256": sha(replacement),
                "equivalent_symbols": names,
            }
        )
        row["sha256"] = sha(replacement)
    result["base_commit"] = target_commit
    if "vllm_commit" in result:
        result["vllm_commit"] = target_commit
    result.pop("vllm_tree", None)
    result["qualification"] = (
        "Implemented source binding: declared interfaces and module globals match "
        "the protected reference; byte-only files remain identical. A passing exact-source "
        "CPU oracle is recorded separately. GPU serving and cache recovery require qualification."
    )
    result["semantic_review"] = {
        "parent_contract_canonical_sha256": sha(encoded(contract)),
        "target_commit": target_commit,
        "candidate_tree_sha256": target_tree,
        "scope": "Named interfaces and byte-only files; not whole-program equivalence.",
    }
    proof = {
        "schema": "sparkring-binding-equivalence/v1",
        "candidate_tree_sha256": target_tree,
        "target_commit": target_commit,
        "oracle": oracle,
        "files": evidence,
        "serving_qualified": False,
        "scope": "Named interface syntax and module globals agree; GPU restore/corruption qualification is still required.",
    }
    return result, proof
