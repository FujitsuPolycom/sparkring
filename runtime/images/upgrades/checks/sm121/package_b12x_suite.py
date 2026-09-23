"""Bind versioned B12X oracle adaptations without modifying frozen inputs."""

import ast
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def invariants(path):
    tree = ast.parse(path.read_text())
    return {
        "assertions": [
            ast.dump(n) for n in ast.walk(tree) if isinstance(n, ast.Assert)
        ],
        "tests": {
            n.name: [ast.dump(d) for d in n.decorator_list]
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
        },
    }


def package():
    root = Path(__file__).resolve().parents[1]
    parent = root / "b12x-kraken-suite.json"
    suite = json.loads(parent.read_text())
    for name, expected in suite["files"].items():
        assert digest(root / name) == expected, "Frozen oracle changed: " + name
    adaptations = []
    for old, new in (
        ("b12x_prepared_contracts.py", "sm121/b12x_prepared_contracts.py"),
        ("kraken/distributed_cache_v2.py", "sm121/distributed_cache_v3.py"),
    ):
        assert invariants(root / old) == invariants(root / new), (
            "Protected assertions/cases changed: " + old
        )
        adaptations.append(
            {
                "original": old,
                "original_sha256": digest(root / old),
                "adapted": new,
                "adapted_sha256": digest(root / new),
                "assertions_and_test_cases_unchanged": True,
            }
        )
        suite["files"][new] = digest(root / new)
        suite["candidate_only"] = [
            new + selection[len(old) :]
            if selection.split("::")[0] == old
            else selection
            for selection in suite["candidate_only"]
        ]
    old_qsa = "kraken/qsa_release3_contracts_v2.py"
    new_qsa = "sm121/qsa_dcp_contracts_v3.py"
    suite["files"][new_qsa] = digest(root / new_qsa)
    suite["candidate_only"] = [
        new_qsa if selection == old_qsa else selection
        for selection in suite["candidate_only"]
    ]
    old_exact = "sm121/b12x_prepared_contracts.py::test_qsa_paired_score_and_draft_bounds_still_match_baseline"
    assert old_exact in suite["candidate_only"]
    suite["candidate_only"].remove(old_exact)
    suite["migration"] = {
        "parent_suite": parent.name,
        "parent_suite_sha256": digest(parent),
        "fixture_adaptations": adaptations,
        "qsa_replacement": {
            "original": old_qsa,
            "adapted": new_qsa,
            "replaced_exact_body_selection": old_exact,
            "replacement_test": "test_score_and_draft_preserve_baseline_math_with_dcp_geometry",
            "contract": "sm121/B12X_SOURCE_V1.md",
            "contract_sha256": digest(root / "sm121/B12X_SOURCE_V1.md"),
        },
        "scope": "Frozen common CPU checks plus baseline-derived DCP safety, adversarial geometry, retained ring/program ownership and current metadata fixtures; no GPU qualification.",
    }
    output = root / "b12x-sm121-suite-v1.json"
    output.write_bytes((json.dumps(suite, indent=2, sort_keys=True) + "\n").encode())
    return output


if __name__ == "__main__":
    print(package())
