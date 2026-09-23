"""Bind current vLLM fixtures and HC ownership admission to a versioned suite."""

import json
from pathlib import Path

if __package__:
    from .package_b12x_suite import digest, invariants
else:
    from package_b12x_suite import digest, invariants


def package():
    root = Path(__file__).resolve().parents[1]
    parent = root / "vllm-kraken-suite.json"
    suite = json.loads(parent.read_bytes())
    for name, expected in suite["files"].items():
        assert digest(root / name) == expected, "Frozen oracle changed: " + name
    adaptations = []
    for old in ("kda_preparation_lifetime.py", "qwen_checkpoint_metadata.py",
                "hybrid_recovery.py"):
        new = "sm121/" + old
        assert invariants(root / old) == invariants(root / new), (
            "Protected assertions/cases changed: " + old
        )
        suite["files"][new] = digest(root / new)
        for group in ("common", "candidate_only"):
            suite[group] = [
                new + selection[len(old):]
                if selection.split("::")[0] == old else selection
                for selection in suite[group]
            ]
        adaptations.append({
            "original": old, "original_sha256": digest(root / old),
            "adapted": new, "adapted_sha256": digest(root / new),
            "assertions_and_test_cases_unchanged": True,
        })
    ownership = "sm121/hc_projection_admission.py"
    suite["files"][ownership] = digest(root / ownership)
    suite["candidate_only"].append(ownership)
    suite["migration"] = {
        "parent_suite": parent.name, "parent_suite_sha256": digest(parent),
        "fixture_adaptations": adaptations,
        "additional_checks": [ownership],
        "scope": "Retained CPU source assertions with current KDA, checkpoint and VMM fixtures; reject simultaneous HC token-row and projection ownership. No GPU or serving qualification.",
    }
    output = root / "vllm-sm121-suite-v1.json"
    output.write_bytes((json.dumps(suite, indent=2, sort_keys=True) + "\n").encode())
    return output


if __name__ == "__main__":
    print(package())
