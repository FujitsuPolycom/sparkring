"""Public upgrade-runner behavior, using small local Git repositories."""

import json
from pathlib import Path
import subprocess

from runtime.images.upgrades import contracts
from runtime.images.upgrades.demo import trial


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def test_policy_rejects_missing_behavior_oracle(tmp_path):
    policy = {
        "schema": "sparkring-image-upgrade/v1",
        "name": "fixture",
        "local_sources": True,
        "budgets": dict(
            run_seconds=30,
            command_seconds=10,
            source_bytes=1000000,
            output_bytes=100000,
            agent_attempts=2,
        ),
        "sources": [
            {
                "id": "engine",
                "repository": str(tmp_path),
                "ref": "refs/heads/main",
                "baseline": "a" * 40,
                "editable_paths": ["engine"],
                "native_paths": ["native"],
                "contracts": [],
            }
        ],
    }
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    try:
        contracts.load_policy(path)
    except contracts.Refused as error:
        assert "Behavior contracts" in str(error)
    else:
        raise AssertionError("Unspecified behavior must not be called reconciled")


def test_three_runs_retain_skip_and_semantically_adapt(tmp_path):
    result = trial(tmp_path / "trial")
    assert result["simulation"] is True
    assert result["statuses"] == [
        "candidate-simulation",
        "unchanged",
        "candidate-simulation",
    ]
    assert result["agent_calls"] == 1
    report = json.loads((Path(result["reports"][-1]) / "report.json").read_text())
    assert report["reconciliation"]["engine"]["disposition"] == "adapt"
    assert report["hardware_qualified"] is False
