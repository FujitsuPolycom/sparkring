"""Contract tests for the external LMCache key-value reuse gate.

The two negative cases named in the gate's docstring — a stalled request with a
high hit count, and a fast byte-identical replay inside a live engine process —
are asserted directly, because both are shapes that read as success to a reader
who checks counters alone.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import lmcache_external_reuse_gate as gate  # noqa: E402

SCRIPT = Path(__file__).resolve().parent / "lmcache_external_reuse_gate.py"

PROMPT_HASH = "b" * 64
OUTPUT_HASH = "c" * 64


def passing_evidence() -> dict:
    return {
        "schema": gate.EVIDENCE_SCHEMA,
        "checkpoint": "example/Checkpoint-A",
        "package_version": "0.5.3+glm52dcp4.2",
        "external_reuse_policy": gate.POLICY_REQUIRED,
        "store_phase": {
            "completed": True,
            "prompt_sha256": PROMPT_HASH,
            "output_sha256": OUTPUT_HASH,
        },
        "teardown": {
            "engine_containers_removed": True,
            "cache_servers_removed": True,
            "memory_tier_cleared": True,
            "filesystem_tier_retained": True,
        },
        "replay_phase": {
            "completed": True,
            "prompt_sha256": PROMPT_HASH,
            "output_sha256": OUTPUT_HASH,
            "engine_process_recreated": True,
            "native_prefix_caching_enabled": False,
            "native_prefix_cache_hits": 0,
            "external_prefix_cache_hits": 47104,
        },
        "logs": {
            "server_size_mismatch_count": 0,
            "engine_retrieve_failed_count": 0,
        },
    }


def test_complete_evidence_passes():
    report = gate.evaluate(passing_evidence())
    assert report["verdict"] == "pass"
    assert report["failed_conditions"] == []
    assert report["checkpoint"] == "example/Checkpoint-A"


def test_stalled_request_with_high_hit_count_fails():
    """The hit counter counts lookups, so a stall with hits is not a slow success."""
    evidence = passing_evidence()
    evidence["replay_phase"]["completed"] = False
    evidence["replay_phase"]["external_prefix_cache_hits"] = 104960

    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert "replay_request_completed" in report["failed_conditions"]
    assert "hit_counter_not_misreporting_a_stall" in report["failed_conditions"]


def test_live_process_replay_fails_even_when_output_is_identical():
    """A replay inside a live engine is served by that engine's own prefix cache."""
    evidence = passing_evidence()
    evidence["replay_phase"]["engine_process_recreated"] = False
    evidence["replay_phase"]["native_prefix_caching_enabled"] = True
    evidence["replay_phase"]["native_prefix_cache_hits"] = 47104
    evidence["replay_phase"]["external_prefix_cache_hits"] = 0

    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert "deployment_destroyed_and_recreated" in report["failed_conditions"]
    assert (
        "native_prefix_cache_cannot_account_for_result"
        in report["failed_conditions"]
    )
    assert "external_tier_recorded_hits" in report["failed_conditions"]
    # The output hash still matches; matching output alone never carries the gate.
    assert "output_hash_matches_store_phase" not in report["failed_conditions"]


def test_native_prefix_cache_left_enabled_fails_attribution():
    evidence = passing_evidence()
    evidence["replay_phase"]["native_prefix_caching_enabled"] = True
    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert (
        "native_prefix_cache_cannot_account_for_result"
        in report["failed_conditions"]
    )


def test_output_divergence_fails():
    evidence = passing_evidence()
    evidence["replay_phase"]["output_sha256"] = "d" * 64
    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert "output_hash_matches_store_phase" in report["failed_conditions"]


def test_different_replay_prompt_fails():
    evidence = passing_evidence()
    evidence["replay_phase"]["prompt_sha256"] = "e" * 64
    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert "replayed_prompt_is_identical" in report["failed_conditions"]


@pytest.mark.parametrize(
    "pointer",
    ["server_size_mismatch_count", "engine_retrieve_failed_count"],
)
def test_nonzero_log_counts_fail(pointer):
    evidence = passing_evidence()
    evidence["logs"][pointer] = 1
    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    expected = {
        "server_size_mismatch_count": "no_transfer_size_mismatch",
        "engine_retrieve_failed_count": "no_retrieve_failures",
    }[pointer]
    assert expected in report["failed_conditions"]


@pytest.mark.parametrize(
    "field",
    [
        "engine_containers_removed",
        "cache_servers_removed",
        "memory_tier_cleared",
        "filesystem_tier_retained",
    ],
)
def test_incomplete_teardown_fails(field):
    evidence = passing_evidence()
    evidence["teardown"][field] = False
    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert "deployment_destroyed_and_recreated" in report["failed_conditions"]


def test_zero_external_hits_fails():
    evidence = passing_evidence()
    evidence["replay_phase"]["external_prefix_cache_hits"] = 0
    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert "external_tier_recorded_hits" in report["failed_conditions"]


@pytest.mark.parametrize("pointer", gate.REQUIRED_POINTERS)
def test_missing_required_field_is_a_config_error_not_a_pass(tmp_path, pointer):
    """An incomplete collection must never read as either a pass or a refutation."""
    evidence = passing_evidence()
    parent = evidence
    tokens = pointer.strip("/").split("/")
    for token in tokens[:-1]:
        parent = parent[token]
    del parent[tokens[-1]]

    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(gate.ConfigError):
        gate.load_evidence(path)


def test_wrong_schema_is_rejected(tmp_path):
    evidence = passing_evidence()
    evidence["schema"] = "something-else/v1"
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(gate.ConfigError):
        gate.load_evidence(path)


def test_malformed_json_is_rejected(tmp_path):
    path = tmp_path / "evidence.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(gate.ConfigError):
        gate.load_evidence(path)


@pytest.mark.parametrize("value", [1, "yes", None])
def test_boolean_fields_reject_non_booleans(tmp_path, value):
    evidence = passing_evidence()
    evidence["replay_phase"]["completed"] = value
    path = tmp_path / "evidence.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(gate.ConfigError):
        gate.evaluate(gate.load_evidence(path))


@pytest.mark.parametrize("value", [-1, True, "12"])
def test_counter_fields_reject_non_counts(value):
    evidence = passing_evidence()
    evidence["replay_phase"]["external_prefix_cache_hits"] = value
    with pytest.raises(gate.ConfigError):
        gate.evaluate(evidence)


def test_plan_lists_every_required_field_and_marks_authorization():
    plan = gate.build_plan()
    assert plan["required_fields"] == list(gate.REQUIRED_POINTERS)
    assert plan["evidence_schema"] == gate.EVIDENCE_SCHEMA
    classes = {step["safety_class"] for step in plan["steps"]}
    assert classes == {"READ-ONLY REMOTE", "STOPS SERVING"}
    assert "authorization" in plan


def test_plan_is_stable():
    assert gate.build_plan() == copy.deepcopy(gate.build_plan())


def test_cli_exit_codes(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps(passing_evidence()), encoding="utf-8")

    failing = passing_evidence()
    failing["replay_phase"]["external_prefix_cache_hits"] = 0
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(failing), encoding="utf-8")

    plan = subprocess.run(
        [sys.executable, str(SCRIPT), "plan"], capture_output=True, text=True
    )
    assert plan.returncode == gate.EXIT_OK
    assert json.loads(plan.stdout)["schema"] == gate.PLAN_SCHEMA

    ok = subprocess.run(
        [sys.executable, str(SCRIPT), "evaluate", "--evidence", str(good)],
        capture_output=True,
        text=True,
    )
    assert ok.returncode == gate.EXIT_OK
    assert json.loads(ok.stdout)["verdict"] == "pass"

    fail = subprocess.run(
        [sys.executable, str(SCRIPT), "evaluate", "--evidence", str(bad)],
        capture_output=True,
        text=True,
    )
    assert fail.returncode == gate.EXIT_FAIL

    missing = subprocess.run(
        [sys.executable, str(SCRIPT), "evaluate"], capture_output=True, text=True
    )
    assert missing.returncode == gate.EXIT_CONFIG_ERROR
    assert "CONFIG ERROR" in missing.stderr


def test_optional_policy_failure_does_not_block_serving():
    """An optional tier that does not qualify withholds reuse and nothing else."""
    evidence = passing_evidence()
    evidence["external_reuse_policy"] = gate.POLICY_OPTIONAL
    evidence["logs"]["server_size_mismatch_count"] = 21525
    evidence["replay_phase"]["completed"] = False

    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert report["serving_blocked"] is False
    assert gate.exit_code_for(report) == gate.EXIT_OPTIONAL_UNAVAILABLE
    assert "unaffected" in report["serving_disposition"]


def test_required_policy_failure_blocks_serving():
    evidence = passing_evidence()
    evidence["logs"]["server_size_mismatch_count"] = 21525

    report = gate.evaluate(evidence)
    assert report["verdict"] == "fail"
    assert report["serving_blocked"] is True
    assert gate.exit_code_for(report) == gate.EXIT_FAIL


@pytest.mark.parametrize("policy", [gate.POLICY_REQUIRED, gate.POLICY_OPTIONAL])
def test_passing_gate_never_blocks_serving_under_either_policy(policy):
    evidence = passing_evidence()
    evidence["external_reuse_policy"] = policy
    report = gate.evaluate(evidence)
    assert report["verdict"] == "pass"
    assert report["serving_blocked"] is False
    assert gate.exit_code_for(report) == gate.EXIT_OK


def test_unknown_policy_is_a_config_error():
    evidence = passing_evidence()
    evidence["external_reuse_policy"] = "preferred"
    with pytest.raises(gate.ConfigError):
        gate.evaluate(evidence)


def test_optional_failure_exit_code_is_distinct_from_every_other_outcome():
    """Exit 4 must not collide with pass, blocking failure, or config error."""
    codes = {
        gate.EXIT_OK,
        gate.EXIT_FAIL,
        gate.EXIT_CONFIG_ERROR,
        gate.EXIT_OPTIONAL_UNAVAILABLE,
    }
    assert len(codes) == 4


def test_cli_optional_failure_exits_four(tmp_path):
    evidence = passing_evidence()
    evidence["external_reuse_policy"] = gate.POLICY_OPTIONAL
    evidence["replay_phase"]["external_prefix_cache_hits"] = 0
    path = tmp_path / "optional.json"
    path.write_text(json.dumps(evidence), encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), "evaluate", "--evidence", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == gate.EXIT_OPTIONAL_UNAVAILABLE
    report = json.loads(result.stdout)
    assert report["verdict"] == "fail"
    assert report["serving_blocked"] is False
