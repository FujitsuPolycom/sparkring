"""Tests for the LMCache CS512 geometry and boundary verification gate."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import exl3_cache_geometry_gate as gate  # noqa: E402

RECIPE = ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json"


def load_recipe() -> dict:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Geometry — verified checks
# ---------------------------------------------------------------------------


def test_verify_geometry_passes_on_published_recipe():
    recipe = load_recipe()
    result = gate.verify_geometry(recipe)
    assert result["passed"] is True
    assert result["category"] == "verified"
    assert len(result["checks"]) == 6
    for check in result["checks"]:
        assert check["passed"] is True


def test_verify_geometry_fails_on_wrong_chunk_size():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["chunk_size"] = 256
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False
    assert any(
        c["check"] == "chunk_size_is_512" and not c["passed"]
        for c in result["checks"]
    )


def test_verify_geometry_fails_on_wrong_predecessor_chunk_size():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["predecessor_chunk_size"] = 512
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False
    assert any(
        c["check"] == "predecessor_chunk_size_is_256" and not c["passed"]
        for c in result["checks"]
    )


def test_verify_geometry_fails_on_non_lazy_l1():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["l1_lazy"] = False
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False


def test_verify_geometry_fails_on_wrong_eviction_policy():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["eviction_policy"] = "FIFO"
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False


def test_verify_geometry_fails_on_missing_eviction_policy():
    recipe = load_recipe()
    del recipe["serving"]["lmcache"]["eviction_policy"]
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False
    assert any(
        c["check"] == "eviction_policy_is_lru" and not c["passed"]
        for c in result["checks"]
    )


def test_verify_geometry_fails_on_wrong_transfer_mode():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["transfer_mode"] = "server_pushed"
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False


def test_verify_geometry_fails_on_wrong_topology():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["server_topology"] = "central-server"
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False


# ---------------------------------------------------------------------------
# Cache isolation — verified checks
# ---------------------------------------------------------------------------


def test_cache_isolation_passes_on_published_recipe():
    recipe = load_recipe()
    result = gate.verify_cache_isolation(recipe)
    assert result["passed"] is True
    assert result["category"] == "verified"
    assert result["sparkcache_disabled"]["observed"] == "0"
    assert result["apc_native_prefix_cache"]["observed"] == "enabled"


def test_cache_isolation_fails_when_sparkcache_enabled():
    recipe = load_recipe()
    recipe["serving"]["environment"]["SPARK_CONTEXT_CACHE_ENABLE"] = "1"
    result = gate.verify_cache_isolation(recipe)
    assert result["passed"] is False
    assert result["sparkcache_disabled"]["passed"] is False


def test_cache_isolation_fails_when_apc_disabled():
    recipe = load_recipe()
    args = recipe["serving"]["vllm_args"]
    args.remove("--enable-prefix-caching")
    result = gate.verify_cache_isolation(recipe)
    assert result["passed"] is False
    assert result["apc_native_prefix_cache"]["passed"] is False
    assert result["apc_native_prefix_cache"]["observed"] == "absent"


def test_cache_isolation_check_name_does_not_say_apc_isolation():
    """The check name must not falsely call this 'apc_isolation'."""
    recipe = load_recipe()
    result = gate.verify_cache_isolation(recipe)
    assert result["check"] == "cache_isolation"
    assert result["check"] != "apc_isolation"


def test_cache_isolation_note_distinguishes_sparkcache_from_apc():
    """The note must explicitly state SparkCache and APC are distinct."""
    recipe = load_recipe()
    result = gate.verify_cache_isolation(recipe)
    assert "SparkCache" in result["note"]
    assert "distinct" in result["note"]


def test_cache_isolation_has_native_apc_clearing_procedure():
    """The check must document how to clear native APC while retaining
    LMCache objects — engine restart with servers preserved."""
    recipe = load_recipe()
    result = gate.verify_cache_isolation(recipe)
    assert "native_apc_clearing_procedure" in result
    procedure = result["native_apc_clearing_procedure"]
    assert "engine" in procedure.lower()
    assert "lmcache" in procedure.lower()
    assert "restart" in procedure.lower()


# ---------------------------------------------------------------------------
# Planned live gates — never report passed: true
# ---------------------------------------------------------------------------


def test_plan_namespace_isolation_is_planned():
    result = gate.plan_namespace_isolation()
    assert result["category"] == "planned_live"
    assert result["status"] == "planned"
    assert "passed" not in result
    assert "unique" in result["note"].lower()


def test_plan_boundary_tests_is_planned():
    result = gate.plan_boundary_tests()
    assert result["category"] == "planned_live"
    assert result["status"] == "planned"
    assert "passed" not in result
    assert result["boundaries"] == [255, 256, 257, 511, 512, 513, 1024, 1025]
    assert "required_live_geometry" in result
    assert "excludes the final prompt token" in result["note"]


def test_plan_dcp_consensus_is_planned():
    result = gate.plan_dcp_consensus()
    assert result["category"] == "planned_live"
    assert result["status"] == "planned"
    assert "passed" not in result
    assert "evidence_path" in result
    assert "limitation" in result


def test_plan_dcp_consensus_does_not_claim_hit_counter():
    """DCP consensus must not claim /status exposes per-rank hit counters."""
    result = gate.plan_dcp_consensus()
    assert "per-rank cache-hit" in result["limitation"]
    assert "cannot be read from /status" in result["limitation"]


def test_plan_dcp_consensus_objects_not_equated_with_hits():
    """The limitation must state objects != hits."""
    result = gate.plan_dcp_consensus()
    assert "objects were stored, not that they were hit" in result["limitation"]


def test_plan_capacity_metrics_is_planned():
    result = gate.plan_capacity_metrics()
    assert result["category"] == "planned_live"
    assert result["status"] == "planned"
    assert "passed" not in result


def test_plan_capacity_metrics_lists_available_fields():
    """Only fields actually in the /status schema are listed as available."""
    result = gate.plan_capacity_metrics()
    assert "total_object_count" in result["available_from_status"]
    assert "memory_used_bytes" in result["available_from_status"]


def test_plan_capacity_metrics_marks_eviction_unavailable():
    """eviction_count must be in unavailable_from_status, not available."""
    result = gate.plan_capacity_metrics()
    assert "eviction_count" in result["unavailable_from_status"]
    assert "eviction_count" not in result["available_from_status"]


def test_plan_capacity_metrics_note_explains_unavailability():
    result = gate.plan_capacity_metrics()
    assert "not exposed" in result["note"].lower()


# ---------------------------------------------------------------------------
# Aggregation — verify_all
# ---------------------------------------------------------------------------


def test_verify_all_passes_on_published_recipe():
    recipe = load_recipe()
    report = gate.verify_all(recipe)
    assert report["verdict"] == "pass"
    assert "recipe_sha256" in report
    assert len(report["recipe_sha256"]) == 64


def test_verify_all_separates_verified_from_planned():
    """verify_all must have verified_checks and planned_live_gates sections."""
    recipe = load_recipe()
    report = gate.verify_all(recipe)
    assert "verified_checks" in report
    assert "planned_live_gates" in report
    # Verified checks have passed verdicts
    for check in report["verified_checks"]:
        assert "passed" in check
        assert check["category"] == "verified"
    # Planned gates have status: planned, never passed: true
    for gate_item in report["planned_live_gates"]:
        assert gate_item["category"] == "planned_live"
        assert gate_item["status"] == "planned"
        assert "passed" not in gate_item


def test_verify_all_verified_checks_count():
    """Only geometry and cache_isolation are verified offline."""
    recipe = load_recipe()
    report = gate.verify_all(recipe)
    assert len(report["verified_checks"]) == 2


def test_verify_all_planned_gates_count():
    """Namespace, boundary, DCP, and capacity are planned live gates."""
    recipe = load_recipe()
    report = gate.verify_all(recipe)
    assert len(report["planned_live_gates"]) == 4


def test_verify_all_fails_on_bad_geometry():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["chunk_size"] = 1024
    report = gate.verify_all(recipe)
    assert report["verdict"] == "fail"


def test_verify_all_evidence_scope_mentions_offline():
    recipe = load_recipe()
    report = gate.verify_all(recipe)
    assert "offline" in report["evidence_scope"].lower()


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------


def test_build_plan_includes_verified_and_planned_sections():
    recipe = load_recipe()
    plan = gate.build_plan(recipe)
    assert "verified_checks" in plan
    assert "planned_live_gates" in plan
    assert "live_timing_cells" in plan
    assert len(plan["live_timing_cells"]) == 8
    labels = [cell["label"] for cell in plan["live_timing_cells"]]
    assert "C1-standard" in labels
    assert "C8-16K" in labels
    assert "C8-64K" in labels


def test_build_plan_cold_warm_flag():
    recipe = load_recipe()
    plan = gate.build_plan(recipe)
    for cell in plan["live_timing_cells"]:
        assert cell["cold_warm"] is True


def test_build_plan_has_dcp_live_note():
    """Plan must note that DCP requires live evidence, not hit counters."""
    recipe = load_recipe()
    plan = gate.build_plan(recipe)
    assert "dcp_live_note" in plan
    assert "hit counters" in plan["dcp_live_note"]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_verify_passes(capsys):
    rc = gate.main(["verify"])
    assert rc == gate.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "pass"
    assert "verified_checks" in report
    assert "planned_live_gates" in report


def test_cli_plan_outputs_plan(capsys):
    rc = gate.main(["plan"])
    assert rc == gate.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == gate.PLAN_SCHEMA
    assert "live_timing_cells" in plan
    assert "verified_checks" in plan
    assert "planned_live_gates" in plan


def test_cli_missing_recipe_config_error(capsys):
    rc = gate.main(["--recipe", "nonexistent.json", "verify"])
    assert rc == gate.EXIT_CONFIG_ERROR


def test_cli_verify_fails_on_bad_recipe(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps({"serving": {"lmcache": {"chunk_size": 999}}}),
        encoding="utf-8",
    )
    rc = gate.main(["--recipe", str(bad), "verify"])
    assert rc == gate.EXIT_FAIL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "fail"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


def test_boundary_token_counts_are_chunk_aware():
    """Boundaries must relate to the chunk size, not be arbitrary."""
    assert 512 in gate.BOUNDARY_TOKEN_COUNTS  # exact chunk
    assert 511 in gate.BOUNDARY_TOKEN_COUNTS  # below chunk
    assert 513 in gate.BOUNDARY_TOKEN_COUNTS  # above chunk
    assert 1024 in gate.BOUNDARY_TOKEN_COUNTS  # exact 2x chunk
    assert 1025 in gate.BOUNDARY_TOKEN_COUNTS  # above 2x chunk


def test_timing_cells_include_16k_and_64k():
    """The timing suite must include 16K and 64K context cells."""
    contexts = {cell["context"] for cell in gate.TIMING_CELLS}
    assert "16K" in contexts
    assert "64K" in contexts
    concurrencies = {cell["concurrency"] for cell in gate.TIMING_CELLS}
    assert {1, 2, 4, 8} == concurrencies


def test_status_available_metrics_match_observed_schema():
    """The available metrics must match fields in parse_launcher_status."""
    # These are the fields exl3_cache_acceptance.py extracts from /status
    assert "total_object_count" in gate.STATUS_AVAILABLE_METRICS
    assert "memory_used_bytes" in gate.STATUS_AVAILABLE_METRICS
    assert "write_locked_count" in gate.STATUS_AVAILABLE_METRICS
    assert "read_locked_count" in gate.STATUS_AVAILABLE_METRICS
    assert "temporary_count" in gate.STATUS_AVAILABLE_METRICS
    assert "is_healthy" in gate.STATUS_AVAILABLE_METRICS
    assert "registered_gpu_ids" in gate.STATUS_AVAILABLE_METRICS


def test_eviction_count_is_unavailable():
    """eviction_count must NOT be in available metrics."""
    assert "eviction_count" in gate.STATUS_UNAVAILABLE_METRICS
    assert "eviction_count" not in gate.STATUS_AVAILABLE_METRICS


def test_dcp_consensus_evidence_path_lists_object_counts_and_ttft():
    """The evidence path must use object counts and TTFT, not hit counters."""
    evidence = gate.DCP_CONSENSUS_EVIDENCE_PATH
    text = " ".join(evidence)
    assert "total_object_count" in text
    assert "TTFT" in text
    assert "hit-length" in text


def test_dcp_consensus_limitation_explains_no_hit_counter():
    limitation = gate.DCP_CONSENSUS_LIMITATION
    assert "does not expose per-rank cache-hit" in limitation
    assert "cannot be read from /status" in limitation
    assert "objects were stored, not that they were hit" in limitation
