"""Tests for the LMCache geometry and boundary verification gate."""

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


def test_verify_geometry_passes_on_published_recipe():
    recipe = load_recipe()
    result = gate.verify_geometry(recipe)
    assert result["passed"] is True
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


def test_verify_geometry_fails_on_wrong_parent_chunk_size():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["parent_chunk_size"] = 512
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False
    assert any(
        c["check"] == "parent_chunk_size_is_256" and not c["passed"]
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
    recipe["serving"]["lmcache"]["transfer_mode"] = "pytorch"
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False


def test_verify_geometry_fails_on_wrong_topology():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["server_topology"] = "central-server"
    result = gate.verify_geometry(recipe)
    assert result["passed"] is False


def test_cache_isolation_passes_on_published_recipe():
    recipe = load_recipe()
    result = gate.verify_cache_isolation(recipe)
    assert result["passed"] is True
    assert result["sparkcache_disabled"]["observed"] == "0"
    assert result["vllm_prefix_cache"]["observed"] == "enabled"


def test_cache_isolation_fails_when_sparkcache_enabled():
    recipe = load_recipe()
    recipe["serving"]["environment"]["SPARK_CONTEXT_CACHE_ENABLE"] = "1"
    result = gate.verify_cache_isolation(recipe)
    assert result["passed"] is False
    assert result["sparkcache_disabled"]["passed"] is False


def test_cache_isolation_fails_when_prefix_cache_disabled():
    recipe = load_recipe()
    recipe["serving"]["vllm_args"].remove("--enable-prefix-caching")
    result = gate.verify_cache_isolation(recipe)
    assert result["passed"] is False
    assert result["vllm_prefix_cache"]["passed"] is False
    assert result["vllm_prefix_cache"]["observed"] == "absent"


def test_cache_isolation_names_distinct_cache_layers():
    result = gate.verify_cache_isolation(load_recipe())
    assert result["check"] == "cache_isolation"
    assert "separate SparkCache" in result["note"]
    assert "vLLM native prefix cache" in result["note"]


def test_namespace_isolation_is_not_claimed_as_automatically_verified():
    result = gate.namespace_isolation_requirement()
    assert result["implementation_status"] == "operator-enforced"
    assert result["automatically_verified"] is False
    assert "cannot detect reuse" in result["note"]


def test_boundary_requests_are_explicitly_unimplemented():
    result = gate.boundary_request_requirement()
    assert result["implementation_status"] == "not-implemented"
    assert result["automatically_verified"] is False
    assert result["boundaries"] == [511, 512, 513, 1024, 1025]


def test_dcp_hit_consensus_is_explicitly_unimplemented():
    result = gate.dcp_hit_consensus_requirement()
    assert result["implementation_status"] == "not-implemented"
    assert result["automatically_verified"] is False
    assert result["minimum_hit_per_rank"] == 1


def test_capacity_metric_coverage_matches_live_parser():
    result = gate.capacity_metric_coverage()
    assert result["implementation_status"] == "partial"
    assert "total_object_count" in result["collected_metrics"]
    assert result["missing_metrics"] == ["eviction_count"]


def test_verify_all_passes_on_published_recipe():
    recipe = load_recipe()
    report = gate.verify_all(recipe)
    assert report["verdict"] == "pass"
    assert "canonical_recipe_sha256" in report
    assert len(report["canonical_recipe_sha256"]) == 64
    assert len(report["offline_checks"]) == 2
    assert len(report["live_requirements"]) == 4


def test_verify_all_fails_on_bad_geometry():
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["chunk_size"] = 1024
    report = gate.verify_all(recipe)
    assert report["verdict"] == "fail"


def test_build_plan_includes_offline_and_live_sections():
    recipe = load_recipe()
    plan = gate.build_plan(recipe)
    assert "offline_checks" in plan
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


def test_cli_verify_passes(capsys):
    rc = gate.main(["verify"])
    assert rc == gate.EXIT_OK
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "pass"


def test_cli_plan_outputs_plan(capsys):
    rc = gate.main(["plan"])
    assert rc == gate.EXIT_OK
    plan = json.loads(capsys.readouterr().out)
    assert plan["schema"] == gate.PLAN_SCHEMA
    assert "live_timing_cells" in plan


def test_cli_missing_recipe_config_error(capsys):
    rc = gate.main(["--recipe", "nonexistent.json", "verify"])
    assert rc == gate.EXIT_CONFIG_ERROR


def test_cli_verify_fails_on_bad_recipe(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["chunk_size"] = 999
    bad.write_text(json.dumps(recipe), encoding="utf-8")
    rc = gate.main(["--recipe", str(bad), "verify"])
    assert rc == gate.EXIT_FAIL
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "fail"


def test_cli_plan_fails_when_offline_geometry_fails(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    recipe = load_recipe()
    recipe["serving"]["lmcache"]["chunk_size"] = 999
    bad.write_text(json.dumps(recipe), encoding="utf-8")
    rc = gate.main(["--recipe", str(bad), "plan"])
    assert rc == gate.EXIT_FAIL
    plan = json.loads(capsys.readouterr().out)
    assert plan["offline_checks"]["verdict"] == "fail"


def test_cli_malformed_nested_recipe_is_bounded_config_error(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"serving": []}), encoding="utf-8")
    rc = gate.main(["--recipe", str(bad), "verify"])
    captured = capsys.readouterr()
    assert rc == gate.EXIT_CONFIG_ERROR
    assert "serving must be an object" in captured.err


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


def test_capacity_metrics_include_eviction():
    """Eviction remains explicit as an unsupported live metric."""
    coverage = gate.capacity_metric_coverage()
    assert coverage["missing_metrics"] == ["eviction_count"]
    assert "memory_used_bytes" in gate.COLLECTED_L1_METRICS
