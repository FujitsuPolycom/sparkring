#!/usr/bin/env python3
"""Dry-run-first LMCache CS512 geometry and boundary verification gate.

Extends the cache acceptance runbook with offline-verifiable configuration
checks for the LMCache CS512 block-256 geometry, token-count boundaries,
DCP minimum-hit consensus, SparkCache isolation with the native vLLM prefix
cache enabled, namespace isolation, and
capacity/eviction metric declarations.  A companion plan mode discloses
the C1/C2/C4/C8 and 16K/64K cold/warm timing cells that require a live
cluster.

This gate is OFFLINE: it reads the recipe and launch profile to verify
geometry and boundary declarations.  It does not contact the cluster.
The live timing and boundary probes are delegated to
``exl3_cache_acceptance.py`` and the acceptance gate's performance
matrix.

Usage::

    # Offline: verify geometry and boundaries from the recipe
    python scripts/exl3_cache_geometry_gate.py verify

    # Offline: produce a plan for the full geometry + timing suite
    python scripts/exl3_cache_geometry_gate.py plan

    # Offline: verify a specific profile
    python scripts/exl3_cache_geometry_gate.py verify \\
        --recipe recipes/glm52-exl3-tr3-3.25bpw.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-exl3-cache-geometry/v1"
PLAN_SCHEMA = "sparkring-exl3-cache-geometry-plan/v1"

EXIT_OK = 0
EXIT_FAIL = 2
EXIT_CONFIG_ERROR = 3

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECIPE = ROOT / "recipes/glm52-exl3-tr3-3.25bpw.json"

# Block-256 is the parent chunk size for this recipe's CS512 configuration.
# The chunk size is 512, the parent chunk size is 256.
REQUIRED_CHUNK_SIZE = 512
REQUIRED_PARENT_CHUNK_SIZE = 256

# Boundary token counts to verify: the cache must handle sequences that
# span exactly, just below, and just above the chunk/parent boundaries.
# These are derived from the chunk geometry, not arbitrary.
BOUNDARY_TOKEN_COUNTS = [511, 512, 513, 1024, 1025]

# DCP minimum-hit consensus: all four ranks must report at least one cache
# hit after a warm probe. The minimum is 1, not 0 — a rank with zero hits
# after a warm probe has a cache miss consensus failure.
DCP_MINIMUM_HIT_PER_RANK = 1

# SparkCache and vLLM's native prefix cache are distinct mechanisms. The
# separate sparkcache/ implementation must be disabled for LMCache attribution,
# while vLLM prefix caching remains enabled in this profile.
SPARKCACHE_DISABLE_ENV = "SPARK_CONTEXT_CACHE_ENABLE"
SPARKCACHE_DISABLED_VALUE = "0"
PREFIX_CACHE_ENABLE_FLAG = "--enable-prefix-caching"

# Namespace isolation: the operator must provide a unique probe ID per run.
# The current acceptance gate records the value but cannot detect reuse.
NAMESPACE_ISOLATION_NOTE = (
    "The probe ID must be unique per acceptance run. The acceptance gate "
    "records the operator-provided value but does not maintain history and "
    "therefore cannot detect reuse."
)

# L1 metrics collected by exl3_cache_acceptance.parse_launcher_status.
COLLECTED_L1_METRICS = [
    "total_object_count",
    "memory_used_bytes",
    "write_locked_count",
    "read_locked_count",
    "temporary_count",
]

# Cold/warm timing cells: C1/C2/C4/C8 at standard and 16K/64K contexts.
# These require a live cluster and are disclosed in the plan only.
TIMING_CELLS = [
    {"label": "C1-standard", "concurrency": 1, "context": "standard",
     "cold_warm": True},
    {"label": "C2-standard", "concurrency": 2, "context": "standard",
     "cold_warm": True},
    {"label": "C4-standard", "concurrency": 4, "context": "standard",
     "cold_warm": True},
    {"label": "C8-standard", "concurrency": 8, "context": "standard",
     "cold_warm": True},
    {"label": "C1-16K", "concurrency": 1, "context": "16K",
     "cold_warm": True},
    {"label": "C1-64K", "concurrency": 1, "context": "64K",
     "cold_warm": True},
    {"label": "C8-16K", "concurrency": 8, "context": "16K",
     "cold_warm": True},
    {"label": "C8-64K", "concurrency": 8, "context": "64K",
     "cold_warm": True},
]


class ConfigError(ValueError):
    """The operator supplied an invalid configuration."""


class GeometryFailure(Exception):
    """A geometry or boundary check failed."""


def load_recipe(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"recipe not found: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConfigError(f"recipe parse error: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError("recipe root must be an object")
    return document


def serving_config(recipe: dict[str, Any]) -> dict[str, Any]:
    """Return the serving object or fail with a bounded configuration error."""
    serving = recipe.get("serving")
    if not isinstance(serving, dict):
        raise GeometryFailure("recipe serving must be an object")
    return serving


def verify_geometry(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify the LMCache CS512 block-256 geometry from the recipe.

    Checks:
    - chunk_size == 512 (CS512)
    - parent_chunk_size == 256 (block-256)
    - L1 is lazy (lazy allocation, not eager)
    - eviction policy is LRU
    - transfer mode is lmcache_driven
    """
    lmcache = serving_config(recipe).get("lmcache")
    if not isinstance(lmcache, dict):
        raise GeometryFailure("recipe serving.lmcache is not an object")

    checks: list[dict[str, Any]] = []

    chunk_size = lmcache.get("chunk_size")
    checks.append({
        "check": "chunk_size_is_512",
        "expected": REQUIRED_CHUNK_SIZE,
        "observed": chunk_size,
        "passed": chunk_size == REQUIRED_CHUNK_SIZE,
    })

    parent_chunk_size = lmcache.get("parent_chunk_size")
    checks.append({
        "check": "parent_chunk_size_is_256",
        "expected": REQUIRED_PARENT_CHUNK_SIZE,
        "observed": parent_chunk_size,
        "passed": parent_chunk_size == REQUIRED_PARENT_CHUNK_SIZE,
    })

    l1_lazy = lmcache.get("l1_lazy")
    checks.append({
        "check": "l1_is_lazy",
        "expected": True,
        "observed": l1_lazy,
        "passed": l1_lazy is True,
    })

    eviction = lmcache.get("eviction_policy")
    checks.append({
        "check": "eviction_policy_is_lru",
        "expected": "LRU",
        "observed": eviction,
        "passed": eviction == "LRU",
    })

    transfer_mode = lmcache.get("transfer_mode")
    checks.append({
        "check": "transfer_mode_is_lmcache_driven",
        "expected": "lmcache_driven",
        "observed": transfer_mode,
        "passed": transfer_mode == "lmcache_driven",
    })

    topology = lmcache.get("server_topology")
    checks.append({
        "check": "server_topology_one_per_rank",
        "expected": "one-local-server-per-rank",
        "observed": topology,
        "passed": topology == "one-local-server-per-rank",
    })

    failures = [c for c in checks if not c["passed"]]
    return {
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def verify_cache_isolation(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify SparkCache is disabled and vLLM prefix caching is enabled."""
    serving = serving_config(recipe)
    env = serving.get("environment")
    vllm_args = serving.get("vllm_args")
    if not isinstance(env, dict):
        raise GeometryFailure("recipe serving.environment must be an object")
    if not isinstance(vllm_args, list) or not all(
        isinstance(arg, str) for arg in vllm_args
    ):
        raise GeometryFailure("recipe serving.vllm_args must be a string list")
    sparkcache_value = env.get(SPARKCACHE_DISABLE_ENV)
    prefix_cache_enabled = PREFIX_CACHE_ENABLE_FLAG in vllm_args
    sparkcache_passed = sparkcache_value == SPARKCACHE_DISABLED_VALUE
    return {
        "check": "cache_isolation",
        "sparkcache_disabled": {
            "expected": SPARKCACHE_DISABLED_VALUE,
            "observed": sparkcache_value,
            "passed": sparkcache_passed,
        },
        "vllm_prefix_cache": {
            "expected": "enabled",
            "observed": "enabled" if prefix_cache_enabled else "absent",
            "passed": prefix_cache_enabled,
        },
        "passed": sparkcache_passed and prefix_cache_enabled,
        "note": (
            "SPARK_CONTEXT_CACHE_ENABLE=0 disables the separate SparkCache "
            "implementation, not vLLM prefix caching. The vLLM native prefix "
            "cache remains enabled through --enable-prefix-caching."
        ),
    }


def namespace_isolation_requirement() -> dict[str, Any]:
    """Describe the operator-owned namespace-isolation requirement."""
    return {
        "requirement": "namespace_isolation",
        "implementation_status": "operator-enforced",
        "automatically_verified": False,
        "note": NAMESPACE_ISOLATION_NOTE,
    }


def boundary_request_requirement() -> dict[str, Any]:
    """Describe exact token-count requests missing from the live gate."""
    return {
        "requirement": "boundary_token_counts",
        "implementation_status": "not-implemented",
        "automatically_verified": False,
        "boundaries": BOUNDARY_TOKEN_COUNTS,
        "note": (
            "A future live gate must test token counts at 511 (below chunk), "
            "512 (exact chunk), 513 (above chunk), 1024 (exact 2x chunk), "
            "and 1025 (above 2x chunk). The current cache acceptance gate "
            "does not execute these exact token-count requests."
        ),
    }


def dcp_hit_consensus_requirement() -> dict[str, Any]:
    """Describe the DCP hit-consensus collector missing from the live gate."""
    return {
        "requirement": "dcp_minimum_hit_consensus",
        "implementation_status": "not-implemented",
        "automatically_verified": False,
        "minimum_hit_per_rank": DCP_MINIMUM_HIT_PER_RANK,
        "note": (
            "A future live gate must collect per-rank hit counters after a "
            "warm probe and require at least one hit on every rank. The "
            "current LMCache status parser collects L1 occupancy and lock "
            "metrics, not cache-hit counters."
        ),
    }


def capacity_metric_coverage() -> dict[str, Any]:
    """Describe metrics collected by the live cache acceptance parser."""
    return {
        "requirement": "capacity_metric_coverage",
        "implementation_status": "partial",
        "automatically_verified": True,
        "collected_metrics": COLLECTED_L1_METRICS,
        "missing_metrics": ["eviction_count"],
        "note": (
            "The live acceptance parser collects occupancy, lock, and "
            "temporary-object metrics from every rank. The LMCache status "
            "contract used here does not expose an eviction counter, so live "
            "eviction attribution remains unsupported."
        ),
    }


def verify_all(recipe: dict[str, Any]) -> dict[str, Any]:
    """Run all offline-verifiable geometry and configuration checks."""
    geometry = verify_geometry(recipe)
    cache_isolation = verify_cache_isolation(recipe)
    offline_checks = [geometry, cache_isolation]
    all_passed = all(check["passed"] for check in offline_checks)

    return {
        "schema": SCHEMA,
        "verdict": "pass" if all_passed else "fail",
        "offline_checks": offline_checks,
        "live_requirements": [
            namespace_isolation_requirement(),
            boundary_request_requirement(),
            dcp_hit_consensus_requirement(),
            capacity_metric_coverage(),
        ],
        "evidence_scope": (
            "Offline geometry and configuration verification from the recipe; "
            "does not contact the cluster or verify live cache behavior"
        ),
        "canonical_recipe_sha256": hashlib.sha256(
            json.dumps(recipe, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def build_plan(recipe: dict[str, Any]) -> dict[str, Any]:
    """Build a plan for the full geometry + timing suite."""
    verify_report = verify_all(recipe)
    return {
        "schema": PLAN_SCHEMA,
        "offline_checks": verify_report,
        "live_timing_cells": TIMING_CELLS,
        "live_timing_note": (
            "C1/C2/C4/C8 at standard and 16K/64K contexts require a live "
            "cluster; they are delegated to exl3_cache_acceptance.py and "
            "the acceptance gate's performance matrix"
        ),
        "live_requirements": verify_report["live_requirements"],
        "evidence_scope": (
            "Offline plan only; no cluster contact, no cache reads, "
            "no API requests"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recipe",
        default=str(DEFAULT_RECIPE),
        help="path to the recipe JSON file",
    )
    parser.add_argument("command", choices=("verify", "plan"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        recipe = load_recipe(Path(args.recipe))
        report = (
            build_plan(recipe) if args.command == "plan" else verify_all(recipe)
        )
    except (ConfigError, GeometryFailure) as exc:
        print(f"exl3-cache-geometry: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.command == "plan":
        print(json.dumps(report, indent=2, sort_keys=True))
        return (
            EXIT_OK
            if report["offline_checks"]["verdict"] == "pass"
            else EXIT_FAIL
        )

    # verify
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "fail":
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
