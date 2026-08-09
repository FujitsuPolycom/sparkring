#!/usr/bin/env python3
"""Dry-run-first LMCache CS512 geometry and boundary verification gate.

Extends the cache acceptance runbook with offline-verifiable configuration
checks for the LMCache CS512 block-256 geometry, token-count boundaries,
DCP minimum-hit consensus, APC isolation, namespace isolation, and
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

# APC (adaptive prefix cache) isolation: the profile must explicitly disable
# SparkCache (SPARK_CONTEXT_CACHE_ENABLE=0) to ensure APC isolation from
# the LMCache layer.
APC_ISOLATION_ENV = "SPARK_CONTEXT_CACHE_ENABLE"
APC_ISOLATION_REQUIRED_VALUE = "0"

# Namespace isolation: the probe ID must be unique per acceptance run to
# prevent cache namespace collision across runs.
NAMESPACE_ISOLATION_NOTE = (
    "The probe ID must be unique per acceptance run; reusing a probe ID "
    "from a prior run produces cache hits that belong to the old namespace, "
    "not the current run's cold/warm attribution"
)

# Capacity/eviction metrics that the live gate should collect.
CAPACITY_METRICS = [
    "total_object_count",
    "memory_used_bytes",
    "write_locked_count",
    "read_locked_count",
    "temporary_count",
    "eviction_count",
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
    except json.JSONDecodeError as exc:
        raise ConfigError(f"recipe parse error: {exc}") from exc
    if not isinstance(document, dict):
        raise ConfigError("recipe root must be an object")
    return document


def verify_geometry(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify the LMCache CS512 block-256 geometry from the recipe.

    Checks:
    - chunk_size == 512 (CS512)
    - parent_chunk_size == 256 (block-256)
    - L1 is lazy (lazy allocation, not eager)
    - eviction policy is LRU
    - transfer mode is lmcache_driven
    """
    lmcache = recipe.get("serving", {}).get("lmcache", {})
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

    eviction = lmcache.get("eviction_policy", "LRU")
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


def verify_apc_isolation(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify APC (SparkCache) is disabled to isolate from LMCache."""
    env = recipe.get("serving", {}).get("environment", {})
    value = env.get(APC_ISOLATION_ENV)
    return {
        "check": "apc_isolation",
        "expected": APC_ISOLATION_REQUIRED_VALUE,
        "observed": value,
        "passed": value == APC_ISOLATION_REQUIRED_VALUE,
        "note": (
            "SPARK_CONTEXT_CACHE_ENABLE=0 ensures native prefix cache "
            "(SparkCache/APC) does not interfere with LMCache attribution"
        ),
    }


def verify_namespace_isolation() -> dict[str, Any]:
    """Disclose namespace isolation requirements."""
    return {
        "check": "namespace_isolation",
        "passed": True,  # structural: the requirement is documented
        "note": NAMESPACE_ISOLATION_NOTE,
        "requirement": (
            "Each acceptance run must use a unique probe ID; "
            "the cache acceptance gate enforces this at run time"
        ),
    }


def verify_boundary_plan() -> dict[str, Any]:
    """Disclose the boundary token counts that the live gate must test."""
    return {
        "check": "boundary_token_counts",
        "passed": True,  # structural: the boundaries are declared
        "boundaries": BOUNDARY_TOKEN_COUNTS,
        "note": (
            "The live gate must test token counts at 511 (below chunk), "
            "512 (exact chunk), 513 (above chunk), 1024 (exact 2x chunk), "
            "and 1025 (above 2x chunk) to verify chunk-boundary handling"
        ),
    }


def verify_dcp_consensus() -> dict[str, Any]:
    """Disclose DCP minimum-hit consensus requirements."""
    return {
        "check": "dcp_minimum_hit_consensus",
        "passed": True,  # structural: the requirement is documented
        "minimum_hit_per_rank": DCP_MINIMUM_HIT_PER_RANK,
        "note": (
            "After a warm probe, all four ranks must report at least "
            f"{DCP_MINIMUM_HIT_PER_RANK} cache hit; a rank with zero hits "
            "has a DCP cache consensus failure"
        ),
    }


def verify_capacity_metrics() -> dict[str, Any]:
    """Disclose the capacity/eviction metrics the live gate must collect."""
    return {
        "check": "capacity_eviction_metrics",
        "passed": True,  # structural: the metrics are declared
        "required_metrics": CAPACITY_METRICS,
        "note": (
            "The live gate must collect these metrics from every rank's "
            "LMCache /status endpoint after each probe phase"
        ),
    }


def verify_all(recipe: dict[str, Any]) -> dict[str, Any]:
    """Run all offline-verifiable geometry and configuration checks."""
    geometry = verify_geometry(recipe)
    apc = verify_apc_isolation(recipe)
    namespace = verify_namespace_isolation()
    boundaries = verify_boundary_plan()
    dcp = verify_dcp_consensus()
    capacity = verify_capacity_metrics()

    all_checks = [geometry, apc, namespace, boundaries, dcp, capacity]
    all_passed = all(c.get("passed", False) for c in all_checks)

    return {
        "schema": SCHEMA,
        "verdict": "pass" if all_passed else "fail",
        "checks": all_checks,
        "evidence_scope": (
            "Offline geometry and configuration verification from the recipe; "
            "does not contact the cluster or verify live cache behavior"
        ),
        "recipe_sha256": hashlib.sha256(
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
        "boundary_live_note": (
            "Boundary token counts (511/512/513/1024/1025) require live "
            "API requests; the offline gate declares them, the live gate "
            "must execute them"
        ),
        "dcp_live_note": (
            "DCP minimum-hit consensus requires reading all four ranks' "
            "LMCache /status after a warm probe; this is a live check"
        ),
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
    except ConfigError as exc:
        print(f"exl3-cache-geometry: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    if args.command == "plan":
        report = build_plan(recipe)
        print(json.dumps(report, indent=2, sort_keys=True))
        return EXIT_OK

    # verify
    report = verify_all(recipe)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["verdict"] == "fail":
        return EXIT_FAIL
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
