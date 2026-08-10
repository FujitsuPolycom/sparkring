#!/usr/bin/env python3
"""Dry-run-first LMCache CS512 configuration and boundary verification gate.

Extends the cache acceptance runbook with offline-verifiable configuration
checks the LMCache CS512 recipe and cache-layer isolation. Physical/APC
geometry is a live-attested property and is never inferred from tuning-arm
provenance.
A companion plan mode discloses the live gates (boundary token counts,
object/TTFT evidence, C1/C2/C4/C8 and 16K/64K timing cells) that require
a live cluster.

This gate is OFFLINE: it reads the recipe and launch profile to verify
geometry and configuration.  It does not contact the cluster.  The live
timing and boundary probes are delegated to ``exl3_cache_acceptance.py``
and the acceptance gate's performance matrix.

**Verified vs planned.** The ``verify`` command reports two categories:

- ``verified_checks``: configuration facts read from the recipe that are
  actually checked offline (geometry, SparkCache disabled, APC enabled).
  These have ``passed: true/false``.
- ``planned_live_gates``: requirements that must be satisfied live but
  cannot be checked offline (boundary token counts, object/TTFT evidence
  after warm probe, cold/warm timing).  These have ``status: "planned"``
  and never report ``passed: true``.

Usage::

    # Offline: verify geometry and configuration from the recipe
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

# CS512 was selected against a predecessor whose LMCache chunk size was 256.
# This is provenance only. It is not vLLM's physical block size or proof of
# native APC alignment.
REQUIRED_CHUNK_SIZE = 512
REQUIRED_PREDECESSOR_CHUNK_SIZE = 256

# Boundary token counts to verify: the cache must handle sequences that
# span exactly, just below, and just above the live-attested DCP-global APC
# alignment and LMCache chunk boundaries. A reusable prefix excludes the final
# prompt token, hence 256 prompt tokens do not yet supply a full 256-token APC
# unit; 257 do.
BOUNDARY_TOKEN_COUNTS = [255, 256, 257, 511, 512, 513, 1024, 1025]

# SparkCache isolation: SPARK_CONTEXT_CACHE_ENABLE=0 disables SparkCache
# (the separate sparkcache/ implementation) so it does not interfere
# with LMCache attribution. This is NOT APC isolation — APC (vLLM's
# native prefix cache via --enable-prefix-caching) is enabled in this
# profile and is distinct from SparkCache.
SPARKCACHE_DISABLE_ENV = "SPARK_CONTEXT_CACHE_ENABLE"
SPARKCACHE_DISABLED_VALUE = "0"

# APC (native prefix cache) is enabled via --enable-prefix-caching in
# the vLLM argument vector. This is distinct from SparkCache.
APC_ENABLE_FLAG = "--enable-prefix-caching"

# Namespace isolation: the probe ID must be unique per acceptance run to
# prevent cache namespace collision across runs.
NAMESPACE_ISOLATION_NOTE = (
    "The probe ID must be unique per acceptance run; reusing a probe ID "
    "from a prior run produces cache hits that belong to the old namespace, "
    "not the current run's cold/warm attribution"
)

# Metrics actually available from the LMCache /status endpoint.
# The /status l1_manager object exposes these fields. This is the
# observed schema, confirmed by exl3_cache_acceptance.py's
# parse_launcher_status, not an aspirational list.
STATUS_AVAILABLE_METRICS = [
    "is_healthy",
    "total_object_count",
    "memory_used_bytes",
    "write_locked_count",
    "read_locked_count",
    "temporary_count",
    "registered_gpu_ids",
]

# Metrics NOT available from /status. eviction_count is not exposed by the
# current LMCache server /status schema. Do not claim it can be collected.
STATUS_UNAVAILABLE_METRICS = [
    "eviction_count",
]

# DCP consensus evidence: the /status endpoint does NOT expose per-rank
# cache-hit counters. DCP minimum-hit consensus cannot be read from
# /status. The truthful evidence path is:
# 1. Object counts (total_object_count > 0 on all ranks after warm probe)
#    proves objects were stored, not that they were hit.
# 2. TTFT ratio (warm < cold) from the live cache acceptance gate
#    provides timing evidence of cache reuse, not a hit counter.
# 3. Connector hit-length logs (if present in the serving image) may
#    provide per-request hit attribution; this is not guaranteed.
DCP_CONSENSUS_EVIDENCE_PATH = [
    "total_object_count > 0 on all four ranks after warm probe (from /status)",
    "TTFT ratio warm < cold (from exl3_cache_acceptance.py timing samples)",
    "connector hit-length logs if present in the serving image (not guaranteed)",
]
DCP_CONSENSUS_LIMITATION = (
    "The LMCache /status endpoint does not expose per-rank cache-hit "
    "counters. DCP minimum-hit consensus cannot be read from /status. "
    "Object counts prove objects were stored, not that they were hit. "
    "TTFT ratios provide timing evidence of reuse, not a hit counter."
)

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


# ---------------------------------------------------------------------------
# Verified checks — configuration facts read from the recipe
# ---------------------------------------------------------------------------


def verify_geometry(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify CS512 configuration and predecessor provenance from the recipe.

    Checks:
    - chunk_size == 512 (CS512)
    - predecessor_chunk_size == 256 (historical tuning-arm provenance only)
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

    predecessor_chunk_size = lmcache.get("predecessor_chunk_size")
    checks.append({
        "check": "predecessor_chunk_size_is_256",
        "scope": "historical-tuning-arm-provenance-not-runtime-geometry",
        "expected": REQUIRED_PREDECESSOR_CHUNK_SIZE,
        "observed": predecessor_chunk_size,
        "passed": predecessor_chunk_size == REQUIRED_PREDECESSOR_CHUNK_SIZE,
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
        "category": "verified",
        "check": "geometry",
        "checks": checks,
        "passed": not failures,
        "failures": failures,
    }


def verify_cache_isolation(recipe: dict[str, Any]) -> dict[str, Any]:
    """Verify SparkCache is disabled and APC (native prefix cache) is enabled.

    These are distinct mechanisms:
    - SparkCache (SPARK_CONTEXT_CACHE_ENABLE) is the separate sparkcache/
      implementation; it must be disabled (=0) to avoid interfering with
      LMCache attribution.
    - APC (vLLM --enable-prefix-caching) is the native prefix cache; it
      is enabled in this profile and is a different layer from SparkCache.

    Native APC remains active during an engine-only restart (LMCache servers
    preserved). To clear native APC while retaining LMCache objects, the
    live procedure must restart engines (clearing APC) while keeping LMCache
    servers alive, or use another proven isolation method.
    """
    env = recipe.get("serving", {}).get("environment", {})
    vllm_args = recipe.get("serving", {}).get("vllm_args", [])
    sparkcache_value = env.get(SPARKCACHE_DISABLE_ENV)
    apc_enabled = APC_ENABLE_FLAG in vllm_args
    sparkcache_passed = sparkcache_value == SPARKCACHE_DISABLED_VALUE
    apc_passed = apc_enabled is True
    return {
        "category": "verified",
        "check": "cache_isolation",
        "sparkcache_disabled": {
            "expected": SPARKCACHE_DISABLED_VALUE,
            "observed": sparkcache_value,
            "passed": sparkcache_passed,
        },
        "apc_native_prefix_cache": {
            "expected": "enabled",
            "observed": "enabled" if apc_enabled else "absent",
            "passed": apc_passed,
        },
        "passed": sparkcache_passed and apc_passed,
        "note": (
            "SPARK_CONTEXT_CACHE_ENABLE=0 disables SparkCache (the "
            "separate sparkcache/ implementation), not APC. APC (vLLM "
            "native prefix cache via --enable-prefix-caching) is enabled "
            "and is a distinct layer from SparkCache."
        ),
        "native_apc_clearing_procedure": (
            "Native APC persists in-engine and is not cleared by LMCache "
            "server restart. To clear APC while retaining LMCache objects, "
            "restart engines (which clears APC state) while keeping LMCache "
            "servers alive. The live cache acceptance gate's engine-only "
            "restart phase exercises this boundary."
        ),
    }


# ---------------------------------------------------------------------------
# Planned live gates — requirements that cannot be checked offline
# ---------------------------------------------------------------------------


def plan_namespace_isolation() -> dict[str, Any]:
    """Disclose namespace isolation requirements for live runs."""
    return {
        "category": "planned_live",
        "check": "namespace_isolation",
        "status": "planned",
        "note": NAMESPACE_ISOLATION_NOTE,
        "requirement": (
            "Each acceptance run must use a unique probe ID; "
            "the cache acceptance gate enforces this at run time"
        ),
    }


def plan_boundary_tests() -> dict[str, Any]:
    """Disclose the boundary token counts that the live gate must test."""
    return {
        "category": "planned_live",
        "check": "boundary_token_counts",
        "status": "planned",
        "boundaries": BOUNDARY_TOKEN_COUNTS,
        "required_live_geometry": {
            "engine_block_rows_per_dcp_rank": "from vllm:cache_config_info",
            "dcp_size": "from exact live argv/config attestation",
            "kv_group_global_tokens_per_manager_block": (
                "engine block rows multiplied by each KV group's DCP shard count"
            ),
            "native_apc_hit_alignment_tokens_global": (
                "LCM of all KV-group global manager-block spans"
            ),
            "lmcache_chunk_tokens_global": "from all four LMCache servers",
        },
        "note": (
            "The live gate must first attest physical, DCP-global APC, and "
            "LMCache chunk geometry. It then tests 255/256/257 around one "
            "global APC unit and 511/512/513/1024/1025 around LMCache chunks. "
            "Because reusable prefix length excludes the final prompt token, "
            "257 prompt tokens are required for one reusable 256-token APC unit."
        ),
    }


def plan_dcp_consensus() -> dict[str, Any]:
    """Disclose DCP consensus evidence requirements for live runs.

    The /status endpoint does NOT expose per-rank cache-hit counters.
    Object counts and TTFT ratios are the truthful evidence path.
    """
    return {
        "category": "planned_live",
        "check": "dcp_consensus_evidence",
        "status": "planned",
        "evidence_path": DCP_CONSENSUS_EVIDENCE_PATH,
        "limitation": DCP_CONSENSUS_LIMITATION,
    }


def plan_capacity_metrics() -> dict[str, Any]:
    """Disclose the capacity metrics available from /status.

    Only the fields actually present in the /status schema are listed as
    available. eviction_count is explicitly marked unavailable.
    """
    return {
        "category": "planned_live",
        "check": "capacity_metrics",
        "status": "planned",
        "available_from_status": STATUS_AVAILABLE_METRICS,
        "unavailable_from_status": STATUS_UNAVAILABLE_METRICS,
        "note": (
            "The live gate collects available metrics from every rank's "
            "LMCache /status endpoint after each probe phase. "
            "eviction_count is not exposed by the current /status schema "
            "and cannot be claimed as collectible."
        ),
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def verify_all(recipe: dict[str, Any]) -> dict[str, Any]:
    """Run all offline-verifiable checks and list planned live gates.

    Returns a report with two sections:
    - ``verified_checks``: configuration facts with ``passed`` verdicts
    - ``planned_live_gates``: future live requirements with ``status: planned``
    """
    verified = [
        verify_geometry(recipe),
        verify_cache_isolation(recipe),
    ]
    planned = [
        plan_namespace_isolation(),
        plan_boundary_tests(),
        plan_dcp_consensus(),
        plan_capacity_metrics(),
    ]

    all_verified_passed = all(c.get("passed", False) for c in verified)

    return {
        "schema": SCHEMA,
        "verdict": "pass" if all_verified_passed else "fail",
        "verified_checks": verified,
        "planned_live_gates": planned,
        "evidence_scope": (
            "Offline configuration verification from the recipe only. "
            "Verified checks are configuration facts; planned live gates "
            "require a live cluster and are not passed by this offline gate."
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
        "verified_checks": verify_report["verified_checks"],
        "planned_live_gates": verify_report["planned_live_gates"],
        "live_timing_cells": TIMING_CELLS,
        "live_timing_note": (
            "C1/C2/C4/C8 at standard and 16K/64K contexts require a live "
            "cluster; they are delegated to exl3_cache_acceptance.py and "
            "the acceptance gate's performance matrix"
        ),
        "boundary_live_note": (
            "Physical/DCP-global geometry and boundary token counts "
            "(255/256/257/511/512/513/1024/1025) require live "
            "API requests; the offline gate declares them, the live gate "
            "must execute them"
        ),
        "dcp_live_note": (
            "DCP consensus requires live evidence: object counts > 0 on "
            "all ranks plus TTFT ratio warm < cold; /status does not "
            "expose per-rank hit counters"
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
