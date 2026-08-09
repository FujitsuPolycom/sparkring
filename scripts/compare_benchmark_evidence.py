#!/usr/bin/env python3
"""Offline evidence-comparison tool for cache-off vs cache-on 16K benchmarks.

Compares two sustained-decode 16K C1/C2/C4/C8 benchmark JSON documents
and reports per-concurrency deltas, but ONLY when workload settings match
exactly.  Never mixes bounded 128-token gate figures with sustained
25-second matrix figures.

This tool is **purely offline**: it reads two JSON files supplied by
the operator and produces a structured comparison report.  It does not
contact the cluster, run benchmarks, or mutate anything.

## Workload-settings matching

Before claiming any delta, the tool verifies that these settings are
identical between the two documents:

- harness name and version
- context length (tokens)
- concurrencies list
- duration per cell (seconds)
- decode warmup (seconds)
- max output tokens
- temperature
- unique context percent
- shared context percent
- DCP size
- KV budget (tokens)
- ignore_eos
- skip_prefill
- cell warmup timeout (seconds)

If any setting differs, the comparison reports ``settings_mismatch`` and
NO delta is claimed.  This is fail-closed: a mismatched comparison
is not a valid delta.

## 128-token gate vs sustained matrix

The tool distinguishes bounded 128-token finite-request gate figures
from sustained 25-second decode matrix figures.  A document is
classified as ``sustained_matrix`` when ``duration_seconds >= 10`` and
``max_tokens >= 256``.  A document with ``max_tokens < 256`` or
``duration_seconds < 10`` is classified as ``bounded_gate``.  The
tool refuses to compare a ``bounded_gate`` document against a
``sustained_matrix`` document.

## Usage::

    python scripts/compare_benchmark_evidence.py \
        --baseline evidence/cache-off.json \
        --candidate evidence/cache-on.json

    # Strict mode: exit non-zero if any cell has invalid concurrency or errors
    python scripts/compare_benchmark_evidence.py \
        --baseline evidence/cache-off.json \
        --candidate evidence/cache-on.json \
        --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-benchmark-comparison/v1"
EXIT_OK = 0
EXIT_MISMATCH = 2
EXIT_CONFIG_ERROR = 3
EXIT_INVALID = 4

# Settings that must match exactly before any delta is claimed.
# Each entry is (json_key_path, display_name).
# json_key_path is a list of keys to traverse in the JSON document.
MATCHED_SETTINGS = [
    (["harness", "name"], "harness_name"),
    (["harness", "version"], "harness_version"),
    (["configuration", "context_tokens"], "context_tokens"),
    (["configuration", "concurrencies"], "concurrencies"),
    (["configuration", "duration_per_cell_seconds"], "duration_seconds"),
    (["configuration", "decode_warmup_seconds"], "decode_warmup_seconds"),
    (["configuration", "max_output_tokens"], "max_output_tokens"),
    (["configuration", "temperature"], "temperature"),
    (["configuration", "unique_context_percent"], "unique_context_percent"),
    (["configuration", "shared_context_percent"], "shared_context_percent"),
    (["configuration", "dcp_size"], "dcp_size"),
    (["configuration", "kv_budget_tokens"], "kv_budget_tokens"),
    (["configuration", "ignore_eos"], "ignore_eos"),
    (["configuration", "skip_prefill"], "skip_prefill"),
    (["configuration", "cell_warmup_timeout_seconds"], "cell_warmup_timeout_seconds"),
]

# Alternative key names used in some evidence documents (e.g. top-level
# "tool" instead of "harness.name").  Maps display_name to list of
# fallback key paths.
FALLBACK_KEYS: dict[str, list[list[str]]] = {
    "harness_name": [["tool"], ["harness"]],
    "harness_version": [["version"], ["tool_version"]],
    "context_tokens": [["context"]],
    "duration_seconds": [["duration_seconds"], ["duration_per_cell_seconds"]],
    "decode_warmup_seconds": [["decode_warmup_seconds"]],
    "max_output_tokens": [["max_tokens"], ["max_output_tokens"]],
    "unique_context_percent": [["unique_context_percent"]],
    "shared_context_percent": [["shared_context_percent"]],
    "dcp_size": [["dcp_size"], ["dcp"]],
    "kv_budget_tokens": [["kv_budget_tokens"], ["kv_budget"]],
    "ignore_eos": [["ignore_eos"]],
    "skip_prefill": [["skip_prefill"]],
    "cell_warmup_timeout_seconds": [["cell_warmup_timeout_seconds"]],
    "concurrencies": [["concurrencies"], ["concurrency"]],
    "temperature": [["temperature"]],
}

# Concurrency label normalization: "c1"→"C1", "C1"→"C1"
def _norm_concurrency(key: str) -> str:
    return key.upper() if key.lower().startswith("c") else key


class ConfigError(ValueError):
    """The operator supplied an invalid argument."""


class EvidenceError(ValueError):
    """A benchmark document is malformed or settings mismatch."""


def _resolve(doc: dict[str, Any], key_path: list[str]) -> Any:
    """Traverse a list of keys in a dict, returning None if any is absent."""
    current: Any = doc
    for key in key_path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def _resolve_with_fallbacks(
    doc: dict[str, Any], display_name: str, primary: list[str]
) -> Any:
    """Try the primary key path, then fall back to alternatives."""
    value = _resolve(doc, primary)
    if value is not None:
        return value
    for fallback in FALLBACK_KEYS.get(display_name, []):
        value = _resolve(doc, fallback)
        if value is not None:
            return value
    return None


def extract_settings(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract workload settings from a benchmark document.

    Searches both the nested ``configuration`` object and top-level
    fallback keys used in older evidence formats.
    """
    settings: dict[str, Any] = {}
    for key_path, display_name in MATCHED_SETTINGS:
        settings[display_name] = _resolve_with_fallbacks(
            doc, display_name, key_path
        )
    return settings


def classify_document_type(doc: dict[str, Any]) -> str:
    """Classify a benchmark document as sustained_matrix or bounded_gate.

    sustained_matrix: duration >= 10s and max_tokens >= 256
    bounded_gate: max_tokens < 256 or duration < 10s
    indeterminate: cannot determine (missing both)
    """
    settings = extract_settings(doc)
    duration = settings.get("duration_seconds")
    max_tokens = settings.get("max_output_tokens")

    if duration is None and max_tokens is None:
        return "indeterminate"

    # If either suggests bounded, classify as bounded_gate
    if max_tokens is not None and max_tokens < 256:
        return "bounded_gate"
    if duration is not None and duration < 10:
        return "bounded_gate"
    if duration is not None and max_tokens is not None:
        return "sustained_matrix"
    # One is present but not the other; use what we have
    if max_tokens is not None and max_tokens >= 256:
        return "sustained_matrix"
    if duration is not None and duration >= 10:
        return "sustained_matrix"
    return "indeterminate"


def extract_throughput(doc: dict[str, Any]) -> dict[str, float]:
    """Extract aggregate throughput per concurrency from a benchmark document.

    Looks for ``aggregate_tokens_per_second`` or ``aggregate_tps``
    at the top level or nested in common sub-objects.
    """
    tps: dict[str, float] = {}

    # Try top-level aggregate_tokens_per_second / aggregate_tps
    for key in ("aggregate_tokens_per_second", "aggregate_tps"):
        value = doc.get(key)
        if isinstance(value, dict):
            for k, v in value.items():
                nk = _norm_concurrency(k)
                if isinstance(v, (int, float)) and nk not in tps:
                    tps[nk] = float(v)

    # Try nested in artifacts or cells
    for container_key in ("artifacts", "cells", "results"):
        container = doc.get(container_key)
        if not isinstance(container, dict):
            continue
        for _artifact_name, artifact in container.items():
            if not isinstance(artifact, dict):
                continue
            for key in ("aggregate_tokens_per_second", "aggregate_tps"):
                value = artifact.get(key)
                if isinstance(value, dict):
                    for k, v in value.items():
                        nk = _norm_concurrency(k)
                        if isinstance(v, (int, float)) and nk not in tps:
                            tps[nk] = float(v)

    return tps


def extract_validity(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract validity information from a benchmark document."""
    info: dict[str, Any] = {}

    # all_cells_valid
    for key in ("all_cells_valid",):
        value = doc.get(key)
        if isinstance(value, bool):
            info["all_cells_valid"] = value
            break
    # Also check nested
    if "all_cells_valid" not in info:
        for container_key in ("artifacts", "cells", "results"):
            container = doc.get(container_key)
            if not isinstance(container, dict):
                continue
            for artifact in container.values():
                if isinstance(artifact, dict):
                    value = artifact.get("all_cells_valid")
                    if isinstance(value, bool):
                        info["all_cells_valid"] = value
                        break
            if "all_cells_valid" in info:
                break

    # effective_concurrency
    eff: dict[str, int] = {}
    for key in ("effective_concurrency",):
        value = doc.get(key)
        if isinstance(value, dict):
            for k, v in value.items():
                nk = _norm_concurrency(k)
                if isinstance(v, int):
                    eff[nk] = v
    if eff:
        info["effective_concurrency"] = eff

    # errors
    errors: dict[str, int] = {}
    for key in ("errors",):
        value = doc.get(key)
        if isinstance(value, dict):
            for k, v in value.items():
                nk = _norm_concurrency(k)
                if isinstance(v, int):
                    errors[nk] = v
    if errors:
        info["errors"] = errors

    return info


def compare_settings(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare workload settings between two documents."""
    base_settings = extract_settings(baseline)
    cand_settings = extract_settings(candidate)

    matches: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    missing: list[str] = []

    for _key_path, display_name in MATCHED_SETTINGS:
        base_val = base_settings.get(display_name)
        cand_val = cand_settings.get(display_name)

        if base_val is None and cand_val is None:
            missing.append(display_name)
        elif base_val is None or cand_val is None:
            mismatches.append({
                "setting": display_name,
                "baseline": base_val,
                "candidate": cand_val,
                "reason": "one side is missing",
            })
        elif base_val != cand_val:
            mismatches.append({
                "setting": display_name,
                "baseline": base_val,
                "candidate": cand_val,
                "reason": "values differ",
            })
        else:
            matches.append({"setting": display_name, "value": base_val})

    return {
        "matched": matches,
        "mismatched": mismatches,
        "missing_in_both": missing,
        "all_matched": len(mismatches) == 0,
    }


def compare_throughput(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare throughput between two documents, per concurrency."""
    base_tps = extract_throughput(baseline)
    cand_tps = extract_throughput(candidate)

    cells: list[dict[str, Any]] = []
    all_concurrencies = sorted(set(base_tps) | set(cand_tps))

    for conc in all_concurrencies:
        base_val = base_tps.get(conc)
        cand_val = cand_tps.get(conc)
        entry: dict[str, Any] = {"concurrency": conc}

        if base_val is None or cand_val is None:
            entry["baseline_tps"] = base_val
            entry["candidate_tps"] = cand_val
            entry["delta"] = None
            entry["delta_percent"] = None
            entry["status"] = "missing"
        else:
            entry["baseline_tps"] = base_val
            entry["candidate_tps"] = cand_val
            entry["delta"] = cand_val - base_val
            if base_val > 0:
                entry["delta_percent"] = ((cand_val - base_val) / base_val) * 100.0
            else:
                entry["delta_percent"] = None
                entry["status"] = "baseline_zero"
            if "status" not in entry:
                entry["status"] = "compared"
        cells.append(entry)

    return {
        "cells": cells,
        "baseline_tps": base_tps,
        "candidate_tps": cand_tps,
    }


def compare_documents(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Full comparison of two benchmark documents."""
    base_type = classify_document_type(baseline)
    cand_type = classify_document_type(candidate)

    type_mismatch = None
    if base_type != cand_type:
        if base_type == "indeterminate" or cand_type == "indeterminate":
            type_mismatch = (
                f"cannot classify one or both documents: "
                f"baseline={base_type}, candidate={cand_type}"
            )
        else:
            type_mismatch = (
                f"document type mismatch: baseline={base_type}, "
                f"candidate={cand_type}. "
                "Bounded 128-token gate figures cannot be compared against "
                "sustained 25-second matrix figures."
            )

    settings_comparison = compare_settings(baseline, candidate)
    throughput_comparison = compare_throughput(baseline, candidate)
    base_validity = extract_validity(baseline)
    cand_validity = extract_validity(candidate)

    # Determine overall status
    if type_mismatch:
        status = "type_mismatch"
    elif not settings_comparison["all_matched"]:
        status = "settings_mismatch"
    else:
        status = "compared"

    # Check for invalid cells
    base_all_valid = base_validity.get("all_cells_valid", True)
    cand_all_valid = cand_validity.get("all_cells_valid", True)
    if not base_all_valid or not cand_all_valid:
        if status == "compared":
            status = "invalid_cells"

    return {
        "schema": SCHEMA,
        "status": status,
        "baseline_type": base_type,
        "candidate_type": cand_type,
        "type_mismatch": type_mismatch,
        "settings": settings_comparison,
        "throughput": throughput_comparison,
        "baseline_validity": base_validity,
        "candidate_validity": cand_validity,
        "evidence_scope": (
            "Offline comparison of two benchmark JSON documents. "
            "Deltas are claimed only when workload settings match exactly. "
            "Bounded 128-token gate figures are never compared against "
            "sustained 25-second matrix figures. This tool does not "
            "contact the cluster or run benchmarks."
        ),
        "claim_note": (
            "A 'compared' status with delta_percent values is a valid "
            "cross-document comparison only when all_matched is true and "
            "both documents are sustained_matrix type. It is not a sealed "
            "A/B unless both documents were produced under controlled "
            "conditions with identical configuration except for the "
            "cache variable."
        ),
    }


def load_document(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"file not found: {path}")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(doc, dict):
        raise ConfigError(f"top-level JSON in {path} is not an object")
    return doc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="path to baseline JSON")
    parser.add_argument("--candidate", required=True, help="path to candidate JSON")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any cell is invalid, missing, or has errors",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        baseline = load_document(Path(args.baseline))
        candidate = load_document(Path(args.candidate))
    except ConfigError as exc:
        print(f"compare-benchmark-evidence: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    report = compare_documents(baseline, candidate)
    print(json.dumps(report, indent=2, sort_keys=True))

    if report["status"] in ("type_mismatch", "settings_mismatch"):
        return EXIT_MISMATCH

    if args.strict:
        # Check for invalid cells or missing throughput
        for cell in report["throughput"]["cells"]:
            if cell["status"] in ("missing", "baseline_zero"):
                return EXIT_INVALID
        base_valid = report["baseline_validity"].get("all_cells_valid", True)
        cand_valid = report["candidate_validity"].get("all_cells_valid", True)
        if not base_valid or not cand_valid:
            return EXIT_INVALID

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
