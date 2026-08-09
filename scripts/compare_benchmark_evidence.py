#!/usr/bin/env python3
"""Offline evidence-comparison tool for cache-off vs cache-on 16K benchmarks.

Compares two sustained-decode 16K C1/C2/C4/C8 benchmark JSON documents
produced by ``llm_decode_bench.py`` v0.4.31 and reports per-concurrency
deltas, but ONLY when document type, workload settings, validity, and exact
cell coverage ALL pass.  Never mixes bounded 128-token gate figures with
sustained 25-second matrix figures.

This tool is **purely offline**: it reads two JSON files supplied by
the operator and produces a structured comparison report.  It does not
contact the cluster, run benchmarks, or mutate anything.

## Raw schema (llm_decode_bench v0.4.31)

- ``metadata``: top-level object with ``version``, ``decode_mode``,
  ``duration_per_test``, ``max_tokens``, ``temperature``,
  ``decode_warmup_seconds``, ``cell_warmup_timeout_seconds``,
  ``unique_context_percent``, ``shared_context_percent``, ``dcp_size``,
  ``max_total_tokens``, ``skip_prefill``, ``ignore_eos``,
  ``concurrency_levels``, ``context_lengths``.
- ``results``: array of per-cell objects, each with ``concurrency``,
  ``context_tokens``, ``aggregate_tps``, ``num_errors``,
  ``effective_concurrency``, ``benchmark_mode``, ``measurement_seconds``,
  ``underfilled``, ``warmup_timed_out``, ``capacity_limited``.
- ``summary_table``: ``{context_tokens: {concurrency: aggregate_tps}}``.

## Fail-closed rules

1. Both documents must be ``sustained_matrix``.  Two ``bounded_gate``
   or two ``indeterminate`` documents must NOT compare.
2. Each document must have exactly one 16K context and exactly C1/C2/C4/C8
   result coverage consistent with ``metadata.concurrency_levels``.
   Absent, duplicated, or unexpected/multi-context results fail closed.
3. Missing ``num_errors``, ``effective_concurrency``, ``underfilled``,
   ``warmup_timed_out``, ``capacity_limited``, ``benchmark_mode``,
   ``measurement_seconds``, or ``max_tokens`` is indeterminate/invalid,
   never default-zero/true.
4. No numeric deltas are computed or emitted until document type,
   complete settings, validity, and exact cell coverage all pass.
5. Classification requires ``metadata.decode_mode == "duration"`` and
   each result ``benchmark_mode == "duration"``, and requires both
   ``duration_per_test`` and ``max_tokens`` (no inference from one).
6. Multi-context documents and any mismatch between metadata context
   list, result contexts, and summary table are rejected.

## Usage::

    python scripts/compare_benchmark_evidence.py \\
        --baseline evidence/cache-off.json \\
        --candidate evidence/cache-on.json

    python scripts/compare_benchmark_evidence.py \\
        --baseline evidence/cache-off.json \\
        --candidate evidence/cache-on.json \\
        --strict
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-benchmark-comparison/v3"
EXIT_OK = 0
EXIT_MISMATCH = 2
EXIT_CONFIG_ERROR = 3
EXIT_INVALID = 4

# Required concurrency cells for a valid 16K matrix comparison.
REQUIRED_CONCURRENCIES = [1, 2, 4, 8]
REQUIRED_CONTEXT = 16384

# Settings that must match exactly before any delta is claimed.
MATCHED_SETTINGS: list[tuple[str, str]] = [
    ("version", "harness_version"),
    ("context_lengths", "context_tokens"),
    ("concurrency_levels", "concurrencies"),
    ("duration_per_test", "duration_seconds"),
    ("decode_warmup_seconds", "decode_warmup_seconds"),
    ("max_tokens", "max_output_tokens"),
    ("temperature", "temperature"),
    ("unique_context_percent", "unique_context_percent"),
    ("shared_context_percent", "shared_context_percent"),
    ("dcp_size", "dcp_size"),
    ("max_total_tokens", "kv_budget_tokens"),
    ("ignore_eos", "ignore_eos"),
    ("skip_prefill", "skip_prefill"),
    ("cell_warmup_timeout_seconds", "cell_warmup_timeout_seconds"),
]

# Validity fields that must be present in each result cell.
# Each is (field_name, expected_type, invalid_if_missing).
VALIDITY_FIELDS: list[tuple[str, tuple[type, ...]]] = [
    ("num_errors", (int,)),
    ("effective_concurrency", (int,)),
    ("underfilled", (bool,)),
    ("warmup_timed_out", (bool,)),
    ("capacity_limited", (bool,)),
    ("benchmark_mode", (str,)),
    ("measurement_seconds", (int, float)),
]


class ConfigError(ValueError):
    """The operator supplied an invalid argument."""


class EvidenceError(ValueError):
    """A benchmark document is malformed or settings mismatch."""


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _get_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the ``metadata`` object from a raw benchmark document.

    Raises EvidenceError if the document has no metadata or metadata is
    not a dict — this is fail-closed.
    """
    meta = doc.get("metadata")
    if not isinstance(meta, dict):
        raise EvidenceError(
            "document has no 'metadata' object or it is not a dict; "
            "cannot extract workload settings"
        )
    return meta


def _first_context_length(meta: dict[str, Any]) -> int | None:
    """Extract the first (primary) context length from metadata."""
    cls = meta.get("context_lengths")
    if isinstance(cls, list) and len(cls) >= 1 and isinstance(cls[0], int):
        return cls[0]
    return None


def extract_settings(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract workload settings from a raw benchmark document's metadata."""
    meta = _get_metadata(doc)
    settings: dict[str, Any] = {}
    for meta_key, display_name in MATCHED_SETTINGS:
        if meta_key == "context_lengths":
            settings[display_name] = _first_context_length(meta)
        else:
            settings[display_name] = meta.get(meta_key)
    return settings


def classify_document_type(doc: dict[str, Any]) -> str:
    """Classify a benchmark document as sustained_matrix or bounded_gate.

    sustained_matrix requires ALL of:
    - metadata.decode_mode == "duration"
    - metadata.duration_per_test >= 10 (present and numeric)
    - metadata.max_tokens >= 256 (present and numeric)
    - every result cell has benchmark_mode == "duration"

    bounded_gate: max_tokens < 256 or duration_per_test < 10.
    indeterminate: metadata missing, decode_mode wrong, fields missing,
    or cannot determine.
    """
    try:
        meta = _get_metadata(doc)
    except EvidenceError:
        return "indeterminate"

    decode_mode = meta.get("decode_mode")
    max_tokens = meta.get("max_tokens")
    duration = meta.get("duration_per_test")

    # Require both duration and max_tokens — no inference from one
    if max_tokens is None or duration is None:
        return "indeterminate"
    if not isinstance(max_tokens, (int, float)) or not isinstance(duration, (int, float)):
        return "indeterminate"

    # If either suggests bounded, classify as bounded_gate
    if max_tokens < 256:
        return "bounded_gate"
    if duration < 10:
        return "bounded_gate"

    # For sustained_matrix, also require decode_mode == "duration"
    if decode_mode != "duration":
        return "indeterminate"

    # Verify every result cell has benchmark_mode == "duration"
    results = doc.get("results")
    if isinstance(results, list):
        for cell in results:
            if isinstance(cell, dict):
                bmode = cell.get("benchmark_mode")
                if bmode is not None and bmode != "duration":
                    return "indeterminate"

    if max_tokens >= 256 and duration >= 10:
        return "sustained_matrix"

    return "indeterminate"


# ---------------------------------------------------------------------------
# Context and cell coverage validation
# ---------------------------------------------------------------------------


def validate_context_coverage(doc: dict[str, Any]) -> dict[str, Any]:
    """Validate that a document has exactly one context and exact cell coverage.

    Checks:
    - metadata.context_lengths has exactly one entry == REQUIRED_CONTEXT
    - every result has context_tokens == REQUIRED_CONTEXT
    - summary_table has exactly one key == str(REQUIRED_CONTEXT)
    - no mismatch between metadata, results, and summary_table
    - exactly the REQUIRED_CONCURRENCIES are present, no duplicates, no extras
    """
    try:
        meta = _get_metadata(doc)
    except EvidenceError:
        return {"valid": False, "reason": "no metadata"}

    # Check context_lengths
    ctx_lengths = meta.get("context_lengths")
    if not isinstance(ctx_lengths, list) or len(ctx_lengths) != 1:
        return {"valid": False, "reason": f"context_lengths must have exactly one entry, got {ctx_lengths}"}
    if ctx_lengths[0] != REQUIRED_CONTEXT:
        return {"valid": False, "reason": f"context_lengths[0]={ctx_lengths[0]} != {REQUIRED_CONTEXT}"}

    # Check results
    results = doc.get("results")
    if not isinstance(results, list) or len(results) == 0:
        return {"valid": False, "reason": "no results"}

    seen_concurrencies: set[int] = set()
    for cell in results:
        if not isinstance(cell, dict):
            return {"valid": False, "reason": "non-dict result entry"}
        ctx = cell.get("context_tokens")
        if ctx is not None and ctx != REQUIRED_CONTEXT:
            return {"valid": False, "reason": f"result context_tokens={ctx} != {REQUIRED_CONTEXT} (multi-context rejected)"}
        conc = cell.get("concurrency")
        if conc is not None:
            if conc in seen_concurrencies:
                return {"valid": False, "reason": f"duplicate concurrency C{conc}"}
            seen_concurrencies.add(conc)

    # Check exact coverage
    expected = set(REQUIRED_CONCURRENCIES)
    if seen_concurrencies != expected:
        missing = expected - seen_concurrencies
        extra = seen_concurrencies - expected
        parts = []
        if missing:
            parts.append(f"missing {[f'C{c}' for c in sorted(missing)]}")
        if extra:
            parts.append(f"unexpected {[f'C{c}' for c in sorted(extra)]}")
        return {"valid": False, "reason": "; ".join(parts)}

    # Check summary_table consistency
    summary = doc.get("summary_table")
    if isinstance(summary, dict):
        ctx_keys = set(summary.keys())
        expected_ctx_key = str(REQUIRED_CONTEXT)
        if len(ctx_keys) > 1:
            return {"valid": False, "reason": f"summary_table has multiple context keys: {sorted(ctx_keys)}"}
        if expected_ctx_key not in ctx_keys:
            return {"valid": False, "reason": f"summary_table missing key '{expected_ctx_key}'"}
        summary_concs = set()
        for k in summary[expected_ctx_key]:
            try:
                summary_concs.add(int(k))
            except (ValueError, TypeError):
                pass
        if summary_concs != expected:
            missing = expected - summary_concs
            extra = summary_concs - expected
            parts = []
            if missing:
                parts.append(f"summary missing {[f'C{c}' for c in sorted(missing)]}")
            if extra:
                parts.append(f"summary unexpected {[f'C{c}' for c in sorted(extra)]}")
            if parts:
                return {"valid": False, "reason": "; ".join(parts)}

    return {"valid": True, "concurrencies": sorted(seen_concurrencies)}


# ---------------------------------------------------------------------------
# Throughput extraction
# ---------------------------------------------------------------------------


def extract_throughput(doc: dict[str, Any]) -> dict[str, float]:
    """Extract aggregate throughput per concurrency from a raw benchmark document.

    Reads ``summary_table`` first (authoritative), then falls back to
    scanning ``results[]`` for ``aggregate_tps`` per concurrency.
    Only returns cells for the REQUIRED_CONCURRENCIES.
    """
    tps: dict[str, float] = {}
    required_labels = {f"C{c}" for c in REQUIRED_CONCURRENCIES}

    # summary_table: {context_tokens: {concurrency: aggregate_tps}}
    summary = doc.get("summary_table")
    if isinstance(summary, dict):
        for _ctx_key, conc_map in summary.items():
            if not isinstance(conc_map, dict):
                continue
            for k, v in conc_map.items():
                try:
                    nk = f"C{int(k)}"
                    val = float(v)
                except (ValueError, TypeError):
                    continue
                if nk in required_labels and nk not in tps:
                    tps[nk] = val

    # Fallback: scan results[] for aggregate_tps
    results = doc.get("results")
    if isinstance(results, list):
        for cell in results:
            if not isinstance(cell, dict):
                continue
            conc = cell.get("concurrency")
            agg = cell.get("aggregate_tps")
            if conc is not None and agg is not None:
                try:
                    nk = f"C{int(conc)}"
                    val = float(agg)
                except (ValueError, TypeError):
                    continue
                if nk in required_labels and nk not in tps:
                    tps[nk] = val

    return tps


# ---------------------------------------------------------------------------
# Validity extraction
# ---------------------------------------------------------------------------


def extract_validity(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract per-cell validity information from a raw benchmark document.

    Every validity field must be present.  Missing fields make the cell
    indeterminate — never default-zero/true.

    Returns:
    - ``cells``: per-concurrency dict with all validity fields.
    - ``all_cells_valid``: True only if every required cell is present and valid.
    - ``zero_cells``: True if no result cells found at all.
    - ``missing_fields``: list of (concurrency, field) for missing fields.
    """
    info: dict[str, Any] = {"cells": {}, "missing_fields": []}
    results = doc.get("results")
    if not isinstance(results, list) or len(results) == 0:
        info["zero_cells"] = True
        return info

    info["zero_cells"] = False
    all_valid = True
    for cell in results:
        if not isinstance(cell, dict):
            all_valid = False
            continue
        conc = cell.get("concurrency")
        if conc is None:
            continue
        try:
            nk = f"C{int(conc)}"
        except (ValueError, TypeError):
            continue

        cell_info: dict[str, Any] = {
            "requested_concurrency": conc,
        }
        cell_valid = True

        for field_name, expected_types in VALIDITY_FIELDS:
            val = cell.get(field_name)
            cell_info[field_name] = val
            if val is None:
                info["missing_fields"].append((nk, field_name))
                cell_valid = False
            elif not isinstance(val, expected_types):
                info["missing_fields"].append((nk, field_name))
                cell_valid = False

        # Check semantic validity: num_errors == 0, effective == requested,
        # not underfilled, not warmup_timed_out, not capacity_limited,
        # benchmark_mode == "duration"
        if cell_info.get("num_errors") is not None:
            if cell_info["num_errors"] > 0:
                cell_valid = False
        if cell_info.get("effective_concurrency") is not None:
            if cell_info["effective_concurrency"] != conc:
                cell_valid = False
        if cell_info.get("underfilled") is True:
            cell_valid = False
        if cell_info.get("warmup_timed_out") is True:
            cell_valid = False
        if cell_info.get("capacity_limited") is True:
            cell_valid = False
        if cell_info.get("benchmark_mode") is not None:
            if cell_info["benchmark_mode"] != "duration":
                cell_valid = False

        cell_info["valid"] = cell_valid
        info["cells"][nk] = cell_info
        if not cell_valid:
            all_valid = False

    info["all_cells_valid"] = all_valid
    return info


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def compare_settings(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compare workload settings between two documents.

    Missing-on-both settings count as a mismatch (fail-closed).
    """
    try:
        base_settings = extract_settings(baseline)
    except EvidenceError:
        base_settings = {}
    try:
        cand_settings = extract_settings(candidate)
    except EvidenceError:
        cand_settings = {}

    matches: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    missing_both: list[str] = []

    for _meta_key, display_name in MATCHED_SETTINGS:
        base_val = base_settings.get(display_name)
        cand_val = cand_settings.get(display_name)

        if base_val is None and cand_val is None:
            missing_both.append(display_name)
            mismatches.append({
                "setting": display_name,
                "baseline": None,
                "candidate": None,
                "reason": "missing on both documents",
            })
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
        "missing_in_both": missing_both,
        "all_matched": len(mismatches) == 0,
    }


def _compute_deltas(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Compute per-concurrency throughput deltas. Only called when all preconditions pass."""
    base_tps = extract_throughput(baseline)
    cand_tps = extract_throughput(candidate)

    cells: list[dict[str, Any]] = []
    for conc in [f"C{c}" for c in REQUIRED_CONCURRENCIES]:
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
                entry["status"] = "compared"
            else:
                entry["delta_percent"] = None
                entry["status"] = "baseline_zero"
        cells.append(entry)

    return {
        "cells": cells,
        "baseline_tps": base_tps,
        "candidate_tps": cand_tps,
    }


def _empty_throughput() -> dict[str, Any]:
    """Return an empty throughput structure with no deltas."""
    return {"cells": [], "baseline_tps": {}, "candidate_tps": {}}


def compare_documents(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Full comparison of two benchmark documents.

    Deltas are ONLY computed and emitted when ALL of the following pass:
    1. Both documents are sustained_matrix
    2. All matched settings are identical
    3. Both documents have valid cells (no errors, underfilled, etc.)
    4. Both documents have exact C1/C2/C4/C8 coverage at 16K
    """
    base_type = classify_document_type(baseline)
    cand_type = classify_document_type(candidate)

    type_mismatch = None
    if base_type != cand_type:
        type_mismatch = (
            f"document type mismatch: baseline={base_type}, "
            f"candidate={cand_type}. "
            "Only two sustained_matrix documents can be compared."
        )
    elif base_type != "sustained_matrix":
        type_mismatch = (
            f"both documents are {base_type}; "
            "only sustained_matrix documents can be compared."
        )

    settings_comparison = compare_settings(baseline, candidate)
    base_validity = extract_validity(baseline)
    cand_validity = extract_validity(candidate)
    base_coverage = validate_context_coverage(baseline)
    cand_coverage = validate_context_coverage(candidate)

    # Determine overall status — deltas are NOT computed until all pass
    if type_mismatch:
        status = "type_mismatch"
    elif not settings_comparison["all_matched"]:
        status = "settings_mismatch"
    elif base_validity.get("zero_cells") or cand_validity.get("zero_cells"):
        status = "no_cells"
    elif not base_coverage["valid"] or not cand_coverage["valid"]:
        status = "coverage_error"
    elif not base_validity.get("all_cells_valid", False) or not cand_validity.get(
        "all_cells_valid", False
    ):
        status = "invalid_cells"
    else:
        status = "compared"

    # Only compute deltas when all preconditions pass
    if status == "compared":
        throughput_comparison = _compute_deltas(baseline, candidate)
    else:
        throughput_comparison = _empty_throughput()

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
        "baseline_coverage": base_coverage,
        "candidate_coverage": cand_coverage,
        "evidence_scope": (
            "Offline comparison of two benchmark JSON documents. "
            "Deltas are claimed only when document type, workload settings, "
            "validity, and exact C1/C2/C4/C8 cell coverage all pass. "
            "Bounded 128-token gate figures are never compared against "
            "sustained 25-second matrix figures. This tool is scoped to "
            "one 16K context and C1/C2/C4/C8 cells. This tool does not "
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

    try:
        report = compare_documents(baseline, candidate)
    except EvidenceError as exc:
        print(f"compare-benchmark-evidence: EVIDENCE ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID

    print(json.dumps(report, indent=2, sort_keys=True))

    if report["status"] in ("type_mismatch", "settings_mismatch"):
        return EXIT_MISMATCH

    if report["status"] in ("no_cells", "coverage_error", "invalid_cells"):
        return EXIT_INVALID

    if args.strict:
        for cell in report["throughput"]["cells"]:
            if cell["status"] in ("missing", "baseline_zero"):
                return EXIT_INVALID

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
