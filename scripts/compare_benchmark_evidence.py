#!/usr/bin/env python3
"""Offline evidence-comparison tool for matched 16K decode benchmarks.

Compares two sustained-decode 16K C1/C2/C4/C8 benchmark JSON documents
produced by ``llm_decode_bench.py`` v0.4.31 and reports per-concurrency
deltas, but ONLY when document type, workload settings, validity, and exact
cell coverage ALL pass. Never mixes bounded 128-token gate figures with
sustained-duration matrix figures.

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
2. ``metadata.concurrency_levels`` must exactly match ``[8, 4, 2, 1]``
   as a set.  Each document must have exactly one 16K context and exactly
   C1/C2/C4/C8 result coverage consistent with metadata.  Absent,
   duplicated, or unexpected/multi-context results fail closed.
3. Missing ``num_errors``, ``effective_concurrency``, ``underfilled``,
   ``warmup_timed_out``, ``capacity_limited``, ``benchmark_mode``,
   ``measurement_seconds``, or ``aggregate_tps`` is indeterminate/invalid,
   never default-zero/true.  ``aggregate_tps`` must be finite, numeric,
   positive, and not bool.  ``num_errors`` must be int (not bool) and 0.
   ``effective_concurrency`` must be int (not bool) and equal requested.
   ``measurement_seconds`` must be finite and > 0.  Boolean flags must
   be actual booleans and false.
4. No numeric deltas are computed or emitted until document type,
   complete settings, validity, and exact cell coverage all pass.
5. Classification requires ``metadata.decode_mode == "duration"`` and
   each result ``benchmark_mode == "duration"``, and requires both
   ``duration_per_test`` and ``max_tokens`` (no inference from one).
6. Multi-context documents and any mismatch between metadata context
   list, result contexts, and summary table are rejected.  If
   ``summary_table`` is present, its values must agree with results
   ``aggregate_tps`` within a tiny float tolerance.  Results are
   canonical after consistency validation.

## Usage::

    python scripts/compare_benchmark_evidence.py \\
        --baseline evidence/cache-off.json \\
        --candidate evidence/cache-on.json

"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

SCHEMA = "sparkring-benchmark-comparison/v1"
SUPPORTED_HARNESS_VERSION = "0.4.31"
SUPPORTED_DECODE_LAYER = "sustained_decode"
EXIT_OK = 0
EXIT_MISMATCH = 2
EXIT_CONFIG_ERROR = 3
EXIT_INVALID = 4

# Required concurrency cells for a valid 16K matrix comparison.
REQUIRED_CONCURRENCIES = [1, 2, 4, 8]
REQUIRED_CONTEXT = 16384
REQUIRED_CONCURRENCY_LEVELS = [8, 4, 2, 1]

# Tolerance for summary_table vs results aggregate_tps agreement.
# Accounts for float serialization round-trip; values are tok/s
# measured to full double precision so 1e-9 relative is generous.
SUMMARY_TOLERANCE_RELATIVE = 1e-9

# Settings that must match exactly before any delta is claimed.
MATCHED_SETTINGS: list[tuple[str, str]] = [
    ("engine", "engine"),
    ("model", "model"),
    ("version", "harness_version"),
    ("primary_decode_layer", "decode_layer"),
    ("decode_mode", "decode_mode"),
    ("context_lengths", "context_tokens"),
    ("concurrency_levels", "concurrencies"),
    ("duration_per_test", "duration_seconds"),
    ("decode_warmup_seconds", "decode_warmup_seconds"),
    ("decode_warmup_context", "decode_warmup_context"),
    ("decode_warmup_concurrency", "decode_warmup_concurrency"),
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


class ConfigError(ValueError):
    """The operator supplied an invalid argument."""


class EvidenceError(ValueError):
    """A benchmark document is malformed or settings mismatch."""


# ---------------------------------------------------------------------------
# Type-check helpers
# ---------------------------------------------------------------------------


def _is_int_not_bool(val: Any) -> bool:
    """True if val is an int but not a bool (bool is a subclass of int)."""
    return isinstance(val, int) and not isinstance(val, bool)


def _is_finite_number(val: Any) -> bool:
    """True if val is a finite int or float but not a bool."""
    if isinstance(val, bool):
        return False
    if isinstance(val, (int, float)):
        try:
            return math.isfinite(float(val))
        except OverflowError:
            return False
    return False


# ---------------------------------------------------------------------------
# Metadata extraction
# ---------------------------------------------------------------------------


def _get_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the ``metadata`` object from a raw benchmark document."""
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


def validate_metadata(doc: dict[str, Any]) -> dict[str, Any]:
    """Validate metadata fields required for a comparable sustained run."""
    try:
        meta = _get_metadata(doc)
    except EvidenceError as exc:
        return {"valid": False, "reason": str(exc)}

    for key in ("version", "engine", "model", "primary_decode_layer"):
        value = meta.get(key)
        if not isinstance(value, str) or not value.strip():
            return {"valid": False, "reason": f"metadata.{key} must be a non-empty string"}

    if meta["version"] != SUPPORTED_HARNESS_VERSION:
        return {
            "valid": False,
            "reason": (
                f"metadata.version must be {SUPPORTED_HARNESS_VERSION!r}, "
                f"got {meta['version']!r}"
            ),
        }

    if meta["primary_decode_layer"] != SUPPORTED_DECODE_LAYER:
        return {
            "valid": False,
            "reason": (
                f"metadata.primary_decode_layer must be {SUPPORTED_DECODE_LAYER!r}, "
                f"got {meta['primary_decode_layer']!r}"
            ),
        }

    if meta.get("decode_mode") != "duration":
        return {"valid": False, "reason": "metadata.decode_mode must be 'duration'"}

    for key, minimum, inclusive in (
        ("duration_per_test", 10.0, True),
        ("decode_warmup_seconds", 0.0, True),
        ("cell_warmup_timeout_seconds", 0.0, False),
    ):
        value = meta.get(key)
        if not _is_finite_number(value):
            return {"valid": False, "reason": f"metadata.{key} must be finite numeric"}
        if (inclusive and float(value) < minimum) or (
            not inclusive and float(value) <= minimum
        ):
            relation = ">=" if inclusive else ">"
            return {"valid": False, "reason": f"metadata.{key} must be {relation} {minimum:g}"}

    for key, minimum in (
        ("max_tokens", 256),
        ("dcp_size", 1),
        ("max_total_tokens", 1),
    ):
        value = meta.get(key)
        if not _is_int_not_bool(value) or value < minimum:
            return {"valid": False, "reason": f"metadata.{key} must be an integer >= {minimum}"}

    for key in ("decode_warmup_context", "decode_warmup_concurrency"):
        value = meta.get(key)
        if not _is_int_not_bool(value) or value < 0:
            return {"valid": False, "reason": f"metadata.{key} must be a non-negative integer"}

    temperature = meta.get("temperature")
    if not _is_finite_number(temperature) or float(temperature) < 0:
        return {"valid": False, "reason": "metadata.temperature must be finite and non-negative"}

    percentages: dict[str, float] = {}
    for key in ("unique_context_percent", "shared_context_percent"):
        value = meta.get(key)
        if not _is_finite_number(value) or not 0 <= float(value) <= 100:
            return {"valid": False, "reason": f"metadata.{key} must be between 0 and 100"}
        percentages[key] = float(value)
    if not math.isclose(sum(percentages.values()), 100.0, abs_tol=1e-9):
        return {
            "valid": False,
            "reason": "metadata unique/shared context percentages must sum to 100",
        }

    for key in ("ignore_eos", "skip_prefill"):
        if not isinstance(meta.get(key), bool):
            return {"valid": False, "reason": f"metadata.{key} must be boolean"}

    return {"valid": True}


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
    if not isinstance(max_tokens, (int, float)) or isinstance(max_tokens, bool):
        return "indeterminate"
    if not isinstance(duration, (int, float)) or isinstance(duration, bool):
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
    - metadata.concurrency_levels exactly matches REQUIRED_CONCURRENCY_LEVELS as a set
    - metadata.context_lengths has exactly one entry == REQUIRED_CONTEXT
    - every result has context_tokens present and == REQUIRED_CONTEXT (missing fails)
    - summary_table has exactly one key == str(REQUIRED_CONTEXT)
    - if summary_table present, its values agree with results aggregate_tps within tolerance
    - no mismatch between metadata, results, and summary_table
    - exactly the REQUIRED_CONCURRENCIES are present, no duplicates, no extras
    """
    try:
        meta = _get_metadata(doc)
    except EvidenceError:
        return {"valid": False, "reason": "no metadata"}

    # Validate concurrency_levels item-by-item before any set conversion.
    # bool (True/False) is a subclass of int in Python, so set([True,2,4,8])
    # silently equals {1,2,4,8} — we must reject bool explicitly.
    # Unhashable types (dict, list) would crash set() — reject first.
    conc_levels = meta.get("concurrency_levels")
    if not isinstance(conc_levels, list):
        return {"valid": False, "reason": f"concurrency_levels must be a list, got {type(conc_levels).__name__}"}
    if len(conc_levels) != len(REQUIRED_CONCURRENCIES):
        return {"valid": False, "reason": f"concurrency_levels must have exactly {len(REQUIRED_CONCURRENCIES)} items, got {len(conc_levels)}: {conc_levels}"}
    allowed = set(REQUIRED_CONCURRENCIES)
    seen_meta_concs: set[int] = set()
    for item in conc_levels:
        if not _is_int_not_bool(item):
            return {"valid": False, "reason": f"concurrency_levels item {item!r} is not integer-not-bool (got {type(item).__name__})"}
        if item not in allowed:
            return {"valid": False, "reason": f"concurrency_levels item {item} is not one of {sorted(allowed)}"}
        if item in seen_meta_concs:
            return {"valid": False, "reason": f"concurrency_levels has duplicate value {item}"}
        seen_meta_concs.add(item)
    # Now safe to compare as a set — all items validated as int-not-bool
    if seen_meta_concs != allowed:
        return {"valid": False, "reason": f"concurrency_levels {conc_levels} does not match required {sorted(allowed)} as a set"}

    # Check context_lengths
    ctx_lengths = meta.get("context_lengths")
    if not isinstance(ctx_lengths, list) or len(ctx_lengths) != 1:
        return {"valid": False, "reason": f"context_lengths must have exactly one entry, got {ctx_lengths}"}
    if not _is_int_not_bool(ctx_lengths[0]):
        return {
            "valid": False,
            "reason": "context_lengths[0] must be an integer, not a numeric alias",
        }
    if ctx_lengths[0] != REQUIRED_CONTEXT:
        return {"valid": False, "reason": f"context_lengths[0]={ctx_lengths[0]} != {REQUIRED_CONTEXT}"}

    # Check results — validate each concurrency before building maps/sets
    results = doc.get("results")
    if not isinstance(results, list) or len(results) == 0:
        return {"valid": False, "reason": "no results"}

    seen_concurrencies: set[int] = set()
    results_tps: dict[str, float] = {}
    for cell in results:
        if not isinstance(cell, dict):
            return {"valid": False, "reason": "non-dict result entry"}
        # context_tokens must be present and exactly REQUIRED_CONTEXT
        ctx = cell.get("context_tokens")
        if ctx is None:
            return {"valid": False, "reason": "result missing context_tokens (must be present)"}
        if not _is_int_not_bool(ctx):
            return {
                "valid": False,
                "reason": "result context_tokens must be an integer, not a numeric alias",
            }
        if ctx != REQUIRED_CONTEXT:
            return {"valid": False, "reason": f"result context_tokens={ctx} != {REQUIRED_CONTEXT}"}
        # concurrency must be present, integer-not-bool, one of allowed,
        # and unique — validated BEFORE any set membership test
        conc = cell.get("concurrency")
        if conc is None:
            return {"valid": False, "reason": "result missing concurrency (must be present)"}
        if not _is_int_not_bool(conc):
            return {"valid": False, "reason": f"result concurrency {conc!r} is not integer-not-bool (got {type(conc).__name__})"}
        if conc not in allowed:
            return {"valid": False, "reason": f"result concurrency {conc} is not one of {sorted(allowed)}"}
        if conc in seen_concurrencies:
            return {"valid": False, "reason": f"duplicate concurrency C{conc}"}
        seen_concurrencies.add(conc)
        # Collect ALL numeric tps for summary comparison — validity
        # (sign/finiteness/positivity) is checked separately by
        # extract_validity, not by coverage.
        agg = cell.get("aggregate_tps")
        if _is_finite_number(agg):
            results_tps[f"C{int(conc)}"] = float(agg)

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

    # Check summary_table consistency — if present, all four numeric values
    # must be present, finite, and agree with results aggregate_tps within
    # tolerance.  Summary never silently overrides results.
    if "summary_table" in doc:
        summary = doc["summary_table"]
        if not isinstance(summary, dict) or not summary:
            return {
                "valid": False,
                "reason": "summary_table must be a non-empty object when present",
            }
        ctx_keys = set(summary.keys())
        expected_ctx_key = str(REQUIRED_CONTEXT)
        if len(ctx_keys) > 1:
            return {"valid": False, "reason": f"summary_table has multiple context keys: {sorted(ctx_keys)}"}
        if expected_ctx_key not in ctx_keys:
            return {"valid": False, "reason": f"summary_table missing key '{expected_ctx_key}'"}
        conc_map = summary[expected_ctx_key]
        if not isinstance(conc_map, dict):
            return {"valid": False, "reason": "summary_table context entry is not a dict"}
        expected_summary_keys = {str(conc) for conc in REQUIRED_CONCURRENCIES}
        actual_summary_keys = set(conc_map.keys())
        if actual_summary_keys != expected_summary_keys:
            return {
                "valid": False,
                "reason": (
                    "summary_table concurrency keys must be exactly "
                    f"{sorted(expected_summary_keys)}, got "
                    f"{sorted(str(key) for key in actual_summary_keys)}"
                ),
            }
        summary_concs: set[int] = set()
        for k, v in conc_map.items():
            conc_int = int(k)
            summary_concs.add(conc_int)
            nk = f"C{conc_int}"
            if nk not in results_tps:
                # Result tps was non-finite/missing — skip summary check
                # for this entry; extract_validity will flag it as invalid.
                continue
            if not _is_finite_number(v):
                return {"valid": False, "reason": f"summary_table {nk} value is not finite numeric, got {type(v).__name__}"}
            summary_val = float(v)
            results_val = results_tps[nk]
            if abs(summary_val - results_val) > abs(results_val) * SUMMARY_TOLERANCE_RELATIVE:
                return {"valid": False, "reason": f"summary_table {nk}={summary_val} disagrees with results {nk}={results_val} beyond tolerance"}
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
# Throughput extraction (results are canonical)
# ---------------------------------------------------------------------------


def extract_throughput(doc: dict[str, Any]) -> dict[str, float]:
    """Extract aggregate throughput per concurrency from results[].

    Results are canonical.  Summary table is validated for consistency
    in ``validate_context_coverage`` but never silently overrides results.
    Only returns cells for the REQUIRED_CONCURRENCIES.
    """
    tps: dict[str, float] = {}
    required_labels = {f"C{c}" for c in REQUIRED_CONCURRENCIES}

    # Scan results[] for aggregate_tps (canonical source)
    results = doc.get("results")
    if isinstance(results, list):
        for cell in results:
            if not isinstance(cell, dict):
                continue
            conc = cell.get("concurrency")
            agg = cell.get("aggregate_tps")
            if conc is not None and agg is not None:
                if not _is_int_not_bool(conc):
                    continue
                try:
                    nk = f"C{int(conc)}"
                    val = float(agg)
                except (ValueError, TypeError):
                    continue
                if nk in required_labels and nk not in tps:
                    tps[nk] = val

    # Fallback: summary_table (only if results didn't provide a value)
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

    return tps


# ---------------------------------------------------------------------------
# Validity extraction
# ---------------------------------------------------------------------------


def extract_validity(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract per-cell validity information from a raw benchmark document.

    Every validity field must be present and correctly typed:
    - ``num_errors``: int (not bool), exactly 0
    - ``effective_concurrency``: int (not bool), exactly requested
    - ``underfilled``: actual bool, must be false
    - ``warmup_timed_out``: actual bool, must be false
    - ``capacity_limited``: actual bool, must be false
    - ``benchmark_mode``: str, must be "duration"
    - ``measurement_seconds``: finite number, must be > 0
    - ``aggregate_tps``: finite number, must be > 0 (not bool, not NaN/inf, not zero/negative)

    Missing or wrong-typed fields make the cell invalid, never default.
    """
    info: dict[str, Any] = {"cells": {}, "missing_fields": [], "invalid_fields": []}
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
        if not _is_int_not_bool(conc):
            # Bool or non-int concurrency — skip cell; coverage will
            # reject it with a structured failure.
            continue
        nk = f"C{int(conc)}"

        cell_info: dict[str, Any] = {
            "requested_concurrency": conc,
        }
        cell_valid = True

        # --- num_errors: int (not bool), exactly 0 ---
        val = cell.get("num_errors")
        cell_info["num_errors"] = val
        if val is None:
            info["missing_fields"].append((nk, "num_errors"))
            cell_valid = False
        elif not _is_int_not_bool(val):
            info["invalid_fields"].append((nk, "num_errors", f"expected int, got {type(val).__name__}"))
            cell_valid = False
        elif val != 0:
            info["invalid_fields"].append((nk, "num_errors", f"expected 0, got {val}"))
            cell_valid = False

        # --- effective_concurrency: int (not bool), == requested ---
        val = cell.get("effective_concurrency")
        cell_info["effective_concurrency"] = val
        if val is None:
            info["missing_fields"].append((nk, "effective_concurrency"))
            cell_valid = False
        elif not _is_int_not_bool(val):
            info["invalid_fields"].append((nk, "effective_concurrency", f"expected int, got {type(val).__name__}"))
            cell_valid = False
        elif val != conc:
            info["invalid_fields"].append((nk, "effective_concurrency", f"expected {conc}, got {val}"))
            cell_valid = False

        # --- measurement_seconds: finite number, > 0 ---
        val = cell.get("measurement_seconds")
        cell_info["measurement_seconds"] = val
        if val is None:
            info["missing_fields"].append((nk, "measurement_seconds"))
            cell_valid = False
        elif not _is_finite_number(val):
            info["invalid_fields"].append((nk, "measurement_seconds", f"expected finite number, got {type(val).__name__}"))
            cell_valid = False
        elif float(val) <= 0:
            info["invalid_fields"].append((nk, "measurement_seconds", f"expected > 0, got {val}"))
            cell_valid = False

        # --- underfilled: actual bool, must be false ---
        val = cell.get("underfilled")
        cell_info["underfilled"] = val
        if val is None:
            info["missing_fields"].append((nk, "underfilled"))
            cell_valid = False
        elif not isinstance(val, bool):
            info["invalid_fields"].append((nk, "underfilled", f"expected bool, got {type(val).__name__}"))
            cell_valid = False
        elif val is True:
            cell_valid = False

        # --- warmup_timed_out: actual bool, must be false ---
        val = cell.get("warmup_timed_out")
        cell_info["warmup_timed_out"] = val
        if val is None:
            info["missing_fields"].append((nk, "warmup_timed_out"))
            cell_valid = False
        elif not isinstance(val, bool):
            info["invalid_fields"].append((nk, "warmup_timed_out", f"expected bool, got {type(val).__name__}"))
            cell_valid = False
        elif val is True:
            cell_valid = False

        # --- capacity_limited: actual bool, must be false ---
        val = cell.get("capacity_limited")
        cell_info["capacity_limited"] = val
        if val is None:
            info["missing_fields"].append((nk, "capacity_limited"))
            cell_valid = False
        elif not isinstance(val, bool):
            info["invalid_fields"].append((nk, "capacity_limited", f"expected bool, got {type(val).__name__}"))
            cell_valid = False
        elif val is True:
            cell_valid = False

        # --- benchmark_mode: str, must be "duration" ---
        val = cell.get("benchmark_mode")
        cell_info["benchmark_mode"] = val
        if val is None:
            info["missing_fields"].append((nk, "benchmark_mode"))
            cell_valid = False
        elif not isinstance(val, str):
            info["invalid_fields"].append((nk, "benchmark_mode", f"expected str, got {type(val).__name__}"))
            cell_valid = False
        elif val != "duration":
            cell_valid = False

        # --- aggregate_tps: finite number, > 0, not bool ---
        val = cell.get("aggregate_tps")
        cell_info["aggregate_tps"] = val
        if val is None:
            info["missing_fields"].append((nk, "aggregate_tps"))
            cell_valid = False
        elif not _is_finite_number(val):
            info["invalid_fields"].append((nk, "aggregate_tps", f"expected finite number, got {type(val).__name__}"))
            cell_valid = False
        elif float(val) <= 0:
            info["invalid_fields"].append((nk, "aggregate_tps", f"expected > 0, got {val}"))
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
            delta = cand_val - base_val
            if base_val > 0:
                delta_percent = (delta / base_val) * 100.0
                if math.isfinite(delta) and math.isfinite(delta_percent):
                    entry["delta"] = delta
                    entry["delta_percent"] = delta_percent
                    entry["status"] = "compared"
                else:
                    entry["delta"] = None
                    entry["delta_percent"] = None
                    entry["status"] = "non_finite_derived_value"
            else:
                entry["delta"] = None
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
    base_metadata = validate_metadata(baseline)
    cand_metadata = validate_metadata(candidate)
    base_validity = extract_validity(baseline)
    cand_validity = extract_validity(candidate)
    base_coverage = validate_context_coverage(baseline)
    cand_coverage = validate_context_coverage(candidate)

    # Determine overall status — deltas are NOT computed until all pass
    if type_mismatch:
        status = "type_mismatch"
    elif baseline == candidate:
        status = "identical_documents"
    elif not base_metadata["valid"] or not cand_metadata["valid"]:
        status = "invalid_metadata"
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
        if any(cell["status"] != "compared" for cell in throughput_comparison["cells"]):
            status = "invalid_delta"
            throughput_comparison = _empty_throughput()
    else:
        throughput_comparison = _empty_throughput()

    return {
        "schema": SCHEMA,
        "status": status,
        "baseline_type": base_type,
        "candidate_type": cand_type,
        "type_mismatch": type_mismatch,
        "settings": settings_comparison,
        "baseline_metadata": base_metadata,
        "candidate_metadata": cand_metadata,
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
            "sustained-duration matrix figures. This tool is scoped to "
            "one 16K context and C1/C2/C4/C8 cells. Results are canonical; "
            "summary_table is validated for consistency but never overrides "
            "results. This tool does not contact the cluster or run benchmarks."
        ),
        "claim_note": (
            "A 'compared' status with delta_percent values is a valid "
            "cross-document comparison only when all_matched is true and "
            "both documents are sustained_matrix type. It is not a sealed "
            "A/B unless both documents were produced under controlled "
            "conditions with identical configuration except for the "
            "declared experimental variable."
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


def _sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of an input evidence file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="path to baseline JSON")
    parser.add_argument("--candidate", required=True, help="path to candidate JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        baseline_path = Path(args.baseline)
        candidate_path = Path(args.candidate)
        baseline = load_document(baseline_path)
        candidate = load_document(candidate_path)
    except ConfigError as exc:
        print(f"compare-benchmark-evidence: CONFIG ERROR: {exc}", file=sys.stderr)
        return EXIT_CONFIG_ERROR

    try:
        report = compare_documents(baseline, candidate)
    except EvidenceError as exc:
        print(f"compare-benchmark-evidence: EVIDENCE ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID

    report["inputs"] = {
        "baseline_sha256": _sha256_file(baseline_path),
        "candidate_sha256": _sha256_file(candidate_path),
    }

    print(json.dumps(report, indent=2, sort_keys=True))

    if report["status"] in (
        "type_mismatch",
        "settings_mismatch",
        "identical_documents",
    ):
        return EXIT_MISMATCH

    if report["status"] in (
        "invalid_metadata",
        "invalid_delta",
        "no_cells",
        "coverage_error",
        "invalid_cells",
    ):
        return EXIT_INVALID

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
