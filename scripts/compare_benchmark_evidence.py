#!/usr/bin/env python3
"""Offline evidence-comparison tool for cache-off vs cache-on 16K benchmarks.

Compares two sustained-decode 16K C1/C2/C4/C8 benchmark JSON documents
produced by ``llm_decode_bench.py`` v0.4.31 and reports per-concurrency
deltas, but ONLY when workload settings match exactly.  Never mixes bounded
128-token gate figures with sustained 25-second matrix figures.

This tool is **purely offline**: it reads two JSON files supplied by
the operator and produces a structured comparison report.  It does not
contact the cluster, run benchmarks, or mutate anything.

## Raw schema (llm_decode_bench v0.4.31)

The tool reads the actual raw JSON schema produced by the benchmark harness:

- ``metadata``: top-level object with ``version``, ``duration_per_test``,
  ``max_tokens``, ``temperature``, ``decode_warmup_seconds``,
  ``cell_warmup_timeout_seconds``, ``unique_context_percent``,
  ``shared_context_percent``, ``dcp_size``, ``max_total_tokens``,
  ``skip_prefill``, ``ignore_eos``, ``concurrency_levels``,
  ``context_lengths``, ``prefill_mode``.
- ``results``: array of per-cell objects, each with ``concurrency``,
  ``context_tokens``, ``aggregate_tps``, ``num_errors``,
  ``effective_concurrency``, ``benchmark_mode``, ``measurement_seconds``.
- ``summary_table``: ``{context_tokens: {concurrency: aggregate_tps}}``.

## Workload-settings matching

Before claiming any delta, the tool verifies that these settings are
identical between the two documents:

- harness version (metadata.version)
- context length (metadata.context_lengths[0])
- concurrencies list (metadata.concurrency_levels)
- duration per cell (metadata.duration_per_test)
- decode warmup (metadata.decode_warmup_seconds)
- max output tokens (metadata.max_tokens)
- temperature (metadata.temperature)
- unique context percent (metadata.unique_context_percent)
- shared context percent (metadata.shared_context_percent)
- DCP size (metadata.dcp_size)
- KV budget (metadata.max_total_tokens)
- ignore_eos (metadata.ignore_eos)
- skip_prefill (metadata.skip_prefill)
- cell warmup timeout (metadata.cell_warmup_timeout_seconds)

If any setting is missing on both sides, it is reported as
``missing_in_both`` and counts as a **mismatch** (fail-closed).
If any setting differs, the comparison reports ``settings_mismatch`` and
NO delta is claimed.

## 128-token gate vs sustained matrix

The tool distinguishes bounded 128-token finite-request gate figures
from sustained 25-second decode matrix figures.  A document is
classified as ``sustained_matrix`` when ``benchmark_mode == "duration"``
and ``duration_per_test >= 10`` and ``max_tokens >= 256``.  A document
with ``max_tokens < 256`` or ``duration_per_test < 10`` is classified
as ``bounded_gate``.  An ``indeterminate`` document (missing metadata)
cannot be compared against anything.

## Usage::

    python scripts/compare_benchmark_evidence.py \\
        --baseline evidence/cache-off.json \\
        --candidate evidence/cache-on.json

    # Strict mode: exit non-zero if any cell is invalid, missing, or has errors
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

SCHEMA = "sparkring-benchmark-comparison/v2"
EXIT_OK = 0
EXIT_MISMATCH = 2
EXIT_CONFIG_ERROR = 3
EXIT_INVALID = 4

# Settings that must match exactly before any delta is claimed.
# Each entry is (metadata_key, display_name).
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
    """Extract workload settings from a raw benchmark document's metadata.

    Raises EvidenceError if the document has no metadata.
    """
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

    sustained_matrix: benchmark_mode == "duration" and
        duration_per_test >= 10 and max_tokens >= 256
    bounded_gate: max_tokens < 256 or duration_per_test < 10
    indeterminate: metadata missing or cannot determine
    """
    try:
        meta = _get_metadata(doc)
    except EvidenceError:
        return "indeterminate"

    max_tokens = meta.get("max_tokens")
    duration = meta.get("duration_per_test")

    if max_tokens is None and duration is None:
        return "indeterminate"

    # If either suggests bounded, classify as bounded_gate
    if max_tokens is not None and isinstance(max_tokens, (int, float)) and max_tokens < 256:
        return "bounded_gate"
    if duration is not None and isinstance(duration, (int, float)) and duration < 10:
        return "bounded_gate"
    if (
        max_tokens is not None
        and isinstance(max_tokens, (int, float))
        and max_tokens >= 256
        and duration is not None
        and isinstance(duration, (int, float))
        and duration >= 10
    ):
        return "sustained_matrix"
    # One is present but not the other; use what we have
    if max_tokens is not None and isinstance(max_tokens, (int, float)) and max_tokens >= 256:
        return "sustained_matrix"
    if duration is not None and isinstance(duration, (int, float)) and duration >= 10:
        return "sustained_matrix"
    return "indeterminate"


# ---------------------------------------------------------------------------
# Throughput extraction from results[] and summary_table
# ---------------------------------------------------------------------------


def extract_throughput(doc: dict[str, Any]) -> dict[str, float]:
    """Extract aggregate throughput per concurrency from a raw benchmark document.

    Reads ``summary_table`` first (authoritative), then falls back to
    scanning ``results[]`` for ``aggregate_tps`` per concurrency.
    """
    tps: dict[str, float] = {}

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
                if nk not in tps:
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
                if nk not in tps:
                    tps[nk] = val

    return tps


def extract_validity(doc: dict[str, Any]) -> dict[str, Any]:
    """Extract per-cell validity information from a raw benchmark document.

    Scans ``results[]`` for each cell's ``num_errors``,
    ``effective_concurrency``, and ``benchmark_mode``. Returns:

    - ``cells``: per-concurrency dict with ``num_errors``,
      ``effective_concurrency``, ``requested_concurrency``.
    - ``all_cells_valid``: True only if every cell has zero errors and
      effective concurrency == requested concurrency. False if any cell
      is invalid. Absent only if no cells are found at all.
    - ``zero_cells``: True if no result cells were found.
    """
    info: dict[str, Any] = {"cells": {}}
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
        errors = cell.get("num_errors", 0)
        eff = cell.get("effective_concurrency")
        requested = conc
        cell_info: dict[str, Any] = {
            "num_errors": errors,
            "requested_concurrency": requested,
            "effective_concurrency": eff,
        }
        info["cells"][nk] = cell_info
        if not isinstance(errors, int) or errors > 0:
            all_valid = False
        if eff is not None and eff != requested:
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
    If either document has no metadata, all settings are mismatched.
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
    elif base_validity.get("zero_cells") or cand_validity.get("zero_cells"):
        status = "no_cells"
    elif not base_validity.get("all_cells_valid", False) or not cand_validity.get(
        "all_cells_valid", False
    ):
        status = "invalid_cells"
    elif len(throughput_comparison["cells"]) == 0:
        status = "no_cells"
    else:
        status = "compared"

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

    try:
        report = compare_documents(baseline, candidate)
    except EvidenceError as exc:
        print(f"compare-benchmark-evidence: EVIDENCE ERROR: {exc}", file=sys.stderr)
        return EXIT_INVALID

    print(json.dumps(report, indent=2, sort_keys=True))

    if report["status"] in ("type_mismatch", "settings_mismatch"):
        return EXIT_MISMATCH

    if report["status"] == "no_cells":
        return EXIT_INVALID if args.strict else EXIT_OK

    if args.strict:
        # Check for invalid cells or missing throughput
        for cell in report["throughput"]["cells"]:
            if cell["status"] in ("missing", "baseline_zero"):
                return EXIT_INVALID
        base_valid = report["baseline_validity"].get("all_cells_valid", False)
        cand_valid = report["candidate_validity"].get("all_cells_valid", False)
        if not base_valid or not cand_valid:
            return EXIT_INVALID

    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
