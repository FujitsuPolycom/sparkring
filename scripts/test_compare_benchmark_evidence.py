"""Tests for the offline benchmark evidence comparison tool.

Fixtures use the actual llm_decode_bench v0.4.31 raw JSON schema:
``metadata``, ``results[]``, and ``summary_table``.  No invented
configuration/aggregate-map shapes are used.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_benchmark_evidence as cmp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — actual llm_decode_bench v0.4.31 raw schema
# ---------------------------------------------------------------------------

def _make_meta(
    *,
    version: str = "0.4.31",
    context_lengths: list[int] | None = None,
    concurrency_levels: list[int] | None = None,
    duration_per_test: float = 25.0,
    decode_warmup_seconds: float = 0.0,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    unique_context_percent: float = 0.0,
    shared_context_percent: float = 100.0,
    dcp_size: int = 4,
    max_total_tokens: int = 562688,
    ignore_eos: bool = True,
    skip_prefill: bool = True,
    cell_warmup_timeout_seconds: float = 600.0,
    benchmark_mode: str = "duration",
    prefill_mode: str = "skipped",
) -> dict:
    return {
        "version": version,
        "engine": "vllm",
        "model": "glm-5.2-exl3-tr3-3.25bpw",
        "timestamp": "2026-08-09T06:02:17.088583",
        "decode_mode": "duration",
        "primary_decode_layer": "sustained_decode",
        "duration_per_test": duration_per_test,
        "decode_warmup_seconds": decode_warmup_seconds,
        "decode_warmup_context": 0 if decode_warmup_seconds == 0 else 16384,
        "decode_warmup_concurrency": 0 if decode_warmup_seconds == 0 else 1,
        "cell_warmup_timeout_seconds": cell_warmup_timeout_seconds,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "ignore_eos": ignore_eos,
        "max_total_tokens": max_total_tokens,
        "dcp_size": dcp_size,
        "unique_context_percent": unique_context_percent,
        "shared_context_percent": shared_context_percent,
        "concurrency_levels": concurrency_levels or [8, 4, 2, 1],
        "context_lengths": context_lengths or [16384],
        "benchmark_mode": benchmark_mode,
        "prefill_mode": prefill_mode,
        "skip_prefill": skip_prefill,
    }


def _make_result(
    conc: int,
    aggregate_tps: float,
    *,
    num_errors: int = 0,
    effective_concurrency: int | None = None,
    context_tokens: int = 16384,
    benchmark_mode: str = "duration",
    measurement_seconds: float = 24.9,
) -> dict:
    return {
        "concurrency": conc,
        "context_tokens": context_tokens,
        "benchmark_mode": benchmark_mode,
        "measurement_seconds": measurement_seconds,
        "aggregate_tps": aggregate_tps,
        "num_errors": num_errors,
        "effective_concurrency": effective_concurrency if effective_concurrency is not None else conc,
        "request_count": conc,
        "completed_request_count": 0,
    }


def _make_doc(
    meta: dict | None = None,
    results: list[dict] | None = None,
) -> dict:
    if meta is None:
        meta = _make_meta()
    if results is None:
        results = [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ]
    # Build summary_table from results
    summary: dict[str, dict[str, float]] = {}
    for r in results:
        ctx = str(r["context_tokens"])
        if ctx not in summary:
            summary[ctx] = {}
        summary[ctx][str(r["concurrency"])] = r["aggregate_tps"]
    return {
        "metadata": meta,
        "results": results,
        "summary_table": summary,
    }

def test_classify_no_duration_but_max_tokens_high():
    """max_tokens >= 256 but duration_per_test=0 → bounded_gate (duration < 10)."""
    doc = {"metadata": _make_meta(duration_per_test=0, max_tokens=1024)}
    assert cmp.classify_document_type(doc) == "bounded_gate"
# Post-upgrade baseline protocol fixtures (actual protocol from 20260809)

SUSTAINED_BASELINE = _make_doc(
    meta=_make_meta(),
    results=[
        _make_result(8, 64.56),
        _make_result(4, 45.13),
        _make_result(2, 29.29),
        _make_result(1, 15.89),
    ],
)

SUSTAINED_CANDIDATE = _make_doc(
    meta=_make_meta(),
    results=[
        _make_result(8, 62.00),
        _make_result(4, 44.00),
        _make_result(2, 28.50),
        _make_result(1, 15.50),
    ],
)

# Bounded gate: 128 tokens, 5s duration
BOUNDED_GATE = {
    "metadata": _make_meta(
        max_tokens=128,
        duration_per_test=5,
        concurrency_levels=[1, 2, 8],
        context_lengths=[512],
        cell_warmup_timeout_seconds=60,
    ),
    "results": [
        _make_result(1, 21.94, context_tokens=512),
        _make_result(2, 30.25, context_tokens=512),
        _make_result(8, 69.39, context_tokens=512),
    ],
    "summary_table": {
        "512": {"1": 21.94, "2": 30.25, "8": 69.39},
    },
}

# Older protocol variant (pre-upgrade: 2048 max tokens, 3s warmup, 300s timeout, 100% unique)
OLDER_PROTOCOL = _make_doc(
    meta=_make_meta(
        max_tokens=2048,
        decode_warmup_seconds=3.0,
        cell_warmup_timeout_seconds=300.0,
        unique_context_percent=100.0,
        shared_context_percent=0.0,
        max_total_tokens=1125632,
    ),
)

# Document with no metadata (malformed)
NO_METADATA: dict = {}

# Document with metadata but no results
NO_RESULTS = {
    "metadata": _make_meta(),
    "results": [],
    "summary_table": {},
}

# Document with errors in one cell
INVALID_CELLS = _make_doc(
    meta=_make_meta(),
    results=[
        _make_result(8, 64.56, num_errors=2),
        _make_result(4, 45.13),
        _make_result(2, 29.29),
        _make_result(1, 15.89),
    ],
)

# Document with underfilled concurrency (effective != requested)
UNDERFILLED = _make_doc(
    meta=_make_meta(),
    results=[
        _make_result(8, 30.0, effective_concurrency=3),
        _make_result(4, 45.13),
        _make_result(2, 29.29),
        _make_result(1, 15.89),
    ],
)


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------


def test_classify_sustained_matrix():
    assert cmp.classify_document_type(SUSTAINED_BASELINE) == "sustained_matrix"


def test_classify_bounded_gate():
    assert cmp.classify_document_type(BOUNDED_GATE) == "bounded_gate"


def test_classify_indeterminate_empty():
    assert cmp.classify_document_type({}) == "indeterminate"


def test_classify_indeterminate_no_metadata():
    assert cmp.classify_document_type(NO_METADATA) == "indeterminate"



def test_classify_bounded_low_max_tokens():
    doc = {"metadata": _make_meta(max_tokens=128, duration_per_test=25)}
    assert cmp.classify_document_type(doc) == "bounded_gate"


def test_classify_bounded_low_duration():
    doc = {"metadata": _make_meta(max_tokens=2048, duration_per_test=5)}
    assert cmp.classify_document_type(doc) == "bounded_gate"


# ---------------------------------------------------------------------------
# Settings extraction and matching
# ---------------------------------------------------------------------------


def test_extract_settings_from_metadata():
    settings = cmp.extract_settings(SUSTAINED_BASELINE)
    assert settings["harness_version"] == "0.4.31"
    assert settings["context_tokens"] == 16384
    assert settings["concurrencies"] == [8, 4, 2, 1]
    assert settings["duration_seconds"] == 25.0
    assert settings["max_output_tokens"] == 1024
    assert settings["temperature"] == 0.0
    assert settings["unique_context_percent"] == 0.0
    assert settings["shared_context_percent"] == 100.0
    assert settings["dcp_size"] == 4
    assert settings["kv_budget_tokens"] == 562688
    assert settings["ignore_eos"] is True
    assert settings["skip_prefill"] is True
    assert settings["cell_warmup_timeout_seconds"] == 600.0


def test_extract_settings_no_metadata_raises():
    with pytest.raises(cmp.EvidenceError):
        cmp.extract_settings({})


def test_settings_match_identical():
    result = cmp.compare_settings(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert result["all_matched"] is True
    assert len(result["mismatched"]) == 0


def test_settings_mismatch_duration():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["duration_per_test"] = 15.0
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False
    assert any(m["setting"] == "duration_seconds" for m in result["mismatched"])


def test_settings_mismatch_concurrencies():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["concurrency_levels"] = [1, 2, 4]
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False
    assert any(m["setting"] == "concurrencies" for m in result["mismatched"])


def test_settings_mismatch_temperature():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["temperature"] = 0.7
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False


def test_settings_mismatch_kv_budget():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["max_total_tokens"] = 1125632
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert any(m["setting"] == "kv_budget_tokens" for m in result["mismatched"])


def test_settings_mismatch_unique_context():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["unique_context_percent"] = 100.0
    candidate["metadata"]["shared_context_percent"] = 0.0
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False


def test_settings_mismatch_max_tokens():
    """Older protocol (2048) vs post-upgrade (1024) must mismatch."""
    result = cmp.compare_settings(SUSTAINED_BASELINE, OLDER_PROTOCOL)
    assert result["all_matched"] is False
    assert any(m["setting"] == "max_output_tokens" for m in result["mismatched"])


def test_settings_missing_on_both_counts_as_mismatch():
    """Two documents missing the same setting must NOT count as matched."""
    # Both documents missing ignore_eos
    base = json.loads(json.dumps(SUSTAINED_BASELINE))
    cand = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    del base["metadata"]["ignore_eos"]
    del cand["metadata"]["ignore_eos"]
    result = cmp.compare_settings(base, cand)
    assert result["all_matched"] is False
    assert any(
        m["reason"] == "missing on both documents" for m in result["mismatched"]
    )


def test_settings_mismatch_decode_warmup():
    """Post-upgrade (0s) vs older (3s) must mismatch."""
    result = cmp.compare_settings(SUSTAINED_BASELINE, OLDER_PROTOCOL)
    assert any(
        m["setting"] == "decode_warmup_seconds" for m in result["mismatched"]
    )


def test_settings_mismatch_cell_warmup_timeout():
    """Post-upgrade (600s) vs older (300s) must mismatch."""
    result = cmp.compare_settings(SUSTAINED_BASELINE, OLDER_PROTOCOL)
    assert any(
        m["setting"] == "cell_warmup_timeout_seconds"
        for m in result["mismatched"]
    )


# ---------------------------------------------------------------------------
# Throughput extraction and comparison
# ---------------------------------------------------------------------------


def test_extract_throughput_summary_table():
    tps = cmp.extract_throughput(SUSTAINED_BASELINE)
    assert tps["C1"] == 15.89
    assert tps["C8"] == 64.56
    assert tps["C4"] == 45.13
    assert tps["C2"] == 29.29


def test_extract_throughput_from_results_no_summary():
    """If summary_table is missing, results[] should be used."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(1, 10.0),
            _make_result(2, 20.0),
        ],
        "summary_table": {},
    }
    tps = cmp.extract_throughput(doc)
    assert tps == {"C1": 10.0, "C2": 20.0}


def test_extract_throughput_empty():
    assert cmp.extract_throughput({}) == {}


def test_throughput_comparison_deltas():
    result = cmp.compare_throughput(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    cells = {c["concurrency"]: c for c in result["cells"]}
    assert cells["C1"]["status"] == "compared"
    assert cells["C1"]["delta"] == 15.50 - 15.89
    assert cells["C1"]["delta_percent"] is not None


def test_throughput_comparison_missing_concurrency():
    """If candidate is missing a concurrency, status is 'missing'."""
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["results"] = [r for r in candidate["results"] if r["concurrency"] != 8]
    candidate["summary_table"]["16384"].pop("8")
    result = cmp.compare_throughput(SUSTAINED_BASELINE, candidate)
    cells = {c["concurrency"]: c for c in result["cells"]}
    assert cells["C8"]["status"] == "missing"
    assert cells["C8"]["delta"] is None


def test_throughput_baseline_zero():
    baseline = json.loads(json.dumps(SUSTAINED_BASELINE))
    for r in baseline["results"]:
        r["aggregate_tps"] = 0.0
    baseline["summary_table"]["16384"]["8"] = 0.0
    baseline["summary_table"]["16384"]["4"] = 0.0
    baseline["summary_table"]["16384"]["2"] = 0.0
    baseline["summary_table"]["16384"]["1"] = 0.0
    result = cmp.compare_throughput(baseline, SUSTAINED_CANDIDATE)
    cells = {c["concurrency"]: c for c in result["cells"]}
    assert cells["C1"]["status"] == "baseline_zero"


# ---------------------------------------------------------------------------
# Validity extraction
# ---------------------------------------------------------------------------


def test_extract_validity_all_valid():
    v = cmp.extract_validity(SUSTAINED_BASELINE)
    assert v["all_cells_valid"] is True
    assert v["zero_cells"] is False


def test_extract_validity_errors():
    v = cmp.extract_validity(INVALID_CELLS)
    assert v["all_cells_valid"] is False
    assert v["cells"]["C8"]["num_errors"] == 2


def test_extract_validity_underfilled():
    v = cmp.extract_validity(UNDERFILLED)
    assert v["all_cells_valid"] is False
    assert v["cells"]["C8"]["effective_concurrency"] == 3


def test_extract_validity_no_results():
    v = cmp.extract_validity(NO_RESULTS)
    assert v["zero_cells"] is True
    assert "all_cells_valid" not in v


def test_extract_validity_empty():
    v = cmp.extract_validity({})
    assert v["zero_cells"] is True
    assert "all_cells_valid" not in v


# ---------------------------------------------------------------------------
# Full document comparison
# ---------------------------------------------------------------------------


def test_compare_matched_sustained():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert result["status"] == "compared"
    assert result["baseline_type"] == "sustained_matrix"
    assert result["candidate_type"] == "sustained_matrix"
    assert result["settings"]["all_matched"] is True


def test_compare_type_mismatch_bounded_vs_sustained():
    """Bounded gate vs sustained matrix must produce type_mismatch."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, BOUNDED_GATE)
    assert result["status"] == "type_mismatch"
    assert result["type_mismatch"] is not None
    assert "128-token" in result["type_mismatch"] or "bounded" in result["type_mismatch"].lower()


def test_compare_type_mismatch_indeterminate():
    """Indeterminate vs sustained must produce type_mismatch."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, NO_METADATA)
    assert result["status"] == "type_mismatch"
    assert "indeterminate" in result["type_mismatch"]


def test_compare_settings_mismatch():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["temperature"] = 1.0
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "settings_mismatch"


def test_compare_protocol_mismatch():
    """Post-upgrade vs older protocol must settings_mismatch."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, OLDER_PROTOCOL)
    assert result["status"] == "settings_mismatch"


def test_compare_invalid_cells():
    result = cmp.compare_documents(SUSTAINED_BASELINE, INVALID_CELLS)
    assert result["status"] == "invalid_cells"


def test_compare_underfilled_cells():
    result = cmp.compare_documents(SUSTAINED_BASELINE, UNDERFILLED)
    assert result["status"] == "invalid_cells"


def test_compare_no_cells():
    """No results → no_cells status."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, NO_RESULTS)
    assert result["status"] == "no_cells"


def test_compare_both_no_cells():
    """Both documents with no results → no_cells."""
    result = cmp.compare_documents(NO_RESULTS, NO_RESULTS)
    assert result["status"] == "no_cells"


def test_compare_claim_note_present():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert "claim_note" in result
    assert "sealed A/B" in result["claim_note"]


def test_compare_evidence_scope_present():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert "evidence_scope" in result
    assert "offline" in result["evidence_scope"].lower()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_matched(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(SUSTAINED_CANDIDATE))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert rc == cmp.EXIT_OK
    assert report["status"] == "compared"


def test_cli_type_mismatch(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(BOUNDED_GATE))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_MISMATCH


def test_cli_settings_mismatch(tmp_path, capsys):
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["temperature"] = 1.0
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(candidate))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_MISMATCH


def test_cli_strict_invalid_cells(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(INVALID_CELLS))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand), "--strict"])
    assert rc == cmp.EXIT_INVALID


def test_cli_strict_no_cells(tmp_path, capsys):
    """Strict mode with no cells must exit non-zero."""
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(NO_RESULTS))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand), "--strict"])
    assert rc == cmp.EXIT_INVALID


def test_cli_no_strict_no_cells(tmp_path, capsys):
    """Non-strict mode with no cells exits 0 (status reported)."""
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(NO_RESULTS))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_OK


def test_cli_missing_file(capsys):
    rc = cmp.main([
        "--baseline", "nonexistent.json",
        "--candidate", "also-nonexistent.json",
    ])
    assert rc == cmp.EXIT_CONFIG_ERROR


def test_cli_invalid_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not valid json {{{")
    good = tmp_path / "good.json"
    good.write_text(json.dumps(SUSTAINED_BASELINE))
    rc = cmp.main(["--baseline", str(bad), "--candidate", str(good)])
    assert rc == cmp.EXIT_CONFIG_ERROR


def test_cli_no_metadata(tmp_path, capsys):
    """Document with no metadata → indeterminate type → mismatch."""
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(NO_METADATA))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_MISMATCH


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema():
    assert cmp.SCHEMA == "sparkring-benchmark-comparison/v2"
