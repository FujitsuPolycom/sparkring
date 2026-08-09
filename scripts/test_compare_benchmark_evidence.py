"""Tests for the offline benchmark evidence comparison tool."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import compare_benchmark_evidence as cmp  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures — realistic sustained-matrix document shapes
# ---------------------------------------------------------------------------

SUSTAINED_BASELINE = {
    "harness": {"name": "llm_decode_bench.py", "version": "0.4.31"},
    "configuration": {
        "context_tokens": 16384,
        "concurrencies": [1, 2, 4, 8],
        "duration_per_cell_seconds": 25,
        "decode_warmup_seconds": 3,
        "max_output_tokens": 2048,
        "temperature": 0,
        "unique_context_percent": 100,
        "shared_context_percent": 0,
        "dcp_size": 4,
        "kv_budget_tokens": 562688,
        "ignore_eos": True,
        "skip_prefill": True,
        "cell_warmup_timeout_seconds": 300,
    },
    "aggregate_tokens_per_second": {
        "C1": 18.33, "C2": 27.61, "C4": 45.11, "C8": 59.40,
    },
    "effective_concurrency": {"C1": 1, "C2": 2, "C4": 4, "C8": 8},
    "errors": {"C1": 0, "C2": 0, "C4": 0, "C8": 0},
    "all_cells_valid": True,
}

SUSTAINED_CANDIDATE = {
    "harness": {"name": "llm_decode_bench.py", "version": "0.4.31"},
    "configuration": {
        "context_tokens": 16384,
        "concurrencies": [1, 2, 4, 8],
        "duration_per_cell_seconds": 25,
        "decode_warmup_seconds": 3,
        "max_output_tokens": 2048,
        "temperature": 0,
        "unique_context_percent": 100,
        "shared_context_percent": 0,
        "dcp_size": 4,
        "kv_budget_tokens": 562688,
        "ignore_eos": True,
        "skip_prefill": True,
        "cell_warmup_timeout_seconds": 300,
    },
    "aggregate_tokens_per_second": {
        "C1": 17.50, "C2": 26.00, "C4": 44.00, "C8": 58.00,
    },
    "effective_concurrency": {"C1": 1, "C2": 2, "C4": 4, "C8": 8},
    "errors": {"C1": 0, "C2": 0, "C4": 0, "C8": 0},
    "all_cells_valid": True,
}

BOUNDED_GATE = {
    "harness": {"name": "exl3_live_gate.py", "version": "1.0"},
    "configuration": {
        "context_tokens": 512,
        "concurrencies": [1, 2, 8],
        "duration_per_cell_seconds": 5,
        "max_output_tokens": 128,
        "temperature": 0,
        "unique_context_percent": 0,
        "shared_context_percent": 100,
        "dcp_size": 4,
        "kv_budget_tokens": 562688,
        "ignore_eos": True,
        "skip_prefill": True,
        "cell_warmup_timeout_seconds": 60,
    },
    "aggregate_tokens_per_second": {
        "C1": 21.94, "C2": 30.25, "C8": 69.39,
    },
    "all_cells_valid": True,
}

# Document with alternative key names (older evidence format)
ALT_FORMAT = {
    "tool": "llm_decode_bench.py",
    "tool_version": "0.4.31",
    "context": 16384,
    "concurrency": [1, 2, 4, 8],
    "duration_seconds": 25,
    "max_tokens": 2048,
    "temperature": 0,
    "unique_context_percent": 100,
    "dcp_size": 4,
    "kv_budget_tokens": 562688,
    "ignore_eos": True,
    "skip_prefill": True,
    "cell_warmup_timeout_seconds": 300,
    "decode_warmup_seconds": 3,
    "shared_context_percent": 0,
    "aggregate_tps": {
        "c1": 18.0, "c2": 27.0, "c4": 44.0, "c8": 58.0,
    },
}


# ---------------------------------------------------------------------------
# Document classification
# ---------------------------------------------------------------------------


def test_classify_sustained_matrix():
    assert cmp.classify_document_type(SUSTAINED_BASELINE) == "sustained_matrix"


def test_classify_bounded_gate():
    assert cmp.classify_document_type(BOUNDED_GATE) == "bounded_gate"


def test_classify_indeterminate_empty():
    assert cmp.classify_document_type({}) == "indeterminate"


def test_classify_sustained_no_duration():
    """max_tokens >= 256 but no duration → sustained (best effort)."""
    doc = {"configuration": {"max_output_tokens": 2048}}
    assert cmp.classify_document_type(doc) == "sustained_matrix"


def test_classify_bounded_low_max_tokens():
    doc = {"configuration": {"max_output_tokens": 128, "duration_per_cell_seconds": 25}}
    assert cmp.classify_document_type(doc) == "bounded_gate"


def test_classify_bounded_low_duration():
    doc = {"configuration": {"max_output_tokens": 2048, "duration_per_cell_seconds": 5}}
    assert cmp.classify_document_type(doc) == "bounded_gate"


# ---------------------------------------------------------------------------
# Settings matching
# ---------------------------------------------------------------------------


def test_settings_match_identical():
    result = cmp.compare_settings(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert result["all_matched"] is True
    assert len(result["mismatched"]) == 0


def test_settings_mismatch_duration():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["configuration"]["duration_per_cell_seconds"] = 15
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False
    assert any(m["setting"] == "duration_seconds" for m in result["mismatched"])


def test_settings_mismatch_concurrencies():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["configuration"]["concurrencies"] = [1, 2, 4]
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False
    assert any(m["setting"] == "concurrencies" for m in result["mismatched"])


def test_settings_mismatch_temperature():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["configuration"]["temperature"] = 1
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False


def test_settings_mismatch_kv_budget():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["configuration"]["kv_budget_tokens"] = 1125632
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False
    assert any(m["setting"] == "kv_budget_tokens" for m in result["mismatched"])


def test_settings_mismatch_unique_context():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["configuration"]["unique_context_percent"] = 50
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False


def test_alt_format_keys_matched():
    """Alternative key names (tool, context, etc.) must be resolved."""
    result = cmp.compare_settings(SUSTAINED_BASELINE, ALT_FORMAT)
    assert result["all_matched"] is True


# ---------------------------------------------------------------------------
# Throughput extraction and comparison
# ---------------------------------------------------------------------------


def test_extract_throughput_top_level():
    tps = cmp.extract_throughput(SUSTAINED_BASELINE)
    assert tps["C1"] == 18.33
    assert tps["C8"] == 59.40


def test_extract_throughput_alt_key():
    tps = cmp.extract_throughput(ALT_FORMAT)
    assert tps["C1"] == 18.0
    assert tps["C8"] == 58.0


def test_extract_throughput_nested():
    doc = {
        "artifacts": {
            "arm_a": {
                "aggregate_tokens_per_second": {"C1": 10.0, "C2": 20.0},
            }
        }
    }
    tps = cmp.extract_throughput(doc)
    assert tps == {"C1": 10.0, "C2": 20.0}


def test_extract_throughput_empty():
    assert cmp.extract_throughput({}) == {}


def test_throughput_comparison_deltas():
    result = cmp.compare_throughput(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    cells = {c["concurrency"]: c for c in result["cells"]}
    assert cells["C1"]["delta"] == pytest.approx(17.50 - 18.33)
    assert cells["C1"]["delta_percent"] == pytest.approx(
        ((17.50 - 18.33) / 18.33) * 100
    )
    assert cells["C1"]["status"] == "compared"


def test_throughput_comparison_missing_concurrency():
    """If candidate is missing a concurrency, status is 'missing'."""
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    del candidate["aggregate_tokens_per_second"]["C8"]
    result = cmp.compare_throughput(SUSTAINED_BASELINE, candidate)
    cells = {c["concurrency"]: c for c in result["cells"]}
    assert cells["C8"]["status"] == "missing"
    assert cells["C8"]["delta"] is None


def test_throughput_baseline_zero():
    baseline = json.loads(json.dumps(SUSTAINED_BASELINE))
    baseline["aggregate_tokens_per_second"]["C1"] = 0.0
    result = cmp.compare_throughput(baseline, SUSTAINED_CANDIDATE)
    cells = {c["concurrency"]: c for c in result["cells"]}
    assert cells["C1"]["delta_percent"] is None
    assert cells["C1"]["status"] == "baseline_zero"


# ---------------------------------------------------------------------------
# Full document comparison
# ---------------------------------------------------------------------------


def test_compare_matched_sustained():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert result["status"] == "compared"
    assert result["baseline_type"] == "sustained_matrix"
    assert result["candidate_type"] == "sustained_matrix"
    assert result["type_mismatch"] is None
    assert result["settings"]["all_matched"] is True


def test_compare_type_mismatch_bounded_vs_sustained():
    """Bounded gate vs sustained matrix must produce type_mismatch."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, BOUNDED_GATE)
    assert result["status"] == "type_mismatch"
    assert result["type_mismatch"] is not None
    assert "128-token" in result["type_mismatch"] or "bounded" in result["type_mismatch"].lower()


def test_compare_settings_mismatch():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["configuration"]["duration_per_cell_seconds"] = 15
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "settings_mismatch"


def test_compare_invalid_cells():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["all_cells_valid"] = False
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "invalid_cells"


def test_compare_claim_note_present():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert "claim_note" in result
    assert "sealed A/B" in result["claim_note"]


def test_compare_evidence_scope_present():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert "evidence_scope" in result
    assert "offline" in result["evidence_scope"].lower()


# ---------------------------------------------------------------------------
# Validity extraction
# ---------------------------------------------------------------------------


def test_extract_validity_all_valid():
    v = cmp.extract_validity(SUSTAINED_BASELINE)
    assert v["all_cells_valid"] is True


def test_extract_validity_effective_concurrency():
    v = cmp.extract_validity(SUSTAINED_BASELINE)
    assert v["effective_concurrency"]["C4"] == 4


def test_extract_validity_errors():
    v = cmp.extract_validity(SUSTAINED_BASELINE)
    assert v["errors"]["C1"] == 0


def test_extract_validity_empty():
    v = cmp.extract_validity({})
    assert "all_cells_valid" not in v


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_matched(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(SUSTAINED_CANDIDATE))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_OK
    report = json.loads(capsys.readouterr().out)
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
    candidate["configuration"]["duration_per_cell_seconds"] = 15
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(candidate))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_MISMATCH


def test_cli_strict_invalid_cells(tmp_path, capsys):
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["all_cells_valid"] = False
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(candidate))
    rc = cmp.main([
        "--baseline", str(base), "--candidate", str(cand), "--strict"
    ])
    assert rc == cmp.EXIT_INVALID


def test_cli_missing_file(capsys):
    rc = cmp.main([
        "--baseline", "nonexistent.json", "--candidate", "also.json"
    ])
    assert rc == cmp.EXIT_CONFIG_ERROR


def test_cli_invalid_json(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text("not json at all")
    rc = cmp.main([
        "--baseline", str(bad), "--candidate", str(bad)
    ])
    assert rc == cmp.EXIT_CONFIG_ERROR


def test_cli_alt_format(tmp_path, capsys):
    """Alternative-format documents must be comparable."""
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(ALT_FORMAT))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_OK


# ---------------------------------------------------------------------------
# Concurrency normalization
# ---------------------------------------------------------------------------


def test_norm_concurrency_uppercase():
    assert cmp._norm_concurrency("c1") == "C1"
    assert cmp._norm_concurrency("C1") == "C1"
    assert cmp._norm_concurrency("c8") == "C8"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema():
    assert cmp.SCHEMA == "sparkring-benchmark-comparison/v1"
