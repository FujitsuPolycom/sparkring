"""Tests for the offline benchmark evidence comparison tool.

Fixtures use the actual llm_decode_bench v0.4.31 raw JSON schema:
``metadata``, ``results[]``, and ``summary_table``. Adversarial tests cover
every fail-closed comparison requirement.
"""

from __future__ import annotations

import json
import hashlib
from typing import Any

import pytest

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
    decode_mode: str = "duration",
) -> dict:
    return {
        "version": version,
        "engine": "vllm",
        "model": "glm-5.2-exl3-tr3-3.25bpw",
        "timestamp": "2026-08-09T06:02:17.088583",
        "decode_mode": decode_mode,
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
    underfilled: bool = False,
    warmup_timed_out: bool = False,
    capacity_limited: bool = False,
) -> dict:
    return {
        "concurrency": conc,
        "context_tokens": context_tokens,
        "benchmark_mode": benchmark_mode,
        "measurement_seconds": measurement_seconds,
        "aggregate_tps": aggregate_tps,
        "num_errors": num_errors,
        "effective_concurrency": effective_concurrency if effective_concurrency is not None else conc,
        "underfilled": underfilled,
        "warmup_timed_out": warmup_timed_out,
        "capacity_limited": capacity_limited,
        "request_count": conc,
        "completed_request_count": 0,
    }


_OMIT = object()  # sentinel to skip auto summary_table generation


def _make_doc(
    meta: dict | None = None,
    results: list[dict] | None = None,
    summary_table: Any = None,
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
    if summary_table is None:
        summary: dict[str, dict[str, float]] = {}
        for r in results:
            ctx = str(r["context_tokens"])
            if ctx not in summary:
                summary[ctx] = {}
            summary[ctx][str(r["concurrency"])] = r["aggregate_tps"]
        summary_table = summary
    elif summary_table is _OMIT:
        pass  # no summary_table key in output
    return {
        "metadata": meta,
        "results": results,
        **({"summary_table": summary_table} if summary_table is not _OMIT else {}),
    }


# Standard valid documents
SUSTAINED_BASELINE = _make_doc(
    results=[
        _make_result(8, 64.56),
        _make_result(4, 45.13),
        _make_result(2, 29.29),
        _make_result(1, 15.89),
    ],
)

SUSTAINED_CANDIDATE = _make_doc(
    results=[
        _make_result(8, 62.00),
        _make_result(4, 44.00),
        _make_result(2, 28.50),
        _make_result(1, 15.50),
    ],
)

# Bounded gate: 128 tokens, 5s duration
BOUNDED_GATE = _make_doc(
    meta=_make_meta(
        max_tokens=128,
        duration_per_test=5,
        concurrency_levels=[1, 2, 8],
        context_lengths=[512],
        cell_warmup_timeout_seconds=60,
    ),
    results=[
        _make_result(1, 21.94, context_tokens=512),
        _make_result(2, 30.25, context_tokens=512),
        _make_result(8, 69.39, context_tokens=512),
    ],
)

# Older protocol variant
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

NO_METADATA: dict = {}

NO_RESULTS = {
    "metadata": _make_meta(),
    "results": [],
    "summary_table": {},
}

INVALID_CELLS = _make_doc(
    results=[
        _make_result(8, 64.56, num_errors=2),
        _make_result(4, 45.13),
        _make_result(2, 29.29),
        _make_result(1, 15.89),
    ],
)

UNDERFILLED = _make_doc(
    results=[
        _make_result(8, 30.0, effective_concurrency=3),
        _make_result(4, 45.13),
        _make_result(2, 29.29),
        _make_result(1, 15.89),
    ],
)


# ---------------------------------------------------------------------------
# Classification
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
    doc = _make_doc(meta=_make_meta(max_tokens=128, duration_per_test=25))
    assert cmp.classify_document_type(doc) == "bounded_gate"


def test_classify_bounded_low_duration():
    doc = _make_doc(meta=_make_meta(max_tokens=2048, duration_per_test=5))
    assert cmp.classify_document_type(doc) == "bounded_gate"


def test_classify_requires_both_duration_and_max_tokens():
    """Missing one of duration/max_tokens → indeterminate, not inferred."""
    doc = {"metadata": _make_meta(max_tokens=1024, duration_per_test=0)}
    # duration_per_test=0 is < 10 → bounded_gate (not sustained)
    assert cmp.classify_document_type(doc) == "bounded_gate"


def test_classify_requires_decode_mode_duration():
    """decode_mode != duration → indeterminate."""
    doc = _make_doc(meta=_make_meta(decode_mode="burst"))
    assert cmp.classify_document_type(doc) == "indeterminate"


def test_classify_requires_result_benchmark_mode_duration():
    """A result with benchmark_mode != duration → indeterminate."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, benchmark_mode="burst"),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    assert cmp.classify_document_type(doc) == "indeterminate"


# ---------------------------------------------------------------------------
# Settings extraction and matching
# ---------------------------------------------------------------------------


def test_extract_settings_from_metadata():
    settings = cmp.extract_settings(SUSTAINED_BASELINE)
    assert settings["engine"] == "vllm"
    assert settings["model"] == "glm-5.2-exl3-tr3-3.25bpw"
    assert settings["harness_version"] == "0.4.31"
    assert settings["decode_layer"] == "sustained_decode"
    assert settings["decode_mode"] == "duration"
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


def test_settings_mismatch_temperature():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["temperature"] = 0.7
    result = cmp.compare_settings(SUSTAINED_BASELINE, candidate)
    assert result["all_matched"] is False


@pytest.mark.parametrize("field", ["engine", "model"])
def test_settings_mismatch_runtime_identity(field):
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"][field] = f"different-{field}"
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "settings_mismatch"
    assert result["throughput"]["cells"] == []
    assert any(item["setting"] == field for item in result["settings"]["mismatched"])


def test_settings_missing_on_both_counts_as_mismatch():
    """Two documents missing the same setting must NOT count as matched."""
    base = json.loads(json.dumps(SUSTAINED_BASELINE))
    cand = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    del base["metadata"]["ignore_eos"]
    del cand["metadata"]["ignore_eos"]
    result = cmp.compare_settings(base, cand)
    assert result["all_matched"] is False
    assert any(
        m["reason"] == "missing on both documents" for m in result["mismatched"]
    )


def test_settings_mismatch_max_tokens():
    """Older protocol (2048) vs post-upgrade (1024) must mismatch."""
    result = cmp.compare_settings(SUSTAINED_BASELINE, OLDER_PROTOCOL)
    assert result["all_matched"] is False


# ---------------------------------------------------------------------------
# Throughput extraction
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
            _make_result(4, 30.0),
            _make_result(8, 40.0),
        ],
        "summary_table": {},
    }
    tps = cmp.extract_throughput(doc)
    assert tps == {"C1": 10.0, "C2": 20.0, "C4": 30.0, "C8": 40.0}


def test_extract_throughput_empty():
    assert cmp.extract_throughput({}) == {}


def test_extract_throughput_ignores_non_required_concurrencies():
    """Concurrencies outside C1/C2/C4/C8 are ignored."""
    doc = {
        "metadata": _make_meta(concurrency_levels=[1, 2, 4, 8, 16]),
        "results": [
            _make_result(1, 10.0),
            _make_result(2, 20.0),
            _make_result(4, 30.0),
            _make_result(8, 40.0),
            _make_result(16, 50.0),
        ],
        "summary_table": {},
    }
    tps = cmp.extract_throughput(doc)
    assert "C16" not in tps


# ---------------------------------------------------------------------------
# Validity extraction
# ---------------------------------------------------------------------------


def test_extract_validity_all_valid():
    v = cmp.extract_validity(SUSTAINED_BASELINE)
    assert v["all_cells_valid"] is True
    assert v["zero_cells"] is False
    assert len(v["missing_fields"]) == 0


def test_extract_validity_errors():
    v = cmp.extract_validity(INVALID_CELLS)
    assert v["all_cells_valid"] is False
    assert v["cells"]["C8"]["num_errors"] == 2


def test_extract_validity_underfilled():
    v = cmp.extract_validity(UNDERFILLED)
    assert v["all_cells_valid"] is False


def test_extract_validity_no_results():
    v = cmp.extract_validity(NO_RESULTS)
    assert v["zero_cells"] is True
    assert "all_cells_valid" not in v


def test_extract_validity_empty():
    v = cmp.extract_validity({})
    assert v["zero_cells"] is True
    assert "all_cells_valid" not in v


def test_extract_validity_missing_num_errors():
    """Missing num_errors is invalid, not default-zero."""
    doc = _make_doc(
        results=[
            {"concurrency": 8, "context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "effective_concurrency": 8, "underfilled": False,
             "warmup_timed_out": False, "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert ("C8", "num_errors") in v["missing_fields"]


def test_extract_validity_missing_effective_concurrency():
    """Missing effective_concurrency is invalid."""
    doc = _make_doc(
        results=[
            {"concurrency": 8, "context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "num_errors": 0, "underfilled": False,
             "warmup_timed_out": False, "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert ("C8", "effective_concurrency") in v["missing_fields"]


def test_extract_validity_missing_underfilled():
    """Missing underfilled is invalid, not default-false."""
    doc = _make_doc(
        results=[
            {"concurrency": 8, "context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "num_errors": 0, "effective_concurrency": 8,
             "warmup_timed_out": False, "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert ("C8", "underfilled") in v["missing_fields"]


def test_extract_validity_missing_benchmark_mode():
    """Missing benchmark_mode is invalid."""
    doc = _make_doc(
        results=[
            {"concurrency": 8, "context_tokens": 16384,
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "num_errors": 0, "effective_concurrency": 8,
             "underfilled": False, "warmup_timed_out": False,
             "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert ("C8", "benchmark_mode") in v["missing_fields"]


def test_extract_validity_warmup_timed_out():
    """warmup_timed_out=True makes cell invalid."""
    doc = _make_doc(
        results=[
            _make_result(8, 30.0, warmup_timed_out=True),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False


def test_extract_validity_capacity_limited():
    """capacity_limited=True makes cell invalid."""
    doc = _make_doc(
        results=[
            _make_result(8, 30.0, capacity_limited=True),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False


# ---------------------------------------------------------------------------
# Context coverage validation
# ---------------------------------------------------------------------------


def test_coverage_valid():
    result = cmp.validate_context_coverage(SUSTAINED_BASELINE)
    assert result["valid"] is True


def test_coverage_no_metadata():
    result = cmp.validate_context_coverage({})
    assert result["valid"] is False


def test_coverage_no_results():
    result = cmp.validate_context_coverage(NO_RESULTS)
    assert result["valid"] is False


def test_coverage_multi_context():
    """Multi-context document (two context_lengths) is rejected."""
    doc = _make_doc(
        meta=_make_meta(context_lengths=[16384, 32768]),
        results=[
            _make_result(8, 64.56, context_tokens=16384),
            _make_result(4, 45.13, context_tokens=32768),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False


def test_coverage_wrong_context():
    """Result with wrong context_tokens is rejected."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, context_tokens=8192),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False


def test_coverage_missing_concurrency():
    """Missing a required concurrency fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            # C1 missing
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "missing" in result["reason"].lower()


def test_coverage_extra_concurrency_metadata():
    """Extra concurrency in metadata concurrency_levels fails at metadata check."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[1, 2, 4, 8, 16]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
            _make_result(16, 80.0),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "concurrency_levels" in result["reason"]


def test_coverage_extra_concurrency_results_only():
    """Extra concurrency in results (metadata correct) fails at results check."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2, 1]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
            _make_result(16, 80.0),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not one of" in result["reason"].lower()


def test_coverage_duplicate_concurrency():
    """Duplicate concurrency fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56),
            _make_result(8, 60.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "duplicate" in result["reason"].lower()


def test_coverage_summary_table_multi_context():
    """Summary table with multiple context keys fails."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": 64.56, "4": 45.13, "2": 29.29, "1": 15.89},
            "32768": {"8": 50.0},
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "multiple" in result["reason"].lower()


def test_coverage_summary_table_missing_concurrency():
    """Summary table missing a concurrency that results have fails."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": 64.56, "4": 45.13, "2": 29.29},
            # C1 missing from summary
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False


# ---------------------------------------------------------------------------
# Full document comparison — no deltas on failure
# ---------------------------------------------------------------------------


def test_compare_matched_sustained():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert result["status"] == "compared"
    assert result["baseline_type"] == "sustained_matrix"
    assert result["candidate_type"] == "sustained_matrix"
    assert result["settings"]["all_matched"] is True
    assert len(result["throughput"]["cells"]) == 4


def test_compare_identical_sustained_documents_rejected():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_BASELINE)
    assert result["status"] == "identical_documents"
    assert result["throughput"]["cells"] == []


@pytest.mark.parametrize(
    ("field", "value", "expected_status"),
    [
        ("engine", "", "invalid_metadata"),
        ("duration_per_test", float("nan"), "type_mismatch"),
        ("max_tokens", True, "type_mismatch"),
        ("ignore_eos", 1, "invalid_metadata"),
    ],
)
def test_compare_invalid_metadata_emits_no_deltas(field, value, expected_status):
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"][field] = value
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == expected_status
    assert result["candidate_metadata"]["valid"] is False
    assert result["throughput"]["cells"] == []


def test_compare_context_percentages_must_sum_to_100():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["unique_context_percent"] = 25.0
    candidate["metadata"]["shared_context_percent"] = 25.0
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "invalid_metadata"
    assert "sum to 100" in result["candidate_metadata"]["reason"]


def test_compare_unsupported_harness_version_emits_no_deltas():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["version"] = "0.5.0"
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "invalid_metadata"
    assert "0.4.31" in result["candidate_metadata"]["reason"]
    assert result["throughput"]["cells"] == []


def test_compare_unsupported_decode_layer_emits_no_deltas():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["primary_decode_layer"] = "prefill"
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "invalid_metadata"
    assert "sustained_decode" in result["candidate_metadata"]["reason"]
    assert result["throughput"]["cells"] == []


def test_compare_huge_integer_metadata_fails_without_exception():
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["duration_per_test"] = 10**1000
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "invalid_metadata"
    assert result["throughput"]["cells"] == []


def test_compare_huge_integer_throughput_fails_without_exception():
    candidate = _make_doc(
        results=[
            _make_result(8, 10**1000),
            _make_result(4, 44.0),
            _make_result(2, 28.5),
            _make_result(1, 15.5),
        ],
    )
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "invalid_cells"
    assert result["throughput"]["cells"] == []


def test_compare_non_finite_derived_percentage_emits_no_deltas():
    baseline = _make_doc(
        results=[_make_result(conc, 1e-308) for conc in (8, 4, 2, 1)],
    )
    candidate = _make_doc(
        results=[_make_result(conc, 1e308) for conc in (8, 4, 2, 1)],
    )
    result = cmp.compare_documents(baseline, candidate)
    assert result["status"] == "invalid_delta"
    assert result["throughput"]["cells"] == []


def test_compare_type_mismatch_bounded_vs_sustained():
    """Bounded gate vs sustained matrix must produce type_mismatch."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, BOUNDED_GATE)
    assert result["status"] == "type_mismatch"


def test_compare_type_mismatch_indeterminate():
    """Indeterminate vs sustained must produce type_mismatch."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, NO_METADATA)
    assert result["status"] == "type_mismatch"


def test_compare_two_bounded_gate_rejected():
    """Two bounded_gate documents must NOT compare."""
    result = cmp.compare_documents(BOUNDED_GATE, BOUNDED_GATE)
    assert result["status"] == "type_mismatch"
    assert "both documents are bounded_gate" in result["type_mismatch"]


def test_compare_two_indeterminate_rejected():
    """Two indeterminate documents must NOT compare."""
    result = cmp.compare_documents({}, {})
    assert result["status"] == "type_mismatch"


def test_compare_settings_mismatch_no_deltas():
    """Settings mismatch must NOT emit any deltas."""
    candidate = json.loads(json.dumps(SUSTAINED_CANDIDATE))
    candidate["metadata"]["temperature"] = 1.0
    result = cmp.compare_documents(SUSTAINED_BASELINE, candidate)
    assert result["status"] == "settings_mismatch"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_protocol_mismatch_no_deltas():
    """Post-upgrade vs older protocol: settings_mismatch, no deltas."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, OLDER_PROTOCOL)
    assert result["status"] == "settings_mismatch"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_invalid_cells_no_deltas():
    """Invalid cells must NOT emit deltas."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, INVALID_CELLS)
    assert result["status"] == "invalid_cells"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_underfilled_no_deltas():
    """Underfilled cells must NOT emit deltas."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, UNDERFILLED)
    assert result["status"] == "invalid_cells"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_no_cells_no_deltas():
    """No results → no_cells, no deltas."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, NO_RESULTS)
    assert result["status"] == "no_cells"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_coverage_error_no_deltas():
    """Coverage error must NOT emit deltas."""
    doc_missing_c1 = _make_doc(
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
        ],
    )
    result = cmp.compare_documents(SUSTAINED_BASELINE, doc_missing_c1)
    assert result["status"] == "coverage_error"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_multi_context_no_deltas():
    """Multi-context document must NOT produce deltas."""
    multi_ctx = _make_doc(
        meta=_make_meta(context_lengths=[16384, 32768]),
        results=[
            _make_result(8, 64.56, context_tokens=16384),
            _make_result(4, 45.13, context_tokens=32768),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.compare_documents(SUSTAINED_BASELINE, multi_ctx)
    assert result["status"] == "coverage_error"
    assert len(result["throughput"]["cells"]) == 0


def test_compare_claim_note_present():
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert "claim_note" in result
    assert "sealed A/B" in result["claim_note"]


def test_compare_coverage_in_report():
    """Coverage validation results are included in the report."""
    result = cmp.compare_documents(SUSTAINED_BASELINE, SUSTAINED_CANDIDATE)
    assert "baseline_coverage" in result
    assert "candidate_coverage" in result
    assert result["baseline_coverage"]["valid"] is True


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
    assert report["inputs"]["baseline_sha256"] == hashlib.sha256(base.read_bytes()).hexdigest()
    assert report["inputs"]["candidate_sha256"] == hashlib.sha256(cand.read_bytes()).hexdigest()


def test_cli_identical_documents_exit_mismatch(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    payload = json.dumps(SUSTAINED_BASELINE)
    base.write_text(payload)
    cand.write_text(payload)
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    report = json.loads(capsys.readouterr().out)
    assert rc == cmp.EXIT_MISMATCH
    assert report["status"] == "identical_documents"


def test_cli_type_mismatch(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(BOUNDED_GATE))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_MISMATCH


def test_cli_two_bounded_rejected(tmp_path, capsys):
    """CLI: two bounded_gate documents exit MISMATCH."""
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(BOUNDED_GATE))
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


def test_cli_coverage_error(tmp_path, capsys):
    """CLI: coverage error exits INVALID."""
    doc_missing_c1 = _make_doc(
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
        ],
    )
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(doc_missing_c1))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_INVALID


def test_cli_invalid_cells(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(INVALID_CELLS))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_INVALID


def test_cli_no_cells(tmp_path, capsys):
    base = tmp_path / "base.json"
    cand = tmp_path / "cand.json"
    base.write_text(json.dumps(SUSTAINED_BASELINE))
    cand.write_text(json.dumps(NO_RESULTS))
    rc = cmp.main(["--baseline", str(base), "--candidate", str(cand)])
    assert rc == cmp.EXIT_INVALID


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
    assert cmp.SCHEMA == "sparkring-benchmark-comparison/v1"

# ---------------------------------------------------------------------------
# Adversarial coverage and validity mutations
# ---------------------------------------------------------------------------


# --- concurrency_levels set/order contract ---


def test_metadata_concurrencies_subset_fails():
    """Metadata concurrency_levels [1,2] with four result cells must fail closed."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[1, 2]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "concurrency_levels" in result["reason"]


def test_metadata_concurrencies_superset_fails():
    """Metadata concurrency_levels with extra entries fails closed."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2, 1, 16]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "concurrency_levels" in result["reason"]


def test_metadata_concurrencies_wrong_order_passes():
    """Metadata concurrency_levels [1,2,4,8] still passes (set comparison, order irrelevant)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[1, 2, 4, 8]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is True


def test_results_metadata_concurrencies_mismatch_fails():
    """Metadata says [8,4,2,1] but results only have C1/C2 — fail closed."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2, 1]),
        results=[
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "missing" in result["reason"].lower()


# --- context_tokens present and exactly 16384 ---


def test_missing_context_tokens_fails():
    """Result missing context_tokens must fail closed, not default."""
    doc = _make_doc(
        summary_table=_OMIT,
        results=[
            {"concurrency": 8, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "num_errors": 0, "effective_concurrency": 8,
             "underfilled": False, "warmup_timed_out": False,
             "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "context_tokens" in result["reason"]



def test_wrong_context_tokens_fails():
    """Result with context_tokens != 16384 fails closed."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, context_tokens=8192),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "context_tokens" in result["reason"]


@pytest.mark.parametrize("context", [16384.0, True, "16384"])
def test_result_context_numeric_alias_fails(context):
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, context_tokens=context),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "integer" in result["reason"]


@pytest.mark.parametrize("context", [16384.0, True, "16384"])
def test_metadata_context_numeric_alias_fails(context):
    doc = _make_doc(meta=_make_meta(context_lengths=[context]))
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "integer" in result["reason"]


def test_all_context_tokens_correct_passes():
    """All results with context_tokens=16384 passes."""
    result = cmp.validate_context_coverage(SUSTAINED_BASELINE)
    assert result["valid"] is True


# --- summary_table agrees with results within tolerance ---


def test_summary_disagrees_with_results_fails():
    """Summary table value that disagrees with results beyond tolerance fails."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": 999.0, "4": 45.13, "2": 29.29, "1": 15.89},
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "disagrees" in result["reason"].lower()


def test_summary_within_tolerance_passes():
    """Summary table values within tolerance of results pass."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": 64.560000001, "4": 45.13, "2": 29.29, "1": 15.89},
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is True


def test_summary_non_numeric_value_fails():
    """Summary table with non-numeric value fails."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": "not_a_number", "4": 45.13, "2": 29.29, "1": 15.89},
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not finite numeric" in result["reason"].lower()


def test_summary_missing_concurrency_fails():
    """Summary table missing a concurrency that results have fails."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": 64.56, "4": 45.13, "2": 29.29},
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False


def test_summary_no_override_of_results():
    """Summary table never silently overrides results — extract_throughput prefers results."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
        "summary_table": {
            "16384": {"8": 999.0, "4": 45.13, "2": 29.29, "1": 15.89},
        },
    }
    tps = cmp.extract_throughput(doc)
    assert tps["C8"] == 64.56  # results value, not summary


def test_summary_absent_passes():
    """If summary_table is absent, coverage validation still passes."""
    doc = {
        "metadata": _make_meta(),
        "results": [
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is True


@pytest.mark.parametrize("summary", [{}, [], None, "not-an-object"])
def test_summary_present_but_empty_or_malformed_fails(summary):
    doc = _make_doc(summary_table=_OMIT)
    doc["summary_table"] = summary
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "non-empty object" in result["reason"]


def test_summary_noncanonical_duplicate_concurrency_alias_fails():
    doc = _make_doc(summary_table=_OMIT)
    doc["summary_table"] = {
        "16384": {
            "1": 15.89,
            "01": 15.89,
            "2": 29.29,
            "4": 45.13,
            "8": 64.56,
        },
    }
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "exactly" in result["reason"]


# --- finite numeric aggregate_tps required ---


def test_missing_aggregate_tps_fails():
    """Missing aggregate_tps fails validity."""
    doc = _make_doc(
        summary_table=_OMIT,
        results=[
            {"concurrency": 8, "context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9,
             "num_errors": 0, "effective_concurrency": 8,
             "underfilled": False, "warmup_timed_out": False,
             "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert ("C8", "aggregate_tps") in v["missing_fields"]


def test_nan_aggregate_tps_fails():
    """NaN aggregate_tps fails validity."""
    doc = _make_doc(
        results=[
            _make_result(8, float("nan")),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "aggregate_tps" for f in v["invalid_fields"])


def test_inf_aggregate_tps_fails():
    """Infinity aggregate_tps fails validity."""
    doc = _make_doc(
        results=[
            _make_result(8, float("inf")),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "aggregate_tps" for f in v["invalid_fields"])


def test_boolean_aggregate_tps_fails():
    """Boolean aggregate_tps fails validity (bool is not a valid number)."""
    doc = _make_doc(
        results=[
            _make_result(8, True),  # type: ignore
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "aggregate_tps" for f in v["invalid_fields"])


def test_negative_aggregate_tps_fails():
    """Negative aggregate_tps fails validity."""
    doc = _make_doc(
        results=[
            _make_result(8, -10.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "aggregate_tps" for f in v["invalid_fields"])


def test_zero_aggregate_tps_fails():
    """Zero aggregate_tps fails validity."""
    doc = _make_doc(
        results=[
            _make_result(8, 0.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "aggregate_tps" for f in v["invalid_fields"])


def test_zero_tps_emits_no_deltas():
    """Zero throughput in one doc → invalid_cells, no deltas, even in normal mode."""
    doc = _make_doc(
        results=[
            _make_result(8, 0.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.compare_documents(SUSTAINED_BASELINE, doc)
    assert result["status"] == "invalid_cells"
    assert len(result["throughput"]["cells"]) == 0


# --- strict type checks ---


def test_num_errors_bool_fails():
    """num_errors as bool (True) fails — bool is not a valid int."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, num_errors=True),  # type: ignore
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "num_errors" for f in v["invalid_fields"])


def test_num_errors_nonzero_fails():
    """num_errors=1 fails validity."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, num_errors=1),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "num_errors" for f in v["invalid_fields"])


def test_effective_concurrency_bool_fails():
    """effective_concurrency as bool fails — bool is not a valid int."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, effective_concurrency=True),  # type: ignore
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "effective_concurrency" for f in v["invalid_fields"])


def test_effective_concurrency_mismatch_fails():
    """effective_concurrency != requested concurrency fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, effective_concurrency=3),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "effective_concurrency" for f in v["invalid_fields"])


def test_measurement_seconds_zero_fails():
    """measurement_seconds=0 fails (must be > 0)."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, measurement_seconds=0.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "measurement_seconds" for f in v["invalid_fields"])


def test_measurement_seconds_negative_fails():
    """measurement_seconds < 0 fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, measurement_seconds=-1.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "measurement_seconds" for f in v["invalid_fields"])


def test_measurement_seconds_nan_fails():
    """NaN measurement_seconds fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, measurement_seconds=float("nan")),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "measurement_seconds" for f in v["invalid_fields"])


def test_underfilled_not_bool_fails():
    """underfilled as non-bool (e.g. int 0) fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56),  # can't pass non-bool via fixture, use raw dict
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    doc["results"][0]["underfilled"] = 0  # int, not bool
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "underfilled" for f in v["invalid_fields"])


def test_warmup_timed_out_not_bool_fails():
    """warmup_timed_out as non-bool fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    doc["results"][0]["warmup_timed_out"] = 1  # int, not bool
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "warmup_timed_out" for f in v["invalid_fields"])


def test_capacity_limited_not_bool_fails():
    """capacity_limited as non-bool fails."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    doc["results"][0]["capacity_limited"] = 0  # int, not bool
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False
    assert any(f[0] == "C8" and f[1] == "capacity_limited" for f in v["invalid_fields"])


def test_underfilled_true_fails():
    """underfilled=True fails validity."""
    doc = _make_doc(
        results=[
            _make_result(8, 64.56, underfilled=True),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    v = cmp.extract_validity(doc)
    assert v["all_cells_valid"] is False


def test_all_flags_false_valid():
    """All boolean flags false, all int fields correct → valid."""
    v = cmp.extract_validity(SUSTAINED_BASELINE)
    assert v["all_cells_valid"] is True
    assert len(v["invalid_fields"]) == 0


# ---------------------------------------------------------------------------
# Concurrency is validated before set conversion
# ---------------------------------------------------------------------------


def test_metadata_bool_true_concurrency_fails():
    """metadata.concurrency_levels=[True,2,4,8] must fail (bool not int)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[True, 2, 4, 8]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_result_bool_true_concurrency_fails():
    """Result concurrency=True must fail (bool not int)."""
    doc = _make_doc(
        summary_table=_OMIT,
        results=[
            {"concurrency": True, "context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 15.89,
             "num_errors": 0, "effective_concurrency": 1,
             "underfilled": False, "warmup_timed_out": False,
             "capacity_limited": False},
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_metadata_duplicate_concurrency_fails():
    """metadata.concurrency_levels=[8,4,1,1] must fail (duplicate, 4 items)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 1, 1]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "duplicate" in result["reason"].lower()


def test_metadata_object_concurrency_fails():
    """metadata.concurrency_levels=[8,4,2,{}] must fail (unhashable, not exception)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2, {}]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_result_object_concurrency_fails():
    """Result concurrency={} must fail (unhashable, not exception)."""
    doc = _make_doc(
        summary_table=_OMIT,
        results=[
            {"concurrency": {}, "context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "num_errors": 0, "effective_concurrency": 8,
             "underfilled": False, "warmup_timed_out": False,
             "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_metadata_float_concurrency_fails():
    """metadata.concurrency_levels=[8.0,4,2,1] must fail (float not int)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8.0, 4, 2, 1]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_metadata_string_concurrency_fails():
    """metadata.concurrency_levels=['8','4','2','1'] must fail (string not int)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=["8", "4", "2", "1"]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_metadata_null_concurrency_fails():
    """metadata.concurrency_levels=[8,4,2,None] must fail (null not int)."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2, None]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "not integer-not-bool" in result["reason"]


def test_result_missing_concurrency_fails():
    """Result missing concurrency key must fail closed."""
    doc = _make_doc(
        summary_table=_OMIT,
        results=[
            {"context_tokens": 16384, "benchmark_mode": "duration",
             "measurement_seconds": 24.9, "aggregate_tps": 64.56,
             "num_errors": 0, "effective_concurrency": 8,
             "underfilled": False, "warmup_timed_out": False,
             "capacity_limited": False},
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "missing concurrency" in result["reason"].lower()


def test_metadata_wrong_concurrency_count_fails():
    """metadata.concurrency_levels=[8,4,2] (3 items) must fail."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "exactly 4" in result["reason"]


def test_result_duplicate_concurrency_fails():
    """Duplicate result concurrency must fail before building sets."""
    doc = _make_doc(
        summary_table=_OMIT,
        results=[
            _make_result(8, 64.56),
            _make_result(8, 60.0),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.validate_context_coverage(doc)
    assert result["valid"] is False
    assert "duplicate" in result["reason"].lower()


def test_bool_concurrency_no_exception_in_compare():
    """Full compare_documents with bool concurrency must not raise."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[True, 2, 4, 8]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.compare_documents(SUSTAINED_BASELINE, doc)
    assert result["status"] != "compared"
    assert len(result["throughput"]["cells"]) == 0


def test_object_concurrency_no_exception_in_compare():
    """Full compare_documents with object concurrency must not raise."""
    doc = _make_doc(
        meta=_make_meta(concurrency_levels=[8, 4, 2, {}]),
        results=[
            _make_result(8, 64.56),
            _make_result(4, 45.13),
            _make_result(2, 29.29),
            _make_result(1, 15.89),
        ],
    )
    result = cmp.compare_documents(SUSTAINED_BASELINE, doc)
    assert result["status"] != "compared"
    assert len(result["throughput"]["cells"]) == 0


def test_valid_concurrencies_pass():
    """Normal valid document still passes after stricter validation."""
    result = cmp.validate_context_coverage(SUSTAINED_BASELINE)
    assert result["valid"] is True
