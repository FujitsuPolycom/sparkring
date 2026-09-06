"""Reuse evidence tests are entirely offline; incomplete quorum never verifies."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("reuse_analysis_test", Path(__file__).with_name("analyze_conversation_reuse.py"))
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def turn(**overrides):
    return {"type": "turn", "request_id": "soak-one", "response_id": "chatcmpl-soak-one", "valid": True,
            "phase": "soak", "cached_tokens_reported": 0, "usage": {"prompt_tokens": 12000},
            "elapsed_seconds": 5, "ttft_seconds": 1, **overrides}


def completion(rank, **overrides):
    return {"schema": analysis.TRACE_SCHEMA, "event": "worker_restore_completed",
            "request_id": "chatcmpl-soak-one", "role": "worker", "rank": rank,
            "dcp_rank": rank % 2, "time_ns": 100, "digest": "0123456789ab",
            "requested_span_tokens": 10000, "verified_span_tokens": 10000, "outcome": "verified",
            "queue_wait_ms": 1 + rank, "service_ms": 20, "end_to_end_ms": 21 + rank,
            "phase_ms": {"read": 12, "place": 8}, **overrides}


def test_log_prefix_and_docker_envelope():
    record = completion(0)
    line = "2026-09-06 (Worker_TP0) INFO " + analysis.MARKER + json.dumps(record)
    assert analysis.parse_trace(line) == record
    assert analysis.parse_trace(json.dumps({"log": line, "stream": "stderr"})) == record
    assert analysis.parse_trace("not a trace") is None
    assert analysis.parse_trace(analysis.MARKER + "{partial") is None


def test_every_physical_rank_required_and_tokens_not_summed():
    report = analysis.analyze([turn()], [completion(rank) for rank in range(4)], [0, 1, 2, 3])
    row = report["turns"][0]
    assert row["source"] == "verified_all_rank_external_restore"
    assert row["verified_span_tokens"] == 10000
    assert row["missing_ranks"] == []
    assert report["summary"]["per_rank_restore_timings"]["3"]["median_queue_wait_ms"] == 4
    assert report["summary"]["per_rank_restore_timings"]["2"]["median_phase_ms"] == {"read": 12, "place": 8}


@pytest.mark.parametrize("traces", [
    [completion(0), completion(1)],
    [completion(rank, outcome="recompute", verified_span_tokens=0) for rank in range(4)],
    [completion(rank, digest="different" if rank == 3 else "0123456789ab") for rank in range(4)],
    [completion(rank, requested_span_tokens=11000 if rank == 3 else 10000) for rank in range(4)],
    [completion(rank, role="scheduler") for rank in range(4)],
])
def test_partial_failed_or_mismatched_completions_do_not_verify(traces):
    row = analysis.analyze([turn()], traces, [0, 1, 2, 3])["turns"][0]
    assert row["source"] == "unknown"
    assert row["verified_span_tokens"] is None


def test_offer_is_not_restore_and_zero_api_not_local_miss():
    offer = {"schema": analysis.TRACE_SCHEMA, "request_id": "chatcmpl-soak-one", "role": "scheduler",
             "event": "external_restore_offer", "selected_span_tokens": 10000}
    row = analysis.analyze([turn()], [offer], [0, 1, 2, 3])["turns"][0]
    assert row["source"] == "unknown"
    assert row["offer_count"] == 1
    assert not row["all_rank_worker_verification"]
    missing = analysis.analyze([turn(cached_tokens_reported=None)], [], [0])["turns"][0]
    assert missing["source"] == "unknown"


def test_gpu_attachment_and_positive_report_are_separate_evidence():
    attached = {"schema": analysis.TRACE_SCHEMA, "request_id": "soak-one", "event": "gpu_lease_attached",
                "role": "scheduler", "lease_span_tokens": 9000}
    row = analysis.analyze([turn(cached_tokens_reported=8000)], [attached], [0, 1, 2, 3])["turns"][0]
    assert row["source"] == "gpu_lease_attached"
    assert not row["all_rank_worker_verification"]
    assert row["lease_span_tokens_observed"] == 9000
    assert analysis.analyze([turn(cached_tokens_reported=8000)], [], [0])["turns"][0]["source"] == "reported_cached"


def test_latest_failure_overrides_prior_success_and_duplicates_do_not_make_quorum():
    records = [completion(rank) for rank in range(4)]
    records += [completion(3, time_ns=200, outcome="recompute", verified_span_tokens=0)]
    report = analysis.analyze([turn()], records + records, [0, 1, 2, 3])
    assert report["turns"][0]["source"] == "unknown"
    assert report["summary"]["duplicate_trace_events"] == 5
    assert analysis.analyze([turn()], [completion(0)] * 4, [0, 1, 2, 3])["turns"][0]["source"] == "unknown"


def test_conflicting_same_time_and_ambiguous_ids_do_not_verify():
    records = [completion(rank) for rank in range(4)] + [completion(0, outcome="recompute", verified_span_tokens=0)]
    row = analysis.analyze([turn()], records, [0, 1, 2, 3])["turns"][0]
    assert row["source"] == "unknown" and row["conflicting_ranks"] == [0]
    report = analysis.analyze([turn(), turn()], [completion(rank) for rank in range(4)], [0, 1, 2, 3])
    assert report["summary"]["ambiguous_turns"] == 2
    assert all(row["source"] == "unknown" for row in report["turns"])


def test_unmatched_id_not_fuzzily_associated_and_rank_set_explicit():
    report = analysis.analyze([turn()], [completion(0, request_id="chatcmpl-soak-one-other")], [0])
    assert report["summary"]["unmatched_trace_events"] == 1
    with pytest.raises(ValueError, match="explicit"):
        analysis.analyze([turn()], [], [])


def test_malformed_completion_never_verifies():
    row = analysis.analyze([turn()], [completion(0, requested_span_tokens=[10000])], [0])["turns"][0]
    assert row["source"] == "unknown"
    row = analysis.analyze([turn()], [completion(0, digest=[])], [0])["turns"][0]
    assert row["source"] == "unknown"


def test_phase_populations_remain_separate_and_retry_timings_retained():
    turns = [turn(), turn(phase="before", request_id="probe-id", response_id="chatcmpl-probe-id", elapsed_seconds=1)]
    report = analysis.analyze(turns, [completion(0, time_ns=50, outcome="recompute", verified_span_tokens=0), completion(0)], [0])
    phases = report["summary"]["by_phase_and_source"]
    assert phases["soak"]["verified_all_rank_external_restore"]["median_latency_seconds"] == 5
    assert phases["before"]["unknown"]["median_latency_seconds"] == 1
    assert report["summary"]["per_rank_restore_timings"]["0"]["outcomes"] == {"recompute": 1, "verified": 1}


def test_observed_response_id_engine_nonce_joins_strictly():
    response_id = "chatcmpl-soak-cc0bdb28896248beab09f145c9418947"
    rows = [completion(rank, request_id=response_id + "-b4884e31") for rank in range(4)]
    report = analysis.analyze([turn(response_id=response_id)], rows, [0, 1, 2, 3])
    assert report["turns"][0]["source"] == "verified_all_rank_external_restore"
    assert report["summary"]["unmatched_trace_events"] == 0
    for suffix in ("-b4884e3", "-b4884e311", "-b4884e3z", "-b4884e31-more"):
        report = analysis.analyze([turn(response_id=response_id)], [completion(0, request_id=response_id + suffix)], [0])
        assert report["turns"][0]["source"] == "unknown"
        assert report["summary"]["unmatched_trace_events"] == 1


def test_engine_nonce_requires_observed_response_id_and_refuses_collisions():
    engine_id = "chatcmpl-soak-one-1234abcd"
    row = analysis.analyze([turn(response_id=None)], [completion(0, request_id=engine_id)], [0])["turns"][0]
    assert row["source"] == "unknown"
    turns = [turn(), turn(request_id=engine_id, response_id="other-api-id")]
    report = analysis.analyze(turns, [completion(0, request_id=engine_id)], [0])
    assert report["summary"]["ambiguous_turns"] == 2
    assert all(row["source"] == "unknown" for row in report["turns"])


def test_different_engine_requests_cannot_be_combined_into_quorum():
    rows = [completion(0, request_id="chatcmpl-soak-one-1234abcd"),
            completion(1, request_id="chatcmpl-soak-one-5678abcd")]
    row = analysis.analyze([turn()], rows, [0, 1])["turns"][0]
    assert row["source"] == "unknown"
    assert not row["all_rank_worker_verification"]
