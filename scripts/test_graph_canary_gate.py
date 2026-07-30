from __future__ import annotations

import copy
import json
from pathlib import Path

import graph_canary_gate as gate


def artifact(mode: str, *, native: bool = False) -> dict:
    ranks = []
    if native:
        ranks = [
            {
                "rank": rank,
                "captured_nodes": 128,
                "before": {
                    "published_sequence": 224,
                    "consumed_sequence": 224,
                    "completed_sequence": 224,
                    "overflow_sequence": 0,
                },
                "after": {
                    "published_sequence": 256,
                    "consumed_sequence": 256,
                    "completed_sequence": 256,
                    "overflow_sequence": 0,
                },
                "submit_affinity_verified": True,
                "progress_affinity_verified": True,
            }
            for rank in range(4)
        ]
    return {
        "schema": gate.SCHEMA,
        "mode": mode,
        "model_identity_sha256": "a" * 64,
        "prompt_sha256": "b" * 64,
        "request": {
            "temperature": 0,
            "seed": 20260730,
            "max_tokens": 64,
            "mtp_enabled": False,
        },
        "response": {
            "http_status": 200,
            "finish_reason": "length",
            "output_token_ids": list(range(32)),
        },
        "expect_native_graph": native,
        "graph_ranks": ranks,
    }


def test_matching_stock_graph_canary_passes() -> None:
    assert gate.compare_artifacts(
        artifact("eager"),
        artifact("graph"),
        minimum_output_tokens=16,
    ) == []


def test_one_token_lock_symptom_fails_sharp_gate() -> None:
    eager = artifact("eager")
    graph = artifact("graph")
    graph["response"]["output_token_ids"] = [42]
    failures = gate.compare_artifacts(eager, graph, minimum_output_tokens=16)
    assert any("graph produced 1 token" in failure for failure in failures)
    assert any("differ from the eager oracle" in failure for failure in failures)


def test_native_graph_requires_progress_and_rank_sync() -> None:
    eager = artifact("eager")
    graph = artifact("graph", native=True)
    graph["graph_ranks"][2]["captured_nodes"] = 127
    graph["graph_ranks"][3]["after"]["completed_sequence"] = 255
    failures = gate.compare_artifacts(eager, graph, minimum_output_tokens=16)
    assert any("not rank-synchronous" in failure for failure in failures)
    assert any(
        "rank 3 replay is not caught up after request" in failure
        for failure in failures
    )


def test_stale_native_sequence_counters_fail() -> None:
    eager = artifact("eager")
    graph = artifact("graph", native=True)
    for rank in graph["graph_ranks"]:
        rank["after"] = copy.deepcopy(rank["before"])
    failures = gate.compare_artifacts(eager, graph, minimum_output_tokens=16)
    assert sum(
        "published_sequence did not advance during request" in failure
        for failure in failures
    ) == 4


def test_native_graph_requires_rank_synchronous_request_delta() -> None:
    eager = artifact("eager")
    graph = artifact("graph", native=True)
    graph["graph_ranks"][2]["after"] = {
        "published_sequence": 255,
        "consumed_sequence": 255,
        "completed_sequence": 255,
        "overflow_sequence": 0,
    }
    failures = gate.compare_artifacts(eager, graph, minimum_output_tokens=16)
    assert any(
        "advancement is not rank-synchronous" in failure for failure in failures
    )


def test_native_graph_rejects_overflow_before_or_after_request() -> None:
    eager = artifact("eager")
    graph = artifact("graph", native=True)
    graph["graph_ranks"][0]["before"]["overflow_sequence"] = 4
    graph["graph_ranks"][1]["after"]["overflow_sequence"] = 1
    failures = gate.compare_artifacts(eager, graph, minimum_output_tokens=16)
    assert any("rank 0 overflow_sequence is nonzero (4->0)" in f for f in failures)
    assert any("rank 1 overflow_sequence is nonzero (0->1)" in f for f in failures)


def test_native_graph_healthy_status_passes() -> None:
    assert gate.compare_artifacts(
        artifact("eager"),
        artifact("graph", native=True),
        minimum_output_tokens=16,
    ) == []


def test_finish_reason_must_match_eager_oracle() -> None:
    eager = artifact("eager")
    graph = artifact("graph")
    graph["response"]["finish_reason"] = "stop"
    failures = gate.compare_artifacts(eager, graph, minimum_output_tokens=16)
    assert any("finish_reason differs" in failure for failure in failures)


def test_empty_finish_reason_is_rejected() -> None:
    graph = artifact("graph")
    graph["response"]["finish_reason"] = ""
    try:
        gate.validate_artifact(graph, expected_mode="graph")
    except gate.ArtifactError as exc:
        assert "expected non-empty string" in str(exc)
    else:
        raise AssertionError("empty finish reason was accepted")


def test_mtp_enabled_is_rejected_for_first_isolation_gate() -> None:
    graph = artifact("graph")
    graph["request"]["mtp_enabled"] = True
    try:
        gate.validate_artifact(graph, expected_mode="graph")
    except gate.ArtifactError as exc:
        assert "first graph canary must disable MTP" in str(exc)
    else:
        raise AssertionError("MTP-on isolation artifact was accepted")


def test_cli_reports_semantic_failure(tmp_path: Path, capsys) -> None:
    eager = artifact("eager")
    graph = artifact("graph")
    graph["response"]["output_token_ids"] = [7]
    eager_path = tmp_path / "eager.json"
    graph_path = tmp_path / "graph.json"
    eager_path.write_text(json.dumps(eager), encoding="utf-8")
    graph_path.write_text(json.dumps(graph), encoding="utf-8")
    assert gate.main(
        ["--eager", str(eager_path), "--graph", str(graph_path)]
    ) == 3
    captured = capsys.readouterr()
    assert "graph canary: FAIL" in captured.err


def test_unknown_fields_fail_closed() -> None:
    candidate = copy.deepcopy(artifact("graph"))
    candidate["silently_ignore_me"] = True
    try:
        gate.validate_artifact(candidate, expected_mode="graph")
    except gate.ArtifactError as exc:
        assert "unknown key" in str(exc)
    else:
        raise AssertionError("unknown graph artifact field was accepted")
