import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def test_local_image_receipt_records_exact_dcp_restart_restores() -> None:
    receipt = json.loads(
        (HERE / "local-image-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["qualified"] is False
    assert receipt["image"]["image_id"] == (
        "sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225"
    )
    assert receipt["defaults"] == {
        "max_model_len": 1048576,
        "max_num_batched_tokens": 8192,
        "prefill_schedule_interval": 8,
        "kv_cache_bytes_per_rank": {
            "dcp1": 27917287424,
            "dcp2": 32212254720,
            "dcp4": 32212254720,
        },
        "full_ckv_gather_max_tokens": 524288,
        "publication_schema": "snapshot-v1",
        "manager_page_identity": "manager-pages-v2",
    }
    for name, expected in (("dcp2", "R8_DCP2_BLUE"), ("dcp4", "R8_DCP4_RED")):
        result = receipt[name]
        assert result["expected_visible_output"] == expected
        assert result["observed_visible_output"] == expected
        assert len(result["restart_restore_service_ms_by_physical_rank"]) == 4
        assert all(
            milliseconds > 0
            for milliseconds in result["restart_restore_service_ms_by_physical_rank"]
        )
    rerun = receipt["dcp4_fresh_restart_rerun"]
    assert rerun["published_span_tokens"] == 14336
    assert rerun["verified_restore_ranks"] == 4
    assert rerun["restore_rejections_observed"] == 0
    assert rerun["recomputations_observed"] == 0
    dcp1 = receipt["dcp1_kv_capacity_sweep"]
    assert dcp1["largest_healthy_capacity_candidate_kv_cache_bytes_per_rank"] == (
        41 * 1024**3
    )
    assert dcp1["largest_healthy_capacity_candidate_gpu_kv_cache_tokens"] == 2056272
    assert dcp1["published_span_tokens"] == 8192
    assert len(dcp1["restart_restore_service_ms_by_physical_rank"]) == 4
    assert dcp1["candidates"][-1]["result"] == (
        "rank-0-oom-during-kv-materialization"
    )
    deep = receipt["dcp1_deep_context"]
    assert deep["kv_cache_bytes_per_rank"] == 26 * 1024**3
    assert deep["gpu_kv_cache_tokens"] == 1303701
    assert deep["actual_prompt_tokens"] == 942767
    assert deep["needle_present"] is True
    assert deep["sparkcache_publication"]["aligned_tokens"] == 942592
    assert deep["sparkcache_publication"]["restart_restore_replayed"] is False


def test_validation_prose_keeps_evidence_boundary_explicit() -> None:
    text = " ".join(
        (HERE / "LIVE_VALIDATION.md").read_text(encoding="utf-8").split()
    )
    assert "not a general qualification" in text
    assert "942,767-token request" in text
    assert "no registry digest" in text


def test_deep_context_record_binds_the_26_gib_result() -> None:
    record = json.loads(
        (
            ROOT
            / "performance/records/glm53-flash"
            / "dcp1-deep-context-boundary-20260831.json"
        ).read_text(encoding="utf-8")
    )
    assert record["image_id"] == (
        "sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225"
    )
    success = record["results"][-1]
    assert success == {
        "kv_gib": 26,
        "requested_depth": 1000000,
        "actual_prompt_tokens": 942767,
        "completion_tokens": 113,
        "elapsed_seconds": 478.1,
        "finish_reason": "stop",
        "needle": "36161615",
        "result": "pass",
    }
    assert record["kv26"]["allocator_capacity_tokens"] == 1303701
    publication = record["sparkcache_publication"]
    assert publication["aligned_tokens"] == 942592
    assert publication["restart_restore_replayed"] is False
