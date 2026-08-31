import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


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
        "kv_cache_bytes_per_rank": 32212254720,
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
    assert dcp1["selected_kv_cache_bytes_per_rank"] == 41 * 1024**3
    assert dcp1["selected_gpu_kv_cache_tokens"] == 2056272
    assert dcp1["published_span_tokens"] == 8192
    assert len(dcp1["restart_restore_service_ms_by_physical_rank"]) == 4
    assert dcp1["candidates"][-1]["result"] == (
        "rank-0-oom-during-kv-materialization"
    )


def test_validation_prose_keeps_evidence_boundary_explicit() -> None:
    text = " ".join(
        (HERE / "LIVE_VALIDATION.md").read_text(encoding="utf-8").split()
    )
    assert "not a general qualification" in text
    assert "does not prove a complete 1M request" in text
    assert "no registry digest" in text
