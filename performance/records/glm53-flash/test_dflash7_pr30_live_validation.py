"""Validate the bounded GLM-5.3 DFlash7 SparkCache record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "dflash7-python-overlay-pr30"
    / "validation.json"
)
RECORD_PATH = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "dflash7-python-overlay-pr30-live-validation.md"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_binds_the_exact_image_and_runtime_sources() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "sparkring-glm53-dflash7-pr30-live-validation/v1"
    artifact = receipt["artifact"]
    assert artifact["image_id"] == (
        "sha256:eef863d8bc578815a80b0e2d9f0d745102b6363415225101fd92171a2e5a55cb"
    )
    assert artifact["published_digest"] is None
    assert artifact["sparkcache_revision"] == (
        "5ec6a9953ad5d39120298bbfc26e95a6fa4b1dc3"
    )
    assert artifact["sparkcache_tree"] == (
        "94c236b9dfbf5f70075eb47877fd9caaa5d8c249"
    )


def test_status_does_not_transfer_concurrency_or_quality() -> None:
    status = _receipt()["status"]
    assert status["startup_health"] == "qualified"
    assert status["persistent_128k_restore"] == "qualified"
    assert status["persistent_256k_restore_correctness"] == "qualified"
    assert status["persistent_256k_restore_performance"] == "research-only"
    assert status["shared_prefix_concurrency"] == "unsupported"
    assert status["dflash_response_quality"] == "unsupported"


def test_receipt_preserves_loader_health_and_restore_observations() -> None:
    measurements = _receipt()["measurements"]
    assert measurements["startup"]["target_fastsafetensors_seconds"] == 53.93
    assert measurements["startup"]["draft_safetensors_seconds"] == 4.85
    assert measurements["startup"]["restart_count_by_rank"] == [0, 0, 0, 0]
    assert measurements["startup"]["oom_killed_by_rank"] == [
        False,
        False,
        False,
        False,
    ]
    assert measurements["arbitrary_page_boundary_restore"]["chunk_count"] == 47
    restore_128k = measurements["persistent_restore_128k"]
    assert restore_128k["restored_tokens"] == 131072
    assert restore_128k["completion_token"] == 13
    restore_256k = measurements["persistent_restore_256k"]
    assert restore_256k["restored_tokens"] == 262144
    assert restore_256k["prime_completion_token"] == 916
    assert restore_256k["replay_completion_token"] == 916
    assert restore_256k["rank_0_chunk_count"] == 1024


def test_record_has_required_evidence_sections_and_scope() -> None:
    record = RECORD_PATH.read_text(encoding="utf-8")
    for heading in (
        "## Conditions",
        "## Measurement",
        "## Result",
        "## Conclusion",
        "## Limitations",
    ):
        assert heading in record
    assert "Concurrency observations from another image do not qualify" in record
    assert "macro-object layout" in record
    assert "prefill-schedule-interval 8" in record
