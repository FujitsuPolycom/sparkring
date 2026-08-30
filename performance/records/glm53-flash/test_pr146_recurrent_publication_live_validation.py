"""Validate the bounded PR146 recurrent-publication evidence record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "pr146-recurrent-publication"
    / "validation.json"
)
RECORD_PATH = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "pr146-recurrent-publication-live-validation.md"
)
QUICKSTART_PATH = (
    ROOT / "docs" / "GLM53_DFLASH7_PYTHON_OVERLAY_SPARKCACHE_TP4_QUICKSTART.md"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_binds_exact_final_artifact_and_source_contracts() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "sparkring-glm53-pr146-live-validation/v2"
    assert receipt["normalization"] == {
        "supersedes_schema": "sparkring-glm53-pr146-live-validation/v1",
        "reason": (
            "The C16 observation is classified as shared exact prefix reuse for "
            "block_pages_v1; it does not demonstrate per_token_rows different-root "
            "descriptor coalescing."
        ),
        "raw_measurements_changed": False,
    }
    artifact = receipt["artifact"]
    assert artifact["image_id"] == (
        "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8"
    )
    assert artifact["published_digest"] is None
    assert artifact["sparkring_revision"] == (
        "d93cb3d98305041081cf572521602625185112ae"
    )
    assert artifact["sparkcache_revision"] == (
        "65b6642df1afc64366430d3aef9aca01f5c5e1c3"
    )
    assert artifact["sparkcache_source_sha256"] == (
        "a2add45a9f97446f6c2a843355161da9a5499ff7501b4750d2163591785d7345"
    )
    assert artifact["sparkcache_vllm_contract_sha256"] == (
        "8adbdfa3fd4b06b213c3aab45255a0b039f1c9940a4b1fad0efd004d263227c9"
    )


def test_status_scopes_functional_and_performance_claims() -> None:
    status = _receipt()["status"]
    for status_name in (
        "semantic_canary",
        "persistent_8k_restore_after_restart",
        "persistent_128k_restore",
        "tail_publication_128k_to_256k",
        "persistent_256k_restore_correctness",
        "shared_exact_prefix_c16",
    ):
        assert status[status_name] == "qualified"
    assert status["restore_performance"] == "research-only"
    assert (
        status["different_root_row_descriptor_coalescing_live_validation"]
        == "unsupported"
    )
    assert status["continuous_availability_during_replacement"] == "unsupported"
    assert status["dflash_response_quality"] == "unsupported"


def test_construction_receipt_precedes_replacement_container_creation() -> None:
    ordering = _receipt()["measurements"][
        "construction_verification_before_container_creation"
    ]
    assert ordering["image_id"] == (
        "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8"
    )
    assert ordering["image_receipt_sha256"] == (
        "2c4a02efe91df5de21c5e3c92f65710b7d41680f25c22baed43ca96c1e5a51d3"
    )
    receipt_time = ordering["image_receipt_modified_at"]
    assert receipt_time == "2026-08-30T05:23:18.699254136-05:00"
    assert ordering["image_receipt_bytes"] == 8690
    assert all(
        created > "2026-08-30T10:23:18.699254136Z"
        for created in ordering["container_created_at_by_rank"].values()
    )
    assert ordering["continuous_availability"] == "unsupported"


def test_clean_restart_8k_restore_is_verified_on_every_rank() -> None:
    restore = _receipt()["measurements"]["persistent_restore_8k_after_clean_relaunch"]
    assert restore["request_id"] == "chatcmpl-9c60514089aed6f5-b033fd19"
    assert restore["restored_tokens"] == 8192
    assert restore["page_bytes_per_rank"] == 103841965
    assert restore["client_elapsed_seconds"] == 1.573
    assert {
        rank: values["end_to_end_ms"]
        for rank, values in restore["rank_restore"].items()
    } == {
        "0": 66.550,
        "1": 59.759,
        "2": 60.924,
        "3": 63.868,
    }
    assert all(
        values["outcome"] == "verified"
        for values in restore["rank_restore"].values()
    )


def test_tail_manifest_proves_bounded_copy_on_write_shape() -> None:
    manifest = _receipt()["measurements"]["tail_publication_128k_to_256k"]
    assert manifest == {
        "manifest_schema": "sparkcache-page-delta-manifest/v2",
        "manifest_sha256": "f21221318441d809c7393582741ff896cec0cf82e5c35c34a6dabd3d5b77eba6",
        "context_digest": "e532f048762ffec5bcc0c69283a3d5afacb652efbbf4267d34afab99737460b9",
        "base_context_digest": "2010bc9db8fee25489d1580d1105cdcc819fa7a04c8edeffaf23d298e52452cd",
        "base_committed_tokens": 131072,
        "result_committed_tokens": 262144,
        "base_root_objects": 512,
        "delta_encoded_bytes": 826457677,
        "delta_object_target_bytes": 67108864,
        "delta_object_count": 13,
        "last_delta_object_bytes": 21151309,
        "delta_sha256": "3976f30ef3575e6f8e80b1f90f742d22c2cac30c7d428b96e69d4ca98bf5a41c",
        "layout_sha256": "0d0b44eeb515963cf262f5d4d5b345caf794b85b1fa0cf11ad18576c1e8e7331",
        "logical_chunk_tokens": 256,
    }


def test_256k_restore_retains_all_rank_correctness_evidence() -> None:
    restore = _receipt()["measurements"]["persistent_restore_256k"]
    assert restore["restored_tokens"] == 262144
    assert restore["page_bytes_per_rank"] == 1575821491
    assert restore["logical_chunk_count"] == 1024
    assert {
        rank: values["end_to_end_ms"]
        for rank, values in restore["rank_restore"].items()
    } == {
        "0": 7149.323,
        "1": 6688.887,
        "2": 7206.548,
        "3": 6738.472,
    }
    assert all(
        values["outcome"] == "verified"
        for values in restore["rank_restore"].values()
    )


def test_shared_exact_prefix_cohort_records_one_logical_restore() -> None:
    receipt = _receipt()
    shared = receipt["measurements"]["shared_exact_prefix_c16"]
    assert shared["scenario"] == "shared_exact_prefix"
    assert shared["storage_mode"] == "block_pages_v1"
    assert receipt["conditions"]["restore_storage_mode"] == "block_pages_v1"
    assert receipt["conditions"]["different_root_row_descriptor_coalescing"] == (
        "implemented with GPU-free coverage; not exercised by this artifact"
    )
    assert shared["concurrency"] == 16
    assert shared["distinct_prompt_sha256_count"] == 16
    assert len(shared["prompt_sha256"]) == len(set(shared["prompt_sha256"])) == 16
    assert shared["http_200"] == 16
    assert shared["unsuccessful"] == 0
    assert len(shared["request_elapsed_seconds"]) == 16
    assert shared["external_restore_events_per_rank"] == 1
    assert shared["external_restore_request_id"] == (
        "cmpl-9ac4e6b523215233-0-9c41ccbc"
    )
    assert set(shared["external_restore"]["rank_restore"]) == {"0", "1", "2", "3"}
    assert shared["recurrent_warning_count"] == 0


def test_predecessor_evidence_is_not_transferred_to_final_image() -> None:
    receipt = _receipt()
    assert receipt["predecessor_artifact"]["image_id"] == (
        "sha256:1b4e58dc0999292da34d7418688b2b7f745a5b4d06e048ceb19f06f9d63a1185"
    )
    predecessor = receipt["measurements"]["predecessor_same_boundary_api"]
    assert [wave["concurrency"] for wave in predecessor["identical_prefix_waves"]] == [
        2,
        8,
        16,
    ]
    assert "not transferred" in predecessor["scope"]
    limitations = "\n".join(receipt["limitations"])
    assert "do not qualify the final image" in limitations


def test_record_follows_evidence_method_and_rejects_speed_claim() -> None:
    record = RECORD_PATH.read_text(encoding="utf-8")
    for heading in (
        "## Conditions",
        "## Measurement",
        "## Result",
        "## Conclusion",
        "## Limitations",
    ):
        assert heading in record
    assert "One logical external restore" in record
    assert "not a speed claim" in record
    assert "shared exact prefix cohort" in record
    assert "does not qualify\n`per_token_rows` different-root" in record
    assert "do not qualify the final image" in record
    assert "`--prefill-schedule-interval 8` was not tested" in record
    assert (
        "../../receipts/glm53-flash/pr146-recurrent-publication/validation.json"
        in record
    )


def test_quickstart_links_the_exact_artifact_evidence_separately() -> None:
    quickstart = QUICKSTART_PATH.read_text(encoding="utf-8")
    assert (
        "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8"
        in quickstart
    )
    assert "d93cb3d98305041081cf572521602625185112ae" in quickstart
    assert "65b6642df1afc64366430d3aef9aca01f5c5e1c3" in quickstart
    assert "pr146-recurrent-publication-live-validation.md" in quickstart
    assert "`block_pages_v1`" in quickstart
    assert "`per_token_rows` different-root descriptor-segment" in quickstart
    assert "rebuilds are not qualified" in quickstart
    assert "### Separate historical artifact" in quickstart
    assert (
        "sha256:eef863d8bc578815a80b0e2d9f0d745102b6363415225101fd92171a2e5a55cb"
        in quickstart
    )
