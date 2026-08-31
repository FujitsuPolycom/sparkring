from __future__ import annotations

import json
import re
import hashlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = HERE / "pins.json"
ARTIFACT = HERE / "qualified-artifact.json"
LAUNCHER = HERE / "launch-qualified-rank.sh"
ROLLBACK = HERE / "rollback-rank.sh"
RECEIPT = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "split-page-shared-base-c8-20260830"
    / "validation.json"
)
CLIENT_RESULTS = RECEIPT.parent / "client-results.json"
RECORD = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "split-page-shared-base-c8-20260830.md"
)


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_artifact_binds_exact_sources_models_and_runtime() -> None:
    artifact = _json(ARTIFACT)
    assert artifact["schema"] == "sparkring-glm53-split-page-sparkcache-artifact/v1"
    assert artifact["status"] == "qualified"
    assert artifact["artifact"]["image_id"] == (
        "sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818"
    )
    assert artifact["artifact"]["published_oci_digest"] is None
    assert artifact["artifact"]["source_digest_sha256"] == (
        "21035ceeeab514b573dffc9a8b415b246fad8cd11ad83c633717b37f2bf6dd1b"
    )
    assert artifact["sources"]["sparkcache"]["commit"] == (
        "59ac0b04db6035a9a9d2a52e92405ceaf84daa40"
    )
    assert artifact["sources"]["sparkcache"]["cache_namespace_change"] == "none"
    assert artifact["sources"]["vllm"]["sparkcache_composition_commit"] == (
        "6da4865d440608a46eada50f27b2fff0e698c574"
    )
    assert artifact["sources"]["b12x"]["commit"] == (
        "b1d541f9e71a35f030d45fae437630fff7507c2a"
    )
    serving = artifact["serving"]
    assert serving["tensor_parallel_size"] == 4
    assert serving["decode_context_parallel_size"] == 1
    assert serving["kv_cache_dtype"] == "fp8"
    assert serving["kv_cache_bytes_per_rank"] == 20 * 1024**3
    assert serving["target_page_size_tokens"] == 512
    assert serving["recurrent_page_size_tokens"] == 512
    assert serving["sparkcache"]["load_lanes"] == 8
    assert serving["sparkcache"]["max_pending_restores"] == 8
    assert serving["sparkcache"]["cuda_arena_bytes_per_lane"] == 256 * 1024**2
    qualification = artifact["qualification"]
    assert qualification["scheduler_inventory_warmup_required_after_restart"] is True
    assert qualification["external_restores"] == 8
    assert qualification["physical_base_reads_per_rank"] == 1
    assert qualification["avoided_base_reads_per_rank"] == 7


def test_public_input_lock_matches_qualified_artifact() -> None:
    pins = _json(PINS)
    artifact = _json(ARTIFACT)
    assert pins["schema"] == "sparkring-glm53-split-page-sparkcache-input-lock/v1"
    assert pins["status"] == "implemented"
    assert pins["source_digest_sha256"] == artifact["artifact"]["source_digest_sha256"]
    assert pins["sources"]["vllm"]["composition_patch_sha256"] == _sha256(
        HERE / pins["sources"]["vllm"]["composition_patch"]
    )
    assert pins["sparkcache_cuda_placement_library_sha256"] == (
        artifact["sources"]["sparkcache_cuda_placement_library_sha256"]
    )
    assert pins["models"]["target"] == artifact["models"]["target"]
    assert pins["models"]["draft"] | {"license": None} == (
        artifact["models"]["draft"] | {"license": None}
    )


def test_c8_receipt_preserves_exactness_and_safe_recompute_boundary() -> None:
    receipt = _json(RECEIPT)
    assert receipt["schema"] == (
        "sparkring-glm53-split-page-shared-base-c8-validation/v1"
    )
    assert receipt["status"]["eight_distinct_request_semantics"] == "qualified"
    assert receipt["status"]["authenticated_shared_base_read"] == "qualified"
    assert receipt["status"]["eight_external_restores"] == "qualified"
    measurement = receipt["measurement"]
    assert measurement["concurrency"] == 8
    assert measurement["exact_response_count"] == 8
    assert measurement["source_client_result_sha256"] == (
        "3e437c65dc5c4eb5f8ec0723d9fe10b58c18f3246a265c1f19fd877c2cef89ad"
    )
    assert measurement["client_result_path"] == CLIENT_RESULTS.name
    assert measurement["client_result_sha256"] == _sha256(CLIENT_RESULTS)
    client = _json(CLIENT_RESULTS)
    assert client["status"] == "verified"
    assert len(client["results"]) == 8
    assert all(result["oracle_match"] for result in client["results"])
    assert measurement["external_restore"]["request_count"] == 8
    assert measurement["external_restore"]["hit_ratio"] == 1.0
    per_rank = measurement["external_restore"]["per_rank"]
    assert per_rank["participants"] == 8
    assert per_rank["physical_base_reads"] == 1
    assert per_rank["avoided_base_reads"] == 7
    assert per_rank["base_bytes"] == 100868258
    assert measurement["restart_readiness"]["worker_manifest_count"] == 20
    assert measurement["restart_readiness"]["warmup_prompt_tokens"] == 5
    assert measurement["restart_readiness"]["warmup_completion_tokens"] == 1
    assert measurement["cold_scheduler_inventory_control"]["safe_recompute_count"] == 1


def test_operator_scripts_pin_image_and_preserve_rollback_containers() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    rollback = ROLLBACK.read_text(encoding="utf-8")
    assert "sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818" in launcher
    assert "spark_cache_load_threads\":8" in launcher
    assert "spark_cache_max_pending_restores\":8" in launcher
    assert "spark_cache_cuda_placement_arena_bytes\":268435456" in launcher
    assert "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512" in launcher
    assert "VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512" in launcher
    assert "glm53-pr535-sc59ac-c8-01-r${rank}" in rollback
    assert "glm53-pr535-sc78-hotpatch-c8-qualified-r${rank}" in rollback
    assert "docker stop" in rollback and "docker start" in rollback


def test_record_has_required_evidence_sections_and_public_prose() -> None:
    text = RECORD.read_text(encoding="utf-8")
    for heading in (
        "## Conditions",
        "## Measurement",
        "## Result",
        "## Conclusion",
        "## Limitations",
    ):
        assert heading in text
    assert "All eight requests returned HTTP 200" in text
    assert "All eight requests restored external state" in text
    assert "HTTP `/health` was already successful before the tiny inference" in text
    assert re.search(r"(?i)\b[A-Z]:\\", text) is None
    assert re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", text) is None
