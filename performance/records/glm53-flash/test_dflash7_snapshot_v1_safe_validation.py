from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECEIPT_ROOT = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "dflash7-snapshot-v1-safe"
)


def _json(name: str) -> dict:
    return json.loads((RECEIPT_ROOT / name).read_text(encoding="utf-8"))


def _sha256(name: str) -> str:
    return hashlib.sha256((RECEIPT_ROOT / name).read_bytes()).hexdigest()


def test_snapshot_v1_receipt_binds_exact_artifact_and_safe_profile() -> None:
    receipt = _json("validation.json")
    assert receipt["schema"] == (
        "sparkring-glm53-dflash7-snapshot-v1-safe-validation/v1"
    )
    assert receipt["status"] == "qualified"
    assert receipt["artifact"]["image_id"] == (
        "sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e"
    )
    assert receipt["artifact"]["sparkcache_commit"] == (
        "a1511d26a1fe2b17b24561bc52e376bf7f54b06a"
    )
    assert receipt["profile"] == {
        **receipt["profile"],
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 1,
        "kv_cache_dtype": "fp8",
        "kv_cache_bytes_per_rank": 20 * 1024**3,
        "publication_schema": "snapshot-v1",
        "storage_mode": "block_pages_v1",
        "load_threads": 1,
        "max_pending_restores": 1,
        "cache_root_role": "isolated snapshot-v1 root",
    }


def test_snapshot_v1_inputs_and_all_rank_restore_are_consistent() -> None:
    receipt = _json("validation.json")
    published = _json("publish.json")
    restored = _json("restore.json")
    readiness = _json("readiness.json")
    manifests = _json("manifest-inspection.json")

    assert readiness["status"] == "verified"
    assert published["prompt_sha256"] == restored["prompt_sha256"]
    assert published["expected_oracle"] == published["observed_oracle"] == "red"
    assert restored["expected_oracle"] == restored["observed_oracle"] == "red"
    assert published["response_sha256"] == restored["response_sha256"]
    assert manifests["status"] == "verified"
    assert len(manifests["results"]) == 4
    assert {item["rank"] for item in manifests["results"]} == {0, 1, 2, 3}
    assert {item["context_digest"] for item in manifests["results"]} == {
        receipt["case"]["context_digest"]
    }
    assert {item["object_count"] for item in manifests["results"]} == {13}
    assert {item["snapshot_encoded_bytes"] for item in manifests["results"]} == {
        813068464
    }

    rank_evidence = receipt["rank_restore_evidence"]
    assert {item["rank"] for item in rank_evidence} == {0, 1, 2, 3}
    assert {item["outcome"] for item in rank_evidence} == {"verified"}
    assert {item["page_bytes"] for item in rank_evidence} == {813068464}
    elapsed = [item["end_to_end_ms"] for item in rank_evidence]
    assert min(elapsed) == 1552.485
    assert max(elapsed) == 1699.752

    repository_inputs = {
        "publish.json": "repository_publish_json",
        "restore.json": "repository_restore_json",
        "readiness.json": "repository_readiness_json",
        "manifest-inspection.json": "repository_manifest_inspection_json",
    }
    for name, field in repository_inputs.items():
        assert _sha256(name) == receipt["input_sha256"][field]


def test_snapshot_v1_receipt_does_not_qualify_research_paths() -> None:
    limitations = _json("validation.json")["limitations"]
    assert limitations["tail_cow_opaque_page_deltas"] == "research-only"
    assert limitations["host_base_read_coalescing"] == "research-only"
    assert limitations["multi_root_concurrent_restore"] == "research-only"
    assert limitations["c2_delta_restore"].startswith("not qualified")
