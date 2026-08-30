"""Validate the bounded GLM-5.3 DFlash7 live-evidence record."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
RECEIPT_PATH = (
    ROOT
    / "performance"
    / "receipts"
    / "glm53-flash"
    / "dflash7-python-overlay-pr25"
    / "validation.json"
)
RECORD_PATH = (
    ROOT
    / "performance"
    / "records"
    / "glm53-flash"
    / "dflash7-python-overlay-pr25-live-validation.md"
)
def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))


def test_receipt_binds_exact_artifact_and_scoped_status() -> None:
    receipt = _receipt()
    assert receipt["schema"] == "sparkring-glm53-dflash7-pr25-live-validation/v1"
    assert receipt["artifact"] == {
        "image": "sparkring-glm53-sparkcache:dflash7-vllm-python-0b67266-native-da4d7be-b12x-b1d541f-arm64",
        "image_id": "sha256:9faa36a9f37aee16d97ab9214ef3153b4d200121126e6b2dee5ebb63109fea18",
        "oci_revision_label": "e2d92fdc7d0306d664d6fd9f296dc2adcaf0fe05",
        "sparkring_revision": "e2d92fdc7d0306d664d6fd9f296dc2adcaf0fe05",
        "vllm_compiled_revision": "da4d7be6c97434f6942292ed8abbf4b32dc44355",
        "vllm_python_revision": "0b67266a0f37d6146a8403fb8482403c62f412d5",
        "b12x_revision": "b1d541f9e71a35f030d45fae437630fff7507c2a",
        "sparkcache_revision": "5d571018de5b63a9a90e5c11e6d6e86bbff4a957",
        "sparkcache_tree": "e864ed9ad64f771188fdb59aa9738e348134d636",
        "sparkcache_source_sha256": "f7c0565521fddeff7085e4cc08043cb8d1e2bde33abc67f83b8608a162d05b88",
        "cuda_placement_library_sha256": "a2e495162bf3d58b01613cd82ac15c8e15031dd7d6de7299700d2c58d905ada8",
        "dflash_loader_patch_sha256": "39b567013ee7aed79f63200ed460129587933dc77fb430decdf19f78178de279",
        "dflash_loader_postimage_sha256": "98acbae2b3bb4482d83f9637c163ce7c92707ccdf6561b7e431f23337f151cf4",
    }
    assert receipt["status"] == {
        "artifact_construction": "implemented",
        "startup_health": "qualified",
        "semantic_smoke": "qualified",
        "flat_restore": "qualified",
        "page_delta_restore_correctness": "qualified",
        "page_delta_restore_performance": "research-only",
        "persistent_128k_restore": "qualified",
        "shared_prefix_waves": "qualified",
        "dflash_quality": "unsupported",
    }
    assert receipt["conditions"]["sparkcache_config_surface"] == (
        "PR25 legacy-key compatibility"
    )
    assert receipt["conditions"]["sparkcache_accepted_legacy_keys"] == [
        "spark_cache_native_restore",
        "spark_cache_native_library",
        "spark_cache_native_library_sha256",
        "spark_cache_native_arena_bytes",
        "spark_cache_native_io_workers",
    ]


def test_receipt_preserves_startup_and_restore_observations() -> None:
    measurements = _receipt()["measurements"]
    assert measurements["startup"] == {
        "target_fastsafetensors_seconds": 69.52,
        "draft_safetensors_seconds": 4.18,
        "total_model_load_seconds": 79.27,
        "warm_ready_seconds_approx": 190.0,
        "healthy_ranks": 4,
        "restart_count_by_rank": [0, 0, 0, 0],
        "oom_killed_by_rank": [False, False, False, False],
    }
    assert measurements["semantic_smoke"]["raw_completion_tokens"] == [2, 2]
    assert measurements["flat_restore"] == {
        "restored_tokens": 11520,
        "rank": 0,
        "restore_milliseconds": 29.258,
    }
    assert measurements["page_delta_restore"] == {
        "restored_tokens": 17152,
        "restore_milliseconds": 703.826,
        "restore_read_milliseconds": 579.259,
        "chunk_count": 67,
        "type_error_observed": False,
    }
    assert measurements["persistent_restore_128k"] == {
        "restored_tokens": 131072,
        "bytes_per_rank": 813068464,
        "rank_restore_milliseconds": {"minimum": 123.69, "maximum": 153.253},
        "logical_tokens_per_second_at_slowest_rank": 855265.476,
    }


def test_each_retained_prefix_wave_records_one_restore() -> None:
    waves = _receipt()["measurements"]["shared_prefix_waves"]
    assert [(wave["concurrency"], wave["wall_seconds"]) for wave in waves] == [
        (2, 0.808),
        (8, 1.506),
        (16, 1.781),
    ]
    for wave in waves:
        assert wave["http_200"] == wave["concurrency"]
        assert wave["completion_token"] == 13
        assert wave["restore_events_observed"] == 1
        assert wave["retention_observed"] is True


def test_receipt_retains_required_limitations() -> None:
    limitations = "\n".join(_receipt()["limitations"])
    for required in (
        "7168-to-12032",
        "SparkCache PR28",
        "6912 tokens",
        "too high for a performance claim",
        "No DFlash quality benchmark",
    ):
        assert required in limitations


def test_record_follows_evidence_method_and_links_receipt() -> None:
    record = RECORD_PATH.read_text(encoding="utf-8")
    for heading in (
        "## Conditions",
        "## Measurement",
        "## Result",
        "## Conclusion",
        "## Limitations",
    ):
        assert heading in record
    assert (
        "../../receipts/glm53-flash/dflash7-python-overlay-pr25/validation.json"
        in record
    )
    assert "one restore\nevent followed by retained-prefix followers" in record


def test_receipt_does_not_transfer_qualification_to_other_images() -> None:
    limitations = "\n".join(_receipt()["limitations"])
    assert "qualifies only image sha256:9faa36" in limitations
    assert "sha256:eef863d8" in limitations
    assert "sha256:ed60be06" in limitations
    record = RECORD_PATH.read_text(encoding="utf-8")
    assert "This record applies only to image" in record
    assert "sha256:9faa36a9" in record
    assert "sha256:eef863d8" in record
    assert "sha256:ed60be06" in record
