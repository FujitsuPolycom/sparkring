from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_module("jj_r8_build_image", HERE / "build_image.py")
verify = load_module("jj_r8_verify_image", HERE / "verify_image.py")
metrics_patch = load_module(
    "jj_r8_metrics_patch",
    HERE / "patch_kv_metrics_logging.py",
)


def test_metrics_patch_uses_connector_owned_compact_lines(tmp_path: Path) -> None:
    target = tmp_path / "metrics.py"
    target.write_text(
        "            xfer_metrics = self.transfer_stats_accumulator.reduce()\n"
        "            xfer_metrics_str = \", \".join(f\"{k}={v}\" for k, v in xfer_metrics.items())\n"
        "            log_fn(\"KV Transfer metrics: %s\", xfer_metrics_str)\n",
        encoding="utf-8",
    )

    metrics_patch.apply_patch(target)

    result = target.read_text(encoding="utf-8")
    assert "format_log_lines" in result
    assert 'log_fn("%s", line)' in result
    assert 'log_fn("KV Transfer metrics: %s", xfer_metrics_str)' in result


def test_image_packages_dflash_warmup_and_rank_zero_waits_for_it() -> None:
    recipe = (HERE / "Dockerfile").read_text(encoding="utf-8")
    builder = (HERE / "build_image.py").read_text(encoding="utf-8")
    launcher = (HERE / "launch-rank.sh").read_text(encoding="utf-8")

    assert "COPY warmup_dflash.py /opt/sparkring/bin/warmup-glm53-dflash.py" in recipe
    assert '"warmup_dflash.py"' in builder
    assert 'container_id="$(docker run -d' in launcher
    assert '"${rank}" == 0 && "${DFLASH_WARMUP}" == 1' in launcher
    assert "docker exec \"${container}\" python3" in launcher
    assert "/opt/sparkring/bin/warmup-glm53-dflash.py" in launcher


def test_pins_bind_effective_sources_and_operator_defaults() -> None:
    pins = json.loads((HERE / "pins.json").read_text(encoding="utf-8"))
    assert pins["vllm"]["commit"] == (
        "22ffe1401ca9bd3e4503e62de7b414deca7661a1"
    )
    assert pins["vllm"]["tree"] == "1bb7f10a5838d348ca2fcb0134b05ad768d3340b"
    assert pins["vllm"]["package_tree"] == (
        "e5ed1db6292c7312571419e101ba719bb5ebb393"
    )
    assert pins["vllm"]["delta_patch_id"] == (
        "44ad5586548465c002d0195d3739992795233ffe"
    )
    assert pins["vllm"]["community_release"] == "Jovian Judgement Community R10"
    assert pins["vllm"]["community_parent_commit"] == (
        "55969c16d4da57da76ee5729f3102d4b2003833c"
    )
    assert pins["vllm"]["sparse_pooled_index_commit"] == (
        "adb69eac865a1a37081fa4edb9f7599a351f7aac"
    )
    assert pins["vllm"]["fwht_scaling_commit"] == (
        "ae37fd6ed8df72d4bd8cdc067c7c241c93408235"
    )
    assert pins["vllm"]["scheduler_prefill_cadence_component_commit"] == (
        "f1191b9090cd02ac49238c8e4f371050759703b6"
    )
    assert pins["vllm"]["scheduler_prefill_cadence_pull_request_head"] == (
        "2412a6f34ab1412f86ed3e4cdd355271a082d93d"
    )
    assert pins["sparkcache"] == {
        "repository": "https://github.com/FujitsuPolycom/sparkcache.git",
        "commit": "737ed1399f559ba036fb0e358541744011afd47d",
        "tree": "decc0e042a4b1807e960551ffa3ef12c8c9114a7",
        "package_tree": "fef9ac1c59526dec49b0c3346cbf7bdb6f22a620",
        "source_tree_sha256": (
            "3cfb8d66db3a437a8b3a886633e64b7006af4c50cccb3ddbf75eb8d73eda5de6"
        ),
        "cuda_placement_sha256": (
            "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
        ),
        "cuda_snapshot_sha256": (
            "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5"
        ),
        "manager_page_lease_contract_sha256": (
            "480bdc463e3722cf28aa460021da689d12d9049ae9fd8238252fcf1db5544b53"
        ),
    }
    defaults = pins["defaults"]
    assert defaults["max_model_len"] == 1048576
    assert defaults["max_num_batched_tokens"] == 8192
    assert defaults["prefill_schedule_interval"] == 2
    assert defaults["shared_prefix_lease_ttl_seconds"] == 300
    assert defaults["kv_cache_bytes_per_rank"] == {
        "dcp1": 27917287424,
        "dcp2": 32212254720,
        "dcp4": 25769803776,
    }
    assert defaults["full_ckv_gather_max_tokens"] == 524288
    assert defaults["async_page_capture"] == {
        "preferred_dcp4_enabled": True,
        "launcher_fallback_enabled": False,
        "slot_bytes_by_dcp": {
            "dcp1": 8589934592,
            "dcp2": 5368709120,
            "dcp4": 3221225472,
        },
        "slot_count": 2,
    }
    assert defaults["dcp"] == {
        "1": {"cp_kv_cache_interleave_size": 1, "full_ckv_gather": False},
        "2": {"cp_kv_cache_interleave_size": 4, "full_ckv_gather": True},
        "4": {"cp_kv_cache_interleave_size": 4, "full_ckv_gather": True},
    }


def test_dockerfile_preserves_native_components_and_binds_overlays() -> None:
    recipe = (HERE / "Dockerfile").read_text(encoding="utf-8")
    for identity in (
        "f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5",
        "22ffe1401ca9bd3e4503e62de7b414deca7661a1",
        "737ed1399f559ba036fb0e358541744011afd47d",
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3",
        "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5",
    ):
        assert identity in recipe
    assert "org.sparkcache.dcp-layouts=\"1,2,4\"" in recipe
    assert "org.sparkcache.cache-geometry=\"manager-pages-v2\"" in recipe
    assert (
        "org.sparkcache.publication-schema="
        '\"snapshot-v1,page-tail-cow-v1,page-tail-cow-v2\"'
        in recipe
    )
    assert "compact-startup-no-deep-ep" in recipe
    assert "fastsafetensors" in recipe


def test_parent_contract_binds_proven_arm64_image() -> None:
    pins = build.load_pins()
    document = {
        "Architecture": "arm64",
        "Os": "linux",
        "Id": pins["parent"]["image_id"],
        "Config": {"Labels": dict(pins["parent"]["required_labels"])},
    }
    build.validate_parent(document, pins)
    document["Config"]["Labels"]["org.sparkring.nccl.sha256"] = "0" * 64
    try:
        build.validate_parent(document, pins)
    except build.BuildError as error:
        assert "nccl.sha256" in str(error)
    else:
        raise AssertionError("transport drift was accepted")


def test_runtime_hashes_are_enforced_inside_image() -> None:
    pins = build.load_pins()
    verifier = (HERE / "verify_image.py").read_text(encoding="utf-8")
    assert 'pins["vllm"]["runtime_file_sha256"]' in verifier
    assert set(pins["vllm"]["runtime_file_sha256"]) == {
        "vllm/config/scheduler.py",
        "vllm/v1/core/sched/interface.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/engine/core.py",
    }
    labels = verify.expected_labels(pins)
    assert labels["org.sparkring.vllm.delta-patch-id"] == (
        "44ad5586548465c002d0195d3739992795233ffe"
    )
    assert labels["org.sparkring.vllm.community-parent"] == (
        "55969c16d4da57da76ee5729f3102d4b2003833c"
    )
    assert labels["org.sparkring.b12x.package-tree"] == (
        "6de9871d15dab093340695518fec0f744289e676"
    )


def test_launcher_keeps_gather_workspace_below_native_context_limit() -> None:
    launcher = (HERE / "launch-rank.sh").read_text(encoding="utf-8")
    environment = (HERE / "runtime.env.example").read_text(encoding="utf-8")
    assert 'MAX_MODEL_LEN:=1048576' in launcher
    assert 'MAX_NUM_BATCHED_TOKENS:=8192' in launcher
    assert 'PREFILL_SCHEDULE_INTERVAL:=2' in launcher
    assert 'KV_CACHE_MEMORY_BYTES:=auto' in launcher
    assert 'B12X_MLA_CKV_GATHER_MAX_TOKENS:=524288' in launcher
    assert 'SPARKCACHE_MAX_SPAN_TOKENS:=1048576' in launcher
    assert 'SPARKCACHE_ASYNC_PAGE_CAPTURE:=0' in launcher
    assert 'SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES:=auto' in launcher
    assert 'SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT:=2' in launcher
    assert 'SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS:=300' in launcher
    assert "spark_cache_async_page_capture_library" in launcher
    assert "vllm-manager-page-async-contract-55969c16.json" in launcher
    for value in (
        "MAX_MODEL_LEN=1048576",
        "MAX_NUM_BATCHED_TOKENS=8192",
        "PREFILL_SCHEDULE_INTERVAL=2",
        "MAX_IMAGES_PER_PROMPT=4",
        "MAX_VIDEOS_PER_PROMPT=1",
        "KV_CACHE_MEMORY_BYTES='auto'",
        "B12X_MLA_CKV_GATHER_MAX_TOKENS=524288",
        "SPARKCACHE_MAX_SPAN_TOKENS=1048576",
        "SPARKCACHE_ASYNC_PAGE_CAPTURE=1",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES='auto'",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT=2",
        "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS=300",
    ):
        assert value in environment
    for text in (launcher, environment):
        assert "glm53-flash-dcp4-snapshot-v1" in text
    assert "IMAGE_REF" in launcher
    assert "IMAGE_ID" in launcher


def test_operator_docs_distinguish_page_tails_from_published_rollback() -> None:
    runtime_readme = (HERE / "README.md").read_text(encoding="utf-8")
    quickstart = (
        ROOT / "docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md"
    ).read_text(encoding="utf-8")
    runtime_index = (ROOT / "runtime/README.md").read_text(encoding="utf-8")

    for document in (runtime_readme, quickstart):
        for schema in ("snapshot-v1", "tail-cow-v1", "tail-cow-v2"):
            assert schema in document
        assert "sparkring-glm53-sparkcache:page-tail-v2-local" in document
        assert "no published registry digest" in document
        assert "sparkcache: capacity" in document
        assert "sparkcache: publications" in document
        assert "sparkcache: writes" in document

    published_digest = (
        "3c377f1e4136285ebf66c32c36c3d01f"
        "d929f8aba0836cd0a16ed63cfd7e1762"
    )
    assert published_digest in runtime_readme
    assert published_digest in quickstart
    assert published_digest in runtime_index
    assert "380283a506aeb8f9" not in runtime_index


def test_multimodal_lease_image_receipt_binds_public_artifact_and_smoke() -> None:
    receipt = json.loads(
        (HERE / "multimodal-lease300-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["status"] == "live-smoke-verified"
    assert receipt["artifact"] == {
        "registry": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@"
            "sha256:3c377f1e4136285ebf66c32c36c3d01fd929f8aba0836cd0a16ed63cfd7e1762"
        ),
        "published_tag": (
            "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:"
            "20260902-r10-multimodal-lease300"
        ),
        "image_id": (
            "sha256:d1a07147c9e25f3d3e0af6b1499c4988b1ae61138e327aa05c9ad9dc568e39a9"
        ),
        "platform": "linux/arm64",
        "archive_sha256": (
            "a88cd040bb38ed0092d8de8fc00aa2ac7e15a4352258a73394fb876fea3756a4"
        ),
        "archive_bytes": 8469866878,
    }
    assert receipt["sources"]["sparkcache_commit"] == (
        "b7d1c188a3f9e78595e6e7b649f3751131e269ea"
    )
    assert receipt["launch"]["shared_prefix_lease_ttl_seconds"] == 300
    assert receipt["launch"]["max_images_per_prompt"] == 4
    assert receipt["launch"]["max_videos_per_prompt"] == 1
    assert receipt["verification"]["physical_ranks"] == 4
    assert receipt["verification"]["running_image_ids_equal"] is True
    assert receipt["verification"]["sparkcache_source_bind_mounts"] == 0
    assert receipt["verification"]["image_probe"] == {
        "input": "448x448 solid-red PNG",
        "multimodal_image_tokens": 256,
        "dominant_color_identified": "red",
    }


def test_page_tail_v2_receipt_binds_the_local_image() -> None:
    receipt = json.loads(
        (HERE / "page-tail-v2-local-image-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert receipt["schema"] == "sparkring-glm53-jj-r8-gb10-image-receipt/v1"
    assert receipt["status"] == "implemented"
    assert receipt["platform"] == "linux/arm64"
    assert receipt["image_id"] == (
        "sha256:7df364ed1bb0036d2514e36d5e40cfa1721c7fb9d841b0d9c4b519b53f5680c8"
    )
    assert receipt["inside_image"]["sparkcache_commit"] == (
        "fb03fd4f007f492608ebef01954365627ab2a2d6"
    )
    assert receipt["labels"]["org.sparkcache.publication-schema"] == (
        "snapshot-v1,page-tail-cow-v1,page-tail-cow-v2"
    )


def test_async_capture_image_receipt_binds_public_artifact_and_live_results() -> None:
    receipt = json.loads(
        (HERE / "async-capture-image-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "qualified"
    assert receipt["artifact"]["registry"].endswith(
        "@sha256:bc7d079f16ff4a418669c58c5250f2da52e989a0c5805569ba9429d41b765f65"
    )
    assert receipt["artifact"]["image_id"] == (
        "sha256:35f397668c01075d0bdd28bbdb3398afd3744df6086646c6f68bcf7ebe7f918f"
    )
    assert receipt["artifact"]["published_tag"].endswith(
        ":20260901-r10-async-telemetry"
    )
    assert receipt["sources"]["sparkring_commit"] == (
        "d2f8911427d64bbb89c275814777fc3f8112fd21"
    )
    assert receipt["sources"]["sparkcache_commit"] == (
        "c5dda75ec46bf235f6ece6e0d0174c1e41bd805a"
    )
    assert receipt["conditions"]["capture_slot_bytes"] == 3 * 1024**3
    assert receipt["conditions"]["operator_template_storage_root"] == (
        "glm53-flash-dcp4-snapshot-v1"
    )
    assert receipt["conditions"]["live_validation_storage_root"] == (
        "jj-r10-async-ab-v1"
    )
    assert receipt["conditions"]["operator_template_storage_root_measured"] is False
    assert receipt["deep_entry_writer"] == {
        "image_id": (
            "sha256:8e586e6ad9b4f30a8ccef1bfd8b76194524e156089c958907872d0f8735a09b2"
        ),
        "archive_sha256": (
            "47c800fd73130c1fe26b707caa2c64f81ed43c951fe2019d8836cd0b883dbe48"
        ),
        "sparkcache_commit": "6d83c7d8cb6ace96e657b3d0150116d0fe4e011c",
        "sparkcache_tree": "0bb871bd1e8d3893a11686f0ba404bd4b6240e4d",
        "sparkcache_source_sha256": (
            "67edb651835b978cbaf2519f92e68251145c1368a22cc0339f706d5c2144f862"
        ),
        "vllm_commit": "22ffe1401ca9bd3e4503e62de7b414deca7661a1",
        "vllm_tree": "1bb7f10a5838d348ca2fcb0134b05ad768d3340b",
        "cuda_snapshot_sha256": (
            "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5"
        ),
        "topology": "TP4/DCP4",
        "publication_schema": "snapshot-v1",
        "storage_root": "jj-r10-async-ab-v1",
        "checkpoint_identity": (
            "same target and draft repositories and revisions recorded in conditions"
        ),
    }
    assert (
        receipt["validation"]["live"]["identical_prompt_prime_before_restore"]
        is False
    )
    assert receipt["validation"]["live"]["startup_inventory"] == {
        "checked": 29,
        "offered": 29,
        "rejected": 0,
    }
    assert receipt["validation"]["live"]["cold_126k_publication"][
        "capture_completion_observed_ms_by_rank"
    ] == [408.5, 404.3, 403.7, 404.3]
    assert receipt["validation"]["live"]["900k_restore"]["needle"] == "passed"
    assert receipt["validation"]["live"]["1m_restore"]["needle"] == "passed"
