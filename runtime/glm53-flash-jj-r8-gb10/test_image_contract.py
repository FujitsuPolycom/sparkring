from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = load_module("jj_r8_build_image", HERE / "build_image.py")
verify = load_module("jj_r8_verify_image", HERE / "verify_image.py")


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
        "commit": "506cc4a16581b5f62ae343cbd90cdd6bea13a6cd",
        "tree": "f0b2f3f35325b45bdc0acc9e13c9851229bdc9d7",
        "package_tree": "b4a2faa9566aaf77c1741021f175d4f53f041c8c",
        "source_tree_sha256": (
            "cfd6720b625232df64a36f37ab86e9c8ae353a1915304642d229ab28ea7bd794"
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
    assert defaults["prefill_schedule_interval"] == 8
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
        "506cc4a16581b5f62ae343cbd90cdd6bea13a6cd",
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3",
        "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5",
    ):
        assert identity in recipe
    assert "org.sparkcache.dcp-layouts=\"1,2,4\"" in recipe
    assert "org.sparkcache.cache-geometry=\"manager-pages-v2\"" in recipe
    assert "org.sparkcache.publication-schema=\"snapshot-v1\"" in recipe
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
    assert 'PREFILL_SCHEDULE_INTERVAL:=8' in launcher
    assert 'KV_CACHE_MEMORY_BYTES:=auto' in launcher
    assert 'B12X_MLA_CKV_GATHER_MAX_TOKENS:=524288' in launcher
    assert 'SPARKCACHE_MAX_SPAN_TOKENS:=1048576' in launcher
    assert 'SPARKCACHE_ASYNC_PAGE_CAPTURE:=0' in launcher
    assert 'SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES:=auto' in launcher
    assert 'SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT:=2' in launcher
    assert "spark_cache_async_page_capture_library" in launcher
    assert "vllm-manager-page-async-contract-55969c16.json" in launcher
    for value in (
        "MAX_MODEL_LEN=1048576",
        "MAX_NUM_BATCHED_TOKENS=8192",
        "PREFILL_SCHEDULE_INTERVAL=8",
        "KV_CACHE_MEMORY_BYTES='auto'",
        "B12X_MLA_CKV_GATHER_MAX_TOKENS=524288",
        "SPARKCACHE_MAX_SPAN_TOKENS=1048576",
        "SPARKCACHE_ASYNC_PAGE_CAPTURE=1",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES='auto'",
        "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT=2",
    ):
        assert value in environment
    for text in (launcher, environment):
        assert "jj-r8-gb10-manager-pages-v2" in text
    assert "IMAGE_REF" in launcher
    assert "IMAGE_ID" in launcher


def test_async_capture_image_receipt_binds_public_artifact_and_live_results() -> None:
    receipt = json.loads(
        (HERE / "async-capture-image-receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "qualified"
    assert receipt["artifact"]["registry"].endswith(
        "@sha256:368973d2e67241479ff49f7898f5026a2a44a37dad78b36f26afa1c6d9684e0e"
    )
    assert receipt["artifact"]["image_id"] == (
        "sha256:4664bcba054d2cf383d3d7940189e26aa32774e755583652a6e93c0058500029"
    )
    assert receipt["sources"]["sparkcache_commit"] == (
        "506cc4a16581b5f62ae343cbd90cdd6bea13a6cd"
    )
    assert receipt["conditions"]["capture_slot_bytes"] == 3 * 1024**3
    assert receipt["validation"]["live"]["prime_request_before_restore"] is False
    assert receipt["validation"]["live"]["900k_restore"]["needle"] == "passed"
    assert receipt["validation"]["live"]["1m_restore"]["needle"] == "passed"
