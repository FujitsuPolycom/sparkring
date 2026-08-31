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
        "55969c16d4da57da76ee5729f3102d4b2003833c"
    )
    assert pins["vllm"]["tree"] == "a8d44216d05cbcd4df25f2c269b807275ec2e4ea"
    assert pins["vllm"]["package_tree"] == (
        "2255ebbbd63c8fe2347e9c742d565f44f4bf2e3d"
    )
    assert pins["vllm"]["delta_patch_id"] == (
        "171dfb66935731a4944335ce0e74307905ee903d"
    )
    assert pins["vllm"]["official_r8_component_commit"] == (
        "f1191b9090cd02ac49238c8e4f371050759703b6"
    )
    assert pins["vllm"]["public_prefill_cadence_pr_head"] == (
        "2412a6f34ab1412f86ed3e4cdd355271a082d93d"
    )
    assert pins["sparkcache"] == {
        "repository": "https://github.com/FujitsuPolycom/sparkcache.git",
        "commit": "65895c87a6d925fcd270fcea202ab847d7fbc2d1",
        "tree": "80daebdeedae62e444a1bfc71a102848660a04f0",
        "package_tree": "2636d7eb0de33e82a10bae8322193a9a392070d5",
        "source_tree_sha256": (
            "40de372dda64dd25f493584b2ba3dae81c4350d424d3cf00cfea92452dac170c"
        ),
        "cuda_placement_sha256": (
            "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c"
        ),
    }
    defaults = pins["defaults"]
    assert defaults["max_model_len"] == 1048576
    assert defaults["max_num_batched_tokens"] == 8192
    assert defaults["prefill_schedule_interval"] == 8
    assert defaults["kv_cache_bytes_per_rank"] == 32212254720
    assert defaults["full_ckv_gather_max_tokens"] == 524288
    assert defaults["dcp"] == {
        "1": {"cp_kv_cache_interleave_size": 1, "full_ckv_gather": False},
        "2": {"cp_kv_cache_interleave_size": 4, "full_ckv_gather": True},
        "4": {"cp_kv_cache_interleave_size": 4, "full_ckv_gather": True},
    }


def test_dockerfile_preserves_native_components_and_binds_overlays() -> None:
    recipe = (HERE / "Dockerfile").read_text(encoding="utf-8")
    for identity in (
        "f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5",
        "55969c16d4da57da76ee5729f3102d4b2003833c",
        "65895c87a6d925fcd270fcea202ab847d7fbc2d1",
        "5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3",
        "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
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
        "171dfb66935731a4944335ce0e74307905ee903d"
    )
    assert labels["org.sparkring.b12x.package-tree"] == (
        "6de9871d15dab093340695518fec0f744289e676"
    )


def test_launcher_keeps_gather_workspace_below_native_context_limit() -> None:
    launcher = (
        HERE.parent / "glm53-flash-jj-r7-gb10/launch-rank.sh"
    ).read_text(encoding="utf-8")
    environment = (
        HERE.parent / "glm53-flash-jj-r7-gb10/runtime.env.example"
    ).read_text(encoding="utf-8")
    assert 'MAX_MODEL_LEN:=1048576' in launcher
    assert 'MAX_NUM_BATCHED_TOKENS:=8192' in launcher
    assert 'PREFILL_SCHEDULE_INTERVAL:=8' in launcher
    assert 'KV_CACHE_MEMORY_BYTES:=32212254720' in launcher
    assert 'B12X_MLA_CKV_GATHER_MAX_TOKENS:=524288' in launcher
    assert 'SPARKCACHE_MAX_SPAN_TOKENS:=1048576' in launcher
    for value in (
        "MAX_MODEL_LEN=1048576",
        "MAX_NUM_BATCHED_TOKENS=8192",
        "PREFILL_SCHEDULE_INTERVAL=8",
        "KV_CACHE_MEMORY_BYTES=32212254720",
        "B12X_MLA_CKV_GATHER_MAX_TOKENS=524288",
        "SPARKCACHE_MAX_SPAN_TOKENS=1048576",
    ):
        assert value in environment
    for text in (launcher, environment):
        assert "jj-r8-gb10-manager-pages-v2" in text
    assert "SPARKCACHE_DCP_IMAGE_REF" in launcher
    assert "SPARKCACHE_DCP_IMAGE_ID" in launcher
