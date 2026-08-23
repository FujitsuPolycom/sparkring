"""GPU-free contracts for the Qwen3.8-27B four-Spark candidate profile."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes" / "qwen38-27b-exl3-k5k6.json"
ENV_PATH = ROOT / "scripts" / "config" / "qwen38-27b-exl3-k5k6.env.example"
QUICKSTART_PATH = ROOT / "docs" / "QWEN38_27B_EXL3_K5K6_QUICKSTART.md"
PROFILE_PATH = ROOT / "docs" / "profiles" / "QWEN38_27B_EXL3_K5K6.md"
LAUNCHER_PATH = ROOT / "scripts" / "qwen38_dgx4_serve.sh"
SMOKE_PATH = ROOT / "scripts" / "qwen38_smoke.py"
BUILDER_PATH = ROOT / "runtime" / "qwen38" / "build-image.sh"
BUILDER_PINS_PATH = ROOT / "runtime" / "qwen38" / "pins.json"
def _recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def test_recipe_declares_the_four_spark_candidate() -> None:
    recipe = _recipe()
    assert recipe["schema"] == "sparkring-recipe/v1"
    assert recipe["recipe_id"] == "qwen38-27b-exl3-k5k6"
    assert recipe["status"] == "candidate"
    assert recipe["hardware"] == {
        "platform": "linux/arm64",
        "cuda_arch": "sm_121",
        "ranks": 4,
        "topology": "direct-cycle-4",
    }

    serving = recipe["serving"]
    assert serving["tensor_parallel_size"] == 4
    assert serving["decode_context_parallel_size"] == 1
    assert serving["node_count"] == 4
    assert serving["distributed_executor_backend"] == "mp"
    assert serving["max_model_len"] == 1048576
    assert serving["context_extension"] == {
        "method": "static-yarn",
        "factor": 4.0,
        "original_max_position_embeddings": 262144,
        "rope_type": "yarn",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_interleaved": True,
        "mrope_section": [11, 11, 10],
    }
    assert serving["max_num_seqs"] == 64
    assert serving["max_num_batched_tokens"] == 8192
    assert serving["enable_chunked_prefill"] is True
    assert serving["async_scheduling"] is True
    assert serving["scheduler_reserve_full_isl"] is True
    assert serving["requested_attention_block_size"] == 16
    assert serving["effective_attention_block_size"] == 1600
    assert serving["effective_mamba_block_size"] == 1600
    assert serving["kv_cache_dtype"] == "fp8"
    assert serving["native_prefix_caching"] is True
    assert serving["mamba_cache_mode"] == "align"
    assert serving["external_kv_cache"] is False
    assert serving["speculation"] == {
        "method": "qwen3_5_mtp",
        "num_speculative_tokens": 3,
        "attention_backend": "TRITON_ATTN",
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
    }
    assert serving["sampling"] == {
        "benchmark_request": {"temperature": 1.0, "top_p": None, "top_k": None},
        "effective_model_defaults": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
        },
        "generation_config_sha256": (
            "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"
        ),
    }


def test_checkpoint_geometry_partitions_across_four_ranks() -> None:
    model = _recipe()["model"]
    assert model["revision"] == "ab3a91a13813df8096cb4c1d560ed3669035d0cf"
    assert model["config_sha256"] == (
        "fbb105334da6554c10784ff1257fda5e3821d4d5426d64469cee2b2ad67ba2b3"
    )
    assert model["weight_index_sha256"] == (
        "ea6e0e1064efbb72d89b4a6f9e0ee76c909a94b3f25047487a2ffb282896a26c"
    )
    assert model["weight_shards"] == 3
    assert model["weights_bytes"] == 21587265510
    geometry = model["geometry"]
    for key in (
        "hidden_size",
        "intermediate_size",
        "num_attention_heads",
        "num_key_value_heads",
        "linear_num_key_heads",
        "linear_num_value_heads",
        "vocab_size",
    ):
        assert geometry[key] % 4 == 0, key
    assert (geometry["hidden_size"] // 4) % 128 == 0
    assert (geometry["intermediate_size"] // 4) % 128 == 0
    assert (geometry["vocab_size"] // 4) % 128 == 0


def test_recipe_records_the_tp4_live_evidence_separately_from_the_pair() -> None:
    evidence = _recipe()["evidence"]
    assert {
        "status",
        "conditions",
        "measurement",
        "result",
        "conclusion",
        "limitations",
        "record",
    } <= set(evidence)
    assert evidence["status"] == "implemented"
    assert "four directly cabled NVIDIA DGX Sparks" in evidence["conditions"]
    assert "8382750 logical key-value tokens" in evidence["result"]
    assert "remains a candidate" in evidence["conclusion"]
    assert evidence["limitations"]


def test_runtime_records_the_public_qwen_image_builder() -> None:
    runtime = _recipe()["runtime"]
    builder_pins = json.loads(BUILDER_PINS_PATH.read_text(encoding="utf-8"))
    assert runtime["base_image"].endswith(
        "@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d"
    )
    assert runtime["recipe_repository"] == (
        "https://github.com/FujitsuPolycom/qwen38-spark-pair"
    )
    assert runtime["recipe_commit"] == (
        "b9e1031b80b6f3f64bfc75ae3922322f56954fd6"
    )
    assert runtime["requirements_freeze_sha256"] == (
        "d773c781bcc1de6cf81a64f9fa6b2ab80535f77eea08c5aeb5b96c2ce4423ba8"
    )
    assert runtime["engine_commit"] == (
        "229effc810ee6b8112f661472f6aace4eb8c787d"
    )
    assert runtime["exllamav3_commit"] == (
        "5f3c537ca9d89893d771256f5c43c93656553fbb"
    )
    assert runtime["environment_template"] == (
        "scripts/config/qwen38-27b-exl3-k5k6.env.example"
    )
    assert runtime["launcher"] == "scripts/qwen38_dgx4_serve.sh"
    assert runtime["image_status"] == (
        "public clean-checkout local image builder; no published image"
    )
    assert runtime["builder_status"].startswith("experimental; offline-validated")
    assert runtime["image_builder"] == "runtime/qwen38/build-image.sh"
    assert runtime["image_pins"] == "runtime/qwen38/pins.json"
    assert runtime["image_default_tag"] == "sparkring-qwen38:arm64-sm121"
    assert runtime["launcher_preflight_argument"] == "--check"
    assert runtime["launcher_run_argument"] == "--run"
    assert runtime["smoke_harness"] == "scripts/qwen38_smoke.py"
    assert BUILDER_PATH.is_file()
    assert SMOKE_PATH.is_file()
    assert builder_pins["sources"]["vllm"]["commit"] == runtime["engine_commit"]
    assert builder_pins["sources"]["exllamav3"]["patched_tree"] == (
        runtime["exllamav3_patched_tree"]
    )
    assert builder_pins["sources"]["nccl"]["patched_tree"] == (
        runtime["nccl_patched_tree"]
    )
    assert {
        "cuda-toolkit-13-2",
        "iproute2",
        "procps",
        "libibverbs1",
        "ibverbs-providers",
        "ibverbs-utils",
    } <= set(runtime["system_packages"])


def test_environment_is_the_patched_nccl_cycle_without_model_foreign_knobs() -> None:
    text = ENV_PATH.read_text(encoding="utf-8")
    required = (
        "LD_PRELOAD=/ws/nccl-patched/libnccl.so.2",
        "VLLM_NCCL_SO_PATH=/ws/nccl-patched/libnccl.so.2",
        "NCCL_SOCKET_IFNAME=<RENDEZVOUS_IFNAME>",
        "VLLM_HOST_IP=<RANK_RENDEZVOUS_IP>",
        "NCCL_IB_HCA=<NCCL_IB_HCA>",
        "NCCL_IB_GID_INDEX=<NCCL_IB_GID_INDEX>",
        "NCCL_IB_SUBNET_AWARE_ROUTING=1",
        "NCCL_ALGO=Ring",
        "NCCL_PROTO=LL,LL128,Simple",
        "NCCL_MIN_NCHANNELS=4",
        "NCCL_MAX_NCHANNELS=4",
        "NCCL_SKIP_TREE_CONNECT=1",
        "VLLM_EXL3_GRAPH_DECODE=1",
        "VLLM_EXL3_PREFILL_FP8=1",
        "VLLM_EXL3_PREFILL_RECONSTRUCT_M=256",
    )
    for value in required:
        assert value in text
    for forbidden in (
        "VLLM_DSPARK_",
        "VLLM_USE_B12X_MOE",
        "SPARK_CONTEXT_CACHE",
        "LMCACHE",
        "VLLM_SPARK_TP4_MODE",
    ):
        assert forbidden not in text


def test_quickstart_command_matches_the_recipe() -> None:
    text = LAUNCHER_PATH.read_text(encoding="utf-8")
    for value in (
        "--tensor-parallel-size 4",
        "--nnodes 4",
        "--distributed-executor-backend mp",
        "--max-model-len 1048576",
        "--max-num-seqs 64",
        "--max-num-batched-tokens 8192",
        "--enable-chunked-prefill",
        "--async-scheduling",
        "--scheduler-reserve-full-isl",
        "--block-size 16",
        "--kv-cache-dtype fp8",
        "--enable-prefix-caching",
        "--mamba-cache-mode align",
        '"method":"qwen3_5_mtp"',
        '"num_speculative_tokens":3',
        '"draft_sample_method":"probabilistic"',
        '"rejection_sample_method":"standard"',
        '"original_max_position_embeddings":262144',
        '"cudagraph_mode":"FULL_DECODE_ONLY"',
        '--mm-processor-kwargs \'{"truncation":false}\'',
        "--enable-auto-tool-choice",
    ):
        assert value in text
    assert "--kv-transfer-config" not in text
    assert "RANK0_RENDEZVOUS_ADDR" in text
    assert "RANK0_FABRIC_ADDR" not in text
    assert "pgrep -f '[v]llm serve'" in text
    assert "command -v pgrep" in text
    assert "229effc810ee6b8112f661472f6aace4eb8c787d" in text
    assert "594b01547b0d801cf95926ea973719354150893121019aba2ad8832bc9f17fdb" in text
    quickstart = QUICKSTART_PATH.read_text(encoding="utf-8")
    assert "scripts/qwen38_dgx4_serve.sh" in quickstart
    for value in (
        "runtime/qwen38/build-image.sh",
        "bash ./runtime/qwen38/build-image.sh",
        "docker save",
        "docker load",
        "--entrypoint /ws/venv/bin/hf",
        '"$IMAGE" --check',
        '"$IMAGE" --run',
        "scripts/qwen38_smoke.py",
        "ATTEMPT_ID must match [A-Za-z0-9_.-]+",
        "container-id-${ATTEMPT_ID}-r${RANK}",
        "{{.State.Running}}",
        "/ws/runtime/source-receipt.json",
        "org.sparkring.source-receipt-sha256",
    ):
        assert value in quickstart
    for shared_container_contract in (
        "--network host --ipc host --shm-size 16g --gpus all",
        "--ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband",
        '"$MODEL_DIR:/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated:ro"',
        '"$CACHE_DIR:/ws/cache"',
        '"$LOG_DIR:/ws/logs"',
        '"$ENV_FILE:/ws/rank.env:ro"',
        '-e RANK="$RANK"',
        '-e RANK0_RENDEZVOUS_ADDR="$RANK0_RENDEZVOUS_ADDR"',
        "--label org.sparkring.profile=qwen38-27b-exl3-k5k6",
        "--entrypoint /ws/qwen38_dgx4_serve.sh",
    ):
        assert quickstart.count(shared_container_contract) == 2


def test_profile_marks_sparkcache_pending_without_publishing_a_composition() -> None:
    profile = PROFILE_PATH.read_text(encoding="utf-8")
    compositions = ROOT / "recipes" / "sparkcache"
    assert "SparkCache" in profile
    assert "Pending" in profile
    assert not (compositions / "qwen38-27b-exl3-k5k6-tp4-dcp1.json").exists()
