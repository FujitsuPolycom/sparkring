"""GPU-free contracts for the Qwen3.8-27B two-Spark research profile."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "qwen38-27b-exl3-k5k6-pair.json"
CYCLE_RECIPE = ROOT / "recipes" / "qwen38-27b-exl3-k5k6.json"
ENV = ROOT / "scripts" / "config" / "qwen38-27b-exl3-k5k6-pair.env.example"
LAUNCHER = ROOT / "scripts" / "qwen38_dgx2_serve.sh"
QUICKSTART = ROOT / "docs" / "QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md"
PROFILE = ROOT / "docs" / "profiles" / "QWEN38_27B_EXL3_K5K6_PAIR.md"
EVIDENCE = (
    ROOT
    / "performance"
    / "records"
    / "qwen38-27b"
    / "dgx2-1m-probmtp-20260823.json"
)


def _recipe() -> dict:
    return json.loads(RECIPE.read_text(encoding="utf-8"))


def test_pair_and_cycle_share_the_normalized_serving_contract() -> None:
    pair = _recipe()["serving"]
    cycle = json.loads(CYCLE_RECIPE.read_text(encoding="utf-8"))["serving"]
    topology_specific = {
        "tensor_parallel_size",
        "node_count",
        "max_num_seqs",
        "sizing_note",
    }
    assert {key: value for key, value in pair.items() if key not in topology_specific} == {
        key: value for key, value in cycle.items() if key not in topology_specific
    }


def test_recipe_declares_the_pair_research_contract() -> None:
    recipe = _recipe()
    assert recipe["schema"] == "sparkring-recipe/v1"
    assert recipe["recipe_id"] == "qwen38-27b-exl3-k5k6-pair"
    assert recipe["status"] == "research-only"
    assert recipe["hardware"] == {
        "platform": "linux/arm64",
        "cuda_arch": "sm_121",
        "ranks": 2,
        "topology": "direct-pair-2",
    }
    serving = recipe["serving"]
    assert serving["tensor_parallel_size"] == 2
    assert serving["decode_context_parallel_size"] == 1
    assert serving["distributed_executor_backend"] == "mp"
    assert serving["max_model_len"] == 1_048_576
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
    assert serving["max_num_seqs"] == 32
    assert serving["max_num_batched_tokens"] == 8192
    assert serving["gpu_memory_utilization"] == 0.7
    assert serving["kv_cache_dtype"] == "fp8"
    assert serving["external_kv_cache"] is False
    assert serving["reasoning_parser"] == "qwen3"
    assert serving["tool_call_parser"] == "qwen3_coder"
    assert serving["auto_tool_choice"] is True
    assert serving["multimodal_processor_kwargs"] == {"truncation": False}
    assert serving["exl3_prefill_fp8"] is True
    assert serving["exl3_prefill_reconstruct_m"] == 256
    assert serving["speculation"] == {
        "method": "qwen3_5_mtp",
        "num_speculative_tokens": 3,
        "attention_backend": "TRITON_ATTN",
        "draft_sample_method": "probabilistic",
        "rejection_sample_method": "standard",
    }


def test_pair_environment_is_single_hca_and_cache_free() -> None:
    text = ENV.read_text(encoding="utf-8")
    for required in (
        "NCCL_SOCKET_IFNAME=<FABRIC_IFNAME>",
        "VLLM_HOST_IP=<RANK_FABRIC_IP>",
        "NCCL_IB_HCA=<NCCL_IB_HCA>",
        "NCCL_IB_GID_INDEX=<GID_INDEX>",
        "NCCL_IB_SUBNET_AWARE_ROUTING=0",
        "NCCL_IB_MERGE_NICS=0",
        "NCCL_PROTO=LL,LL128,Simple",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN=1",
        "VLLM_EXL3_PREFILL_RECONSTRUCT_M=256",
    ):
        assert required in text
    for forbidden in (
        "NCCL_ALGO=",
        "NCCL_MIN_NCHANNELS",
        "NCCL_MAX_NCHANNELS",
        "NCCL_SKIP_TREE_CONNECT",
        "LMCACHE",
        "SPARK_CONTEXT_CACHE",
    ):
        assert forbidden not in text


def test_pair_launcher_matches_recipe_and_fails_closed() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "RANK must be 0 or 1",
        "--tensor-parallel-size 2",
        "--decode-context-parallel-size 1",
        "--nnodes 2",
        "--distributed-executor-backend mp",
        "--max-model-len 1048576",
        "--max-num-seqs 32",
        "--max-num-batched-tokens 8192",
        "--gpu-memory-utilization 0.70",
        "--kv-cache-dtype fp8",
        '"draft_sample_method":"probabilistic"',
        '"rejection_sample_method":"standard"',
        '"factor":4.0',
        "verify_runtime.py --imports",
        "sha256sum --check --strict --status SHA256SUMS",
        "libibverbs.so.1",
        "torch.cuda.device_count()",
        "torch.cuda.get_device_capability(0)",
        "no infiniband uverbs device is available",
        "rank-0 rendezvous address must equal rank-0 VLLM_HOST_IP",
        "cycle-only transport value must be unset on a pair",
        "qwen38 pair preflight passed",
    ):
        assert required in text
    assert "--kv-transfer-config" not in text


def test_pair_launcher_rejects_an_invalid_rank_before_touching_runtime() -> None:
    result = subprocess.run(
        [
            "bash",
            "-c",
            "export RANK=2; bash scripts/qwen38_dgx2_serve.sh --check",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert result.returncode == 20
    assert "RANK must be 0 or 1; got 2" in result.stderr


def _run_early_pair_preflight(
    tmp_path: Path,
    *,
    overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    values = {
        "LD_PRELOAD": "/ws/nccl-patched/libnccl.so.2",
        "VLLM_NCCL_SO_PATH": "/ws/nccl-patched/libnccl.so.2",
        "NCCL_SOCKET_IFNAME": "lo",
        "GLOO_SOCKET_IFNAME": "lo",
        "VLLM_HOST_IP": "127.0.0.1",
        "NCCL_NET": "IB",
        "NCCL_NET_PLUGIN": "none",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_HCA": "fakehca",
        "NCCL_IB_GID_INDEX": "3",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "0",
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_PROTO": "LL,LL128,Simple",
        "NCCL_P2P_LEVEL": "SYS",
        "NCCL_CROSS_NIC": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "VLLM_EXL3_GRAPH_DECODE": "1",
        "VLLM_EXL3_PREFILL_FP8": "1",
        "VLLM_EXL3_PREFILL_RECONSTRUCT_M": "256",
        "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
    }
    values.update(overrides or {})
    env_file = tmp_path / "rank.env"
    env_file.write_text(
        "\n".join(f"{key}={value}" for key, value in values.items()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    def shell_path(path: Path) -> str:
        if os.name != "nt":
            return str(path)
        resolved = path.resolve()
        drive = resolved.drive.rstrip(":").lower()
        relative = resolved.as_posix()[2:].lstrip("/")
        return f"/mnt/{drive}/{relative}"

    assignments = {
        "RANK": "0",
        "RANK0_RENDEZVOUS_ADDR": "127.0.0.1",
        "QWEN_ENV_FILE": shell_path(env_file),
        "QWEN_INFINIBAND_SYS_ROOT": shell_path(tmp_path / "infiniband"),
        "QWEN_INFINIBAND_DEV_ROOT": shell_path(tmp_path / "dev-infiniband"),
    }
    exports = "; ".join(
        f"export {key}={shlex.quote(value)}" for key, value in assignments.items()
    )
    return subprocess.run(
        ["bash", "-c", f"{exports}; bash scripts/qwen38_dgx2_serve.sh --check"],
        check=False,
        capture_output=True,
        text=True,
        cwd=ROOT,
    )


def test_pair_launcher_rejects_a_preload_that_omits_patched_nccl(
    tmp_path: Path,
) -> None:
    result = _run_early_pair_preflight(
        tmp_path,
        overrides={"LD_PRELOAD": "/tmp/not-nccl.so"},
    )
    assert result.returncode == 20
    assert "LD_PRELOAD must include VLLM_NCCL_SO_PATH" in result.stderr


def test_pair_launcher_rejects_inherited_cycle_transport(
    tmp_path: Path,
) -> None:
    result = _run_early_pair_preflight(
        tmp_path,
        overrides={"NCCL_ALGO": "Ring"},
    )
    assert result.returncode == 20
    assert "cycle-only transport value must be unset" in result.stderr


def test_pair_launcher_rejects_an_empty_gid_before_runtime_checks(
    tmp_path: Path,
) -> None:
    gid_root = tmp_path / "infiniband" / "fakehca" / "ports" / "1"
    (gid_root / "gids").mkdir(parents=True)
    (gid_root / "gid_attrs" / "types").mkdir(parents=True)
    (gid_root / "gid_attrs" / "ndevs").mkdir(parents=True)
    (gid_root / "gids" / "3").write_text(
        "0000:0000:0000:0000:0000:0000:0000:0000\n",
        encoding="utf-8",
        newline="\n",
    )
    (gid_root / "gid_attrs" / "types" / "3").write_text(
        "RoCE v2\n",
        encoding="utf-8",
        newline="\n",
    )
    (gid_root / "gid_attrs" / "ndevs" / "3").write_text(
        "lo\n",
        encoding="utf-8",
        newline="\n",
    )
    result = _run_early_pair_preflight(tmp_path)
    assert result.returncode == 20
    assert "selected GID entry is empty" in result.stderr


def test_quickstart_separates_normalized_and_shared_prefix_benchmarks() -> None:
    text = QUICKSTART.read_text(encoding="utf-8")
    for required in (
        "runtime/qwen38/build-image.sh",
        "scripts/config/qwen38-27b-exl3-k5k6-pair.env.example",
        "/ws/qwen38_dgx2_serve.sh",
        "--expected-max-model-len 1048576",
        "temperature 1",
        "100% unique",
        "defer admission to the server",
        "top-p 0.95",
        "top-k 20",
    ):
        assert required in text
    assert "shared-prefix" in text
    assert "No 1M prompt" in text


def test_evidence_is_research_only_and_publishes_no_throughput() -> None:
    evidence = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert evidence["status"] == "research-only"
    assert evidence["startup"]["kv_tokens"] == 4_093_750
    assert evidence["startup"]["maximum_full_context_concurrency"] == 4.09
    assert evidence["probabilistic_mtp_probe"]["draft_acceptance_percent"] == 67.5
    assert "benchmark" not in evidence
    assert "throughput" not in evidence
    assert PROFILE.is_file()


def test_pair_sparkcache_composition_is_not_published() -> None:
    composition = (
        ROOT
        / "recipes"
        / "sparkcache"
        / "qwen38-27b-exl3-k5k6-tp2-dcp1.json"
    )
    assert not composition.exists()
