#!/usr/bin/env python3
"""Fail-closed four-rank launcher for the public-functional SparkRing lane.

Dry-run planning is the default. Remote mutation requires both a mutating
subcommand and ``--execute``.
"""


from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess  # noqa: F401 — kept for test monkeypatch compatibility
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import SiteConfig, SiteConfigError, load_site  # noqa: E402

SCHEMA = "sparkring-public-launch/v1"
_GLM52_INDEX_TOPK_PATTERN = (
    "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
    "FSSSFSSSFSSS"
)
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")
_ABS = re.compile(r"^/[A-Za-z0-9._/+@:-]*[A-Za-z0-9._+@:-]$")
_ENV = re.compile(r"^[A-Z_][A-Z0-9_]*$")
_PLACEHOLDER = re.compile(r"{([a-z0-9_]+)}")
_ALLOWED_PLACEHOLDERS = {
    "api_port",
    "draft_path",
    "master_addr",
    "master_port",
    "model_path",
    "peer0_addr",
    "peer0_device",
    "peer0_gid",
    "peer0_rank",
    "peer1_addr",
    "peer1_device",
    "peer1_gid",
    "peer1_rank",
    "rank",
    "world_size",
}
_SITE_DERIVED_ENVIRONMENT = {
    "GLOO_SOCKET_IFNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_IB_GID_INDEX",
    "NCCL_IB_HCA",
    "NCCL_IB_SUBNET_PREFIX_LEN",
    "NCCL_SOCKET_IFNAME",
    "RANK",
    "SPARK_TP4_DEVICE0",
    "SPARK_TP4_DEVICE1",
    "SPARK_TP4_GID0",
    "SPARK_TP4_GID1",
    "SPARK_TP4_PEER0",
    "SPARK_TP4_PEER1",
    "WORLD_SIZE",
}
_NF3_1M_STARTUP_CAP_HOST_PATH = (
    "/var/tmp/sparkring-nf3-1m/spark_nf3_startup_profile_cap.py"
)
_NF3_1M_STARTUP_CAP_CONTAINER_PATH = (
    "/opt/spark-vllm/spark_nf3_startup_profile_cap.py"
)


class LaunchConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LaunchConfig:
    engine: str
    container_name: str
    model_host_path: str
    mtp_draft_host_path: str
    shm_size: str
    startup_timeout_seconds: int
    environment: dict[str, str | None]
    extra_vllm_args: tuple[str, ...]


# RemoteAction is shared from sparkring_runtime (F8 extraction).
RemoteAction = runtime.RemoteAction


def _exact_keys(value: dict, expected: set[str], where: str) -> None:
    unknown = sorted(set(value) - expected)
    missing = sorted(expected - set(value))
    if unknown:
        raise LaunchConfigError(f"{where}: unknown key {unknown[0]!r}")
    if missing:
        raise LaunchConfigError(f"{where}: missing key {missing[0]!r}")


def load_launch(path: Path) -> LaunchConfig:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LaunchConfigError(f"{path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise LaunchConfigError(f"{path}: root must be an object")
    _exact_keys(
        raw,
        {
            "schema",
            "engine",
            "container_name",
            "model_host_path",
            "mtp_draft_host_path",
            "shm_size",
            "startup_timeout_seconds",
            "environment",
            "extra_vllm_args",
        },
        str(path),
    )
    if raw["schema"] != SCHEMA:
        raise LaunchConfigError(f"{path}: unsupported schema {raw['schema']!r}")
    engine = raw["engine"]
    if engine not in ("docker", "podman"):
        raise LaunchConfigError(f"{path}: engine must be docker or podman")
    name = raw["container_name"]
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise LaunchConfigError(f"{path}: invalid container_name")
    model_host_path = raw["model_host_path"]
    if not isinstance(model_host_path, str) or not _ABS.fullmatch(model_host_path):
        raise LaunchConfigError(f"{path}: model_host_path must be shell-safe absolute")
    mtp_draft_host_path = raw["mtp_draft_host_path"]
    if (
        not isinstance(mtp_draft_host_path, str)
        or not _ABS.fullmatch(mtp_draft_host_path)
    ):
        raise LaunchConfigError(
            f"{path}: mtp_draft_host_path must be shell-safe absolute"
        )
    if mtp_draft_host_path == model_host_path:
        raise LaunchConfigError(
            f"{path}: target and MTP draft host paths must be distinct"
        )
    shm_size = raw["shm_size"]
    if not isinstance(shm_size, str) or not re.fullmatch(r"[1-9][0-9]*[gGmM]", shm_size):
        raise LaunchConfigError(f"{path}: shm_size must look like 16g")
    timeout = raw["startup_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, int) or not 30 <= timeout <= 7200:
        raise LaunchConfigError(
            f"{path}: startup_timeout_seconds must be an integer in [30,7200]"
        )
    environment = raw["environment"]
    if not isinstance(environment, dict):
        raise LaunchConfigError(f"{path}: environment must be an object")
    checked_env: dict[str, str | None] = {}
    for key, value in environment.items():
        if not isinstance(key, str) or not _ENV.fullmatch(key):
            raise LaunchConfigError(f"{path}: invalid environment key {key!r}")
        if value is not None:
            if not isinstance(value, str) or "\x00" in value or "\n" in value:
                raise LaunchConfigError(
                    f"{path}: environment {key} must be one line or null"
                )
            _validate_placeholders(value, f"environment.{key}")
        checked_env[key] = value
    derived_override = sorted(set(checked_env) & _SITE_DERIVED_ENVIRONMENT)
    if derived_override:
        raise LaunchConfigError(
            f"{path}: environment {derived_override[0]} is derived from "
            "the validated site and cannot be overridden"
        )
    extra = raw["extra_vllm_args"]
    if not isinstance(extra, list) or not all(
        isinstance(value, str) and value and "\x00" not in value and "\n" not in value
        for value in extra
    ):
        raise LaunchConfigError(f"{path}: extra_vllm_args must be non-empty strings")
    for index, value in enumerate(extra):
        _validate_placeholders(value, f"extra_vllm_args[{index}]")
    return LaunchConfig(
        engine=engine,
        container_name=name,
        model_host_path=model_host_path,
        mtp_draft_host_path=mtp_draft_host_path,
        shm_size=shm_size,
        startup_timeout_seconds=timeout,
        environment=checked_env,
        extra_vllm_args=tuple(extra),
    )


def _validate_placeholders(value: str, where: str) -> None:
    observed = set(_PLACEHOLDER.findall(value))
    unknown = sorted(observed - _ALLOWED_PLACEHOLDERS)
    if unknown:
        raise LaunchConfigError(f"{where}: unknown placeholder {{{unknown[0]}}}")


def _context(site: SiteConfig, rank_id: int) -> dict[str, str]:
    rank = site.rank(rank_id)
    peers_by_rank = {peer.rank: peer for peer in rank.transport_peers}
    # Native TP4 consumes peer slots in recursive-doubling round order:
    # round 0 is rank^1 and round 1 is rank^3. Sorting by rank silently
    # reverses both slots on ranks 2 and 3.
    peers = [peers_by_rank[rank_id ^ 1], peers_by_rank[rank_id ^ 3]]
    ports = {port.peer_rank: port for port in rank.ring_ports}
    master = site.rank(site.serving.master_rank)
    return {
        "api_port": str(site.serving.api_port),
        "draft_path": "/mtp-draft",
        "master_addr": str(master.management.address),
        "master_port": str(site.serving.master_port),
        "model_path": site.runtime.model_path,
        "peer0_addr": str(peers[0].address),
        "peer0_device": ports[peers[0].rank].rdma_device,
        "peer0_gid": str(ports[peers[0].rank].roce_gid_index),
        "peer0_rank": str(peers[0].rank),
        "peer1_addr": str(peers[1].address),
        "peer1_device": ports[peers[1].rank].rdma_device,
        "peer1_gid": str(ports[peers[1].rank].roce_gid_index),
        "peer1_rank": str(peers[1].rank),
        "rank": str(rank_id),
        "world_size": str(len(site.ranks)),
    }


def _expand(value: str, context: dict[str, str]) -> str:
    return _PLACEHOLDER.sub(lambda match: context[match.group(1)], value)


def container_name(config: LaunchConfig, rank: int) -> str:
    return f"{config.container_name}-r{rank}"


def _base_environment(site: SiteConfig, rank_id: int) -> dict[str, str]:
    context = _context(site, rank_id)
    rank = site.rank(rank_id)
    if context["peer0_gid"] != context["peer1_gid"]:
        raise LaunchConfigError(
            f"rank {rank_id} uses different RoCE GID indices on its two "
            "ring ports; NCCL_IB_GID_INDEX is rank-global"
        )
    model_recipe = _model_recipe(site)
    draft_recipe = model_recipe["mtp_draft"]
    return {
        "GLOO_SOCKET_IFNAME": rank.management.interface,
        "NCCL_IB_GID_INDEX": context["peer0_gid"],
        "NCCL_IB_HCA": f"{context['peer0_device']},{context['peer1_device']}",
        "NCCL_IB_SUBNET_PREFIX_LEN": "24",
        "NCCL_SOCKET_IFNAME": rank.management.interface,
        "RANK": context["rank"],
        "WORLD_SIZE": context["world_size"],
        "MASTER_ADDR": context["master_addr"],
        "MASTER_PORT": context["master_port"],
        "SPARKRING_IMAGE_DIGEST": site.runtime.container_image_digest,
        "SPARKRING_DRAFT_CONFIG_SHA256": str(draft_recipe["config_sha256"]),
        "SPARKRING_DRAFT_INDEX_SHA256": str(draft_recipe["index_sha256"]),
        "SPARKRING_DRAFT_WEIGHT_SHA256": str(draft_recipe["weight_sha256"]),
        "SPARKRING_MODEL_CONFIG_SHA256": str(model_recipe["config_sha256"]),
        "SPARKRING_MODEL_INDEX_SHA256": str(model_recipe["index_sha256"]),
        "SPARKRING_MODEL_PATH": site.runtime.model_path,
        "SPARKRING_MODEL_REPOSITORY": site.runtime.model_repo,
        "SPARKRING_MODEL_REVISION": site.runtime.model_revision,
        "SPARKRING_MTP_DRAFT_PATH": "/mtp-draft",
        "SPARKRING_RUNTIME_MANIFEST": "/opt/sparkring/runtime-manifest.json",
        "SPARK_TP4_DEVICE0": context["peer0_device"],
        "SPARK_TP4_DEVICE1": context["peer1_device"],
        "SPARK_TP4_GID0": context["peer0_gid"],
        "SPARK_TP4_GID1": context["peer1_gid"],
        "SPARK_TP4_PEER0": context["peer0_addr"],
        "SPARK_TP4_PEER1": context["peer1_addr"],
    }


def _model_recipe(site: SiteConfig) -> dict:
    # The deployment recipe owns target-model identity. The runtime lock may
    # still describe the ARM64 base image from which the derived NF3 image was
    # built, so using it here would incorrectly couple the target checkpoint
    # to that base image's historical model.
    recipe = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "recipes/glm52-nf3-hybrid.json"
        ).read_text(encoding="utf-8")
    )
    model = recipe["model"]
    if (
        recipe["recipe_id"] != "glm52-nf3-hybrid"
        or model["repository"] != site.runtime.model_repo
        or model["revision"] != site.runtime.model_revision
    ):
        raise LaunchConfigError(
            "site model identity differs from recipes/glm52-nf3-hybrid.json"
        )
    return model


def _model_config_sha(site: SiteConfig) -> str:
    return str(_model_recipe(site)["config_sha256"])


def _option_values(arguments: tuple[str, ...], option: str) -> list[str]:
    return [
        arguments[index + 1]
        for index, value in enumerate(arguments[:-1])
        if value == option
    ]


def _validate_pinned_model_launch(
    site: SiteConfig,
    config: LaunchConfig,
) -> None:
    # Establish that this site names the exact checkpoint pinned by the
    # deployment recipe before enforcing its checkpoint-specific contract.
    _model_config_sha(site)

    overrides = _option_values(config.extra_vllm_args, "--hf-overrides")
    if len(overrides) != 1:
        raise LaunchConfigError(
            "pinned GLM-5.2 checkpoint requires exactly one --hf-overrides"
        )
    try:
        override_document = json.loads(overrides[0])
    except json.JSONDecodeError as exc:
        raise LaunchConfigError(
            "pinned GLM-5.2 checkpoint has invalid --hf-overrides JSON"
        ) from exc
    if override_document != {
        "index_topk_pattern": _GLM52_INDEX_TOPK_PATTERN
    }:
        raise LaunchConfigError(
            "pinned GLM-5.2 checkpoint requires its exact 78-layer "
            "index_topk_pattern"
        )

    candidate_1m = (
        config.environment.get("VLLM_SPARK_RUNTIME_ID")
        == "glm52-nf3-nvfp4-rope8-1m-candidate"
    )
    kv_profile = config.environment.get("VLLM_SPARK_KV_PROFILE")
    kv_contracts = {
        "fp8": {
            "dtype": "fp8",
            "environment": {},
            "forbidden": (
                "VLLM_SPARK_KV_CACHE_DTYPE",
                "VLLM_NVFP4_MLA_PER_TOKEN_SCALE",
                "VLLM_SPARK_KV_SCALE_MODE",
            ),
        },
        "nvfp4-rope8": {
            "dtype": "nvfp4_ds_mla",
            "environment": {
                "VLLM_SPARK_KV_CACHE_DTYPE": "nvfp4_ds_mla",
                "VLLM_NVFP4_MLA_PER_TOKEN_SCALE": "1",
                "VLLM_SPARK_KV_SCALE_MODE": "per-token",
            },
            "forbidden": (),
        },
    }
    if kv_profile not in kv_contracts:
        raise LaunchConfigError(
            "pinned NF3 launch requires VLLM_SPARK_KV_PROFILE=fp8 or "
            "nvfp4-rope8"
        )
    if candidate_1m and kv_profile != "nvfp4-rope8":
        raise LaunchConfigError(
            "NF3 1M candidate requires VLLM_SPARK_KV_PROFILE=nvfp4-rope8"
        )
    kv_contract = kv_contracts[kv_profile]
    kv_dtypes = _option_values(config.extra_vllm_args, "--kv-cache-dtype")
    if kv_dtypes != [kv_contract["dtype"]]:
        raise LaunchConfigError(
            "pinned NF3 launch profile "
            f"{kv_profile} requires exactly one --kv-cache-dtype "
            f"{kv_contract['dtype']}"
        )
    for name, expected in kv_contract["environment"].items():
        if config.environment.get(name) != expected:
            raise LaunchConfigError(
                f"pinned NF3 {kv_profile} launch requires {name}={expected}"
            )
    for name in kv_contract["forbidden"]:
        if name in config.environment:
            raise LaunchConfigError(
                f"pinned NF3 fp8 launch forbids {name}"
            )
    load_formats = _option_values(config.extra_vllm_args, "--load-format")
    if load_formats != ["fastsafetensors"]:
        raise LaunchConfigError(
            "pinned NF3 launch requires exactly one "
            "--load-format fastsafetensors"
        )

    if "--enforce-eager" in config.extra_vllm_args:
        raise LaunchConfigError(
            "the pinned NF3 C8 contract uses CUDA graphs, not --enforce-eager"
        )
    graph_options = {
        "--max-cudagraph-capture-size": "40",
        "--compilation-config": '{"pass_config":{"fuse_allreduce_rms":true}}',
        "--kernel-config": '{"enable_flashinfer_autotune":false}',
    }
    if "--no-enable-flashinfer-autotune" in config.extra_vllm_args:
        raise LaunchConfigError(
            "pinned NF3 launch configures FlashInfer autotune only through "
            "--kernel-config; the duplicate CLI flag is rejected by vLLM"
        )
    for option, expected in graph_options.items():
        if _option_values(config.extra_vllm_args, option) != [expected]:
            raise LaunchConfigError(
                f"pinned NF3 C8 launch requires exactly one {option} {expected}"
            )
    max_batched_tokens = "3072" if candidate_1m else "4096"
    pinned_options = {
        "--attention-backend": "B12X_MLA_SPARSE",
        "--dcp-comm-backend": "ag_rs",
        "--max-num-batched-tokens": max_batched_tokens,
        "--speculative-config": (
            '{"model":"{draft_path}","method":"mtp",'
            '"num_speculative_tokens":4,'
            '"draft_attention_backend":"B12X_MLA_SPARSE",'
            '"adaptive_speculative_tokens_window":32}'
        ),
    }
    for option, expected in pinned_options.items():
        if _option_values(config.extra_vllm_args, option) != [expected]:
            raise LaunchConfigError(
                f"pinned NF3 C8 launch requires exactly one {option} {expected}"
            )

    expected_environment = {
        "B12X_NSA_CONTIGUOUS_PREFILL_BLOCK_K": "auto",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "FASTSAFETENSORS_UNIFIED_MEM": "0",
        "HYBRID_KEPT": "b12x_nf3",
        "HYBRID_NF3": "b12x_nf3",
        "HYBRID_TIER": "both",
        "NCCL_ALGO": "Ring",
        "NCCL_CROSS_NIC": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_DEBUG": "WARN",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_LOCAL_INFERENCE_PATH": "/opt/libnccl-local-inference.so.2.30.4",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_NET": "IB",
        "NCCL_NET_PLUGIN": "none",
        "NCCL_P2P_LEVEL": None,
        "NCCL_PR2127_PATH": "/opt/libnccl-local-inference.so.2.30.4",
        "NCCL_PROTO": None,
        "NCCL_SKIP_TREE_CONNECT": "1",
        "SPARK_ADAPTIVE_MTP_CONTROL": "1",
        "SPARK_CONTEXT_CACHE_ENABLE": "0",
        "SPARK_GLM52_MTP_INDEX_REUSE": "1",
        "SPARK_TP4_ALLGATHER_BASE_PORT": "10200",
        "SPARK_TP4_ALLGATHER_ENABLE_CKV": "0",
        "SPARK_TP4_ALLGATHER_SHADOW_COLLECTIVES": "8",
        "SPARK_TP4_ALLGATHER_SHADOW_PROMOTE": "0",
        "SPARK_TP4_FLIGHT_RECORDER": "0",
        "SPARK_TP4_GRAPH_CONTROL_PORT0": "9970",
        "SPARK_TP4_GRAPH_CONTROL_PORT1": "9971",
        "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT0": "9462",
        "SPARK_TP4_GRAPH_INDEXER_CONTROL_PORT1": "9463",
        "SPARK_TP4_GRAPH_INDEXER_PROGRESS_CPU": "14",
        "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
        "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
        "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0": "10110",
        "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1": "10111",
        "SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU": "12",
        "SPARK_TP4_MAX_INFLIGHT": "64",
        "SPARK_TP4_PERSISTENT_OUTPUT_SLOTS": "0",
        "SPARK_TP4_TRACE_ALLGATHER_SHAPES": "0",
        "SPARK_TP4_VOCAB_CONTROL_PORT0": "9990",
        "SPARK_TP4_VOCAB_CONTROL_PORT1": "9991",
        "SPARK_TP4_VOCAB_SHADOW_COLLECTIVES": "8",
        "SPARK_TP4_VOCAB_SHADOW_PROMOTE": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_ADAPTIVE_SPEC_DEPTHS": "2,4",
        "VLLM_B12X_MLA_CKV_GATHER": "1",
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": (
            "1048576" if candidate_1m else "458752"
        ),
        "VLLM_B12X_MLA_DECODE_GATHER_V2": "0",
        "VLLM_B12X_MLA_DECODE_SPARSE_GATHER": "0",
        "VLLM_CPP_AR_1STAGE_NCCL_CUTOFF": "56KB",
        "VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS": "0",
        "VLLM_DISABLED_KERNELS": "MarlinFP8ScaledMMLinearKernel",
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM": "0",
        "VLLM_DSPARK_CONFIDENCE_SCHEDULER": "off",
        "VLLM_DSPARK_CONFIDENCE_THRESHOLD": "0.0",
        "VLLM_DSPARK_FP8_BATCH_ADAPTIVE": "0",
        "VLLM_DSPARK_FP8_LM_HEAD": "1",
        "VLLM_DSPARK_FP8_MAIN_PROJ": "1",
        "VLLM_DSPARK_FUSED_MARKOV_ARGMAX": "0",
        "VLLM_DSPARK_GREEDY_DRAFT": "0",
        "VLLM_DSPARK_IMPL": "upstream",
        "VLLM_DSPARK_KV_QAT": "1",
        "VLLM_DSPARK_MARKOV_SCALE": "1.0",
        "VLLM_DSPARK_MARKOV_W2_BF16": "1",
        "VLLM_DSPARK_MARKOV_W2_FP8": "1",
        "VLLM_DSPARK_PREFIX_HIT_HOLDBACK": "0",
        "VLLM_DSPARK_REPLICATE_MARKOV_W1": "1",
        "VLLM_DSPARK_REPLICATE_MARKOV_W2": "1",
        "VLLM_ENABLE_PCIE_ALLREDUCE": "1",
        "VLLM_ENGINE_READY_TIMEOUT_S": "3600",
        "VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS": "1800",
        "VLLM_FASTSAFETENSORS_QUEUE_SIZE": "-1",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_PCIE_ALLREDUCE_BACKEND": "cpp",
        "VLLM_RTX6K_FUSED_ALLREDUCE_ADD": "0",
        "VLLM_RTX6K_FUSED_ALLREDUCE_ADD_END_BARRIER": "0",
        "VLLM_SPARK_ENABLE_CUDAGRAPH": "1",
        "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
        "VLLM_SPARK_MAX_CUDAGRAPH_CAPTURE_SIZE": "40",
        "VLLM_SPARK_COMPILATION_PROFILE": "reference-fuse-allreduce-rms-q40",
        "VLLM_SPARK_DECODE_CAPTURE_SIZES": (
            "1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40"
        ),
        "VLLM_SPARK_FULL_DECODE_CAPTURE_SIZES": (
            "5,10,15,20,25,30,35,40"
        ),
        "VLLM_SPARK_GRAPH_CAPTURE_SIZES": (
            "1,2,3,4,5,6,8,10,12,15,16,20,24,25,30,32,35,40"
        ),
        "VLLM_SPARK_DCP_SIZE": "4",
        "VLLM_SPARK_ENABLE_PROFILER": "0",
        "VLLM_SPARK_KV_CACHE_MEMORY_BYTES": (
            "9000000000" if candidate_1m else "7000000000"
        ),
        "VLLM_SPARK_LOAD_FORMAT": "fastsafetensors",
        "VLLM_SPARK_MAX_MODEL_LEN": "1048576" if candidate_1m else "262144",
        "VLLM_SPARK_MAX_NUM_BATCHED_TOKENS": max_batched_tokens,
        "VLLM_SPARK_MAX_NUM_SEQS": "8",
        "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "32",
        "VLLM_SPARK_MTP_MODE_ID": "adaptive-mtp2-4-window32",
        "VLLM_SPARK_MTP_TOKENS": "4",
        "VLLM_SPARK_MTP_DRAFT_SAFETENSORS": "0",
        "VLLM_SPARK_NCCL_TRANSPORT_MODE": "switchless_ib",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_TP4_GRAPH_Q1": "1",
        "VLLM_SPARK_NF3_PROFILE": "reference-four-spark-adaptive-2-4-c8",
        "VLLM_SPARK_NF3_SINGLE_COMPILE_RANGE": "1",
        "VLLM_SPARK_NF3_STARTUP_PROFILE_MAX_TOKENS": "2",
        "VLLM_SPARK_NF3_WORKSPACE_RESERVE_BYTES": "805306368",
        "VLLM_SPARK_PREFILL_PIECEWISE_CAPTURE_SIZES": "",
        "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "1",
        "VLLM_SPARK_TP4_ALLGATHER_POLICY": "spark-custom",
        "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": "0",
        "VLLM_SPARK_TP4_PREFILL_Q512": "0",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "0",
    }
    if candidate_1m:
        expected_environment.update(
            {
                "HYBRID_B12X_MAX_TOKENS": "3072",
                "VLLM_SPARK_RUNTIME_ID": "glm52-nf3-nvfp4-rope8-1m-candidate",
            }
        )
        if site.serving.max_model_len != 1_048_576:
            raise LaunchConfigError(
                "NF3 1M candidate requires site max_model_len=1048576"
            )
        if site.serving.kv_cache_bytes_per_rank != 9_000_000_000:
            raise LaunchConfigError(
                "NF3 1M candidate requires site kv_cache_bytes_per_rank=9000000000"
            )
    for name, expected in expected_environment.items():
        if config.environment.get(name) != expected:
            raise LaunchConfigError(
                f"pinned NF3 launch requires {name}={expected}"
            )
    if site.serving.max_num_seqs != 8:
        raise LaunchConfigError(
            "pinned NF3 first-launch contract requires max_num_seqs=8"
        )

    if (
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL" not in config.environment
        or config.environment["VLLM_PREFIX_CACHE_RETENTION_INTERVAL"] is not None
    ):
        raise LaunchConfigError(
            "VLLM_PREFIX_CACHE_RETENTION_INTERVAL must be null so the pinned "
            "GLM-5.2 launch removes the incompatible base-image value"
        )

    if config.extra_vllm_args.count("--enable-auto-tool-choice") != 1:
        raise LaunchConfigError(
            "pinned GLM-5.2 launch requires exactly one "
            "--enable-auto-tool-choice"
        )

    parser_contract = {
        "--reasoning-parser": "glm47" if candidate_1m else "glm45",
        "--tool-call-parser": "glm47",
    }
    for option, expected in parser_contract.items():
        if _option_values(config.extra_vllm_args, option) != [expected]:
            raise LaunchConfigError(
                f"pinned NF3 launch requires exactly one {option} {expected}"
            )


def start_actions(site: SiteConfig, config: LaunchConfig) -> list[RemoteAction]:
    _validate_pinned_model_launch(site, config)
    actions: list[RemoteAction] = []
    for rank in site.ranks:
        context = _context(site, rank.id)
        environment: dict[str, str | None] = _base_environment(site, rank.id)
        environment.update(
            {
                key: None if value is None else _expand(value, context)
                for key, value in config.environment.items()
            }
        )
        candidate_1m = (
            config.environment.get("VLLM_SPARK_RUNTIME_ID")
            == "glm52-nf3-nvfp4-rope8-1m-candidate"
        )
        argv = [
            config.engine,
            "run",
            "--detach",
            "--name",
            container_name(config, rank.id),
            "--label",
            "org.sparkring.managed=true",
            "--label",
            f"org.sparkring.site={site.name}",
            "--network",
            "host",
            "--ipc",
            "host",
            "--gpus",
            "all",
            "--shm-size",
            config.shm_size,
            "--ulimit",
            "memlock=-1:-1",
            "--cap-add",
            "IPC_LOCK",
            "--device",
            "/dev/infiniband:/dev/infiniband",
            "--volume",
            f"{config.model_host_path}:{site.runtime.model_path}:ro",
            "--volume",
            f"{config.mtp_draft_host_path}:/mtp-draft:ro",
            "--volume",
            f"{site.paths.jit_cache_dir}:/cache/jit",
            "--volume",
            f"{site.paths.context_cache_dir}:{site.paths.context_cache_dir}",
        ]
        if candidate_1m:
            argv.extend(
                (
                    "--volume",
                    f"{_NF3_1M_STARTUP_CAP_HOST_PATH}:"
                    f"{_NF3_1M_STARTUP_CAP_CONTAINER_PATH}:ro",
                    "--entrypoint",
                    "/opt/venv/bin/vllm",
                )
            )
        for key, value in sorted(environment.items()):
            argv.extend(("--env", key if value is None else f"{key}={value}"))
        argv.extend(
            (
                site.runtime.container_image,
                "serve",
                site.runtime.model_path,
                "--tensor-parallel-size",
                str(site.serving.tensor_parallel_size),
                "--decode-context-parallel-size",
                str(site.serving.decode_context_parallel_size),
                "--max-model-len",
                str(site.serving.max_model_len),
                "--kv-cache-memory-bytes",
                str(site.serving.kv_cache_bytes_per_rank),
                "--max-num-seqs",
                str(site.serving.max_num_seqs),
                "--port",
                str(site.serving.api_port),
                "--distributed-executor-backend",
                "mp",
                "--nnodes",
                str(len(site.ranks)),
                "--node-rank",
                str(rank.id),
                "--master-addr",
                context["master_addr"],
                "--master-port",
                context["master_port"],
            )
        )
        argv.extend(_expand(value, context) for value in config.extra_vllm_args)
        if rank.id != site.serving.master_rank:
            argv.append("--headless")
        actions.append(RemoteAction(rank.id, rank.ssh_target, tuple(argv)))
    return actions


def simple_actions(
    site: SiteConfig, config: LaunchConfig, operation: str
) -> list[RemoteAction]:
    actions = []
    for rank in site.ranks:
        name = container_name(config, rank.id)
        if operation == "stop":
            inspect = shlex.join(
                (
                    config.engine,
                    "container",
                    "inspect",
                    "--format",
                    '{{index .Config.Labels "org.sparkring.managed"}}',
                    name,
                )
            )
            remove = shlex.join((config.engine, "rm", "--force", name))
            script = (
                f"managed=$({inspect} 2>/dev/null) || exit 0; "
                '[ "$managed" = true ] || exit 73; '
                f"exec {remove}"
            )
            argv = ("sh", "-c", script)
        elif operation == "status":
            argv = (
                config.engine,
                "inspect",
                "--format",
                "{{.State.Status}}",
                name,
            )
        elif operation == "verify-rollback":
            argv = (
                "sh",
                "-c",
                f"! {shlex.join((config.engine, 'container', 'inspect', name))} "
                ">/dev/null 2>&1",
            )
        else:
            raise ValueError(operation)
        actions.append(RemoteAction(rank.id, rank.ssh_target, tuple(argv)))
    return actions


# run_remote, execute, and action_succeeded are shared from
# sparkring_runtime (F8 extraction).  The shared implementations are
# byte-identical to the originals; compatibility re-exports keep
# existing callers and tests working.
run_remote = runtime.run_remote
execute = runtime.execute
action_succeeded = runtime.action_succeeded


def plan_document(command: str, actions: list[RemoteAction]) -> dict:
    return {
        "schema": "sparkring-public-launch-plan/v1",
        "command": command,
        "mutates_remote": command in ("start", "stop"),
        "actions": [
            {
                "rank": action.rank,
                "ssh_target": action.ssh_target,
                "remote_command": action.shell_command,
            }
            for action in actions
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--site", required=True)
    parser.add_argument("--launch-config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("command", choices=("plan", "start", "stop", "status", "verify-rollback"))
    args = parser.parse_args(argv)
    try:
        site = load_site(args.site)
        config = load_launch(Path(args.launch_config))
        actions = (
            start_actions(site, config)
            if args.command in ("plan", "start")
            else simple_actions(site, config, args.command)
        )
    except (OSError, KeyError, json.JSONDecodeError, SiteConfigError, LaunchConfigError) as exc:
        parser.error(str(exc))

    if not args.execute:
        print(json.dumps(plan_document(args.command, actions), indent=2))
        if args.command != "plan":
            print(
                f"DRY RUN: {args.command} made no remote connection; add --execute",
                file=sys.stderr,
            )
        return 0
    if args.command == "plan":
        parser.error("plan never executes; omit --execute")

    results = execute(actions, config.startup_timeout_seconds)
    failed = [
        rank for rank, result in results.items()
        if not action_succeeded(args.command, result)
    ]
    if args.command == "start" and failed:
        started = {
            rank for rank, result in results.items()
            if action_succeeded(args.command, result)
        }
        rollback = [
            action
            for action in simple_actions(site, config, "stop")
            if action.rank in started
        ]
        rollback_results = execute(rollback, config.startup_timeout_seconds)
    else:
        rollback_results = None
    print(
        json.dumps(
            {
                "schema": "sparkring-public-launch-result/v1",
                "command": args.command,
                "passed": not failed,
                "results": results,
                "rollback_results": rollback_results,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
