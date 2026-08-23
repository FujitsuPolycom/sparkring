#!/usr/bin/env python3
"""Derive the complete GLM-5.2 EXL3 3.5-bpw pre-exact-Q40 profile offline.

The public interface accepts a resolved candidate template, a complete site,
the four hash-bound SIRCL artifacts, and an output directory. Planning is
dry-run by default. Execution keeps each transformation in-process while
preserving the staged generators' profile, site, rollback, bundle, and receipt
bytes.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "scripts/config/exl3-r7-pins.json"
RECIPE_PATH = ROOT / "recipes/glm52-exl3-r7-3.5bpw.json"
TEMPLATE_PATH = ROOT / "scripts/config/exl3-r7-candidate.example.json"
PROFILE_NAME = "glm52-exl3-3.5bpw-fixed-mtp4-foundation"
_TEMPLATE_SCHEMA = "sparkring-exl3-r7-candidate-template/v1"
_PROFILE_SCHEMA = "sparkring-runtime-profile/v1"
_HEX64 = re.compile(r"[0-9a-f]{64}")
_IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

_CUDA_COMPAT_LIBRARY = "/usr/local/cuda/compat/libcuda.so.1"
_NCCL_LIBRARY = "/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
_PATCHED_NCCL_LIBRARY = "/opt/sparkring/nccl/libnccl.so.2"
_SIRCL_LIBRARY = "/opt/sparkring/spark_transport/libspark_transport_capi.so"
_ENTRYPOINT_CONTAINER_PATH = "/usr/local/bin/sparkring-r7-entrypoint"
_ENTRYPOINT_SHA256 = "bbc72446e9a7d811c903e76e37e7d9dfce3d21108b2ea7c3db278bb71e84f95e"
_WEIGHT_UTILS_CONTAINER_PATH = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/weight_utils.py"
_WEIGHT_UTILS_SHA256 = (
    "da5e6c3429293870d0de611183818fa57c0e9e0ad896784bc739c8a812343102"
)
_EXL3_SCRATCH_CONTAINER_PATH = "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py"
_EXL3_SCRATCH_SHA256 = (
    "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4"
)
_CUDAGRAPH_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/cudagraph_utils.py"
)
_CUDAGRAPH_UTILS_SHA256 = (
    "ef03d64297ed2d1a5161847b48a435bf8ae5feda7a5b81b668d00ae9a1d65a2a"
)
_QUACK_LAYOUT_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/quack/layout_utils.py"
)
_QUACK_LAYOUT_UTILS_SHA256 = (
    "3199dc3f55f346183e3d284f6da98f4394eaf14f28b7616d147e6e49ec896194"
)
_QUACK_COPY_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/quack/copy_utils.py"
)
_QUACK_COPY_UTILS_SHA256 = (
    "2ce88b0d7ee9afe025e52c02fcb32e772a429f1ee626b59546ab8b61d7a37929"
)
_DCP_AUDIT_CONTAINER_PATH = "/opt/spark-vllm/spark_dcp_collective_audit.py"
_DCP_AUDIT_SHA256 = "077a234e4edff8b8dd44784953aef713884b4dd7a3f7c46589b14c6bb8b40745"
_DCP_GRAPH_STATUS_PATH = "/cache/jit/sparkring-r7-dcp4-stock-graph-status.json"
_TVM_FFI_CONTAINER_PATH = "/opt/sparkring-r7-tvm-ffi"
_TVM_FFI_VERSION = "0.1.10"
_GLM52_INDEX_TOPK_PATTERN = (
    "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
)
_ONLINE_QUANTIZATION_CONFIG = {
    "linear": {"weight": "mxfp8"},
    "shared_experts": {"weight": "mxfp8"},
    "ignore": [
        r"re:.*\.fused_qkv_a_proj$",
        r"re:.*\.q_a_proj$",
        r"re:.*kv_a_proj_with_mqa",
        r"re:.*\.mlp\.gate$",
        "model.layers.78.eh_proj",
        "lm_head",
    ],
}
_HF_OVERRIDES = {
    "use_index_cache": True,
    "index_topk_pattern": _GLM52_INDEX_TOPK_PATTERN,
}
_MTP_CONFIG = {
    "method": "mtp",
    "num_speculative_tokens": 2,
    "draft_tensor_parallel_size": 4,
    "quantization": "exl3",
    "moe_backend": "b12x",
    "attention_backend": "B12X_MLA_SPARSE",
    "use_local_argmax_reduction": False,
    "draft_sample_method": "greedy",
}
_MTP2_ENVIRONMENT = {
    "SPARK_ADAPTIVE_MTP_CONTROL": "0",
    "SPARK_GLM52_MTP_INDEX_REUSE": "0",
    "VLLM_ADAPTIVE_SPEC_DEPTHS": None,
    "VLLM_SPARK_MAX_QUERY_ROWS": "24",
    "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "0",
    "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp2",
    "VLLM_SPARK_MTP_TOKENS": "2",
    "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "0",
}
_SHARED_CAPTURE_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/distributed/parallel_state.py"
)
_SHARED_CAPTURE_SHA256 = (
    "b087e93463e9a2d9bede71d3a6e4d696c8f2657449e8dc1119b38613d5750e4e"
)
_EXPECTED_IDENTITY = {
    "model_repository": "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
    "model_revision": "9ab9579774cc432df91567a36f6e9e863e0d4c9f",
    "model_config_sha256": "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126",
    "model_index_sha256": "9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd",
}

_NVFP4_SOURCE_MODEL_LEN = 65_536
_NVFP4_MODEL_LEN = 1_048_576
_NVFP4_SOURCE_BATCHED_TOKENS = 2_048
_NVFP4_BATCHED_TOKENS = 4_096
_NVFP4_KV_CACHE_DTYPE = "nvfp4_ds_mla"
_NVFP4_KV_CONTRACT = "nvfp4-dynamic-per-token+fp8-rope-368-byte"
_REPORTED_KV_CAPACITY_TOKENS = 1_156_864

_CKV_SOURCE_PROFILE_ID = (
    "glm52-exl3-r7-3.5bpw-fixed-mtp4-nvfp4-rope8-ctx1m-b4096"
)
_CKV_PROFILE_ID = f"{_CKV_SOURCE_PROFILE_ID}-ckv-gather"
_CKV_LABEL = "org.sparkring.r7.ckv-prefill-contract"
_CKV_LABEL_VALUE = "dcp4-full-ckv-prefetch1-max1048576"
_CKV_GATHER_MAX_TOKENS = 1_048_576
_CKV_PREFETCH_DEPTH = 1
_CKV_RECORD_BYTES = 368
_DCP_WORLD_SIZE = 4
_MAX_NUM_SEQS = 16
_DCP_KV_INTERLEAVE_SIZE = 1
_KV_BLOCK_SIZE = 64
_CKV_EXECUTION_LANES = 2
_KV_CACHE_BYTES_PER_RANK = 9_250_000_000
_CKV_LOCAL_CAPACITY_TOKENS = (
    (
        (
            (_CKV_GATHER_MAX_TOKENS + _DCP_WORLD_SIZE - 1) // _DCP_WORLD_SIZE
            + _MAX_NUM_SEQS * _DCP_KV_INTERLEAVE_SIZE
            + _KV_BLOCK_SIZE
            - 1
        )
        // _KV_BLOCK_SIZE
    )
    * _KV_BLOCK_SIZE
)
_CKV_WORKSPACE_BYTES_PER_LANE = (
    1 + (_CKV_PREFETCH_DEPTH + 1) * _DCP_WORLD_SIZE
) * _CKV_LOCAL_CAPACITY_TOKENS * _CKV_RECORD_BYTES
_CKV_WORKSPACE_POOL_BYTES_PER_RANK = (
    _CKV_WORKSPACE_BYTES_PER_LANE * _CKV_EXECUTION_LANES
)

_SIRCL_REMOTE_ROOT = "/var/tmp/sparkring-sircl-tiered-v1"
_SIRCL_ARTIFACTS = {
    "transport_library": (
        "libspark_transport_capi.so",
        "/opt/sparkring/spark_transport/libspark_transport_capi.so",
    ),
    "backend": ("spark_tp4_backend.py", "/opt/spark-vllm/spark_tp4_backend.py"),
    "port_namespace": (
        "spark_tp4_port_namespace.py",
        "/opt/spark-vllm/spark_tp4_port_namespace.py",
    ),
    "query_row_provider": (
        "spark_tp4_query_row_provider.py",
        "/opt/spark-vllm/spark_tp4_query_row_provider.py",
    ),
}
_SIRCL_ENVIRONMENT_DELTA = {
    "SPARK_TP4_CONTROL_PORT0": "11100",
    "SPARK_TP4_CONTROL_PORT1": "11101",
    "VLLM_SPARK_TP4_GRAPH_ALLREDUCE_PROTOCOL": "two_slot_deferred_ack",
    "VLLM_SPARK_TP4_GRAPH_KERNEL_STRATEGY": "tiered_64k",
}
_SIRCL_LABEL_DELTA = {
    "org.sparkring.sircl-graph-protocol": "two-slot-deferred-ack",
    "org.sparkring.sircl-graph-kernel": "tiered-64k",
}


class ProfileError(ValueError):
    """The supplied inputs cannot produce a fail-closed GLM profile."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _require_absolute(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ProfileError(f"{name} must be an absolute remote path")
    return value


def _option_values(arguments: list[str], option: str) -> list[str]:
    values: list[str] = []
    for index, value in enumerate(arguments):
        if value == option:
            if index + 1 >= len(arguments):
                raise ProfileError(f"{option} has no value")
            values.append(arguments[index + 1])
        elif value.startswith(option + "="):
            values.append(value.split("=", 1)[1])
    return values


def _require_option(arguments: list[str], option: str, expected: str) -> None:
    if _option_values(arguments, option) != [expected]:
        raise ProfileError(f"profile requires {option}={expected}")


def _replace_option(arguments: list[str], option: str, value: str) -> None:
    if len(_option_values(arguments, option)) != 1:
        raise ProfileError(f"profile requires exactly one {option}")
    arguments[arguments.index(option) + 1] = value


def _single_json_option(arguments: list[str], option: str, role: str) -> dict[str, Any]:
    values = _option_values(arguments, option)
    if len(values) != 1:
        raise ProfileError(f"profile requires exactly one {role}")
    try:
        value = json.loads(values[0])
    except json.JSONDecodeError as exc:
        raise ProfileError(f"{role} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise ProfileError(f"{role} must be a JSON object")
    return value


def _base_profile(
    template: dict[str, Any], pins: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, Any]:
    """Build the conservative candidate whose exact derivatives are validated below."""
    if template.get("schema") != _TEMPLATE_SCHEMA:
        raise ProfileError("wrong candidate template schema")
    if (
        pins.get("schema_version") != 1
        or recipe.get("recipe_id") != "glm52-exl3-r7-3.5bpw"
    ):
        raise ProfileError("pins or recipe schema is wrong")
    image_id = template.get("image_id")
    if not isinstance(image_id, str) or not _IMAGE_ID.fullmatch(image_id):
        raise ProfileError("image_id must be an exact local ARM64 image ID")
    identity: dict[str, str] = {}
    for field in ("model_config_sha256", "model_index_sha256"):
        value = template.get(field)
        if not isinstance(value, str) or not _HEX64.fullmatch(value):
            raise ProfileError(f"{field} must be 64 lowercase hex characters")
        identity[field] = value
    if template.get("transport", "sircl") not in {
        "sircl",
        "sircl-nccl-ib",
        "stock-nccl-ib",
    }:
        raise ProfileError("transport must be a supported profile transport")
    # The foundation always chooses the qualified hybrid transport.
    runtime_nccl = _PATCHED_NCCL_LIBRARY
    ld_preload = f"{_CUDA_COMPAT_LIBRARY}:{runtime_nccl}"
    model_path = "/models/glm52-exl3-r7-3.5bpw"
    attest_lines = (
        f"{_ENTRYPOINT_SHA256}  {_ENTRYPOINT_CONTAINER_PATH}",
        f"{_WEIGHT_UTILS_SHA256}  {_WEIGHT_UTILS_CONTAINER_PATH}",
        f"{_EXL3_SCRATCH_SHA256}  {_EXL3_SCRATCH_CONTAINER_PATH}",
        f"{_CUDAGRAPH_UTILS_SHA256}  {_CUDAGRAPH_UTILS_CONTAINER_PATH}",
        f"{_QUACK_LAYOUT_UTILS_SHA256}  {_QUACK_LAYOUT_UTILS_CONTAINER_PATH}",
        f"{_QUACK_COPY_UTILS_SHA256}  {_QUACK_COPY_UTILS_CONTAINER_PATH}",
        f"{_DCP_AUDIT_SHA256}  {_DCP_AUDIT_CONTAINER_PATH}",
        f"{identity['model_config_sha256']}  {model_path}/config.json",
        f"{identity['model_index_sha256']}  {model_path}/model.safetensors.index.json",
    )
    attest_command = (
        f"test -r {_CUDA_COMPAT_LIBRARY} && test -r {runtime_nccl} && printf '%s\\n' "
        + " ".join(repr(line) for line in attest_lines)
        + " | sha256sum --check --strict -"
        + f" && PYTHONPATH={_TVM_FFI_CONTAINER_PATH} /opt/venv/bin/python -c "
        + repr(
            "import importlib.metadata as md; import pathlib; import tvm_ffi; "
            f'assert md.version("apache-tvm-ffi") == "{_TVM_FFI_VERSION}"; '
            f'assert tvm_ffi.__version__ == "{_TVM_FFI_VERSION}"; '
            "assert pathlib.Path(tvm_ffi.__file__).resolve().is_relative_to("
            f'pathlib.Path("{_TVM_FFI_CONTAINER_PATH}"))'
        )
        + " && env "
        + f"TORCH_USE_RTLD_GLOBAL=1 LD_PRELOAD={ld_preload} VLLM_NCCL_SO_PATH={runtime_nccl} /opt/venv/bin/python -c "
        + repr(
            "import pathlib; import torch; import instanttensor; import vllm; "
            "mapped = set(line.rsplit(None, 1)[-1] for line in "
            'pathlib.Path("/proc/self/maps").read_text().splitlines() '
            'if "libnccl" in line); '
            f'patched = str(pathlib.Path("{_PATCHED_NCCL_LIBRARY}").resolve()); '
            f'bundled = str(pathlib.Path("{_NCCL_LIBRARY}").resolve()); '
            "assert patched in mapped, (patched, sorted(mapped)); "
            "assert bundled not in mapped, (bundled, sorted(mapped)); "
            "assert len(mapped) == 1, sorted(mapped)"
        )
    )
    env: dict[str, Any] = {
        "B12X_DENSE_SPLITK_TURBO": "1",
        "B12X_MLA_SM120_UNIFIED": "1",
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_DEVICE_MAX_CONNECTIONS": "32",
        "CUTE_DSL_ARCH": "sm_121a",
        "LD_PRELOAD": ld_preload,
        "INSTANTTENSOR_BACKEND": "BUFFERED",
        "INSTANTTENSOR_BUFFER_SIZE": "536870912",
        "INSTANTTENSOR_COPY": "0",
        "INSTANTTENSOR_MAX_FREE_MEM_USAGE": "0.05",
        "LOAD_FORMAT": "instanttensor",
        "MOE_MODE": "a16",
        "B12X_MOE_FORCE_A8": "0",
        "B12X_MOE_FORCE_A16": "1",
        "B12X_W4A16_TC_DECODE": "1",
        "B12X_W4A8_TINY_DECODE": "1",
        "NCCL_ALGO": "Ring",
        "NCCL_CROSS_NIC": "1",
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_IB_DISABLE": "0",
        "NCCL_IB_MERGE_NICS": "0",
        "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
        "NCCL_IGNORE_CPU_AFFINITY": "1",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_NET": "IB",
        "NCCL_NET_PLUGIN": "none",
        "NCCL_SKIP_TREE_CONNECT": "1",
        "ONLINE_QUANT": "exl3-b6",
        "OMP_NUM_THREADS": "16",
        "PYTHONPATH": f"{_TVM_FFI_CONTAINER_PATH}:/opt/spark-vllm",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "SPARK_CONTEXT_CACHE_ENABLE": "0",
        "SAFETENSORS_FAST_GPU": "1",
        "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "10",
        "SPARK_TP4_FLIGHT_RECORDER": "0",
        "SPARK_TP4_DCP_COLLECTIVE_AUDIT": "1",
        "SPARK_TP4_GRAPH_STATUS_PATH": _DCP_GRAPH_STATUS_PATH,
        "SPARK_TP4_GRAPH_CONTROL_PORT0": "9970",
        "SPARK_TP4_GRAPH_CONTROL_PORT1": "9971",
        "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
        "SPARK_TP4_GRAPH_SUBMIT_CPU": "10",
        "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT0": "10110",
        "SPARK_TP4_GRAPH_VOCAB_CONTROL_PORT1": "10111",
        "SPARK_TP4_GRAPH_VOCAB_PROGRESS_CPU": "12",
        "SPARK_TP4_LIBRARY": _SIRCL_LIBRARY,
        "SPARK_TP4_MAX_INFLIGHT": "64",
        "SPARK_TP4_PERSISTENT_OUTPUT_SLOTS": "0",
        "SPARK_TP4_VOCAB_CONTROL_PORT0": "9990",
        "SPARK_TP4_VOCAB_CONTROL_PORT1": "9991",
        "SPARK_TP4_VOCAB_EAGER_STAGING_TIMEOUT_SECONDS": "1200",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_USE_RTLD_GLOBAL": "1",
        "VLLM_USE_AOT_COMPILE": "1",
        "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
        "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
        "VLLM_MEMORY_PROFILE_INCLUDE_ATTN": "1",
        "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        "VLLM_NCCL_SO_PATH": runtime_nccl,
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": None,
        "VLLM_B12X_ABSORB_BMM": "0",
        "VLLM_DCP_GLOBAL_TOPK": "1",
        "VLLM_DCP_SHARD_DRAFT": "1",
        "VLLM_USE_B12X_DCP_A2A": "1",
        "VLLM_USE_B12X_FP8_GEMM": "1",
        "VLLM_USE_B12X_MHC": "1",
        "VLLM_USE_B12X_MOE": "1",
        "VLLM_USE_B12X_SPARSE_INDEXER": "1",
        "VLLM_USE_B12X_WO_PROJECTION": "1",
        "VLLM_USE_FLASHINFER_SAMPLER": "1",
        "VLLM_USE_V2_MODEL_RUNNER": "1",
        "VLLM_NO_USAGE_STATS": "1",
        "VLLM_EXL3_ENCODER_SOURCE": "/opt/exllamav3-python/exllamav3",
        "VLLM_EXL3_ONLINE_CACHE_DIR": "/cache/exl3-online",
        "VLLM_EXL3_ONLINE_CACHE_MODE": "readwrite",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
        "VLLM_EXL3_PREFILL_CAPACITY": "2048",
        "VLLM_SPARK_MAX_QUERY_ROWS": "8",
        "VLLM_SPARK_MTP_TOKENS": "0",
        "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
        "VLLM_SPARK_TP4_GRAPH_Q1": "1",
        "VLLM_SPARK_TP4_MODE": "custom",
        "VLLM_SPARK_TP4_PREFILL_Q512": "0",
        "VLLM_SPARK_TP4_VOCAB_MODE": "custom",
        "XDG_CACHE_HOME": "/cache/jit",
    }
    return {
        "schema": _PROFILE_SCHEMA,
        "profile_id": recipe["recipe_id"],
        "model_family": "exl3-r7",
        "engine": "docker",
        "container_name": "glm52-sparkring-exl3-r7-candidate",
        "image": template["image"],
        "image_id": image_id,
        "model_host_path": _require_absolute(
            template.get("model_host_path"), "model_host_path"
        ),
        "model_container_path": model_path,
        "shm_size": "16g",
        "startup_timeout_seconds": 3600,
        "environment": env,
        "extra_vllm_args": [
            "--trust-remote-code",
            "--quantization",
            "exl3",
            "--quantization-config",
            json.dumps(_ONLINE_QUANTIZATION_CONFIG, separators=(",", ":")),
            "--moe-backend",
            "b12x",
            "--attention-backend",
            "B12X_MLA_SPARSE",
            "--kv-cache-dtype",
            "auto",
            "--compilation-config",
            json.dumps(
                {
                    "cudagraph_mode": "FULL_AND_PIECEWISE",
                    "cudagraph_capture_sizes": list(range(1, 33)),
                    "custom_ops": ["all"],
                    "pass_config": {"fuse_allreduce_rms": True},
                },
                separators=(",", ":"),
            ),
            "--enable-chunked-prefill",
            "--no-async-scheduling",
            "--max-cudagraph-capture-size",
            "32",
            "--load-format",
            "instanttensor",
            "--served-model-name",
            "glm-5.2-exl3-r7-3.5bpw",
            "--enable-auto-tool-choice",
            "--reasoning-parser",
            "glm45",
            "--tool-call-parser",
            "glm47",
            "--default-chat-template-kwargs",
            json.dumps({"reasoning_effort": "high"}, separators=(",", ":")),
            "--hf-overrides",
            json.dumps(_HF_OVERRIDES, separators=(",", ":")),
            "--host",
            "0.0.0.0",
            "--gpu-memory-utilization",
            str(recipe["serving"]["gpu_memory_utilization"]),
            "--block-size",
            str(recipe["serving"]["block_size"]),
            "--max-num-batched-tokens",
            "2048",
        ],
        "extra_volumes": [
            {
                "host": _require_absolute(
                    template.get("jit_cache_host_path"), "jit_cache_host_path"
                ),
                "container": "/cache/jit",
                "mode": "rw",
            },
            {
                "host": _require_absolute(
                    template.get("online_weight_cache_host_path"),
                    "online_weight_cache_host_path",
                ),
                "container": "/cache/exl3-online",
                "mode": "rw",
            },
        ],
        "extra_labels": {"org.sparkring.candidate": "exl3-r7-3.5bpw"},
        "privileged": False,
        "entrypoint": None,
        "confirmation": "START-EXL3-R7-CANDIDATE-ALL-FOUR",
        "identity": {
            "model_repository": recipe["model"]["repository"],
            "model_revision": recipe["model"]["revision"],
            **identity,
        },
        "attestation_hook": ["/bin/sh", "-c", attest_command],
        "health_check": ["/opt/venv/bin/python", "/opt/sparkring-r7/verify_runtime.py"],
    }


def _validate_stock(profile: dict[str, Any]) -> None:
    if (
        profile.get("schema") != _PROFILE_SCHEMA
        or profile.get("identity") != _EXPECTED_IDENTITY
    ):
        raise ProfileError("stock DCP4 identity or schema drift")
    env = profile.get("environment")
    args = profile.get("extra_vllm_args")
    if not isinstance(env, dict) or not isinstance(args, list):
        raise ProfileError("stock DCP4 has malformed environment or arguments")
    required_env = {
        "LD_PRELOAD": f"{_CUDA_COMPAT_LIBRARY}:{_PATCHED_NCCL_LIBRARY}",
        "NCCL_ALGO": "Ring",
        "NCCL_CROSS_NIC": "1",
        "NCCL_IB_DISABLE": "0",
        "NCCL_MAX_NCHANNELS": "4",
        "NCCL_MIN_NCHANNELS": "4",
        "NCCL_NET": "IB",
        "ONLINE_QUANT": "exl3-b6",
        "TORCH_USE_RTLD_GLOBAL": "1",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
        "VLLM_NCCL_SO_PATH": _PATCHED_NCCL_LIBRARY,
        "VLLM_SPARK_MAX_QUERY_ROWS": "8",
        "VLLM_SPARK_MTP_TOKENS": "0",
        "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
    }
    if any(env.get(key) != value for key, value in required_env.items()):
        raise ProfileError("stock DCP4 environment drift")
    for option, value in (
        ("--dcp-comm-backend", "ag_rs"),
        ("--dcp-kv-cache-interleave-size", "1"),
        ("--kv-cache-dtype", "fp8_ds_mla"),
        ("--quantization", "exl3"),
        ("--moe-backend", "b12x"),
    ):
        _require_option(args, option, value)
    if _option_values(args, "--speculative-config") or "--enforce-eager" in args:
        raise ProfileError("stock DCP4 speculation or graph contract drift")
    if 24 not in _single_json_option(
        args, "--compilation-config", "compilation config"
    ).get("cudagraph_capture_sizes", []):
        raise ProfileError("stock DCP4 must capture Q24")


def _derive_stock(
    template: dict[str, Any], pins: dict[str, Any], recipe: dict[str, Any]
) -> dict[str, Any]:
    profile = copy.deepcopy(_base_profile(template, pins, recipe))
    args = profile["extra_vllm_args"]
    args[args.index("--moe-backend") + 2 : args.index("--moe-backend") + 2] = [
        "--dcp-comm-backend",
        "ag_rs",
        "--dcp-kv-cache-interleave-size",
        "1",
    ]
    _replace_option(args, "--kv-cache-dtype", "fp8_ds_mla")
    _validate_stock(profile)
    return profile


def _derive_mtp2(stock: dict[str, Any]) -> dict[str, Any]:
    _validate_stock(stock)
    candidate = copy.deepcopy(stock)
    candidate["profile_id"] += "-fixed-mtp2"
    candidate["environment"].update(_MTP2_ENVIRONMENT)
    candidate["extra_labels"].update(
        {
            "org.sparkring.r7.online-k6-scope": "target-only",
            "org.sparkring.r7.target-weight-contract": "checkpoint-exl3-routed+online-k6-eligible-bf16",
            "org.sparkring.r7.draft-weight-contract": "checkpoint-exl3-routed+producer-bf16-nonexpert",
            "org.sparkring.r7.capture-stream-contract": "process-device-shared-target+draft",
        }
    )
    hook = candidate["attestation_hook"][2]
    marker = " | sha256sum --check --strict -"
    if hook.count(marker) != 1 or _SHARED_CAPTURE_PATH in hook:
        raise ProfileError("stock attestation cannot accept the shared capture stream")
    candidate["attestation_hook"][2] = hook.replace(
        marker, f" '{_SHARED_CAPTURE_SHA256}  {_SHARED_CAPTURE_PATH}'{marker}", 1
    )
    spec = {"model": stock["model_container_path"], **_MTP_CONFIG}
    args = candidate["extra_vllm_args"]
    index = args.index("--max-num-batched-tokens")
    args[index:index] = [
        "--speculative-config",
        json.dumps(spec, separators=(",", ":")),
    ]
    return candidate


def _validate_mtp(profile: dict[str, Any], depth: int) -> None:
    suffix = f"-fixed-mtp{depth}"
    if not isinstance(profile.get("profile_id"), str) or not profile[
        "profile_id"
    ].endswith(suffix):
        raise ProfileError(f"fixed-MTP{depth} profile identity drift")
    env = profile.get("environment")
    args = profile.get("extra_vllm_args")
    if not isinstance(env, dict) or not isinstance(args, list):
        raise ProfileError(f"fixed-MTP{depth} profile is malformed")
    query_rows = (depth + 1) * 8
    if env.get("VLLM_SPARK_MTP_TOKENS") != str(depth) or env.get(
        "VLLM_SPARK_MAX_QUERY_ROWS"
    ) != str(query_rows):
        raise ProfileError(f"fixed-MTP{depth} serving geometry drift")
    spec = _single_json_option(args, "--speculative-config", "speculative config")
    if spec != {
        "model": profile["model_container_path"],
        **{**_MTP_CONFIG, "num_speculative_tokens": depth},
    }:
        raise ProfileError(f"fixed-MTP{depth} draft contract drift")
    captures = _single_json_option(
        args, "--compilation-config", "compilation config"
    ).get("cudagraph_capture_sizes")
    if depth >= 3 and captures != list(range(1, query_rows + 1)):
        raise ProfileError(f"fixed-MTP{depth} graph capture drift")


def _derive_mtp3(stock: dict[str, Any], mtp2: dict[str, Any]) -> dict[str, Any]:
    expected = _derive_mtp2(stock)
    if mtp2 != expected:
        raise ProfileError("fixed-MTP2 source is not the exact stock derivative")
    candidate = copy.deepcopy(mtp2)
    candidate["profile_id"] = (
        candidate["profile_id"].removesuffix("-fixed-mtp2") + "-fixed-mtp3"
    )
    candidate["environment"].update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": "32",
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp3",
            "VLLM_SPARK_MTP_TOKENS": "3",
        }
    )
    spec = _single_json_option(
        candidate["extra_vllm_args"], "--speculative-config", "speculative config"
    )
    spec["num_speculative_tokens"] = 3
    _replace_option(
        candidate["extra_vllm_args"],
        "--speculative-config",
        json.dumps(spec, separators=(",", ":")),
    )
    _validate_mtp(candidate, 3)
    return candidate


def _derive_mtp4(mtp3: dict[str, Any]) -> dict[str, Any]:
    _validate_mtp(mtp3, 3)
    candidate = copy.deepcopy(mtp3)
    candidate["profile_id"] = (
        candidate["profile_id"].removesuffix("-fixed-mtp3") + "-fixed-mtp4"
    )
    candidate["environment"].update(
        {
            "VLLM_SPARK_MAX_QUERY_ROWS": "40",
            "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp4",
            "VLLM_SPARK_MTP_TOKENS": "4",
        }
    )
    args = candidate["extra_vllm_args"]
    spec = _single_json_option(args, "--speculative-config", "speculative config")
    spec["num_speculative_tokens"] = 4
    _replace_option(
        args, "--speculative-config", json.dumps(spec, separators=(",", ":"))
    )
    compilation = _single_json_option(
        args, "--compilation-config", "compilation config"
    )
    compilation["cudagraph_capture_sizes"] = list(range(1, 41))
    _replace_option(
        args, "--compilation-config", json.dumps(compilation, separators=(",", ":"))
    )
    _replace_option(args, "--max-cudagraph-capture-size", "40")
    _validate_mtp(candidate, 4)
    return candidate

def _validate_nvfp4_source(profile: dict[str, Any]) -> None:
    """Require the complete fixed-MTP4 contract before changing KV format."""

    _validate_mtp(profile, 4)
    if profile.get("profile_id") != _CKV_SOURCE_PROFILE_ID.removesuffix(
        "-nvfp4-rope8-ctx1m-b4096"
    ):
        raise ProfileError("source fixed-MTP4 profile identity drift")
    environment = profile.get("environment")
    arguments = profile.get("extra_vllm_args")
    labels = profile.get("extra_labels")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ProfileError("source environment or arguments are malformed")
    if not isinstance(labels, dict):
        raise ProfileError("source labels are malformed")
    required_environment = {
        "ONLINE_QUANT": "exl3-b6",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
        "VLLM_EXL3_PREFILL_CAPACITY": str(_NVFP4_SOURCE_BATCHED_TOKENS),
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_MTP_TOKENS": "4",
        "VLLM_USE_B12X_DCP_A2A": "1",
    }
    if any(
        environment.get(key) != value
        for key, value in required_environment.items()
    ):
        raise ProfileError("source fixed-MTP4 environment drift")
    _require_option(
        arguments,
        "--max-num-batched-tokens",
        str(_NVFP4_SOURCE_BATCHED_TOKENS),
    )
    _require_option(arguments, "--kv-cache-dtype", "fp8_ds_mla")
    _require_option(arguments, "--dcp-comm-backend", "ag_rs")
    _require_option(arguments, "--dcp-kv-cache-interleave-size", "1")
    for key in ("KV_FP8_ROPE", "VLLM_NVFP4_MLA_DYNAMIC_SCALE"):
        if key in environment:
            raise ProfileError(f"source profile already declares {key}")
    if "org.sparkring.r7.kv-contract" in labels:
        raise ProfileError("source profile already carries a KV contract label")


def _derive_nvfp4(source: dict[str, Any]) -> dict[str, Any]:
    """Return the exact 1M-context dynamic-NVFP4 derivative."""

    _validate_nvfp4_source(source)
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = f"{source['profile_id']}-nvfp4-rope8-ctx1m-b4096"
    candidate["environment"].update(
        {
            "KV_FP8_ROPE": "1",
            "VLLM_EXL3_PREFILL_CAPACITY": str(_NVFP4_BATCHED_TOKENS),
            "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
        }
    )
    _replace_option(
        candidate["extra_vllm_args"],
        "--max-num-batched-tokens",
        str(_NVFP4_BATCHED_TOKENS),
    )
    _replace_option(
        candidate["extra_vllm_args"], "--kv-cache-dtype", _NVFP4_KV_CACHE_DTYPE
    )
    candidate["extra_labels"]["org.sparkring.r7.kv-contract"] = (
        _NVFP4_KV_CONTRACT
    )
    _validate_nvfp4_candidate(source, candidate)
    return candidate


def _validate_nvfp4_candidate(
    source: dict[str, Any], candidate: dict[str, Any]
) -> None:
    """Require exactly the dynamic-NVFP4 semantic delta."""

    _validate_nvfp4_source(source)
    expected = copy.deepcopy(source)
    expected["profile_id"] = f"{source['profile_id']}-nvfp4-rope8-ctx1m-b4096"
    expected["environment"].update(
        {
            "KV_FP8_ROPE": "1",
            "VLLM_EXL3_PREFILL_CAPACITY": str(_NVFP4_BATCHED_TOKENS),
            "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
        }
    )
    _replace_option(
        expected["extra_vllm_args"],
        "--max-num-batched-tokens",
        str(_NVFP4_BATCHED_TOKENS),
    )
    _replace_option(
        expected["extra_vllm_args"], "--kv-cache-dtype", _NVFP4_KV_CACHE_DTYPE
    )
    expected["extra_labels"]["org.sparkring.r7.kv-contract"] = (
        _NVFP4_KV_CONTRACT
    )
    if candidate != expected:
        raise ProfileError("candidate differs outside the dynamic-NVFP4 allowlist")


def _derive_nvfp4_site(source: str) -> str:
    required = (
        "  tensor_parallel_size: 4",
        "  decode_context_parallel_size: 4",
        '  mtp_mode: "static"',
        "  mtp_tokens: 4",
        "  kv_cache_bytes_per_rank: 9250000000",
        "  max_num_seqs: 16",
    )
    for line in required:
        if source.count(line) != 1:
            raise ProfileError(
                f"source site must declare exactly one {line.strip()}"
            )
    return _replace_site_value(
        source,
        f"  max_model_len: {_NVFP4_SOURCE_MODEL_LEN}",
        f"  max_model_len: {_NVFP4_MODEL_LEN}",
        "dynamic-NVFP4",
    )


def _validate_ckv_source(profile: dict[str, Any]) -> None:
    """Require the exact MTP4/NVFP4/DCP4 behavior before CKV gather."""

    if profile.get("profile_id") != _CKV_SOURCE_PROFILE_ID:
        raise ProfileError("source profile_id is not the live b4096 profile")
    environment = profile.get("environment")
    arguments = profile.get("extra_vllm_args")
    labels = profile.get("extra_labels")
    if not isinstance(environment, dict) or not isinstance(arguments, list):
        raise ProfileError("source profile environment/args are malformed")
    if not isinstance(labels, dict):
        raise ProfileError("source profile labels are malformed")
    required_environment = {
        "ONLINE_QUANT": "exl3-b6",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
        "VLLM_EXL3_PREFILL_CAPACITY": "4096",
        "VLLM_SPARK_MAX_QUERY_ROWS": "40",
        "VLLM_SPARK_MTP_TOKENS": "4",
        "VLLM_USE_B12X_DCP_A2A": "1",
        "KV_FP8_ROPE": "1",
        "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
    }
    descriptions = {
        "VLLM_EXL3_PREFILL_CAPACITY": "prefill capacity",
        "KV_FP8_ROPE": "FP8-RoPE contract",
        "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "dynamic NVFP4 contract",
    }
    for key, expected in required_environment.items():
        if environment.get(key) != expected:
            raise ProfileError(
                f"source {descriptions.get(key, key)} must be {expected}"
            )
    for key in (
        "VLLM_B12X_MLA_CKV_GATHER",
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS",
        "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH",
        "VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB",
    ):
        if key in environment:
            raise ProfileError(f"source profile already declares {key}")
    _require_option(arguments, "--max-num-batched-tokens", "4096")
    _require_option(arguments, "--kv-cache-dtype", _NVFP4_KV_CACHE_DTYPE)
    if _single_json_option(
        arguments, "--speculative-config", "speculative config"
    ).get("num_speculative_tokens") != 4:
        raise ProfileError("source speculative depth must be fixed-MTP4")
    if labels.get("org.sparkring.r7.kv-contract") != _NVFP4_KV_CONTRACT:
        raise ProfileError("source 368-byte NVFP4 KV label is missing")
    if _CKV_LABEL in labels:
        raise ProfileError("source profile already carries a CKV-prefill label")


def _derive_ckv_gather(source: dict[str, Any]) -> dict[str, Any]:
    """Enable bounded full-CKV prefill gather without changing serving geometry."""

    _validate_ckv_source(source)
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = _CKV_PROFILE_ID
    candidate["environment"]["VLLM_B12X_MLA_CKV_GATHER"] = "1"
    candidate["environment"]["VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS"] = str(
        _CKV_GATHER_MAX_TOKENS
    )
    candidate["extra_labels"][_CKV_LABEL] = _CKV_LABEL_VALUE
    return candidate


def _validate_ckv_site(site: str) -> None:
    for line in (
        "  decode_context_parallel_size: 4",
        "  max_model_len: 1048576",
        "  kv_cache_bytes_per_rank: 9250000000",
        "  max_num_seqs: 16",
    ):
        if site.count(line) != 1:
            raise ProfileError(
                f"source site must declare exactly one {line.strip()}"
            )


def _validate_sircl_source(profile: dict[str, Any]) -> None:
    """Require the full-CKV derivative before enabling native transport."""

    environment = profile.get("environment")
    labels = profile.get("extra_labels")
    volumes = profile.get("extra_volumes")
    if (
        not isinstance(environment, dict)
        or not isinstance(labels, dict)
        or not isinstance(volumes, list)
    ):
        raise ProfileError("source SIRCL profile fields are malformed")
    ckv_source = copy.deepcopy(profile)
    ckv_source["profile_id"] = _CKV_SOURCE_PROFILE_ID
    ckv_source["environment"].pop("VLLM_B12X_MLA_CKV_GATHER", None)
    ckv_source["environment"].pop("VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS", None)
    ckv_source["extra_labels"].pop(_CKV_LABEL, None)
    _validate_ckv_source(ckv_source)
    if profile.get("profile_id") != _CKV_PROFILE_ID:
        raise ProfileError("source profile_id is not the full-CKV derivative")
    if environment.get("VLLM_B12X_MLA_CKV_GATHER") != "1":
        raise ProfileError("source full-CKV gather must be enabled")
    if environment.get("VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS") != str(
        _CKV_GATHER_MAX_TOKENS
    ):
        raise ProfileError(
            f"source CKV gather ceiling must be {_CKV_GATHER_MAX_TOKENS}"
        )
    if labels.get(_CKV_LABEL) != _CKV_LABEL_VALUE:
        raise ProfileError("source CKV label is missing")
    for key in _SIRCL_ENVIRONMENT_DELTA:
        if key in environment:
            raise ProfileError(f"source profile already declares {key}")
    for _filename, container in _SIRCL_ARTIFACTS.values():
        if any(volume.get("container") == container for volume in volumes):
            raise ProfileError(f"source profile already mounts {container}")


def _inject_sircl_attestation(
    command: str, artifact_digests: dict[str, str]
) -> str:
    marker = " | sha256sum --check --strict -"
    if command.count(marker) != 1:
        raise ProfileError(
            "source attestation checksum marker is absent or non-unique"
        )
    additions = [
        repr(f"{artifact_digests[name]}  {container}")
        for name, (_filename, container) in _SIRCL_ARTIFACTS.items()
    ]
    return command.replace(marker, " " + " ".join(additions) + marker, 1)


def _derive_sircl_tiered(
    source: dict[str, Any], *, artifact_digests: dict[str, str]
) -> dict[str, Any]:
    """Return the exact tiered/deferred SIRCL derivative."""

    _validate_sircl_source(source)
    if set(artifact_digests) != set(_SIRCL_ARTIFACTS):
        raise ProfileError("SIRCL artifact digest inventory is incomplete")
    for name, digest in artifact_digests.items():
        if not _SHA256.fullmatch(digest):
            raise ProfileError(f"{name} SHA-256 is invalid")
    candidate = copy.deepcopy(source)
    candidate["profile_id"] = f"{source['profile_id']}-sircl-tiered"
    candidate["container_name"] = "glm52-sparkring-sircl-tiered"
    candidate["confirmation"] = "START-SIRCL-TIERED-ALL-FOUR"
    candidate["environment"].update(_SIRCL_ENVIRONMENT_DELTA)
    candidate["extra_labels"].update(_SIRCL_LABEL_DELTA)
    for filename, container in _SIRCL_ARTIFACTS.values():
        candidate["extra_volumes"].append(
            {
                "host": f"{_SIRCL_REMOTE_ROOT}/{filename}",
                "container": container,
                "mode": "ro",
            }
        )
    hook = candidate.get("attestation_hook")
    if not isinstance(hook, list) or len(hook) != 3 or not isinstance(hook[2], str):
        raise ProfileError("source attestation hook shape drifted")
    hook[2] = _inject_sircl_attestation(hook[2], artifact_digests)
    _validate_sircl_candidate(
        source, candidate, artifact_digests=artifact_digests
    )
    return candidate


def _validate_sircl_candidate(
    source: dict[str, Any],
    candidate: dict[str, Any],
    *,
    artifact_digests: dict[str, str],
) -> None:
    expected = copy.deepcopy(source)
    expected["profile_id"] = f"{source['profile_id']}-sircl-tiered"
    expected["container_name"] = "glm52-sparkring-sircl-tiered"
    expected["confirmation"] = "START-SIRCL-TIERED-ALL-FOUR"
    expected["environment"].update(_SIRCL_ENVIRONMENT_DELTA)
    expected["extra_labels"].update(_SIRCL_LABEL_DELTA)
    for filename, container in _SIRCL_ARTIFACTS.values():
        expected["extra_volumes"].append(
            {
                "host": f"{_SIRCL_REMOTE_ROOT}/{filename}",
                "container": container,
                "mode": "ro",
            }
        )
    expected["attestation_hook"][2] = _inject_sircl_attestation(
        expected["attestation_hook"][2], artifact_digests
    )
    if candidate != expected:
        raise ProfileError("candidate differs outside the tiered SIRCL allowlist")


def _stock_site(site: str) -> str:
    replacements = (
        ('  mtp_mode: "static"', '  mtp_mode: "off"'),
        ("  mtp_tokens: 4", "  mtp_tokens: 0"),
        (
            "  kv_cache_bytes_per_rank: 9250000000",
            "  kv_cache_bytes_per_rank: 9000000000",
        ),
    )
    for before, after in replacements:
        if site.count(before) != 1:
            raise ProfileError(
                f"complete site requires exactly one {before.strip()} declaration"
            )
        site = site.replace(before, after)
    return site


def _replace_site_value(site: str, before: str, after: str, role: str) -> str:
    if site.count(before) != 1 or after in site:
        raise ProfileError(f"{role} site declaration is not exact")
    candidate = site.replace(before, after)
    if candidate.replace(after, before) != site:
        raise ProfileError(f"{role} site contains semantic drift")
    return candidate


def _validate_resolved_inputs(site: Path, template: Path) -> None:
    from sparkring_site import load_site

    if not site.is_file() or not template.is_file():
        raise ProfileError("--site and --template must name existing files")
    try:
        document = json.loads(template.read_text(encoding="utf-8"))
        parsed_site = load_site(site)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProfileError(f"invalid resolved profile input: {exc}") from exc
    image = document.get("image")
    image_id = document.get("image_id")
    if (
        not isinstance(image, str)
        or "REPLACE" in image.upper()
        or image_id == "sha256:" + "1" * 64
    ):
        raise ProfileError("candidate template image is unresolved")
    expected = {
        "image": (image, parsed_site.runtime.container_image),
        "image_id": (image_id, parsed_site.runtime.container_image_digest),
        "model_host_path": (
            document.get("model_host_path"),
            parsed_site.runtime.model_path,
        ),
        "jit_cache_host_path": (
            document.get("jit_cache_host_path"),
            parsed_site.paths.jit_cache_dir,
        ),
    }
    for field, (profile_value, site_value) in expected.items():
        if profile_value != site_value:
            raise ProfileError(
                f"candidate template {field} does not match the complete site"
            )


def _json_bytes(document: dict[str, Any]) -> bytes:
    return (json.dumps(document, indent=2) + "\n").encode("utf-8")


def plan(
    *,
    template: Path | None = None,
    site: Path | None = None,
    output_dir: Path | None = None,
    artifact_paths: dict[str, Path] | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Plan or execute the complete pre-exact-Q40 profile derivation."""

    if output_dir is None:
        output_dir = ROOT / ".sparkring" / "exl3-r7"
    if template is None:
        template = TEMPLATE_PATH
    if not dry_run:
        if site is None:
            raise ProfileError("--execute requires a complete ignored --site file")
        _validate_resolved_inputs(site, template)
        if artifact_paths is None or set(artifact_paths) != set(_SIRCL_ARTIFACTS):
            raise ProfileError("SIRCL artifact path inventory is incomplete")
        for name, path in artifact_paths.items():
            if not path.is_file():
                raise ProfileError(f"{name} is not a regular file")
        bundle = output_dir / "pre-q40-bundle"
        if bundle.exists():
            raise ProfileError(f"refusing to replace bundle {bundle}")

    receipt: dict[str, Any] = {"dry_run": dry_run, "steps": []}
    recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
    pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
    receipt["steps"].extend(
        (
            {"step": "profile-recipe", "safety": "OFFLINE"},
            {"step": "source-pins", "safety": "OFFLINE"},
            {"step": "speculation-off-stock-dcp4", "safety": "OFFLINE"},
            {"step": "fixed-mtp2-intermediate", "safety": "OFFLINE"},
            {"step": "fixed-mtp3-rollback", "safety": "OFFLINE"},
            {"step": "kv-cache-9.25gb-per-rank", "safety": "OFFLINE"},
            {"step": "fixed-mtp4", "safety": "OFFLINE"},
            {"step": "dynamic-nvfp4-key-value-layout", "safety": "OFFLINE"},
            {"step": "full-compressed-key-value-gather", "safety": "OFFLINE"},
            {"step": "tiered-sircl-tp-all-reduce", "safety": "OFFLINE"},
        )
    )
    if dry_run:
        if site and site.exists():
            receipt["steps"].extend(
                (
                    {"step": "site-validation", "safety": "OFFLINE"},
                    {"step": "read-only-preflight-plan", "safety": "READ-ONLY REMOTE"},
                )
            )
    else:
        assert site is not None
        assert artifact_paths is not None
        stock = _derive_stock(
            json.loads(template.read_text(encoding="utf-8")), pins, recipe
        )
        mtp2 = _derive_mtp2(stock)
        mtp3 = _derive_mtp3(stock, mtp2)
        stock_site = _stock_site(site.read_text(encoding="utf-8"))
        mtp2_site = _replace_site_value(
            _replace_site_value(
                stock_site,
                '  mtp_mode: "off"',
                '  mtp_mode: "static"',
                "stock",
            ),
            "  mtp_tokens: 0",
            "  mtp_tokens: 2",
            "stock",
        )
        mtp3_site = _replace_site_value(
            mtp2_site, "  mtp_tokens: 2", "  mtp_tokens: 3", "MTP2"
        )
        kv925_site = _replace_site_value(
            mtp3_site,
            "  kv_cache_bytes_per_rank: 9000000000",
            "  kv_cache_bytes_per_rank: 9250000000",
            "MTP3",
        )
        mtp4 = _derive_mtp4(mtp3)
        mtp4_site = _replace_site_value(
            kv925_site, "  mtp_tokens: 3", "  mtp_tokens: 4", "MTP3 KV9.25"
        )
        artifact_digests = {
            name: _sha256_bytes(path.read_bytes())
            for name, path in artifact_paths.items()
        }
        nvfp4 = _derive_nvfp4(mtp4)
        nvfp4_site = _derive_nvfp4_site(mtp4_site)
        _validate_ckv_site(nvfp4_site)
        ckv = _derive_ckv_gather(nvfp4)
        pre_q40 = _derive_sircl_tiered(
            ckv, artifact_digests=artifact_digests
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        foundation_outputs = {
            "stock-dcp4-profile.json": _json_bytes(stock),
            "stock-site.yaml": stock_site.encode(),
            "mtp2-profile.json": _json_bytes(mtp2),
            "mtp2-site.yaml": mtp2_site.encode(),
            "mtp2-rollback.json": _json_bytes(stock),
            "mtp3-profile.json": _json_bytes(mtp3),
            "mtp3-site.yaml": mtp3_site.encode(),
            "mtp3-rollback.json": _json_bytes(mtp2),
            "mtp3-kv925-profile.json": _json_bytes(mtp3),
            "mtp3-kv925-site.yaml": kv925_site.encode(),
            "mtp3-kv925-rollback.json": _json_bytes(mtp3),
            "mtp3-kv925-rollback-site.yaml": mtp3_site.encode(),
            "mtp4-kv925-profile.json": _json_bytes(mtp4),
            "mtp4-kv925-site.yaml": mtp4_site.encode(),
            "mtp4-kv925-rollback.json": _json_bytes(mtp3),
            "mtp4-kv925-rollback-site.yaml": kv925_site.encode(),
        }
        for name, payload in foundation_outputs.items():
            # The foundation generators used platform text newlines.
            (output_dir / name).write_text(payload.decode("utf-8"), encoding="utf-8")

        mtp4_profile_bytes = (output_dir / "mtp4-kv925-profile.json").read_bytes()
        mtp4_site_bytes = (output_dir / "mtp4-kv925-site.yaml").read_bytes()
        nvfp4_profile_bytes = _json_bytes(nvfp4)
        nvfp4_site_bytes = _derive_nvfp4_site(
            mtp4_site_bytes.decode("utf-8")
        ).encode("utf-8")
        stage_outputs = {
            "mtp4-nvfp4-profile.json": nvfp4_profile_bytes,
            "mtp4-nvfp4-site.yaml": nvfp4_site_bytes,
            "mtp4-nvfp4-rollback.json": mtp4_profile_bytes,
            "mtp4-nvfp4-rollback-site.yaml": mtp4_site_bytes,
            "mtp4-ckv-gather-profile.json": _json_bytes(ckv),
            "mtp4-ckv-gather-site.yaml": nvfp4_site_bytes,
            "mtp4-ckv-gather-rollback.json": nvfp4_profile_bytes,
            "mtp4-ckv-gather-rollback-site.yaml": nvfp4_site_bytes,
        }
        for name, payload in stage_outputs.items():
            (output_dir / name).write_bytes(payload)

        bundle.mkdir(parents=True)
        files: dict[str, dict[str, str | int]] = {}
        for name, source_path in artifact_paths.items():
            filename, _container = _SIRCL_ARTIFACTS[name]
            destination = bundle / filename
            shutil.copyfile(source_path, destination)
            files[name] = {
                "path": str(destination.resolve()),
                "sha256": _sha256_bytes(destination.read_bytes()),
                "bytes": destination.stat().st_size,
            }
        ckv_profile_path = output_dir / "mtp4-ckv-gather-profile.json"
        ckv_profile_sha256 = _sha256_bytes(ckv_profile_path.read_bytes())
        manifest = {
            "schema": "sparkring-r7-sircl-tiered-bundle/v1",
            "maturity": "offline-validated",
            "remote_root": _SIRCL_REMOTE_ROOT,
            "base_profile": {
                "path": str(ckv_profile_path.resolve()),
                "sha256": ckv_profile_sha256,
            },
            "candidate_profile_id": pre_q40["profile_id"],
            "files": files,
            "policy": {
                "graph_protocol": "two_slot_deferred_ack",
                "kernel_strategy": "tiered_64k",
                "dual_port": False,
                "prefill_capacity_pool": False,
            },
            "rollback": {
                "profile": str(ckv_profile_path.resolve()),
                "sha256": ckv_profile_sha256,
            },
        }
        pre_q40_profile = output_dir / "pre-q40-profile.json"
        pre_q40_site = output_dir / "pre-q40-site.yaml"
        pre_q40_receipt = output_dir / "pre-q40-receipt.json"
        pre_q40_profile.write_text(
            json.dumps(pre_q40, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        pre_q40_site.write_bytes(nvfp4_site_bytes)
        pre_q40_receipt.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        rollback_profile = output_dir / "mtp4-kv925-rollback.json"
        rollback_site = output_dir / "mtp4-kv925-rollback-site.yaml"
        source_profile = output_dir / "mtp3-kv925-profile.json"
        source_site = output_dir / "mtp3-kv925-site.yaml"
        if (
            rollback_profile.read_bytes() != source_profile.read_bytes()
            or rollback_site.read_bytes() != source_site.read_bytes()
        ):
            raise ProfileError(
                "fixed-MTP4 rollback is not byte-identical to MTP3 KV9.25"
            )
        if (
            (output_dir / "mtp4-nvfp4-rollback.json").read_bytes()
            != mtp4_profile_bytes
            or (output_dir / "mtp4-ckv-gather-rollback.json").read_bytes()
            != nvfp4_profile_bytes
        ):
            raise ProfileError("pre-Q40 rollback bytes drifted")

        receipt.update(
            {
                "stock_profile_sha256": _sha256_bytes(
                    (output_dir / "stock-dcp4-profile.json").read_bytes()
                ),
                "mtp4_profile_sha256": _sha256_bytes(mtp4_profile_bytes),
                "mtp4_site_sha256": _sha256_bytes(mtp4_site_bytes),
                "rollback_profile_sha256": _sha256_bytes(
                    rollback_profile.read_bytes()
                ),
                "rollback_site_sha256": _sha256_bytes(rollback_site.read_bytes()),
                "rollback_identity": (
                    "MTP4 rollback is byte-identical to MTP3 KV9.25"
                ),
                "nvfp4_profile_sha256": _sha256_bytes(nvfp4_profile_bytes),
                "nvfp4_site_sha256": _sha256_bytes(nvfp4_site_bytes),
                "ckv_gather_profile_sha256": ckv_profile_sha256,
                "ckv_workspace_pool_bytes_per_rank": (
                    _CKV_WORKSPACE_POOL_BYTES_PER_RANK
                ),
                "reported_kv_capacity_tokens": _REPORTED_KV_CAPACITY_TOKENS,
                "sircl_artifact_sha256": artifact_digests,
                "pre_q40_profile_sha256": _sha256_bytes(
                    pre_q40_profile.read_bytes()
                ),
                "pre_q40_site_sha256": _sha256_bytes(pre_q40_site.read_bytes()),
                "pre_q40_receipt_sha256": _sha256_bytes(
                    pre_q40_receipt.read_bytes()
                ),
            }
        )
        receipt["steps"].extend(
            (
                {"step": "site-validation", "safety": "OFFLINE"},
                {"step": "read-only-preflight-plan", "safety": "READ-ONLY REMOTE"},
            )
        )
    receipt.update(
        {
            "operator_deployment_status": "qualified",
            "generated_profile_status": "implemented",
            "note": "Operator qualification applies to the documented image and four-Spark appliance. A profile generated for another image ID requires the live promotion gate.",
        }
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan",))
    parser.add_argument("--template", type=Path, default=None)
    parser.add_argument("--site", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--transport-library", type=Path, default=None)
    parser.add_argument("--backend", type=Path, default=None)
    parser.add_argument("--port-namespace", type=Path, default=None)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    try:
        artifact_paths = None
        if args.execute:
            supplied = (
                args.transport_library,
                args.backend,
                args.port_namespace,
            )
            if any(path is None for path in supplied):
                raise ProfileError(
                    "--execute requires the transport library, backend, and port namespace"
                )
            assert args.backend is not None
            artifact_paths = {
                "transport_library": args.transport_library.resolve(),
                "backend": args.backend.resolve(),
                "port_namespace": args.port_namespace.resolve(),
                "query_row_provider": args.backend.resolve().parent
                / "spark_tp4_query_row_provider.py",
            }
        receipt = plan(
            template=args.template,
            site=args.site,
            output_dir=args.output_dir,
            artifact_paths=artifact_paths,
            dry_run=not args.execute,
        )
    except (OSError, json.JSONDecodeError, ProfileError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
