#!/usr/bin/env python3
"""Generate an offline-only generic launch profile for the R7 EXL3 candidate."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PINS_PATH = ROOT / "scripts/config/exl3-r7-pins.json"
RECIPE_PATH = ROOT / "recipes/glm52-exl3-r7-3.5bpw.json"
TEMPLATE_PATH = ROOT / "scripts/config/exl3-r7-candidate.example.json"
TEMPLATE_SCHEMA = "sparkring-exl3-r7-candidate-template/v1"
OUTPUT_SCHEMA = "sparkring-runtime-profile/v1"
HEX64 = re.compile(r"[0-9a-f]{64}")
IMAGE_ID = re.compile(r"sha256:[0-9a-f]{64}")
CUDA_COMPAT_LIBRARY = "/usr/local/cuda/compat/libcuda.so.1"
NCCL_LIBRARY = "/opt/venv/lib/python3.12/site-packages/nvidia/nccl/lib/libnccl.so.2"
PATCHED_NCCL_LIBRARY = "/opt/sparkring/nccl/libnccl.so.2"
SIRCL_LIBRARY = "/opt/sparkring/spark_transport/libspark_transport_capi.so"
LD_PRELOAD = CUDA_COMPAT_LIBRARY
ENTRYPOINT_HOST_PATH = "/var/tmp/sparkring-r7-entrypoint-hotfix"
ENTRYPOINT_CONTAINER_PATH = "/usr/local/bin/sparkring-r7-entrypoint"
ENTRYPOINT_SHA256 = "bbc72446e9a7d811c903e76e37e7d9dfce3d21108b2ea7c3db278bb71e84f95e"
WEIGHT_UTILS_HOST_PATH = "/var/tmp/sparkring-r7-weight-utils-local-io.py"
WEIGHT_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/model_loader/weight_utils.py"
)
WEIGHT_UTILS_SHA256 = "da5e6c3429293870d0de611183818fa57c0e9e0ad896784bc739c8a812343102"
EXL3_SCRATCH_HOST_PATH = "/var/tmp/sparkring-r7-exl3-sm121-dense-scratch.py"
EXL3_SCRATCH_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/layers/quantization/exl3.py"
)
EXL3_SCRATCH_SHA256 = "8e0051faf9b8bac9eefd6f38a5f0133a30bca4c0b5ab41962537e2f13cf968f4"
CUDAGRAPH_UTILS_HOST_PATH = "/var/tmp/sparkring-r7-cudagraph-utils-shared-stream.py"
CUDAGRAPH_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/worker/gpu/cudagraph_utils.py"
)
CUDAGRAPH_UTILS_SHA256 = "ef03d64297ed2d1a5161847b48a435bf8ae5feda7a5b81b668d00ae9a1d65a2a"
QUACK_LAYOUT_UTILS_HOST_PATH = "/var/tmp/sparkring-r7-quack-layout_utils-annotations.py"
QUACK_LAYOUT_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/quack/layout_utils.py"
)
QUACK_LAYOUT_UTILS_SHA256 = "3199dc3f55f346183e3d284f6da98f4394eaf14f28b7616d147e6e49ec896194"
QUACK_COPY_UTILS_HOST_PATH = "/var/tmp/sparkring-r7-quack-copy_utils-annotations.py"
QUACK_COPY_UTILS_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/quack/copy_utils.py"
)
QUACK_COPY_UTILS_SHA256 = "2ce88b0d7ee9afe025e52c02fcb32e772a429f1ee626b59546ab8b61d7a37929"
NONFINITE_TRACE_HOST_PATH = "/var/tmp/sparkring-r7-deepseek-v2-nonfinite-trace.py"
NONFINITE_TRACE_CONTAINER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py"
)
NONFINITE_TRACE_SHA256 = "90f4591b71bd8da9e2e37c866bcac17c89583db5d70ae2c80b095a7c35eae01b"
DCP_AUDIT_HOST_PATH = "/var/tmp/sparkring-r7-dcp-audit-head-major.py"
DCP_AUDIT_CONTAINER_PATH = "/opt/spark-vllm/spark_dcp_collective_audit.py"
DCP_AUDIT_SHA256 = "077a234e4edff8b8dd44784953aef713884b4dd7a3f7c46589b14c6bb8b40745"
DCP_GRAPH_STATUS_PATH = "/cache/jit/sparkring-r7-dcp4-stock-graph-status.json"
TVM_FFI_HOST_PATH = "/var/tmp/sparkring-r7-tvm-ffi-0.1.10-r1"
TVM_FFI_CONTAINER_PATH = "/opt/sparkring-r7-tvm-ffi"
TVM_FFI_VERSION = "0.1.10"
TVM_FFI_WHEEL_SHA256 = "3829216a8500c2f61062e48c627f6db6c3fa49416b3ffa85bc04243ae5d759f7"
GLM52_INDEX_TOPK_PATTERN = (
    "FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"
)
ONLINE_QUANTIZATION_CONFIG = {
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
HF_OVERRIDES = {
    "use_index_cache": True,
    "index_topk_pattern": GLM52_INDEX_TOPK_PATTERN,
}
DEFAULT_CHAT_TEMPLATE_KWARGS = {"reasoning_effort": "high"}


class CandidateError(ValueError):
    """The candidate template cannot produce a fail-closed launch profile."""


def _absolute(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.startswith("/"):
        raise CandidateError(f"{name} must be an absolute remote path")
    return value


def generate(template: dict, pins: dict, recipe: dict) -> dict:
    if template.get("schema") != TEMPLATE_SCHEMA:
        raise CandidateError("wrong R7 candidate template schema")
    if pins.get("schema_version") != 1:
        raise CandidateError("wrong R7 pins schema")
    if recipe.get("recipe_id") != "glm52-exl3-r7-3.5bpw":
        raise CandidateError("wrong R7 recipe")
    image_id = template.get("image_id")
    if not isinstance(image_id, str) or not IMAGE_ID.fullmatch(image_id):
        raise CandidateError("image_id must be an exact local ARM64 image ID")
    identity = {}
    for field in ("model_config_sha256", "model_index_sha256"):
        value = template.get(field)
        if not isinstance(value, str) or not HEX64.fullmatch(value):
            raise CandidateError(f"{field} must be 64 lowercase hex characters")
        identity[field] = value
    model = recipe["model"]
    serving = recipe["serving"]
    # The recipe records the qualified serving contract. This generator emits
    # the conservative MTP0/DCP1 baseline from which the separately validated
    # derivative-profile generators build DCP and MTP candidates. Only shared
    # operator defaults are inherited here; baseline KV remains explicit.
    kv_cache_dtype = template.get("kv_cache_dtype", "auto")
    if kv_cache_dtype not in {"auto", "fp8_ds_mla"}:
        raise CandidateError("kv_cache_dtype must be auto or fp8_ds_mla")
    nonfinite_trace = template.get("nonfinite_trace", False)
    if not isinstance(nonfinite_trace, bool):
        raise CandidateError("nonfinite_trace must be true or false")
    dcp_collective_audit = template.get("dcp_collective_audit", True)
    if not isinstance(dcp_collective_audit, bool):
        raise CandidateError("dcp_collective_audit must be true or false")
    online_quantization = template.get("online_quantization", "exl3-b6")
    if online_quantization not in {"exl3-b6", "none"}:
        raise CandidateError("online_quantization must be exl3-b6 or none")
    online_k6 = online_quantization == "exl3-b6"
    transport = template.get("transport", "sircl")
    if transport not in {"sircl", "sircl-nccl-ib", "stock-nccl-ib"}:
        raise CandidateError(
            "transport must be sircl, sircl-nccl-ib, or stock-nccl-ib"
        )
    sircl = transport in {"sircl", "sircl-nccl-ib"}
    nccl_ib = transport in {"sircl-nccl-ib", "stock-nccl-ib"}
    indexer_graph_custom = template.get("indexer_graph_custom", False)
    if not isinstance(indexer_graph_custom, bool):
        raise CandidateError("indexer_graph_custom must be true or false")
    if indexer_graph_custom and not sircl:
        raise CandidateError("indexer_graph_custom requires a SIRCL transport")
    runtime_nccl_library = PATCHED_NCCL_LIBRARY if nccl_ib else NCCL_LIBRARY
    runtime_ld_preload = (
        f"{CUDA_COMPAT_LIBRARY}:{runtime_nccl_library}"
        if nccl_ib
        else LD_PRELOAD
    )
    model_container_path = "/models/glm52-exl3-r7-3.5bpw"
    attest_lines = (
        f"{ENTRYPOINT_SHA256}  {ENTRYPOINT_CONTAINER_PATH}",
        f"{WEIGHT_UTILS_SHA256}  {WEIGHT_UTILS_CONTAINER_PATH}",
        f"{EXL3_SCRATCH_SHA256}  {EXL3_SCRATCH_CONTAINER_PATH}",
        f"{CUDAGRAPH_UTILS_SHA256}  {CUDAGRAPH_UTILS_CONTAINER_PATH}",
        f"{QUACK_LAYOUT_UTILS_SHA256}  {QUACK_LAYOUT_UTILS_CONTAINER_PATH}",
        f"{QUACK_COPY_UTILS_SHA256}  {QUACK_COPY_UTILS_CONTAINER_PATH}",
        *(
            (f"{NONFINITE_TRACE_SHA256}  {NONFINITE_TRACE_CONTAINER_PATH}",)
            if nonfinite_trace
            else ()
        ),
        *(
            (f"{DCP_AUDIT_SHA256}  {DCP_AUDIT_CONTAINER_PATH}",)
            if dcp_collective_audit
            else ()
        ),
        f"{identity['model_config_sha256']}  {model_container_path}/config.json",
        f"{identity['model_index_sha256']}  {model_container_path}/model.safetensors.index.json",
    )
    attest_command = (
        f"test -r {CUDA_COMPAT_LIBRARY} && test -r {runtime_nccl_library} && printf '%s\\n' "
        + " ".join(repr(line) for line in attest_lines)
        + " | sha256sum --check --strict -"
        + f" && PYTHONPATH={TVM_FFI_CONTAINER_PATH} /opt/venv/bin/python -c "
        + repr(
            "import importlib.metadata as md; import pathlib; import tvm_ffi; "
            f'assert md.version("apache-tvm-ffi") == "{TVM_FFI_VERSION}"; '
            f'assert tvm_ffi.__version__ == "{TVM_FFI_VERSION}"; '
            "assert pathlib.Path(tvm_ffi.__file__).resolve().is_relative_to("
            f'pathlib.Path("{TVM_FFI_CONTAINER_PATH}"))'
        )
        + (
            " && /opt/venv/bin/python -c "
            + repr(
                "import ctypes; "
                f'library = ctypes.CDLL("{SIRCL_LIBRARY}"); '
                "assert all(hasattr(library, name) for name in "
                '("spark_tp4_indexer_graph_create", '
                '"spark_tp4_indexer_capture_allgather", '
                '"spark_tp4_indexer_get_graph_status", '
                '"spark_tp4_indexer_graph_destroy"))'
            )
            if indexer_graph_custom
            else ""
        )
        + (
            " && env "
            f"TORCH_USE_RTLD_GLOBAL=1 LD_PRELOAD={runtime_ld_preload} "
            f"VLLM_NCCL_SO_PATH={runtime_nccl_library} "
            "/opt/venv/bin/python -c "
            + repr(
                "import pathlib; import torch; import instanttensor; import vllm; "
                "mapped = set(line.rsplit(None, 1)[-1] for line in "
                'pathlib.Path("/proc/self/maps").read_text().splitlines() '
                'if "libnccl" in line); '
                f'patched = str(pathlib.Path("{PATCHED_NCCL_LIBRARY}").resolve()); '
                f'bundled = str(pathlib.Path("{NCCL_LIBRARY}").resolve()); '
                "assert patched in mapped, (patched, sorted(mapped)); "
                "assert bundled not in mapped, (bundled, sorted(mapped)); "
                "assert len(mapped) == 1, sorted(mapped)"
            )
            if nccl_ib
            else ""
        )
    )
    return {
        "schema": OUTPUT_SCHEMA,
        "profile_id": recipe["recipe_id"],
        "model_family": "exl3-r7",
        "engine": "docker",
        "container_name": "glm52-sparkring-exl3-r7-candidate",
        "image": template["image"],
        "image_id": image_id,
        "model_host_path": _absolute(template.get("model_host_path"), "model_host_path"),
        "model_container_path": model_container_path,
        "shm_size": "16g",
        "startup_timeout_seconds": 3600,
        "environment": {
            "B12X_DENSE_SPLITK_TURBO": "1",
            "B12X_MLA_SM120_UNIFIED": "1",
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_DEVICE_MAX_CONNECTIONS": "32",
            "CUTE_DSL_ARCH": "sm_121a",
            "LD_PRELOAD": runtime_ld_preload,
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
            "NCCL_CROSS_NIC": "1" if nccl_ib else "0",
            "NCCL_CUMEM_ENABLE": "0",
            "NCCL_IB_DISABLE": "0" if nccl_ib else "1",
            "NCCL_IB_MERGE_NICS": "0",
            "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
            "NCCL_IGNORE_CPU_AFFINITY": "1",
            "NCCL_MAX_NCHANNELS": "4" if nccl_ib else None,
            "NCCL_MIN_NCHANNELS": "4" if nccl_ib else None,
            "NCCL_NET": "IB" if nccl_ib else "Socket",
            "NCCL_NET_PLUGIN": "none",
            "NCCL_SKIP_TREE_CONNECT": "1",
            "ONLINE_QUANT": online_quantization,
            "OMP_NUM_THREADS": "16",
            "PYTHONPATH": f"{TVM_FFI_CONTAINER_PATH}:/opt/spark-vllm",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "SPARK_CONTEXT_CACHE_ENABLE": "0",
            "SAFETENSORS_FAST_GPU": "1",
            "SPARK_TP4_ALLGATHER_BASE_PORT": "10200",
            "SPARK_TP4_ALLGATHER_ENABLE_CKV": "0",
            "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "10",
            "SPARK_TP4_FLIGHT_RECORDER": "0",
            "SPARK_TP4_DCP_COLLECTIVE_AUDIT": (
                "1" if dcp_collective_audit else None
            ),
            "SPARK_TP4_GRAPH_STATUS_PATH": (
                DCP_GRAPH_STATUS_PATH if dcp_collective_audit else None
            ),
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
            "SPARK_TP4_LIBRARY": SIRCL_LIBRARY,
            "SPARK_TP4_MAX_INFLIGHT": "64",
            "SPARK_TP4_PERSISTENT_OUTPUT_SLOTS": "0",
            "SPARK_TP4_TRACE_ALLGATHER_SHAPES": "0",
            "SPARK_TP4_VOCAB_CONTROL_PORT0": "9990",
            "SPARK_TP4_VOCAB_CONTROL_PORT1": "9991",
            "SPARK_TP4_VOCAB_EAGER_STAGING_TIMEOUT_SECONDS": "1200",
            "TORCHINDUCTOR_COMPILE_THREADS": "1",
            "TORCH_USE_RTLD_GLOBAL": "1" if nccl_ib else None,
            "VLLM_USE_AOT_COMPILE": "1",
            "VLLM_USE_BREAKABLE_CUDAGRAPH": "0",
            "VLLM_USE_MEGA_AOT_ARTIFACT": "1",
            "VLLM_MEMORY_PROFILE_INCLUDE_ATTN": "1",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
            "VLLM_NCCL_SO_PATH": runtime_nccl_library,
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
            "VLLM_EXL3_ENCODER_SOURCE": None if not online_k6 else "/opt/exllamav3-python/exllamav3",
            "VLLM_EXL3_ONLINE_CACHE_DIR": None if not online_k6 else "/cache/exl3-online",
            "VLLM_EXL3_ONLINE_CACHE_MODE": None if not online_k6 else "readwrite",
            "VLLM_EXL3_ONLINE_TRELLIS_BITS": None if not online_k6 else "6",
            "VLLM_EXL3_PREFILL_CAPACITY": "2048",
            **({"VLLM_SPARK_R7_NONFINITE_TRACE": "1"} if nonfinite_trace else {}),
            "VLLM_SPARK_MAX_QUERY_ROWS": "8" if sircl else None,
            "VLLM_SPARK_MTP_TOKENS": "0",
            "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1",
            "VLLM_SPARK_TP4_ALLGATHER_MODE": "custom" if sircl else None,
            "VLLM_SPARK_TP4_ALLGATHER_POLICY": "spark-custom" if sircl else None,
            "VLLM_SPARK_TP4_GRAPH_Q1": "1" if sircl else None,
            "VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM": (
                "1" if indexer_graph_custom else "0"
            ) if sircl else None,
            "VLLM_SPARK_TP4_MODE": "custom" if sircl else None,
            "VLLM_SPARK_TP4_PREFILL_Q512": "0" if sircl else None,
            "VLLM_SPARK_TP4_VOCAB_MODE": "custom" if sircl else None,
            "XDG_CACHE_HOME": "/cache/jit"
        },
        "extra_vllm_args": [
            "--trust-remote-code",
            "--quantization", "exl3",
            *(
                [
                    "--quantization-config",
                    json.dumps(ONLINE_QUANTIZATION_CONFIG, separators=(",", ":")),
                ]
                if online_k6
                else []
            ),
            "--moe-backend", "b12x",
            "--attention-backend", "B12X_MLA_SPARSE",
            "--kv-cache-dtype", kv_cache_dtype,
            "--compilation-config", json.dumps({
                "cudagraph_mode": "FULL_AND_PIECEWISE",
                "cudagraph_capture_sizes": list(range(1, 33)),
                "custom_ops": ["all"],
                "pass_config": {"fuse_allreduce_rms": True},
            }, separators=(",", ":")),
            "--enable-chunked-prefill",
            "--no-async-scheduling",
            "--max-cudagraph-capture-size", "32",
            "--load-format", "instanttensor",
            "--served-model-name", "glm-5.2-exl3-r7-3.5bpw",
            "--enable-auto-tool-choice",
            "--reasoning-parser", "glm45",
            "--tool-call-parser", "glm47",
            "--default-chat-template-kwargs", json.dumps(
                DEFAULT_CHAT_TEMPLATE_KWARGS, separators=(",", ":")
            ),
            "--hf-overrides", json.dumps(HF_OVERRIDES, separators=(",", ":")),
            "--host", "0.0.0.0",
            "--gpu-memory-utilization", str(serving["gpu_memory_utilization"]),
            # The public chain starts from the conservative 2K prefill profile.
            # The operator recipe records the accepted 4K endpoint; the
            # dynamic-NVFP4 derivative changes this value under an explicit
            # one-variable allowlist.
            "--max-num-batched-tokens", "2048"
        ],
        "extra_volumes": [
            {"host": _absolute(template.get("jit_cache_host_path"), "jit_cache_host_path"), "container": "/cache/jit", "mode": "rw"},
            {"host": _absolute(template.get("online_weight_cache_host_path"), "online_weight_cache_host_path"), "container": "/cache/exl3-online", "mode": "rw"},
            {"host": ENTRYPOINT_HOST_PATH, "container": ENTRYPOINT_CONTAINER_PATH, "mode": "ro"},
            {"host": WEIGHT_UTILS_HOST_PATH, "container": WEIGHT_UTILS_CONTAINER_PATH, "mode": "ro"},
            {"host": EXL3_SCRATCH_HOST_PATH, "container": EXL3_SCRATCH_CONTAINER_PATH, "mode": "ro"},
            {"host": CUDAGRAPH_UTILS_HOST_PATH, "container": CUDAGRAPH_UTILS_CONTAINER_PATH, "mode": "ro"},
            {"host": QUACK_LAYOUT_UTILS_HOST_PATH, "container": QUACK_LAYOUT_UTILS_CONTAINER_PATH, "mode": "ro"},
            {"host": QUACK_COPY_UTILS_HOST_PATH, "container": QUACK_COPY_UTILS_CONTAINER_PATH, "mode": "ro"},
            *(
                [{"host": NONFINITE_TRACE_HOST_PATH, "container": NONFINITE_TRACE_CONTAINER_PATH, "mode": "ro"}]
                if nonfinite_trace
                else []
            ),
            {"host": TVM_FFI_HOST_PATH, "container": TVM_FFI_CONTAINER_PATH, "mode": "ro"},
            *(
                [{"host": DCP_AUDIT_HOST_PATH, "container": DCP_AUDIT_CONTAINER_PATH, "mode": "ro"}]
                if dcp_collective_audit
                else []
            ),
        ],
        "extra_labels": {"org.sparkring.candidate": "exl3-r7-3.5bpw"},
        "privileged": False,
        "entrypoint": None,
        "confirmation": "START-EXL3-R7-CANDIDATE-ALL-FOUR",
        "identity": {
            "model_repository": model["repository"],
            "model_revision": model["revision"],
            **identity
        },
        "attestation_hook": ["/bin/sh", "-c", attest_command],
        "health_check": ["/opt/venv/bin/python", "/opt/sparkring-r7/verify_runtime.py"]
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        template = json.loads(args.template.read_text(encoding="utf-8"))
        pins = json.loads(PINS_PATH.read_text(encoding="utf-8"))
        recipe = json.loads(RECIPE_PATH.read_text(encoding="utf-8"))
        document = generate(template, pins, recipe)
    except (OSError, json.JSONDecodeError, KeyError, CandidateError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE: {args.output}")
    print("RESEARCH-ONLY: inspect the generic launcher plan before any live action")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
