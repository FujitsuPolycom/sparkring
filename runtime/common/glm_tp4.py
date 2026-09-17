"""Pure container planning for the four source-bound GLM TP4 profiles.

The caller verifies the image receipt, installed contract, model files and private
site before supplying a rendered rank mapping. This owner resolves launch defaults
and constructs argv directly. It never sources configuration, invokes a launcher,
reads process environment, or contacts Docker. Legacy and diagnostic variants are
deliberately outside this adapter's admission boundary.
"""
from __future__ import annotations

from collections.abc import Mapping
import json
import math
from pathlib import PurePosixPath
import re

from runtime.common.container_spec import Bind, ContainerSpec
from runtime.common import glm_targets, glm_source_candidate

PROFILES = frozenset(("tp4-dcp1", "tp4-dcp1-sparkcache", "tp4-dcp4", "tp4-dcp4-sparkcache"))
NCCL_PATH = "/opt/local-inference/nccl/lib/libnccl.so.2"
SIRCL_ROOT = "/opt/sparkring/sircl"
TARGET_FINGERPRINT = "357f6a86160ebd5caff25d9a10d9f29e8547b16c6c73e78751fa69fde11ac4e4"

# Launch defaults not necessarily present in the managed rank mapping. Profile,
# image, topology and model identities must always come from the validated input.
DEFAULTS = {
    "PORT": "8015", "MASTER_PORT": "29775", "SHM_SIZE": "32g",
    "PIPELINE_PARALLEL_SIZE": "1", "CP_KV_CACHE_INTERLEAVE_SIZE": "auto",
    "B12X_MLA_CKV_GATHER": "auto", "B12X_FUSED_INDEXER": "1",
    "B12X_MLA_CKV_GATHER_MAX_TOKENS": "524288", "MAX_NUM_SEQS": "16",
    "MAX_NUM_BATCHED_TOKENS": "8192", "PREFILL_SCHEDULE_INTERVAL": "2",
    "MAX_IMAGES_PER_PROMPT": "4", "MAX_VIDEOS_PER_PROMPT": "1",
    "KV_CACHE_MEMORY_BYTES": "auto", "GPU_MEMORY_UTILIZATION": "0.80",
    "KV_CACHE_DTYPE": "fp8", "DRAFT_TENSOR_PARALLEL_SIZE": "4",
    "DRAFT_KV_CACHE_DTYPE": "auto", "DRAFT_SAMPLE_METHOD": "probabilistic",
    "REJECTION_SAMPLE_METHOD": "standard", "ATTENTION_BACKEND": "B12X",
    "MOE_BACKEND": "b12x", "LINEAR_BACKEND": "b12x", "KDA_PREFILL_BACKEND": "b12x",
    "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE", "JIT_MONITOR_VERBOSE": "0",
    "DFLASH_WARMUP": "0", "DFLASH_WARMUP_CONCURRENCIES": "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16",
    "DFLASH_WARMUP_SHAPE_WORDS": "8,24,56,120,248", "DFLASH_WARMUP_MAX_TOKENS": "16",
    "DFLASH_WARMUP_TIMEOUT_SECONDS": "600", "SPARKRING_WARMUP_TEMPERATURE": "0",
    "SPARKRING_LIVENESS_ENABLED": "1", "SPARKRING_LIVENESS_PORT": "8016",
    "SPARKRING_LIVENESS_BLOCKED_SECONDS": "60", "SPARKRING_LIVENESS_OUTPUT_SECONDS": "300",
    "SPARKRING_IDLE_KV_WARN_SECONDS": "330", "SPARKRING_LIVENESS_STALE_SECONDS": "15",
    "SPARKRING_LIVENESS_SAMPLE_SECONDS": "10", "SPARK_TP4_GID0": "3", "SPARK_TP4_GID1": "3",
    "SPARK_TP4_GRAPH_CONTROL_PORT0": "9970", "SPARK_TP4_GRAPH_CONTROL_PORT1": "9971",
    "SPARK_TP4_GRAPH_SUBMIT_CPU": "10", "SPARK_TP4_GRAPH_PROGRESS_CPU": "11",
    "SPARK_TP4_MAX_INFLIGHT": "64", "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS": "10",
    "SPARK_TP4_GRAPH_DIRECT_DOORBELL": "0", "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0": "19000",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1": "19001",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0": "3",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1": "3",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0": "19100",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1": "19101",
    "SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS": "120",
    "SPARK_CUDAGRAPH_REPLAY_TIMING": "0", "SPARK_CONTEXT_CACHE_TRACE_REUSE": "0",
    "SPARKCACHE_ACCESS_MODE": "read-write", "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS": "300",
    "SPARKCACHE_PUBLICATION_SCHEMA": "tail-cow-v2", "SPARKCACHE_CLEAR_ONCE": "auto",
    "SPARKCACHE_MAX_BYTES": "42949672960", "SPARKCACHE_LOW_WATERMARK_BYTES": "34359738368",
    "SPARKCACHE_TTL_SECONDS": "0", "SPARKCACHE_MIN_SPAN_TOKENS": "4096",
    "SPARKCACHE_MAX_SPAN_TOKENS": "1048576", "SPARKCACHE_LOAD_THREADS": "8",
    "SPARKCACHE_MAX_PENDING_RESTORES": "8", "SPARKCACHE_CUDA_RESTORE_IO_WORKERS": "8",
    "SPARKCACHE_CUDA_ARENA_BYTES": "268435456", "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES": "auto",
    "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT": "2", "SPARKCACHE_BUFFER_BUDGET_BYTES": "0",
    "MULTIMODAL_INPUTS": "1", "NCCL_IB_GID_INDEX": "3", "NCCL_MIN_NCHANNELS": "4",
    "NCCL_MAX_NCHANNELS": "4", "NCCL_DEBUG": "WARN", "NCCL_DEBUG_SUBSYS": "NET,INIT,GRAPH",
    "VLLM_BLOCK_SIZE": "256", "OMP_NUM_THREADS": "16", "TORCHINDUCTOR_COMPILE_THREADS": "1",
    "FASTSAFETENSORS_QUEUE_SIZE": "1", "ENABLE_PROMPT_TOKENS_DETAILS": "1",
    "VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT": "0", "R33_PROFILE_CONTRACT_HOST_ROOT": "",
    "API_KEYS_FILE": "", "CHAT_TEMPLATE_HOST_PATH": "", "SIRCL_BUNDLE_HOST_ROOT": "",
    "SPARKCACHE_SOURCE_OVERLAY": "", "VLLM_KV_METRICS_OVERLAY": "",
    "SPARKCACHE_SOURCE_LEASE_CONTRACT": "",
    "SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT": "", "DFLASH_MODEL_HOST_PATH": "",
}
SOURCE_FLAGS = tuple("""
VLLM_B12X_KDA_PREFILL_COALESCING VLLM_GLM53_MHC_PREFILL_SHARD
VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS VLLM_GLM53_KDA_GATE_SIDE_STREAM
VLLM_DCP_TOPK_OWNER_MERGE VLLM_DCP_OWNER_FUSED_ENDPOINTS
VLLM_DCP_COMPACT_INDEX_CACHE_OWNER VLLM_DCP_COMPACT_INDEX_TENSOR_VOTE
VLLM_DCP_COMPACT_INDEX_LOCAL_WIDTHS VLLM_DCP_COMPACT_INDEX_PROFILE
NCCL_IB_EXTENDED_IPV4_GIDS NCCL_IB_PRESERVE_PCI_DOMAIN NCCL_IB_ROUTE_DIAGNOSTICS
""".split())
SIRCL_FIELDS = tuple("""
SPARK_TP4_PEER0 SPARK_TP4_PEER1 SPARK_TP4_DEVICE0 SPARK_TP4_DEVICE1
SPARK_TP4_GID0 SPARK_TP4_GID1 SPARK_TP4_GRAPH_CONTROL_PORT0 SPARK_TP4_GRAPH_CONTROL_PORT1
SPARK_TP4_GRAPH_SUBMIT_CPU SPARK_TP4_GRAPH_PROGRESS_CPU SPARK_TP4_MAX_INFLIGHT
SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS SPARK_TP4_GRAPH_DIRECT_DOORBELL
SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0 SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1
SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0 SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1
SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0 SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1
SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0 SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1
SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0 SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1
SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS
""".split())


def _require(values, key, expected):
    if values.get(key) != str(expected):
        raise ValueError(f"{key} must be {expected} for the selected normal GLM TP4 profile")


def _uint(values, key, *, positive=True, maximum=9223372036854775807):
    value = values.get(key, "")
    if not re.fullmatch(r"0|[1-9][0-9]*", value) or len(value) > 19:
        raise ValueError(f"{key} must be a canonical unsigned decimal integer")
    number = int(value)
    if number < int(positive) or number > maximum:
        raise ValueError(f"{key} is outside its supported integer range")
    return number


def _path(value, key):
    if (not value.startswith("/") or any(c in value for c in ":,\\\r\n\0")
            or ".." in PurePosixPath(value).parts or str(PurePosixPath(value)) != value):
        raise ValueError(f"{key} must be an unambiguous absolute POSIX bind path")


def normalize_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Return effective defaults and auto values without modifying the input.

    Callers may retain this full mapping in their canonical render record; only
    the explicit container environment assembled by ``build_spec`` goes to Docker.
    """
    if not isinstance(environment, Mapping) or any(
        not isinstance(k, str) or not re.fullmatch(r"[A-Z][A-Z0-9_]*", k)
        or not isinstance(v, str) or any(c in v for c in "\0\r\n")
        for k, v in environment.items()
    ):
        raise ValueError("GLM rank environment must contain literal string assignments")
    values = dict(environment)
    for key, value in DEFAULTS.items():
        if key not in values or (not values[key] and key != "SPARKCACHE_CLEAR_ONCE"):
            values[key] = value
    for key in SOURCE_FLAGS:
        values[key] = values.get(key) or "0"
    if values.get("DECODE_CONTEXT_PARALLEL_SIZE") not in ("1", "4"):
        raise ValueError("Normal GLM TP4 profiles support DCP1 or DCP4")
    dcp = int(values["DECODE_CONTEXT_PARALLEL_SIZE"])
    for key, resolved in (
        ("KV_CACHE_MEMORY_BYTES", "25769803776"),
        ("CP_KV_CACHE_INTERLEAVE_SIZE", "1" if dcp == 1 else "4"),
        ("B12X_MLA_CKV_GATHER", "0" if dcp == 1 else "1"),
        ("SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES", "8589934592" if dcp == 1 else "3221225472"),
        ("SPARKCACHE_CLEAR_ONCE", values.get("SPARKCACHE_CACHE_NAMESPACE", "")),
    ):
        if values[key] == "auto":
            values[key] = resolved
    if values.get("SPARKCACHE_ASYNC_PAGE_CAPTURE") == "auto":
        values["SPARKCACHE_ASYNC_PAGE_CAPTURE"] = str(int(
            values.get("SPARKCACHE_ENABLED") == "1" and values["SPARKCACHE_ACCESS_MODE"] == "read-write"))
    return values


def _validate(values, image_record, contract):
    variant = values.get("TARGET_MODEL_VARIANT", glm_targets.DEFAULT)
    glm_targets.require_image(variant, image_record)
    profile = values.get("SOURCE_IMAGE_PROFILE")
    if profile not in PROFILES:
        raise ValueError("Unsupported source-bound GLM TP4 profile")
    selected = contract.get("profiles", {}).get(profile)
    if not isinstance(selected, Mapping) or selected.get("host_domains") != "dual":
        raise ValueError("Verified contract lacks the selected dual-domain TP4 profile")
    schema = image_record.get("schema")
    release = {"sparkring-r35-image-receipt/v1": "r35", "sparkring-candidate-image-receipt/v1": "candidate", glm_source_candidate.SCHEMA: "candidate"}.get(schema)
    if release is None:
        raise ValueError("GLM TP4 structured planning requires a verified R35 or candidate image")
    _require(values, "SPARKRING_RUNTIME_RELEASE", release)
    if schema == glm_source_candidate.SCHEMA:
        glm_source_candidate.validate_profile_capabilities(image_record, profile)
        if contract != glm_source_candidate.contract_for_receipt(image_record):
            raise ValueError("GLM source profile contract differs from admitted source")
        if profile in glm_source_candidate.CACHE_PROFILES:
            namespace = glm_source_candidate.cache_namespace(image_record["image_id"], profile)
            _require(values, "SPARKCACHE_CACHE_NAMESPACE", namespace + "-" + glm_targets.target_for_image(image=image_record)["revision"][:12])
        for key, expected in glm_source_candidate.DISABLED.items():
            _require(values, key, expected)
    for key, field in (("IMAGE_ID", "image_id"), ("IMAGE_REF", "image_reference")):
        if not image_record.get(field):
            raise ValueError("Verified image identity is absent")
        _require(values, key, image_record[field])
    if (not re.fullmatch(r"sha256:[0-9a-f]{64}", values["IMAGE_ID"])
            or image_record.get("platform") != "linux/arm64"):
        raise ValueError("GLM TP4 requires an immutable ARM64 image identity")
    for key in ("SIRCL_BUNDLE_HOST_ROOT", "SPARKCACHE_SOURCE_OVERLAY", "VLLM_KV_METRICS_OVERLAY",
                "R33_PROFILE_CONTRACT_HOST_ROOT", "SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT", "DFLASH_MODEL_HOST_PATH"):
        _require(values, key, "")
    fixed = {
        "SPARKRING_PROFILE_MODE": "custom", "SPARKRING_MANAGED_MESH_RENDERED": "1",
        "SPECULATION_METHOD": "mtp", "TARGET_MODEL_VARIANT": variant,
        "NUM_SPECULATIVE_TOKENS": "3", "DRAFT_TENSOR_PARALLEL_SIZE": "4",
        "PIPELINE_PARALLEL_SIZE": "1", "SIRCL_ENABLED": "1", "VLLM_SPARK_TP4_MODE": "custom",
        "VLLM_SPARK_TP4_VOCAB_MODE": "custom", "VLLM_B12X_KDA_PREFILL_COALESCING": "1",
        "VLLM_GLM53_MHC_PREFILL_SHARD": "1", "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH": "1",
        "NCCL_LIBRARY_PATH": NCCL_PATH, "NCCL_IB_PRESERVE_PCI_DOMAIN": "1",
        "NCCL_DEBUG_SUBSYS": "NET,INIT,GRAPH", "SPARK_CUDAGRAPH_REPLAY_TIMING": "0",
        "SPARK_CONTEXT_CACHE_TRACE_REUSE": "0", "SPARKCACHE_ACCESS_MODE": "read-write",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL": "1",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE": "dual",
        "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE": "fused",
        "CUDAGRAPH_MODE": "FULL_AND_PIECEWISE", "ATTENTION_BACKEND": "B12X",
        "LOAD_FORMAT": "safetensors" if variant == "nvidia-nvfp4" else contract["model"]["loader"]["load_format"],
    }
    for key, value in fixed.items():
        _require(values, key, value)
    for key, field in (("TENSOR_PARALLEL_SIZE", "tensor_parallel_size"),
                       ("NODE_COUNT", "node_count"), ("DECODE_CONTEXT_PARALLEL_SIZE", "decode_context_parallel_size"),
                       ("KV_CACHE_MEMORY_BYTES", "kv_cache_memory_bytes")):
        _require(values, key, selected[field])
    if selected["tensor_parallel_size"] != 4 or selected["node_count"] != 4:
        raise ValueError("Verified profile must select four ranks and TP4")
    _require(values, "MAX_MODEL_LEN", contract["model"]["max_model_len"])
    cache = profile.endswith("-sparkcache")
    if selected.get("sparkcache") is not cache:
        raise ValueError("SparkCache selection differs from the verified profile")
    _require(values, "SPARKCACHE_ENABLED", int(cache))
    _require(values, "SPARKCACHE_ASYNC_PAGE_CAPTURE", int(cache))
    rank = _uint(values, "NODE_RANK", positive=False, maximum=3)
    _require(values, "SPARKRING_NODE_RANK", rank)
    for key in ("HOST_IP", "MASTER_ADDR", "SOCKET_IFNAME", "NCCL_IB_HCA", "SERVED_MODEL_NAME"):
        if not values.get(key) or any(c.isspace() for c in values[key]) or "REPLACE" in values[key]:
            raise ValueError(f"{key} requires a resolved rank value")
    hcas = values["NCCL_IB_HCA"].removeprefix("=").split(",")
    if (not values["NCCL_IB_HCA"].startswith("=") or len(set(hcas)) != 4
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+:1", hca) for hca in hcas)):
        raise ValueError("NCCL_IB_HCA must select the four connected HCA ports exactly")
    for key in ("TARGET_MODEL_HOST_PATH", "CACHE_HOST_ROOT"):
        _path(values.get(key, ""), key)
    if values["TARGET_MODEL_HOST_PATH"] == values["CACHE_HOST_ROOT"]:
        raise ValueError("Model and writable cache paths must differ")
    if values["CHAT_TEMPLATE_HOST_PATH"]:
        _path(values["CHAT_TEMPLATE_HOST_PATH"], "CHAT_TEMPLATE_HOST_PATH")
    for key in ("CONTAINER_PREFIX", "JIT_CACHE_NAMESPACE", "SPARKCACHE_CACHE_NAMESPACE", "SPARKCACHE_CLEAR_ONCE"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", values.get(key, "")):
            raise ValueError(f"{key} requires a resolved namespace or container prefix")
    for key in (*SOURCE_FLAGS, "B12X_MLA_CKV_GATHER", "B12X_FUSED_INDEXER", "MULTIMODAL_INPUTS",
                "JIT_MONITOR_VERBOSE", "ENABLE_PROMPT_TOKENS_DETAILS", "DFLASH_WARMUP",
                "SPARKRING_LIVENESS_ENABLED", "SPARK_TP4_GRAPH_DIRECT_DOORBELL"):
        if values[key] not in ("0", "1"):
            raise ValueError(f"{key} must be 0 or 1")
    if values["NCCL_DEBUG"] not in ("WARN", "INFO"):
        raise ValueError("NCCL_DEBUG must be WARN or INFO")
    for key in """MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS PREFILL_SCHEDULE_INTERVAL
        B12X_MLA_CKV_GATHER_MAX_TOKENS SPARKCACHE_MAX_BYTES SPARKCACHE_MIN_SPAN_TOKENS
        SPARKCACHE_MAX_SPAN_TOKENS SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES
        SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES
        SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS OMP_NUM_THREADS
        TORCHINDUCTOR_COMPILE_THREADS FASTSAFETENSORS_QUEUE_SIZE SPARKRING_LIVENESS_BLOCKED_SECONDS
        SPARKRING_LIVENESS_OUTPUT_SECONDS SPARKRING_IDLE_KV_WARN_SECONDS SPARKRING_LIVENESS_STALE_SECONDS
        SPARKRING_LIVENESS_SAMPLE_SECONDS SPARK_TP4_MAX_INFLIGHT SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS
        SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS DFLASH_WARMUP_MAX_TOKENS DFLASH_WARMUP_TIMEOUT_SECONDS""".split():
        _uint(values, key)
    for key in ("SPARKCACHE_LOW_WATERMARK_BYTES", "SPARKCACHE_TTL_SECONDS", "SPARKCACHE_BUFFER_BUDGET_BYTES",
                "MAX_IMAGES_PER_PROMPT", "MAX_VIDEOS_PER_PROMPT", "VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT"):
        _uint(values, key, positive=False)
    _uint(values, "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS", maximum=300)
    for key in ("PORT", "MASTER_PORT", "SPARKRING_LIVENESS_PORT", "SPARK_TP4_GRAPH_CONTROL_PORT0", "SPARK_TP4_GRAPH_CONTROL_PORT1"):
        _uint(values, key, maximum=65535)
    if values["PORT"] == values["SPARKRING_LIVENESS_PORT"]:
        raise ValueError("API and liveness ports must differ")
    for key in ("SPARK_TP4_GRAPH_SUBMIT_CPU", "SPARK_TP4_GRAPH_PROGRESS_CPU"):
        _uint(values, key, positive=False, maximum=2147483647)
    if values["SPARK_TP4_GRAPH_SUBMIT_CPU"] == values["SPARK_TP4_GRAPH_PROGRESS_CPU"]:
        raise ValueError("SIRCL graph submit and progress CPUs must differ")
    for stem in ("PEER", "DEVICE"):
        endpoints = [values.get(prefix + stem + str(slot), "")
                     for prefix in ("SPARK_TP4_", "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_") for slot in (0, 1)]
        if len(set(endpoints)) != 4 or any(not item or "REPLACE" in item for item in endpoints):
            raise ValueError(f"SIRCL dual-rail {stem.lower()} values must be resolved and distinct")
    for prefix in ("SPARK_TP4_", "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_"):
        for slot in (0, 1):
            _uint(values, prefix + "GID" + str(slot), positive=False, maximum=255)
    _uint(values, "NCCL_IB_GID_INDEX", positive=False)
    for prefix in ("SPARK_TP4_BIDIRECTIONAL_PREFILL_", "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_"):
        for slot in (0, 1):
            _uint(values, prefix + "CONTROL_PORT" + str(slot), maximum=65529)
    interleave = _uint(values, "CP_KV_CACHE_INTERLEAVE_SIZE", maximum=256)
    if 256 % interleave or (values["DECODE_CONTEXT_PARALLEL_SIZE"] == "4" and interleave % 4):
        raise ValueError("CP_KV_CACHE_INTERLEAVE_SIZE violates scheduler geometry")
    if values["VLLM_BLOCK_SIZE"] not in ("256", "512"):
        raise ValueError("VLLM_BLOCK_SIZE must be 256 or 512")
    maximum = _uint(values, "MAX_CUDAGRAPH_CAPTURE_SIZE", maximum=64)
    if maximum % 4 or list(range(4, maximum + 1, 4)) != selected["cudagraph_capture_sizes"]:
        raise ValueError("CUDA graph capture sizes differ from the verified MTP3 profile")
    if "CUDAGRAPH_CAPTURE_SIZES" in values:
        _require(values, "CUDAGRAPH_CAPTURE_SIZES", ",".join(map(str, selected["cudagraph_capture_sizes"])))
    utilization = float(values["GPU_MEMORY_UTILIZATION"])
    if not math.isfinite(utilization) or not 0 < utilization <= 1:
        raise ValueError("GPU_MEMORY_UTILIZATION must be greater than zero and at most one")
    for low, high in (("SPARKCACHE_LOW_WATERMARK_BYTES", "SPARKCACHE_MAX_BYTES"),
                      ("SPARKCACHE_MIN_SPAN_TOKENS", "SPARKCACHE_MAX_SPAN_TOKENS"), ("NCCL_MIN_NCHANNELS", "NCCL_MAX_NCHANNELS")):
        if int(values[low]) > int(values[high]):
            raise ValueError(f"{low} cannot exceed {high}")
    checked = image_record.get("verification", {}).get("checked_files", {})
    for key, digest in (
        ("NCCL_LIBRARY_SHA256", checked.get(NCCL_PATH + ".31.2")),
        ("SPARKRING_DECLARED_SIRCL_NATIVE_SHA256", checked.get(SIRCL_ROOT + "/libspark_transport_capi.so")),
        ("SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256", image_record.get("bundle_manifest_sha256")),
    ):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Verified image does not bind {key}")
        _require(values, key, digest)
    if cache:
        native = contract["sparkcache_native"]
        for stem in ("placement", "snapshot"):
            if checked.get(native[stem + "_path"]) != native[stem + "_sha256"]:
                raise ValueError("Verified image does not bind the SparkCache native contract")
            for suffix in ("path", "sha256"):
                _require(values, f"SPARKCACHE_{stem.upper()}_LIBRARY_{suffix.upper()}", native[stem + "_" + suffix])
        _require(values, "SPARKCACHE_VLLM_ROOT", native["vllm_root"])
        _require(values, "SPARKCACHE_SOURCE_LEASE_CONTRACT", native["lease_contract"])
        for key, value in {
            "SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT": 2, "SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES": 536870912,
            "SPARKCACHE_LOAD_THREADS": 2, "SPARKCACHE_MAX_PENDING_RESTORES": 2,
            "SPARKCACHE_CUDA_RESTORE_IO_WORKERS": 2, "SPARKCACHE_CUDA_ARENA_BYTES": 67108864,
            "SPARKCACHE_BUFFER_BUDGET_BYTES": 1342177280, "SPARKCACHE_MAX_BYTES": 8589934592,
            "SPARKCACHE_LOW_WATERMARK_BYTES": 6442450944, "SPARKCACHE_MIN_SPAN_TOKENS": 4096,
            "SPARKCACHE_MAX_SPAN_TOKENS": 65536, "SPARKCACHE_PUBLICATION_SCHEMA": "tail-cow-v2",
        }.items():
            _require(values, key, value)
    else:
        _require(values, "SPARKCACHE_SOURCE_LEASE_CONTRACT", "")
    return profile, rank, release


def _json(value):
    return json.dumps(value, separators=(",", ":"))


def _cache_config(values, image_record):
    target = glm_targets.target_for_image(values["TARGET_MODEL_VARIANT"], image_record)
    extra = {
        "spark_cache_root": "/cache/jit/sparkcache-context/" + values["SPARKCACHE_CACHE_NAMESPACE"],
        "spark_cache_model_profile": "glm53-flash-hybrid",
        "spark_cache_publication_schema": values["SPARKCACHE_PUBLICATION_SCHEMA"],
        "spark_cache_target_checkpoint_sha256": target["checkpoint_identity"],
        "spark_cache_draft_checkpoint_sha256": target["checkpoint_identity"],
        "spark_cache_draft_policy": "separate", "spark_cache_access_mode": values["SPARKCACHE_ACCESS_MODE"],
        "spark_cache_scheduler_probe": "none", "spark_cache_streaming_snapshots": False,
        "spark_cache_cuda_restore": True, "spark_cache_clear_once": values["SPARKCACHE_CLEAR_ONCE"],
        "spark_cache_async_page_capture": True, "spark_cache_async_page_capture_lease_mode": "connector-jobs",
        "spark_cache_cuda_restore_arena_budget_bytes": 268435456, "spark_cache_page_snapshot_interval_tokens": 0,
    }
    for output, source in {
        "shared_prefix_lease_ttl_seconds": "SHARED_PREFIX_LEASE_TTL_SECONDS",
        "max_bytes": "MAX_BYTES", "low_watermark_bytes": "LOW_WATERMARK_BYTES", "ttl_seconds": "TTL_SECONDS",
        "min_span_tokens": "MIN_SPAN_TOKENS", "max_span_tokens": "MAX_SPAN_TOKENS",
        "cuda_placement_arena_bytes": "CUDA_ARENA_BYTES", "cuda_restore_io_workers": "CUDA_RESTORE_IO_WORKERS",
        "load_threads": "LOAD_THREADS", "max_pending_restores": "MAX_PENDING_RESTORES",
        "async_page_capture_slot_bytes": "ASYNC_CAPTURE_SLOT_BYTES", "async_page_capture_slot_count": "ASYNC_CAPTURE_SLOT_COUNT",
    }.items():
        extra["spark_cache_" + output] = int(values["SPARKCACHE_" + source])
    for output, source in {
        "cuda_placement_library": "PLACEMENT_LIBRARY_PATH", "cuda_placement_library_sha256": "PLACEMENT_LIBRARY_SHA256",
        "async_page_capture_library": "SNAPSHOT_LIBRARY_PATH", "async_page_capture_library_sha256": "SNAPSHOT_LIBRARY_SHA256",
        "async_page_capture_vllm_root": "VLLM_ROOT", "async_page_capture_lease_contract": "SOURCE_LEASE_CONTRACT",
    }.items():
        extra["spark_cache_" + output] = values["SPARKCACHE_" + source]
    return {"kv_connector": "SparkContextCacheConnector", "kv_role": "kv_both", "kv_load_failure_policy": "recompute",
            "kv_connector_module_path": "sparkcache.spark_context_cache_connector", "kv_connector_extra_config": extra}


def build_spec(environment: Mapping[str, str], *, image_record: Mapping, contract: Mapping,
               api_keys: tuple[str, ...] = (), model_config: bytes | None = None,
               model_index: bytes | None = None) -> ContainerSpec:
    """Construct the effective container from a canonical rank and verified inputs.

    Secret-file permission checks and reads belong to the caller. A selected key
    file requires explicit keys, and keys cannot appear without that selection.
    """
    values = normalize_environment(environment)
    profile, rank, release = _validate(values, image_record, contract)
    variant = values["TARGET_MODEL_VARIANT"]
    override = glm_targets.verified_override(variant, model_config, model_index) if variant != glm_targets.DEFAULT else None
    if (not isinstance(api_keys, tuple) or bool(api_keys) != bool(values["API_KEYS_FILE"])
            or any(not isinstance(key, str) or not key or key.startswith("-")
                   or any(c.isspace() or c == "\0" for c in key) for key in api_keys)):
        raise ValueError("API_KEYS_FILE selection requires explicit validated, nonempty API keys")
    env = {
        "VLLM_SPARK_TP4_MODE": "custom", "VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH": "1",
        "VLLM_SPARK_SHARED_CAPTURE_STREAM": "1", "VLLM_SPARK_TP4_GRAPH_Q1": "0",
        "VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40": "0", "SPARK_TP4_CAPABILITY_VOTE": "1",
        "SPARK_TP4_HEALTH_GATE": "1", "SPARK_TP4_FLIGHT_RECORDER": "0",
        "SPARKRING_SIRCL_NATIVE_SHA256": values["SPARKRING_DECLARED_SIRCL_NATIVE_SHA256"],
        "SPARKRING_SIRCL_MANIFEST_SHA256": values["SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256"],
        "SPARK_TP4_GRAPH_STATUS_PATH": f"/cache/jit/sircl-graph-rank{rank}.json",
        "PYTHONPATH": SIRCL_ROOT + "/python", "SPARK_TP4_LIBRARY": SIRCL_ROOT + "/libspark_transport_capi.so",
        "VLLM_SPARK_TP4_VOCAB_MODE": "custom", "SPARK_TP4_CONTROL_PORT0": values["SPARK_TP4_GRAPH_CONTROL_PORT0"],
        "SPARK_TP4_CONTROL_PORT1": values["SPARK_TP4_GRAPH_CONTROL_PORT1"],
        "SPARKRING_NODE_RANK": str(rank), "VLLM_HOST_IP": values["HOST_IP"],
        "VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE": "512", "VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE": "512",
        "VLLM_B12X_MLA_CKV_GATHER": values["B12X_MLA_CKV_GATHER"],
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": values["B12X_MLA_CKV_GATHER_MAX_TOKENS"],
        "XDG_CACHE_HOME": "/cache/jit", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "VLLM_NO_USAGE_STATS": "1", "VLLM_PLUGINS": "", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "CMAKE_CUDA_ARCHITECTURES": "121", "TORCH_CUDA_ARCH_LIST": "12.1a", "CUTE_DSL_ARCH": "sm_121a",
        "FLASHINFER_CUDA_ARCH_LIST": "12.1f", "VLLM_B12X_MOE_FP4_FORCE_A16": "0",
        "VLLM_ENABLE_PCIE_ALLREDUCE": "0", "VLLM_ALLREDUCE_USE_FLASHINFER": "0",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0", "PYTHONUNBUFFERED": "1", "NODE_RANK": str(rank),
        "NCCL_LOCAL_INFERENCE_PATH": NCCL_PATH, "VLLM_NCCL_SO_PATH": NCCL_PATH, "LD_PRELOAD": NCCL_PATH,
        "NCCL_NET": "IB", "NCCL_NET_PLUGIN": "none", "NCCL_IB_DISABLE": "0", "NCCL_IB_SUBNET_AWARE_ROUTING": "1",
        "NCCL_IB_MERGE_NICS": "0", "NCCL_CROSS_NIC": "1", "NCCL_SOCKET_IFNAME": values["SOCKET_IFNAME"],
        "GLOO_SOCKET_IFNAME": values["SOCKET_IFNAME"], "NCCL_P2P_LEVEL": "SYS", "NCCL_PROTO": "LL,LL128,Simple",
        "NCCL_ALGO": "Ring", "NCCL_SWITCHLESS_RING_ONLY": "1", "NCCL_CUMEM_ENABLE": "0",
        "NCCL_IGNORE_CPU_AFFINITY": "1", "VLLM_FASTSAFETENSORS_QUEUE_SIZE": values["FASTSAFETENSORS_QUEUE_SIZE"],
    }
    passthrough = """PORT SERVED_MODEL_NAME DFLASH_WARMUP DFLASH_WARMUP_CONCURRENCIES DFLASH_WARMUP_SHAPE_WORDS
        DFLASH_WARMUP_MAX_TOKENS DFLASH_WARMUP_TIMEOUT_SECONDS SPARKRING_WARMUP_TEMPERATURE
        SPARKRING_LIVENESS_ENABLED SPARKRING_LIVENESS_PORT SPARKRING_LIVENESS_BLOCKED_SECONDS
        SPARKRING_LIVENESS_OUTPUT_SECONDS SPARKRING_IDLE_KV_WARN_SECONDS SPARKRING_LIVENESS_STALE_SECONDS
        SPARKRING_LIVENESS_SAMPLE_SECONDS B12X_FUSED_INDEXER LOAD_FORMAT OMP_NUM_THREADS TORCHINDUCTOR_COMPILE_THREADS
        SOURCE_IMAGE_PROFILE SIRCL_ENABLED SPARKCACHE_ENABLED MASTER_ADDR SPARKRING_PROFILE_MODE
        SPARKRING_MANAGED_MESH_RENDERED VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH R33_PROFILE_CONTRACT_HOST_ROOT
        NCCL_DEBUG NCCL_DEBUG_SUBSYS SPARK_CONTEXT_CACHE_TRACE_REUSE NCCL_IB_HCA NCCL_IB_GID_INDEX
        NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS VLLM_B12X_KDA_PREFILL_COALESCING_LOG_LIMIT
        VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE
        VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE""".split()
    passthrough += list(SOURCE_FLAGS)
    passthrough += list(SIRCL_FIELDS)
    if values["SPARKCACHE_ENABLED"] == "1":
        passthrough += """SPARKCACHE_CACHE_NAMESPACE SPARKCACHE_PLACEMENT_LIBRARY_PATH SPARKCACHE_PLACEMENT_LIBRARY_SHA256
            SPARKCACHE_SNAPSHOT_LIBRARY_PATH SPARKCACHE_SNAPSHOT_LIBRARY_SHA256 SPARKCACHE_VLLM_ROOT
            SPARKCACHE_SOURCE_LEASE_CONTRACT""".split()
    env.update((key, values[key]) for key in passthrough)
    if image_record.get("schema") == glm_source_candidate.SCHEMA:
        env.update(glm_source_candidate.DISABLED)
    for key, leaf in (("VLLM_CACHE_ROOT", "vllm"), ("B12X_CUTE_COMPILE_CACHE_DIR", "b12x"),
                      ("TRITON_CACHE_DIR", "triton"), ("TORCHINDUCTOR_CACHE_DIR", "torchinductor")):
        env[key] = f"/cache/jit/{leaf}/{values['JIT_CACHE_NAMESPACE']}"
    if api_keys:
        env["SPARKRING_WARMUP_API_KEY"] = api_keys[0]
    for key in env.keys() & values.keys():
        if env[key] != values[key]:
            raise ValueError(f"{key} conflicts with the effective source-bound container environment")
    command = ["/opt/sparkring/bin/" + ("sparkring" if release == "r35" else "candidate-image.py"), "serve", "/models/target",
               "--served-model-name", values["SERVED_MODEL_NAME"]]
    if image_record.get("schema") == glm_source_candidate.SCHEMA:
        command[0] = glm_source_candidate.ENTRYPOINT
    if api_keys:
        command.extend(("--api-key", *api_keys))
    command.extend(("--host", "0.0.0.0", "--port", values["PORT"]))
    for option, key in (("tensor-parallel-size", "TENSOR_PARALLEL_SIZE"), ("pipeline-parallel-size", "PIPELINE_PARALLEL_SIZE"),
                        ("decode-context-parallel-size", "DECODE_CONTEXT_PARALLEL_SIZE"), ("cp-kv-cache-interleave-size", "CP_KV_CACHE_INTERLEAVE_SIZE")):
        command.extend(("--" + option, values[key]))
    command.extend(("--distributed-executor-backend", "mp", "--nnodes", "4", "--node-rank", str(rank),
                    "--master-addr", values["MASTER_ADDR"], "--master-port", values["MASTER_PORT"],
                    "--disable-custom-all-reduce", "--mamba-cache-mode", "align"))
    if values["MULTIMODAL_INPUTS"] == "1":
        command.extend(("--limit-mm-per-prompt", _json({"image": int(values["MAX_IMAGES_PER_PROMPT"]), "video": int(values["MAX_VIDEOS_PER_PROMPT"])})))
    else:
        command.append("--language-model-only")
    command.extend(("--mamba-block-size", values["VLLM_BLOCK_SIZE"], "--recurrent-checkpoint-policy", "aligned", "--prefix-cache-retention-interval", "0"))
    mounts = [Bind(values["TARGET_MODEL_HOST_PATH"], "/models/target", True), Bind(values["CACHE_HOST_ROOT"], "/cache/jit")]
    if values["CHAT_TEMPLATE_HOST_PATH"]:
        mounts.append(Bind(values["CHAT_TEMPLATE_HOST_PATH"], "/opt/sparkring/chat_template.jinja", True))
        command.extend(("--chat-template", "/opt/sparkring/chat_template.jinja"))
    command.extend(("--enable-chunked-prefill", "--dtype", "bfloat16", "--kv-cache-dtype", values["KV_CACHE_DTYPE"],
                    "--quantization", "modelopt" if variant == "nvidia-nvfp4" else "modelopt_mixed", "--attention-backend", values["ATTENTION_BACKEND"],
                    "--block-size", values["VLLM_BLOCK_SIZE"], "--moe-backend", values["MOE_BACKEND"],
                    "--linear-backend", values["LINEAR_BACKEND"], "--no-enable-flashinfer-autotune", "--load-format", values["LOAD_FORMAT"],
                    "--enable-auto-tool-choice", "--tool-call-parser", "glm47", "--reasoning-parser", "glm45"))
    if override is not None:
        command.extend(("--hf-overrides", _json(override)))
    for key in ("KDA_PREFILL_BACKEND", "GPU_MEMORY_UTILIZATION", "KV_CACHE_MEMORY_BYTES", "MAX_MODEL_LEN", "MAX_NUM_SEQS",
                "MAX_NUM_BATCHED_TOKENS", "PREFILL_SCHEDULE_INTERVAL"):
        command.extend(("--" + key.lower().replace("_", "-"), values[key]))
    speculation = {"method": "mtp", "num_speculative_tokens": 3, "draft_tensor_parallel_size": 4,
                   "kv_cache_dtype": values["DRAFT_KV_CACHE_DTYPE"], "draft_sample_method": values["DRAFT_SAMPLE_METHOD"],
                   "rejection_sample_method": values["REJECTION_SAMPLE_METHOD"], "draft_load_config": {"load_format": "safetensors"}, "attention_backend": "B12X"}
    compilation = {"cudagraph_mode": values["CUDAGRAPH_MODE"], "cudagraph_capture_sizes": contract["profiles"][profile]["cudagraph_capture_sizes"],
                   "custom_ops": ["all"], "pass_config": {"fuse_allreduce_rms": False}}
    command.extend(("--speculative-config", _json(speculation), "--compilation-config", _json(compilation),
                    "--max-cudagraph-capture-size", values["MAX_CUDAGRAPH_CAPTURE_SIZE"], "--async-scheduling", "--enable-prefix-caching", "--cudagraph-metrics"))
    for key, option in (("JIT_MONITOR_VERBOSE", "--jit-monitor-verbose"), ("ENABLE_PROMPT_TOKENS_DETAILS", "--enable-prompt-tokens-details")):
        if values[key] == "1":
            command.append(option)
    if values["SPARKCACHE_ENABLED"] == "1":
        command.extend(("--kv-transfer-config", _json(_cache_config(values, image_record))))
    if rank:
        command.append("--headless")
    labels = {"org.sparkring.runtime": f"glm53-flash-spark-jovian-{release}-{profile}", "org.sparkring.rank": str(rank),
              "org.sparkring.sircl.native-sha256": values["SPARKRING_DECLARED_SIRCL_NATIVE_SHA256"],
              "org.sparkring.sircl.manifest-sha256": values["SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256"]}
    for label, key in (("sparkcache.enabled", "SPARKCACHE_ENABLED"), ("sparkcache.access-mode", "SPARKCACHE_ACCESS_MODE"),
                       ("sparkcache.shared-prefix-lease-seconds", "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS"),
                       ("multimodal-inputs", "MULTIMODAL_INPUTS"), ("sircl.enabled", "SIRCL_ENABLED"),
                       ("sircl.direct-doorbell", "SPARK_TP4_GRAPH_DIRECT_DOORBELL"),
                       ("sircl.prefill-exposure", "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE"),
                       ("sircl.prefill-rail-mode", "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE")):
        labels["org.sparkring." + label] = values[key]
    shm = re.fullmatch(r"([1-9][0-9]*)([bkmgBKMG]?)", values["SHM_SIZE"])
    if shm is None:
        raise ValueError("SHM_SIZE requires a positive byte count or b/k/m/g suffix")
    shm_bytes = int(shm[1]) * 1024 ** {"": 0, "b": 0, "k": 1, "m": 2, "g": 3}[shm[2].lower()]
    health = f'''python3 -S -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:{values["PORT"]}/health", timeout=4).close()' '''.rstrip()
    return ContainerSpec(name=f"{values['CONTAINER_PREFIX']}-r{rank}", image_id=values["IMAGE_ID"], entrypoint=("/opt/venv/bin/python",),
                         command=tuple(command), environment=env, mounts=tuple(mounts), memory=None, memory_swap=None,
                         shm_size=shm_bytes, cap_add=("IPC_LOCK",), security_opt=("label=disable",),
                         health_mode="shell" if rank == 0 else "inherit", health_command=(health,) if rank == 0 else (),
                         health_interval=10, health_timeout=6, health_start_period=1800, health_retries=3, labels=labels)
