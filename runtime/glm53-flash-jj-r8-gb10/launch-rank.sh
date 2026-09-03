#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: launch-rank.sh RANK [CONFIG_FILE]}"
config_file="${2:-${SPARKRING_CONFIG_FILE:-}}"
if (( $# > 2 )); then
  printf 'usage: launch-rank.sh RANK [CONFIG_FILE]\n' >&2
  exit 2
fi
if [[ -n "${config_file}" ]]; then
  [[ -r "${config_file}" && -f "${config_file}" ]] || {
    printf 'configuration file is not a readable regular file: %s\n' "${config_file}" >&2
    exit 78
  }
  # This is executable shell configuration. Review it before sourcing.
  # shellcheck source=/dev/null
  source "${config_file}"
fi

: "${HOST_IP:?set HOST_IP to this rank's routable address}"
: "${MASTER_ADDR:?set MASTER_ADDR to rank 0's routable address}"
: "${TARGET_MODEL_HOST_PATH:?set TARGET_MODEL_HOST_PATH to the pinned target checkpoint}"
: "${DFLASH_MODEL_HOST_PATH:?set DFLASH_MODEL_HOST_PATH to the pinned BF16 draft checkpoint}"
: "${CACHE_HOST_ROOT:?set CACHE_HOST_ROOT to a dedicated rank-local directory}"

: "${IMAGE_REF:=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:4ce98659c30d9e9c313b1018a2675e5f135a0404e7cc00951b4ade161c0a711f}"
: "${IMAGE_ID:=sha256:c3f85b2350609b6ff1201b8c5998f881ff4cef8b671d6783b543f841040915c0}"
: "${CONTAINER_PREFIX:=glm53-jj-r8-gb10}"
: "${SERVED_MODEL_NAME:=glm-5.3-flash}"
: "${PORT:=8015}"
: "${MASTER_PORT:=29775}"
: "${SHM_SIZE:=32g}"
: "${TENSOR_PARALLEL_SIZE:=4}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${DECODE_CONTEXT_PARALLEL_SIZE:=4}"
: "${CP_KV_CACHE_INTERLEAVE_SIZE:=auto}"
: "${B12X_MLA_CKV_GATHER:=auto}"
: "${B12X_MLA_CKV_GATHER_MAX_TOKENS:=524288}"
: "${NODE_COUNT:=4}"
: "${MAX_MODEL_LEN:=1048576}"
: "${MAX_NUM_SEQS:=16}"
: "${MAX_NUM_BATCHED_TOKENS:=8192}"
: "${PREFILL_SCHEDULE_INTERVAL:=2}"
: "${MAX_IMAGES_PER_PROMPT:=4}"
: "${MAX_VIDEOS_PER_PROMPT:=1}"
: "${KV_CACHE_MEMORY_BYTES:=auto}"
: "${GPU_MEMORY_UTILIZATION:=0.80}"
: "${KV_CACHE_DTYPE:=fp8}"
: "${SPECULATION_METHOD:=dflash}"
: "${NUM_SPECULATIVE_TOKENS:=7}"
: "${DRAFT_TENSOR_PARALLEL_SIZE:=4}"
: "${DRAFT_KV_CACHE_DTYPE:=auto}"
: "${DRAFT_SAMPLE_METHOD:=probabilistic}"
: "${REJECTION_SAMPLE_METHOD:=standard}"
: "${ATTENTION_BACKEND:=B12X}"
: "${MOE_BACKEND:=b12x}"
: "${LINEAR_BACKEND:=b12x}"
: "${KDA_PREFILL_BACKEND:=flashkda}"
: "${LOAD_FORMAT:=fastsafetensors}"
: "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}"
: "${MAX_CUDAGRAPH_CAPTURE_SIZE:=128}"
: "${SPARKCACHE_CACHE_NAMESPACE:=glm53-flash-dcp4-page-tail-cow-v2}"
: "${JIT_CACHE_NAMESPACE:=glm53-flash-sm121-vllm-22ffe140-b12x-6255090a}"
: "${JIT_MONITOR_VERBOSE:=0}"
: "${DFLASH_WARMUP:=0}"
: "${DFLASH_WARMUP_CONCURRENCIES:=1,2,4,8,16}"
: "${DFLASH_WARMUP_SHAPE_WORDS:=8,24,56,120,248}"
: "${DFLASH_WARMUP_MAX_TOKENS:=16}"
: "${DFLASH_WARMUP_TIMEOUT_SECONDS:=600}"
: "${SPARKCACHE_ENABLED:=1}"
: "${SPARKCACHE_ACCESS_MODE:=read-write}"
: "${SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS:=300}"
: "${SPARKCACHE_PUBLICATION_SCHEMA:=tail-cow-v2}"
: "${SPARKCACHE_CLEAR_ONCE:=auto}"
: "${SPARKCACHE_MAX_BYTES:=42949672960}"
: "${SPARKCACHE_LOW_WATERMARK_BYTES:=34359738368}"
: "${SPARKCACHE_TTL_SECONDS:=0}"
: "${SPARKCACHE_MIN_SPAN_TOKENS:=4096}"
: "${SPARKCACHE_MAX_SPAN_TOKENS:=1048576}"
: "${SPARKCACHE_LOAD_THREADS:=8}"
: "${SPARKCACHE_MAX_PENDING_RESTORES:=8}"
: "${SPARKCACHE_CUDA_RESTORE_IO_WORKERS:=8}"
: "${SPARKCACHE_CUDA_ARENA_BYTES:=268435456}"
: "${SPARKCACHE_ASYNC_PAGE_CAPTURE:=0}"
: "${SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES:=auto}"
: "${SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT:=2}"
: "${SPARKCACHE_SOURCE_OVERLAY:=}"
: "${VLLM_KV_METRICS_OVERLAY:=}"
: "${MULTIMODAL_INPUTS:=1}"
: "${SOCKET_IFNAME:=enP7s7}"
: "${NCCL_IB_HCA:=rocep1s0f0,rocep1s0f1}"
: "${NCCL_IB_GID_INDEX:=3}"
: "${NCCL_MIN_NCHANNELS:=4}"
: "${NCCL_MAX_NCHANNELS:=4}"
: "${OMP_NUM_THREADS:=16}"
: "${TORCHINDUCTOR_COMPILE_THREADS:=1}"
: "${FASTSAFETENSORS_QUEUE_SIZE:=1}"
: "${ENABLE_PROMPT_TOKENS_DETAILS:=1}"
: "${API_KEYS_FILE:=}"

die() {
  printf '%s\n' "$*" >&2
  exit 78
}

require_uint() {
  local name="$1" value="${!1}"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an unsigned integer"
}

require_positive_uint() {
  require_uint "$1"
  (( ${!1} > 0 )) || die "$1 must be greater than zero"
}

for name in \
  PORT MASTER_PORT TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE \
  DECODE_CONTEXT_PARALLEL_SIZE NODE_COUNT MAX_MODEL_LEN MAX_NUM_SEQS \
  MAX_NUM_BATCHED_TOKENS PREFILL_SCHEDULE_INTERVAL \
  NUM_SPECULATIVE_TOKENS \
  DRAFT_TENSOR_PARALLEL_SIZE MAX_CUDAGRAPH_CAPTURE_SIZE \
  B12X_MLA_CKV_GATHER_MAX_TOKENS \
  SPARKCACHE_MAX_BYTES SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS \
  SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES \
  SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS \
  SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES \
  SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT \
  NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS OMP_NUM_THREADS \
  TORCHINDUCTOR_COMPILE_THREADS FASTSAFETENSORS_QUEUE_SIZE
do
  require_positive_uint "${name}"
done
require_uint SPARKCACHE_LOW_WATERMARK_BYTES
require_uint SPARKCACHE_TTL_SECONDS
require_uint NCCL_IB_GID_INDEX
require_uint MAX_IMAGES_PER_PROMPT
require_uint MAX_VIDEOS_PER_PROMPT
(( SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS <= 300 )) || \
  die 'SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS must be between 1 and 300'

case "${DECODE_CONTEXT_PARALLEL_SIZE}" in
  1|2|4) ;;
  *) die 'DECODE_CONTEXT_PARALLEL_SIZE must be 1, 2, or 4 for this profile' ;;
esac
(( TENSOR_PARALLEL_SIZE % DECODE_CONTEXT_PARALLEL_SIZE == 0 )) || \
  die 'DECODE_CONTEXT_PARALLEL_SIZE must divide TENSOR_PARALLEL_SIZE'

if [[ "${KV_CACHE_MEMORY_BYTES}" == auto ]]; then
  if (( DECODE_CONTEXT_PARALLEL_SIZE == 1 )); then
    # This reservation completed a 942,767-token request without host OOM.
    KV_CACHE_MEMORY_BYTES=27917287424
  elif (( DECODE_CONTEXT_PARALLEL_SIZE == 2 )); then
    # DCP2 retains the recorded 30 GiB capacity configuration.
    KV_CACHE_MEMORY_BYTES=32212254720
  else
    # DCP4 uses 24 GiB to retain host-memory headroom under concurrent serving.
    KV_CACHE_MEMORY_BYTES=25769803776
  fi
else
  require_positive_uint KV_CACHE_MEMORY_BYTES
fi

if [[ "${SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES}" == auto ]]; then
  if (( DECODE_CONTEXT_PARALLEL_SIZE == 1 )); then
    SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES=8589934592
  elif (( DECODE_CONTEXT_PARALLEL_SIZE == 2 )); then
    SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES=5368709120
  else
    SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES=3221225472
  fi
else
  require_positive_uint SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES
fi

if [[ "${CP_KV_CACHE_INTERLEAVE_SIZE}" == auto ]]; then
  if (( DECODE_CONTEXT_PARALLEL_SIZE == 1 )); then
    CP_KV_CACHE_INTERLEAVE_SIZE=1
  else
    CP_KV_CACHE_INTERLEAVE_SIZE=4
  fi
else
  require_positive_uint CP_KV_CACHE_INTERLEAVE_SIZE
fi
(( CP_KV_CACHE_INTERLEAVE_SIZE <= 256 && 256 % CP_KV_CACHE_INTERLEAVE_SIZE == 0 )) || \
  die 'CP_KV_CACHE_INTERLEAVE_SIZE must divide the 256-token scheduler block size'
if (( DECODE_CONTEXT_PARALLEL_SIZE > 1 && CP_KV_CACHE_INTERLEAVE_SIZE % 4 != 0 )); then
  die 'GLM-5.3 DCP2/DCP4 requires CP_KV_CACHE_INTERLEAVE_SIZE divisible by 4'
fi
if [[ "${B12X_MLA_CKV_GATHER}" == auto ]]; then
  if (( DECODE_CONTEXT_PARALLEL_SIZE == 1 )); then
    B12X_MLA_CKV_GATHER=0
  else
    B12X_MLA_CKV_GATHER=1
  fi
fi
case "${B12X_MLA_CKV_GATHER}" in
  0|1) ;;
  *) die 'B12X_MLA_CKV_GATHER must be auto, 0, or 1' ;;
esac

[[ "${rank}" =~ ^[0-9]+$ ]] || die 'rank must be an unsigned integer'
(( rank < NODE_COUNT )) || die "rank must be between 0 and $((NODE_COUNT - 1))"
(( PORT <= 65535 && MASTER_PORT <= 65535 )) || die 'ports must be at most 65535'
(( SPARKCACHE_LOW_WATERMARK_BYTES <= SPARKCACHE_MAX_BYTES )) || \
  die 'SPARKCACHE_LOW_WATERMARK_BYTES cannot exceed SPARKCACHE_MAX_BYTES'
(( SPARKCACHE_MIN_SPAN_TOKENS <= SPARKCACHE_MAX_SPAN_TOKENS )) || \
  die 'SPARKCACHE_MIN_SPAN_TOKENS cannot exceed SPARKCACHE_MAX_SPAN_TOKENS'
(( NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS )) || \
  die 'NCCL_MIN_NCHANNELS cannot exceed NCCL_MAX_NCHANNELS'
[[ "${CONTAINER_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  die 'CONTAINER_PREFIX is not a valid Docker container-name prefix'
[[ "${SPECULATION_METHOD}" == dflash ]] || \
  die 'this runtime launcher supports SPECULATION_METHOD=dflash'
case "${SPARKCACHE_PUBLICATION_SCHEMA}" in
  snapshot-v1|tail-cow-v1|tail-cow-v2) ;;
  *) die 'SPARKCACHE_PUBLICATION_SCHEMA must be snapshot-v1, tail-cow-v1, or tail-cow-v2' ;;
esac
case "${SPARKCACHE_ENABLED}" in
  0|1) ;;
  *) die 'SPARKCACHE_ENABLED must be 0 or 1' ;;
esac
case "${ENABLE_PROMPT_TOKENS_DETAILS}" in
  0|1) ;;
  *) die 'ENABLE_PROMPT_TOKENS_DETAILS must be 0 or 1' ;;
esac
case "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" in
  0|1) ;;
  *) die 'SPARKCACHE_ASYNC_PAGE_CAPTURE must be 0 or 1' ;;
esac
case "${SPARKCACHE_ACCESS_MODE}" in
  read-write|restore-only|store-only|disabled) ;;
  *) die 'SPARKCACHE_ACCESS_MODE must be read-write, restore-only, store-only, or disabled' ;;
esac
if [[ "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" == 1 ]]; then
  [[ "${SPARKCACHE_ENABLED}" == 1 ]] || \
    die 'asynchronous page capture requires SPARKCACHE_ENABLED=1'
  case "${SPARKCACHE_ACCESS_MODE}" in
    read-write|store-only) ;;
    *) die 'asynchronous page capture requires a publication-capable access mode' ;;
  esac
fi
case "${MULTIMODAL_INPUTS}" in
  0|1) ;;
  *) die 'MULTIMODAL_INPUTS must be 0 (text only) or 1 (images and video)' ;;
esac
case "${JIT_MONITOR_VERBOSE}" in
  0|1) ;;
  *) die 'JIT_MONITOR_VERBOSE must be 0 or 1' ;;
esac
case "${DFLASH_WARMUP}" in
  0|1) ;;
  *) die 'DFLASH_WARMUP must be 0 or 1' ;;
esac
for name in SPARKCACHE_CACHE_NAMESPACE SPARKCACHE_CLEAR_ONCE JIT_CACHE_NAMESPACE
do
  [[ "${!name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "${name} must contain only letters, digits, dot, underscore, or hyphen"
done
api_key_args=()
api_keys=()
if [[ -n "${API_KEYS_FILE}" ]]; then
  [[ -f "${API_KEYS_FILE}" && -r "${API_KEYS_FILE}" ]] || \
    die "API_KEYS_FILE is not a readable regular file: ${API_KEYS_FILE}"
  api_keys_file_mode="$(stat -c %a -- "${API_KEYS_FILE}" 2>/dev/null \
    || stat -f %Lp -- "${API_KEYS_FILE}" 2>/dev/null || true)"
  [[ "${api_keys_file_mode}" == 600 ]] || \
    die "API_KEYS_FILE must be mode 0600, got 0${api_keys_file_mode:-???}"
  mapfile -t api_keys < <(awk 'NF {print}' "${API_KEYS_FILE}")
  (( ${#api_keys[@]} > 0 )) || die 'API_KEYS_FILE contains no non-empty keys'
  for api_key in "${api_keys[@]}"; do
    [[ "${api_key}" != *[[:space:]]* ]] || \
      die 'API_KEYS_FILE contains whitespace in a key'
  done
  # vLLM parses --api-key with nargs="+"; keep every key in one occurrence so a
  # later option cannot truncate the accepted set.
  api_key_args=(--api-key "${api_keys[@]}")
fi
warmup_api_key_env=()
if (( ${#api_keys[@]} > 0 )); then
  warmup_api_key_env=(-e "SPARKRING_WARMUP_API_KEY=${api_keys[0]}")
fi
for name in TARGET_MODEL_HOST_PATH DFLASH_MODEL_HOST_PATH CACHE_HOST_ROOT; do
  value="${!name}"
  [[ "${value}" == /* ]] || die "${name} must be an absolute host path"
  [[ "${value}" != *:* && "${value}" != *$'\n'* ]] || \
    die "${name} cannot be represented safely as a Docker bind mount"
done
sparkcache_source_args=()
if [[ -n "${SPARKCACHE_SOURCE_OVERLAY}" ]]; then
  [[ "${SPARKCACHE_SOURCE_OVERLAY}" == /* ]] || \
    die 'SPARKCACHE_SOURCE_OVERLAY must be an absolute host path'
  [[ -d "${SPARKCACHE_SOURCE_OVERLAY}" ]] || \
    die 'SPARKCACHE_SOURCE_OVERLAY must be a directory'
  sparkcache_source_args=(
    -v
    "${SPARKCACHE_SOURCE_OVERLAY}:/usr/local/lib/python3.12/dist-packages/sparkcache:ro"
  )
fi
vllm_metrics_args=()
if [[ -n "${VLLM_KV_METRICS_OVERLAY}" ]]; then
  [[ "${VLLM_KV_METRICS_OVERLAY}" == /* ]] || \
    die 'VLLM_KV_METRICS_OVERLAY must be an absolute host path'
  [[ -f "${VLLM_KV_METRICS_OVERLAY}" ]] || \
    die 'VLLM_KV_METRICS_OVERLAY must be a regular file'
  vllm_metrics_args=(
    -v
    "${VLLM_KV_METRICS_OVERLAY}:/usr/local/lib/python3.12/dist-packages/vllm/distributed/kv_transfer/kv_connector/v1/metrics.py:ro"
  )
fi
command -v python3 >/dev/null 2>&1 || die 'python3 is required to encode JSON configuration safely'
command -v sha256sum >/dev/null 2>&1 || die 'sha256sum is required to verify model inputs'
python3 - "${GPU_MEMORY_UTILIZATION}" <<'PY' || exit 78
import math
import sys

try:
    value = float(sys.argv[1])
except ValueError:
    print("GPU_MEMORY_UTILIZATION must be a number", file=sys.stderr)
    raise SystemExit(1)
if not math.isfinite(value) or not 0 < value <= 1:
    print("GPU_MEMORY_UTILIZATION must be greater than zero and at most one", file=sys.stderr)
    raise SystemExit(1)
PY

[[ -n "${IMAGE_REF}" && "${IMAGE_REF}" != *[[:space:]]* ]] || \
  die 'IMAGE_REF must not contain whitespace'
[[ "${IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]] || \
  die 'IMAGE_ID must be an immutable local image ID'
if [[ "${SPARKCACHE_CLEAR_ONCE}" == auto ]]; then
  SPARKCACHE_CLEAR_ONCE="${SPARKCACHE_CACHE_NAMESPACE}"
fi

actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_REF}")"
[[ "${actual_image_id}" == "${IMAGE_ID}" ]] || \
  die "image identity mismatch: expected ${IMAGE_ID}, got ${actual_image_id}"
for directory in "${TARGET_MODEL_HOST_PATH}" "${DFLASH_MODEL_HOST_PATH}" "${CACHE_HOST_ROOT}"; do
  [[ -d "${directory}" ]] || die "required directory is missing: ${directory}"
done

verify_file_sha256() {
  local role="$1" path="$2" expected="$3" actual
  [[ -f "${path}" ]] || die "${role} is missing: ${path}"
  actual="$(sha256sum -- "${path}")"
  actual="${actual%% *}"
  [[ "${actual}" == "${expected}" ]] || \
    die "${role} identity mismatch: expected ${expected}, got ${actual}"
}
verify_file_sha256 \
  'target config.json' \
  "${TARGET_MODEL_HOST_PATH}/config.json" \
  '676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996'
verify_file_sha256 \
  'target model.safetensors.index.json' \
  "${TARGET_MODEL_HOST_PATH}/model.safetensors.index.json" \
  '0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb'
verify_file_sha256 \
  'draft config.json' \
  "${DFLASH_MODEL_HOST_PATH}/config.json" \
  'c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573'
verify_file_sha256 \
  'draft model.safetensors' \
  "${DFLASH_MODEL_HOST_PATH}/model.safetensors" \
  'b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b'

container="${CONTAINER_PREFIX}-r${rank}"
if docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'container already exists: %s\n' "${container}" >&2
  exit 3
fi

export NUM_SPECULATIVE_TOKENS DRAFT_TENSOR_PARALLEL_SIZE DRAFT_KV_CACHE_DTYPE
export DRAFT_SAMPLE_METHOD REJECTION_SAMPLE_METHOD
speculative_config="$(python3 - <<'PY'
import json
import os

print(json.dumps({
    "method": "dflash",
    "model": "/dflash-draft",
    "num_speculative_tokens": int(os.environ["NUM_SPECULATIVE_TOKENS"]),
    "draft_tensor_parallel_size": int(os.environ["DRAFT_TENSOR_PARALLEL_SIZE"]),
    "kv_cache_dtype": os.environ["DRAFT_KV_CACHE_DTYPE"],
    "draft_sample_method": os.environ["DRAFT_SAMPLE_METHOD"],
    "rejection_sample_method": os.environ["REJECTION_SAMPLE_METHOD"],
    "draft_load_config": {"load_format": "safetensors"},
}, separators=(",", ":")))
PY
)"

export CUDAGRAPH_MODE MAX_CUDAGRAPH_CAPTURE_SIZE
compilation_config="$(python3 - <<'PY'
import json
import os

maximum = int(os.environ["MAX_CUDAGRAPH_CAPTURE_SIZE"])
capture_sizes = []
size = 8
while size < maximum:
    capture_sizes.append(size)
    size *= 2
capture_sizes.append(maximum)
print(json.dumps({
    "cudagraph_mode": os.environ["CUDAGRAPH_MODE"],
    "cudagraph_capture_sizes": capture_sizes,
    "custom_ops": ["all"],
    "pass_config": {"fuse_allreduce_rms": False},
}, separators=(",", ":")))
PY
)"

kv_transfer_args=()
if [[ "${SPARKCACHE_ENABLED}" == 1 ]]; then
  export SPARKCACHE_CACHE_NAMESPACE SPARKCACHE_CLEAR_ONCE SPARKCACHE_MAX_BYTES
  export SPARKCACHE_ACCESS_MODE
  export SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS
  export SPARKCACHE_PUBLICATION_SCHEMA
  export SPARKCACHE_LOW_WATERMARK_BYTES SPARKCACHE_TTL_SECONDS
  export SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS
  export SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES
  export SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES
  export SPARKCACHE_ASYNC_PAGE_CAPTURE
  export SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT
  kv_transfer_config="$(python3 - <<'PY'
import json
import os

def integer(name: str) -> int:
    return int(os.environ[name])

extra = {
    "spark_cache_root": f"/cache/jit/sparkcache-context/{os.environ['SPARKCACHE_CACHE_NAMESPACE']}",
    "spark_cache_model_profile": "glm53-flash-hybrid",
    "spark_cache_publication_schema": os.environ["SPARKCACHE_PUBLICATION_SCHEMA"],
    "spark_cache_target_checkpoint_sha256": "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9",
    "spark_cache_draft_checkpoint_sha256": "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b",
    "spark_cache_draft_policy": "separate",
    "spark_cache_access_mode": os.environ["SPARKCACHE_ACCESS_MODE"],
    "spark_cache_shared_prefix_lease_ttl_seconds": integer(
        "SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS"
    ),
    "spark_cache_scheduler_probe": "none",
    "spark_cache_streaming_snapshots": False,
    "spark_cache_cuda_restore": True,
    "spark_cache_max_bytes": integer("SPARKCACHE_MAX_BYTES"),
    "spark_cache_low_watermark_bytes": integer("SPARKCACHE_LOW_WATERMARK_BYTES"),
    "spark_cache_ttl_seconds": integer("SPARKCACHE_TTL_SECONDS"),
    "spark_cache_min_span_tokens": integer("SPARKCACHE_MIN_SPAN_TOKENS"),
    "spark_cache_max_span_tokens": integer("SPARKCACHE_MAX_SPAN_TOKENS"),
    "spark_cache_cuda_placement_library": "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so",
    "spark_cache_cuda_placement_library_sha256": "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
    "spark_cache_cuda_placement_arena_bytes": integer("SPARKCACHE_CUDA_ARENA_BYTES"),
    "spark_cache_cuda_restore_io_workers": integer("SPARKCACHE_CUDA_RESTORE_IO_WORKERS"),
    "spark_cache_load_threads": integer("SPARKCACHE_LOAD_THREADS"),
    "spark_cache_max_pending_restores": integer("SPARKCACHE_MAX_PENDING_RESTORES"),
    "spark_cache_clear_once": os.environ["SPARKCACHE_CLEAR_ONCE"],
    "spark_cache_async_page_capture": os.environ["SPARKCACHE_ASYNC_PAGE_CAPTURE"] == "1",
    "spark_cache_async_page_capture_library": "/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_snapshot.so",
    "spark_cache_async_page_capture_library_sha256": "4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5",
    "spark_cache_async_page_capture_slot_bytes": integer("SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES"),
    "spark_cache_async_page_capture_slot_count": integer("SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT"),
    "spark_cache_async_page_capture_vllm_root": "/usr/local/lib/python3.12/dist-packages",
    "spark_cache_async_page_capture_lease_contract": "/usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-manager-page-async-contract-55969c16.json",
}
print(json.dumps({
    "kv_connector": "SparkContextCacheConnector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
    "kv_connector_extra_config": extra,
}, separators=(",", ":")))
PY
)"
  kv_transfer_args=(--kv-transfer-config "${kv_transfer_config}")
fi

# Text-only mode avoids loading the vision tower. Multimodal mode uses the
# independently configurable image and video request limits.
multimodal_args=(--language-model-only)
if [[ "${MULTIMODAL_INPUTS}" == 1 ]]; then
  multimodal_args=(
    --limit-mm-per-prompt
    "{\"image\":${MAX_IMAGES_PER_PROMPT},\"video\":${MAX_VIDEOS_PER_PROMPT}}"
  )
fi

headless=()
[[ "${rank}" == 0 ]] || headless=(--headless)

prompt_tokens_details=()
if [[ "${ENABLE_PROMPT_TOKENS_DETAILS}" == 1 ]]; then
  prompt_tokens_details=(--enable-prompt-tokens-details)
fi
jit_monitor_args=()
if [[ "${JIT_MONITOR_VERBOSE}" == 1 ]]; then
  jit_monitor_args=(--jit-monitor-verbose)
fi

container_id="$(docker run -d \
  --name "${container}" \
  --entrypoint /opt/sparkring/bin/serve-with-warmup.py \
  --network host --ipc host --shm-size "${SHM_SIZE}" --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  --security-opt label=disable --init \
  -v "${TARGET_MODEL_HOST_PATH}:/models/target:ro" \
  -v "${DFLASH_MODEL_HOST_PATH}:/dflash-draft:ro" \
  -v "${CACHE_HOST_ROOT}:/cache/jit" \
  "${sparkcache_source_args[@]}" \
  "${vllm_metrics_args[@]}" \
  -e "SPARKRING_NODE_RANK=${rank}" \
  -e "PORT=${PORT}" -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  -e "DFLASH_WARMUP=${DFLASH_WARMUP}" \
  -e "DFLASH_WARMUP_CONCURRENCIES=${DFLASH_WARMUP_CONCURRENCIES}" \
  -e "DFLASH_WARMUP_SHAPE_WORDS=${DFLASH_WARMUP_SHAPE_WORDS}" \
  -e "DFLASH_WARMUP_MAX_TOKENS=${DFLASH_WARMUP_MAX_TOKENS}" \
  -e "DFLASH_WARMUP_TIMEOUT_SECONDS=${DFLASH_WARMUP_TIMEOUT_SECONDS}" \
  "${warmup_api_key_env[@]}" \
  -e "VLLM_HOST_IP=${HOST_IP}" \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512 \
  -e "VLLM_B12X_MLA_CKV_GATHER=${B12X_MLA_CKV_GATHER}" \
  -e "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=${B12X_MLA_CKV_GATHER_MAX_TOKENS}" \
  -e "VLLM_CACHE_ROOT=/cache/jit/vllm/${JIT_CACHE_NAMESPACE}" \
  -e "B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x/${JIT_CACHE_NAMESPACE}" \
  -e "TRITON_CACHE_DIR=/cache/jit/triton/${JIT_CACHE_NAMESPACE}" \
  -e "TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor/${JIT_CACHE_NAMESPACE}" \
  -e XDG_CACHE_HOME=/cache/jit -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 -e VLLM_PLUGINS= \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CMAKE_CUDA_ARCHITECTURES=121 -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a -e FLASHINFER_CUDA_ARCH_LIST=12.1f \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2 \
  -e LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2 \
  -e NCCL_DEBUG=WARN -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none \
  -e NCCL_IB_DISABLE=0 -e "NCCL_IB_HCA=${NCCL_IB_HCA}" \
  -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_IB_MERGE_NICS=0 -e NCCL_CROSS_NIC=1 \
  -e "NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}" -e "GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}" \
  -e NCCL_P2P_LEVEL=SYS -e NCCL_PROTO=LL,LL128,Simple -e NCCL_ALGO=Ring \
  -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}" \
  -e "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}" \
  -e NCCL_SWITCHLESS_RING_ONLY=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e "VLLM_FASTSAFETENSORS_QUEUE_SIZE=${FASTSAFETENSORS_QUEUE_SIZE}" \
  --label org.sparkring.runtime=glm53-jj-r8-gb10-sparkcache \
  --label org.sparkring.sparkcache.enabled="${SPARKCACHE_ENABLED}" \
  --label org.sparkring.sparkcache.access-mode="${SPARKCACHE_ACCESS_MODE}" \
  --label org.sparkring.sparkcache.shared-prefix-lease-seconds="${SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS}" \
  --label org.sparkring.rank="${rank}" \
  --label org.sparkring.multimodal-inputs="${MULTIMODAL_INPUTS}" \
  "${IMAGE_REF}" \
  /models/target \
  --served-model-name "${SERVED_MODEL_NAME}" "${api_key_args[@]}" \
  --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}" \
  --cp-kv-cache-interleave-size "${CP_KV_CACHE_INTERLEAVE_SIZE}" \
  --distributed-executor-backend mp --nnodes "${NODE_COUNT}" --node-rank "${rank}" \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}" \
  --disable-custom-all-reduce --mamba-cache-mode align "${multimodal_args[@]}" \
  --enable-chunked-prefill --dtype bfloat16 --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --quantization modelopt_mixed --attention-backend "${ATTENTION_BACKEND}" \
  --block-size 256 --moe-backend "${MOE_BACKEND}" --linear-backend "${LINEAR_BACKEND}" \
  --no-enable-flashinfer-autotune --load-format "${LOAD_FORMAT}" \
  --enable-auto-tool-choice --tool-call-parser glm47 --reasoning-parser glm45 \
  --kda-prefill-backend "${KDA_PREFILL_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --prefill-schedule-interval "${PREFILL_SCHEDULE_INTERVAL}" \
  --speculative-config "${speculative_config}" \
  --compilation-config "${compilation_config}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --async-scheduling --enable-prefix-caching --cudagraph-metrics \
  "${jit_monitor_args[@]}" \
  "${prompt_tokens_details[@]}" \
  "${kv_transfer_args[@]}" "${headless[@]}")"

if [[ "${rank}" == 0 && "${DFLASH_WARMUP}" == 1 ]]; then
  readiness_deadline=$((SECONDS + DFLASH_WARMUP_TIMEOUT_SECONDS + 120))
  while true; do
    health="$(docker inspect --format '{{.State.Health.Status}}' "${container}" 2>/dev/null || true)"
    [[ "${health}" == healthy ]] && break
    state="$(docker inspect --format '{{.State.Status}}' "${container}" 2>/dev/null || true)"
    if [[ "${health}" == unhealthy || "${state}" == exited || "${state}" == dead ]]; then
      die "rank-0 engine readiness failed: state=${state:-unknown} health=${health:-unknown}"
    fi
    (( SECONDS < readiness_deadline )) || \
      die 'rank-0 engine readiness timed out'
    sleep 1
  done
fi
printf '%s\n' "${container_id}"
