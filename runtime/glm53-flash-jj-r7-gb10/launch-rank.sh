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

: "${IMAGE_VARIANT:=base}"
: "${BASE_IMAGE_REF:=ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0}"
: "${SPARKCACHE_IMAGE_REF:=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5}"
: "${CONTAINER_PREFIX:=glm53-jj-r7-gb10}"
: "${SERVED_MODEL_NAME:=glm-5.3-flash-nvfp4-dflash2-bf16-t7-jj-r7-gb10}"
: "${PORT:=8015}"
: "${MASTER_PORT:=29775}"
: "${SHM_SIZE:=32g}"
: "${TENSOR_PARALLEL_SIZE:=4}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${DECODE_CONTEXT_PARALLEL_SIZE:=1}"
: "${NODE_COUNT:=4}"
: "${MAX_MODEL_LEN:=262144}"
: "${MAX_NUM_SEQS:=16}"
: "${MAX_NUM_BATCHED_TOKENS:=4096}"
: "${KV_CACHE_MEMORY_BYTES:=21474836480}"
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
: "${BASE_CACHE_NAMESPACE:=jj-r7-gb10-base-v1}"
: "${SPARKCACHE_CACHE_NAMESPACE:=jj-r7-gb10-page-tail-cow-v1}"
: "${SPARKCACHE_CLEAR_ONCE:=${SPARKCACHE_CACHE_NAMESPACE}}"
: "${SPARKCACHE_MAX_BYTES:=42949672960}"
: "${SPARKCACHE_LOW_WATERMARK_BYTES:=34359738368}"
: "${SPARKCACHE_TTL_SECONDS:=0}"
: "${SPARKCACHE_MIN_SPAN_TOKENS:=4096}"
: "${SPARKCACHE_MAX_SPAN_TOKENS:=262144}"
: "${SPARKCACHE_LOAD_THREADS:=8}"
: "${SPARKCACHE_MAX_PENDING_RESTORES:=8}"
: "${SPARKCACHE_CUDA_RESTORE_IO_WORKERS:=8}"
: "${SPARKCACHE_CUDA_ARENA_BYTES:=268435456}"
: "${SOCKET_IFNAME:=enP7s7}"
: "${NCCL_IB_HCA:=rocep1s0f0,rocep1s0f1}"
: "${NCCL_IB_GID_INDEX:=3}"
: "${NCCL_MIN_NCHANNELS:=4}"
: "${NCCL_MAX_NCHANNELS:=4}"
: "${OMP_NUM_THREADS:=16}"
: "${TORCHINDUCTOR_COMPILE_THREADS:=1}"
: "${FASTSAFETENSORS_QUEUE_SIZE:=1}"

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
  MAX_NUM_BATCHED_TOKENS KV_CACHE_MEMORY_BYTES NUM_SPECULATIVE_TOKENS \
  DRAFT_TENSOR_PARALLEL_SIZE MAX_CUDAGRAPH_CAPTURE_SIZE \
  SPARKCACHE_MAX_BYTES SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS \
  SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES \
  SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES \
  NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS OMP_NUM_THREADS \
  TORCHINDUCTOR_COMPILE_THREADS FASTSAFETENSORS_QUEUE_SIZE
do
  require_positive_uint "${name}"
done
require_uint SPARKCACHE_LOW_WATERMARK_BYTES
require_uint SPARKCACHE_TTL_SECONDS
require_uint NCCL_IB_GID_INDEX

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
  die 'the smoke-verified images support SPECULATION_METHOD=dflash in this launcher'
for name in BASE_CACHE_NAMESPACE SPARKCACHE_CACHE_NAMESPACE SPARKCACHE_CLEAR_ONCE; do
  [[ "${!name}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
    die "${name} must contain only letters, digits, dot, underscore, or hyphen"
done
for name in TARGET_MODEL_HOST_PATH DFLASH_MODEL_HOST_PATH CACHE_HOST_ROOT; do
  value="${!name}"
  [[ "${value}" == /* ]] || die "${name} must be an absolute host path"
  [[ "${value}" != *:* && "${value}" != *$'\n'* ]] || \
    die "${name} cannot be represented safely as a Docker bind mount"
done
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

case "${IMAGE_VARIANT}" in
  base)
    image_ref="${BASE_IMAGE_REF}"
    expected_image_id='sha256:8cff7a250f16bfb89df23d29f9233dbb1c700a780dcec86a64c535a71aee88be'
    cache_namespace="${BASE_CACHE_NAMESPACE}"
    ;;
  sparkcache)
    image_ref="${SPARKCACHE_IMAGE_REF}"
    expected_image_id='sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4'
    cache_namespace="${SPARKCACHE_CACHE_NAMESPACE}"
    ;;
  *) die 'IMAGE_VARIANT must be base or sparkcache' ;;
esac

actual_image_id="$(docker image inspect --format '{{.Id}}' "${image_ref}")"
[[ "${actual_image_id}" == "${expected_image_id}" ]] || \
  die "image identity mismatch for ${IMAGE_VARIANT}: expected ${expected_image_id}, got ${actual_image_id}"
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

container="${CONTAINER_PREFIX}-${IMAGE_VARIANT}-r${rank}"
if docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'container already exists: %s\n' "${container}" >&2
  exit 3
fi

launch_status='implemented-tp4-smoke-verified'
modified_settings=()
mark_if_modified() {
  local name="$1" expected="$2"
  [[ "${!name}" == "${expected}" ]] || modified_settings+=("${name}")
}
while IFS='=' read -r name expected; do
  mark_if_modified "${name}" "${expected}"
done <<'SMOKE_DEFAULTS'
SERVED_MODEL_NAME=glm-5.3-flash-nvfp4-dflash2-bf16-t7-jj-r7-gb10
PORT=8015
MASTER_PORT=29775
SHM_SIZE=32g
TENSOR_PARALLEL_SIZE=4
PIPELINE_PARALLEL_SIZE=1
DECODE_CONTEXT_PARALLEL_SIZE=1
NODE_COUNT=4
MAX_MODEL_LEN=262144
MAX_NUM_SEQS=16
MAX_NUM_BATCHED_TOKENS=4096
KV_CACHE_MEMORY_BYTES=21474836480
GPU_MEMORY_UTILIZATION=0.80
KV_CACHE_DTYPE=fp8
SPECULATION_METHOD=dflash
NUM_SPECULATIVE_TOKENS=7
DRAFT_TENSOR_PARALLEL_SIZE=4
DRAFT_KV_CACHE_DTYPE=auto
DRAFT_SAMPLE_METHOD=probabilistic
REJECTION_SAMPLE_METHOD=standard
ATTENTION_BACKEND=B12X
MOE_BACKEND=b12x
LINEAR_BACKEND=b12x
KDA_PREFILL_BACKEND=flashkda
LOAD_FORMAT=fastsafetensors
CUDAGRAPH_MODE=FULL_AND_PIECEWISE
MAX_CUDAGRAPH_CAPTURE_SIZE=128
SOCKET_IFNAME=enP7s7
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
NCCL_IB_GID_INDEX=3
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
OMP_NUM_THREADS=16
TORCHINDUCTOR_COMPILE_THREADS=1
FASTSAFETENSORS_QUEUE_SIZE=1
SMOKE_DEFAULTS
if [[ "${IMAGE_VARIANT}" == base ]]; then
  mark_if_modified BASE_CACHE_NAMESPACE jj-r7-gb10-base-v1
else
  mark_if_modified SPARKCACHE_CACHE_NAMESPACE jj-r7-gb10-page-tail-cow-v1
  mark_if_modified SPARKCACHE_CLEAR_ONCE jj-r7-gb10-page-tail-cow-v1
  mark_if_modified SPARKCACHE_MAX_BYTES 42949672960
  mark_if_modified SPARKCACHE_LOW_WATERMARK_BYTES 34359738368
  mark_if_modified SPARKCACHE_TTL_SECONDS 0
  mark_if_modified SPARKCACHE_MIN_SPAN_TOKENS 4096
  mark_if_modified SPARKCACHE_MAX_SPAN_TOKENS 262144
  mark_if_modified SPARKCACHE_LOAD_THREADS 8
  mark_if_modified SPARKCACHE_MAX_PENDING_RESTORES 8
  mark_if_modified SPARKCACHE_CUDA_RESTORE_IO_WORKERS 8
  mark_if_modified SPARKCACHE_CUDA_ARENA_BYTES 268435456
fi
if (( ${#modified_settings[@]} > 0 )); then
  launch_status='user-modified-unqualified'
  printf 'settings differ from the smoke-verified artifact: %s\n' \
    "$(IFS=,; printf '%s' "${modified_settings[*]}")" >&2
fi
modified_label='none'
if (( ${#modified_settings[@]} > 0 )); then
  modified_label="$(IFS=,; printf '%s' "${modified_settings[*]}")"
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

variant_args=()
if [[ "${IMAGE_VARIANT}" == sparkcache ]]; then
  export SPARKCACHE_CACHE_NAMESPACE SPARKCACHE_CLEAR_ONCE SPARKCACHE_MAX_BYTES
  export SPARKCACHE_LOW_WATERMARK_BYTES SPARKCACHE_TTL_SECONDS
  export SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS
  export SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES
  export SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES
  kv_transfer_config="$(python3 - <<'PY'
import json
import os

def integer(name: str) -> int:
    return int(os.environ[name])

extra = {
    "spark_cache_root": f"/cache/jit/sparkcache-context/{os.environ['SPARKCACHE_CACHE_NAMESPACE']}",
    "spark_cache_model_profile": "glm53-flash-hybrid",
    "spark_cache_publication_schema": "page-tail-cow-v1",
    "spark_cache_target_checkpoint_sha256": "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9",
    "spark_cache_draft_checkpoint_sha256": "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b",
    "spark_cache_draft_policy": "separate",
    "spark_cache_store": True,
    "spark_cache_restore": True,
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
  variant_args=(--kv-transfer-config "${kv_transfer_config}")
fi

headless=()
[[ "${rank}" == 0 ]] || headless=(--headless)

exec docker run -d \
  --name "${container}" \
  --network host --ipc host --shm-size "${SHM_SIZE}" --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  --security-opt label=disable --init \
  -v "${TARGET_MODEL_HOST_PATH}:/models/target:ro" \
  -v "${DFLASH_MODEL_HOST_PATH}:/dflash-draft:ro" \
  -v "${CACHE_HOST_ROOT}:/cache/jit" \
  -e "VLLM_HOST_IP=${HOST_IP}" \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512 \
  -e "VLLM_CACHE_ROOT=/cache/jit/vllm/${cache_namespace}" \
  -e "B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x/6255090a/${cache_namespace}" \
  -e "TRITON_CACHE_DIR=/cache/jit/triton/${cache_namespace}" \
  -e "TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor/${cache_namespace}" \
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
  --label org.sparkring.image-variant="${IMAGE_VARIANT}" \
  --label org.sparkring.rank="${rank}" \
  --label org.sparkring.launch.status="${launch_status}" \
  --label org.sparkring.launch.modified-settings="${modified_label}" \
  "${image_ref}" \
  /models/target \
  --served-model-name "${SERVED_MODEL_NAME}" --host 0.0.0.0 --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}" \
  --distributed-executor-backend mp --nnodes "${NODE_COUNT}" --node-rank "${rank}" \
  --master-addr "${MASTER_ADDR}" --master-port "${MASTER_PORT}" \
  --disable-custom-all-reduce --mamba-cache-mode align --language-model-only \
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
  --speculative-config "${speculative_config}" \
  --compilation-config "${compilation_config}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --async-scheduling --enable-prefix-caching --cudagraph-metrics \
  "${variant_args[@]}" "${headless[@]}"
