#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: launch-qualified-rank.sh RANK [CONFIG_FILE]}"
config_file="${2:-${SPARKRING_CONFIG_FILE:-}}"
if (( $# > 2 )); then
  printf 'usage: launch-qualified-rank.sh RANK [CONFIG_FILE]\n' >&2
  exit 2
fi
if [[ -n "${config_file}" ]]; then
  [[ -r "${config_file}" && -f "${config_file}" ]] || {
    printf 'configuration file is not a readable regular file: %s\n' "${config_file}" >&2
    exit 78
  }
  # This is a shell environment file. Review it before use; sourcing preserves
  # quoted paths without reparsing values through eval.
  # shellcheck source=/dev/null
  source "${config_file}"
fi

: "${HOST_IP:?set HOST_IP to this rank's routable management address}"
: "${MASTER_ADDR:?set MASTER_ADDR to rank 0's routable management address}"
: "${TARGET_MODEL_HOST_PATH:?set TARGET_MODEL_HOST_PATH to the pinned target checkpoint}"
: "${DFLASH_MODEL_HOST_PATH:?set DFLASH_MODEL_HOST_PATH to the pinned BF16 draft checkpoint}"
: "${CACHE_HOST_ROOT:?set CACHE_HOST_ROOT to a dedicated rank-local cache directory}"

# Defaults reproduce the artifact described by qualified-artifact.json.
: "${IMAGE_REF:=sparkring-glm53-sparkcache:pr535-6da4865-sc59ac0b0-c8-exact-arm64}"
: "${CONTAINER_PREFIX:=glm53-pr535-sc59ac-c8-01}"
: "${SERVED_MODEL_NAME:=glm-5.3-flash-pr535-sc78-tp4}"
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
: "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}"
: "${MAX_CUDAGRAPH_CAPTURE_SIZE:=128}"
: "${CACHE_NAMESPACE:=pr535-sc78-tp4-01}"
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
  SPARKCACHE_MAX_BYTES SPARKCACHE_LOW_WATERMARK_BYTES \
  SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS \
  SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES \
  SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES \
  NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS OMP_NUM_THREADS \
  TORCHINDUCTOR_COMPILE_THREADS FASTSAFETENSORS_QUEUE_SIZE
do
  require_positive_uint "${name}"
done
require_uint SPARKCACHE_TTL_SECONDS
require_uint NCCL_IB_GID_INDEX

(( PORT <= 65535 )) || die 'PORT must be at most 65535'
(( MASTER_PORT <= 65535 )) || die 'MASTER_PORT must be at most 65535'
[[ "${rank}" =~ ^[0-9]+$ ]] || die 'rank must be an unsigned integer'
(( rank < NODE_COUNT )) || die "rank must be between 0 and $((NODE_COUNT - 1))"
(( SPARKCACHE_LOW_WATERMARK_BYTES <= SPARKCACHE_MAX_BYTES )) || \
  die 'SPARKCACHE_LOW_WATERMARK_BYTES cannot exceed SPARKCACHE_MAX_BYTES'
(( SPARKCACHE_MIN_SPAN_TOKENS <= SPARKCACHE_MAX_SPAN_TOKENS )) || \
  die 'SPARKCACHE_MIN_SPAN_TOKENS cannot exceed SPARKCACHE_MAX_SPAN_TOKENS'
(( NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS )) || \
  die 'NCCL_MIN_NCHANNELS cannot exceed NCCL_MAX_NCHANNELS'
[[ "${CACHE_NAMESPACE}" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || \
  die 'CACHE_NAMESPACE must contain only letters, digits, dot, underscore, or hyphen'
[[ "${CONTAINER_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  die 'CONTAINER_PREFIX is not a valid Docker container-name prefix'
[[ "${SPECULATION_METHOD}" == dflash ]] || \
  die 'this qualified launcher supports SPECULATION_METHOD=dflash only'
for path_name in TARGET_MODEL_HOST_PATH DFLASH_MODEL_HOST_PATH CACHE_HOST_ROOT; do
  path_value="${!path_name}"
  [[ "${path_value}" == /* ]] || die "${path_name} must be an absolute host path"
  [[ "${path_value}" != *:* && "${path_value}" != *$'\n'* ]] || \
    die "${path_name} contains a character that cannot be used in a Docker bind mount"
done
command -v python3 >/dev/null 2>&1 || die 'python3 is required to encode JSON configuration safely'
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

expected_image_id='sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818'
actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_REF}")"
if [[ "${actual_image_id}" != "${expected_image_id}" ]]; then
  printf 'image identity mismatch: expected %s, got %s\n' \
    "${expected_image_id}" "${actual_image_id}" >&2
  exit 78
fi

for directory in \
  "${TARGET_MODEL_HOST_PATH}" \
  "${DFLASH_MODEL_HOST_PATH}" \
  "${CACHE_HOST_ROOT}"
do
  [[ -d "${directory}" ]] || die "required directory is missing: ${directory}"
done

container="${CONTAINER_PREFIX}-r${rank}"
if docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'container already exists: %s\n' "${container}" >&2
  exit 3
fi

# Runtime changes do not inherit the artifact's bounded qualification. Site
# addresses, bind mounts, image aliases resolving to the verified ID, and
# container names do not change serving behavior.
qualification_status='qualified-bounded'
modified_settings=()
mark_if_modified() {
  local name="$1" expected="$2"
  if [[ "${!name}" != "${expected}" ]]; then
    modified_settings+=("${name}")
  fi
}
while IFS='=' read -r name expected; do
  mark_if_modified "${name}" "${expected}"
done <<'QUALIFIED_DEFAULTS'
SERVED_MODEL_NAME=glm-5.3-flash-pr535-sc78-tp4
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
CUDAGRAPH_MODE=FULL_AND_PIECEWISE
MAX_CUDAGRAPH_CAPTURE_SIZE=128
CACHE_NAMESPACE=pr535-sc78-tp4-01
SPARKCACHE_MAX_BYTES=42949672960
SPARKCACHE_LOW_WATERMARK_BYTES=34359738368
SPARKCACHE_TTL_SECONDS=0
SPARKCACHE_MIN_SPAN_TOKENS=4096
SPARKCACHE_MAX_SPAN_TOKENS=262144
SPARKCACHE_LOAD_THREADS=8
SPARKCACHE_MAX_PENDING_RESTORES=8
SPARKCACHE_CUDA_RESTORE_IO_WORKERS=8
SPARKCACHE_CUDA_ARENA_BYTES=268435456
SOCKET_IFNAME=enP7s7
NCCL_IB_HCA=rocep1s0f0,rocep1s0f1
NCCL_IB_GID_INDEX=3
NCCL_MIN_NCHANNELS=4
NCCL_MAX_NCHANNELS=4
OMP_NUM_THREADS=16
TORCHINDUCTOR_COMPILE_THREADS=1
FASTSAFETENSORS_QUEUE_SIZE=1
QUALIFIED_DEFAULTS
if (( ${#modified_settings[@]} > 0 )); then
  qualification_status='user-modified-unqualified'
  printf 'runtime settings differ from the qualified artifact: %s\n' \
    "$(IFS=,; printf '%s' "${modified_settings[*]}")" >&2
fi

export CACHE_NAMESPACE SPARKCACHE_MAX_BYTES SPARKCACHE_LOW_WATERMARK_BYTES
export SPARKCACHE_TTL_SECONDS SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS
export SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES
export SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES
kv_transfer_config="$(python3 - <<'PY'
import json
import os

def integer(name: str) -> int:
    return int(os.environ[name])

extra = {
    "spark_cache_root": f"/cache/jit/sparkcache-context/{os.environ['CACHE_NAMESPACE']}",
    "spark_cache_model_profile": "glm53-flash-hybrid",
    "spark_cache_publication_schema": "tail-cow-v1",
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
    "spark_cache_clear_once": os.environ["CACHE_NAMESPACE"],
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

export CUDAGRAPH_MODE
compilation_config="$(python3 - <<'PY'
import json
import os

print(json.dumps({
    "cudagraph_mode": os.environ["CUDAGRAPH_MODE"],
    "cudagraph_capture_sizes": [8, 16, 32, 64, 128],
    "custom_ops": ["all"],
    "pass_config": {"fuse_allreduce_rms": False},
}, separators=(",", ":")))
PY
)"

headless=()
if [[ "${rank}" != 0 ]]; then
  headless=(--headless)
fi
modified_label='none'
if (( ${#modified_settings[@]} > 0 )); then
  modified_label="$(IFS=,; printf '%s' "${modified_settings[*]}")"
fi

exec docker run -d \
  --name "${container}" \
  --network host \
  --ipc host \
  --shm-size "${SHM_SIZE}" \
  --gpus all \
  --ulimit memlock=-1:-1 \
  --cap-add IPC_LOCK \
  --device /dev/infiniband \
  --security-opt label=disable \
  --init \
  -v "${TARGET_MODEL_HOST_PATH}:/models/target:ro" \
  -v "${DFLASH_MODEL_HOST_PATH}:/dflash-draft:ro" \
  -v "${CACHE_HOST_ROOT}:/cache/jit" \
  -e "VLLM_HOST_IP=${HOST_IP}" \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512 \
  -e "VLLM_CACHE_ROOT=/cache/jit/vllm/${CACHE_NAMESPACE}" \
  -e "B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x/b1d541f9/${CACHE_NAMESPACE}" \
  -e "TRITON_CACHE_DIR=/cache/jit/triton/${CACHE_NAMESPACE}" \
  -e "TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor/${CACHE_NAMESPACE}" \
  -e XDG_CACHE_HOME=/cache/jit \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_PLUGINS= \
  -e "OMP_NUM_THREADS=${OMP_NUM_THREADS}" \
  -e "TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e CMAKE_CUDA_ARCHITECTURES=121 \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e CUTE_DSL_ARCH=sm_121a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1f \
  -e VLLM_B12X_MOE_FP4_FORCE_A16=0 \
  -e VLLM_ENABLE_PCIE_ALLREDUCE=0 \
  -e VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  -e VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2 \
  -e LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2 \
  -e NCCL_DEBUG=WARN \
  -e NCCL_NET=IB \
  -e NCCL_NET_PLUGIN=none \
  -e NCCL_IB_DISABLE=0 \
  -e "NCCL_IB_HCA=${NCCL_IB_HCA}" \
  -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX}" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CROSS_NIC=1 \
  -e "NCCL_SOCKET_IFNAME=${SOCKET_IFNAME}" \
  -e "GLOO_SOCKET_IFNAME=${SOCKET_IFNAME}" \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_ALGO=Ring \
  -e "NCCL_MIN_NCHANNELS=${NCCL_MIN_NCHANNELS}" \
  -e "NCCL_MAX_NCHANNELS=${NCCL_MAX_NCHANNELS}" \
  -e NCCL_SWITCHLESS_RING_ONLY=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e "VLLM_FASTSAFETENSORS_QUEUE_SIZE=${FASTSAFETENSORS_QUEUE_SIZE}" \
  --label org.sparkring.attempt="${CONTAINER_PREFIX}" \
  --label org.sparkring.rank="${rank}" \
  --label org.sparkring.qualification-status="${qualification_status}" \
  --label org.sparkring.modified-settings="${modified_label}" \
  --label org.sparkcache.restore-concurrency="${SPARKCACHE_LOAD_THREADS}" \
  "${IMAGE_REF}" \
  serve /models/target \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --pipeline-parallel-size "${PIPELINE_PARALLEL_SIZE}" \
  --decode-context-parallel-size "${DECODE_CONTEXT_PARALLEL_SIZE}" \
  --distributed-executor-backend mp \
  --nnodes "${NODE_COUNT}" \
  --node-rank "${rank}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port "${MASTER_PORT}" \
  --disable-custom-all-reduce \
  --mamba-cache-mode align \
  --language-model-only \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --quantization modelopt_mixed \
  --attention-backend "${ATTENTION_BACKEND}" \
  --block-size 256 \
  --moe-backend "${MOE_BACKEND}" \
  --linear-backend "${LINEAR_BACKEND}" \
  --no-enable-flashinfer-autotune \
  --load-format fastsafetensors \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kda-prefill-backend "${KDA_PREFILL_BACKEND}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --speculative-config "${speculative_config}" \
  --compilation-config "${compilation_config}" \
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}" \
  --async-scheduling \
  --enable-prefix-caching \
  --cudagraph-metrics \
  --kv-transfer-config "${kv_transfer_config}" \
  "${headless[@]}"
