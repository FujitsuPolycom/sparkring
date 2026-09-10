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
: "${DFLASH_MODEL_HOST_PATH:=}"
: "${CACHE_HOST_ROOT:?set CACHE_HOST_ROOT to a dedicated rank-local directory}"

: "${IMAGE_REF:=ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:0d4029b3b7023cf32c37ac20279469c9a2ee16a057f25aae3bcfee9ee5fb660f}"
: "${IMAGE_ID:=sha256:5e32aaa1bbe3559e81db7706ed4286248f18d27cfdb186f6b851bf786eb43075}"
: "${CONTAINER_PREFIX:=glm53-jj-r8-gb10}"
: "${SPARKRING_CREATE_ONLY:=0}"
: "${SPARKRING_PRINT_CONTAINER_SPEC:=0}"
: "${SPARKRING_OFFLINE_SPEC:=0}"
case "${SPARKRING_OFFLINE_SPEC}" in
  0) ;;
  1) [[ "${SPARKRING_PRINT_CONTAINER_SPEC}" == 1 ]] || { printf 'offline rendering requires SPARKRING_PRINT_CONTAINER_SPEC=1\n' >&2; exit 78; } ;;
  *) printf 'SPARKRING_OFFLINE_SPEC must be 0 or 1\n' >&2; exit 78 ;;
esac
: "${SPARKRING_PRINT_MEMORY_PLAN:=0}"
case "${SPARKRING_PRINT_MEMORY_PLAN}" in
  0|1) ;;
  *) printf 'SPARKRING_PRINT_MEMORY_PLAN must be 0 or 1\n' >&2; exit 78 ;;
esac
case "${SPARKRING_PRINT_CONTAINER_SPEC}" in
  0|1) ;;
  *) printf 'SPARKRING_PRINT_CONTAINER_SPEC must be 0 or 1\n' >&2; exit 78 ;;
esac
: "${SERVED_MODEL_NAME:=glm-5.3-flash}"
: "${PORT:=8015}"
: "${MASTER_PORT:=29775}"
: "${SHM_SIZE:=32g}"
: "${TENSOR_PARALLEL_SIZE:=4}"
: "${PIPELINE_PARALLEL_SIZE:=1}"
: "${DECODE_CONTEXT_PARALLEL_SIZE:=4}"
: "${CP_KV_CACHE_INTERLEAVE_SIZE:=auto}"
: "${B12X_MLA_CKV_GATHER:=auto}"
: "${B12X_FUSED_INDEXER:=1}"
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
: "${TARGET_MODEL_VARIANT:=nvfp4}"
: "${NUM_SPECULATIVE_TOKENS:=7}"
: "${DRAFT_TENSOR_PARALLEL_SIZE:=4}"
: "${DRAFT_KV_CACHE_DTYPE:=auto}"
: "${DRAFT_SAMPLE_METHOD:=probabilistic}"
: "${REJECTION_SAMPLE_METHOD:=standard}"
: "${ATTENTION_BACKEND:=B12X}"
: "${MOE_BACKEND:=b12x}"
: "${LINEAR_BACKEND:=b12x}"
: "${KDA_PREFILL_BACKEND:=b12x}"
: "${LOAD_FORMAT:=fastsafetensors}"
: "${CUDAGRAPH_MODE:=FULL_AND_PIECEWISE}"
: "${MAX_CUDAGRAPH_CAPTURE_SIZE:=128}"
: "${SPARKCACHE_CACHE_NAMESPACE:=glm53-flash-vllm-e02b1746-b12x-9ae41c5c-dcp4-page-tail-cow-v2}"
: "${JIT_CACHE_NAMESPACE:=glm53-flash-sm121-vllm-e02b1746-b12x-9ae41c5c}"
: "${JIT_MONITOR_VERBOSE:=0}"
: "${DFLASH_WARMUP:=0}"
: "${DFLASH_WARMUP_CONCURRENCIES:=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16}"
: "${DFLASH_WARMUP_SHAPE_WORDS:=8,24,56,120,248}"
: "${DFLASH_WARMUP_MAX_TOKENS:=16}"
: "${DFLASH_WARMUP_TIMEOUT_SECONDS:=600}"
: "${SPARKRING_WARMUP_TEMPERATURE:=0}"
: "${SPARKRING_LIVENESS_ENABLED:=1}"
: "${SPARKRING_LIVENESS_PORT:=8016}"
: "${SPARKRING_LIVENESS_BLOCKED_SECONDS:=60}"
: "${SPARKRING_LIVENESS_OUTPUT_SECONDS:=300}"
: "${SPARKRING_IDLE_KV_WARN_SECONDS:=330}"
: "${SPARKRING_LIVENESS_STALE_SECONDS:=15}"
: "${SPARKRING_LIVENESS_SAMPLE_SECONDS:=10}"
: "${SIRCL_ENABLED:=0}"
: "${SIRCL_BUNDLE_HOST_ROOT:=}"
: "${SPARK_TP4_PEER0:=}"
: "${SPARK_TP4_PEER1:=}"
: "${SPARK_TP4_DEVICE0:=rocep1s0f0}"
: "${SPARK_TP4_DEVICE1:=rocep1s0f1}"
: "${SPARK_TP4_GID0:=3}"
: "${SPARK_TP4_GID1:=3}"
: "${SPARK_TP4_GRAPH_CONTROL_PORT0:=9970}"
: "${SPARK_TP4_GRAPH_CONTROL_PORT1:=9971}"
: "${SPARK_TP4_GRAPH_SUBMIT_CPU:=10}"
: "${SPARK_TP4_GRAPH_PROGRESS_CPU:=11}"
: "${SPARK_TP4_MAX_INFLIGHT:=64}"
: "${SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS:=10}"
: "${SPARK_TP4_GRAPH_DIRECT_DOORBELL:=0}"
: "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL:=0}"
: "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE:=single}"
: "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE:=sync}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0:=19000}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1:=19001}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0:=}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1:=}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0:=}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1:=}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0:=3}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1:=3}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0:=19100}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1:=19101}"
: "${SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS:=120}"
: "${SPARK_CUDAGRAPH_REPLAY_TIMING:=0}"
: "${SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES:=512}"
: "${SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT:=}"
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
: "${SPARKCACHE_BUFFER_BUDGET_BYTES:=0}"
: "${SPARKCACHE_SOURCE_OVERLAY:=}"
: "${SPARKCACHE_SOURCE_LEASE_CONTRACT:=}"
: "${SPARKCACHE_PLACEMENT_LIBRARY_PATH:=/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so}"
: "${SPARKCACHE_PLACEMENT_LIBRARY_SHA256:=}"
: "${SPARKCACHE_SNAPSHOT_LIBRARY_PATH:=/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_snapshot.so}"
: "${SPARKCACHE_SNAPSHOT_LIBRARY_SHA256:=4398f18b8913e743e7bf1ed8fe29560d4580e61b6a1e2ab8b16684b19b6573b5}"
: "${SPARKCACHE_VLLM_ROOT:=/usr/local/lib/python3.12/dist-packages}"
: "${VLLM_KV_METRICS_OVERLAY:=}"
: "${MULTIMODAL_INPUTS:=1}"
: "${SOCKET_IFNAME:=enP7s7}"
: "${NCCL_IB_HCA:=rocep1s0f0,rocep1s0f1}"
: "${NCCL_IB_GID_INDEX:=3}"
: "${NCCL_MIN_NCHANNELS:=4}"
: "${NCCL_MAX_NCHANNELS:=4}"
: "${NCCL_LIBRARY_PATH:=/opt/sparkring/nccl/libnccl.so.2}"
: "${NCCL_LIBRARY_SHA256:=}"
: "${NCCL_DEBUG:=WARN}"
: "${NCCL_DEBUG_SUBSYS:=NET,INIT,GRAPH}"
: "${SOURCE_IMAGE_PROFILE:=}"
: "${VLLM_BLOCK_SIZE:=256}"
: "${OMP_NUM_THREADS:=16}"
: "${TORCHINDUCTOR_COMPILE_THREADS:=1}"
: "${FASTSAFETENSORS_QUEUE_SIZE:=1}"
: "${ENABLE_PROMPT_TOKENS_DETAILS:=1}"
: "${API_KEYS_FILE:=}"
: "${CHAT_TEMPLATE_HOST_PATH:=}"

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

source_environment=()
for name in VLLM_B12X_KDA_PREFILL_COALESCING VLLM_GLM53_MHC_PREFILL_SHARD \
  VLLM_GLM53_MHC_PREFILL_DIAGNOSTICS VLLM_GLM53_KDA_GATE_SIDE_STREAM \
  VLLM_DCP_TOPK_OWNER_MERGE VLLM_DCP_OWNER_FUSED_ENDPOINTS \
  VLLM_DCP_COMPACT_INDEX_CACHE_OWNER VLLM_DCP_COMPACT_INDEX_TENSOR_VOTE \
  VLLM_DCP_COMPACT_INDEX_LOCAL_WIDTHS VLLM_DCP_COMPACT_INDEX_PROFILE \
  NCCL_IB_EXTENDED_IPV4_GIDS NCCL_IB_PRESERVE_PCI_DOMAIN NCCL_IB_ROUTE_DIAGNOSTICS; do
  value="${!name:-0}"
  [[ "${value}" == 0 || "${value}" == 1 ]] || die "${name} must be 0 or 1"
  source_environment+=(-e "${name}=${value}")
done
case "${NCCL_LIBRARY_PATH}" in
  /opt/sparkring/nccl/libnccl.so.2) ;;
  /opt/local-inference/nccl/lib/libnccl.so.2)
    [[ "${NCCL_LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || \
      die 'R33 NCCL requires its receipt-bound SHA-256' ;;
  /opt/sparkring/nccl-pci/libnccl.so.2.30.7)
    [[ "${NCCL_LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ && -n "${SOURCE_IMAGE_PROFILE}" ]] || \
      die 'Source-composed NCCL requires its receipt-bound profile and SHA-256' ;;
  *) die 'NCCL library path is not supported by this launcher' ;;
esac
case "${VLLM_BLOCK_SIZE}" in 256|512) ;; *) die 'VLLM_BLOCK_SIZE must be 256 or 512' ;; esac
case "${NCCL_DEBUG}" in WARN|INFO) ;; *) die 'NCCL_DEBUG must be WARN or INFO' ;; esac
[[ "${NCCL_DEBUG_SUBSYS}" == NET,INIT,GRAPH ]] || die 'NCCL_DEBUG_SUBSYS must be NET,INIT,GRAPH'
r33_profile=0
case "${SOURCE_IMAGE_PROFILE}" in
  '') ;;
  tp4-dcp1-mtp3-prefill|tp4-dcp4-mtp3-prefill)
    [[ "${SPARKCACHE_ENABLED}" == 0 && "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" == 0 ]] || \
      die 'Source-composed prefill profiles require SparkCache and capture disabled'
    [[ "${SPECULATION_METHOD}" == mtp && "${NUM_SPECULATIVE_TOKENS}" == 3 && "${TENSOR_PARALLEL_SIZE}" == 4 ]] || \
      die 'Source-composed prefill profiles require TP4 and native MTP3'
    [[ "${SOURCE_IMAGE_PROFILE}" == "tp4-dcp${DECODE_CONTEXT_PARALLEL_SIZE}-mtp3-prefill" ]] || \
      die 'DCP size differs from the source-composed profile'
    [[ "${VLLM_B12X_KDA_PREFILL_COALESCING:-0}" == 1 && "${VLLM_GLM53_MHC_PREFILL_SHARD:-0}" == 1 ]] || \
      die 'Source-composed prefill profile requires both coalescing and mHC sharding' ;;
  tp4-dcp1-mtp3-sparkcache)
    [[ "${SPECULATION_METHOD}" == mtp && "${NUM_SPECULATIVE_TOKENS}" == 3 && \
       "${TENSOR_PARALLEL_SIZE}" == 4 && "${DECODE_CONTEXT_PARALLEL_SIZE}" == 1 && \
       "${PIPELINE_PARALLEL_SIZE}" == 1 && "${NODE_COUNT}" == 4 && \
       "${DRAFT_TENSOR_PARALLEL_SIZE}" == 4 ]] || \
      die 'Source SparkCache profile requires TP4/DCP1/PP1 and native MTP3'
    [[ "${TARGET_MODEL_VARIANT}" == nvfp4-spark && "${VLLM_BLOCK_SIZE}" == 512 ]] || \
      die 'Source SparkCache profile requires NVFP4-Spark and 512-token blocks'
    [[ "${SPARKCACHE_SOURCE_LEASE_CONTRACT}" == /usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-connector-jobs-source-contract.json ]] || \
      die 'Source SparkCache profile requires the installed source-image connector-job contract'
    [[ "${SPARKCACHE_PLACEMENT_LIBRARY_SHA256:-}" == 2657cdd2e54a097c9544e4c79ae62c0646db6db123ff24e4f0c384238c3a1e8d ]] || \
      die 'Source SparkCache profile requires its receipt-bound placement library'
    [[ -z "${SPARKCACHE_SOURCE_OVERLAY}" && -z "${VLLM_KV_METRICS_OVERLAY}" ]] || \
      die 'Source SparkCache profile cannot replace receipt-bound package files'
    [[ "${VLLM_B12X_KDA_PREFILL_COALESCING:-0}" == 1 && "${VLLM_GLM53_MHC_PREFILL_SHARD:-0}" == 1 && \
       "${VLLM_DCP_COMPACT_INDEX_CACHE_OWNER:-0}" == 0 ]] || \
      die 'Source SparkCache profile requires coalescing, mHC sharding, and compact index cache disabled'
    [[ "${NCCL_LIBRARY_PATH}" == /opt/sparkring/nccl-pci/libnccl.so.2.30.7 && \
       "${NCCL_IB_EXTENDED_IPV4_GIDS:-0}" == 1 && "${NCCL_IB_PRESERVE_PCI_DOMAIN:-0}" == 1 ]] || \
      die 'Source SparkCache profile requires its dual-domain NCCL settings' ;;
  tp4-dcp1)
    r33_profile=1
    [[ "${SPARKRING_PROFILE_MODE:-}" == custom && "${SPARKRING_MANAGED_MESH_RENDERED:-0}" == 1 ]] || \
      die 'R33 TP4 requires the canonical managed custom profile'
    [[ "${SPARKCACHE_ENABLED}" == 0 && "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" == 0 ]] || \
      die 'R33 tp4-dcp1 requires SparkCache disabled'
    [[ "${VLLM_SPARK_TP4_MODE}" == custom && "${VLLM_SPARK_TP4_VOCAB_MODE}" == custom ]] || \
      die 'R33 TP4 requires custom all-reduce and vocabulary transports'
    [[ "${VLLM_B12X_KDA_PREFILL_COALESCING:-0}" == 1 && \
       "${VLLM_GLM53_MHC_PREFILL_SHARD:-0}" == 1 && \
       "${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH:-0}" == 1 ]] || \
      die 'R33 TP4 requires coalescing, mHC, and GDN metadata fast path'
    [[ "${NCCL_LIBRARY_PATH}" == /opt/local-inference/nccl/lib/libnccl.so.2 ]] || \
      die 'R33 TP4 requires installed NCCL 2.31.2' ;;
  tp4-dcp1-sparkcache)
    r33_profile=1
    [[ "${SPARKRING_PROFILE_MODE:-}" == custom && "${SPARKRING_MANAGED_MESH_RENDERED:-0}" == 1 ]] || \
      die 'R33 SparkCache requires the canonical managed custom profile'
    [[ "${SPARKCACHE_ENABLED}" == 1 && "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" == 1 && \
       "${SPARKCACHE_ACCESS_MODE}" == read-write ]] || \
      die 'R33 SparkCache requires bounded read-write asynchronous capture'
    [[ "${VLLM_SPARK_TP4_MODE}" == custom && "${VLLM_SPARK_TP4_VOCAB_MODE}" == custom ]] || \
      die 'R33 SparkCache requires custom all-reduce and vocabulary transports'
    [[ "${VLLM_B12X_KDA_PREFILL_COALESCING:-0}" == 1 && \
       "${VLLM_GLM53_MHC_PREFILL_SHARD:-0}" == 1 && \
       "${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH:-0}" == 1 ]] || \
      die 'R33 SparkCache requires coalescing, mHC, and GDN metadata fast path'
    [[ "${NCCL_LIBRARY_PATH}" == /opt/local-inference/nccl/lib/libnccl.so.2 ]] || \
      die 'R33 SparkCache requires installed NCCL 2.31.2'
    [[ "${SPARKCACHE_PLACEMENT_LIBRARY_PATH}" == /opt/sparkring/sparkcache/lib/libspark_cache_placement.so && \
       "${SPARKCACHE_PLACEMENT_LIBRARY_SHA256}" == d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93 && \
       "${SPARKCACHE_SNAPSHOT_LIBRARY_PATH}" == /opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so && \
       "${SPARKCACHE_SNAPSHOT_LIBRARY_SHA256}" == 7da9e72f096ae679906ba71336c16e7894a247eb5b0d217aaccd115b85058953 && \
       "${SPARKCACHE_VLLM_ROOT}" == /opt/venv/lib/python3.12/site-packages && \
       "${SPARKCACHE_SOURCE_LEASE_CONTRACT}" == /opt/venv/lib/python3.12/site-packages/sparkcache/runtime_patches/vllm-connector-jobs-source-contract.json ]] || \
      die 'R33 SparkCache native and lease paths differ from its receipt' ;;
  *) die 'Unsupported source-composed runtime profile' ;;
esac
if [[ -n "${SPARKCACHE_SOURCE_LEASE_CONTRACT}" && \
      "${SOURCE_IMAGE_PROFILE}" != tp4-dcp1-mtp3-sparkcache && \
      "${SOURCE_IMAGE_PROFILE}" != tp4-dcp1-sparkcache ]]; then
  die 'SPARKCACHE_SOURCE_LEASE_CONTRACT requires the source SparkCache profile'
fi
if [[ -n "${SOURCE_IMAGE_PROFILE}" ]]; then
  source_environment+=(-e "SOURCE_IMAGE_PROFILE=${SOURCE_IMAGE_PROFILE}" -e PYTHONUNBUFFERED=1)
fi

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
  TORCHINDUCTOR_COMPILE_THREADS FASTSAFETENSORS_QUEUE_SIZE \
  SPARKRING_LIVENESS_PORT SPARKRING_LIVENESS_BLOCKED_SECONDS SPARKRING_LIVENESS_OUTPUT_SECONDS \
  SPARKRING_IDLE_KV_WARN_SECONDS SPARKRING_LIVENESS_STALE_SECONDS \
  SPARKRING_LIVENESS_SAMPLE_SECONDS
do
  require_positive_uint "${name}"
done
require_uint SPARKCACHE_LOW_WATERMARK_BYTES
require_uint SPARKCACHE_TTL_SECONDS
require_uint SPARKCACHE_BUFFER_BUDGET_BYTES
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
  # Keep the per-rank allocation uniform while DCP controls cache geometry.
  KV_CACHE_MEMORY_BYTES=25769803776
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
case "${B12X_FUSED_INDEXER}" in
  0|1) ;;
  *) die 'B12X_FUSED_INDEXER must be 0 or 1' ;;
esac

[[ "${rank}" =~ ^[0-9]+$ ]] || die 'rank must be an unsigned integer'
(( rank < NODE_COUNT )) || die "rank must be between 0 and $((NODE_COUNT - 1))"
(( PORT <= 65535 && MASTER_PORT <= 65535 && SPARKRING_LIVENESS_PORT <= 65535 )) || \
  die 'ports must be at most 65535'
(( PORT != SPARKRING_LIVENESS_PORT )) || \
  die 'SPARKRING_LIVENESS_PORT must differ from PORT'
(( SPARKCACHE_LOW_WATERMARK_BYTES <= SPARKCACHE_MAX_BYTES )) || \
  die 'SPARKCACHE_LOW_WATERMARK_BYTES cannot exceed SPARKCACHE_MAX_BYTES'
(( SPARKCACHE_MIN_SPAN_TOKENS <= SPARKCACHE_MAX_SPAN_TOKENS )) || \
  die 'SPARKCACHE_MIN_SPAN_TOKENS cannot exceed SPARKCACHE_MAX_SPAN_TOKENS'
(( NCCL_MIN_NCHANNELS <= NCCL_MAX_NCHANNELS )) || \
  die 'NCCL_MIN_NCHANNELS cannot exceed NCCL_MAX_NCHANNELS'
[[ "${CONTAINER_PREFIX}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || \
  die 'CONTAINER_PREFIX is not a valid Docker container-name prefix'
case "${SPECULATION_METHOD}" in
  dflash) : "${DFLASH_MODEL_HOST_PATH:?set DFLASH_MODEL_HOST_PATH to the pinned BF16 draft checkpoint}" ;;
  mtp) ;;
  *) die 'SPECULATION_METHOD must be dflash or mtp' ;;
esac
case "${TARGET_MODEL_VARIANT}" in
  nvfp4)
    target_config_sha256=676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996
    target_index_sha256=0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb
    TARGET_CHECKPOINT_FINGERPRINT=a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9
    ;;
  nvfp4-spark)
    target_config_sha256=e1c0246a44ebefb5fd6383fb57aebbf7ac69ff6e7b23e989c0571b279a0eca23
    target_index_sha256=db30fc7c5a70ccfb3b1c46637bb4ddb04226b95a5dfc451dffccb96a4f0ff544
    TARGET_CHECKPOINT_FINGERPRINT=357f6a86160ebd5caff25d9a10d9f29e8547b16c6c73e78751fa69fde11ac4e4
    ;;
  *) die 'TARGET_MODEL_VARIANT must be nvfp4 or nvfp4-spark' ;;
esac
# Native MTP loads its predictor from the target checkpoint. The separate
# cache policy describes registered layer roles, not separate weight files.
DRAFT_CHECKPOINT_FINGERPRINT=b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b
if [[ "${SPECULATION_METHOD}" == mtp ]]; then
  DRAFT_CHECKPOINT_FINGERPRINT="${TARGET_CHECKPOINT_FINGERPRINT}"
fi
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
  auto|0|1) ;;
  *) die 'SPARKCACHE_ASYNC_PAGE_CAPTURE must be auto, 0, or 1' ;;
esac
case "${SPARKCACHE_ACCESS_MODE}" in
  read-write|restore-only|store-only|disabled) ;;
  *) die 'SPARKCACHE_ACCESS_MODE must be read-write, restore-only, store-only, or disabled' ;;
esac
if [[ "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" == auto ]]; then
  SPARKCACHE_ASYNC_PAGE_CAPTURE=0
  if [[ "${SPARKCACHE_ENABLED}" == 1 ]]; then
    case "${SPARKCACHE_ACCESS_MODE}" in
      read-write|store-only) SPARKCACHE_ASYNC_PAGE_CAPTURE=1 ;;
    esac
  fi
fi
if [[ "${SPARKCACHE_ASYNC_PAGE_CAPTURE}" == 1 ]]; then
  [[ "${SPARKCACHE_ENABLED}" == 1 ]] || \
    die 'asynchronous page capture requires SPARKCACHE_ENABLED=1'
  case "${SPARKCACHE_ACCESS_MODE}" in
    read-write|store-only) ;;
    *) die 'asynchronous page capture requires a publication-capable access mode' ;;
  esac
fi

if [[ "${SOURCE_IMAGE_PROFILE}" == tp4-dcp1-mtp3-sparkcache || \
      "${SOURCE_IMAGE_PROFILE}" == tp4-dcp1-sparkcache ]]; then
  # These are the bounded capacities named by the source-image profile. Reject
  # inherited operator defaults instead of allocating larger unqualified buffers.
  for setting in SPARKCACHE_ENABLED=1 SPARKCACHE_ACCESS_MODE=read-write \
    SPARKCACHE_ASYNC_PAGE_CAPTURE=1 SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT=2 \
    SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES=536870912 SPARKCACHE_LOAD_THREADS=2 \
    SPARKCACHE_MAX_PENDING_RESTORES=2 SPARKCACHE_CUDA_RESTORE_IO_WORKERS=2 \
    SPARKCACHE_CUDA_ARENA_BYTES=67108864 SPARKCACHE_BUFFER_BUDGET_BYTES=1342177280 \
    SPARKCACHE_MAX_BYTES=8589934592 SPARKCACHE_LOW_WATERMARK_BYTES=6442450944 \
    SPARKCACHE_MIN_SPAN_TOKENS=4096 SPARKCACHE_MAX_SPAN_TOKENS=65536 \
    SPARKCACHE_PUBLICATION_SCHEMA=tail-cow-v2 KV_CACHE_MEMORY_BYTES=25769803776 \
    MAX_MODEL_LEN=1048576; do
    name="${setting%%=*}"
    [[ "${!name}" == "${setting#*=}" ]] || die "Source SparkCache profile requires ${setting}"
  done
fi

# Resolve the payload buffers before inspecting checkpoints or contacting Docker.
# Python integers avoid overflow when comparing operator-supplied byte budgets.
command -v python3 >/dev/null 2>&1 || die 'python3 is required to resolve the memory plan'
export SPARKCACHE_ENABLED SPARKCACHE_ACCESS_MODE SPARKCACHE_LOAD_THREADS
export SPARKCACHE_CUDA_ARENA_BYTES SPARKCACHE_ASYNC_PAGE_CAPTURE
export SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT
export SPARKCACHE_BUFFER_BUDGET_BYTES KV_CACHE_MEMORY_BYTES
export TENSOR_PARALLEL_SIZE PIPELINE_PARALLEL_SIZE DECODE_CONTEXT_PARALLEL_SIZE
memory_plan="$(python3 - <<'PY'
import json
import os
import sys

def integer(name):
    return int(os.environ[name])

enabled = os.environ["SPARKCACHE_ENABLED"] == "1"
mode = os.environ["SPARKCACHE_ACCESS_MODE"]
# The pinned manager-page connector caps active placement lanes at eight.
requested_lanes = integer("SPARKCACHE_LOAD_THREADS")
lanes = min(8, requested_lanes) if enabled and mode in ("read-write", "restore-only") else 0
slots = integer("SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT") if os.environ["SPARKCACHE_ASYNC_PAGE_CAPTURE"] == "1" else 0
restore = lanes * 2 * integer("SPARKCACHE_CUDA_ARENA_BYTES")
capture = slots * integer("SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES")
buffers = restore + capture
budget = integer("SPARKCACHE_BUFFER_BUDGET_BYTES")
ranks = integer("TENSOR_PARALLEL_SIZE") * integer("PIPELINE_PARALLEL_SIZE")
kv = integer("KV_CACHE_MEMORY_BYTES")
print(json.dumps({
    "status": "implemented",
    "basis": "configured payload capacities; not measured resident memory",
    "access_mode": mode if enabled else "disabled",
    "async_page_capture": slots > 0,
    "dcp_degree": integer("DECODE_CONTEXT_PARALLEL_SIZE"),
    "rank_count": ranks,
    "requested_restore_lanes_per_rank": requested_lanes,
    "restore_lanes_per_rank": lanes,
    "arenas_per_restore_lane": 2,
    "capture_slots_per_rank": slots,
    "restore_payload_bytes_per_rank": restore,
    "capture_payload_bytes_per_rank": capture,
    "sparkcache_payload_bytes_per_rank": buffers,
    "sparkcache_payload_bytes_all_ranks": buffers * ranks,
    "buffer_budget_bytes_per_rank": budget or None,
    "within_buffer_budget": budget == 0 or buffers <= budget,
    "kv_cache_bytes_per_rank": kv,
    "kv_and_payload_bytes_per_rank": kv + buffers,
    "kv_and_payload_bytes_all_ranks": (kv + buffers) * ranks,
    "excluded": ["model weights", "CUDA control arrays", "Python objects and read buffers",
                 "shared base retention", "transport", "compilation workspaces", "allocator overhead"],
}, sort_keys=True))
if budget and buffers > budget:
    print(f"SparkCache payload buffers require {buffers} bytes per rank, exceeding "
          f"SPARKCACHE_BUFFER_BUDGET_BYTES={budget}", file=sys.stderr)
    sys.exit(78)
PY
)" || { printf '%s\n' "${memory_plan}"; exit 78; }
if [[ "${SPARKRING_PRINT_MEMORY_PLAN}" == 1 ]]; then
  printf '%s\n' "${memory_plan}"
  exit 0
fi
printf 'sparkcache: memory_plan %s\n' "${memory_plan}" >&2
if [[ -n "${CHAT_TEMPLATE_HOST_PATH}" ]]; then
  [[ "${CHAT_TEMPLATE_HOST_PATH}" == /* ]] || \
    die 'CHAT_TEMPLATE_HOST_PATH must be an absolute host path when set'
  [[ "${CHAT_TEMPLATE_HOST_PATH}" != *:* && "${CHAT_TEMPLATE_HOST_PATH}" != *$'\n'* ]] || \
    die 'CHAT_TEMPLATE_HOST_PATH cannot be represented safely as a Docker bind mount'
  [[ -f "${CHAT_TEMPLATE_HOST_PATH}" && -r "${CHAT_TEMPLATE_HOST_PATH}" ]] || \
    die 'CHAT_TEMPLATE_HOST_PATH is not a readable regular file'
  [[ -s "${CHAT_TEMPLATE_HOST_PATH}" ]] || \
    die 'CHAT_TEMPLATE_HOST_PATH is empty'
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
case "${SPARKRING_LIVENESS_ENABLED}" in
  0|1) ;;
  *) die 'SPARKRING_LIVENESS_ENABLED must be 0 or 1' ;;
esac
case "${SIRCL_ENABLED}" in
  0|1) ;;
  *) die 'SIRCL_ENABLED must be 0 or 1' ;;
esac
case "${SPARK_TP4_GRAPH_DIRECT_DOORBELL}" in
  0|1) ;;
  *) die 'SPARK_TP4_GRAPH_DIRECT_DOORBELL must be 0 or 1' ;;
esac
case "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL}" in
  0|1) ;;
  *) die 'VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL must be 0 or 1' ;;
esac
case "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE}" in
  single|dual) ;;
  *) die 'VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE must be single or dual' ;;
esac
case "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE}" in
  sync|fused) ;;
  *) die 'VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE must be sync or fused' ;;
esac
if [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL}" == 1 && "${SIRCL_ENABLED}" != 1 ]]; then
  die 'bidirectional prefill requires SIRCL_ENABLED=1'
fi
if [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE}" == dual && "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL}" != 1 ]]; then
  die 'dual-rail bidirectional prefill requires VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=1'
fi
if [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE}" == fused ]]; then
  [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL}" == 1 ]] || \
    die 'fused prefill exposure requires VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=1'
  [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE}" == dual ]] || \
    die 'fused prefill exposure requires dual rail mode'
fi
case "${SPARK_CUDAGRAPH_REPLAY_TIMING}" in
  0|1) ;;
  *) die 'SPARK_CUDAGRAPH_REPLAY_TIMING must be 0 or 1' ;;
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
model_path_names=(TARGET_MODEL_HOST_PATH CACHE_HOST_ROOT)
if [[ "${SPECULATION_METHOD}" == dflash ]]; then
  model_path_names+=(DFLASH_MODEL_HOST_PATH)
fi
for name in "${model_path_names[@]}"; do
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
sircl_args=()
sircl_native_sha256='disabled'
sircl_manifest_sha256='disabled'
sircl_container_root='/opt/spark-sircl'
if [[ "${r33_profile}" == 1 ]]; then
  sircl_container_root='/opt/sparkring/sircl'
fi
sircl_bundle_is_external=0
if [[ "${SIRCL_ENABLED}" == 1 ]]; then
  [[ "${TENSOR_PARALLEL_SIZE}" == 4 ]] || \
    die 'SIRCL width-4096 mode requires TENSOR_PARALLEL_SIZE=4'
  [[ -n "${SPARK_TP4_PEER0}" && -n "${SPARK_TP4_PEER1}" ]] || \
    die 'SIRCL requires SPARK_TP4_PEER0 and SPARK_TP4_PEER1'
  [[ -n "${SPARK_TP4_DEVICE0}" && -n "${SPARK_TP4_DEVICE1}" ]] || \
    die 'SIRCL requires SPARK_TP4_DEVICE0 and SPARK_TP4_DEVICE1'
  for name in \
    SPARK_TP4_GID0 SPARK_TP4_GID1 \
    SPARK_TP4_GRAPH_CONTROL_PORT0 SPARK_TP4_GRAPH_CONTROL_PORT1 \
    SPARK_TP4_GRAPH_SUBMIT_CPU SPARK_TP4_GRAPH_PROGRESS_CPU \
    SPARK_TP4_MAX_INFLIGHT SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS
  do
    require_positive_uint "${name}"
  done
  (( SPARK_TP4_GRAPH_CONTROL_PORT0 <= 65535 && SPARK_TP4_GRAPH_CONTROL_PORT1 <= 65535 )) || \
    die 'SIRCL graph control ports must be at most 65535'
  (( SPARK_TP4_GRAPH_SUBMIT_CPU != SPARK_TP4_GRAPH_PROGRESS_CPU )) || \
    die 'SIRCL graph submit and progress CPUs must be distinct'
  if [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL}" == 1 ]]; then
    for name in \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS
    do
      require_positive_uint "${name}"
    done
    for name in \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1
    do
      (( ${!name} <= 65529 )) || \
        die "${name} must be at most 65529 to reserve the complete prefill port range"
    done
  fi
  if [[ "${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE}" == dual ]]; then
    for name in \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1
    do
      [[ -n "${!name}" ]] || die "dual-rail bidirectional prefill requires ${name}"
    done
    for name in \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0 \
      SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1
    do
      require_uint "${name}"
      (( ${!name} <= 255 )) || die "${name} must be at most 255"
    done
    [[ "${SPARK_TP4_PEER0}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0}" && \
       "${SPARK_TP4_PEER0}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1}" && \
       "${SPARK_TP4_PEER1}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0}" && \
       "${SPARK_TP4_PEER1}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1}" && \
       "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1}" ]] || \
      die 'dual-rail primary and secondary peer addresses must be distinct'
    [[ "${SPARK_TP4_DEVICE0}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0}" && \
       "${SPARK_TP4_DEVICE0}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1}" && \
       "${SPARK_TP4_DEVICE1}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0}" && \
       "${SPARK_TP4_DEVICE1}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1}" && \
       "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0}" != "${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1}" ]] || \
      die 'dual-rail primary and secondary devices must be distinct'
  fi
  if [[ -n "${SIRCL_BUNDLE_HOST_ROOT}" ]]; then
    [[ "${SIRCL_BUNDLE_HOST_ROOT}" == /* ]] || \
      die 'SIRCL_BUNDLE_HOST_ROOT must be an absolute host path when set'
    [[ "${SIRCL_BUNDLE_HOST_ROOT}" != *:* && "${SIRCL_BUNDLE_HOST_ROOT}" != *$'\n'* ]] || \
      die 'SIRCL_BUNDLE_HOST_ROOT cannot be represented safely as a Docker bind mount'
    [[ -d "${SIRCL_BUNDLE_HOST_ROOT}" ]] || \
      die 'SIRCL_BUNDLE_HOST_ROOT must be a directory'
    for required in \
      sitecustomize.py \
      spark_collective_audit.py \
      spark_graph_status_reporter.py \
      spark_persistent_output_ring.py \
      spark_tp4_backend.py \
      spark_tp4_capability.py \
      spark_tp4_health_gate.py \
      spark_tp4_port_namespace.py \
      spark_tp4_query_contract.py \
      spark_tp4_query_row_provider.py \
      sparkring-overlay-manifest.json \
      libspark_transport_capi.so
    do
      [[ -f "${SIRCL_BUNDLE_HOST_ROOT}/${required}" ]] || \
        die "SIRCL bundle is missing ${required}"
    done
    sircl_native_sha256="$(
      sha256sum -- "${SIRCL_BUNDLE_HOST_ROOT}/libspark_transport_capi.so" |
        cut -d' ' -f1
    )"
    sircl_manifest_sha256="$(
      sha256sum -- "${SIRCL_BUNDLE_HOST_ROOT}/sparkring-overlay-manifest.json" |
        cut -d' ' -f1
    )"
    sircl_args+=(
      -v "${SIRCL_BUNDLE_HOST_ROOT}:${sircl_container_root}:ro"
    )
    sircl_bundle_is_external=1
  elif [[ "${r33_profile}" == 1 ]]; then
    sircl_native_sha256="${SPARKRING_DECLARED_SIRCL_NATIVE_SHA256:?R33 rendering requires declared native identity}"
    sircl_manifest_sha256="${SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256:?R33 rendering requires declared manifest identity}"
    [[ "${sircl_native_sha256}" =~ ^[0-9a-f]{64}$ && "${sircl_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || \
      die 'invalid R33 SIRCL identity'
  elif [[ "${SPARKRING_OFFLINE_SPEC}" == 1 ]]; then
    sircl_native_sha256="${SPARKRING_DECLARED_SIRCL_NATIVE_SHA256:?offline rendering requires declared native identity}"
    sircl_manifest_sha256="${SPARKRING_DECLARED_SIRCL_MANIFEST_SHA256:?offline rendering requires declared manifest identity}"
    [[ "${sircl_native_sha256}" =~ ^[0-9a-f]{64}$ && "${sircl_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || die 'invalid declared SIRCL identity'
  else
    sircl_native_sha256="$(
      docker image inspect --format \
        '{{index .Config.Labels "org.sparkring.sircl.native-sha256"}}' \
        "${IMAGE_REF}"
    )"
    sircl_manifest_sha256="$(
      docker image inspect --format \
        '{{index .Config.Labels "org.sparkring.sircl.manifest-sha256"}}' \
        "${IMAGE_REF}"
    )"
    [[ "${sircl_native_sha256}" =~ ^[0-9a-f]{64}$ && \
       "${sircl_manifest_sha256}" =~ ^[0-9a-f]{64}$ ]] || \
      die 'image has no receipt-bound embedded SIRCL bundle; set SIRCL_BUNDLE_HOST_ROOT or disable SIRCL'
  fi
  sircl_args=(
    "${sircl_args[@]}"
    -e "PYTHONPATH=${sircl_container_root}"
    -e "SPARK_TP4_LIBRARY=${sircl_container_root}/libspark_transport_capi.so"
    -e VLLM_SPARK_TP4_MODE=custom
    -e VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH=1
    -e VLLM_SPARK_SHARED_CAPTURE_STREAM=1
    -e VLLM_SPARK_TP4_GRAPH_Q1=0
    -e VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40=0
    -e SPARK_TP4_CAPABILITY_VOTE=1
    -e SPARK_TP4_HEALTH_GATE=1
    -e "SPARKRING_SIRCL_NATIVE_SHA256=${sircl_native_sha256}"
    -e "SPARKRING_SIRCL_MANIFEST_SHA256=${sircl_manifest_sha256}"
    -e "SPARK_TP4_PEER0=${SPARK_TP4_PEER0}"
    -e "SPARK_TP4_PEER1=${SPARK_TP4_PEER1}"
    -e "SPARK_TP4_DEVICE0=${SPARK_TP4_DEVICE0}"
    -e "SPARK_TP4_DEVICE1=${SPARK_TP4_DEVICE1}"
    -e "SPARK_TP4_GID0=${SPARK_TP4_GID0}"
    -e "SPARK_TP4_GID1=${SPARK_TP4_GID1}"
    -e "SPARK_TP4_GRAPH_CONTROL_PORT0=${SPARK_TP4_GRAPH_CONTROL_PORT0}"
    -e "SPARK_TP4_GRAPH_CONTROL_PORT1=${SPARK_TP4_GRAPH_CONTROL_PORT1}"
    -e "SPARK_TP4_GRAPH_SUBMIT_CPU=${SPARK_TP4_GRAPH_SUBMIT_CPU}"
    -e "SPARK_TP4_GRAPH_PROGRESS_CPU=${SPARK_TP4_GRAPH_PROGRESS_CPU}"
    -e "SPARK_TP4_MAX_INFLIGHT=${SPARK_TP4_MAX_INFLIGHT}"
    -e "SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS=${SPARK_TP4_CONTROL_CONNECT_TIMEOUT_SECONDS}"
    -e "SPARK_TP4_GRAPH_DIRECT_DOORBELL=${SPARK_TP4_GRAPH_DIRECT_DOORBELL}"
    -e "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL=${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL}"
    -e "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE=${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE}"
    -e "VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE=${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0=${SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT0}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1=${SPARK_TP4_BIDIRECTIONAL_PREFILL_CONTROL_PORT1}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER0}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_PEER1}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE0}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_DEVICE1}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID0}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_GID1}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT0}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1=${SPARK_TP4_BIDIRECTIONAL_PREFILL_SECONDARY_CONTROL_PORT1}"
    -e "SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS=${SPARK_TP4_BIDIRECTIONAL_PREFILL_TIMEOUT_SECONDS}"
    -e SPARK_TP4_FLIGHT_RECORDER=0
    -e "SPARK_TP4_GRAPH_STATUS_PATH=/cache/jit/sircl-graph-rank${rank}.json"
  )
  if [[ "${r33_profile}" == 1 ]]; then
    sircl_args+=(
      -e PYTHONPATH=/opt/sparkring/sircl/python
      -e SPARK_TP4_LIBRARY=/opt/sparkring/sircl/libspark_transport_capi.so
      -e VLLM_SPARK_TP4_VOCAB_MODE=custom
      -e "SPARK_TP4_CONTROL_PORT0=${SPARK_TP4_GRAPH_CONTROL_PORT0}"
      -e "SPARK_TP4_CONTROL_PORT1=${SPARK_TP4_GRAPH_CONTROL_PORT1}"
    )
  fi
fi
replay_timing_args=()
if [[ "${SPARK_CUDAGRAPH_REPLAY_TIMING}" == 1 ]]; then
  require_positive_uint SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES
  replay_timing_bundle="${SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT}"
  replay_timing_container_root='/opt/spark-replay-timing'
  replay_timing_bundle_is_external=1
  if [[ "${SIRCL_ENABLED}" == 1 ]]; then
    replay_timing_bundle="${SIRCL_BUNDLE_HOST_ROOT}"
    replay_timing_container_root="${sircl_container_root}"
    replay_timing_bundle_is_external="${sircl_bundle_is_external}"
  else
    [[ "${replay_timing_bundle}" == /* ]] || \
      die 'SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT must be an absolute host path'
    [[ "${replay_timing_bundle}" != *:* && "${replay_timing_bundle}" != *$'\n'* ]] || \
      die 'SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT cannot be represented safely as a Docker bind mount'
    [[ -d "${replay_timing_bundle}" ]] || \
      die 'SPARK_CUDAGRAPH_REPLAY_TIMING_BUNDLE_HOST_ROOT must be a directory'
    replay_timing_args+=(
      -v "${replay_timing_bundle}:${replay_timing_container_root}:ro"
      -e "PYTHONPATH=${replay_timing_container_root}"
    )
  fi
  if [[ "${replay_timing_bundle_is_external}" == 1 ]]; then
    [[ -f "${replay_timing_bundle}/spark_graph_status_reporter.py" ]] || \
      die 'CUDA graph replay timing bundle is missing spark_graph_status_reporter.py'
    for required in sitecustomize.py spark_cudagraph_replay_timing.py; do
      [[ -f "${replay_timing_bundle}/${required}" ]] || \
        die "CUDA graph replay timing bundle is missing ${required}"
    done
  fi
  replay_timing_args+=(
    -e "SPARK_CUDAGRAPH_REPLAY_TIMING_STATUS_PATH=/cache/jit/cudagraph-replay-rank${rank}.json"
    -e SPARK_CUDAGRAPH_REPLAY_TIMING=1
    -e "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES=${SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES}"
    -e SPARK_CUDAGRAPH_REPLAY_TIMING_ARM_PATH=/cache/jit/sircl-replay-timing.arm
  )
fi
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

if [[ "${SPARKRING_OFFLINE_SPEC}" == 0 ]]; then
actual_image_id="$(docker image inspect --format '{{.Id}}' "${IMAGE_REF}")"
[[ "${actual_image_id}" == "${IMAGE_ID}" ]] || \
  die "image identity mismatch: expected ${IMAGE_ID}, got ${actual_image_id}"
for name in "${model_path_names[@]}"; do
  directory="${!name}"
  [[ -d "${directory}" ]] || die "required directory is missing: ${directory}"
done
fi

verify_file_sha256() {
  local role="$1" path="$2" expected="$3" actual
  # Offline output records intended arguments; image and file checks belong to
  # the consumer's preflight before it starts any rank.
  [[ "${SPARKRING_OFFLINE_SPEC}" == 0 ]] || return 0
  [[ -f "${path}" ]] || die "${role} is missing: ${path}"
  actual="$(sha256sum -- "${path}")"
  actual="${actual%% *}"
  [[ "${actual}" == "${expected}" ]] || \
    die "${role} identity mismatch: expected ${expected}, got ${actual}"
}
verify_file_sha256 \
  'target config.json' \
  "${TARGET_MODEL_HOST_PATH}/config.json" \
  "${target_config_sha256}"
verify_file_sha256 \
  'target model.safetensors.index.json' \
  "${TARGET_MODEL_HOST_PATH}/model.safetensors.index.json" \
  "${target_index_sha256}"
draft_mount_args=()
if [[ "${SPECULATION_METHOD}" == dflash ]]; then
verify_file_sha256 \
  'draft config.json' \
  "${DFLASH_MODEL_HOST_PATH}/config.json" \
  'c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573'
verify_file_sha256 \
  'draft model.safetensors' \
  "${DFLASH_MODEL_HOST_PATH}/model.safetensors" \
  'b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b'
  draft_mount_args=(-v "${DFLASH_MODEL_HOST_PATH}:/dflash-draft:ro")
fi

container="${CONTAINER_PREFIX}-r${rank}"
if [[ "${SPARKRING_PRINT_CONTAINER_SPEC}" == 0 ]] && docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'container already exists: %s\n' "${container}" >&2
  exit 3
fi

export NUM_SPECULATIVE_TOKENS DRAFT_TENSOR_PARALLEL_SIZE DRAFT_KV_CACHE_DTYPE
export DRAFT_SAMPLE_METHOD REJECTION_SAMPLE_METHOD SPECULATION_METHOD
speculative_config="$(python3 - <<'PY'
import json
import os

config = {
    "method": os.environ["SPECULATION_METHOD"],
    "num_speculative_tokens": int(os.environ["NUM_SPECULATIVE_TOKENS"]),
    "draft_tensor_parallel_size": int(os.environ["DRAFT_TENSOR_PARALLEL_SIZE"]),
    "kv_cache_dtype": os.environ["DRAFT_KV_CACHE_DTYPE"],
    "draft_sample_method": os.environ["DRAFT_SAMPLE_METHOD"],
    "rejection_sample_method": os.environ["REJECTION_SAMPLE_METHOD"],
    "draft_load_config": {"load_format": "safetensors"},
}
if config["method"] == "dflash":
    config["model"] = "/dflash-draft"
else:
    config["attention_backend"] = "B12X"
print(json.dumps(config, separators=(",", ":")))
PY
)"

export CUDAGRAPH_MODE MAX_CUDAGRAPH_CAPTURE_SIZE NUM_SPECULATIVE_TOKENS
compilation_config="$(python3 - <<'PY'
import json
import os

maximum = int(os.environ["MAX_CUDAGRAPH_CAPTURE_SIZE"])
rows_per_request = int(os.environ["NUM_SPECULATIVE_TOKENS"]) + 1
if maximum % rows_per_request:
    raise SystemExit(
        "MAX_CUDAGRAPH_CAPTURE_SIZE must be divisible by "
        "NUM_SPECULATIVE_TOKENS + 1"
    )
capture_sizes = list(range(rows_per_request, maximum + 1, rows_per_request))
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
  export SPARKCACHE_PLACEMENT_LIBRARY_SHA256="${SPARKCACHE_PLACEMENT_LIBRARY_SHA256:-d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c}"
  [[ "${SPARKCACHE_PLACEMENT_LIBRARY_SHA256}" =~ ^[0-9a-f]{64}$ ]] || die 'SPARKCACHE_PLACEMENT_LIBRARY_SHA256 must be a SHA-256 digest'
  export SPARKCACHE_SHARED_PREFIX_LEASE_TTL_SECONDS
  export SPARKCACHE_PUBLICATION_SCHEMA
  export SPARKCACHE_LOW_WATERMARK_BYTES SPARKCACHE_TTL_SECONDS
  export SPARKCACHE_MIN_SPAN_TOKENS SPARKCACHE_MAX_SPAN_TOKENS
  export SPARKCACHE_LOAD_THREADS SPARKCACHE_MAX_PENDING_RESTORES
  export SPARKCACHE_CUDA_RESTORE_IO_WORKERS SPARKCACHE_CUDA_ARENA_BYTES
  export SPARKCACHE_ASYNC_PAGE_CAPTURE
  export SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT
  export SOURCE_IMAGE_PROFILE SPARKCACHE_SOURCE_LEASE_CONTRACT
  export SPARKCACHE_PLACEMENT_LIBRARY_PATH SPARKCACHE_SNAPSHOT_LIBRARY_PATH
  export SPARKCACHE_SNAPSHOT_LIBRARY_SHA256 SPARKCACHE_VLLM_ROOT
  export TARGET_CHECKPOINT_FINGERPRINT DRAFT_CHECKPOINT_FINGERPRINT
  kv_transfer_config="$(python3 - <<'PY'
import json
import os

def integer(name: str) -> int:
    return int(os.environ[name])

extra = {
    "spark_cache_root": f"/cache/jit/sparkcache-context/{os.environ['SPARKCACHE_CACHE_NAMESPACE']}",
    "spark_cache_model_profile": "glm53-flash-hybrid",
    "spark_cache_publication_schema": os.environ["SPARKCACHE_PUBLICATION_SCHEMA"],
    "spark_cache_target_checkpoint_sha256": os.environ["TARGET_CHECKPOINT_FINGERPRINT"],
    "spark_cache_draft_checkpoint_sha256": os.environ["DRAFT_CHECKPOINT_FINGERPRINT"],
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
    "spark_cache_cuda_placement_library": os.environ["SPARKCACHE_PLACEMENT_LIBRARY_PATH"],
    "spark_cache_cuda_placement_library_sha256": os.environ["SPARKCACHE_PLACEMENT_LIBRARY_SHA256"],
    "spark_cache_cuda_placement_arena_bytes": integer("SPARKCACHE_CUDA_ARENA_BYTES"),
    "spark_cache_cuda_restore_io_workers": integer("SPARKCACHE_CUDA_RESTORE_IO_WORKERS"),
    "spark_cache_load_threads": integer("SPARKCACHE_LOAD_THREADS"),
    "spark_cache_max_pending_restores": integer("SPARKCACHE_MAX_PENDING_RESTORES"),
    "spark_cache_clear_once": os.environ["SPARKCACHE_CLEAR_ONCE"],
    "spark_cache_async_page_capture": os.environ["SPARKCACHE_ASYNC_PAGE_CAPTURE"] == "1",
    "spark_cache_async_page_capture_library": os.environ["SPARKCACHE_SNAPSHOT_LIBRARY_PATH"],
    "spark_cache_async_page_capture_library_sha256": os.environ["SPARKCACHE_SNAPSHOT_LIBRARY_SHA256"],
    "spark_cache_async_page_capture_slot_bytes": integer("SPARKCACHE_ASYNC_CAPTURE_SLOT_BYTES"),
    "spark_cache_async_page_capture_slot_count": integer("SPARKCACHE_ASYNC_CAPTURE_SLOT_COUNT"),
    "spark_cache_async_page_capture_vllm_root": os.environ["SPARKCACHE_VLLM_ROOT"],
    "spark_cache_async_page_capture_lease_contract": "/usr/local/lib/python3.12/dist-packages/sparkcache/runtime_patches/vllm-manager-page-async-contract-55969c16.json",
}
if os.environ["SOURCE_IMAGE_PROFILE"] == "tp4-dcp1-mtp3-sparkcache":
    extra.update({
        "spark_cache_async_page_capture_lease_mode": "connector-jobs",
        "spark_cache_async_page_capture_lease_contract": os.environ["SPARKCACHE_SOURCE_LEASE_CONTRACT"],
        "spark_cache_async_page_capture_library": "/opt/sparkcache-native/libspark_cache_snapshot.so",
        "spark_cache_async_page_capture_library_sha256": "cc44b69c9e01aaeb6b94f46cd649e2ec5972fc6b29d1a41b7b033a82ce788f39",
        "spark_cache_cuda_restore_arena_budget_bytes": 268435456,
        "spark_cache_page_snapshot_interval_tokens": 0,
    })
elif os.environ["SOURCE_IMAGE_PROFILE"] == "tp4-dcp1-sparkcache":
    extra.update({
        "spark_cache_async_page_capture_lease_mode": "connector-jobs",
        "spark_cache_async_page_capture_lease_contract": os.environ["SPARKCACHE_SOURCE_LEASE_CONTRACT"],
        "spark_cache_cuda_restore_arena_budget_bytes": 268435456,
        "spark_cache_page_snapshot_interval_tokens": 0,
    })
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
chat_template_mount=()
chat_template_args=()
if [[ -n "${CHAT_TEMPLATE_HOST_PATH}" ]]; then
  chat_template_mount=(-v "${CHAT_TEMPLATE_HOST_PATH}:/opt/sparkring/chat_template.jinja:ro")
  chat_template_args=(--chat-template /opt/sparkring/chat_template.jinja)
fi
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

case "${SPARKRING_CREATE_ONLY}" in
  0) container_action=(run -d) ;;
  1) container_action=(create) ;;
  *) die 'SPARKRING_CREATE_ONLY must be 0 or 1' ;;
esac

serving_entrypoint=/opt/sparkring/bin/serve-with-warmup.py
serving_prefix=()
source_recurrent_args=()
if [[ -n "${SOURCE_IMAGE_PROFILE}" ]]; then
  # Source-bound profiles must verify installed files before importing serving code.
  if [[ "${r33_profile}" == 1 ]]; then
    serving_entrypoint=/opt/sparkring/bin/sparkring-r33
    serving_prefix=(serve)
  else
    serving_entrypoint=python3
    serving_prefix=(-S -B /opt/sparkcache-jj-runtime/verify_sources.py --serve)
  fi
  # Automatic request-boundary checkpoints exclude continuation coalescing.
  source_recurrent_args=(--mamba-block-size "${VLLM_BLOCK_SIZE}" --recurrent-checkpoint-policy aligned --prefix-cache-retention-interval 0)
fi
r33_environment=()
if [[ "${r33_profile}" == 1 ]]; then
  r33_environment=(
    -e "SIRCL_ENABLED=${SIRCL_ENABLED}"
    -e "SPARKCACHE_ENABLED=${SPARKCACHE_ENABLED}"
    -e "NODE_RANK=${rank}"
    -e "MASTER_ADDR=${MASTER_ADDR}"
    -e "SPARKRING_PROFILE_MODE=${SPARKRING_PROFILE_MODE}"
    -e "SPARKRING_MANAGED_MESH_RENDERED=${SPARKRING_MANAGED_MESH_RENDERED}"
    -e "VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH=${VLLM_GDN_SPEC_DECODE_METADATA_FASTPATH}"
    -e "NCCL_LOCAL_INFERENCE_PATH=${NCCL_LIBRARY_PATH}"
  )
  if [[ "${SOURCE_IMAGE_PROFILE}" == tp4-dcp1-sparkcache ]]; then
    r33_environment+=(
      -e "SPARKCACHE_CACHE_NAMESPACE=${SPARKCACHE_CACHE_NAMESPACE}"
      -e "SPARKCACHE_PLACEMENT_LIBRARY_PATH=${SPARKCACHE_PLACEMENT_LIBRARY_PATH}"
      -e "SPARKCACHE_PLACEMENT_LIBRARY_SHA256=${SPARKCACHE_PLACEMENT_LIBRARY_SHA256}"
      -e "SPARKCACHE_SNAPSHOT_LIBRARY_PATH=${SPARKCACHE_SNAPSHOT_LIBRARY_PATH}"
      -e "SPARKCACHE_SNAPSHOT_LIBRARY_SHA256=${SPARKCACHE_SNAPSHOT_LIBRARY_SHA256}"
      -e "SPARKCACHE_VLLM_ROOT=${SPARKCACHE_VLLM_ROOT}"
      -e "SPARKCACHE_SOURCE_LEASE_CONTRACT=${SPARKCACHE_SOURCE_LEASE_CONTRACT}"
    )
  fi
fi

container_command=(docker "${container_action[@]}" \
  --name "${container}" \
  --entrypoint "${serving_entrypoint}" \
  --network host --ipc host --shm-size "${SHM_SIZE}" --gpus all \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband \
  --security-opt label=disable --init \
  -v "${TARGET_MODEL_HOST_PATH}:/models/target:ro" \
  "${draft_mount_args[@]}" \
  -v "${CACHE_HOST_ROOT}:/cache/jit" \
  "${chat_template_mount[@]}" \
  "${sparkcache_source_args[@]}" \
  "${vllm_metrics_args[@]}" \
  "${sircl_args[@]}" \
  "${replay_timing_args[@]}" \
  -e "SPARKRING_NODE_RANK=${rank}" \
  -e "PORT=${PORT}" -e "SERVED_MODEL_NAME=${SERVED_MODEL_NAME}" \
  -e "DFLASH_WARMUP=${DFLASH_WARMUP}" \
  -e "DFLASH_WARMUP_CONCURRENCIES=${DFLASH_WARMUP_CONCURRENCIES}" \
  -e "DFLASH_WARMUP_SHAPE_WORDS=${DFLASH_WARMUP_SHAPE_WORDS}" \
  -e "DFLASH_WARMUP_MAX_TOKENS=${DFLASH_WARMUP_MAX_TOKENS}" \
  -e "DFLASH_WARMUP_TIMEOUT_SECONDS=${DFLASH_WARMUP_TIMEOUT_SECONDS}" \
  -e "SPARKRING_WARMUP_TEMPERATURE=${SPARKRING_WARMUP_TEMPERATURE}" \
  -e "SPARKRING_LIVENESS_ENABLED=${SPARKRING_LIVENESS_ENABLED}" \
  -e "SPARKRING_LIVENESS_PORT=${SPARKRING_LIVENESS_PORT}" \
  -e "SPARKRING_LIVENESS_BLOCKED_SECONDS=${SPARKRING_LIVENESS_BLOCKED_SECONDS}" \
  -e "SPARKRING_LIVENESS_OUTPUT_SECONDS=${SPARKRING_LIVENESS_OUTPUT_SECONDS}" \
  -e "SPARKRING_IDLE_KV_WARN_SECONDS=${SPARKRING_IDLE_KV_WARN_SECONDS}" \
  -e "SPARKRING_LIVENESS_STALE_SECONDS=${SPARKRING_LIVENESS_STALE_SECONDS}" \
  -e "SPARKRING_LIVENESS_SAMPLE_SECONDS=${SPARKRING_LIVENESS_SAMPLE_SECONDS}" \
  "${warmup_api_key_env[@]}" \
  -e "VLLM_HOST_IP=${HOST_IP}" \
  -e VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512 \
  -e VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512 \
  -e "VLLM_B12X_MLA_CKV_GATHER=${B12X_MLA_CKV_GATHER}" \
  -e "B12X_FUSED_INDEXER=${B12X_FUSED_INDEXER}" \
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
  "${source_environment[@]}" \
  "${r33_environment[@]}" \
  -e "VLLM_NCCL_SO_PATH=${NCCL_LIBRARY_PATH}" \
  -e "LD_PRELOAD=${NCCL_LIBRARY_PATH}" \
  -e "NCCL_DEBUG=${NCCL_DEBUG}" -e "NCCL_DEBUG_SUBSYS=${NCCL_DEBUG_SUBSYS}" \
  -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none \
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
  --label org.sparkring.sircl.enabled="${SIRCL_ENABLED}" \
  --label org.sparkring.sircl.direct-doorbell="${SPARK_TP4_GRAPH_DIRECT_DOORBELL}" \
  --label org.sparkring.sircl.prefill-exposure="${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_EXPOSURE}" \
  --label org.sparkring.sircl.prefill-rail-mode="${VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE}" \
  --label org.sparkring.sircl.native-sha256="${sircl_native_sha256}" \
  --label org.sparkring.sircl.manifest-sha256="${sircl_manifest_sha256}" \
  "${IMAGE_REF}" \
  "${serving_prefix[@]}" \
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
  "${source_recurrent_args[@]}" \
  "${chat_template_args[@]}" \
  --enable-chunked-prefill --dtype bfloat16 --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --quantization modelopt_mixed --attention-backend "${ATTENTION_BACKEND}" \
  --block-size "${VLLM_BLOCK_SIZE}" --moe-backend "${MOE_BACKEND}" --linear-backend "${LINEAR_BACKEND}" \
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
  "${kv_transfer_args[@]}" "${headless[@]}")

# The inspection path emits the same argument array used for execution. It
# performs identity checks above, but never creates a container or waits for a model.
if [[ "${SPARKRING_PRINT_CONTAINER_SPEC}" == 1 ]]; then
  python3 - "${container_command[@]}" <<'PY'
import json
import sys
print(json.dumps({"schema": "sparkring-container-command/v1", "argv": sys.argv[1:]}))
PY
  exit 0
fi
container_id="$("${container_command[@]}")"

if [[ "${SPARKRING_CREATE_ONLY}" == 0 && "${rank}" == 0 && "${DFLASH_WARMUP}" == 1 ]]; then
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
