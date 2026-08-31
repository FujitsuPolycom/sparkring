#!/usr/bin/env bash
set -euo pipefail

rank="${1:?usage: launch-qualified-rank.sh RANK}"
case "${rank}" in
  0|1|2|3) ;;
  *) printf 'rank must be 0, 1, 2, or 3\n' >&2; exit 2 ;;
esac

: "${HOST_IP:?set HOST_IP to this rank's routable management address}"
: "${MASTER_ADDR:?set MASTER_ADDR to rank 0's routable management address}"
: "${TARGET_MODEL_HOST_PATH:?set TARGET_MODEL_HOST_PATH to the pinned target checkpoint}"
: "${DFLASH_MODEL_HOST_PATH:?set DFLASH_MODEL_HOST_PATH to the pinned BF16 draft checkpoint}"
: "${CACHE_HOST_ROOT:?set CACHE_HOST_ROOT to a dedicated rank-local cache directory}"

expected_image_id='sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818'
image_ref="${IMAGE_REF:-sparkring-glm53-sparkcache:pr535-6da4865-sc59ac0b0-c8-exact-arm64}"
actual_image_id="$(docker image inspect --format '{{.Id}}' "${image_ref}")"
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
  [[ -d "${directory}" ]] || {
    printf 'required directory is missing: %s\n' "${directory}" >&2
    exit 78
  }
done

attempt='glm53-pr535-sc59ac-c8-01'
container="${attempt}-r${rank}"
if docker container inspect "${container}" >/dev/null 2>&1; then
  printf 'container already exists: %s\n' "${container}" >&2
  exit 3
fi

headless=()
if [[ "${rank}" != 0 ]]; then
  headless=(--headless)
fi

exec docker run -d \
  --name "${container}" \
  --network host \
  --ipc host \
  --shm-size 32g \
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
  -e VLLM_CACHE_ROOT=/cache/jit/vllm/pr535-sc78-tp4-01 \
  -e B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x/b1d541f9/pr535-sc78-tp4-01 \
  -e TRITON_CACHE_DIR=/cache/jit/triton/pr535-sc78-tp4-01 \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor/pr535-sc78-tp4-01 \
  -e XDG_CACHE_HOME=/cache/jit \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_NO_USAGE_STATS=1 \
  -e VLLM_PLUGINS= \
  -e OMP_NUM_THREADS=16 \
  -e TORCHINDUCTOR_COMPILE_THREADS=1 \
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
  -e "NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f0,rocep1s0f1}" \
  -e "NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3}" \
  -e NCCL_IB_SUBNET_AWARE_ROUTING=1 \
  -e NCCL_IB_MERGE_NICS=0 \
  -e NCCL_CROSS_NIC=1 \
  -e "NCCL_SOCKET_IFNAME=${SOCKET_IFNAME:-enP7s7}" \
  -e "GLOO_SOCKET_IFNAME=${SOCKET_IFNAME:-enP7s7}" \
  -e NCCL_P2P_LEVEL=SYS \
  -e NCCL_PROTO=LL,LL128,Simple \
  -e NCCL_ALGO=Ring \
  -e NCCL_MIN_NCHANNELS=4 \
  -e NCCL_MAX_NCHANNELS=4 \
  -e NCCL_SWITCHLESS_RING_ONLY=1 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e VLLM_FASTSAFETENSORS_QUEUE_SIZE=1 \
  --label org.sparkring.attempt="${attempt}" \
  --label org.sparkring.rank="${rank}" \
  --label org.sparkring.qualification-status=qualified-bounded \
  --label org.sparkcache.restore-concurrency=8 \
  "${image_ref}" \
  serve /models/target \
  --served-model-name glm-5.3-flash-pr535-sc78-tp4 \
  --host 0.0.0.0 \
  --port 8015 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --distributed-executor-backend mp \
  --nnodes 4 \
  --node-rank "${rank}" \
  --master-addr "${MASTER_ADDR}" \
  --master-port 29775 \
  --disable-custom-all-reduce \
  --mamba-cache-mode align \
  --language-model-only \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --attention-backend B12X \
  --block-size 256 \
  --moe-backend b12x \
  --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  --load-format fastsafetensors \
  --enable-auto-tool-choice \
  --tool-call-parser glm47 \
  --reasoning-parser glm45 \
  --kda-prefill-backend flashkda \
  --gpu-memory-utilization 0.80 \
  --kv-cache-memory-bytes 21474836480 \
  --max-model-len 262144 \
  --max-num-seqs 16 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"dflash","model":"/dflash-draft","num_speculative_tokens":7,"draft_tensor_parallel_size":4,"kv_cache_dtype":"auto","draft_sample_method":"probabilistic","rejection_sample_method":"standard","draft_load_config":{"load_format":"safetensors"}}' \
  --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","cudagraph_capture_sizes":[8,16,32,64,128],"custom_ops":["all"],"pass_config":{"fuse_allreduce_rms":false}}' \
  --max-cudagraph-capture-size 128 \
  --async-scheduling \
  --enable-prefix-caching \
  --cudagraph-metrics \
  --kv-transfer-config '{"kv_connector":"SparkContextCacheConnector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_module_path":"sparkcache.spark_context_cache_connector","kv_connector_extra_config":{"spark_cache_root":"/cache/jit/sparkcache-context/pr535-sc78-tp4-01","spark_cache_model_profile":"glm53-flash-hybrid","spark_cache_publication_schema":"tail-cow-v1","spark_cache_target_checkpoint_sha256":"a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9","spark_cache_draft_checkpoint_sha256":"b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b","spark_cache_draft_policy":"separate","spark_cache_store":true,"spark_cache_restore":true,"spark_cache_scheduler_probe":"none","spark_cache_streaming_snapshots":false,"spark_cache_cuda_restore":true,"spark_cache_max_bytes":42949672960,"spark_cache_low_watermark_bytes":34359738368,"spark_cache_ttl_seconds":0,"spark_cache_min_span_tokens":4096,"spark_cache_max_span_tokens":262144,"spark_cache_cuda_placement_library":"/opt/sparkcache-src/sparkcache/native/build-cuda/libspark_cache_placement.so","spark_cache_cuda_placement_library_sha256":"d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c","spark_cache_cuda_placement_arena_bytes":268435456,"spark_cache_cuda_restore_io_workers":8,"spark_cache_load_threads":8,"spark_cache_max_pending_restores":8,"spark_cache_clear_once":"glm53-pr535-sc78-tp4-01"}}' \
  "${headless[@]}"
