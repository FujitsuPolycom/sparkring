#!/usr/bin/env bash
# Validate or start one rank of the DeepSeek-V4.1-Flash four-Spark cycle profile.
# The per-rank env file is both the host launch contract and the container
# environment, so operator-facing paths and serving values have one source.
#
# Stock upstream vLLM image (runtime/deepseek-v41-gb10) + seven bind-mounted
# patches (Engram tables on NVMe, SM12x page sizes, SM12x top-k) + SparkRing's
# patched NCCL preloaded from a host path. --check is offline (no docker needed).
set -euo pipefail

usage() { echo "usage: deepseek_v41_cycle_serve.sh [--check|--run] ENV_FILE" >&2; }
die() { echo "deepseek v41 cycle launcher: $*" >&2; exit 20; }

case "$#" in
    1) mode=--check; env_file=$1 ;;
    2) mode=$1; env_file=$2 ;;
    *) usage; exit 64 ;;
esac
case "$mode" in --check|--run) ;; *) usage; exit 64 ;; esac

[ -f "$env_file" ] || die "environment file is missing: $env_file"
env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")
if grep -Ev '^[[:space:]]*(#|$)' "$env_file" | grep -Eq '<[A-Za-z0-9_]+>|REPLACE_WITH_'; then
    die "environment file contains unresolved placeholders: $env_file"
fi
# shellcheck disable=SC1090
. "$env_file"

require_value() { local v=${!1-}; [ -n "$v" ] || die "required value is empty: $1"; }
require_directory() { local v=${!1-}; case "$v" in /*) ;; *) die "$1 must be an absolute host path: $v" ;; esac; [ -d "$v" ] || die "$1 directory does not exist: $v"; }
require_positive_integer() { local v=${!1-}; case "$v" in ''|*[!0-9]*) die "$1 must be a positive integer: $v" ;; esac; [ "$((10#$v))" -gt 0 ] || die "$1 must be greater than zero"; }
require_port() { require_positive_integer "$1"; [ "$((10#${!1}))" -le 65535 ] || die "$1 must be in 1..65535"; }

for name in NODE_RANK MASTER_ADDR VLLM_HOST_IP MODEL_HOST_PATH CACHE_HOST_PATH PATCH_DIR \
    NCCL_SO_HOST_PATH IMAGE API_PORT MASTER_PORT SERVED_MODEL_NAME MAX_MODEL_LEN MAX_NUM_SEQS \
    MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION NUM_SPECULATIVE_TOKENS ENFORCE_EAGER TEXT_ONLY \
    THINKING_DEFAULT DRAFT_SAMPLE_METHOD ENGRAM_DISK_THREADS ENGRAM_DISK_CHUNK ENGRAM_BALANCED NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
    NCCL_IB_HCA NCCL_IB_GID_INDEX NCCL_IB_SUBNET_PREFIX_LEN NCCL_IB_SUBNET_AWARE_ROUTING \
    NCCL_IB_MERGE_NICS NCCL_CROSS_NIC NCCL_ALGO NCCL_PROTO NCCL_P2P_LEVEL NCCL_MIN_NCHANNELS \
    NCCL_MAX_NCHANNELS NCCL_SKIP_TREE_CONNECT NCCL_SWITCHLESS_RING_ONLY NCCL_CUMEM_ENABLE \
    NCCL_IGNORE_CPU_AFFINITY; do
    require_value "$name"
done
case "$NODE_RANK" in 0|1|2|3) ;; *) die "NODE_RANK must be 0, 1, 2, or 3: $NODE_RANK" ;; esac
require_directory MODEL_HOST_PATH; require_directory CACHE_HOST_PATH; require_directory PATCH_DIR
[ -w "$CACHE_HOST_PATH" ] || die "CACHE_HOST_PATH is not writable: $CACHE_HOST_PATH"
[ -f "$MODEL_HOST_PATH/config.json" ] || die "MODEL_HOST_PATH has no config.json: $MODEL_HOST_PATH"
grep -q '"DeepseekV41ForCausalLM"' "$MODEL_HOST_PATH/config.json" || die "config.json is not DeepseekV41ForCausalLM"
[ -f "$MODEL_HOST_PATH/model-00048-of-00048.safetensors" ] || die "checkpoint incomplete: shard 48 missing (Engram rows are read from shards 47/48 at serve time)"
[ -f "$NCCL_SO_HOST_PATH" ] || die "patched NCCL library missing: $NCCL_SO_HOST_PATH"
[ -f "$PATCH_DIR/mounts.txt" ] || die "no mounts.txt in PATCH_DIR: $PATCH_DIR"
[ -f "$PATCH_DIR/MD5SUMS" ] || die "no MD5SUMS in PATCH_DIR: $PATCH_DIR"
command -v md5sum >/dev/null 2>&1 || die "md5sum is required to verify patch bytes"
(cd "$PATCH_DIR" && md5sum -c MD5SUMS --quiet) || die "patch md5 mismatch in $PATCH_DIR"
for name in MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS NUM_SPECULATIVE_TOKENS ENGRAM_DISK_THREADS ENGRAM_DISK_CHUNK; do require_positive_integer "$name"; done
require_port API_PORT; require_port MASTER_PORT
[ "$((10#$API_PORT))" -ne "$((10#$MASTER_PORT))" ] || die "API_PORT and MASTER_PORT must differ"
case "$GPU_MEMORY_UTILIZATION" in 0.[0-9]|0.[0-9][0-9]) ;; *) die "GPU_MEMORY_UTILIZATION must look like 0.80: $GPU_MEMORY_UTILIZATION" ;; esac
case "$ENFORCE_EAGER" in 0|1) ;; *) die "ENFORCE_EAGER must be 0 or 1" ;; esac
case "$TEXT_ONLY" in 0|1) ;; *) die "TEXT_ONLY must be 0 or 1" ;; esac
case "$THINKING_DEFAULT" in true|false) ;; *) die "THINKING_DEFAULT must be true or false" ;; esac
case "$DRAFT_SAMPLE_METHOD" in greedy|probabilistic) ;; *) die "DRAFT_SAMPLE_METHOD must be greedy or probabilistic" ;; esac
case "$ENGRAM_BALANCED" in 0|1) ;; *) die "ENGRAM_BALANCED must be 0 or 1" ;; esac
case "${ENGRAM_PACKED_DIR:-}" in ""|/cache/*) ;; *) die "ENGRAM_PACKED_DIR must be empty or a path under /cache (the CACHE_HOST_PATH mount): $ENGRAM_PACKED_DIR" ;; esac
[ "$((10#$NUM_SPECULATIVE_TOKENS % 5))" -eq 0 ] || die "NUM_SPECULATIVE_TOKENS must be a multiple of the checkpoint's dspark_block_size (5)"
[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] || die "NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME must match"
[ "$NCCL_IB_SUBNET_PREFIX_LEN" = 24 ] || die "NCCL_IB_SUBNET_PREFIX_LEN must be 24"
[ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 1 ] || die "cycle subnet-aware routing must be enabled"
[ "$NCCL_IB_MERGE_NICS" = 0 ] || die "cycle NIC merging must be disabled"
[ "$NCCL_CROSS_NIC" = 1 ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_ALGO" = Ring ] || die "NCCL_ALGO must be Ring"
[ "$NCCL_P2P_LEVEL" = SYS ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_SKIP_TREE_CONNECT" = 1 ] || die "NCCL_SKIP_TREE_CONNECT must be 1"
[ "$NCCL_SWITCHLESS_RING_ONLY" = 1 ] || die "NCCL_SWITCHLESS_RING_ONLY must be 1"
[ "$NCCL_CUMEM_ENABLE" = 0 ] || die "NCCL_CUMEM_ENABLE must be 0"
[ "$NODE_RANK" != 0 ] || [ "$MASTER_ADDR" = "$VLLM_HOST_IP" ] || die "rank-0 MASTER_ADDR must equal rank-0 VLLM_HOST_IP"
IFS=',' read -r -a hca_specs <<< "$NCCL_IB_HCA"
[ "${#hca_specs[@]}" = 2 ] || die "the cycle environment must name exactly two RoCE devices"
[ "${hca_specs[0]}" != "${hca_specs[1]}" ] || die "the cycle environment must name two distinct RoCE devices"
case "$NCCL_IB_GID_INDEX" in ''|*[!0-9]*) die "NCCL_IB_GID_INDEX must be a decimal integer" ;; esac

container_name="deepseek-v41-flash-r$NODE_RANK"
model_container_path=/models/DeepSeek-V4.1-Flash
site=/usr/local/lib/python3.12/dist-packages/vllm
headless=(); [ "$NODE_RANK" = 0 ] || headless=(--headless)

patch_mounts=()
declare -A patch_targets=()
while read -r f rel; do
    [ -z "$f" ] && continue
    [[ $f != */* && $f != . && $f != .. && -n $rel && $rel != /* && "/$rel/" != *"/../"* ]] \
        || die "mounts.txt contains an unsafe patch path"
    [ -z "${patch_targets[$rel]+present}" ] || die "mounts.txt repeats a patch destination"
    patch_targets[$rel]=1
    [ -f "$PATCH_DIR/$f" ] || die "patch file missing: $PATCH_DIR/$f"
    patch_mounts+=(-v "$PATCH_DIR/$f:$site/$rel:ro")
done < "$PATCH_DIR/mounts.txt"
[ "${#patch_mounts[@]}" -eq 14 ] || die "mounts.txt must list the seven patch files"

# Profiler: VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0 stops vLLM reserving an estimated
# 1.5 GiB for graphs that measure 0.54 GiB on this profile; the KV pool gains the difference.
# DSpark: every decode batch is a multiple of k (draft) or k+1 (target) tokens, so
# capturing exactly those sizes leaves no padded rows (padded spec batches can hang
# the SM120 sparse-MLA kernel, flashinfer #5015); adaptive verification stays off.
k=$NUM_SPECULATIVE_TOKENS
capture_sizes=$( { seq "$k" "$k" $((k * MAX_NUM_SEQS)); seq $((k + 1)) $((k + 1)) $(((k + 1) * MAX_NUM_SEQS)); } | sort -n -u | paste -sd, - )
if [ "$ENFORCE_EAGER" = 1 ]; then
    graph_args=(--enforce-eager); graph_env=()
else
    graph_args=(--compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"cudagraph_capture_sizes\":[$capture_sizes]}")
    graph_env=(-e VLLM_USE_BREAKABLE_CUDAGRAPH=1)
fi
text_args=(); [ "$TEXT_ONLY" = 1 ] && text_args=(--language-model-only)
mm_args=(); [ "$TEXT_ONLY" = 1 ] || mm_args=(--limit-mm-per-prompt '{"image":4}' --mm-processor-cache-gb 1)
# shellcheck disable=SC2206
served=($SERVED_MODEL_NAME); served_args=(); for n in "${served[@]}"; do served_args+=(--served-model-name "$n"); done
# Optional bearer-key enforcement: API_KEY_FILE holds one key per line and every
# non-empty line becomes an accepted key (vLLM: --api-key K1 K2 ...). Unset = open
# server; do not put an open rank 0 behind a public route.
key_args=()
if [ -n "${API_KEY_FILE:-}" ]; then
    [ -r "$API_KEY_FILE" ] || die "API_KEY_FILE is not readable: $API_KEY_FILE"
    api_keys=()
    while IFS= read -r line || [ -n "$line" ]; do
        line=${line%$'\r'}
        case "$line" in *[![:space:]]*) api_keys+=("$line") ;; esac
    done < "$API_KEY_FILE"
    [ "${#api_keys[@]}" -gt 0 ] || die "API_KEY_FILE has no keys: $API_KEY_FILE"
    key_args=(--api-key "${api_keys[@]}")
fi

command=(
    docker run -d --name "$container_name" --restart no --pull never
    --network host --ipc host --shm-size 32g --oom-score-adj 500
    --gpus all --ulimit memlock=-1:-1 --cap-add IPC_LOCK --device /dev/infiniband:/dev/infiniband
    -v "$MODEL_HOST_PATH:$model_container_path:ro"
    -v "$CACHE_HOST_PATH:/cache"
    -v "$NCCL_SO_HOST_PATH:/opt/sparkring/nccl/libnccl.so.2:ro"
    "${patch_mounts[@]}"
    --env-file "$env_file"
    -e LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2 -e VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2
    -e NCCL_NET=IB -e NCCL_NET_PLUGIN=none -e NCCL_IB_DISABLE=0 -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1
    -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
    -e VLLM_CACHE_ROOT=/cache/vllm -e TILELANG_CACHE_DIR=/cache/tilelang -e TRITON_CACHE_DIR=/cache/triton
    -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    -e VLLM_USE_RUST_FRONTEND=0 -e VLLM_HAS_FLASHINFER_CUBIN=1 -e VLLM_USE_FLASHINFER_SAMPLER=0
    -e MAX_JOBS=2 -e FLASHINFER_NVCC_THREADS=1 -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0
    -e TORCH_CUDA_ARCH_LIST=12.1a -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e FLASHINFER_DISABLE_VERSION_CHECK=1
    -e DSV41_ENGRAM_DISK=1 -e "DSV41_ENGRAM_DISK_THREADS=$ENGRAM_DISK_THREADS" -e "DSV41_ENGRAM_DISK_CHUNK=$ENGRAM_DISK_CHUNK"
    -e "DSV41_ENGRAM_BALANCED=$ENGRAM_BALANCED" -e "DSV41_ENGRAM_PACKED_DIR=${ENGRAM_PACKED_DIR:-}"
    ${graph_env[@]+"${graph_env[@]}"}
    "$IMAGE"
    "$model_container_path"
    --host 0.0.0.0 --port "$API_PORT"
    "${served_args[@]}"
    ${key_args[@]+"${key_args[@]}"}
    --tensor-parallel-size 4 --nnodes 4 --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --distributed-executor-backend mp
    --load-format safetensors
    --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" --block-size 128
    --engram-config '{"cpu_offload": false}'
    --speculative-config "{\"method\":\"dspark\",\"num_speculative_tokens\":$k,\"draft_sample_method\":\"$DRAFT_SAMPLE_METHOD\",\"rejection_sample_method\":\"block\",\"enable_adaptive_verification\":false}"
    --tool-call-parser deepseek_v41 --enable-auto-tool-choice --reasoning-parser deepseek_v41
    --default-chat-template-kwargs "{\"thinking\": $THINKING_DEFAULT}"
    ${text_args[@]+"${text_args[@]}"} ${mm_args[@]+"${mm_args[@]}"} "${graph_args[@]}" ${headless[@]+"${headless[@]}"}
)

if [ "$mode" = --check ]; then
    echo "# rank $NODE_RANK image=$IMAGE model=$MODEL_HOST_PATH"
    echo "# MAX_MODEL_LEN=$MAX_MODEL_LEN MAX_NUM_SEQS=$MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS=$MAX_NUM_BATCHED_TOKENS GPU_MEMORY_UTILIZATION=$GPU_MEMORY_UTILIZATION NUM_SPECULATIVE_TOKENS=$NUM_SPECULATIVE_TOKENS ENFORCE_EAGER=$ENFORCE_EAGER TEXT_ONLY=$TEXT_ONLY"
    display=("${command[@]}")
    for ((index = 0; index < ${#display[@]}; index++)); do
        if [ "${display[index]}" = --api-key ]; then
            for ((key_index = 1; key_index <= ${#api_keys[@]}; key_index++)); do
                display[index + key_index]='<redacted>'
            done
            break
        fi
    done
    printf '%q ' "${display[@]}"; echo
    exit 0
fi

command -v docker >/dev/null || die "docker is not installed"
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image not present locally: $IMAGE (build it with runtime/deepseek-v41-gb10/build-image.sh)"
if [ -n "${IMAGE_ID:-}" ]; then
    actual=$(docker image inspect --format '{{.Id}}' "$IMAGE")
    [ "$actual" = "$IMAGE_ID" ] || die "image identity mismatch: expected $IMAGE_ID, got $actual"
fi
if docker ps -a --format '{{.Names}}' | grep -qx "$container_name"; then
    die "container $container_name already exists; remove it intentionally before relaunching"
fi
avail_gib=$(( $(awk '/MemAvailable/{print $2}' /proc/meminfo) / 1048576 ))
[ "$avail_gib" -ge 100 ] || die "MemAvailable ${avail_gib} GiB < 100 GiB; the rank needs ~86 GiB for weights and buffers plus the KV pool (reboot the rank first)"
"${command[@]}"
echo "launched $container_name rank=$NODE_RANK image=$IMAGE MemAvailable=${avail_gib}GiB"
