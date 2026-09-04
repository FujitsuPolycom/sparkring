#!/usr/bin/env bash
# Validate or start one rank of the DeepSeek-V4-Flash-0731 two-Spark profile.
# The per-rank env file is both the host launch contract and the container
# environment, so operator-facing paths and serving values have one source.
set -euo pipefail

usage() {
    echo "usage: deepseek_v4_pair_serve.sh [--check|--run] ENV_FILE" >&2
}

die() {
    echo "deepseek pair launcher: $*" >&2
    exit 20
}

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# shellcheck source=scripts/nccl_ib_gid_policy.sh
. "$script_dir/nccl_ib_gid_policy.sh"

case "$#" in
    1) mode=--check; env_file=$1 ;;
    2) mode=$1; env_file=$2 ;;
    *) usage; exit 64 ;;
esac
case "$mode" in
    --check|--run) ;;
    *) usage; exit 64 ;;
esac

[ -f "$env_file" ] || die "environment file is missing: $env_file"
env_file=$(cd "$(dirname "$env_file")" && pwd)/$(basename "$env_file")

if grep -Ev '^[[:space:]]*(#|$)' "$env_file" \
    | grep -Eq '<[A-Za-z0-9_]+>|REPLACE_WITH_'; then
    die "environment file contains unresolved placeholders: $env_file"
fi

# shellcheck disable=SC1090
. "$env_file"
# Keep the container-only preload value available for validation without
# exporting it to host commands such as grep and docker.
export -n LD_PRELOAD

require_value() {
    local name=$1 value
    value=${!name-}
    [ -n "$value" ] || die "required value is empty: $name"
}

require_directory() {
    local name=$1 value
    value=${!name-}
    case "$value" in
        /*) ;;
        *) die "$name must be an absolute host path: $value" ;;
    esac
    [ -d "$value" ] || die "$name directory does not exist: $value"
}

require_positive_integer() {
    local name=$1 value
    value=${!name-}
    case "$value" in
        ''|*[!0-9]*) die "$name must be a positive integer: $value" ;;
    esac
    [ "$((10#$value))" -gt 0 ] || die "$name must be greater than zero"
}

require_port() {
    local name=$1 value
    require_positive_integer "$name"
    value=${!name-}
    [ "$((10#$value))" -le 65535 ] || die "$name must be in 1..65535: $value"
}

for name in \
    NODE_RANK MASTER_ADDR MODEL_HOST_PATH CACHE_HOST_PATH API_PORT MASTER_PORT \
    NUM_SPECULATIVE_TOKENS MAX_MODEL_LEN MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS \
    LD_PRELOAD VLLM_NCCL_SO_PATH NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
    VLLM_HOST_IP NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA \
    NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_MERGE_NICS \
    NCCL_PROTO NCCL_P2P_LEVEL NCCL_CROSS_NIC NCCL_CUMEM_ENABLE \
    NCCL_IGNORE_CPU_AFFINITY; do
    require_value "$name"
done

case "$NODE_RANK" in
    0|1) ;;
    *) die "NODE_RANK must be 0 or 1: $NODE_RANK" ;;
esac
require_directory MODEL_HOST_PATH
require_directory CACHE_HOST_PATH
[ -r "$MODEL_HOST_PATH" ] || die "MODEL_HOST_PATH is not readable: $MODEL_HOST_PATH"
[ -w "$CACHE_HOST_PATH" ] || die "CACHE_HOST_PATH is not writable: $CACHE_HOST_PATH"

for name in NUM_SPECULATIVE_TOKENS MAX_MODEL_LEN MAX_NUM_SEQS \
    MAX_NUM_BATCHED_TOKENS; do
    require_positive_integer "$name"
done
require_port API_PORT
require_port MASTER_PORT
[ "$API_PORT" != "$MASTER_PORT" ] || die "API_PORT and MASTER_PORT must differ"

[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] \
    || die "NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME must match on a pair"
[ "$NCCL_NET" = IB ] || die "NCCL_NET must be IB"
[ "$NCCL_NET_PLUGIN" = none ] || die "NCCL_NET_PLUGIN must be none"
[ "$NCCL_IB_DISABLE" = 0 ] || die "NCCL_IB_DISABLE must be 0"
[ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 0 ] \
    || die "pair subnet-aware routing must be disabled"
[ "$NCCL_IB_MERGE_NICS" = 0 ] || die "pair NIC merging must be disabled"
[ "$NCCL_PROTO" = LL,LL128,Simple ] \
    || die "NCCL_PROTO must be LL,LL128,Simple"
[ "$NCCL_P2P_LEVEL" = SYS ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_CROSS_NIC" = 1 ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_CUMEM_ENABLE" = 0 ] || die "NCCL_CUMEM_ENABLE must be 0"
[ "$NCCL_IGNORE_CPU_AFFINITY" = 1 ] \
    || die "NCCL_IGNORE_CPU_AFFINITY must be 1"
case ":$LD_PRELOAD:" in
    *":$VLLM_NCCL_SO_PATH:"*) ;;
    *) die "LD_PRELOAD must include VLLM_NCCL_SO_PATH ($VLLM_NCCL_SO_PATH)" ;;
esac
[ "$NODE_RANK" != 0 ] || [ "$MASTER_ADDR" = "$VLLM_HOST_IP" ] \
    || die "rank-0 MASTER_ADDR must equal rank-0 VLLM_HOST_IP"

sparkring_validate_nccl_gid_policy
if [ "$NCCL_IB_GID_AUTO" = 1 ]; then
    [ "$SPARKRING_NCCL_SELECTED_COUNT" = 1 ] \
        || die "the pair NCCL_IB_HCA selector must resolve to exactly one active HCA/port; resolved $SPARKRING_NCCL_SELECTED_COUNT"
    # A bare --env overrides the env-file entry. Automatic policy unsets the
    # host NCCL_IB_GID_INDEX, so Docker omits it instead of treating an empty
    # value as NCCL's index 0.
    gid_env_args=(--env NCCL_IB_GID_INDEX)
else
    case "$NCCL_IB_HCA" in
        *,*) die "the pair environment must name exactly one RoCE device" ;;
    esac
    gid_env_args=()
fi

image=ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:827a8e8c5749b78529cc0015dd174e1b19a0accc116bc142282f8b75428f98bd
container_name="deepseek-v4-flash-r$NODE_RANK"
model_container_path=/models/deepseek-v4-flash-0731
speculative_config=$(printf \
    '{"method":"dspark","num_speculative_tokens":%s,"moe_backend":"b12x"}' \
    "$NUM_SPECULATIVE_TOKENS")

command=(
    docker run -d
    --name "$container_name"
    --pull never
    --network host
    --ipc host
    --shm-size 16g
    --gpus all
    --ulimit memlock=-1:-1
    --device /dev/infiniband
    -v "$MODEL_HOST_PATH:$model_container_path:ro"
    -v "$CACHE_HOST_PATH:/cache"
    --env-file "$env_file"
    "${gid_env_args[@]}"
    --entrypoint /opt/venv/bin/vllm
    "$image"
    serve "$model_container_path"
    --tensor-parallel-size 2
    --nnodes 2
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --distributed-executor-backend mp
    --dtype bfloat16
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-seqs "$MAX_NUM_SEQS"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --async-scheduling
    --scheduler-reserve-full-isl
    --gpu-memory-utilization 0.70
    --kv-cache-memory-bytes 17179869184
    --kv-cache-dtype fp8_ds_mla
    --block-size 256
    --tokenizer-mode deepseek_v4
    --kernel-config '{"enable_cutedsl_warmup":false}'
    --enable-auto-tool-choice
    --tool-call-parser deepseek_v4
    --speculative-config "$speculative_config"
    --served-model-name deepseek-v4-flash-0731
)

if [ "$NODE_RANK" = 0 ]; then
    command+=(--host 0.0.0.0 --port "$API_PORT")
else
    command+=(--headless)
fi

printf "Local rank input checks passed.\n"
printf '  rank: %s\n' "$NODE_RANK"
printf '  model: %s\n' "$MODEL_HOST_PATH"
printf '  cache: %s\n' "$CACHE_HOST_PATH"
printf '  MAX_MODEL_LEN: %s\n' "$MAX_MODEL_LEN"
printf '  MAX_NUM_SEQS: %s\n' "$MAX_NUM_SEQS"
printf '  MAX_NUM_BATCHED_TOKENS: %s\n' "$MAX_NUM_BATCHED_TOKENS"
printf '  NUM_SPECULATIVE_TOKENS: %s\n' "$NUM_SPECULATIVE_TOKENS"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

[ "$mode" = --run ] || exit 0
command -v docker >/dev/null 2>&1 || die "docker is unavailable"
docker image inspect "$image" >/dev/null 2>&1 \
    || die "pinned image is not present; pull it before launching: $image"
if docker container inspect "$container_name" >/dev/null 2>&1; then
    die "container already exists; remove it intentionally before relaunch: $container_name"
fi
exec "${command[@]}"
