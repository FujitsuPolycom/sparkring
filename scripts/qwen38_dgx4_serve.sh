#!/usr/bin/env bash
# Start one rank of the Qwen3.8-27B EXL3 K5/K6 four-Spark candidate.
# Run inside the prepared runtime container on every rank. Rank 0 serves the
# API; ranks 1-3 run headless workers. Pass --check to validate the complete
# local rank contract and print the command without starting vLLM.
set -euo pipefail

usage() {
    echo "usage: qwen38_dgx4_serve.sh [--check|--run]" >&2
}

check_only=0
case "$#" in
    0) ;;
    1)
        if [ "$1" = "--check" ]; then
            check_only=1
        elif [ "$1" = "--run" ]; then
            check_only=0
        else
            usage
            exit 64
        fi
        ;;
    *)
        usage
        exit 64
        ;;
esac

die() {
    echo "qwen38 preflight: $*" >&2
    exit 20
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_file() {
    [ -f "$1" ] || die "$2: $1"
}

require_directory() {
    [ -d "$1" ] || die "$2: $1"
}

check_sha256() {
    local label=$1
    local path=$2
    local expected=$3
    local output actual

    require_file "$path" "$label is missing"
    output=$(sha256sum "$path") || die "could not hash $label: $path"
    actual=${output%% *}
    [ "$actual" = "$expected" ] || {
        die "$label SHA-256 mismatch: expected $expected, got $actual"
    }
}

require_env_value() {
    local name=$1
    local value=${!name-}
    [ -n "$value" ] || die "required rank environment value is empty: $name"
    case "$value" in
        *'<'*'>'*|*REPLACE_WITH_*)
            die "required rank environment value is unresolved: $name=$value"
            ;;
    esac
}

case "${RANK-}" in
    ''|0|1|2|3) ;;
    *) die "RANK must be 0, 1, 2, or 3; got $RANK" ;;
esac
env_file=${QWEN_ENV_FILE:-/ws/rank.env}
require_file "$env_file" "rank environment file is missing"
if grep -Ev '^[[:space:]]*(#|$)' "$env_file" | \
    grep -Eq '<[A-Za-z0-9_]+>|REPLACE_WITH_'; then
    die "rank environment file contains unresolved placeholders: $env_file"
fi
set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

rank=${RANK:-}
rank0_rendezvous_addr=${RANK0_RENDEZVOUS_ADDR:-}
model_path=${QWEN_MODEL_PATH:-/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated}
chat_template=${QWEN_CHAT_TEMPLATE:-/ws/chat_template_agentic.jinja}
venv=${QWEN_VENV:-/ws/venv}
vllm_source=${QWEN_VLLM_SOURCE:-/ws/src/vllm-gg}
exllamav3_source=${QWEN_EXLLAMAV3_SOURCE:-/ws/src/exllamav3}
infiniband_dev_root=${QWEN_INFINIBAND_DEV_ROOT:-/dev/infiniband}
infiniband_sys_root=${QWEN_INFINIBAND_SYS_ROOT:-/sys/class/infiniband}
api_port=${API_PORT:-${QWEN_API_PORT:-8000}}
master_port=${MASTER_PORT:-${QWEN_MASTER_PORT:-29500}}
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-3}
max_model_len=${MAX_MODEL_LEN:-1048576}
max_num_seqs=${MAX_NUM_SEQS:-64}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}

expected_vllm_commit=229effc810ee6b8112f661472f6aace4eb8c787d
expected_exllamav3_commit=5f3c537ca9d89893d771256f5c43c93656553fbb
expected_exllamav3_patch_sha256=594b01547b0d801cf95926ea973719354150893121019aba2ad8832bc9f17fdb
expected_model_manifest_sha256=7626d18481e7f995fd1d9ff211083b7fd57f044daba39e107fb29a48207f24c4
expected_model_config_sha256=fbb105334da6554c10784ff1257fda5e3821d4d5426d64469cee2b2ad67ba2b3
expected_model_index_sha256=ea6e0e1064efbb72d89b4a6f9e0ee76c909a94b3f25047487a2ffb282896a26c
expected_chat_template_sha256=4f9201169f5bacd1a494c8824470a1ef899c7024d23a2b166e42493e7efd9ac9
expected_nccl_sha256=e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54

case "$rank" in
    0|1|2|3) ;;
    *) die "RANK must be 0, 1, 2, or 3; got ${rank:-empty}" ;;
esac
[ -n "$rank0_rendezvous_addr" ] || die "RANK0_RENDEZVOUS_ADDR is required"

command -v pgrep >/dev/null 2>&1 || {
    die "required command is unavailable: pgrep (install procps)"
}
for command_name in awk cut git grep ip sha256sum; do
    require_command "$command_name"
done

for name in \
    LD_PRELOAD VLLM_NCCL_SO_PATH NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
    VLLM_HOST_IP NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA \
    NCCL_IB_GID_INDEX \
    NCCL_IB_SUBNET_PREFIX_LEN NCCL_IB_SUBNET_AWARE_ROUTING \
    NCCL_IB_MERGE_NICS NCCL_ALGO NCCL_PROTO NCCL_P2P_LEVEL \
    NCCL_MIN_NCHANNELS \
    NCCL_MAX_NCHANNELS NCCL_CROSS_NIC NCCL_CUMEM_ENABLE \
    NCCL_SKIP_TREE_CONNECT NCCL_IGNORE_CPU_AFFINITY \
    VLLM_EXL3_GRAPH_DECODE VLLM_EXL3_PREFILL_FP8 \
    VLLM_EXL3_PREFILL_RECONSTRUCT_M VLLM_ALLOW_LONG_MAX_MODEL_LEN; do
    require_env_value "$name"
done

for value_name in num_speculative_tokens max_model_len max_num_seqs \
    max_num_batched_tokens; do
    value=${!value_name}
    case "$value" in
        ''|*[!0-9]*) die "$value_name must be a positive integer: $value" ;;
    esac
    [ "$((10#$value))" -gt 0 ] || die "$value_name must be greater than zero"
done

[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] || {
    die "NCCL_SOCKET_IFNAME and GLOO_SOCKET_IFNAME must name the same management interface"
}
[ "$NCCL_NET" = "IB" ] || die "NCCL_NET must be IB"
[ "$NCCL_NET_PLUGIN" = "none" ] || die "NCCL_NET_PLUGIN must be none"
[ "$NCCL_IB_DISABLE" = "0" ] || die "NCCL_IB_DISABLE must be 0"
[ "$NCCL_IB_SUBNET_PREFIX_LEN" = "24" ] || {
    die "NCCL_IB_SUBNET_PREFIX_LEN must be 24"
}
[ "$NCCL_IB_SUBNET_AWARE_ROUTING" = "1" ] || {
    die "NCCL_IB_SUBNET_AWARE_ROUTING must be 1 on the four-rank cycle"
}
[ "$NCCL_IB_MERGE_NICS" = "0" ] || die "NCCL_IB_MERGE_NICS must be 0"
[ "$NCCL_ALGO" = "Ring" ] || die "NCCL_ALGO must be Ring"
[ "$NCCL_PROTO" = "LL,LL128,Simple" ] || {
    die "NCCL_PROTO must be LL,LL128,Simple"
}
[ "$NCCL_P2P_LEVEL" = "SYS" ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_MIN_NCHANNELS" = "4" ] && [ "$NCCL_MAX_NCHANNELS" = "4" ] || {
    die "NCCL_MIN_NCHANNELS and NCCL_MAX_NCHANNELS must both be 4"
}
[ "$NCCL_CROSS_NIC" = "1" ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_CUMEM_ENABLE" = "0" ] || die "NCCL_CUMEM_ENABLE must be 0"
[ "$NCCL_SKIP_TREE_CONNECT" = "1" ] || die "NCCL_SKIP_TREE_CONNECT must be 1"
[ "$NCCL_IGNORE_CPU_AFFINITY" = "1" ] || {
    die "NCCL_IGNORE_CPU_AFFINITY must be 1"
}
[ "$VLLM_EXL3_GRAPH_DECODE" = "1" ] || die "VLLM_EXL3_GRAPH_DECODE must be 1"
[ "$VLLM_EXL3_PREFILL_FP8" = "1" ] || die "VLLM_EXL3_PREFILL_FP8 must be 1"
[ "$VLLM_EXL3_PREFILL_RECONSTRUCT_M" = "256" ] || {
    die "VLLM_EXL3_PREFILL_RECONSTRUCT_M must be 256"
}
[ "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" = "1" ] || {
    die "VLLM_ALLOW_LONG_MAX_MODEL_LEN must be 1 for the 1048576-token profile"
}
case "$NCCL_IB_GID_INDEX" in
    ''|*[!0-9]*) die "NCCL_IB_GID_INDEX must be a non-negative integer" ;;
esac

case ":$LD_PRELOAD:" in
    *":$VLLM_NCCL_SO_PATH:"*) ;;
    *) die "LD_PRELOAD must include VLLM_NCCL_SO_PATH ($VLLM_NCCL_SO_PATH)" ;;
esac

require_directory "$model_path" "model directory is missing"
require_file "$chat_template" "chat template is missing"
require_file "$venv/bin/activate" "Python environment activate script is missing"
require_file "$venv/bin/python" "Python executable is missing"
require_file "$venv/bin/vllm" "vLLM entrypoint is missing"
require_directory "$vllm_source" "vLLM source tree is missing"
require_directory "$exllamav3_source" "ExLlamaV3 source tree is missing"

vllm_commit=$(git -c safe.directory="$vllm_source" \
    -C "$vllm_source" rev-parse HEAD 2>/dev/null || true)
[ "$vllm_commit" = "$expected_vllm_commit" ] || {
    die "vLLM base commit mismatch: expected $expected_vllm_commit, got ${vllm_commit:-unavailable}"
}
if ! vllm_status=$(git -c safe.directory="$vllm_source" \
    -C "$vllm_source" status --porcelain --untracked-files=all); then
    die "could not inspect vLLM source state"
fi
[ -z "$vllm_status" ] || die "vLLM source tree differs from the pinned clean commit"

exllamav3_commit=$(git -c safe.directory="$exllamav3_source" \
    -C "$exllamav3_source" rev-parse HEAD 2>/dev/null || true)
[ "$exllamav3_commit" = "$expected_exllamav3_commit" ] || {
    die "ExLlamaV3 base commit mismatch: expected $expected_exllamav3_commit, got ${exllamav3_commit:-unavailable}"
}

declare -A expected_exllamav3_paths=(
    ["exllamav3/exllamav3_ext/avx2_target.cpp"]=1
    ["exllamav3/exllamav3_ext/avx512_target.cpp"]=1
    ["exllamav3/exllamav3_ext/cpu/arm_stubs.cpp"]=1
    ["exllamav3/exllamav3_ext/cpu/moe_handoff.cu"]=1
    ["exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu"]=1
    ["setup.py"]=1
)
declare -A expected_exllamav3_sha256=(
    ["exllamav3/exllamav3_ext/avx2_target.cpp"]="b26342bc6cb300587e5ed4ff77d75c21debfe034ee40634635ec455280ae6e8c"
    ["exllamav3/exllamav3_ext/avx512_target.cpp"]="9ba59543263693598713de192028627c8a249cc62f6311c682f64fbb0d69df8e"
    ["exllamav3/exllamav3_ext/cpu/arm_stubs.cpp"]="4abb18d5b6a99c9ce0e0b0f118a33e77a33417c6b471a02dc06792c78a007f2f"
    ["exllamav3/exllamav3_ext/cpu/moe_handoff.cu"]="71382fdc782877a6a0f8173615082964f70353a50e09f02cbe98ae0a0e7d8051"
    ["exllamav3/exllamav3_ext/parallel/all_reduce_cpu.cu"]="18e72f0c39c3a2447ab1d7de0f87c13e69f9216defa1c49f8bb5e31a40a9f14c"
    ["setup.py"]="29e339ee9205df20715d2cb876452569751270fbf62f4f045b358ed9949fb308"
)
declare -A seen_exllamav3_paths=()
while IFS= read -r line; do
    [ -n "$line" ] || continue
    path=${line:3}
    [ "${expected_exllamav3_paths[$path]-}" = "1" ] || {
        die "ExLlamaV3 source tree contains an unexpected change: $line"
    }
    seen_exllamav3_paths["$path"]=1
done < <(git -c safe.directory="$exllamav3_source" \
    -C "$exllamav3_source" status --porcelain --untracked-files=all)

for path in "${!expected_exllamav3_paths[@]}"; do
    [ "${seen_exllamav3_paths[$path]-}" = "1" ] || {
        die "ExLlamaV3 ARM patch state is missing: $path"
    }
    check_sha256 "ExLlamaV3 ARM patch file" \
        "$exllamav3_source/$path" "${expected_exllamav3_sha256[$path]}"
done

check_sha256 "model SHA256SUMS" "$model_path/SHA256SUMS" \
    "$expected_model_manifest_sha256"
check_sha256 "model config" "$model_path/config.json" \
    "$expected_model_config_sha256"
check_sha256 "model weight index" "$model_path/model.safetensors.index.json" \
    "$expected_model_index_sha256"
manifest_entries=$(grep -Ec '^[0-9a-fA-F]{64}  ' "$model_path/SHA256SUMS" || true)
[ "$manifest_entries" = "16" ] || {
    die "model SHA256SUMS must contain 16 entries; found $manifest_entries"
}
(cd "$model_path" && sha256sum --check --strict --status SHA256SUMS) || {
    die "one or more model files failed SHA256SUMS verification"
}

check_sha256 "chat template" "$chat_template" "$expected_chat_template_sha256"
check_sha256 "patched NCCL library" "$VLLM_NCCL_SO_PATH" "$expected_nccl_sha256"

"$venv/bin/python" - <<'PY' || die "runtime import or libibverbs check failed"
import ctypes

import torch
import vllm  # noqa: F401
from exllamav3_ext import exl3_gemm  # noqa: F401

ctypes.CDLL("libibverbs.so.1")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available inside the rank container")
if torch.cuda.device_count() != 1:
    raise RuntimeError(
        f"expected exactly one visible GPU, got {torch.cuda.device_count()}"
    )
capability = torch.cuda.get_device_capability(0)
if capability != (12, 1):
    raise RuntimeError(f"expected CUDA capability (12, 1), got {capability}")
PY

require_directory "$infiniband_dev_root" "/dev/infiniband equivalent is missing"
compgen -G "$infiniband_dev_root/uverbs*" >/dev/null || {
    die "no uverbs device exists under $infiniband_dev_root"
}
require_directory "$infiniband_sys_root" "infiniband sysfs root is missing"

"$venv/bin/python" - "$VLLM_HOST_IP" "$rank0_rendezvous_addr" \
    "$api_port" "$master_port" <<'PY' || die "management address or port is invalid"
import ipaddress
import sys

ipaddress.IPv4Address(sys.argv[1])
ipaddress.IPv4Address(sys.argv[2])
for value in sys.argv[3:]:
    port = int(value)
    if not 1 <= port <= 65535:
        raise ValueError(f"invalid port: {port}")
PY

ip link show dev "$NCCL_SOCKET_IFNAME" >/dev/null 2>&1 || {
    die "management interface does not exist: $NCCL_SOCKET_IFNAME"
}
management_ips=$(ip -o -4 addr show dev "$NCCL_SOCKET_IFNAME" | \
    awk '{print $4}' | cut -d/ -f1)
grep -Fxq "$VLLM_HOST_IP" <<< "$management_ips" || {
    die "VLLM_HOST_IP $VLLM_HOST_IP is not configured on $NCCL_SOCKET_IFNAME"
}
if [ "$rank" = "0" ] && [ "$VLLM_HOST_IP" != "$rank0_rendezvous_addr" ]; then
    die "rank 0 rendezvous address must equal rank 0 VLLM_HOST_IP"
fi

IFS=',' read -r -a hca_specs <<< "$NCCL_IB_HCA"
[ "${#hca_specs[@]}" = "2" ] || {
    die "NCCL_IB_HCA must name exactly two cycle-facing devices"
}
declare -A seen_hcas=()
for spec in "${hca_specs[@]}"; do
    case "$spec" in
        *' '*) die "NCCL_IB_HCA entries must not contain spaces: $spec" ;;
    esac
    hca=${spec%%:*}
    if [ "$hca" = "$spec" ]; then
        port=1
    else
        port=${spec#*:}
    fi
    [ -n "$hca" ] || die "NCCL_IB_HCA contains an empty device"
    case "$port" in
        ''|*[!0-9]*) die "invalid HCA port in NCCL_IB_HCA: $spec" ;;
    esac
    [ -z "${seen_hcas[$hca]-}" ] || die "NCCL_IB_HCA repeats device $hca"
    seen_hcas["$hca"]=1

    hca_port_root="$infiniband_sys_root/$hca/ports/$port"
    require_directory "$hca_port_root" "infiniband HCA/port is missing"
    gid_file="$hca_port_root/gids/$NCCL_IB_GID_INDEX"
    gid_type_file="$hca_port_root/gid_attrs/types/$NCCL_IB_GID_INDEX"
    gid_ndev_file="$hca_port_root/gid_attrs/ndevs/$NCCL_IB_GID_INDEX"
    require_file "$gid_file" "GID entry is missing for $spec"
    require_file "$gid_type_file" "GID type is missing for $spec"
    require_file "$gid_ndev_file" "GID netdev is missing for $spec"
    gid=$(<"$gid_file")
    gid_type=$(<"$gid_type_file")
    gid_ndev=$(<"$gid_ndev_file")
    [ -n "$gid" ] && [ "$gid" != "::" ] || die "GID entry is empty for $spec"
    [[ "$gid_type" == *"RoCE v2"* ]] || {
        die "GID index $NCCL_IB_GID_INDEX on $spec is not RoCE v2: $gid_type"
    }
    [ -n "$gid_ndev" ] || die "GID index $NCCL_IB_GID_INDEX on $spec has no netdev"
    ip link show dev "$gid_ndev" >/dev/null 2>&1 || {
        die "fabric netdev for $spec does not exist: $gid_ndev"
    }
    fabric_ips=$(ip -o -4 addr show dev "$gid_ndev" | \
        awk '{print $4}' | cut -d/ -f1)
    [ -n "$fabric_ips" ] || {
        die "fabric netdev for $spec has no IPv4 address: $gid_ndev"
    }
    "$venv/bin/python" - "$gid" $fabric_ips <<'PY' || {
import ipaddress
import sys

mapped = ipaddress.IPv6Address(sys.argv[1]).ipv4_mapped
if mapped is None or str(mapped) not in sys.argv[2:]:
    raise ValueError("RoCEv2 GID does not encode an IPv4 address on its netdev")
PY
        die "GID index $NCCL_IB_GID_INDEX on $spec does not match an IPv4 address on $gid_ndev"
    }
done

if pgrep -f '[v]llm serve' >/dev/null 2>&1; then
    die "a vLLM serving process is already running in this container"
fi

if [ "$rank" = "0" ]; then
    for port in "$api_port" "$master_port"; do
        "$venv/bin/python" -c \
            'import socket, sys; sock = socket.socket(); sock.bind(("0.0.0.0", int(sys.argv[1]))); sock.close()' \
            "$port" 2>/dev/null || die "required rank-0 port is already bound: $port"
    done
fi

# shellcheck disable=SC1091
. "$venv/bin/activate"

endpoint_args=(--headless)
if [ "$rank" = "0" ]; then
    endpoint_args=(--host 0.0.0.0 --port "$api_port")
fi

hf_overrides='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
speculation=$(printf '{"method":"qwen3_5_mtp","num_speculative_tokens":%s,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' "$num_speculative_tokens")

command=(
    "$venv/bin/vllm" serve "$model_path"
    --served-model-name qwen38
    --quantization exl3
    --tensor-parallel-size 4
    --decode-context-parallel-size 1
    --nnodes 4
    --node-rank "$rank"
    --master-addr "$rank0_rendezvous_addr"
    --master-port "$master_port"
    --distributed-executor-backend mp
    --max-model-len "$max_model_len"
    --hf-overrides "$hf_overrides"
    --max-num-seqs "$max_num_seqs"
    --max-num-batched-tokens "$max_num_batched_tokens"
    --enable-chunked-prefill
    --async-scheduling
    --scheduler-reserve-full-isl
    --block-size 16
    --gpu-memory-utilization 0.70
    --kv-cache-dtype fp8
    --enable-prefix-caching
    --mamba-cache-mode align
    --attention-backend TRITON_ATTN
    --mm-processor-kwargs '{"truncation":false}'
    --mm-encoder-attn-backend TORCH_SDPA
    --speculative-config "$speculation"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --chat-template "$chat_template"
    --reasoning-parser qwen3
    --tool-call-parser qwen3_coder
    --enable-auto-tool-choice
    "${endpoint_args[@]}"
)

printf 'qwen38 preflight passed for rank %s\n' "$rank"
printf 'verified ExLlamaV3 ARM patch SHA-256: %s\n' "$expected_exllamav3_patch_sha256"
printf 'MAX_MODEL_LEN=%s MAX_NUM_SEQS=%s MAX_NUM_BATCHED_TOKENS=%s NUM_SPECULATIVE_TOKENS=%s\n' \
    "$max_model_len" "$max_num_seqs" "$max_num_batched_tokens" \
    "$num_speculative_tokens"
printf 'resolved command:'
printf ' %q' "${command[@]}"
printf '\n'

if [ "$check_only" = "1" ]; then
    exit 0
fi

exec "${command[@]}"
