#!/usr/bin/env bash
# Start one rank of the Qwen3.8-27B EXL3 K5/K6 two-Spark research profile.
# Rank 0 serves the API; rank 1 runs headless. Pass --check to verify immutable
# inputs, pair transport, model bytes and port availability without starting.
set -euo pipefail

usage() { echo "usage: qwen38_dgx2_serve.sh [--check|--run]" >&2; }
case "${1:---run}" in
    --check) check_only=1 ;;
    --run) check_only=0 ;;
    *) usage; exit 64 ;;
esac

die() { echo "qwen38 pair preflight: $*" >&2; exit 20; }
require_file() { [ -f "$1" ] || die "$2: $1"; }
require_directory() { [ -d "$1" ] || die "$2: $1"; }
require_env() {
    local name=$1
    local value
    value=$(printenv "$name" 2>/dev/null || true)
    [ -n "$value" ] || die "required rank environment value is empty: $name"
    case "$value" in *'<'*'>'*|*REPLACE_WITH_*) die "unresolved value: $name=$value" ;; esac
}
check_sha256() {
    local label=$1 path=$2 expected=$3 actual
    require_file "$path" "$label is missing"
    actual=$(sha256sum "$path" | cut -d' ' -f1)
    [ "$actual" = "$expected" ] || die "$label SHA-256 mismatch: expected $expected, got $actual"
}

rank=${RANK:-}
case "$rank" in 0|1) ;; *) die "RANK must be 0 or 1; got ${rank:-empty}" ;; esac
rank0_rendezvous_addr=${RANK0_RENDEZVOUS_ADDR:-}
[ -n "$rank0_rendezvous_addr" ] || die "RANK0_RENDEZVOUS_ADDR is required"

env_file=${QWEN_ENV_FILE:-/ws/rank.env}
model_path=${QWEN_MODEL_PATH:-/ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated}
chat_template=${QWEN_CHAT_TEMPLATE:-/ws/chat_template_agentic.jinja}
venv=${QWEN_VENV:-/ws/venv}
infiniband_root=${QWEN_INFINIBAND_SYS_ROOT:-/sys/class/infiniband}
infiniband_dev_root=${QWEN_INFINIBAND_DEV_ROOT:-/dev/infiniband}
api_port=${QWEN_API_PORT:-8000}
master_port=${QWEN_MASTER_PORT:-29500}

expected_manifest=7626d18481e7f995fd1d9ff211083b7fd57f044daba39e107fb29a48207f24c4
expected_config=fbb105334da6554c10784ff1257fda5e3821d4d5426d64469cee2b2ad67ba2b3
expected_index=ea6e0e1064efbb72d89b4a6f9e0ee76c909a94b3f25047487a2ffb282896a26c
expected_chat=4f9201169f5bacd1a494c8824470a1ef899c7024d23a2b166e42493e7efd9ac9
expected_nccl=e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54

for command_name in cut find grep ip pgrep printenv python3 sha256sum; do
    command -v "$command_name" >/dev/null 2>&1 || die "required command is unavailable: $command_name"
done
require_file "$env_file" "rank environment file is missing"
if grep -Ev '^[[:space:]]*(#|$)' "$env_file" | grep -Eq '<[A-Za-z0-9_]+>|REPLACE_WITH_'; then
    die "rank environment file contains unresolved placeholders: $env_file"
fi
set -a
# shellcheck disable=SC1090
. "$env_file"
set +a

for name in \
    LD_PRELOAD VLLM_NCCL_SO_PATH NCCL_SOCKET_IFNAME GLOO_SOCKET_IFNAME \
    VLLM_HOST_IP NCCL_NET NCCL_NET_PLUGIN NCCL_IB_DISABLE NCCL_IB_HCA \
    NCCL_IB_GID_INDEX NCCL_IB_SUBNET_AWARE_ROUTING NCCL_IB_MERGE_NICS \
    NCCL_PROTO NCCL_P2P_LEVEL NCCL_CROSS_NIC NCCL_CUMEM_ENABLE \
    NCCL_IGNORE_CPU_AFFINITY VLLM_EXL3_GRAPH_DECODE \
    VLLM_EXL3_PREFILL_FP8 VLLM_EXL3_PREFILL_RECONSTRUCT_M \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN; do
    require_env "$name"
done

[ "$NCCL_SOCKET_IFNAME" = "$GLOO_SOCKET_IFNAME" ] || die "NCCL and Gloo interfaces differ"
case ":$LD_PRELOAD:" in
    *":$VLLM_NCCL_SO_PATH:"*) ;;
    *) die "LD_PRELOAD must include VLLM_NCCL_SO_PATH ($VLLM_NCCL_SO_PATH)" ;;
esac
[ "$NCCL_NET" = IB ] || die "NCCL_NET must be IB"
[ "$NCCL_NET_PLUGIN" = none ] || die "NCCL_NET_PLUGIN must be none"
[ "$NCCL_IB_DISABLE" = 0 ] || die "NCCL_IB_DISABLE must be 0"
[ "$NCCL_IB_SUBNET_AWARE_ROUTING" = 0 ] || die "pair subnet-aware routing must be 0"
[ "$NCCL_IB_MERGE_NICS" = 0 ] || die "pair NIC merging must be 0"
[ "$NCCL_PROTO" = LL,LL128,Simple ] || die "NCCL_PROTO must be LL,LL128,Simple"
[ "$NCCL_P2P_LEVEL" = SYS ] || die "NCCL_P2P_LEVEL must be SYS"
[ "$NCCL_CROSS_NIC" = 1 ] || die "NCCL_CROSS_NIC must be 1"
[ "$NCCL_CUMEM_ENABLE" = 0 ] || die "NCCL_CUMEM_ENABLE must be 0"
[ "$NCCL_IGNORE_CPU_AFFINITY" = 1 ] || die "NCCL_IGNORE_CPU_AFFINITY must be 1"
[ "$VLLM_EXL3_GRAPH_DECODE" = 1 ] || die "VLLM_EXL3_GRAPH_DECODE must be 1"
[ "$VLLM_EXL3_PREFILL_FP8" = 1 ] || die "VLLM_EXL3_PREFILL_FP8 must be 1"
[ "$VLLM_EXL3_PREFILL_RECONSTRUCT_M" = 256 ] || die "reconstruction tile must be 256"
[ "$VLLM_ALLOW_LONG_MAX_MODEL_LEN" = 1 ] || die "long-context override gate must be 1"
for forbidden in NCCL_ALGO NCCL_MIN_NCHANNELS NCCL_MAX_NCHANNELS \
    NCCL_SKIP_TREE_CONNECT NCCL_IB_SUBNET_PREFIX_LEN; do
    [ -z "$(printenv "$forbidden" 2>/dev/null || true)" ] || {
        die "cycle-only transport value must be unset on a pair: $forbidden"
    }
done

python3 - "$VLLM_HOST_IP" "$rank0_rendezvous_addr" "$api_port" "$master_port" <<'PY' || die "invalid address or port"
import ipaddress
import sys

ipaddress.IPv4Address(sys.argv[1])
ipaddress.IPv4Address(sys.argv[2])
for raw in sys.argv[3:]:
    value = int(raw)
    if not 1 <= value <= 65535:
        raise ValueError(f"port outside 1..65535: {value}")
if sys.argv[3] == sys.argv[4]:
    raise ValueError("API and master ports must differ")
PY
[ "$rank" != 0 ] || [ "$rank0_rendezvous_addr" = "$VLLM_HOST_IP" ] || {
    die "rank-0 rendezvous address must equal rank-0 VLLM_HOST_IP"
}

case "$NCCL_IB_HCA" in *,*) die "the pair environment must name exactly one RoCE device" ;; esac
hca=${NCCL_IB_HCA%%:*}
port=${NCCL_IB_HCA#*:}
[ "$port" != "$NCCL_IB_HCA" ] || port=1
case "$port" in ''|*[!0-9]*) die "invalid HCA port: $NCCL_IB_HCA" ;; esac
case "$NCCL_IB_GID_INDEX" in ''|*[!0-9]*) die "GID index must be numeric" ;; esac

gid_root="$infiniband_root/$hca/ports/$port"
require_directory "$gid_root" "RoCE device/port is missing"
require_file "$gid_root/gids/$NCCL_IB_GID_INDEX" "GID entry is missing"
require_file "$gid_root/gid_attrs/types/$NCCL_IB_GID_INDEX" "GID type is missing"
require_file "$gid_root/gid_attrs/ndevs/$NCCL_IB_GID_INDEX" "GID netdev is missing"
gid=$(<"$gid_root/gids/$NCCL_IB_GID_INDEX")
gid_type=$(<"$gid_root/gid_attrs/types/$NCCL_IB_GID_INDEX")
gid_ndev=$(<"$gid_root/gid_attrs/ndevs/$NCCL_IB_GID_INDEX")
[ -n "$gid" ] && [ "$gid" != :: ] && [ "$gid" != 0000:0000:0000:0000:0000:0000:0000:0000 ] || die "selected GID entry is empty"
[[ "$gid_type" == *"RoCE v2"* ]] || die "selected GID is not RoCE v2: $gid_type"
[ "$gid_ndev" = "$NCCL_SOCKET_IFNAME" ] || die "GID netdev differs from rendezvous interface"
require_directory "$infiniband_dev_root" "infiniband device root is missing"
find "$infiniband_dev_root" -maxdepth 1 -type c -name 'uverbs*' -print -quit | grep -q . || {
    die "no infiniband uverbs device is available"
}
ip link show dev "$NCCL_SOCKET_IFNAME" >/dev/null 2>&1 || die "fabric interface is missing"
ip -o -4 addr show dev "$NCCL_SOCKET_IFNAME" | grep -Fq " $VLLM_HOST_IP/" || die "VLLM_HOST_IP is not configured on $NCCL_SOCKET_IFNAME"
python3 - "$gid" "$VLLM_HOST_IP" <<'PY' || die "selected GID does not encode VLLM_HOST_IP"
import ipaddress
import sys

mapped = ipaddress.IPv6Address(sys.argv[1]).ipv4_mapped
if mapped is None or str(mapped) != sys.argv[2]:
    raise ValueError("RoCEv2 GID does not encode the rank fabric address")
PY

require_file "$venv/bin/vllm" "vLLM entrypoint is missing"
require_directory "$model_path" "model directory is missing"
check_sha256 "model SHA256SUMS" "$model_path/SHA256SUMS" "$expected_manifest"
check_sha256 "model config" "$model_path/config.json" "$expected_config"
check_sha256 "model weight index" "$model_path/model.safetensors.index.json" "$expected_index"
check_sha256 "chat template" "$chat_template" "$expected_chat"
check_sha256 "patched NCCL library" "$VLLM_NCCL_SO_PATH" "$expected_nccl"
entries=$(grep -Ec '^[0-9a-fA-F]{64}  ' "$model_path/SHA256SUMS" || true)
[ "$entries" = 16 ] || die "model SHA256SUMS must contain 16 entries; found $entries"
(cd "$model_path" && sha256sum --check --strict --status SHA256SUMS) || die "one or more model files failed SHA256SUMS verification"
"$venv/bin/python" /ws/runtime/verify_runtime.py --imports >/dev/null || die "runtime identity or import verification failed"
"$venv/bin/python" - <<'PY' || die "CUDA, SM121, or libibverbs verification failed"
import ctypes
import torch

ctypes.CDLL("libibverbs.so.1")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable")
if torch.cuda.device_count() != 1:
    raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
if torch.cuda.get_device_capability(0) != (12, 1):
    raise RuntimeError(f"expected SM121, got {torch.cuda.get_device_capability(0)}")
PY
pgrep -f '[v]llm serve' >/dev/null 2>&1 && die "a vLLM serving process is already running"

if [ "$rank" = 0 ]; then
    for port_number in "$api_port" "$master_port"; do
        "$venv/bin/python" -c 'import socket,sys;s=socket.socket();s.bind(("0.0.0.0",int(sys.argv[1])));s.close()' "$port_number" 2>/dev/null || die "required rank-0 port is bound: $port_number"
    done
fi

endpoint=(--headless)
[ "$rank" = 0 ] && endpoint=(--host 0.0.0.0 --port "$api_port")
hf_overrides='{"text_config":{"rope_parameters":{"mrope_interleaved":true,"mrope_section":[11,11,10],"rope_type":"yarn","rope_theta":10000000,"partial_rotary_factor":0.25,"factor":4.0,"original_max_position_embeddings":262144}}}'
speculation='{"method":"qwen3_5_mtp","num_speculative_tokens":3,"attention_backend":"TRITON_ATTN","draft_sample_method":"probabilistic","rejection_sample_method":"standard"}'

command=(
    "$venv/bin/vllm" serve "$model_path"
    --served-model-name qwen38 --quantization exl3
    --tensor-parallel-size 2 --decode-context-parallel-size 1
    --nnodes 2 --node-rank "$rank"
    --master-addr "$rank0_rendezvous_addr" --master-port "$master_port"
    --distributed-executor-backend mp
    --max-model-len 1000000 --hf-overrides "$hf_overrides"
    --max-num-seqs 32 --max-num-batched-tokens 8192
    --enable-chunked-prefill --async-scheduling --scheduler-reserve-full-isl
    --block-size 16 --gpu-memory-utilization 0.70 --kv-cache-dtype fp8
    --enable-prefix-caching --mamba-cache-mode align
    --attention-backend TRITON_ATTN
    --mm-processor-kwargs '{"truncation":false}'
    --mm-encoder-attn-backend TORCH_SDPA
    --speculative-config "$speculation"
    --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'
    --chat-template "$chat_template" --reasoning-parser qwen3
    --tool-call-parser qwen3_coder --enable-auto-tool-choice
    "${endpoint[@]}"
)

printf 'qwen38 pair preflight passed for rank %s\n' "$rank"
printf 'resolved command:'
printf ' %q' "${command[@]}"
printf '\n'
[ "$check_only" = 1 ] && exit 0
exec "${command[@]}"
