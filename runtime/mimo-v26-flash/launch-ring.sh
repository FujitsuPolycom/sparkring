#!/bin/bash
# Start one rank of MiMo-V2.6-Flash-RL with its bundled DFlash draft on a
# four-Spark cycle (TP4) with SIRCL collectives over the hardware-forwarded
# managed mesh.
#
#   bash launch-ring.sh --check RANK_ENV_FILE   # validate inputs, start nothing
#   bash launch-ring.sh --run RANK_ENV_FILE     # start this rank's container
#
# RANK_ENV_FILE is a rank-local copy of ring.env.example with every REPLACE_
# value filled in; it names the rank's SIRCL environment file (a filled copy
# of sircl-rank.env.example). Start ranks 3, 2 and 1 before rank 0; rank 0
# serves the API.
#
# The image entrypoint attests the site-packages hashes and rejects the model
# overlay, so the vLLM CLI is exec'd directly with the argv the entrypoint
# would exec, from a login shell so /etc/shinit_v2 activates the CUDA
# forward-compatibility driver (see launch-pair.sh for the reason).
#
# Overlay files (this directory's overlay/) and the switches that select them
# are described in README.md. Serving settings are recorded in
# profiles/mimo-v26-flash-rl-tp4/recipe.json; tunables below default to that
# recipe.
set -euo pipefail
[ $# -eq 2 ] && { [ "$1" = --check ] || [ "$1" = --run ]; } || { echo "usage: bash launch-ring.sh --check|--run RANK_ENV_FILE" >&2; exit 2; }
ACTION="$1"; RANK_ENV_FILE="$2"
[ -f "$RANK_ENV_FILE" ] || { echo "missing $RANK_ENV_FILE" >&2; exit 2; }
if grep -v "^#" "$RANK_ENV_FILE" | grep -q "REPLACE_"; then echo "$RANK_ENV_FILE: replace every REPLACE_ value" >&2; exit 2; fi
# shellcheck disable=SC1090
. "$RANK_ENV_FILE"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY="${OVERLAY:-$HERE/overlay}"
for name in RANK HOST_IP MASTER_ADDR MGMT_IFNAME NCCL_IB_HCA_LIST MODEL_DIR CACHE_DIR IMAGE; do
  [ -n "${!name:-}" ] || { echo "$RANK_ENV_FILE: set $name" >&2; exit 2; }
done
case "$RANK" in 0|1|2|3) ;; *) echo "RANK must be 0..3" >&2; exit 2 ;; esac
NAME="${CONTAINER_NAME:-mimo-v26-flash-rl-tp4-r${RANK}}"
PORT="${PORT:-8020}"
MASTER_PORT="${MASTER_PORT:-29776}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-v2.6-flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
# 20 GiB of KV per rank leaves room for both SIRCL sessions (graph-only and
# fused prefill) beside the weights; a 30 GiB reservation exhausts device
# memory on one rank during warmup once those sessions are resident.
KV_BYTES="${KV_BYTES:-21474836480}"
GPU_MEM="${GPU_MEM:-0.80}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
ATTN="${ATTN:-TRITON_ATTN}"
# A C=8 decode step with a 5-token draft is 48 query rows; a 64-token capture
# ceiling keeps that step on the captured width-4096 SIRCL graph lane.
CG_CAP="${CG_CAP:-64}"
MM="${MM:-1}"
case "$MM" in
  1)     OMNI_FILE=mimo_v2_omni_mm.py; MM_ARGS=(--limit-mm-per-prompt '{"image":3,"video":1,"audio":1}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data --media-io-kwargs '{"video":{"num_frames":16},"audio":{"audio_backend":"torchcodec"}}') ;;
  image) OMNI_FILE=mimo_v2_omni_mm.py; MM_ARGS=(--limit-mm-per-prompt '{"image":3,"video":0,"audio":0}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data) ;;
  0)     OMNI_FILE=mimo_v2_omni.py;    MM_ARGS=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --mm-processor-cache-gb 0) ;;
  *) echo "MM must be 1, image or 0" >&2; exit 2 ;;
esac
if [ "$MM" != 0 ]; then AOT_COMPILE=0; else AOT_COMPILE="${AOT_COMPILE:-1}"; fi
# PAD_V=0 (ring default) runs the DiffKV kernel with the PR 839 split-KV
# dispatch; PAD_V=1 selects the padded-V symmetric kernel, which on this
# topology gains nothing and costs 17 percent of KV capacity.
PAD_V="${PAD_V:-0}"
if [ "$PAD_V" = 1 ]; then
  MIMO_V2_FILE=mimo_v2_padded_v.py
  KERNEL_MOUNT=(-v "$OVERLAY/triton_unified_attention.py":/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention.py:ro)
else
  MIMO_V2_FILE=mimo_v2.py
  KERNEL_MOUNT=(-v "$OVERLAY/triton_unified_attention_diffkv.py":/opt/venv/lib/python3.12/site-packages/vllm/v1/attention/ops/triton_unified_attention_diffkv.py:ro)
fi
EXTRA_ENV_ARGS=()
IFS=',' read -r -a _extra_kv <<< "${EXTRA_ENV:-}"
for kv in "${_extra_kv[@]}"; do [ -n "$kv" ] && EXTRA_ENV_ARGS+=(--env "$kv"); done
# SIRCL=1 (default) enables the SparkRing TP4 transport: the rank's SIRCL
# environment file (SPARK_TP4_* peers, devices, GIDs and ports; custom mode;
# vocabulary all-gather backend) plus PYTHONPATH to the image's SIRCL Python
# bundle, whose sitecustomize composes SIRCL with the RoCEnante
# virtual-diagonal overlay. That overlay needs the managed mesh's hardware
# relay rule (EtherType 0x88b5) on the rails; the launcher refuses to start
# without one. SIRCL=0 serves on PyNCCL over the four rails.
SIRCL="${SIRCL:-1}"
SIRCL_ARGS=(--env SIRCL_ENABLED=0)
if [ "$SIRCL" = 1 ]; then
  [ -n "${SIRCL_ENV_FILE:-}" ] && [ -f "$SIRCL_ENV_FILE" ] || { echo "$RANK_ENV_FILE: set SIRCL_ENV_FILE to the rank's SIRCL environment file" >&2; exit 2; }
  if grep -v "^#" "$SIRCL_ENV_FILE" | grep -q "REPLACE_"; then echo "$SIRCL_ENV_FILE: replace every REPLACE_ value" >&2; exit 2; fi
  # docker gives --env-file precedence over --env for duplicate names, so
  # names given in EXTRA_ENV are dropped from a per-launch copy of the file.
  SIRCL_EFFECTIVE="$CACHE_DIR/sircl.effective.env"
  mkdir -p "$CACHE_DIR"
  cp "$SIRCL_ENV_FILE" "$SIRCL_EFFECTIVE"
  for kv in "${_extra_kv[@]}"; do [ -n "$kv" ] && sed -i "/^${kv%%=*}=/d" "$SIRCL_EFFECTIVE"; done
  SIRCL_ARGS=(--env-file "$SIRCL_EFFECTIVE" --env PYTHONPATH=/opt/sparkring/sircl/python)
  found=0
  for d in /sys/class/net/*; do
    tc filter show dev "$(basename "$d")" ingress 2>/dev/null | grep -q 88b5 && found=1
  done
  [ "$found" = 1 ] || { echo "no EtherType 0x88b5 relay rule on any interface: start the managed mesh fabric service first, or launch with SIRCL=0" >&2; exit 1; }
fi
# At TP4 a global-attention layer's fused QKV slice is 3392 wide (16 query
# heads x 192 + one 192-wide K head + one 128-wide V head); the b12x fp8
# block-scaled GEMM requires multiples of 128, so linear kernels are chosen
# per layer.
LINEAR_BACKEND="${LINEAR_BACKEND:-auto}"
for f in config.json dflash/config.json dflash/dflash_draft_model.safetensors model.safetensors.index.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "missing $MODEL_DIR/$f" >&2; exit 1; }
done
for f in "$MIMO_V2_FILE" mimo_v2_mtp.py "$OMNI_FILE" qwen3_dflash.py; do
  [ -f "$OVERLAY/$f" ] || { echo "missing overlay file $OVERLAY/$f" >&2; exit 1; }
done
if [ "$(find "$MODEL_DIR" -name '*.incomplete' | wc -l)" != 0 ]; then
  echo "incomplete download files under $MODEL_DIR" >&2; exit 1
fi
mkdir -p "$CACHE_DIR"
DFLASH_MOUNT=()
if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$MODEL_DIR/dflash/config.json" 2>/dev/null; then
  python3 "$HERE/fix_dflash_config.py" "$MODEL_DIR/dflash/config.json" "$CACHE_DIR/dflash-config.json"
  DFLASH_MOUNT=(-v "$CACHE_DIR/dflash-config.json":/models/target/dflash/config.json:ro)
fi
if [ "$ACTION" = --check ]; then
  docker image inspect "$IMAGE" --format '{{.Id}}' >/dev/null || { echo "image $IMAGE is not present" >&2; exit 1; }
  echo "rank $RANK: inputs valid (SIRCL=$SIRCL MM=$MM PAD_V=$PAD_V KV_BYTES=$KV_BYTES SPEC_TOKENS=$SPEC_TOKENS CG_CAP=$CG_CAP LINEAR_BACKEND=$LINEAR_BACKEND)"
  exit 0
fi
P=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models
docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d --name "$NAME" \
  --gpus all --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  --env VLLM_HOST_IP="$HOST_IP" \
  --env NCCL_SOCKET_IFNAME="$MGMT_IFNAME" --env GLOO_SOCKET_IFNAME="$MGMT_IFNAME" \
  --env NCCL_NET=IB --env NCCL_NET_PLUGIN=none --env NCCL_IB_DISABLE=0 \
  --env NCCL_IB_HCA="=$NCCL_IB_HCA_LIST" \
  --env NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-3}" --env NCCL_IB_EXTENDED_IPV4_GIDS=0 \
  --env NCCL_IB_SUBNET_AWARE_ROUTING=1 --env NCCL_IB_MERGE_NICS=0 \
  --env NCCL_IB_PRESERVE_PCI_DOMAIN=1 --env NCCL_IB_ROUTE_DIAGNOSTICS=1 \
  --env NCCL_SWITCHLESS_RING_ONLY=1 --env NCCL_ALGO=Ring \
  --env NCCL_PROTO=LL,LL128,Simple --env NCCL_P2P_LEVEL=SYS \
  --env NCCL_MIN_NCHANNELS=4 --env NCCL_MAX_NCHANNELS=4 --env NCCL_CROSS_NIC=1 \
  --env NCCL_CUMEM_ENABLE=0 --env NCCL_IGNORE_CPU_AFFINITY=1 \
  --env NCCL_DEBUG=WARN --env NCCL_DEBUG_SUBSYS=NET,INIT,GRAPH \
  --env VLLM_ENABLE_PCIE_ALLREDUCE=0 --env VLLM_ALLREDUCE_USE_FLASHINFER=0 \
  --env VLLM_ALLREDUCE_USE_SYMM_MEM=0 \
  "${SIRCL_ARGS[@]}" --env SPARKCACHE_ENABLED=0 \
  --env VLLM_PLUGINS=b12x_loader --env CUTE_DSL_ARCH=sm_121a \
  --env FLASHINFER_CUDA_ARCH_LIST=12.1f --env OMP_NUM_THREADS=16 \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env XDG_CACHE_HOME=/cache/jit --env VLLM_CACHE_ROOT=/cache/jit/vllm \
  --env TRITON_CACHE_DIR=/cache/jit/triton --env TORCHINDUCTOR_CACHE_DIR=/cache/jit/torchinductor \
  --env B12X_ROCE_CACHE_DIR=/cache/jit/roce --env CUTE_DSL_CACHE_DIR=/cache/jit/cute \
  --env B12X_COMPILE_CACHE_DIR=/cache/jit/b12x --env B12X_CUTE_COMPILE_CACHE_DIR=/cache/jit/b12x \
  --env SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  --env VLLM_KV_CACHE_LAYOUT=BLHNC --env VLLM_USE_AOT_COMPILE="$AOT_COMPILE" \
  --env VLLM_MIMO_PAD_V="$PAD_V" --env VLLM_MIMO_PASS_CACHE_CONFIG=0 \
  "${EXTRA_ENV_ARGS[@]}" \
  -v "$MODEL_DIR":/models/target:ro \
  -v "$CACHE_DIR":/cache/jit \
  "${KERNEL_MOUNT[@]}" "${DFLASH_MOUNT[@]}" \
  -v "$OVERLAY/$MIMO_V2_FILE":$P/mimo_v2.py:ro \
  -v "$OVERLAY/mimo_v2_mtp.py":$P/mimo_v2_mtp.py:ro \
  -v "$OVERLAY/$OMNI_FILE":$P/mimo_v2_omni.py:ro \
  -v "$OVERLAY/qwen3_dflash.py":$P/qwen3_dflash.py:ro \
  --entrypoint /bin/sh "$IMAGE" -lc 'exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$@"' sh /models/target \
  --served-model-name "$SERVED_MODEL_NAME" --trust-remote-code \
  --node-rank "$RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --nnodes 4 \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --disable-custom-all-reduce \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --dtype bfloat16 --kv-cache-dtype fp8 --block-size 64 \
  --gpu-memory-utilization "$GPU_MEM" --kv-cache-memory-bytes "$KV_BYTES" \
  --attention-backend "$ATTN" --linear-backend "$LINEAR_BACKEND" --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching --enable-chunked-prefill \
  "${MM_ARGS[@]}" \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --generation-config vllm \
  --speculative-config "{\"model\":\"/models/target/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":${SPEC_TOKENS},\"attention_backend\":\"${ATTN}\"}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":${CG_CAP}}"
