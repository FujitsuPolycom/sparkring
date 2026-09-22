#!/bin/bash
# Start one rank of MiMo-V2.6-Flash-RL with its bundled DFlash draft on a
# two-Spark pair (TP2, one direct cable, RoCEnante one-shot all-reduce).
#
#   bash launch-pair.sh --check RANK_ENV_FILE   # validate inputs, start nothing
#   bash launch-pair.sh --run RANK_ENV_FILE     # start this rank's container
#
# RANK_ENV_FILE is a rank-local copy of pair.env.example with every REPLACE_
# value filled in. Start rank 1 first, then rank 0; rank 0 serves the API.
#
# The image entrypoint attests the site-packages hashes and rejects the model
# overlay, so the vLLM CLI is exec'd directly with the argv the entrypoint
# would exec. It runs from a login shell so /etc/shinit_v2 activates the CUDA
# forward-compatibility driver the image carries: the FlashAttention-2 kernel
# used by the vision and audio encoders is CUDA 13.3 PTX, which the host
# driver alone cannot JIT ("the provided PTX was compiled with an unsupported
# toolchain"). Preloading libcuda without shinit mixes driver components and
# segfaults in the JIT entry point.
#
# Overlay files (this directory's overlay/, mounted read-only over the
# image's vllm package) and the switches that select them are described in
# README.md. Serving settings are recorded in
# profiles/mimo-v26-flash-rl-tp2/recipe.json; tunables below default to that
# recipe.
set -euo pipefail
[ $# -eq 2 ] && { [ "$1" = --check ] || [ "$1" = --run ]; } || { echo "usage: bash launch-pair.sh --check|--run RANK_ENV_FILE" >&2; exit 2; }
ACTION="$1"; RANK_ENV_FILE="$2"
[ -f "$RANK_ENV_FILE" ] || { echo "missing $RANK_ENV_FILE" >&2; exit 2; }
if grep -v "^#" "$RANK_ENV_FILE" | grep -q "REPLACE_"; then echo "$RANK_ENV_FILE: replace every REPLACE_ value" >&2; exit 2; fi
# shellcheck disable=SC1090
. "$RANK_ENV_FILE"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OVERLAY="${OVERLAY:-$HERE/overlay}"
for name in RANK HOST_IP MASTER_ADDR MGMT_IFNAME ROCE_HCA_PAIR MODEL_DIR CACHE_DIR IMAGE; do
  [ -n "${!name:-}" ] || { echo "$RANK_ENV_FILE: set $name" >&2; exit 2; }
done
case "$RANK" in 0|1) ;; *) echo "RANK must be 0 or 1" >&2; exit 2 ;; esac
NAME="${CONTAINER_NAME:-mimo-v26-flash-rl-tp2-r${RANK}}"
PORT="${PORT:-8000}"
MASTER_PORT="${MASTER_PORT:-29639}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-mimo-v2.6-flash}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_BATCHED="${MAX_BATCHED:-8192}"
# Per-rank memory: 121.7 GiB unified memory holds the weights (81 GiB), the
# encoders, the compiled graphs and the host processes. 12 GiB of KV with
# multimodal inputs enabled keeps about 10 GiB available on the host, which
# is what the pair's earlyoom threshold (4 percent) tolerates. 16 GiB is the
# text-only reservation.
KV_BYTES="${KV_BYTES:-12884901888}"
GPU_MEM="${GPU_MEM:-0.88}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
ATTN="${ATTN:-TRITON_ATTN}"
CG_CAP="${CG_CAP:-32}"
ROCE_AR="${ROCE_AR:-1}"
# MM=1 enables image (3), video (1) and audio (1) inputs through the omni
# class with the multimodal overlay; MM=image limits inputs to images; MM=0
# is text-only with the smaller omni overlay and the 16 GiB KV default.
MM="${MM:-1}"
case "$MM" in
  1)     OMNI_FILE=mimo_v2_omni_mm.py; MM_ARGS=(--limit-mm-per-prompt '{"image":3,"video":1,"audio":1}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data --media-io-kwargs '{"video":{"num_frames":16},"audio":{"audio_backend":"torchcodec"}}') ;;
  image) OMNI_FILE=mimo_v2_omni_mm.py; MM_ARGS=(--limit-mm-per-prompt '{"image":3,"video":0,"audio":0}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data) ;;
  0)     OMNI_FILE=mimo_v2_omni.py;    MM_ARGS=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --mm-processor-cache-gb 0)
         [ -n "${KV_BYTES_EXPLICIT:-}" ] || KV_BYTES=17179869184 ;;
  *) echo "MM must be 1, image or 0" >&2; exit 2 ;;
esac
# The ahead-of-time compiled language-model forward is specialised on the
# text-only dummy run; the multimodal profiling run (embeddings, no token
# ids) fails inside it, so AOT compile is off whenever multimodal is on.
if [ "$MM" != 0 ]; then AOT_COMPILE=0; else AOT_COMPILE="${AOT_COMPILE:-1}"; fi
# PAD_V=1 (pair default) zero-pads V from 128 to 192 so the target runs the
# symmetric TRITON_ATTN kernel; PAD_V=0 selects the DiffKV kernel path with
# the PR 839 split-KV dispatch. Target KV stays bf16 in both cases: the
# checkpoint carries no K/V scales and an fp8 target cache loops on long
# output. The draft's KV cache is fp8.
PAD_V="${PAD_V:-1}"
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
for f in config.json dflash/config.json dflash/dflash_draft_model.safetensors model.safetensors.index.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "missing $MODEL_DIR/$f" >&2; exit 1; }
done
for f in "$MIMO_V2_FILE" mimo_v2_mtp.py "$OMNI_FILE" qwen3_dflash.py; do
  [ -f "$OVERLAY/$f" ] || { echo "missing overlay file $OVERLAY/$f" >&2; exit 1; }
done
mkdir -p "$CACHE_DIR"/{triton,inductor,roce,cute,b12x}
# Checkpoint revisions before b2674c72 ship dflash/config.json with a trailing
# comma. When the file is not valid JSON, a corrected copy is mounted over it.
DFLASH_MOUNT=()
if ! python3 -c 'import json,sys; json.load(open(sys.argv[1]))' "$MODEL_DIR/dflash/config.json" 2>/dev/null; then
  python3 "$HERE/fix_dflash_config.py" "$MODEL_DIR/dflash/config.json" "$CACHE_DIR/dflash-config.json"
  DFLASH_MOUNT=(-v "$CACHE_DIR/dflash-config.json":/models/target/dflash/config.json:ro)
fi
if [ "$ACTION" = --check ]; then
  docker image inspect "$IMAGE" --format '{{.Id}}' >/dev/null || { echo "image $IMAGE is not present" >&2; exit 1; }
  echo "rank $RANK: inputs valid (MM=$MM PAD_V=$PAD_V KV_BYTES=$KV_BYTES SPEC_TOKENS=$SPEC_TOKENS CG_CAP=$CG_CAP ROCE_AR=$ROCE_AR)"
  exit 0
fi
P=/opt/venv/lib/python3.12/site-packages/vllm/model_executor/models
C=/cache/mimo26
docker rm -f "$NAME" >/dev/null 2>&1 || true
run_container() {
docker run -d --name "$NAME" \
  --gpus all --network host --ipc host --shm-size 64m --ulimit memlock=-1:-1 \
  --device /dev/infiniband:/dev/infiniband \
  --env VLLM_HOST_IP="$HOST_IP" \
  --env NCCL_SOCKET_IFNAME="$MGMT_IFNAME" --env GLOO_SOCKET_IFNAME="$MGMT_IFNAME" \
  --env B12X_ROCE_HCA="$ROCE_HCA_PAIR" \
  --env TRITON_CACHE_DIR=$C/triton --env TORCHINDUCTOR_CACHE_DIR=$C/inductor \
  --env B12X_ROCE_CACHE_DIR=$C/roce --env CUTE_DSL_CACHE_DIR=$C/cute \
  --env B12X_COMPILE_CACHE_DIR=$C/b12x --env B12X_CUTE_COMPILE_CACHE_DIR=$C/b12x \
  --env SERVED_MODEL_NAME="$SERVED_MODEL_NAME" \
  --env VLLM_KV_CACHE_LAYOUT=BLHNC --env VLLM_USE_AOT_COMPILE="$AOT_COMPILE" \
  --env VLLM_ENABLE_ROCE_ALLREDUCE="$ROCE_AR" \
  --env VLLM_MIMO_PAD_V="$PAD_V" --env VLLM_MIMO_PASS_CACHE_CONFIG=0 \
  "${EXTRA_ENV_ARGS[@]}" \
  -v "$MODEL_DIR":/models/target:ro \
  -v "$CACHE_DIR":$C \
  "${KERNEL_MOUNT[@]}" "${DFLASH_MOUNT[@]}" \
  -v "$OVERLAY/$MIMO_V2_FILE":$P/mimo_v2.py:ro \
  -v "$OVERLAY/mimo_v2_mtp.py":$P/mimo_v2_mtp.py:ro \
  -v "$OVERLAY/$OMNI_FILE":$P/mimo_v2_omni.py:ro \
  -v "$OVERLAY/qwen3_dflash.py":$P/qwen3_dflash.py:ro \
  --entrypoint /bin/sh "$IMAGE" -lc 'exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$@"' sh /models/target \
  --served-model-name "$SERVED_MODEL_NAME" --trust-remote-code \
  --node-rank "$RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --nnodes 2 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --dtype bfloat16 --kv-cache-dtype fp8 --block-size 64 \
  --gpu-memory-utilization "$GPU_MEM" --kv-cache-memory-bytes "$KV_BYTES" \
  --attention-backend "$ATTN" --linear-backend b12x --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching --enable-chunked-prefill \
  "${MM_ARGS[@]}" \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --generation-config vllm \
  --speculative-config "{\"model\":\"/models/target/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":${SPEC_TOKENS},\"attention_backend\":\"${ATTN}\"}" \
  --compilation-config "{\"cudagraph_mode\":\"FULL_AND_PIECEWISE\",\"max_cudagraph_capture_size\":${CG_CAP}}"
}
# The first GPU container start after removing the previous one can fail with
# "open /run/nvidia-persistenced/socket: no such file or directory" while the
# socket exists; a retry succeeds.
for attempt in 1 2 3; do
  run_container && exit 0
  echo "docker run failed (attempt $attempt); retrying in 5s" >&2
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  sleep 5
done
exit 1
