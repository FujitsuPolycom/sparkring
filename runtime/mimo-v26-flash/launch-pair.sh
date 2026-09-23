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
# Invoke the derived image's CLI through a login shell so /etc/shinit_v2
# activates its CUDA compatibility driver. The inherited base-image
# attestation does not describe the replaced Python packages.
# Settings and source pins are recorded in recipe.json and b12x-image.json.
set -euo pipefail
[ $# -eq 2 ] && { [ "$1" = --check ] || [ "$1" = --run ]; } || { echo "usage: bash launch-pair.sh --check|--run RANK_ENV_FILE" >&2; exit 2; }
ACTION="$1"; RANK_ENV_FILE="$2"
[ -f "$RANK_ENV_FILE" ] || { echo "missing $RANK_ENV_FILE" >&2; exit 2; }
if grep -v "^#" "$RANK_ENV_FILE" | grep -q "REPLACE_"; then echo "$RANK_ENV_FILE: replace every REPLACE_ value" >&2; exit 2; fi
# shellcheck disable=SC1090
. "$RANK_ENV_FILE"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
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
# Retained starting reservation; BF16 draft KV and 64-row graphs need
# TP2 memory qualification on this source composition.
KV_BYTES="${KV_BYTES:-12884901888}"
GPU_MEM="${GPU_MEM:-0.88}"
SPEC_TOKENS="${SPEC_TOKENS:-5}"
ATTN="${ATTN:-B12X}"
CG_CAP="${CG_CAP:-64}"
ROCE_AR="${ROCE_AR:-1}"
# MM selects image, video and audio request limits.
MM="${MM:-1}"
case "$MM" in
  1)     MM_ARGS=(--limit-mm-per-prompt '{"image":3,"video":1,"audio":1}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data --media-io-kwargs '{"video":{"num_frames":16},"audio":{"audio_backend":"torchcodec"}}') ;;
  image) MM_ARGS=(--limit-mm-per-prompt '{"image":3,"video":0,"audio":0}' --mm-processor-cache-gb 0 --mm-encoder-tp-mode data) ;;
  0)     MM_ARGS=(--limit-mm-per-prompt '{"image":0,"video":0,"audio":0}' --mm-processor-cache-gb 0)
         [ -n "${KV_BYTES_EXPLICIT:-}" ] || KV_BYTES=17179869184 ;;
  *) echo "MM must be 1, image or 0" >&2; exit 2 ;;
esac
# The ahead-of-time compiled language-model forward is specialised on the
# text-only dummy run; the multimodal profiling run (embeddings, no token
# ids) fails inside it, so AOT compile is off whenever multimodal is on.
if [ "$MM" != 0 ]; then AOT_COMPILE=0; else AOT_COMPILE="${AOT_COMPILE:-1}"; fi
# Native B12X accepts packed Q/K-192, V-128 pages without padding V.
# Target and draft KV remain BF16; FP8 B12X draft KV is not qualified here.
EXTRA_ENV_ARGS=()
IFS=',' read -r -a _extra_kv <<< "${EXTRA_ENV:-}"
for kv in "${_extra_kv[@]}"; do [ -n "$kv" ] && EXTRA_ENV_ARGS+=(--env "$kv"); done
for f in config.json dflash/config.json dflash/dflash_draft_model.safetensors model.safetensors.index.json; do
  [ -f "$MODEL_DIR/$f" ] || { echo "missing $MODEL_DIR/$f" >&2; exit 1; }
done
python3 "$HERE/check_image.py" "$IMAGE"
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
  echo "rank $RANK: inputs valid (MM=$MM ATTN=$ATTN KV_BYTES=$KV_BYTES SPEC_TOKENS=$SPEC_TOKENS CG_CAP=$CG_CAP ROCE_AR=$ROCE_AR)"
  exit 0
fi
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
  --env VLLM_USE_V2_MODEL_RUNNER=1 --env VLLM_PLUGINS=b12x_loader \
  --env CUTE_DSL_ARCH=sm_121a --env FLASHINFER_CUDA_ARCH_LIST=12.1f \
  --env XDG_CACHE_HOME=$C --env VLLM_CACHE_ROOT=$C/vllm \
  --env VLLM_KV_CACHE_LAYOUT=BLHNC --env VLLM_USE_AOT_COMPILE="$AOT_COMPILE" \
  --env VLLM_ENABLE_ROCE_ALLREDUCE="$ROCE_AR" \
  "${EXTRA_ENV_ARGS[@]}" \
  -v "$MODEL_DIR":/models/target:ro \
  -v "$CACHE_DIR":$C \
  "${DFLASH_MOUNT[@]}" \
  --entrypoint /bin/sh "$IMAGE" -lc 'exec /opt/venv/bin/python -m vllm.entrypoints.cli.main serve "$@"' sh /models/target \
  --served-model-name "$SERVED_MODEL_NAME" --trust-remote-code \
  --node-rank "$RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT" --nnodes 2 \
  --tensor-parallel-size 2 --pipeline-parallel-size 1 --distributed-executor-backend mp \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MAX_BATCHED" \
  --dtype bfloat16 --load-format safetensors --kv-cache-dtype bfloat16 --block-size 64 \
  --gpu-memory-utilization "$GPU_MEM" --kv-cache-memory-bytes "$KV_BYTES" \
  --attention-backend "$ATTN" --linear-backend b12x --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --enable-prefix-caching --enable-chunked-prefill \
  "${MM_ARGS[@]}" \
  --reasoning-parser mimo --tool-call-parser mimo --enable-auto-tool-choice \
  --generation-config vllm \
  --speculative-config "{\"model\":\"/models/target/dflash\",\"method\":\"dflash\",\"num_speculative_tokens\":${SPEC_TOKENS},\"kv_cache_dtype\":\"auto\",\"attention_backend\":\"B12X\"}" \
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
