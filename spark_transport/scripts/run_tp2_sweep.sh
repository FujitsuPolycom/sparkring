#!/usr/bin/env bash
set -euo pipefail

SERVER_SSH="${SERVER_SSH:-${SPARKRING_TARGETS:?set SPARKRING_TARGETS to a comma-separated user@host list (the first entry is used as the remote server) or set SERVER_SSH=user@host}}"
SERVER_SSH="${SERVER_SSH%%,*}"
PEER_FABRIC_IP="${PEER_FABRIC_IP:?set PEER_FABRIC_IP to the fabric IP of the server node}"
DEVICE="${DEVICE:-rocep1s0f0}"
GID_INDEX="${GID_INDEX:-3}"
IMAGE="${IMAGE:?set IMAGE to your vLLM container image tag}"
BINARY="${BINARY:-/tmp/spark_tp2_probe}"
WARMUP="${WARMUP:-1000}"
ITERATIONS="${ITERATIONS:-10000}"
BASE_PORT="${BASE_PORT:-9450}"
SIZES="${SIZES:-4096 8192 12288 16384 24576 32768 65536}"

index=0
for bytes in ${SIZES}; do
  port=$((BASE_PORT + index))
  server_name="spark-tp2-sweep-server-${bytes}"
  client_name="spark-tp2-sweep-client-${bytes}"

  ssh "${SERVER_SSH}" docker run -d \
    --name "${server_name}" \
    --privileged --gpus all --network host --ipc host \
    --ulimit memlock=-1 \
    -v "${BINARY}:/probe:ro" \
    "${IMAGE}" /probe \
    --server \
    --device "${DEVICE}" \
    --gid "${GID_INDEX}" \
    --control-port "${port}" \
    --bytes "${bytes}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" >/dev/null

  docker run \
    --name "${client_name}" \
    --privileged --gpus all --network host --ipc host \
    --ulimit memlock=-1 \
    -v "${BINARY}:/probe:ro" \
    "${IMAGE}" /probe \
    --client "${PEER_FABRIC_IP}" \
    --device "${DEVICE}" \
    --gid "${GID_INDEX}" \
    --control-port "${port}" \
    --bytes "${bytes}" \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" |
    grep -E '^(TP2_BF16|PHASE)'

  ssh "${SERVER_SSH}" docker logs "${server_name}" 2>&1 |
    grep '^TP2_BF16'
  docker rm "${client_name}" >/dev/null
  ssh "${SERVER_SSH}" docker rm "${server_name}" >/dev/null
  index=$((index + 1))
done
