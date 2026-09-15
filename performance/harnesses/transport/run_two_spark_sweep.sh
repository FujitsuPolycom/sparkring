#!/usr/bin/env bash
set -euo pipefail

SERVER_SSH="${SERVER_SSH:-${SPARKRING_TARGETS:?set SPARKRING_TARGETS to a comma-separated user@host list (the first entry is used as the remote server) or set SERVER_SSH=user@host}}"
SERVER_SSH="${SERVER_SSH%%,*}"
PEER_FABRIC_IP="${PEER_FABRIC_IP:?set PEER_FABRIC_IP to the fabric IP of the server node}"
DEVICE="${DEVICE:-rocep1s0f0}"
GID_INDEX="${GID_INDEX:-3}"
IMAGE="${IMAGE:?set IMAGE to your vLLM container image tag}"
BINARY="${BINARY:-/tmp/spark_transport_probe}"
MEMORY="${MEMORY:-cuda-mapped}"
WARMUP="${WARMUP:-1000}"
ITERATIONS="${ITERATIONS:-10000}"
BASE_PORT="${BASE_PORT:-9420}"
SIZES="${SIZES:-4096 8192 12288 16384 24576 32768 65536}"

# Each invocation owns only containers carrying its nonce. Cleanup uses immutable
# container IDs so a reused name cannot redirect removal to another workload.
run_id="$$-${RANDOM}-${RANDOM}"
client_name=""
server_name=""
remote_docker() {
  local argument quoted command="docker"
  for argument in "$@"; do
    printf -v quoted '%q' "$argument"
    command+=" ${quoted}"
  done
  ssh "${SERVER_SSH}" "$command"
}
remove_owned() {
  local location="$1" name="$2" record container_id owner
  [[ -n "$name" ]] || return 0
  if [[ "$location" == remote ]]; then
    record=$(remote_docker inspect --format '{{.Id}} {{index .Config.Labels "org.sparkring.sweep-run"}}' "$name" 2>/dev/null) || return 0
  else
    record=$(docker inspect --format '{{.Id}} {{index .Config.Labels "org.sparkring.sweep-run"}}' "$name" 2>/dev/null) || return 0
  fi
  read -r container_id owner <<< "$record"
  [[ "$container_id" =~ ^[0-9a-f]{64}$ && "$owner" == "$run_id" ]] || return 0
  if [[ "$location" == remote ]]; then
    remote_docker rm -f "$container_id" >/dev/null
  else
    docker rm -f "$container_id" >/dev/null
  fi
}
cleanup() {
  local status=$?
  trap - EXIT
  set +e
  remove_owned local "$client_name"
  remove_owned remote "$server_name"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

index=0
for bytes in ${SIZES}; do
  port=$((BASE_PORT + index))
  server_name="spark-sweep-server-${run_id}-${bytes}"
  client_name="spark-sweep-client-${run_id}-${bytes}"

  remote_docker run -d \
    --name "${server_name}" \
    --label "org.sparkring.sweep-run=${run_id}" \
    --privileged --gpus all --network host --ipc host \
    --ulimit memlock=-1 \
    -v "${BINARY}:/probe:ro" \
    "${IMAGE}" /probe \
    --server \
    --device "${DEVICE}" \
    --gid "${GID_INDEX}" \
    --control-port "${port}" \
    --bytes "${bytes}" \
    --memory "${MEMORY}" \
    --gpu-roundtrip \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" >/dev/null

  docker run \
    --name "${client_name}" \
    --label "org.sparkring.sweep-run=${run_id}" \
    --privileged --gpus all --network host --ipc host \
    --ulimit memlock=-1 \
    -v "${BINARY}:/probe:ro" \
    "${IMAGE}" /probe \
    --client "${PEER_FABRIC_IP}" \
    --device "${DEVICE}" \
    --gid "${GID_INDEX}" \
    --control-port "${port}" \
    --bytes "${bytes}" \
    --memory "${MEMORY}" \
    --gpu-roundtrip \
    --warmup "${WARMUP}" \
    --iterations "${ITERATIONS}" |
    grep -E '^(GPU_ROUNDTRIP|PHASE)'

  remote_docker logs "${server_name}" 2>&1 |
    grep '^VERIFY'
  remove_owned local "$client_name"
  remove_owned remote "$server_name"
  client_name=""
  server_name=""
  index=$((index + 1))
done
