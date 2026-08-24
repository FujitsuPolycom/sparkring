#!/usr/bin/env bash
set -euo pipefail

here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "${here}/../.." && pwd)
engine=${SPARKRING_CONTAINER_ENGINE:-docker}
target=${SPARKRING_DEEPSEEK_GB10_TARGET:-native}
image_tag=${1:-sparkring/deepseek-v4-flash-0731-gb10-hardened:local}
max_jobs=${MAX_JOBS:-10}
source_revision=$(git -C "${repo_root}" rev-parse HEAD)

case "${target}" in
  native|thin) ;;
  *)
    printf 'SPARKRING_DEEPSEEK_GB10_TARGET must be native or thin, got %s\n' "${target}" >&2
    exit 2
    ;;
esac

"${engine}" build \
  --pull \
  --file "${here}/Containerfile" \
  --target "${target}" \
  --build-arg "MAX_JOBS=${max_jobs}" \
  --build-arg "SPARKRING_SOURCE_REVISION=${source_revision}" \
  --tag "${image_tag}" \
  "${repo_root}"

verify_args=(
  /opt/sparkring-deepseek-gb10/verify_image.py
  --require-launch-env
  --contract /opt/sparkring-deepseek-gb10/runtime-contract.json
)
if [[ "${target}" == native ]]; then
  verify_args+=(--expect-native)
fi

"${engine}" run --rm \
  --entrypoint python3 \
  --env 'LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1:/opt/sparkring/nccl/libnccl.so.2' \
  "${image_tag}" \
  "${verify_args[@]}"

"${engine}" image inspect --format '{{.Id}} {{.Size}}' "${image_tag}"
