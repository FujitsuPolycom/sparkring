#!/usr/bin/env bash
# Launch the bounded R33 FlashInfer ARM64 build after the active native build.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-flashinfer-build-20260910
if [[ ${SPARKRING_FLASHINFER_RESUME:-0} == 1 ]]; then
  test -d "$root/artifacts/flashinfer"
  test -f "$root/build/flashinfer-work/build/aot/cached_ops/.ninja_log"
else
  test ! -e "$root/artifacts/flashinfer"
  test ! -e "$root/build/flashinfer-work"
  mkdir -p "$root/artifacts/flashinfer" "$root/build/flashinfer-work"
fi
docker run --detach --name "$name" \
  --cpus 16 --memory 80g --memory-swap 80g -e MAX_JOBS=16 \
  -e SPARKRING_FLASHINFER_RESUME="${SPARKRING_FLASHINFER_RESUME:-0}" \
  --mount type=bind,src="$root/sources/flashinfer",dst=/source,readonly \
  --mount type=bind,src="$root/build/flashinfer-work",dst=/work \
  --mount type=bind,src="$root/artifacts/flashinfer",dst=/out \
  --mount type=bind,src="$root/scripts/build_flashinfer.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
