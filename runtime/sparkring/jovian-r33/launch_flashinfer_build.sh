#!/usr/bin/env bash
# Launch the bounded R33 FlashInfer ARM64 build after the active native build.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-flashinfer-build-20260910
test ! -e "$root/artifacts/flashinfer"
mkdir -p "$root/artifacts/flashinfer"
docker run --detach --name "$name" \
  --cpus 6 --memory 80g --memory-swap 80g -e MAX_JOBS=6 \
  --mount type=bind,src="$root/sources/flashinfer",dst=/source \
  --mount type=bind,src="$root/artifacts/flashinfer",dst=/out \
  --mount type=bind,src="$root/scripts/build_flashinfer.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
