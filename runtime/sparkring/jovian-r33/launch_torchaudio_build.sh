#!/usr/bin/env bash
# Launch the source build required by the CUDA 13.3 media compatibility gate.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-torchaudio-build-20260910
test ! -e "$root/artifacts/torchaudio"
mkdir -p "$root/artifacts/torchaudio"
docker run --detach --name "$name" \
  --cpus 8 --memory 32g --memory-swap 32g -e MAX_JOBS=8 \
  --mount type=bind,src="$root/sources/audio",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/torchaudio",dst=/out \
  --mount type=bind,src="$root/scripts/build_torchaudio.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
