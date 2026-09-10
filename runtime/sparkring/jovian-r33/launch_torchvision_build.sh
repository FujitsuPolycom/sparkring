#!/usr/bin/env bash
# Launch the bounded R33 Torchvision build after the vLLM native build releases CPUs.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-torchvision-build-20260910
test ! -e "$root/artifacts/torchvision"
mkdir -p "$root/artifacts/torchvision"
docker run --detach --name "$name" \
  --cpus 6 --memory 48g --memory-swap 48g -e MAX_JOBS=6 \
  --mount type=bind,src="$root/sources/vision",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/torchvision",dst=/out \
  --mount type=bind,src="$root/scripts/build_torchvision.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
