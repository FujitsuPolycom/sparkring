#!/usr/bin/env bash
# Launch exact InstantTensor packaging without GPU access.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-instanttensor-build-20260910
test ! -e "$root/artifacts/instanttensor"
mkdir -p "$root/artifacts/instanttensor"
docker run --detach --name "$name" --network none \
  --cpus 4 --memory 12g --memory-swap 12g \
  --mount type=bind,src="$root/sources/instanttensor",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/instanttensor",dst=/out \
  --mount type=bind,src="$root/scripts/build_instanttensor.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
