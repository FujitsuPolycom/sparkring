#!/usr/bin/env bash
# Launch the complete R33 LMCache ARM64 build after the active native build.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-lmcache-build-20260910
test ! -e "$root/artifacts/lmcache"
mkdir -p "$root/artifacts/lmcache"
docker run --detach --name "$name" \
  --cpus 6 --memory 64g --memory-swap 64g -e MAX_JOBS=6 \
  --mount type=bind,src="$root/sources/lmcache",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/lmcache",dst=/out \
  --mount type=bind,src="$root/scripts/build_lmcache.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
