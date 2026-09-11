#!/usr/bin/env bash
# Launch deterministic SparkCache packaging without GPU or network access.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-sparkcache-build-20260910
test ! -e "$root/artifacts/sparkcache"
mkdir -p "$root/artifacts/sparkcache"
docker run --detach --name "$name" --network none \
  --cpus 2 --memory 4g --memory-swap 4g \
  --mount type=bind,src="$root/sources/sparkcache",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/sparkcache",dst=/out \
  --mount type=bind,src="$root/scripts/build_sparkcache.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
