#!/usr/bin/env bash
# Launch the exact R33 B12X packaging job without GPU access.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-b12x-build-20260910
test ! -e "$root/artifacts/b12x"
mkdir -p "$root/artifacts/b12x"
docker run --detach --name "$name" --network none \
  --cpus 2 --memory 4g --memory-swap 4g \
  --mount type=bind,src="$root/sources/b12x",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/b12x",dst=/out \
  --mount type=bind,src="$root/scripts/build_b12x.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
