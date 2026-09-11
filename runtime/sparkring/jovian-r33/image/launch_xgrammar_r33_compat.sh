#!/usr/bin/env bash
set -euo pipefail
root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-xgrammar-compat-build-20260910
test ! -e "$root/artifacts/xgrammar-r33"
test ! -e "$root/build/xgrammar-r33"
mkdir -p "$root/artifacts/xgrammar-r33" "$root/build/xgrammar-r33"
docker run --detach --name "$name" --cpus 8 --memory 32g --memory-swap 32g -e MAX_JOBS=8 \
  --mount type=bind,src="$root/sources/xgrammar",dst=/source,readonly \
  --mount type=bind,src="$root/build/xgrammar-r33",dst=/work \
  --mount type=bind,src="$root/artifacts/xgrammar-r33",dst=/out \
  --mount type=bind,src="$root/scripts/build_xgrammar_r33_compat.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
