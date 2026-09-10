#!/usr/bin/env bash
# Launch exact XGrammar packaging without GPU access.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-xgrammar-build-20260910
test ! -e "$root/artifacts/xgrammar"
mkdir -p "$root/artifacts/xgrammar"
docker run --detach --name "$name" \
  --cpus 8 --memory 24g --memory-swap 24g \
  --mount type=bind,src="$root/sources/xgrammar",dst=/source,readonly \
  --mount type=bind,src="$root/artifacts/xgrammar",dst=/out \
  --mount type=bind,src="$root/scripts/build_xgrammar.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
