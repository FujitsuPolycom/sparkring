#!/usr/bin/env bash
# Build and test the locked SIRCL source on an idle GB10 without starting a model.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-sircl-build-20260910
test ! -e "$root/build/sircl-cu133-sm121"
docker run --detach --name "$name" --network none --gpus all --ipc host \
  --cpus 6 --memory 32g --memory-swap 32g \
  --mount type=bind,src="$root/sources/sparkring",dst=/source,readonly \
  --mount type=bind,src="$root/build",dst=/work \
  --mount type=bind,src="$root/scripts/build-sircl-cu133-sm121.sh",dst=/scripts/build.sh,readonly \
  -e CUDA_HOME=/usr/local/cuda-13.3 -e SPARKRING_SIRCL_BUILD_JOBS=6 \
  --entrypoint bash local/sparkring:r33-arm64-foundation \
  /scripts/build.sh /source /work/sircl-cu133-sm121
