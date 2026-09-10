#!/usr/bin/env bash
# Launch the bounded ARM64 R33 native build without GPU access.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-vllm-native-build-20260910
test ! -e "$root/build/vllm-native"
test ! -e "$root/artifacts/vllm-native"
mkdir -p "$root/build/vllm-native" "$root/artifacts/vllm-native"
docker run --detach --name "$name" \
  --cpus 14 --memory 80g --memory-swap 80g \
  --mount type=bind,src="$root/sources/vllm-sparkring",dst=/source,readonly \
  --mount type=bind,src="$root/sources/cutlass",dst=/cutlass,readonly \
  --mount type=bind,src="$root/build/vllm-native",dst=/build \
  --mount type=bind,src="$root/artifacts/vllm-native",dst=/out \
  --mount type=bind,src="$root/scripts/build_vllm_native.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
