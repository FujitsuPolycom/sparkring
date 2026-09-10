#!/usr/bin/env bash
# Continue the configured vLLM build after focused native targets succeed.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-vllm-all-native-build-20260910
test -f "$root/build/vllm-native/build.ninja"
test -f "$root/artifacts/vllm-native/source-receipt.txt"
docker run --detach --name "$name" \
  --cpus 14 --memory 80g --memory-swap 80g \
  --mount type=bind,src="$root/sources/vllm-sparkring",dst=/source,readonly \
  --mount type=bind,src="$root/sources/cutlass",dst=/cutlass,readonly \
  --mount type=bind,src="$root/build/vllm-native",dst=/build \
  --mount type=bind,src="$root/artifacts/vllm-native",dst=/out \
  --mount type=bind,src="$root/scripts/build_vllm_all_native.sh",dst=/scripts/build.sh,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation /scripts/build.sh
