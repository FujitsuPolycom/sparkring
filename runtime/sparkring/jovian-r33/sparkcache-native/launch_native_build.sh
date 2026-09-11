#!/usr/bin/env bash
# Launch one bounded, network-isolated SparkCache CUDA build on an idle GB10 host.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
name=sparkring-r33-sparkcache-native-20260910
source_dir="$root/sources/sparkcache"
build_dir="$root/build/sparkcache-native"
out_dir="$root/artifacts/sparkcache-native"
script_dir="$root/scripts/sparkcache-native"
pytest_site_packages="$root/work/venv-payload-probe-image-context-002/lib/python3.12/site-packages"

test -d "$source_dir/.git"
test -x "$script_dir/build_native.sh"
test -f "$script_dir/make_receipt.py"
test -d "$pytest_site_packages/pytest"
test ! -e "$build_dir"
test ! -e "$out_dir"
mkdir -p "$build_dir" "$out_dir"

foundation_id=$(docker image inspect local/sparkring:r33-arm64-foundation --format '{{.Id}}')
docker run --detach --name "$name" --network none --gpus all \
  --cpus 4 --memory 12g --memory-swap 12g --shm-size 2g \
  --env CUDA_VISIBLE_DEVICES=0 \
  --env SPARKCACHE_FOUNDATION_IMAGE_ID="$foundation_id" \
  --mount type=bind,src="$source_dir",dst=/source,readonly \
  --mount type=bind,src="$build_dir",dst=/build \
  --mount type=bind,src="$out_dir",dst=/out \
  --mount type=bind,src="$script_dir",dst=/scripts,readonly \
  --mount type=bind,src="$pytest_site_packages",dst=/pytest-site-packages,readonly \
  --entrypoint bash local/sparkring:r33-arm64-foundation \
  /scripts/build_native.sh
