#!/usr/bin/env bash
# Launch the bounded R33 Rust build on an ARM64 Spark rank.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
image=${SPARKRING_R33_FOUNDATION_IMAGE:-local/sparkring:r33-arm64-foundation}
name=${SPARKRING_R33_RUST_CONTAINER:-sparkring-r33-rust-build-20260910}
base="$root/sources/vllm"
source_dir="$root/sources/vllm-rust"
patch="$root/inputs/r33-prefill-sparkcache.patch"
out="$root/artifacts/rust"
build="$root/build/rust"
script="$root/scripts/build_rust_frontend.sh"
expected_commit=ae89131442359dc332d9c46009be3c1f8cdee0b4
expected_patch=eb61c41be57bd52aff57a6ad65e0e01ec12458c77b98d3b529e12688c0e1fa05
expected_tree=386191c06df9c4232cb2f48012968f48cfdc6eee

test "$(uname -m)" = aarch64
test "$(git -C "$base" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$base" status --porcelain)"
printf '%s  %s\n' "$expected_patch" "$patch" | sha256sum -c -
test -f "$script"

if [[ ! -d "$source_dir/.git" ]]; then
  test ! -e "$source_dir"
  git clone --no-hardlinks "$base" "$source_dir"
  git -C "$source_dir" checkout --detach "$expected_commit"
  git -C "$source_dir" apply --index "$patch"
fi
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir" write-tree)" = "$expected_tree"
git -C "$source_dir" diff --cached --check

mkdir -p "$out" "$build"
test -z "$(docker ps -aq --filter name=^/${name}$)"
docker run -d \
  --name "$name" \
  --cpus 4 \
  --memory 24g \
  --pids-limit 2048 \
  --shm-size 2g \
  --mount type=bind,src="$source_dir",dst=/source \
  --mount type=bind,src="$build",dst=/build \
  --mount type=bind,src="$out",dst=/out \
  --mount type=bind,src="$script",dst=/scripts/build.sh,readonly \
  "$image" /bin/bash /scripts/build.sh
