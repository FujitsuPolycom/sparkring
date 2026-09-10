#!/usr/bin/env bash
# Materialize the reviewed R33 vLLM source composition in an isolated checkout.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
base="$root/sources/vllm"
target="$root/sources/vllm-sparkring"
patch="$root/inputs/r33-prefill-sparkcache.patch"
receipt="$root/artifacts/vllm-sparkring-source.txt"
base_commit=ae89131442359dc332d9c46009be3c1f8cdee0b4
patch_sha=eb61c41be57bd52aff57a6ad65e0e01ec12458c77b98d3b529e12688c0e1fa05
expected_tree=386191c06df9c4232cb2f48012968f48cfdc6eee

test "$(git -C "$base" rev-parse HEAD)" = "$base_commit"
test -z "$(git -C "$base" status --porcelain)"
echo "$patch_sha  $patch" | sha256sum -c -
test ! -e "$target"
git clone --no-hardlinks "$base" "$target"
git -C "$target" checkout --detach "$base_commit"
git -C "$target" apply --index "$patch"
actual_tree=$(git -C "$target" write-tree)
test "$actual_tree" = "$expected_tree"
{
  printf 'status=implemented-source-only-gpu-qualification-pending\n'
  printf 'base.commit=%s\n' "$base_commit"
  printf 'patch.sha256=%s\n' "$patch_sha"
  printf 'result.tree=%s\n' "$actual_tree"
} > "$receipt"
git -C "$target" diff --cached --check
printf '%s\n' "$actual_tree"
