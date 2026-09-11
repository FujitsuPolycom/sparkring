#!/usr/bin/env bash
# Materialize the reviewed R33 vLLM source composition in an isolated checkout.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
script_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
base="$root/sources/vllm"
target="$root/sources/vllm-sparkring"
patch=${SPARKRING_R33_VLLM_PATCH:-$script_root/patches/vllm-r33-sparkring.patch}
manifest=${SPARKRING_R33_VLLM_MANIFEST:-$script_root/patches/vllm-r33-sparkring.manifest.json}
receipt="$root/artifacts/vllm-sparkring-source.txt"
base_commit=ae89131442359dc332d9c46009be3c1f8cdee0b4
patch_sha=73df604d183a3309109ac3e97e1a952e99d111b5761ef2e67fbf41a7957f4ae2
manifest_sha=dfd47ca3883ece53feda318c3ce8d1ea0ac1bce9a2145bb36c77e9a7c0531dab
expected_tree=0511a78617bb755ea2901ef3c5db7547bc1e148d

test "$(git -C "$base" rev-parse HEAD)" = "$base_commit"
test -z "$(git -C "$base" status --porcelain)"
echo "$patch_sha  $patch" | sha256sum -c -
echo "$manifest_sha  $manifest" | sha256sum -c -
test "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["result"]["tree"])' "$manifest")" = "$expected_tree"
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
  printf 'patch.manifest.sha256=%s\n' "$(sha256sum "$manifest" | cut -d' ' -f1)"
  printf 'continuation.port.commit=%s\n' b611611a643502542c2d900057eb47e407b8379e
  printf 'scheduler.config.compatibility.commit=%s\n' 3049b639bdbc513319f7bae896c4e239992bc7bb
  printf 'prefix.hit.metadata.compatibility.commit=%s\n' 4405a965e54f14df00d28e5e23f9793f866aae39
  printf 'result.tree=%s\n' "$actual_tree"
} > "$receipt"
git -C "$target" diff --cached --check
printf '%s\n' "$actual_tree"
