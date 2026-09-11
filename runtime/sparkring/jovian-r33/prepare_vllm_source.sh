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
patch_sha=26c21814b7b3239ce8085cfee667ca60710e434586623c8f6d9db8aed987b97e
manifest_sha=49d1be5949f7327e62f617ad4961a762ff30173dfbbb181d9d0adb3fff56c7b8
expected_tree=4f1813fcd2fa1cfc94fdc69a256f2266e394ff90

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
  printf 'result.tree=%s\n' "$actual_tree"
} > "$receipt"
git -C "$target" diff --cached --check
printf '%s\n' "$actual_tree"
