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
patch_sha=ca66931518af4392d3cad42faf9fe8e4136cc8221edb7ac72491fde2000a6040
manifest_sha=8e17816bc60f8a14b6bfe9ead7af9436903b6ce4482c4f99afff6b3748710ce4
expected_tree=667ee2f6652efa065c57a7adc0193991f6cde6ac

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
  printf 'prefix.hit.metadata.compatibility.commit=%s\n' 58c087102cd3039245240e34c750c7f77ce07ed7
  printf 'result.tree=%s\n' "$actual_tree"
} > "$receipt"
git -C "$target" diff --cached --check
printf '%s\n' "$actual_tree"
