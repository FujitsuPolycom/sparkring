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
patch_sha=a0840292894c227036b8a2f12fe9705450c456061cee12e08474e85f34287ab9
manifest_sha=4e996963617c3c41e2aa32b7d492d3bcda6a0578af0aa23bd21f3fb2c4ace207
expected_tree=6c54193a3e9b842fa381095efa25dbb7f741402d

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
  printf 'result.tree=%s\n' "$actual_tree"
} > "$receipt"
git -C "$target" diff --cached --check
printf '%s\n' "$actual_tree"
