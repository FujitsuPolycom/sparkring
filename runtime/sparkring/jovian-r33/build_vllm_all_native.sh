#!/usr/bin/env bash
# Finish every configured R33 vLLM native target using the verified build tree.
set -euo pipefail

source_dir=/source
cutlass_dir=/cutlass
build_dir=/build
out=/out
expected_tree=386191c06df9c4232cb2f48012968f48cfdc6eee
git config --global --add safe.directory "$source_dir"
git config --global --add safe.directory "$cutlass_dir"
test "$(git -C "$source_dir" write-tree)" = "$expected_tree"
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = e6233cbac5d7c7a865c19c91cd684ceece19513c
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification-all.json"
export VLLM_CUTLASS_SRC_DIR="$cutlass_dir"
export TORCH_CUDA_ARCH_LIST=12.1a CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
cmake --build "$build_dir" --parallel "${MAX_JOBS:-14}"
rm -rf "$out/modules"
mkdir -p "$out/modules"
cd "$build_dir"
while IFS= read -r -d '' module; do
  destination="$out/modules/$module"
  mkdir -p "$(dirname "$destination")"
  install -m755 "$module" "$destination"
done < <(find . -type f -name '*.so' -print0)
find "$out/modules" -type f -name '*.so' -print0 | sort -z | xargs -0 sha256sum > "$out/ALL-SHA256SUMS"
find "$out/modules" -type f -name '*.so' -print0 | while IFS= read -r -d '' module; do
  readelf -h "$module" | grep -F 'AArch64' >/dev/null
done
printf 'status=compiled-all-cmake-targets-cpu-import-and-gpu-qualification-pending\nresult.tree=%s\n' \
  "$expected_tree" > "$out/all-source-receipt.txt"
