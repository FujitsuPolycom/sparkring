#!/usr/bin/env bash
# Build R33 native vLLM operators against the verified ARM64 foundation.
set -euo pipefail

source_dir=/source
cutlass_dir=/cutlass
build_dir=/build
out=/out
expected_tree=386191c06df9c4232cb2f48012968f48cfdc6eee
expected_cutlass=e6233cbac5d7c7a865c19c91cd684ceece19513c

git config --global --add safe.directory "$source_dir"
git config --global --add safe.directory "$cutlass_dir"
test "$(git -C "$source_dir" write-tree)" = "$expected_tree"
test "$(git -C "$cutlass_dir" rev-parse HEAD)" = "$expected_cutlass"
test -z "$(git -C "$cutlass_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
mkdir -p "$build_dir" "$out"

export VLLM_CUTLASS_SRC_DIR="$cutlass_dir"
export TORCH_CUDA_ARCH_LIST=12.1a
export CUDA_VISIBLE_DEVICES=
export NVIDIA_VISIBLE_DEVICES=void
export MAX_JOBS=${MAX_JOBS:-14}
export NVCC_THREADS=1
cmake -S "$source_dir" -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DVLLM_TARGET_DEVICE=cuda \
  -DVLLM_PYTHON_EXECUTABLE=/usr/bin/python3 \
  -DNVCC_THREADS=1
cmake --build "$build_dir" \
  --target _C_stable_libtorch _flashkda_C \
  --parallel "$MAX_JOBS"

install -m755 "$build_dir/_C_stable_libtorch.abi3.so" "$out/"
install -m755 "$build_dir/_flashkda_C.abi3.so" "$out/"
flashkda_source="$build_dir/_deps/flashkda-src"
test "$(git -C "$flashkda_source" rev-parse HEAD)" = 3b225bf26bb8e218928a1fe14751cb48cf31d11b
test -n "$(grep -F checkpoint_indptr "$flashkda_source/csrc/flash_kda.h")"
readelf -h "$out/_C_stable_libtorch.abi3.so" > "$out/stable-elf.txt"
readelf -h "$out/_flashkda_C.abi3.so" > "$out/flashkda-elf.txt"
grep -F 'AArch64' "$out/stable-elf.txt"
grep -F 'AArch64' "$out/flashkda-elf.txt"
sha256sum "$out"/*.so > "$out/SHA256SUMS"
{
  printf 'status=compiled-cpu-import-and-gpu-qualification-pending\n'
  printf 'vllm.result.tree=%s\n' "$expected_tree"
  printf 'cutlass.commit=%s\n' "$expected_cutlass"
  printf 'flashkda.base.commit=%s\n' "$(git -C "$flashkda_source" rev-parse HEAD)"
  printf 'flashkda.patch.sha256=%s\n' "$(sha256sum "$source_dir/cmake/external_projects/patches/flashkda-packed-checkpoints.patch" | cut -d' ' -f1)"
} > "$out/source-receipt.txt"
