#!/usr/bin/env bash
# Build the R33 FlashInfer Python and JIT-cache wheels for ARM64 SM121.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=803c4664f4771ddc418f20a57f752469a237a825
git config --global --add safe.directory "$source_dir"
git config --global --add safe.directory "$source_dir/3rdparty/cccl"
git config --global --add safe.directory "$source_dir/3rdparty/cutlass"
git config --global --add safe.directory "$source_dir/3rdparty/nixl"
git config --global --add safe.directory "$source_dir/3rdparty/spdlog"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
python3 -m pip install --upgrade \
  'setuptools>=77,<81' 'packaging>=24' wheel tqdm ninja requests numpy \
  nvidia-ml-py 'apache-tvm-ffi==0.1.11' \
  'nvidia-cutlass-dsl[cu13]==4.6.2'
python3 -m pip install --force-reinstall --no-deps \
  'nvidia-cutlass-dsl-libs-cu13==4.6.2'
python3 -m pip freeze > "$out/build-dependencies.txt"
export TORCH_CUDA_ARCH_LIST=12.1a FLASHINFER_CUDA_ARCH_LIST=12.1f
export FLASHINFER_LOCAL_VERSION=cu133 FLASHINFER_DISABLE_VERSION_CHECK=1
export FLASHINFER_ENABLE_SM90=0 BUILD_NVEP=0 BUILD_NCCL_EP=0 BUILD_NIXL_EP=0
export NVCC_THREADS=1 MAX_JOBS=${MAX_JOBS:-14}
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
cd "$source_dir"
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" .
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" ./flashinfer-jit-cache
test "$(find "$out" -maxdepth 1 -name 'flashinfer_python-*.whl' | wc -l)" -eq 1
test "$(find "$out" -maxdepth 1 -name 'flashinfer_jit_cache-*.whl' | wc -l)" -eq 1
sha256sum "$out"/*.whl > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-gpu-qualification-pending\nsource.commit=%s\n' \
  "$expected_commit" > "$out/source-receipt.txt"
