#!/usr/bin/env bash
set -euo pipefail

source_dir=${TORCH_SOURCE_DIR:-/work/sources/pytorch}
prefix=${NCCL_INSTALL_PREFIX:-/opt/local-inference/nccl}
out=${TORCH_WHEEL_OUTPUT:-/work/artifacts/torch}
jobs=${MAX_JOBS:-16}
test "$(uname -m)" = aarch64
git config --global --add safe.directory "$source_dir"
git config --global --add safe.directory "$source_dir/*"
test "$(git -C "$source_dir" rev-parse HEAD)" = cf30153c4c131c8164ee7798e5022d810682e2cb
test -z "$(git -C "$source_dir" status --porcelain)"
git -C "$source_dir" submodule status --recursive > /work/torch-submodules.txt
if grep -Eq '^[-+U]' /work/torch-submodules.txt; then
  echo 'Source submodules are incomplete or differ from their pinned commits.' >&2
  exit 1
fi
nvcc --version | grep -F 'release 13.3'
# NVCC's listing omits architecture-specific suffixes that it accepts.
printf '__global__ void probe() {}\n' > /tmp/sparkring-torch-arch.cu
nvcc -arch=sm_121a -cubin /tmp/sparkring-torch-arch.cu -o /tmp/sparkring-torch-arch.cubin
test -f "$prefix/build-receipt.json"
mkdir -p "$out" /work/ccache/torch
if ! command -v ccache >/dev/null; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ccache
fi
python3 -m pip install -r "$source_dir/requirements.txt" 'cmake>=3.26,<4' ninja packaging wheel
python3 -m pip freeze > "$out/build-dependencies.txt"
export CCACHE_DIR=/work/ccache/torch
ccache -M 10G
export NCCL_ROOT="$prefix" NCCL_INCLUDE_DIR="$prefix/include" NCCL_LIB_DIR="$prefix/lib"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export USE_CUDA=1 USE_CUDNN=1 USE_CUSPARSELT=1 USE_NCCL=1 USE_SYSTEM_NCCL=1
export USE_DISTRIBUTED=1 USE_KINETO=1 BUILD_TEST=0 BUILD_CAFFE2=0
export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="$jobs"
export PYTORCH_BUILD_VERSION=2.13.0 PYTORCH_BUILD_NUMBER=1
export CMAKE_C_COMPILER_LAUNCHER=ccache CMAKE_CXX_COMPILER_LAUNCHER=ccache CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export TORCH_NVCC_FLAGS='-Xfatbin -compress-all'
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
cd "$source_dir"
python3 setup.py bdist_wheel -d "$out"
sha256sum "$out"/*.whl > "$out/SHA256SUMS"
printf '%s\n' 'Wheel built. Runtime source, CUDA, ABI, library ownership and GPU qualification remain required.'
