#!/usr/bin/env bash
# Build the R33 Torchvision source against the verified ARM64 Torch foundation.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=8fb87713a24951e639c494b0f2a8a81b5f8e33a6
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS:-12}
export BUILD_VERSION=0.28.0 CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
cd "$source_dir"
python3 setup.py bdist_wheel -d "$out"
sha256sum "$out"/torchvision-0.28.0-*.whl > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-gpu-qualification-pending\nsource.commit=%s\n' \
  "$expected_commit" > "$out/source-receipt.txt"
