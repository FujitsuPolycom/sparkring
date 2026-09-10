#!/usr/bin/env bash
# Build the R33 Torchvision source against the verified ARM64 Torch foundation.
set -euo pipefail

source_input=/source
source_dir=/tmp/vision-source
out=/out
expected_commit=8fb87713a24951e639c494b0f2a8a81b5f8e33a6
git config --global --add safe.directory "$source_input"
test "$(git -C "$source_input" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_input" status --porcelain)"
source_tree=$(git -C "$source_input" rev-parse HEAD^{tree})
submodules=$(git -C "$source_input" submodule status --recursive)
cp -a "$source_input" "$source_dir"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
export FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS:-12}
export BUILD_VERSION=0.28.0 PYTORCH_VERSION=2.13.0 CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
cd "$source_dir"
python3 setup.py bdist_wheel -d "$out"
test "$(git -C "$source_input" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_input" rev-parse HEAD^{tree})" = "$source_tree"
test "$(git -C "$source_input" submodule status --recursive)" = "$submodules"
test -z "$(git -C "$source_input" status --porcelain)"
sha256sum "$out"/torchvision-0.28.0-*.whl > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-gpu-qualification-pending\nsource.commit=%s\nsource.tree=%s\nsource.submodules.sha256=%s\nsource.post-build-identical=true\n' \
  "$expected_commit" "$source_tree" "$(printf '%s' "$submodules" | sha256sum | cut -d' ' -f1)" > "$out/source-receipt.txt"
