#!/usr/bin/env bash
# Build TorchAudio 2.11.0 against the exact CUDA 13.3 R33 Torch foundation.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=34c52a67e8941bbd8e6adaca0eb0b9eabec11d78
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
cp -a "$source_dir" /tmp/audio
rm -rf /tmp/audio/.git
cd /tmp/audio
export BUILD_VERSION=2.11.0+cu133 PYTORCH_VERSION=2.13.0
export USE_CUDA=1 TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS:-8}
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void BUILD_CPP_TEST=0
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" .
wheel=$(find "$out" -maxdepth 1 -name 'torchaudio-2.11.0+cu133-*.whl')
test -n "$wheel"
python3 - "$wheel" /tmp/torchaudio-wheel <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[2]); root.mkdir()
with zipfile.ZipFile(sys.argv[1]) as wheel:
    native=[name for name in wheel.namelist() if name.startswith('torchaudio/') and name.endswith('.so')]
    assert native, native
    for name in native: wheel.extract(name,root)
PY
find /tmp/torchaudio-wheel -type f -name '*.so' -print0 | while IFS= read -r -d '' module; do
  readelf -h "$module" | grep -F AArch64 >/dev/null
done
sha256sum "$wheel" > "$out/SHA256SUMS"
printf 'status=compiled-import-and-functional-qualification-pending\nsource.commit=%s\n' \
  "$expected_commit" > "$out/source-receipt.txt"
