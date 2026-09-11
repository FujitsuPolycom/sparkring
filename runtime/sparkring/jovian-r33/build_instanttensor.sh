#!/usr/bin/env bash
# Build exact InstantTensor and its pinned I/O submodules for ARM64.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=49b4010afc1cae0441e71fe0b0bffc24fa05e932
expected_libaio=1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448
git config --global --add safe.directory "$source_dir"
git config --global --add safe.directory "$source_dir/csrc/third_party/libaio"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir/csrc/third_party/libaio" rev-parse HEAD)" = "$expected_libaio"
test -z "$(git -C "$source_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
cp -a "$source_dir" /tmp/instanttensor
rm -rf /tmp/instanttensor/.git
cd /tmp/instanttensor
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" .
wheel=$(find "$out" -maxdepth 1 -name 'instanttensor-0.1.9-*.whl')
test -n "$wheel"
python3 - "$wheel" /tmp/instanttensor-wheel <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[2]); root.mkdir()
with zipfile.ZipFile(sys.argv[1]) as wheel:
    native=[name for name in wheel.namelist() if name.startswith('instanttensor/') and name.endswith('.so')]
    assert len(native)==1, native
    wheel.extract(native[0],root)
    print(root/native[0])
PY
native=$(find /tmp/instanttensor-wheel -name '*.so')
readelf -h "$native" | grep -F AArch64
readelf -d "$native" > "$out/native-dynamic.txt"
sha256sum "$wheel" > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-gpu-qualification-pending\nsource.commit=%s\nlibaio.commit=%s\n' \
  "$expected_commit" "$expected_libaio" > "$out/source-receipt.txt"
