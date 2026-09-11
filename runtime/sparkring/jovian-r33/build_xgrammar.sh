#!/usr/bin/env bash
# Build the R33 XGrammar revision for ARM64.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=2ea71da4ccb997a06928c9fb69b99f330da56697
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
python3 -m pip install 'scikit-build-core>=0.10.0' 'apache-tvm-ffi==0.1.11' build
python3 -m pip freeze > "$out/build-dependencies.txt"
cp -a "$source_dir" /tmp/xgrammar
rm -rf /tmp/xgrammar/.git
cd /tmp/xgrammar
export CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS:-8}
python3 -m build --wheel --no-isolation --outdir "$out"
wheel=$(find "$out" -maxdepth 1 -name 'xgrammar-0.2.5-*.whl')
test -n "$wheel"
python3 - "$wheel" /tmp/xgrammar-wheel <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[2]); root.mkdir()
with zipfile.ZipFile(sys.argv[1]) as wheel:
    native=[name for name in wheel.namelist() if name.endswith(('.so','.so.0'))]
    assert native, native
    for name in native: wheel.extract(name,root)
PY
find /tmp/xgrammar-wheel -type f -name '*.so*' -print0 | while IFS= read -r -d '' module; do
  readelf -h "$module" | grep -F AArch64 >/dev/null
done
sha256sum "$wheel" > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-runtime-qualification-pending\nsource.commit=%s\n' \
  "$expected_commit" > "$out/source-receipt.txt"
