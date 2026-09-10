#!/usr/bin/env bash
# Package exact R33 B12X sources as a platform-independent wheel.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=d95137245253d5c145e4b0700d677ce13b87ebea
expected_tree=8706b0426aa11483bf0d50a205588382d2d0cfbd
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir" rev-parse HEAD^{tree})" = "$expected_tree"
test -z "$(git -C "$source_dir" status --porcelain)"
cp -a "$source_dir" /tmp/b12x
rm -rf /tmp/b12x/.git
cd /tmp/b12x
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" .
test "$(find "$out" -maxdepth 1 -name 'b12x-1.3.0-*.whl' | wc -l)" -eq 1
python3 - "$out"/b12x-1.3.0-*.whl <<'PY'
import sys,zipfile
with zipfile.ZipFile(sys.argv[1]) as wheel:
    names=set(wheel.namelist())
    assert any(name.startswith('b12x/policy/_profiles/data/') and name.endswith('.json.gz') for name in names)
    assert any(name.startswith('b12x/comm/roce/') and name.endswith('.c') for name in names)
    assert any(name.startswith('b12x/loader/') and name.endswith('.c') for name in names)
PY
sha256sum "$out"/*.whl > "$out/SHA256SUMS"
printf 'status=packaged-cpu-import-and-gpu-qualification-pending\nsource.commit=%s\nsource.tree=%s\n' \
  "$expected_commit" "$expected_tree" > "$out/source-receipt.txt"
