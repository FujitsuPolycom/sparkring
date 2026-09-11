#!/usr/bin/env bash
# Package the SparkCache source used by the R33 connector contract.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=f220230a5a85b94af8a296187241b6aacc3ed724
expected_tree=86ef46de45dd0f4ed776b86109a30df6f83db557
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir" rev-parse HEAD^{tree})" = "$expected_tree"
test -z "$(git -C "$source_dir" status --porcelain)"
cp -a "$source_dir" /tmp/sparkcache
rm -rf /tmp/sparkcache/.git
cd /tmp/sparkcache
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" .
test "$(find "$out" -maxdepth 1 -name 'sparkcache-0.1.0a3-*.whl' | wc -l)" -eq 1
sha256sum "$out"/*.whl > "$out/SHA256SUMS"
printf 'status=packaged-runtime-qualification-pending\nsource.commit=%s\nsource.tree=%s\n' \
  "$expected_commit" "$expected_tree" > "$out/source-receipt.txt"
