#!/usr/bin/env bash
# Materialize CMake's authoritative package layout for the completed vLLM build.
set -euo pipefail

build_dir=/build
out=/out
prefix="$out/install"
test -f "$build_dir/build.ninja"
test ! -e "$prefix"
cmake --install "$build_dir" --prefix "$prefix"
find "$prefix" -type f -print0 | sort -z | xargs -0 sha256sum > "$out/INSTALL-SHA256SUMS"
test -n "$(find "$prefix" -type f -name '_C_stable_libtorch*.so' -print -quit)"
test -n "$(find "$prefix" -type f -name '_flashkda_C*.so' -print -quit)"
find "$prefix" -type f -name '*.so' -print0 | while IFS= read -r -d '' module; do
  readelf -h "$module" | grep -F AArch64 >/dev/null
done
printf 'status=cmake-install-layout-materialized-runtime-import-pending\n' \
  > "$out/install-receipt.txt"
