#!/usr/bin/env bash
# Build and qualify SparkCache's exact CUDA placement and snapshot libraries.
set -euo pipefail

source_dir=/source
build_dir=/build
out_dir=/out
expected_commit=f220230a5a85b94af8a296187241b6aacc3ed724
expected_tree=86ef46de45dd0f4ed776b86109a30df6f83db557

exec > >(tee "$out_dir/build.log") 2>&1

git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$source_dir" status --porcelain)"
test "$(uname -m)" = aarch64
test "${CUDA_VISIBLE_DEVICES:-}" = 0

git -C "$source_dir" archive --format=tar "$expected_commit" | gzip -n > \
  "$out_dir/sparkcache-f220230a-source.tar.gz"

cmake -S "$source_dir/sparkcache/native" -B "$build_dir" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=ON \
  -DSPARK_CACHE_PLACEMENT_GPU_TESTS=ON
cmake --build "$build_dir" --parallel 2

ctest --test-dir "$build_dir" --exclude-regex 'page_copy' --output-on-failure | \
  tee "$out_dir/ctest.log"
PYTHONPATH=/pytest-site-packages python3 -m pytest -q \
  "$source_dir/sparkcache/native/tests" | tee "$out_dir/native-pytest.log"

install -m 0755 "$build_dir/libspark_cache_placement.so" \
  "$out_dir/libspark_cache_placement.so"
install -m 0755 "$build_dir/libspark_cache_snapshot.so" \
  "$out_dir/libspark_cache_snapshot.so"
for probe in \
  spark_cache_native_probe \
  spark_cache_hybrid_page_probe \
  spark_cache_snapshot_probe \
  spark_cache_snapshot_matrix_probe \
  spark_cache_page_copy_benchmark; do
  install -m 0755 "$build_dir/$probe" "$out_dir/$probe"
done

(
  cd "$out_dir"
  sha256sum libspark_cache_placement.so \
    libspark_cache_snapshot.so \
    sparkcache-f220230a-source.tar.gz
) > "$out_dir/SHA256SUMS"

readelf -h "$out_dir/libspark_cache_placement.so" > "$out_dir/placement-readelf-header.txt"
readelf -d "$out_dir/libspark_cache_placement.so" > "$out_dir/placement-readelf-dynamic.txt"
readelf -Ws "$out_dir/libspark_cache_placement.so" > "$out_dir/placement-readelf-symbols.txt"
readelf -h "$out_dir/libspark_cache_snapshot.so" > "$out_dir/snapshot-readelf-header.txt"
readelf -d "$out_dir/libspark_cache_snapshot.so" > "$out_dir/snapshot-readelf-dynamic.txt"
readelf -Ws "$out_dir/libspark_cache_snapshot.so" > "$out_dir/snapshot-readelf-symbols.txt"
cuobjdump --list-elf "$out_dir/libspark_cache_placement.so" > "$out_dir/placement-cuobjdump.txt"
cuobjdump --list-elf "$out_dir/libspark_cache_snapshot.so" > "$out_dir/snapshot-cuobjdump.txt"

PYTHONPATH="$source_dir" python3 \
  "$source_dir/sparkcache/native/app/spark_cache_ctypes_probe.py" \
  "$out_dir/libspark_cache_placement.so" | tee "$out_dir/placement-ctypes.log"
snapshot_sha=$(sha256sum "$out_dir/libspark_cache_snapshot.so" | awk '{print $1}')
PYTHONPATH="$source_dir" python3 \
  "$source_dir/sparkcache/native/app/spark_cache_snapshot_ctypes_probe.py" \
  "$out_dir/libspark_cache_snapshot.so" --sha256 "$snapshot_sha" | \
  tee "$out_dir/snapshot-ctypes.log"

"$out_dir/spark_cache_native_probe" | tee "$out_dir/placement-gpu-probe.log"
"$out_dir/spark_cache_hybrid_page_probe" | tee "$out_dir/hybrid-page-gpu-probe.log"
"$out_dir/spark_cache_snapshot_probe" | tee "$out_dir/snapshot-gpu-probe.log"
# At f220230a the benchmark's first 257 MiB fixture requires at least 650 MiB
# by its own accounting, then rejects allocations >=512 MiB. Preserve that
# exact-source contradiction as a qualification limit while relying on the
# dedicated hybrid-page C API probe for bounded page-copy correctness.
set +e
ctest --test-dir "$build_dir" -L page-copy --output-on-failure 2>&1 | \
  tee "$out_dir/page-copy-upstream-known-failure.log"
page_copy_status=${PIPESTATUS[0]}
set -e
test "$page_copy_status" -eq 8
test "$(grep -c 'allocation ceiling exceeded' \
  "$out_dir/page-copy-upstream-known-failure.log")" -eq 3
"$out_dir/spark_cache_snapshot_matrix_probe" \
  --arena mapped --slots 2 --rank 0 --rows 64 --iterations 3 \
  --compare-every 1 --pipeline-depth 2 --writer-hold-us 0 \
  --profile compact --slot-mib 2 --saturation-cycles 1 --overlap-samples 1 | \
  tee "$out_dir/snapshot-matrix.jsonl"

python3 /scripts/make_receipt.py \
  --source "$source_dir" --build "$build_dir" --out "$out_dir"
