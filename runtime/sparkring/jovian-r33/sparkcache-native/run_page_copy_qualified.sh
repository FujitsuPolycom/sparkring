#!/usr/bin/env bash
# Qualify the exact placement library with a test-only corrected benchmark ceiling.
set -euo pipefail

source_dir=/source
test_source=/test-work/source
test_build=/test-build
artifact_dir=/artifacts
out_dir=/out
patch_file=/scripts/page-copy-benchmark-1g-ceiling-64mib-slabs.patch
expected_commit=f220230a5a85b94af8a296187241b6aacc3ed724
expected_tree=86ef46de45dd0f4ed776b86109a30df6f83db557
expected_library_sha=d89c9fdae8dc99ae3f7a151cc3dd9e92fdc8fd0b994069fc263027fd4d056c93

exec > >(tee "$out_dir/run.log") 2>&1
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir" rev-parse 'HEAD^{tree}')" = "$expected_tree"
test -z "$(git -C "$source_dir" status --porcelain)"
test "$(sha256sum "$artifact_dir/libspark_cache_placement.so" | awk '{print $1}')" = \
  "$expected_library_sha"

mkdir -p "$test_source" "$test_build"
git -C "$source_dir" archive "$expected_commit" | tar -x -C "$test_source"
patch -d "$test_source" -p1 --forward --batch < "$patch_file"
grep -q 'reuse_arena = std::strcmp' \
  "$test_source/sparkcache/native/app/spark_cache_page_copy_benchmark.cu"
grep -q 'explicit_allocations >= 1024 \* MiB' \
  "$test_source/sparkcache/native/app/spark_cache_page_copy_benchmark.cu"

cmake -S "$test_source/sparkcache/native" -B "$test_build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=ON \
  -DSPARK_CACHE_PLACEMENT_GPU_TESTS=OFF
cmake --build "$test_build" --target spark_cache_page_copy_benchmark --parallel 2

for mode in 1 2 3; do
  python3 -c 'import json, torch; free, total = torch.cuda.mem_get_info(); print(json.dumps({"free_bytes": free, "total_bytes": total}))' \
    > "$out_dir/mode-${mode}-cuda-before.json"
  "$test_build/spark_cache_page_copy_benchmark" \
    "$artifact_dir/libspark_cache_placement.so" 3 "$mode" | \
    tee "$out_dir/mode-${mode}.jsonl"
  test "$(grep -c '"byte_equal":true' "$out_dir/mode-${mode}.jsonl")" -eq 6
  python3 -c 'import json, torch; free, total = torch.cuda.mem_get_info(); print(json.dumps({"free_bytes": free, "total_bytes": total}))' \
    > "$out_dir/mode-${mode}-cuda-after.json"
done

test "$(sha256sum "$artifact_dir/libspark_cache_placement.so" | awk '{print $1}')" = \
  "$expected_library_sha"
install -m 0644 "$patch_file" \
  "$out_dir/page-copy-benchmark-1g-ceiling-64mib-slabs.patch"
sha256sum "$out_dir/page-copy-benchmark-1g-ceiling-64mib-slabs.patch" > \
  "$out_dir/PATCH-SHA256"
python3 /scripts/make_page_copy_receipt.py --out "$out_dir"
