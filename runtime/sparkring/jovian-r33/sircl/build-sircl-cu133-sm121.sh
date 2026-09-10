#!/usr/bin/env bash
# Build a source-locked SIRCL artifact inside the ARM64 R33 foundation image.
set -euo pipefail

readonly public_commit=f26895a158f586918b34136367fc63c060af6a60
readonly transport_tree=3ea295fb7adc7e8c1823412f9b1cc549679eea77
readonly mesh_pins_blob=3267552b436bd67f61b8bdc3c0a3e2f7ea0f6596
readonly mesh_source_manifest_blob=6da6f3ac05510e4461f49c763816be2882325489
readonly mesh_bundle_manifest_blob=069c85bd5bc0faeb07ee3e5d58a50578daf44b97

source_repo=${1:?usage: build-sircl-cu133-sm121.sh SOURCE_REPOSITORY OUTPUT_DIRECTORY}
output_dir=${2:?usage: build-sircl-cu133-sm121.sh SOURCE_REPOSITORY OUTPUT_DIRECTORY}
cuda_root=${CUDA_HOME:-/usr/local/cuda-13.3}
parallelism=${SPARKRING_SIRCL_BUILD_JOBS:-8}

fail() {
  printf 'SIRCL R33 build refused: %s\n' "$*" >&2
  exit 2
}

test "$(uname -m)" = aarch64 || fail "host architecture must be aarch64"
test ! -e "$output_dir" || fail "output directory already exists: $output_dir"
test -x "$cuda_root/bin/nvcc" || fail "nvcc is missing at $cuda_root/bin/nvcc"
test -d "$source_repo/.git" || fail "source repository is not a Git checkout"
git -C "$source_repo" cat-file -e "${public_commit}^{commit}" 2>/dev/null || \
  fail "public source commit is unavailable: $public_commit"
test "$(git -C "$source_repo" rev-parse "$public_commit:spark_transport")" = \
  "$transport_tree" || fail "spark_transport tree does not match the lock"
test "$(git -C "$source_repo" rev-parse "$public_commit:runtime/glm53-spark-mtp3-mesh/pins.json")" = \
  "$mesh_pins_blob" || fail "mesh pins do not match the lock"
test "$(git -C "$source_repo" rev-parse "$public_commit:runtime/glm53-spark-mtp3-mesh/performance/transport/source-manifest.json")" = \
  "$mesh_source_manifest_blob" || fail "mesh source manifest does not match the lock"
test "$(git -C "$source_repo" rev-parse "$public_commit:runtime/glm53-spark-mtp3-mesh/performance/transport/bundle-source/sparkring-overlay-manifest.json")" = \
  "$mesh_bundle_manifest_blob" || fail "mesh bundle manifest does not match the lock"

nvcc_version=$($cuda_root/bin/nvcc --version)
printf '%s\n' "$nvcc_version" | grep -Eq 'release 13\.3|V13\.3\.' || \
  fail "nvcc is not CUDA 13.3"

mkdir -p "$output_dir/source" "$output_dir/build" "$output_dir/artifacts" \
  "$output_dir/logs" "$output_dir/inspection"
git -C "$source_repo" archive --format=tar \
  --output="$output_dir/source/sparkring-${public_commit}.tar" \
  "$public_commit" LICENSE spark_transport
tar -xf "$output_dir/source/sparkring-${public_commit}.tar" \
  -C "$output_dir/source"

cmake -S "$output_dir/source/spark_transport" -B "$output_dir/build" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$cuda_root/bin/nvcc" \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DBUILD_TESTING=ON \
  -DSPARK_TP4_ENABLE_FUSED_STREAM_SWITCH_SMOKE=ON \
  -DSPARK_TP4_ENABLE_FUSED_PREFILL_PROBE=ON \
  -DSPARK_TILED_PREFILL_ENABLE_BIDIRECTIONAL_CUDA_SMOKE=ON \
  2>&1 | tee "$output_dir/logs/cmake-configure.log"
cmake --build "$output_dir/build" --parallel "$parallelism" \
  2>&1 | tee "$output_dir/logs/cmake-build.log"
ctest --test-dir "$output_dir/build" --output-on-failure \
  2>&1 | tee "$output_dir/logs/ctest.log"

artifact="$output_dir/build/libspark_transport_capi.so"
test -f "$artifact" || fail "libspark_transport_capi.so was not produced"
cp "$artifact" "$output_dir/artifacts/libspark_transport_capi.so"
cp "$output_dir/build/spark_tp4_fused_prefill_probe" \
  "$output_dir/artifacts/spark_tp4_fused_prefill_probe"
cp "$output_dir/build/tp4_fused_stream_switch_smoke" \
  "$output_dir/artifacts/tp4_fused_stream_switch_smoke"

file "$artifact" | tee "$output_dir/inspection/file.txt"
readelf -h -d -n "$artifact" > "$output_dir/inspection/readelf.txt"
ldd "$artifact" > "$output_dir/inspection/ldd.txt"
if grep -q 'not found' "$output_dir/inspection/ldd.txt"; then
  fail "artifact has unresolved dynamic dependencies"
fi
"$cuda_root/bin/cuobjdump" --list-elf "$artifact" \
  > "$output_dir/inspection/cuobjdump-list-elf.txt"
grep -q 'sm_121' "$output_dir/inspection/cuobjdump-list-elf.txt" || \
  fail "artifact does not contain an sm_121 CUDA image"
grep -Eq 'Machine:[[:space:]]+AArch64' "$output_dir/inspection/readelf.txt" || \
  fail "artifact is not AArch64"

sha256sum "$output_dir/source/sparkring-${public_commit}.tar" \
  "$output_dir/artifacts/"* "$output_dir/logs/"* \
  "$output_dir/inspection/"* > "$output_dir/SHA256SUMS"

PUBLIC_COMMIT="$public_commit" TRANSPORT_TREE="$transport_tree" \
CUDA_ROOT="$cuda_root" OUTPUT_DIR="$output_dir" python3 -S - <<'PY'
import hashlib
import json
import os
import pathlib
import platform
import re
import subprocess

out = pathlib.Path(os.environ["OUTPUT_DIR"])
artifact = out / "artifacts" / "libspark_transport_capi.so"
ctest = (out / "logs" / "ctest.log").read_text(errors="replace")
match = re.search(r"(\d+)% tests passed, (\d+) tests failed out of (\d+)", ctest)
if match is None or match.group(2) != "0":
    raise SystemExit("cannot produce receipt: CTest summary is missing or failed")

def command(*args: str) -> str:
    return subprocess.check_output(args, text=True).strip()

def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

receipt = {
    "schema": "sparkring-r33-sircl-native-build/v1",
    "status": "native-built-tested",
    "source": {
        "repository": "https://github.com/FujitsuPolycom/sparkring.git",
        "public_commit": os.environ["PUBLIC_COMMIT"],
        "spark_transport_tree": os.environ["TRANSPORT_TREE"],
        "source_archive_sha256": digest(
            out / "source" / f"sparkring-{os.environ['PUBLIC_COMMIT']}.tar"
        ),
    },
    "native": {
        "artifact": "libspark_transport_capi.so",
        "sha256": digest(artifact),
        "host_architecture": platform.machine(),
        "cuda": command(os.environ["CUDA_ROOT"] + "/bin/nvcc", "--version"),
        "cmake": command("cmake", "--version").splitlines()[0],
        "cxx": command("c++", "--version").splitlines()[0],
        "generator": "Ninja",
        "build_type": "Release",
        "cuda_architectures": [121],
    },
    "validation": {
        "ctest_total": int(match.group(3)),
        "ctest_passed": int(match.group(3)) - int(match.group(2)),
        "ctest_failed": int(match.group(2)),
        "sm121_image_present": True,
        "fused_endpoint_four_rank": "pending",
        "fused_c_api_alternating_streams": "pending",
    },
    "qualification": "research-only; four-rank RDMA and serving gates remain",
}
(out / "build-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
PY
sha256sum "$output_dir/build-receipt.json" >> "$output_dir/SHA256SUMS"
printf 'Built %s\n' "$output_dir/artifacts/libspark_transport_capi.so"
