#!/usr/bin/env bash
# Package the exact staged R33 vLLM tree with already-built ARM64 native/Rust artifacts.
set -euo pipefail

root=${SPARKRING_R33_ROOT:-/var/tmp/sparkring-r33-20260910}
source_dir=${VLLM_SOURCE_DIR:-$root/sources/vllm-sparkring}
native_dir=${VLLM_NATIVE_DIR:-$root/artifacts/vllm-native}
rust_dir=${VLLM_RUST_DIR:-$root/artifacts/rust}
work_root=${VLLM_PACKAGE_WORK_ROOT:-$root/work/vllm-package-0511a786}
out_dir=${VLLM_PACKAGE_OUT_DIR:-$root/artifacts/vllm-package}
foundation_image=${SPARKRING_R33_FOUNDATION_IMAGE:-local/sparkring:r33-arm64-foundation}
version=${VLLM_VERSION_OVERRIDE:-0.26.1rc0+sparkring.r33.0511a786}
expected_head=ae89131442359dc332d9c46009be3c1f8cdee0b4
expected_package_tree=0511a78617bb755ea2901ef3c5db7547bc1e148d
expected_native_tree=386191c06df9c4232cb2f48012968f48cfdc6eee
expected_native_inputs_sha=bf4b2150ac4937b325d9b0ff1b0d676e12405e2f4ca940159b959cc7a4bde9e1
flash_attn_source_dir=${VLLM_FLASH_ATTN_SOURCE_DIR:-$root/build/vllm-native/_deps/vllm-flash-attn-src}
expected_flash_attn_commit=f3e1a4f74c99145c0717709860bf765de1703779
expected_rust_bin_sha=cd1cdb2539c79793a92292479b4ec6b99d4e3be3e1a2cadb0fc3f592a99fba5d
expected_rust_parser_sha=b52494b9f599acc71ccf9e63523fa3b2b173cf395d155eca37b0941236e2f28c

test "$(uname -m)" = aarch64
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_head"
test "$(git -C "$source_dir" write-tree)" = "$expected_package_tree"
test "$(git -C "$source_dir" diff --cached --binary | sha256sum | cut -d' ' -f1)" = 73df604d183a3309109ac3e97e1a952e99d111b5761ef2e67fbf41a7957f4ae2
test "$(git -C "$source_dir" status --porcelain=v1 --untracked-files=all | sha256sum | cut -d' ' -f1)" = 0eda60c3efcc6a9fbc65e131f1b0164c582cc7de4783fb284cdfa82df6a6bcfe
test "$(git -C "$source_dir" status --porcelain=v1 --untracked-files=all | wc -l)" = 33
test "$(git -C "$source_dir" ls-tree -r "$expected_package_tree" -- CMakeLists.txt cmake csrc rust | sha256sum | cut -d' ' -f1)" = "$expected_native_inputs_sha"
test "$(git -C "$source_dir" ls-tree -r "$expected_native_tree" -- CMakeLists.txt cmake csrc rust | sha256sum | cut -d' ' -f1)" = "$expected_native_inputs_sha"
test -d "$native_dir/install/vllm"
test -d "$native_dir/modules"
test "$(git -c safe.directory="$flash_attn_source_dir" -C "$flash_attn_source_dir" rev-parse HEAD)" = "$expected_flash_attn_commit"
printf '%s  %s\n' "$expected_rust_bin_sha" "$rust_dir/vllm-rs" | sha256sum --check --strict
printf '%s  %s\n' "$expected_rust_parser_sha" "$rust_dir/_rust_tool_parser.abi3.so" | sha256sum --check --strict

# Every run gets a new, task-owned staging directory; source and native build trees stay read-only inputs.
test ! -e "$work_root"
mkdir -p "$work_root/source" "$work_root/dist-initial" "$work_root/dist-final" "$work_root/evidence" "$out_dir"
git -C "$source_dir" archive "$expected_package_tree" | tar -x -C "$work_root/source"

# vLLM imports the Python helpers built from its commit-pinned FlashAttention
# dependency. The vLLM repository contains only the package facade, while CMake
# supplies layers/ and ops/ from this external source. Materialize those exact
# Python files even when native modules are staged without a full CMake install.
mkdir -p "$work_root/flash-attn-python"
git -c safe.directory="$flash_attn_source_dir" -C "$flash_attn_source_dir" \
  archive "$expected_flash_attn_commit" -- vllm_flash_attn/layers vllm_flash_attn/ops \
  | tar -x -C "$work_root/flash-attn-python"
find "$work_root/flash-attn-python/vllm_flash_attn" -type f ! -name '*.py' -delete
cp -a "$work_root/flash-attn-python/vllm_flash_attn/." \
  "$work_root/source/vllm/vllm_flash_attn/"

# Start from CMake's install layout, then add every completed native target. Some optional targets have
# component-only CMake installs, so the all-target build receipt is the authoritative supplement.
cp -a "$native_dir/install/." "$work_root/source/"
install -m755 "$native_dir/modules/_C_stable_libtorch.abi3.so" "$work_root/source/vllm/"
install -m755 "$native_dir/modules/cumem_allocator.abi3.so" "$work_root/source/vllm/"
install -m755 "$native_dir/modules/fs_io_C.abi3.so" "$work_root/source/vllm/"
install -m755 "$native_dir/modules/_flashkda_C.abi3.so" "$work_root/source/vllm/"
install -m755 "$native_dir/modules/_moe_C_stable_libtorch.abi3.so" "$work_root/source/vllm/"
install -m755 "$native_dir/modules/_qutlass_C.abi3.so" "$work_root/source/vllm/"
install -m755 "$native_dir/modules/spinloop.abi3.so" "$work_root/source/vllm/"
install -D -m755 "$native_dir/modules/deepgemm_C_cpython-312-aarch64-linux-gnu/_C.cpython-312-aarch64-linux-gnu.so" \
  "$work_root/source/vllm/third_party/deep_gemm/_C.cpython-312-aarch64-linux-gnu.so"
install -D -m755 "$native_dir/modules/vllm-flash-attn/_vllm_fa2_C.abi3.so" \
  "$work_root/source/vllm/vllm_flash_attn/_vllm_fa2_C.abi3.so"
install -D -m755 "$native_dir/modules/vllm-flash-attn/_vllm_fa3_C.abi3.so" \
  "$work_root/source/vllm/vllm_flash_attn/_vllm_fa3_C.abi3.so"
install -m755 "$rust_dir/vllm-rs" "$work_root/source/vllm/vllm-rs"
install -m755 "$rust_dir/_rust_tool_parser.abi3.so" "$work_root/source/vllm/_rust_tool_parser.abi3.so"

# This packaging-only setup shim makes build_ext a no-op and enumerates staged binary/data files.
# It is outside the locked source tree and is not included in the wheel.
python3 - "$work_root/source/setup.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
old = "USE_PRECOMPILED_EXTENSIONS = envs.VLLM_USE_PRECOMPILED"
new = "USE_PRECOMPILED_EXTENSIONS = (envs.VLLM_USE_PRECOMPILED or os.getenv('SPARKRING_PACKAGE_LOCAL_NATIVE') == '1')"
if text.count(old) != 1:
    raise SystemExit("unexpected USE_PRECOMPILED_EXTENSIONS assignment")
text = text.replace(old, new)
needle = "for rust_extension_path in get_precompiled_rust_extension_paths():\n    add_vllm_package_data(rust_extension_path.name)\n"
addition = needle + "\nif os.getenv('SPARKRING_PACKAGE_LOCAL_NATIVE') == '1':\n    for package_file in sorted((ROOT_DIR / 'vllm').rglob('*')):\n        if package_file.is_file():\n            add_vllm_package_data(package_file.relative_to(ROOT_DIR / 'vllm').as_posix())\n"
if text.count(needle) != 1:
    raise SystemExit("unexpected package-data hook")
path.write_text(text.replace(needle, addition))
PY

# The foundation supplies the exact Torch/CUDA ABI. Install only small Python packaging helpers into
# the task-owned mount, then build with all native/Rust compilation disabled by the shim above.
docker run --rm --network host \
  -v "$work_root:/work" \
  -w /work/source \
  "$foundation_image" \
  bash -ceu '
    python3 -m pip install --disable-pip-version-check --no-cache-dir --target /work/build-deps \
      "setuptools==80.9.0" "setuptools-scm==9.2.0" "setuptools-rust==1.12.0" "build==1.3.0"
    export PYTHONPATH=/work/build-deps
    export VLLM_TARGET_DEVICE=cuda
    export VLLM_MAIN_CUDA_VERSION=13.3
    export VLLM_VERSION_OVERRIDE="$1"
    export VLLM_RS_BUILD_VERSION="$1"
    export SPARKRING_PACKAGE_LOCAL_NATIVE=1
    export VLLM_REQUIRE_RUST_FRONTEND=1
    python3 -m build --wheel --no-isolation --outdir /work/dist-initial
    wheel tags --python-tag cp312 --abi-tag cp312 --platform-tag linux_aarch64 --remove /work/dist-initial/*.whl
    mv /work/dist-initial/*.whl /work/dist-final/
  ' package "$version"

wheel=$(find "$work_root/dist-final" -maxdepth 1 -type f -name '*.whl' -print -quit)
test -n "$wheel"
python3 "$(dirname "$0")/verify_vllm_wheel.py" \
  --wheel "$wheel" \
  --version "$version" \
  --source-tree "$expected_package_tree" \
  --source-dir "$source_dir" \
  --native-tree "$expected_native_tree" \
  --native-install "$native_dir/install" \
  --native-modules "$native_dir/modules" \
  --flash-attn-source-dir "$flash_attn_source_dir" \
  --flash-attn-commit "$expected_flash_attn_commit" \
  --rust-bin-sha256 "$expected_rust_bin_sha" \
  --rust-parser-sha256 "$expected_rust_parser_sha" \
  --output "$work_root/evidence/verification.json"

cp -a "$wheel" "$out_dir/"
cp -a "$work_root/evidence/." "$out_dir/"
(
  cd "$out_dir"
  sha256sum "$(basename "$wheel")" > SHA256SUMS
)
printf '%s\n' "$wheel"
