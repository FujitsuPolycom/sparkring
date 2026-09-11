#!/usr/bin/env bash
# Build R33 vLLM's Rust frontend and Python parser from locked ARM64 sources.
set -euo pipefail

source_dir=/source
build_dir=/build
out=/out
toolchain_dir="$build_dir/rust-1.95.0"
dist_dir="$build_dir/dist"

expected_commit=ae89131442359dc332d9c46009be3c1f8cdee0b4
expected_tree=386191c06df9c4232cb2f48012968f48cfdc6eee
expected_rust_tree=85c3cd52db223217d45377d3f7f884e756641de3
expected_toolchain_blob=4933b3ba170755e38ab3fc39cfc6bf952aabedcc
manifest_url=https://static.rust-lang.org/dist/channel-rust-1.95.0.toml
manifest_sha=821ff14e4c4a1cbe1e8915f35aff0a3fbbdf8d293ad48ab8f31e3b0440c581f9
toolchain_url=https://static.rust-lang.org/dist/2026-04-16/rust-1.95.0-aarch64-unknown-linux-gnu.tar.xz
toolchain_sha=094c9c36531911c5cc7dd6ab2d3069ab8dcd744d6239b0bda1387b243dfc391e

mkdir -p "$build_dir" "$dist_dir" "$out"
git config --global --add safe.directory "$source_dir"
test "$(uname -m)" = aarch64
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test "$(git -C "$source_dir" write-tree)" = "$expected_tree"
test "$(git -C "$source_dir" rev-parse HEAD:rust)" = "$expected_rust_tree"
test "$(git -C "$source_dir" rev-parse HEAD:rust-toolchain.toml)" = "$expected_toolchain_blob"
test "$(git -C "$source_dir" diff --cached --name-only -- rust rust-toolchain.toml | wc -l)" = 0
test "$(git -C "$source_dir" diff --name-only -- rust rust-toolchain.toml | wc -l)" = 0
test "$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')" = 3.12

curl --fail --location --retry 5 --retry-all-errors -o "$dist_dir/channel-rust-1.95.0.toml" "$manifest_url"
printf '%s  %s\n' "$manifest_sha" "$dist_dir/channel-rust-1.95.0.toml" | sha256sum -c -
grep -F "xz_url = \"$toolchain_url\"" "$dist_dir/channel-rust-1.95.0.toml"
grep -F "xz_hash = \"$toolchain_sha\"" "$dist_dir/channel-rust-1.95.0.toml"

toolchain_archive="$dist_dir/rust-1.95.0-aarch64-unknown-linux-gnu.tar.xz"
if [[ ! -f "$toolchain_archive" ]] || ! printf '%s  %s\n' "$toolchain_sha" "$toolchain_archive" | sha256sum -c -; then
  rm -f "$toolchain_archive"
  curl --fail --location --retry 5 --retry-all-errors -o "$toolchain_archive" "$toolchain_url"
fi
printf '%s  %s\n' "$toolchain_sha" "$toolchain_archive" | sha256sum -c -

if [[ ! -x "$toolchain_dir/bin/rustc" ]]; then
  rm -rf "$build_dir/toolchain-unpack"
  mkdir -p "$build_dir/toolchain-unpack"
  tar -xJf "$toolchain_archive" -C "$build_dir/toolchain-unpack"
  "$build_dir/toolchain-unpack/rust-1.95.0-aarch64-unknown-linux-gnu/install.sh" \
    --prefix="$toolchain_dir" --without=rust-docs --disable-ldconfig
fi

export PATH="$toolchain_dir/bin:$PATH"
export CARGO_HOME="$build_dir/cargo-home"
export CARGO_TARGET_DIR="$build_dir/target"
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
export PYO3_PYTHON=/usr/bin/python3
export VLLM_RS_BUILD_VERSION=0.1.0+ae891314

rustc --version --verbose | tee "$out/rustc-version.txt"
cargo --version --verbose | tee "$out/cargo-version.txt"
test "$(rustc --version | awk '{print $2}')" = 1.95.0

cargo build --manifest-path "$source_dir/rust/Cargo.toml" --locked --release \
  -p vllm-cmd --features native-tls-vendored
cargo build --manifest-path "$source_dir/rust/Cargo.toml" --locked --release \
  -p vllm-tool-parser-py --features pyo3/abi3-py38,pyo3/extension-module

install -m755 "$CARGO_TARGET_DIR/release/vllm-rs" "$out/vllm-rs"
parser_lib=$(find "$CARGO_TARGET_DIR/release" -maxdepth 1 -type f \
  \( -name 'lib_rust_tool_parser.so' -o -name '_rust_tool_parser.so' \) -print -quit)
test -n "$parser_lib"
install -m755 "$parser_lib" "$out/_rust_tool_parser.abi3.so"

readelf -h "$out/vllm-rs" > "$out/vllm-rs-elf.txt"
readelf -h "$out/_rust_tool_parser.abi3.so" > "$out/rust-tool-parser-elf.txt"
grep -F AArch64 "$out/vllm-rs-elf.txt"
grep -F AArch64 "$out/rust-tool-parser-elf.txt"
ldd "$out/vllm-rs" > "$out/vllm-rs-ldd.txt"
ldd "$out/_rust_tool_parser.abi3.so" > "$out/rust-tool-parser-ldd.txt"
! grep -F 'not found' "$out/vllm-rs-ldd.txt" "$out/rust-tool-parser-ldd.txt"
! grep -F 'libpython' "$out/rust-tool-parser-ldd.txt"
"$out/vllm-rs" --version | tee "$out/vllm-rs-version.txt"
PYTHONPATH="$out" python3 -c 'import _rust_tool_parser; print(_rust_tool_parser.__name__)' \
  | tee "$out/rust-tool-parser-import.txt"

sha256sum "$source_dir/rust/Cargo.lock" > "$out/Cargo.lock.sha256"
sha256sum "$out/vllm-rs" "$out/_rust_tool_parser.abi3.so" > "$out/SHA256SUMS"
test "$(git -C "$source_dir" write-tree)" = "$expected_tree"
test "$(git -C "$source_dir" diff --name-only -- rust rust-toolchain.toml | wc -l)" = 0
{
  printf 'status=compiled-and-cpu-verified\n'
  printf 'architecture=aarch64\n'
  printf 'python.version=%s\n' "$(python3 --version 2>&1)"
  printf 'vllm.commit=%s\n' "$expected_commit"
  printf 'vllm.result.tree=%s\n' "$expected_tree"
  printf 'vllm.rust.tree=%s\n' "$expected_rust_tree"
  printf 'vllm.rust_toolchain.blob=%s\n' "$expected_toolchain_blob"
  printf 'rust.manifest.url=%s\n' "$manifest_url"
  printf 'rust.manifest.sha256=%s\n' "$manifest_sha"
  printf 'rust.toolchain.url=%s\n' "$toolchain_url"
  printf 'rust.toolchain.sha256=%s\n' "$toolchain_sha"
  printf 'rustc.version=%s\n' "$(rustc --version)"
  printf 'cargo.version=%s\n' "$(cargo --version)"
} > "$out/source-receipt.txt"
