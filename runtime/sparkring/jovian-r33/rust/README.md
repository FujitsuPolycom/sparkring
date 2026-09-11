# R33 ARM64 Rust frontend build

`launch_rust_build.sh` creates an isolated source composition at
`/var/tmp/sparkring-r33-20260910/sources/vllm-rust`, applies the reviewed
Python-only overlay, verifies its exact Git tree before and after the build, and starts a four-CPU,
24-GiB container against the R33 ARM64 foundation.

`build_rust_frontend.sh` downloads the Rust 1.95.0 aarch64 toolchain named by
Rust's release manifest, verifies both the manifest and toolchain SHA-256,
builds with `Cargo.lock` and `--locked`, enables PyO3's extension-module and
ABI3-Python-3.8 features, and emits the `vllm-rs` executable and
`_rust_tool_parser.abi3.so`. It verifies both as AArch64 ELF files, resolves
their dynamic libraries (including absence of a direct `libpython` dependency), runs the CLI version command, and imports the parser
with Python 3.12 without loading CUDA.

Artifacts and receipts are written to
`/var/tmp/sparkring-r33-20260910/artifacts/rust` on the build rank.
