#!/usr/bin/env bash
# Build every R33 LMCache extension and its cuMem interposer for ARM64.
set -euo pipefail

source_dir=/source
out=/out
expected_commit=29bc5a2efde737c436b04499eb62cd1776cebeec
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected_commit"
test -z "$(git -C "$source_dir" status --porcelain)"
python3 /opt/sparkring-build/verify_torch.py > "$out/foundation-verification.json"
python3 -m pip install 'setuptools>=77,<81' 'setuptools-scm>=8' ninja packaging wheel
cp -a "$source_dir" /tmp/lmcache
rm -rf /tmp/lmcache/.git
cd /tmp/lmcache
unset NO_NATIVE_EXT NO_GPU_EXT NO_CUDA_EXT
export BUILD_WITH_CUDA=1 LMCACHE_CUDA_MAJOR=13 ENABLE_CXX11_ABI=1
export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS=${MAX_JOBS:-12} NVCC_THREADS=1
export CUDA_VISIBLE_DEVICES= NVIDIA_VISIBLE_DEVICES=void
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.5.5.dev0+glm53checkpoints.29bc5a2e
python3 -m pip wheel --no-build-isolation --no-deps -w "$out" .
wheel=$(find "$out" -maxdepth 1 -name 'lmcache-0.5.5.dev0+glm53checkpoints.29bc5a2e-*.whl')
test -n "$wheel"
python3 - "$wheel" /tmp/lmcache-wheel <<'PY'
import pathlib,sys,zipfile
root=pathlib.Path(sys.argv[2]); root.mkdir()
with zipfile.ZipFile(sys.argv[1]) as wheel:
    native=[name for name in wheel.namelist() if name.startswith('lmcache/') and name.endswith('.so')]
    stems={pathlib.PurePosixPath(name).name.split('.')[0] for name in native}
    assert {'cuda_ops','lmcache_native','lmcache_fs','lmcache_redis'} <= stems, (stems,native)
    for name in native: wheel.extract(name,root)
PY
find /tmp/lmcache-wheel -type f -name '*.so' -print0 | while IFS= read -r -d '' module; do
  readelf -h "$module" | grep -F AArch64 >/dev/null
done
make -C csrc/cumem_ipc_interposer CUDA_HOME=/usr/local/cuda
install -m755 csrc/cumem_ipc_interposer/liblmcache_cumem_shareable.so "$out/"
readelf -h "$out/liblmcache_cumem_shareable.so" | grep -F AArch64
nm -D "$out/liblmcache_cumem_shareable.so" | grep -E ' T cuMemCreate$'
sha256sum "$wheel" "$out/liblmcache_cumem_shareable.so" > "$out/SHA256SUMS"
printf 'status=compiled-cpu-import-and-gpu-qualification-pending\nsource.commit=%s\n' \
  "$expected_commit" > "$out/source-receipt.txt"
