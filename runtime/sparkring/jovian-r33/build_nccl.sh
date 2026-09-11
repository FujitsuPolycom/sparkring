#!/usr/bin/env bash
set -euo pipefail

source_dir=${NCCL_SOURCE_DIR:-/work/sources/nccl-canonical}
prefix=${NCCL_INSTALL_PREFIX:-/opt/local-inference/nccl}
jobs=${MAX_JOBS:-16}
expected=fb6f40999a2a9e63104d4ae4a84118bce61528f8
# The checkout is bind-mounted from the host user into this root build container.
git config --global --add safe.directory "$source_dir"
test "$(git -C "$source_dir" rev-parse HEAD)" = "$expected"
test -z "$(git -C "$source_dir" status --porcelain --untracked-files=no)"
test "$(uname -m)" = aarch64
nvcc --list-gpu-code | grep -qx sm_121

make -C "$source_dir" -j"$jobs" src.build CUDA_HOME=/usr/local/cuda \
  NVCC_GENCODE='-gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_121,code=compute_121'
mkdir -p "$prefix/include" "$prefix/lib"
cp -a "$source_dir/build/include/." "$prefix/include/"
cp "$source_dir/build/lib/libnccl.so.2.31.2" "$prefix/lib/libnccl.so.2.31.2"
ln -sfn libnccl.so.2.31.2 "$prefix/lib/libnccl.so.2"
ln -sfn libnccl.so.2 "$prefix/lib/libnccl.so"
readelf -h "$prefix/lib/libnccl.so.2.31.2" | grep -q AArch64
python3 -S - "$source_dir" "$prefix" <<'PY'
import ctypes,hashlib,json,pathlib,subprocess,sys
source,prefix=map(pathlib.Path,sys.argv[1:])
library=prefix/'lib/libnccl.so.2.31.2'
loaded=ctypes.CDLL(str(library));version=ctypes.c_int()
assert loaded.ncclGetVersion(ctypes.byref(version))==0 and version.value==23102
exports=subprocess.check_output(['nm','-D','--defined-only',str(library)],text=True)
names=sorted({line.split()[-1] for line in exports.splitlines() if line.split()})
receipt={'schema':'sparkring-r33-arm64-nccl-build/v1','status':'built-not-gpu-qualified',
 'source_commit':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip(),
 'source_tree':subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD^{tree}'],text=True).strip(),
 'version':version.value,'platform':'linux/arm64','cuda_architectures':['sm_121','compute_121'],
 'library_sha256':hashlib.sha256(library.read_bytes()).hexdigest(),
 'header_sha256':hashlib.sha256((prefix/'include/nccl.h').read_bytes()).hexdigest(),
 'exported_symbols':names,'nvcc':subprocess.check_output(['nvcc','--version'],text=True),
 'gpu_devices_supplied':False}
(prefix/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='exported_symbols'}))
PY
