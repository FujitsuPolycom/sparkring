"""Verify the ARM64 R33 Torch foundation without initializing a CUDA device."""

import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform

import torch

EXPECTED_SOURCE = "cf30153c4c131c8164ee7798e5022d810682e2cb"
assert platform.machine() == "aarch64", platform.machine()
assert torch.__version__ == "2.13.0", torch.__version__
assert torch.version.git_version == EXPECTED_SOURCE, torch.version.git_version
assert torch.version.cuda == "13.3", torch.version.cuda
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
assert not torch.cuda.is_initialized()

library = Path("/opt/local-inference/nccl/lib/libnccl.so.2").resolve(strict=True)
nccl = ctypes.CDLL(str(library))
version = ctypes.c_int()
nccl.ncclGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
nccl.ncclGetVersion.restype = ctypes.c_int
assert nccl.ncclGetVersion(ctypes.byref(version)) == 0
assert version.value == 23102, version.value
mapped = {
    Path(line.split()[-1]).resolve()
    for line in Path("/proc/self/maps").read_text().splitlines()
    if "/libnccl.so" in line
}
assert mapped == {library}, sorted(map(str, mapped))
assert not torch.cuda.is_initialized()
with library.open("rb") as handle:
    nccl_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
receipt = {
    "status": "qualified-cpu-import-only",
    "torch_version": torch.__version__,
    "torch_source": torch.version.git_version,
    "torch_origin": torch.__file__,
    "cuda_toolkit": torch.version.cuda,
    "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    "nccl_version": version.value,
    "nccl_library": str(library),
    "nccl_sha256": nccl_sha256,
    "cuda_initialized": torch.cuda.is_initialized(),
    "platform": platform.machine(),
}
print(json.dumps(receipt, indent=2))
if output := os.environ.get("SPARKRING_TORCH_RECEIPT"):
    Path(output).write_text(json.dumps(receipt, indent=2) + "\n")
