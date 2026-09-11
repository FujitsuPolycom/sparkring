"""Load the candidate's required ARM native modules on one GB10."""

import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path

import torch


assert torch.cuda.is_available()
assert torch.cuda.get_device_capability(0) == (12, 1)
modules = [
    "vllm",
    "vllm._C_stable_libtorch",
    "vllm._moe_C_stable_libtorch",
    "vllm._qutlass_C",
    "vllm._flashkda_C",
    "vllm.vllm_flash_attn._vllm_fa2_C",
    "vllm.vllm_flash_attn._vllm_fa3_C",
    "vllm.vllm_flash_attn.layers.rotary",
    "vllm.vllm_flash_attn.ops.triton.rotary",
    "vllm._rust_tool_parser",
    "b12x",
    "flashinfer",
    "flashinfer_jit_cache",
    "lmcache",
    "lmcache.cuda_ops",
    "lmcache.lmcache_native",
    "lmcache.lmcache_fs",
    "lmcache.lmcache_redis",
    "instanttensor",
    "xgrammar",
    "sparkcache",
]
origins = {}
for name in modules:
    module = importlib.import_module(name)
    origins[name] = getattr(module, "__file__", None)

libraries = {
    "nccl": Path("/opt/local-inference/nccl/lib/libnccl.so.2.31.2"),
    "sircl": Path("/opt/sparkring/sircl/libspark_transport_capi.so"),
    "sparkcache_placement": Path(
        "/opt/sparkring/sparkcache/lib/libspark_cache_placement.so"
    ),
    "sparkcache_snapshot": Path(
        "/opt/sparkring/sparkcache/lib/libspark_cache_snapshot.so"
    ),
    "lmcache_cumem": Path("/opt/lmcache/lib/liblmcache_cumem_shareable.so"),
}
library_hashes = {}
for name, path in libraries.items():
    ctypes.CDLL(str(path), mode=ctypes.RTLD_LOCAL)
    with path.open("rb") as stream:
        library_hashes[name] = hashlib.file_digest(stream, "sha256").hexdigest()

value = torch.arange(1024, device="cuda", dtype=torch.bfloat16)
assert torch.equal((value + 1).cpu(), torch.arange(1024, dtype=torch.bfloat16) + 1)
receipt = {
            "status": "qualified-single-gpu-native-import",
            "device": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "versions": {
                name: importlib.metadata.version(name)
                for name in (
                    "vllm",
                    "b12x",
                    "flashinfer-python",
                    "flashinfer-jit-cache",
                    "lmcache",
                    "instanttensor",
                    "xgrammar",
                    "sparkcache",
                )
            },
            "origins": origins,
            "library_sha256": library_hashes,
            "limits": "One GB10 import and elementary CUDA operation; no distributed collective or model execution.",
}
rendered = json.dumps(receipt, indent=2)
if output := os.environ.get("SPARKRING_GPU_IMPORT_RECEIPT"):
    Path(output).write_text(rendered + "\n")
print(rendered)
