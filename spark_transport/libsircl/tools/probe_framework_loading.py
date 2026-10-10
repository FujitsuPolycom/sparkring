#!/usr/bin/env python3
"""Qualify library loading in installed frameworks without creating a CUDA context.

This is a host loading check, not distributed/GPU/framework execution qualification.
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
from pathlib import Path
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--expected-version", type=int, default=22705)
    parser.add_argument("--skip-torch", action="store_true")
    parser.add_argument("--skip-vllm", action="store_true")
    options = parser.parse_args()
    library = options.library.resolve()
    if not library.is_file():
        parser.error(f"Library does not exist: {library}")
    if os.environ.get("_SIRCL_LOADING_PROBE_CHILD") != str(library):
        env = os.environ.copy()
        env["_SIRCL_LOADING_PROBE_CHILD"] = str(library)
        env["LD_PRELOAD"] = str(library)
        env["CUDA_VISIBLE_DEVICES"] = ""
        os.execve(sys.executable, [sys.executable, *sys.argv], env)
    expected = options.expected_version
    handle = C.CDLL("libnccl.so.2")
    get = handle.ncclGetVersion
    get.argtypes, get.restype = [C.POINTER(C.c_int)], C.c_int
    value = C.c_int(-1)
    assert get(C.byref(value)) == 0 and value.value == expected, (value.value, expected)
    report = {"scope": "host loading only; no CUDA context, ranks, or collectives",
              "library": str(library), "ctypes_soname_version": value.value}
    torch = None
    if not options.skip_torch or not options.skip_vllm:
        import torch
        assert not torch.cuda.is_initialized(), "A CUDA context already exists"
        def forbid_cuda_initialization(*args, **kwargs):
            raise AssertionError("GPU initialization is forbidden in this host loading probe")
        torch.cuda._lazy_init = forbid_cuda_initialization
    if not options.skip_torch:
        import torch.distributed
        assert torch.distributed.is_nccl_available(), "Installed torch has no ProcessGroupNCCL"
        # torch.cuda.nccl.version() reports build-header macros, not runtime:
        # https://github.com/pytorch/pytorch/blob/v2.10.0/torch/csrc/cuda/nccl.cpp
        version = torch.cuda.nccl.version()
        assert hasattr(torch.distributed, "ProcessGroupNCCL")
        report["torch"] = {"package_version": torch.__version__, "build_header_nccl_version": version,
                           "ProcessGroupNCCL_available": True}
        # The installed ELF exposes this ordinary no-argument C++ function.
        # c10d uses ncclGetVersion, unlike the compile-time Python version query.
        cuda_library = C.CDLL(str(Path(torch.__file__).parent / "lib/libtorch_cuda.so"))
        try:
            runtime_version = getattr(cuda_library, "_ZN4c10d20getNcclVersionNumberEv")
        except AttributeError:
            report["torch"]["c10d_runtime_version"] = "symbol not exposed in this build"
        else:
            runtime_version.argtypes, runtime_version.restype = [], C.c_int
            actual = runtime_version()
            assert actual == expected, (actual, expected)
            report["torch"]["c10d_runtime_version"] = actual
        uid = bytes(value & 255 for value in torch.cuda.nccl.unique_id())
        assert len(uid) == 128 and uid[:8] == b"SCCLBO01", uid[:8]
        report["torch"]["sircl_unique_id_magic"] = uid[:8].decode("ascii")
    if not options.skip_vllm:
        from vllm.distributed.device_communicators.pynccl_wrapper import NCCLLibrary
        wrapper = NCCLLibrary()
        assert wrapper.ncclGetRawVersion() == expected
        uid = wrapper.ncclGetUniqueId()
        assert C.sizeof(uid) == 128
        report["vllm"] = {"nccl_version": wrapper.ncclGetRawVersion(),
                          "loaded_functions": len(wrapper._funcs), "unique_id_size": C.sizeof(uid)}
    if torch is not None:
        assert not torch.cuda.is_initialized(), "Probe unexpectedly initialized CUDA"
        report["cuda_initialized"] = False
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
