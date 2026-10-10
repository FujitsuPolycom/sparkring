#!/usr/bin/env python3
"""Communicator creation on a thread with no current CUDA context, as vLLM's PyNccl creates its first one.

vLLM's GPU worker selects its device with ``torch.accelerator.set_device_index`` and creates the first PyNccl
communicator before any CUDA allocation: ``ncclCommInitRank`` runs inside ``torch.accelerator.device_index``
and the first tensor is the warm-up all-reduce after it. torch's set_device skips ``cudaSetDevice`` when the
device is already current, so no context exists at the call. libsircl then runs the communicator on the
current device's primary context, as NVIDIA NCCL does.

This check does the same on one rank (``LIBSIRCL_TRANSPORT=emulation``): the calling thread must have no
current context before the call and the primary context after it, torch's first allocation must land in that
same context, and the warm-up (a one-element float32 sum on torch's current stream) must return 1.0. Prints
one line per step and exits 0 when every step holds.

    LIBSIRCL_TRANSPORT=emulation python3 tests/emulation/init_without_context.py --library build/libsircl.so
"""
import argparse
import ctypes
import sys

import torch


class UniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    cuda = ctypes.CDLL("libcuda.so.1")
    cuda.cuInit(0)

    def current():
        ctx = ctypes.c_void_p()
        cuda.cuCtxGetCurrent(ctypes.byref(ctx))
        return ctx.value

    lib = ctypes.CDLL(args.library)
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]

    torch.accelerator.set_device_index(torch.device("cuda:0"))
    uid = UniqueId()
    if lib.ncclGetUniqueId(ctypes.byref(uid)):
        print("ncclGetUniqueId failed")
        return 1
    comm = ctypes.c_void_p()
    with torch.accelerator.device_index(0):
        before = current()
        result = lib.ncclCommInitRank(ctypes.byref(comm), 1, uid, 0)
        after = current()
        print(f"context before ncclCommInitRank: {'none' if not before else 'present'}; result {result} "
              f"({lib.ncclGetErrorString(result).decode()}); context after: {'none' if not after else 'present'}")
        if result:
            print(f"ncclGetLastError: {lib.ncclGetLastError(None).decode()}")
            return 1
        data = torch.ones(1, device="cuda:0")
        torch_context = current()
        stream = torch.cuda.current_stream()
        reduced = lib.ncclAllReduce(ctypes.c_void_p(data.data_ptr()), ctypes.c_void_p(data.data_ptr()), 1, 7, 0,
                                    comm, ctypes.c_void_p(stream.cuda_stream))
        stream.synchronize()
    destroyed = lib.ncclCommDestroy(comm)
    same = torch_context == after
    value = data.item()
    print(f"torch's first allocation in libsircl's context: {same}; warm-up all-reduce {reduced}, value {value}; "
          f"ncclCommDestroy {destroyed}")
    ok = not before and after and same and reduced == 0 and value == 1.0 and destroyed == 0
    print("init without a context: " + ("every step holds" if ok else "a step failed"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
