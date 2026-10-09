#!/usr/bin/env python3
"""CPU contract tests for the bounded library milestone; no GPU is opened."""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import unittest

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "tests/api_manifest.json").read_text())
LIBRARY = ROOT / "build/libsircl.so"
LIB = None
SUCCESS, INVALID_ARGUMENT, INVALID_USAGE = 0, 4, 5


class UniqueId(C.Structure):
    _fields_ = [("internal", C.c_ubyte * 128)]


class Config(C.Structure):
    _fields_ = [
        ("size", C.c_size_t), ("magic", C.c_uint), ("version", C.c_uint),
        ("blocking", C.c_int), ("cgaClusterSize", C.c_int), ("minCTAs", C.c_int), ("maxCTAs", C.c_int),
        ("netName", C.c_char_p), ("splitShare", C.c_int), ("trafficClass", C.c_int), ("commName", C.c_char_p),
        *[(name, C.c_int) for name in ("collnetEnable", "CTAPolicy", "shrinkShare", "nvlsCTAs", "nChannelsPerNetPeer", "nvlinkCentricSched", "graphUsageMode", "numRmaCtx", "maxP2pPeers", "graphStreamOrdering", "launchOrderImplicit", "numRmaSig", "rmaEagerInit", "hostCftMode", "nvlsHostMode")],
    ]


def config(size: int = 120) -> Config:
    value = Config()
    for name, kind in Config._fields_:
        if kind is C.c_int:
            setattr(value, name, -2147483648)
    value.size, value.magic, value.version = size, 0xcafebeef, 23203
    return value


def initialize_library() -> None:
    global LIB
    LIB = C.CDLL(str(LIBRARY))
    signatures = {
        "GetVersion": ([C.POINTER(C.c_int)], C.c_int),
        "GetUniqueId": ([C.POINTER(UniqueId)], C.c_int),
        "GetErrorString": ([C.c_int], C.c_char_p),
        "GetLastError": ([C.c_void_p], C.c_char_p),
        "CommInitRank": ([C.POINTER(C.c_void_p), C.c_int, UniqueId, C.c_int], C.c_int),
        "CommInitRankConfig": ([C.POINTER(C.c_void_p), C.c_int, UniqueId, C.c_int, C.POINTER(Config)], C.c_int),
        "CommInitAll": ([C.POINTER(C.c_void_p), C.c_int, C.POINTER(C.c_int)], C.c_int),
        "CommRevoke": ([C.c_void_p, C.c_int], C.c_int),
        "CommGetAsyncError": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
        "CommCount": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
        "CommUserRank": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
        "CommCuDevice": ([C.c_void_p, C.POINTER(C.c_int)], C.c_int),
        **{name: ([C.c_void_p], C.c_int) for name in ("CommFinalize", "CommDestroy", "CommAbort")},
        "GroupStart": ([], C.c_int), "GroupEnd": ([], C.c_int),
    }
    for name, (args, result) in signatures.items():
        for prefix in ("nccl", "pnccl"):
            function = getattr(LIB, prefix + name)
            function.argtypes, function.restype = args, result


def new_comm(value: Config | None = None) -> C.c_void_p:
    uid = UniqueId()
    assert LIB.ncclGetUniqueId(C.byref(uid)) == SUCCESS
    handle = C.c_void_p()
    if value is None:
        result = LIB.ncclCommInitRank(C.byref(handle), 1, uid, 0)
    else:
        result = LIB.ncclCommInitRankConfig(C.byref(handle), 1, uid, 0, C.byref(value))
    assert result == SUCCESS, (result, LIB.ncclGetLastError(None))
    return handle


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["LIBSIRCL_BOOTSTRAP_ONLY"] = "1"
        os.environ.pop("LIBSIRCL_NCCL_API_VERSION", None)
        initialize_library()

    def test_version_and_twin(self):
        for prefix in ("nccl", "pnccl"):
            value = C.c_int(-1)
            self.assertEqual(getattr(LIB, prefix + "GetVersion")(C.byref(value)), SUCCESS)
            self.assertEqual(value.value, 22705)
            self.assertEqual(getattr(LIB, prefix + "GetVersion")(None), INVALID_ARGUMENT)

    def test_unique_ids_and_null_check(self):
        values = []
        for _ in range(32):
            uid = UniqueId()
            self.assertEqual(LIB.ncclGetUniqueId(C.byref(uid)), SUCCESS)
            values.append(bytes(uid))
        self.assertEqual(len(set(values)), len(values))
        self.assertEqual(C.sizeof(UniqueId), 128)
        self.assertEqual(LIB.ncclGetUniqueId(None), INVALID_ARGUMENT)

    def test_runtime_version_override_and_validation(self):
        script = '''import ctypes as C, sys
l=C.CDLL(sys.argv[1]); l.ncclGetVersion.argtypes=[C.POINTER(C.c_int)]
v=C.c_int(-1); result=l.ncclGetVersion(C.byref(v))
assert result==int(sys.argv[2]), (result, v.value)
assert v.value==int(sys.argv[3]), (result, v.value)
'''
        for override, result, value in [("29901", SUCCESS, 29901), ("bad", INVALID_ARGUMENT, -1), ("0", INVALID_ARGUMENT, -1)]:
            env = os.environ.copy()
            env["LIBSIRCL_NCCL_API_VERSION"] = override
            subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(result), str(value)], env=env, check=True, capture_output=True, text=True)

    def test_error_strings_cover_every_result(self):
        descriptions = [LIB.ncclGetErrorString(result) for result in range(9)]
        self.assertTrue(all(descriptions))
        self.assertEqual(len(set(descriptions)), 9)
        for result in [-1, 9, 2147483647]:
            self.assertTrue(LIB.ncclGetErrorString(result))
        for result in range(9):
            self.assertEqual(LIB.pncclGetErrorString(result), descriptions[result])

    def test_bootstrap_only_requires_explicit_opt_in(self):
        script = '''import ctypes as C, sys
class U(C.Structure): _fields_=[("b",C.c_ubyte*128)]
l=C.CDLL(sys.argv[1]); l.ncclCommInitRank.argtypes=[C.POINTER(C.c_void_p),C.c_int,U,C.c_int]
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==5
assert c.value is None
'''
        env = os.environ.copy()
        env.pop("LIBSIRCL_BOOTSTRAP_ONLY", None)
        subprocess.run([sys.executable, "-c", script, str(LIBRARY)], env=env, check=True, capture_output=True, text=True)

    @unittest.skipUnless(hasattr(os, "fork"), "Linux fork containment regression")
    def test_fork_child_refuses_bootstrap_and_exit_preserves_parent(self):
        script = '''import ctypes as C, os, sys
class U(C.Structure): _fields_=[("b",C.c_ubyte*128)]
l=C.CDLL(sys.argv[1]); l.ncclGetUniqueId.argtypes=[C.POINTER(U)]
l.ncclCommInitRank.argtypes=[C.POINTER(C.c_void_p),C.c_int,U,C.c_int]
l.ncclCommDestroy.argtypes=[C.c_void_p]
uid=U(); assert l.ncclGetUniqueId(C.byref(uid))==0
pid=os.fork()
if pid==0:
    child_uid=U(); assert l.ncclGetUniqueId(C.byref(child_uid))==5
    child_comm=C.c_void_p(); assert l.ncclCommInitRank(C.byref(child_comm),1,uid,0)==5
    assert child_comm.value is None
    # libc.exit invokes DSO destructors, unlike os._exit. The inherited wake
    # pipe must not be written, because it would stop the parent's broker.
    libc=C.CDLL(None); libc.exit.argtypes=[C.c_int]; libc.exit(0)
_, status=os.waitpid(pid,0); assert status==0, status
comm=C.c_void_p(); assert l.ncclCommInitRank(C.byref(comm),1,uid,0)==0
assert l.ncclCommDestroy(comm)==0
'''
        env = os.environ.copy()
        env["SIRCL_BOOTSTRAP_TIMEOUT_MS"] = "1000"
        subprocess.run([sys.executable, "-c", script, str(LIBRARY)], env=env, check=True, capture_output=True, text=True, timeout=5)

    def test_current_and_legacy_config_bounds(self):
        self.assertEqual(C.sizeof(Config), 120)
        for size in [72, 120]:
            value = config(size)
            handle = new_comm(value)
            self.assertEqual(LIB.ncclCommDestroy(handle), SUCCESS)
        # Real 72-byte storage: a guard page catches reads beyond the supplied size.
        script = '''import ctypes as C, mmap, os, sys
sys.path.insert(0,sys.argv[2]); import test_api as t
t.LIBRARY=t.Path(sys.argv[1]); t.initialize_library()
p=mmap.PAGESIZE; m=mmap.mmap(-1,p*2); base=C.addressof(C.c_char.from_buffer(m))
libc=C.CDLL(None); libc.mprotect.argtypes=[C.c_void_p,C.c_size_t,C.c_int]
assert libc.mprotect(base+p,p,0)==0
v=t.config(72); C.memmove(base+p-72,C.byref(v),72)
u=t.UniqueId(); assert t.LIB.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); ptr=C.cast(base+p-72,C.POINTER(t.Config))
assert t.LIB.ncclCommInitRankConfig(C.byref(c),1,u,0,ptr)==0
assert t.LIB.ncclCommDestroy(c)==0
assert libc.mprotect(base+p,p,3)==0
'''
        subprocess.run([sys.executable, "-c", script, str(LIBRARY), str(ROOT / "tests")], check=True, capture_output=True, text=True)

    def test_invalid_configs_and_init_arguments(self):
        uid = UniqueId()
        self.assertEqual(LIB.ncclGetUniqueId(C.byref(uid)), SUCCESS)
        for ranks, rank in [(0, 0), (-1, 0), (1, -1), (1, 1)]:
            handle = C.c_void_p()
            self.assertEqual(LIB.ncclCommInitRank(C.byref(handle), ranks, uid, rank), INVALID_ARGUMENT)
            self.assertIsNone(handle.value)
        self.assertEqual(LIB.ncclCommInitRank(None, 1, uid, 0), INVALID_ARGUMENT)
        for invalid in [config(0), config(8), config(15)]:
            handle = C.c_void_p()
            self.assertEqual(LIB.ncclCommInitRankConfig(C.byref(handle), 1, uid, 0, C.byref(invalid)), INVALID_ARGUMENT)
        invalid = config()
        invalid.magic = 0
        handle = C.c_void_p()
        self.assertEqual(LIB.ncclCommInitRankConfig(C.byref(handle), 1, uid, 0, C.byref(invalid)), INVALID_ARGUMENT)

    def test_metadata_finalize_destroy_and_stale_handle(self):
        handle = new_comm()
        for name, expected in [("CommCount", 1), ("CommUserRank", 0), ("CommCuDevice", 0), ("CommGetAsyncError", SUCCESS)]:
            output = C.c_int(-1)
            self.assertEqual(getattr(LIB, "nccl" + name)(handle, C.byref(output)), SUCCESS)
            self.assertEqual(output.value, expected)
            self.assertEqual(getattr(LIB, "nccl" + name)(handle, None), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclCommFinalize(handle), SUCCESS)
        self.assertEqual(LIB.ncclCommDestroy(handle), SUCCESS)
        self.assertEqual(LIB.ncclCommDestroy(handle), INVALID_ARGUMENT)
        output = C.c_int(-1)
        self.assertEqual(LIB.ncclCommCount(handle, C.byref(output)), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclCommCount(C.c_void_p(1), C.byref(output)), INVALID_ARGUMENT)

    def test_init_all_preserves_logical_device_and_rank_order(self):
        handles = (C.c_void_p * 3)()
        devices = (C.c_int * 3)(4, 2, 9)
        self.assertEqual(LIB.ncclCommInitAll(handles, 3, devices), SUCCESS)
        try:
            for rank, device in enumerate(devices):
                for name, expected in [("CommCount", 3), ("CommUserRank", rank), ("CommCuDevice", device)]:
                    value = C.c_int(-1)
                    self.assertEqual(getattr(LIB, "nccl" + name)(handles[rank], C.byref(value)), SUCCESS)
                    self.assertEqual(value.value, expected)
        finally:
            for handle in handles:
                self.assertEqual(LIB.ncclCommDestroy(handle), SUCCESS)
        self.assertEqual(LIB.ncclCommInitAll(None, 1, None), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclCommInitAll(handles, 0, None), INVALID_ARGUMENT)
        devices[1] = -1
        self.assertEqual(LIB.ncclCommInitAll(handles, 3, devices), INVALID_ARGUMENT)
        self.assertEqual(list(handles), [None, None, None])

    def test_revoke_then_finalize_and_abort(self):
        handle = new_comm()
        self.assertEqual(LIB.ncclCommRevoke(handle, 1), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclCommRevoke(handle, 0), SUCCESS)
        self.assertEqual(LIB.ncclCommFinalize(handle), INVALID_USAGE)
        self.assertEqual(LIB.ncclCommAbort(handle), SUCCESS)
        self.assertEqual(LIB.ncclCommAbort(handle), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclCommAbort(None), INVALID_ARGUMENT)

    def test_thread_local_nested_group_and_error(self):
        barrier = threading.Barrier(2)
        def grouped():
            self.assertEqual(LIB.ncclGroupStart(), SUCCESS)
            self.assertEqual(LIB.ncclGroupStart(), SUCCESS)
            barrier.wait(timeout=5)
            barrier.wait(timeout=5)
            self.assertEqual(LIB.ncclGroupEnd(), SUCCESS)
            self.assertEqual(LIB.ncclGroupEnd(), SUCCESS)
        def separate():
            barrier.wait(timeout=5)
            self.assertEqual(LIB.ncclGroupEnd(), INVALID_USAGE)
            barrier.wait(timeout=5)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            tasks = [pool.submit(grouped), pool.submit(separate)]
            for task in tasks:
                task.result(timeout=10)
        self.assertEqual(LIB.ncclGroupStart(), SUCCESS)
        f = LIB.ncclCommGetUniqueId
        f.argtypes = [C.c_void_p, C.c_void_p]
        self.assertEqual(f(None, None), INVALID_USAGE)
        self.assertEqual(LIB.ncclGroupEnd(), INVALID_USAGE)
        self.assertEqual(LIB.ncclGroupStart(), SUCCESS)
        self.assertEqual(LIB.ncclGroupEnd(), SUCCESS)

    def test_unsupported_surface_preserves_every_output(self):
        sentinel = C.create_string_buffer(b"Q" * 256)
        before = bytes(sentinel)
        for api in MANIFEST["functions"]:
            # ncclRedOpCreatePreMulSum writes its op (test_premulsum_refusal_returns_a_refused_op).
            if api["implemented"] or api["name"] == "ncclRedOpCreatePreMulSum":
                continue
            for prefix in ["", "p"]:
                with self.subTest(function=prefix + api["name"]):
                    args, types = [], []
                    for argument in api["arguments"]:
                        kind = argument["type"].removeprefix("const ")
                        if "*" in kind or kind in ("ncclComm_t", "ncclWindow_t", "ncclParamHandle_t", "cudaStream_t"):
                            types.append(C.c_void_p)
                            args.append(C.cast(sentinel, C.c_void_p))
                        elif kind == "size_t":
                            types.append(C.c_size_t)
                            args.append(1)
                        elif kind == "unsigned int":
                            types.append(C.c_uint)
                            args.append(0)
                        else:
                            types.append(C.c_int)
                            args.append(0)
                    function = getattr(LIB, prefix + api["name"])
                    function.argtypes = types
                    function.restype = None if api["return"] == "void" else C.c_int
                    result = function(*args)
                    self.assertEqual(result, None if api["return"] == "void" else INVALID_USAGE)
                    self.assertEqual(bytes(sentinel), before)
                    diagnostic = LIB.ncclGetLastError(None)
                    self.assertTrue(diagnostic and api["name"].encode() in diagnostic)

    def test_premulsum_refusal_returns_a_refused_op(self):
        # A caller that ignores the refusal (a handle initialized to 0, ncclSum) must not get a plain sum:
        # the op written is ncclNumOps (5), which every collective refuses with ncclInvalidArgument.
        for prefix in ("", "p"):
            function = getattr(LIB, prefix + "ncclRedOpCreatePreMulSum")
            function.argtypes = [C.POINTER(C.c_int), C.c_void_p, C.c_int, C.c_int, C.c_void_p]
            function.restype = C.c_int
            op, scalar = C.c_int(0), C.c_float(0.5)
            self.assertEqual(function(C.byref(op), C.byref(scalar), 7, 1, None), INVALID_USAGE)
            self.assertEqual(op.value, 5)
            self.assertEqual(function(None, C.byref(scalar), 7, 1, None), INVALID_USAGE)

    def test_invalid_argument_inside_a_group_leaves_the_group(self):
        # A refused call did nothing: ncclGroupEnd still succeeds (an unsupported call, ncclInvalidUsage,
        # remains the group's result; test_thread_local_nested_group_and_error).
        LIB.ncclAllReduce.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_void_p,
                                      C.c_void_p]
        LIB.ncclSend.argtypes = [C.c_void_p, C.c_size_t, C.c_int, C.c_int, C.c_void_p, C.c_void_p]
        stale = C.c_void_p(0x1234)
        self.assertEqual(LIB.ncclGroupStart(), SUCCESS)
        self.assertEqual(LIB.ncclAllReduce(None, None, 1, 7, 0, stale, None), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclSend(None, 1, 7, 0, stale, None), INVALID_ARGUMENT)
        self.assertEqual(LIB.ncclGroupEnd(), SUCCESS)
        self.assertIn(b"unknown communicator", LIB.ncclGetLastError(None))

    def test_unsupported_diagnostic_is_logged_once_per_function(self):
        script = '''import ctypes as C, sys
l=C.CDLL(sys.argv[1]); l.ncclCommGetUniqueId.argtypes=[C.c_void_p,C.c_void_p]
l.pncclCommGetUniqueId.argtypes=l.ncclCommGetUniqueId.argtypes
for f in [l.ncclCommGetUniqueId,l.pncclCommGetUniqueId,l.ncclCommGetUniqueId]: assert f(None,None)==5
'''
        output = subprocess.run([sys.executable, "-c", script, str(LIBRARY)], check=True, capture_output=True, text=True)
        self.assertEqual(output.stderr.count("ncclCommGetUniqueId unsupported"), 1)

    def test_registration_hints_and_memory_without_cuda(self):
        """Registration of buffers and windows is accepted on a CPU-only communicator (the handle and the
        window are the buffer's address); ncclMemAlloc without a current CUDA context names the reason."""
        script = '''import ctypes as C, sys, os
class U(C.Structure): _fields_=[("b",C.c_ubyte*128)]
l=C.CDLL(sys.argv[1]); l.ncclGetLastError.restype=C.c_char_p; l.ncclGetLastError.argtypes=[C.c_void_p]
l.ncclCommInitRank.argtypes=[C.POINTER(C.c_void_p),C.c_int,U,C.c_int]
for name in ("ncclCommRegister","ncclCommWindowRegister"): getattr(l,name).restype=C.c_int
l.ncclCommRegister.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_void_p)]
l.ncclCommDeregister.argtypes=[C.c_void_p,C.c_void_p]
l.ncclCommWindowRegister.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.POINTER(C.c_void_p),C.c_int]
l.ncclCommWindowDeregister.argtypes=[C.c_void_p,C.c_void_p]
l.ncclWinGetUserPtr.argtypes=[C.c_void_p,C.c_void_p,C.POINTER(C.c_void_p)]
l.ncclMemAlloc.argtypes=[C.POINTER(C.c_void_p),C.c_size_t]
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==0
h=C.c_void_p(); w=C.c_void_p(); p=C.c_void_p()
codes=[l.ncclCommRegister(c,0x1000,64,C.byref(h)), l.ncclCommWindowRegister(c,0x2000,64,C.byref(w),1),
       l.ncclWinGetUserPtr(c,w,C.byref(p)), l.ncclCommDeregister(c,h), l.ncclCommWindowDeregister(c,w),
       l.ncclCommRegister(None,0x1000,64,C.byref(h))]
m=C.c_void_p(); alloc=l.ncclMemAlloc(C.byref(m),1024)
print(codes, h.value, w.value, p.value, alloc, l.ncclGetLastError(None).decode())
assert l.ncclCommDestroy(c)==0
'''
        environment = dict(os.environ, LIBSIRCL_BOOTSTRAP_ONLY="1")
        output = subprocess.run([sys.executable, "-c", script, str(LIBRARY)], capture_output=True, text=True,
                                env=environment)
        self.assertEqual(output.returncode, 0, output.stderr)
        self.assertIn("[0, 0, 0, 0, 0, 4] 4096 8192 8192", output.stdout)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=LIBRARY)
    options, extra = parser.parse_known_args()
    LIBRARY = options.library.resolve()
    unittest.main(argv=[sys.argv[0]] + extra)
