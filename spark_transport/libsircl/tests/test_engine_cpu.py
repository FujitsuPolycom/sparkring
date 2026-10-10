#!/usr/bin/env python3
"""CPU checks of the collective entry points and the engine's refusals. No GPU or RDMA device is used.

Each check runs in a fresh process: the library must load and answer without a CUDA context, create a
communicator on a thread without one on the current device's primary context (and refuse, naming the
reason, when that device is unknown), and refuse collectives on CPU-only test communicators and unknown
handles with NCCL's result codes. The checks of communicator creation run on the stand-in CUDA driver
(tests/fake_cuda.c, build/fake-cuda/libcuda.so.1), so they do not depend on the host's GPU.
"""
import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[1] / "build" / "libsircl.so"
FAKE_DRIVER = LIBRARY.parent / "fake-cuda"
PRELUDE = '''import ctypes as C, json, sys
class U(C.Structure): _fields_=[("b",C.c_ubyte*128)]
l=C.CDLL(sys.argv[1])
l.ncclGetLastError.restype=C.c_char_p; l.ncclGetLastError.argtypes=[C.c_void_p]
l.sirclGetInfo.restype=C.c_char_p
l.ncclCommInitRank.argtypes=[C.POINTER(C.c_void_p),C.c_int,U,C.c_int]
l.ncclAllReduce.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
l.ncclAllGather.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_void_p,C.c_void_p]
l.ncclReduceScatter.argtypes=[C.c_void_p,C.c_void_p,C.c_size_t,C.c_int,C.c_int,C.c_void_p,C.c_void_p]
l.sirclGetReceipt.argtypes=[C.c_void_p,C.c_char_p,C.c_size_t,C.POINTER(C.c_size_t)]
l.sirclSetWaitRegime.argtypes=[C.c_void_p,C.c_char_p]
l.ncclCommDestroy.argtypes=[C.c_void_p]
'''


def run(script, env=None):
    environment = dict(os.environ)
    environment.update(env or {})
    return subprocess.run([sys.executable, "-c", PRELUDE + script, str(LIBRARY)], env=environment,
                          capture_output=True, text=True, timeout=60)


class EngineCpuTests(unittest.TestCase):
    def test_info_names_the_collectives(self):
        out = run("print(l.sirclGetInfo().decode())")
        self.assertEqual(out.returncode, 0, out.stderr)
        info = json.loads(out.stdout)
        self.assertIn("all_reduce", info["collectives"])
        self.assertIn("reduce_scatter", info["collectives"])
        self.assertIn("all_gather", info["collectives"])
        self.assertEqual(info["transports"], ["verbs", "emulation"])

    def test_device_api_host_entry_points(self):
        """ncclCommQueryProperties reports the communicator and no device API; the teams are the world,
        the rank alone (LSA) and the world again (rail); ncclDevCommCreate and ncclDevCommDestroy refuse."""
        out = run('''class Props(C.Structure):
    _fields_ = [("size", C.c_size_t), ("magic", C.c_uint), ("version", C.c_uint), ("rank", C.c_int),
                ("nRanks", C.c_int), ("cudaDev", C.c_int), ("nvmlDev", C.c_int), ("deviceApiSupport", C.c_bool),
                ("multimemSupport", C.c_bool), ("ginType", C.c_int), ("nLsaTeams", C.c_int),
                ("hostRmaSupport", C.c_bool), ("railedGinType", C.c_int)]
class Team(C.Structure): _fields_ = [("nRanks", C.c_int), ("rank", C.c_int), ("stride", C.c_int)]
for name in ("ncclTeamWorld", "ncclTeamLsa", "ncclTeamRail", "pncclTeamLsa"):
    getattr(l, name).restype = Team; getattr(l, name).argtypes = [C.c_void_p]
l.ncclCommQueryProperties.argtypes = [C.c_void_p, C.POINTER(Props)]
l.ncclDevCommCreate.argtypes = [C.c_void_p, C.c_void_p, C.c_void_p]
l.ncclDevCommDestroy.argtypes = [C.c_void_p, C.c_void_p]
l.ncclTeamRankToWorld.argtypes = [C.c_void_p, Team, C.c_int]
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==0
p = Props(size=C.sizeof(Props), magic=0xcafebeef, version=23203)
r = l.ncclCommQueryProperties(c, C.byref(p))
bad = Props(size=C.sizeof(Props), magic=0, version=23203)
teams = [(t.nRanks, t.rank, t.stride) for t in (l.ncclTeamWorld(c), l.ncclTeamLsa(c), l.ncclTeamRail(c), l.pncclTeamLsa(c))]
print(json.dumps({"query": r, "bad": l.ncclCommQueryProperties(c, C.byref(bad)), "rank": p.rank, "nRanks": p.nRanks,
                  "device": p.deviceApiSupport, "lsa": p.nLsaTeams, "rma": p.hostRmaSupport, "teams": teams,
                  "world0": l.ncclTeamRankToWorld(c, l.ncclTeamWorld(c), 0),
                  "create": l.ncclDevCommCreate(c, None, None), "destroy": l.ncclDevCommDestroy(c, None)}))
assert l.ncclCommDestroy(c)==0''', {"LIBSIRCL_BOOTSTRAP_ONLY": "1"})
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout.strip().splitlines()[-1])
        self.assertEqual((data["query"], data["bad"], data["rank"], data["nRanks"]), (0, 4, 0, 1))
        self.assertEqual((data["device"], data["lsa"], data["rma"]), (False, 1, False))
        self.assertEqual(data["teams"], [[1, 0, 1], [1, 0, 1], [1, 0, 1], [1, 0, 1]])
        self.assertEqual((data["world0"], data["create"], data["destroy"]), (0, 5, 5))

    def test_earlier_environment_names_and_identifying_line(self):
        """SIRCL_CCL_* names are read when the LIBSIRCL_* name is unset (until release 0.7.0), and under
        NCCL_DEBUG the first communicator creation writes one line naming libsircl and the API level."""
        script = '''u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); r=l.ncclCommInitRank(C.byref(c),1,u,0)
v=U(); assert l.ncclGetUniqueId(C.byref(v))==0
d=C.c_void_p(); r2=l.ncclCommInitRank(C.byref(d),1,v,0)
print(r, r2)'''
        out = run(script, {"SIRCL_CCL_BOOTSTRAP_ONLY": "1", "NCCL_DEBUG": "INFO", "SIRCL_NCCLAPI_VERSION_CODE": "22801"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["0", "0"])
        lines = [line for line in out.stderr.splitlines() if line.startswith("libsircl ")]
        self.assertEqual(len(lines), 1, out.stderr)
        self.assertIn("(SIRCL's NCCL-compatible C API; not NVIDIA NCCL), NCCL API level 22801", lines[0])
        quiet = run(script, {"LIBSIRCL_BOOTSTRAP_ONLY": "1"})
        self.assertEqual(quiet.returncode, 0, quiet.stderr)
        self.assertNotIn("libsircl ", quiet.stderr)

    def creation(self, env, *, runtime=False):
        """One ncclCommInitRank of a one-rank communicator on the stand-in driver, on a thread without a current
        context; LIBSIRCL_TRANSPORT=bogus makes the engine refuse right after the context step, before any device
        work. ``runtime`` brings the stand-in's cudaGetDevice into the global scope, as a framework's CUDA runtime
        is. Returns (result, ncclGetLastError, (retains, retained device, cuCtxSetCurrent calls), stderr)."""
        load = (f"g=C.CDLL({str(FAKE_DRIVER / 'libcuda.so.1')!r}, mode=C.RTLD_GLOBAL)\n" if runtime else
                f"g=C.CDLL({str(FAKE_DRIVER / 'libcuda.so.1')!r})\n")
        environment = {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "bogus", "FAKE_CUDA_NO_CONTEXT": "1",
                       "LD_LIBRARY_PATH": f"{FAKE_DRIVER}:{os.environ.get('LD_LIBRARY_PATH', '')}", "NCCL_DEBUG": ""}
        environment.update(env)
        out = run(load + '''u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); r=l.ncclCommInitRank(C.byref(c),1,u,0)
n=[C.c_int(),C.c_int(),C.c_int()]; g.sccl_fake_cuda_contexts(*[C.byref(v) for v in n])
print(json.dumps({"result":r,"message":l.ncclGetLastError(None).decode(),"contexts":[v.value for v in n]}))''',
                  environment)
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        return data["result"], data["message"], tuple(data["contexts"]), out.stderr

    def test_init_without_cuda_context_uses_the_only_devices_primary_context(self):
        # vLLM's PyNccl creates its first communicator after torch selected the device but before any CUDA
        # allocation, so no context is current: NVIDIA NCCL then uses the current device, and so does libsircl.
        result, message, contexts, stderr = self.creation({"FAKE_CUDA_DEVICES": "1"})
        self.assertNotIn("no current CUDA context", message)
        self.assertIn("LIBSIRCL_TRANSPORT=bogus", message)   # the engine ran past the context step
        self.assertEqual(contexts, (1, 0, 1))                # device 0's primary context, retained, made current
        self.assertNotEqual(result, 0)
        # The refused creation names its reason on stderr whatever NCCL_DEBUG says.
        self.assertIn("libsircl: communicator creation failed (", stderr)
        self.assertIn("LIBSIRCL_TRANSPORT=bogus", stderr)

    def test_init_without_cuda_context_uses_the_runtimes_current_device(self):
        result, message, contexts, _ = self.creation({"FAKE_CUDA_DEVICES": "2", "FAKE_CUDA_RUNTIME_DEVICE": "1"},
                                                     runtime=True)
        self.assertIn("LIBSIRCL_TRANSPORT=bogus", message)
        self.assertEqual(contexts, (1, 1, 1))

    def test_init_without_cuda_context_is_refused_when_the_device_is_unknown(self):
        # Two devices and no CUDA runtime in the process: no device can be chosen, so the creation is refused,
        # naming the reason, and no primary context is retained.
        result, message, contexts, stderr = self.creation({"FAKE_CUDA_DEVICES": "2"})
        self.assertEqual(result, 5)
        self.assertIn("no current CUDA context, and the current CUDA device is unknown", message)
        self.assertEqual(contexts, (0, -1, 0))
        self.assertIn("libsircl: communicator creation failed (invalid usage): no current CUDA context", stderr)

    def test_init_without_a_cuda_driver_is_refused_with_reason(self):
        out = run('''u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); r=l.ncclCommInitRank(C.byref(c),1,u,0)
print(r, c.value is None, l.ncclGetLastError(None).decode())''',
                  {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LD_LIBRARY_PATH": str(FAKE_DRIVER / "absent"),
                   "LD_PRELOAD": "", "LIBSIRCL_TRANSPORT": "bogus"})
        self.assertEqual(out.returncode, 0, out.stderr)
        code, empty, message = out.stdout.split(" ", 2)
        if "CUDA driver is unavailable" not in message:
            self.skipTest("this host's libcuda.so.1 is found outside LD_LIBRARY_PATH")
        self.assertEqual((code, empty), ("5", "True"))
        self.assertIn("no current CUDA context, and the CUDA driver is unavailable", message)

    def test_nccl_debug_names_the_reason_of_every_failed_call(self):
        quiet = run("print(l.ncclGetVersion(None))", {"NCCL_DEBUG": ""})
        loud = run("print(l.ncclGetVersion(None))", {"NCCL_DEBUG": "WARN"})
        self.assertEqual((quiet.stdout.strip(), loud.stdout.strip()), ("4", "4"))
        self.assertNotIn("version output is NULL", quiet.stderr)
        self.assertIn("libsircl: invalid argument: version output is NULL", loud.stderr)

    def test_collectives_on_cpu_communicators_and_unknown_handles(self):
        out = run('''u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==0
codes=[l.ncclAllReduce(None,None,1,7,0,c,None), l.ncclAllGather(None,None,1,7,c,None),
       l.ncclReduceScatter(None,None,1,7,0,c,None)]
message=l.ncclGetLastError(None).decode()
unknown=[l.ncclAllReduce(None,None,1,7,0,None,None), l.ncclAllGather(None,None,1,7,None,None),
         l.ncclReduceScatter(None,None,1,7,0,None,None)]
need=C.c_size_t(0)
extension=[l.sirclGetReceipt(c,None,0,C.byref(need)), l.sirclSetWaitRegime(c,b"serving")]
assert l.ncclCommDestroy(c)==0
print(json.dumps({"codes":codes,"message":message,"unknown":unknown,"extension":extension}))''',
                  {"LIBSIRCL_BOOTSTRAP_ONLY": "1"})
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        self.assertEqual(data["codes"], [5, 5, 5])
        self.assertIn("CPU-only test communicators carry no collectives", data["message"])
        self.assertEqual(data["unknown"], [4, 4, 4])
        self.assertEqual(data["extension"], [5, 5])

    def test_cuda_free_process_never_initializes_cuda_on_load(self):
        """Loading the library and calling its CUDA-free entry points maps no CUDA driver and initializes
        no CUDA. The child runs as ``python -S -E`` without ``LD_PRELOAD``, so site packages and the
        environment add nothing; whatever is mapped before the load is the baseline the library is held
        to. When the driver is already mapped, ``cuDeviceGetCount`` returns 3 (not initialized) until some
        code calls ``cuInit``, so an initialization by the library still shows."""
        script = r'''
import ctypes as C, json, os, sys
def state():
    names = sorted({line.split()[-1] for line in open("/proc/self/maps") if "/libcuda.so" in line})
    status = None
    if names:
        count = C.c_int(0)
        status = C.CDLL("libcuda.so.1", mode=os.RTLD_NOLOAD | os.RTLD_LAZY).cuDeviceGetCount(C.byref(count))
    return {"mapped": names, "status": status}
before = state()
l = C.CDLL(sys.argv[1])
v = C.c_int(0); assert l.ncclGetVersion(C.byref(v)) == 0
class U(C.Structure): _fields_ = [("b", C.c_ubyte * 128)]
u = U(); assert l.ncclGetUniqueId(C.byref(u)) == 0
l.ncclGetErrorString.restype = C.c_char_p; l.ncclGetErrorString(0)
l.sirclGetInfo.restype = C.c_char_p; l.sirclGetInfo()
print(json.dumps({"before": before, "after": state()}))
'''
        environment = {k: v for k, v in os.environ.items() if k != "LD_PRELOAD" and not k.startswith("PYTHON")}
        out = subprocess.run([sys.executable, "-S", "-E", "-c", script, str(LIBRARY)], env=environment,
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        record = json.loads(out.stdout)
        before, after = record["before"], record["after"]
        self.assertEqual(after["mapped"], before["mapped"], f"the library mapped the CUDA driver: {record}")
        if before["status"] == 0:
            self.skipTest(f"the interpreter alone initializes CUDA before the library loads: {record}")
        self.assertEqual(after["status"], before["status"], f"CUDA was initialized after the load: {record}")

    def test_channel_setup_checks_the_native_feature_word(self):
        """The default build compiles the point-to-point channels' setup check of the native library's local
        feature word (Makefile P2P_FEATURES=1, CMake LIBSIRCL_P2P_FEATURES): its refusal text is in the
        library only when the check is compiled."""
        text = b"does not count failed verbs calls (p2p_local_features bit 0)"
        self.assertIn(text, LIBRARY.read_bytes(), f"{LIBRARY} was built without the feature check")

    def test_setup_checks_the_native_proxy_feature_word(self):
        """Every build checks the native proxy's local feature word (roce_local_features, SIRCL change LF) at
        communicator setup: bits 0 and 1, whose refusal names both."""
        text = (b"this library needs 0x%x (bit 0: roce_destroy counts failed verbs calls; bit 1: flags-only own "
                b"items)")
        self.assertIn(text, LIBRARY.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=LIBRARY)
    options, extra = parser.parse_known_args()
    LIBRARY = options.library.resolve()
    unittest.main(argv=[sys.argv[0], *extra], verbosity=2)
