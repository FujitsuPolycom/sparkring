#!/usr/bin/env python3
"""CPU checks of the collective entry points and the engine's refusals. No GPU or RDMA device is used.

Each check runs in a fresh process: the library must load and answer without a CUDA context, refuse
communicator creation without one (naming the reason), and refuse collectives on CPU-only test
communicators and unknown handles with NCCL's result codes.
"""
import argparse
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

LIBRARY = Path(__file__).resolve().parents[1] / "build" / "libsircl.so"
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

    def test_init_without_cuda_context_is_refused_with_reason(self):
        out = run('''u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); r=l.ncclCommInitRank(C.byref(c),1,u,0)
print(r, c.value is None, l.ncclGetLastError(None).decode())''', {"LIBSIRCL_BOOTSTRAP_ONLY": ""})
        self.assertEqual(out.returncode, 0, out.stderr)
        code, empty, message = out.stdout.split(" ", 2)
        self.assertEqual((code, empty), ("5", "True"))
        self.assertIn("no current CUDA context", message)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, default=LIBRARY)
    options, extra = parser.parse_known_args()
    LIBRARY = options.library.resolve()
    unittest.main(argv=[sys.argv[0], *extra], verbosity=2)
