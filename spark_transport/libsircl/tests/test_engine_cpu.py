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
import tempfile
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
    if os.environ.get("LIBSIRCL_TEST_ASAN"):
        environment["LD_PRELOAD"] = os.environ["LD_PRELOAD"]
    return subprocess.run([sys.executable, "-c", PRELUDE + script, str(LIBRARY)], env=environment,
                          capture_output=True, text=True, timeout=60)


class EngineCpuTests(unittest.TestCase):
    def test_finalize_and_destroy_wait_for_the_context(self):
        out = run('''g=C.CDLL("libcuda.so.1")
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==0
assert l.ncclCommFinalize(c)==0
before=g.sccl_fake_cuda_context_syncs()
assert l.ncclCommDestroy(c)==0
print(json.dumps([before,g.sccl_fake_cuda_context_syncs()]))
''', {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "verbs",
       "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
        self.assertEqual(out.returncode, 0, out.stderr)
        before, after = json.loads(out.stdout)
        self.assertGreaterEqual(before, 1)
        self.assertGreater(after, before)

    def test_one_pass_ring_relay_needs_the_research_opt_in(self):
        """LIBSIRCL_RING_REDUCE_PASSES=1 selects the one-pass ring relay, which is research-only: creation is
        refused, naming the opt-in, unless LIBSIRCL_RING_REDUCE_ONE_PASS_RESEARCH=1; 2 (the default) and the
        opted-in 1 create a communicator on the stand-in driver."""
        script = '''u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); r=l.ncclCommInitRank(C.byref(c),1,u,0)
print(json.dumps([r, l.ncclGetLastError(None).decode()]))
if c.value: l.ncclCommDestroy(c)
'''
        base = {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "verbs", "LD_LIBRARY_PATH": str(FAKE_DRIVER)}
        for passes, opt_in, refused in (("1", "", True), ("1", "0", True), ("1", "1", False), ("2", "", False)):
            with self.subTest(passes=passes, opt_in=opt_in):
                out = run(script, {**base, "LIBSIRCL_RING_REDUCE_PASSES": passes,
                                   "LIBSIRCL_RING_REDUCE_ONE_PASS_RESEARCH": opt_in})
                self.assertEqual(out.returncode, 0, out.stderr)
                result, message = json.loads(out.stdout)
                if refused:
                    self.assertNotEqual(result, 0)
                    self.assertIn("LIBSIRCL_RING_REDUCE_ONE_PASS_RESEARCH=1", message)
                else:
                    self.assertEqual(result, 0, message)

    def test_failed_creations_release_their_bootstrap(self):
        """Seventy creations that fail in session setup (LIBSIRCL_TRANSPORT=bogus), whose handles are never
        destroyed, leave no socket open and no bootstrap root held: a further unique id is still granted
        (the process holds at most 64 roots)."""
        out = run('''import os, time
before=len(os.listdir("/proc/self/fd"))
handles=[]
for _ in range(70):
    u=U(); assert l.ncclGetUniqueId(C.byref(u))==0, l.ncclGetLastError(None)
    c=C.c_void_p(); r=l.ncclCommInitRank(C.byref(c),1,u,0); assert r!=0 and c.value; handles.append(c)
time.sleep(0.3)
after=len(os.listdir("/proc/self/fd"))
u=U(); last=l.ncclGetUniqueId(C.byref(u))
print(json.dumps([before, after, last, l.ncclGetLastError(None).decode()]))
''', {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "bogus", "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        before, after, last, message = json.loads(out.stdout)
        self.assertEqual(last, 0, message)
        self.assertEqual(after, before)

    def test_failed_initall_leaves_no_handles(self):
        """An ncclCommInitAll whose communicators fail in session setup aborts them and clears every output."""
        out = run("c=(C.c_void_p*2)(); r=l.ncclCommInitAll(c,2,(C.c_int*2)(0,0)); "
                  "print(json.dumps([r,list(c),l.ncclGetLastError(None).decode()]))",
                  {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "bogus", "FAKE_CUDA_DEVICES": "1",
                   "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
        self.assertEqual(out.returncode, 0, out.stderr)
        result, handles, message = json.loads(out.stdout)
        self.assertNotIn(result, (0, 7))
        self.assertEqual(handles, [None, None])
        self.assertIn("group initialization failed", message)

    def test_destroy_without_its_context_frees_the_host_bookkeeping(self):
        """Destroying a communicator whose context can no longer be made current (the stand-in driver fails
        every push, as for a destroyed context) frees its engine: 1,000 such destroys leave the process heap
        within 1 MiB of where it started. Not under AddressSanitizer, whose allocator mallinfo2 does not see."""
        if os.environ.get("LIBSIRCL_TEST_ASAN"):
            self.skipTest("mallinfo2 does not measure AddressSanitizer's allocator")
        out = run('''g=C.CDLL("libcuda.so.1")
class Info(C.Structure): _fields_=[(n, C.c_size_t) for n in ("arena","ordblks","smblks","hblks","hblkhd","usmblks","fsmblks","uordblks","fordblks","keepcost")]
libc=C.CDLL(None); libc.mallinfo2.restype=Info
def used(): return libc.mallinfo2().uordblks + libc.mallinfo2().hblkhd
def cycle():
    u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
    c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==0
    g.sccl_fake_cuda_fail_push(1); l.ncclCommDestroy(c); g.sccl_fake_cuda_fail_push(0)
for _ in range(5): cycle()
before=used()
for _ in range(1000): cycle()
print(json.dumps([before, used()]))
''', {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "verbs", "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        before, after = json.loads(out.stdout)
        self.assertLess(after - before, 1 << 20, (before, after))

    def test_initall_refuses_distinct_cuda_devices(self):
        for devices in ("None", "(C.c_int*2)(0,1)"):
            out = run("c=(C.c_void_p*2)(); r=l.ncclCommInitAll(c,2," + devices + "); "
                      "print(json.dumps([r,list(c),l.ncclGetLastError(None).decode()]))",
                      {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "bogus",
                       "FAKE_CUDA_DEVICES": "2", "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
            self.assertEqual(out.returncode, 0, out.stderr)
            result, handles, message = json.loads(out.stdout)
            self.assertEqual(result, 5)
            self.assertEqual(handles, [None, None])
            self.assertIn("distinct devices", message)

    def test_setup_refuses_out_of_range_peer_positions(self):
        for position in (-1, 64, 2147483647):
            with self.subTest(position=position):
                out = run('''import socket, struct, threading
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
server=socket.socket(); server.bind(("127.0.0.1", 0)); server.listen()
u.b[16:18]=struct.pack("!H",server.getsockname()[1])
records=[]
def broker():
    peer,_=server.accept(); peer.settimeout(3)
    def read(n):
        data=b""
        while len(data)<n:
            part=peer.recv(n-len(data))
            assert part
            data+=part
        return data
    read(136); peer.sendall(b"SCC1"+struct.pack("!I",0))
    for number in range(2):
        h=read(16); size=struct.unpack("!I",h[8:12])[0]; mine=read(size)
        records.append(mine)
        other=struct.pack("=iiQ", POSITION,0,0) if number==0 else bytes(size)
        body=struct.pack("!I",len(mine))+mine+struct.pack("!I",len(other))+other
        peer.sendall(b"SCR1"+struct.pack("!III",number,2,len(body))+body)
    peer.close()
t=threading.Thread(target=broker); t.start()
c=C.c_void_p(); result=l.ncclCommInitRank(C.byref(c),2,u,0)
t.join(); server.close()
print(json.dumps({"result":result,"local":records[1][:400].split(bytes([0]))[0].decode()}))
if c.value: l.ncclCommAbort(c)
'''.replace("POSITION", str(position)),
                          {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "bogus",
                           "NCCL_SOCKET_IFNAME": "lo", "LIBSIRCL_SETUP_TIMEOUT_MS": "2000",
                           "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
                self.assertEqual(out.returncode, 0, out.stderr)
                data = json.loads(out.stdout)
                self.assertNotEqual(data["result"], 0)
                self.assertIn("rank 1 position", data["local"])
                self.assertIn("0-63", data["local"])

    def test_route_and_check_settings_reach_the_setup_exchange(self):
        """LIBSIRCL_RELAYED_LANES marks a peer position as relayed in the first setup exchange's record (which
        the cycle plan reads) without any forward window; a malformed value, or a LIBSIRCL_RING_WINDOW above the
        native cap, a LIBSIRCL_LANE_CHECK_MS out of range, or an invalid LIBSIRCL_FAIL_STOP or LIBSIRCL_FAIL_STOP_POLL_MS
        fails the first verdict naming the variable."""
        script = '''import socket, struct, threading
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
server=socket.socket(); server.bind(("127.0.0.1", 0)); server.listen()
u.b[16:18]=struct.pack("!H",server.getsockname()[1])
records=[]
def broker():
    peer,_=server.accept(); peer.settimeout(3)
    def read(n):
        data=b""
        while len(data)<n:
            part=peer.recv(n-len(data))
            assert part
            data+=part
        return data
    read(136); peer.sendall(b"SCC1"+struct.pack("!I",0))
    for number in range(2):
        h=read(16); size=struct.unpack("!I",h[8:12])[0]; mine=read(size)
        records.append(mine)
        other=struct.pack("=iiQ",1,0,0) if number==0 else bytes(size)
        body=struct.pack("!I",len(mine))+mine+struct.pack("!I",len(other))+other
        peer.sendall(b"SCR1"+struct.pack("!III",number,2,len(body))+body)
    peer.close()
t=threading.Thread(target=broker); t.start()
c=C.c_void_p(); result=l.ncclCommInitRank(C.byref(c),2,u,0)
t.join(); server.close()
print(json.dumps({"result":result,"relayed":struct.unpack("=iIQ", records[0])[2],
                  "local":records[1][:400].split(bytes([0]))[0].decode()}))
if c.value: l.ncclCommAbort(c)
'''
        base = {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "bogus", "NCCL_SOCKET_IFNAME": "lo",
                "LIBSIRCL_SETUP_TIMEOUT_MS": "1000", "LD_LIBRARY_PATH": str(FAKE_DRIVER)}
        out = run(script, {**base, "LIBSIRCL_RELAYED_LANES": "1=1/0,5=0/1"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout)["relayed"], (1 << 1) | (1 << 5))
        # The verbs transport: its refusal comes after the settings, so the settings' own refusal is the first.
        out = run(script, {**base, "LIBSIRCL_TRANSPORT": "verbs", "LIBSIRCL_RELAYED_LANES": "1=2"})
        self.assertEqual(out.returncode, 0, out.stderr)
        data = json.loads(out.stdout)
        self.assertNotEqual(data["result"], 0)
        self.assertIn("LIBSIRCL_RELAYED_LANES", data["local"])
        # A ring window above the native cap of 60 chunks of 32 KiB is refused, naming the variable.
        out = run(script, {**base, "LIBSIRCL_TRANSPORT": "verbs", "LIBSIRCL_RING_WINDOW": str(61 * 32768)})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("LIBSIRCL_RING_WINDOW=1998848 must be 0 or hold 1 to 60 chunks", json.loads(out.stdout)["local"])
        # The lane check's bound is read with the settings, so a bad value fails the first verdict.
        out = run(script, {**base, "LIBSIRCL_TRANSPORT": "verbs", "LIBSIRCL_LANE_CHECK_MS": "0"})
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("LIBSIRCL_LANE_CHECK_MS=0 must be an integer from 1", json.loads(out.stdout)["local"])
        # Fail-stop's settings: refused before the first verdict, naming the variable.
        for extra, text in (({"LIBSIRCL_FAIL_STOP": "yes"}, "LIBSIRCL_FAIL_STOP=yes: 0, 1 or abort"),
                            ({"LIBSIRCL_FAIL_STOP": "1", "LIBSIRCL_FAIL_STOP_POLL_MS": "0"},
                             "LIBSIRCL_FAIL_STOP_POLL_MS=0 must be an integer from 1")):
            with self.subTest(**extra):
                out = run(script, {**base, "LIBSIRCL_TRANSPORT": "verbs", **extra})
                self.assertEqual(out.returncode, 0, out.stderr)
                self.assertIn(text, json.loads(out.stdout)["local"])

    def test_receipts_never_follow_a_planted_link(self):
        """A symbolic link planted where a receipt's partial file was named (<receipt>.partial) is left as it
        is and the file it names is not written: receipts are written under a name of the writer's own,
        opened with O_EXCL and O_NOFOLLOW, then renamed over the receipt."""
        with tempfile.TemporaryDirectory() as directory:
            out = run('''import os, sys
directory=sys.argv[2]
target=os.path.join(directory, "victim"); open(target, "w").write("untouched")
prefix=os.path.join(directory, "r")
receipt=f"{prefix}.rank0.{os.getpid()}.c1.json"
os.symlink(target, receipt + ".partial")
os.environ["LIBSIRCL_RECEIPT"]=prefix
u=U(); assert l.ncclGetUniqueId(C.byref(u))==0
c=C.c_void_p(); assert l.ncclCommInitRank(C.byref(c),1,u,0)==0
assert l.ncclCommDestroy(c)==0
print(json.dumps([open(target).read(), os.path.islink(receipt + ".partial"), os.path.isfile(receipt),
                  sorted(name for name in os.listdir(directory) if ".partial." in name)]))
'''.replace("sys.argv[2]", repr(directory)), {"LIBSIRCL_BOOTSTRAP_ONLY": "", "LIBSIRCL_TRANSPORT": "verbs",
                                                    "LD_LIBRARY_PATH": str(FAKE_DRIVER)})
            self.assertEqual(out.returncode, 0, out.stderr[-2000:])
            victim, still_link, written, leftovers = json.loads(out.stdout)
            self.assertEqual(victim, "untouched")
            self.assertTrue(still_link)
            self.assertTrue(written)
            self.assertEqual(leftovers, [])

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
        if os.environ.get("LIBSIRCL_TEST_ASAN"):
            environment["LD_PRELOAD"] = os.environ["LD_PRELOAD"]
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
    FAKE_DRIVER = LIBRARY.parent / "fake-cuda"
    unittest.main(argv=[sys.argv[0], *extra], verbosity=2)
