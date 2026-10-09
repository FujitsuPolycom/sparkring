#!/usr/bin/env python3
"""CPU tests of SIRCL's point-to-point native library (src/transport/sircl_p2p_proxy.c) across processes.

Each rank is a process running the library's emulation build on an arena in shared memory over the
shared-memory verbs stand-in; host threads play the channel kernels' part (tests/emulation/p2p_probe.c).
The group sets up as a communicator does (connection records, queue-pair connection, lane check, windows,
progress thread) and every rank sends every channel peer messages of 0 to 40,000 bytes through 4 slots of
8,192 bytes, so messages of several items wait for credits; every byte is checked. Further cases: channels
between ring neighbors only with two ranks exchanging while the others idle, windowed lanes, and a size
mismatch that one rank records, after which every rank's progress thread stops naming that rank. The probe
also checks that p2p_layout equals SIRCL's p2p/protocol.py layout. No GPU is used.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class P2PNativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="sircl-p2p-native-")
        cls.probe = Path(cls.build.name) / "p2p_probe"
        transport = ROOT / "src" / "transport"
        cc = os.environ.get("CC", "cc")
        common = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-Wno-format-truncation", "-pthread", "-I",
                  str(transport / "fake_verbs"), "-I", str(transport)]
        subprocess.run([cc, *common, str(ROOT / "tests" / "emulation" / "p2p_probe.c"), str(transport / "shm_verbs.c"),
                        str(transport / "p2p_emu.c"), "-o", str(cls.probe)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def run_group(self, world, lanes, messages, mode, **settings):
        with tempfile.TemporaryDirectory(prefix="sircl-p2p-group-") as directory:
            env = dict(os.environ, SIRCL_EMU_FABRIC=f"/sircl-emu-p2p-{os.getpid()}-{world}-{lanes}-{mode}", **settings)
            processes = [subprocess.Popen([str(self.probe), str(world), str(rank), str(lanes), str(messages), mode,
                                           directory], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                          text=True)
                         for rank in range(world)]
            results = [p.communicate(timeout=300) for p in processes]
            fabric = Path("/dev/shm") / env["SIRCL_EMU_FABRIC"].lstrip("/")
            if fabric.exists():
                fabric.unlink()
        for rank, (process, (out, err)) in enumerate(zip(processes, results)):
            self.assertEqual(process.returncode, 0, f"rank {rank}: {out}{err}")
        leftovers = [p.name for p in Path("/dev/shm").glob("sircl-emu-seg-*")
                     if any(p.name.startswith(f"sircl-emu-seg-{process.pid}-") for process in processes)]
        self.assertEqual(leftovers, [])
        return [out for out, _ in results]

    def test_four_ranks_every_pair_two_lanes(self):
        outputs = self.run_group(4, 2, 24, "all")
        self.assertTrue(all("24 messages per channel exact" in out for out in outputs), outputs)

    def test_eight_ranks_every_pair_one_lane(self):
        self.run_group(8, 1, 12, "all")

    def test_ring_channels_with_idle_ranks(self):
        self.run_group(4, 2, 32, "ring")

    def test_windowed_lanes(self):
        # Stripes of up to 4,096 bytes through windows of one 4,096-byte chunk, each write executed 100 us
        # after it was posted: lanes wait for room, and credits prove writes delivered.
        outputs = self.run_group(4, 2, 16, "windows", SIRCL_EMU_LATENCY_NS="100000")
        self.assertTrue(any("window waits 0," not in out for out in outputs), outputs)

    def test_size_mismatch_stops_every_rank(self):
        outputs = self.run_group(4, 1, 1, "size")
        for rank, out in enumerate(outputs):
            self.assertIn("channels stopped 1", out)
            if rank != 1:
                self.assertIn("abort from rank 1", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
