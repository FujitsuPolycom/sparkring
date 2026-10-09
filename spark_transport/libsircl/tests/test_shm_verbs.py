#!/usr/bin/env python3
"""CPU tests of the shared-memory verbs stand-in (src/transport/shm_verbs.c) across processes.

Each rank is a process running SIRCL's native proxy (the library's emulation build of it) on an arena
in shared memory; the host plays the one-shot kernel's part (tests/emulation/shm_verbs_probe.c). The
group sets up exactly as a communicator does (connection records, queue-pair connection, lane check,
progress thread) and runs ops of 16 B to 64 KiB, alternating slots, with every peer's bytes checked.
No GPU is used.
"""
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ShmVerbsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="sircl-shm-verbs-")
        cls.probe = Path(cls.build.name) / "probe"
        transport = ROOT / "src" / "transport"
        cc = os.environ.get("CC", "cc")
        common = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "-I", str(transport / "fake_verbs"),
                  "-I", str(transport)]
        subprocess.run([cc, *common, str(ROOT / "tests" / "emulation" / "shm_verbs_probe.c"),
                        str(transport / "shm_verbs.c"), str(transport / "proxy_emu.c"), "-o", str(cls.probe)],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def run_group(self, world, lanes, ops):
        with tempfile.TemporaryDirectory(prefix="sircl-shm-group-") as directory:
            env = dict(os.environ, SIRCL_EMU_FABRIC=f"/sircl-emu-test-{os.getpid()}-{world}-{lanes}")
            processes = [subprocess.Popen([str(self.probe), str(world), str(rank), str(lanes), str(ops), directory],
                                          env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                         for rank in range(world)]
            results = [p.communicate(timeout=120) for p in processes]
            fabric = Path("/dev/shm") / env["SIRCL_EMU_FABRIC"].lstrip("/")
            if fabric.exists():
                fabric.unlink()
        for rank, (process, (out, err)) in enumerate(zip(processes, results)):
            self.assertEqual(process.returncode, 0, f"rank {rank}: {err}")
            self.assertIn(f"{ops} ops exact from {world - 1} peers over {lanes} lane(s)", out)
        leftovers = [p.name for p in Path("/dev/shm").glob("sircl-emu-seg-*")
                     if any(p.name.startswith(f"sircl-emu-seg-{process.pid}-") for process in processes)]
        self.assertEqual(leftovers, [])

    def test_pair_one_lane(self):
        self.run_group(2, 1, 64)

    def test_pair_two_lanes(self):
        self.run_group(2, 2, 64)

    def test_four_ranks_two_lanes(self):
        self.run_group(4, 2, 32)

    def test_eight_ranks_one_lane(self):
        self.run_group(8, 1, 16)

    def test_stale_segments_of_the_same_names_are_skipped(self):
        # Segments that a removed container left in a shared /dev/shm under the names these ranks try first
        # (SIRCL_EMU_SEGMENT_TAG fixes the <pid>-<start> part): the group still runs exact, takes other
        # names, and leaves the stale segments as they were.
        world, lanes, ops = 2, 2, 16
        tags = [f"staletest{os.getpid()}r{rank}" for rank in range(world)]
        stale = [Path("/dev/shm") / f"sircl-emu-seg-{tag}-{serial}" for tag in tags for serial in range(1, 5)]
        for path in stale:
            path.write_bytes(b"stale segment")
        try:
            with tempfile.TemporaryDirectory(prefix="sircl-shm-group-") as directory:
                fabric = f"/sircl-emu-test-stale-{os.getpid()}"
                processes = [subprocess.Popen([str(self.probe), str(world), str(rank), str(lanes), str(ops), directory],
                                              env=dict(os.environ, SIRCL_EMU_FABRIC=fabric, SIRCL_EMU_SEGMENT_TAG=tags[rank]),
                                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                             for rank in range(world)]
                results = [p.communicate(timeout=120) for p in processes]
                shared = Path("/dev/shm") / fabric.lstrip("/")
                if shared.exists():
                    shared.unlink()
            for rank, (process, (out, err)) in enumerate(zip(processes, results)):
                self.assertEqual(process.returncode, 0, f"rank {rank}: {err}")
                self.assertIn(f"{ops} ops exact from {world - 1} peers over {lanes} lane(s)", out)
            self.assertTrue(all(path.read_bytes() == b"stale segment" for path in stale))
            leftovers = [p.name for p in Path("/dev/shm").glob("sircl-emu-seg-staletest*")
                         if any(p.name.startswith(f"sircl-emu-seg-{tag}-") for tag in tags) and p not in stale]
            self.assertEqual(leftovers, [])
        finally:
            for path in stale:
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
