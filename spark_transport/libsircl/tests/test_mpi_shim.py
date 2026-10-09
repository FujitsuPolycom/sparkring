#!/usr/bin/env python3
"""CPU tests of the MPI shim (tools/mpi-shim): every call nccl-tests makes, across 1, 3 and 4 processes.

The probe program calls the shim the way nccl-tests v2.21.1 does (in-place all-gather of host hashes,
a split with one color, the unique-id broadcast from rank 0 of the split, barriers, in-place
all-reduces of int, long, long long, int64 and double with sum, min and max, gathers of doubles and
bytes to rank 0) and prints its results; every rank must print the expected values. The split probe
does what nccl-tests' communicator-operations test does: splits with two colors and with MPI_UNDEFINED,
a split of a split, all-gathers of different sizes (MPI_Allgatherv of MPI_CHAR), reductions to a root
(MPI_Reduce), broadcasts on a sub-communicator whose rank 0 is not world rank 0, and MPI_Error_string.
"""
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = r'''
#include <mpi.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int size, rank, nsize, nrank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  uint64_t hashes[8] = {0};
  hashes[rank] = 1000 + rank;
  if (MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hashes, sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD)) return 2;
  MPI_Comm comm;
  if (MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm)) return 3;
  MPI_Comm_size(comm, &nsize);
  MPI_Comm_rank(comm, &nrank);
  unsigned char id[128];
  memset(id, nrank == 0 ? 0x5a : 0, sizeof id);
  if (MPI_Bcast(id, sizeof id, MPI_BYTE, 0, comm)) return 4;
  MPI_Barrier(MPI_COMM_WORLD);
  int i = rank + 1, imin = rank + 1;
  long l = 10 * (rank + 1);
  long long ll = 100 * (rank + 1);
  int64_t i64 = rank;
  double d = 0.5 * (rank + 1);
  MPI_Allreduce(MPI_IN_PLACE, &i, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &imin, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &l, 1, MPI_LONG, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &ll, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &i64, 1, MPI_INT64_T, MPI_MAX, MPI_COMM_WORLD);
  MPI_Allreduce(MPI_IN_PLACE, &d, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
  double mine[2] = {rank * 1.0, rank * 2.0}, all[16];
  MPI_Gather(mine, 2, MPI_DOUBLE, all, 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  char line[8], lines[64];
  snprintf(line, sizeof line, "r%d", rank);
  MPI_Gather(line, 8, MPI_BYTE, lines, 8, MPI_BYTE, 0, MPI_COMM_WORLD);
  unsigned long hsum = 0;
  for (int r = 0; r < size; ++r) hsum += hashes[r];
  printf("size=%d rank=%d nsize=%d nrank=%d hsum=%lu id=%d i=%d imin=%d l=%ld ll=%lld i64=%lld d=%g", size, rank,
         nsize, nrank, hsum, id[127], i, imin, l, ll, (long long)i64, d);
  if (rank == 0) {
    printf(" gathered=");
    for (int r = 0; r < size; ++r) printf("%g,%g,%s;", all[2 * r], all[2 * r + 1], lines + 8 * r);
  }
  printf("\n");
  MPI_Comm_free(&comm);
  MPI_Finalize();
  return 0;
}
'''

SPLIT_PROBE = r"""
#include <mpi.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size, crank, csize;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm half;
  if (MPI_Comm_split(MPI_COMM_WORLD, rank % 2, rank, &half)) return 2;
  MPI_Comm_rank(half, &crank);
  MPI_Comm_size(half, &csize);
  /* Broadcast from the group's rank 0 (world rank 0 or 1). */
  int bc = crank == 0 ? 7 + rank % 2 : 0;
  if (MPI_Bcast(&bc, 1, MPI_INT, 0, half)) return 3;
  /* Contributions of different sizes: "a" from group rank 0, "bb" from group rank 1. */
  int counts[2] = {1, 2}, displs[2] = {0, 2};
  char mine[2] = {'a', 'a'}, all[5] = {0};
  if (crank == 1) mine[0] = mine[1] = 'b';
  if (MPI_Allgatherv(mine, counts[crank], MPI_CHAR, all, counts, displs, MPI_CHAR, half)) return 4;
  char v[16];
  snprintf(v, sizeof v, "%.1s,%.2s,", all, all + 2);
  /* Sum to group rank 0 of the even group. */
  int one = 1, red = 0;
  if (MPI_Reduce(&one, &red, 1, MPI_INT, MPI_SUM, 0, half)) return 5;
  /* A split with MPI_UNDEFINED: world rank 3 joins nothing; the rest, ordered by descending key. */
  MPI_Comm some;
  if (MPI_Comm_split(MPI_COMM_WORLD, rank == 3 ? MPI_UNDEFINED : 0, -rank, &some)) return 6;
  char und[16] = "null";
  if (some != MPI_COMM_NULL) {
    int n, r;
    MPI_Comm_size(some, &n);
    MPI_Comm_rank(some, &r);
    snprintf(und, sizeof und, "%d:%d", n, r);
  }
  /* A split of a split: each half splits by its own rank; every part is one process. */
  MPI_Comm sub;
  if (MPI_Comm_split(half, crank, 0, &sub)) return 7;
  int sub_size, sub_rank;
  MPI_Comm_size(sub, &sub_size);
  MPI_Comm_rank(sub, &sub_rank);
  double d = rank, dmax = 0;
  if (MPI_Allreduce(&d, &dmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD)) return 8;
  char err[MPI_MAX_ERROR_STRING];
  int length = 0;
  MPI_Error_string(MPI_Bcast(&bc, 1, MPI_INT, 9, half), err, &length);
  if (crank == 0 && rank % 2 == 0)
    printf("rank=%d color=%d size=%d crank=%d bc=%d v=%s red=%d und=%s sub=%d:%d max=%g err=%.12s\n", rank,
           rank % 2, csize, crank, bc, v, red, und, size / 2, sub_rank, dmax, err);
  else
    printf("rank=%d color=%d size=%d crank=%d bc=%d v=%s und=%s sub=%d:%d max=%g err=%.12s\n", rank, rank % 2,
           csize, crank, bc, v, und, size / 2, sub_rank, dmax, err);
  (void)sub_size;
  MPI_Comm_free(&sub);
  if (some != MPI_COMM_NULL) MPI_Comm_free(&some);
  MPI_Comm_free(&half);
  MPI_Finalize();
  return 0;
}
"""


class MpiShimTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.build = tempfile.TemporaryDirectory(prefix="sircl-mpi-shim-")
        cls.probe = Path(cls.build.name) / "probe"
        source = Path(cls.build.name) / "probe.c"
        source.write_text(PROBE)
        shim = ROOT / "tools" / "mpi-shim"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror", "-O2", "-pthread",
                        "-I", str(shim / "include"), str(source), str(shim / "mpi_shim.c"), "-o", str(cls.probe)],
                       check=True)

    @classmethod
    def tearDownClass(cls):
        cls.build.cleanup()

    def run_group(self, size, binary=None):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        processes = []
        for rank in range(size):
            env = dict(os.environ, SIRCL_MPI_SIZE=str(size), SIRCL_MPI_RANK=str(rank),
                       SIRCL_MPI_ROOT=f"127.0.0.1:{port}", SIRCL_MPI_TIMEOUT_S="20")
            processes.append(subprocess.Popen([str(binary or self.probe)], env=env, stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE, text=True))
        outputs = []
        for process in processes:
            out, err = process.communicate(timeout=60)
            self.assertEqual(process.returncode, 0, err)
            outputs.append(out.strip())
        return outputs

    def test_three_processes(self):
        outputs = self.run_group(3)
        for rank, line in enumerate(outputs):
            self.assertIn(f"size=3 rank={rank} nsize=3 nrank={rank} hsum=3003 id=90 i=6 imin=1 l=10 ll=600 "
                          f"i64=2 d=1.5", line)
        self.assertIn("gathered=0,0,r0;1,2,r1;2,4,r2;", outputs[0])

    def test_single_process_without_environment(self):
        env = {k: v for k, v in os.environ.items() if not k.startswith("SIRCL_MPI_")}
        out = subprocess.run([str(self.probe)], env=env, capture_output=True, text=True, check=True).stdout
        self.assertIn("size=1 rank=0 nsize=1 nrank=0 hsum=1000 id=90 i=1 imin=1 l=10 ll=100 i64=0 d=0.5", out)

    def test_splits_and_rooted_collectives(self):
        source = Path(self.build.name) / "split.c"
        source.write_text(SPLIT_PROBE)
        binary = Path(self.build.name) / "split"
        shim = ROOT / "tools" / "mpi-shim"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread",
                        "-I", str(shim / "include"), str(source), str(shim / "mpi_shim.c"), "-o", str(binary)],
                       check=True)
        outputs = self.run_group(4, binary)
        # Colors: even and odd world ranks; the odd group's rank 0 is world rank 1. World rank 3 takes no part in
        # the MPI_UNDEFINED split; the others form one ordered by descending key.
        self.assertIn("rank=0 color=0 size=2 crank=0 bc=7 v=a,bb, red=2 und=3:2", outputs[0])
        self.assertIn("rank=1 color=1 size=2 crank=0 bc=8 v=a,bb, und=3:1", outputs[1])
        self.assertIn("rank=2 color=0 size=2 crank=1 bc=7 v=a,bb, und=3:0", outputs[2])
        self.assertIn("rank=3 color=1 size=2 crank=1 bc=8 v=a,bb, und=null", outputs[3])
        for line in outputs:
            self.assertIn("sub=2:", line)
            self.assertIn("err=MPI_ERR_ROOT", line)

    def _compile(self, name, source):
        path = Path(self.build.name) / f"{name}.c"
        path.write_text(source)
        binary = Path(self.build.name) / name
        shim = ROOT / "tools" / "mpi-shim"
        subprocess.run([os.environ.get("CC", "cc"), "-std=c11", "-Wall", "-Wextra", "-Werror", "-pthread", "-I",
                        str(shim / "include"), str(path), str(shim / "mpi_shim.c"), "-o", str(binary)], check=True)
        return binary

    def _start(self, binary, rank, port, **env):
        environment = dict(os.environ, SIRCL_MPI_SIZE="2", SIRCL_MPI_RANK=str(rank),
                           SIRCL_MPI_ROOT=f"127.0.0.1:{port}", SIRCL_MPI_TIMEOUT_S="20")
        environment.update(env)
        return subprocess.Popen([str(binary)], env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)

    def test_a_peer_ending_outside_finalize_ends_the_other(self):
        """Rank 1 ends right after MPI_Init; rank 0, busy outside MPI, ends within seconds and names it."""
        binary = self._compile("crash", """#include <mpi.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 1) _exit(3);
  sleep(30);
  return 0;
}
""")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        started = time.monotonic()
        ranks = [self._start(binary, r, port) for r in (1, 0)]
        results = [p.communicate(timeout=60) for p in ranks]
        elapsed = time.monotonic() - started
        self.assertEqual([p.returncode for p in ranks], [3, 1])
        self.assertIn("rank 1's process ended outside MPI_Finalize", results[1][1])
        self.assertLess(elapsed, 15)

    def test_distinct_host_names_for_ranks_of_one_host(self):
        """SIRCL_MPI_DISTINCT_HOSTS=1: each rank's gethostname carries its rank; without it, the host's name."""
        binary = self._compile("host", """#define _DEFAULT_SOURCE
#include <mpi.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  char name[256];
  if (gethostname(name, sizeof name) != 0) return 2;
  printf("%s\\n", name);
  MPI_Finalize();
  return 0;
}
""")
        host = socket.gethostname().split(".")[0]
        for distinct, expect in (("1", lambda r: f"{host}-rank{r}"), ("", lambda r: host)):
            with socket.socket() as s:
                s.bind(("127.0.0.1", 0))
                port = s.getsockname()[1]
            ranks = [self._start(binary, r, port, SIRCL_MPI_DISTINCT_HOSTS=distinct) for r in (0, 1)]
            results = [p.communicate(timeout=60) for p in ranks]
            self.assertEqual([p.returncode for p in ranks], [0, 0], results)
            self.assertEqual([out.strip().split(".")[0] for out, _ in results], [expect(0), expect(1)])

    def test_processes_of_different_jobs_refuse_each_other(self):
        """Rank 1 of job "b" never joins rank 0 of job "a": both fail by the deadline."""
        binary = self._compile("job", """#include <mpi.h>
int main(int argc, char **argv) { MPI_Init(&argc, &argv); MPI_Finalize(); return 0; }
""")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        ranks = [self._start(binary, 0, port, SIRCL_MPI_JOB="a", SIRCL_MPI_TIMEOUT_S="3"),
                 self._start(binary, 1, port, SIRCL_MPI_JOB="b", SIRCL_MPI_TIMEOUT_S="3")]
        results = [p.communicate(timeout=60) for p in ranks]
        self.assertTrue(all(p.returncode != 0 for p in ranks))
        self.assertIn("timed out", results[0][1])
        self.assertTrue("did not accept this rank" in results[1][1] or "connecting to a peer" in results[1][1],
                        results[1][1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
