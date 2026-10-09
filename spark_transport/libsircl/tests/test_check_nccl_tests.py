#!/usr/bin/env python3
"""CPU tests of tools/check_nccl_tests.py on synthetic outputs of tools/nccl_tests_pair.sh: a complete pair
(an all-reduce sweep, an all-to-all whose in-place rows are N/A, receipts whose all-reduces ran as ring and
fold ops) passes; sendrecv and alltoallv sweeps whose in-place rows are N/A, as nccl-tests v2.21.1 prints
them, pass with those rows counted as not covered, also when the job names its binary by path; a missing
manifest, a job that failed after a partial log, a job absent from the expected lines and job lists that
differ between the ranks fail; a nonzero #wrong of a test without an in-place result, out of place or in
place, fails, and an in-place N/A of a test with one (all-reduce, hypercube) fails."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import check_nccl_tests  # noqa: E402

HEADER = """#       size         count      type   redop    root     time   algbw   busbw  #wrong     time   algbw   busbw  #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)             (us)  (GB/s)  (GB/s)
"""
FOOTER = """# Out of bounds values : 0 OK
# Avg bus bandwidth    : 1.5
#
# Collective test concluded: {name}
#
"""
ALL_REDUCE = "all_reduce_perf -b 8 -e 32 -f 2 -g 1"
ALLTOALL = "alltoall_perf -b 8 -e 16 -f 2 -g 1"
SENDRECV = "sendrecv_perf -b 8 -e 32 -f 2 -g 1"
ALLTOALLV = "alltoallv_perf -b 8 -e 16 -f 2 -g 1"
HYPERCUBE = "hypercube_perf -b 8 -e 16 -f 2 -g 1"


def log_name(job: str) -> str:
    name, *rest = job.split()
    return Path(name).name + "".join("_" + word for word in rest) + ".log"


def sweep(name: str, sizes, in_wrong="0", stop=False, out_wrong="0", redop="sum") -> str:
    """nccl-tests' rows; ``redop=""`` leaves the column blank, as tests without a reduction op print it."""
    text = HEADER
    for size in sizes:
        text += (f"{size:12d}  {size // 4:12d}     float  {redop:>6}      -1     9.0    0.00    0.00    {out_wrong:>4}"
                 f"     9.0    0.00    0.00    {in_wrong}\n")
    return text if stop else text + FOOTER.format(name=name)


def receipt(rank: int) -> dict:
    return {"schema": "libsircl-receipt/v1", "forwarded": 0, "refused": {"op": 0, "capture_stream": 0},
            "healthy": True, "all_reduce": {"calls": 3, "ops": {}}, "fold": {"ops": {"int32/max": 1}},
            "chain": {"ops": 0}, "links": {"ops": {"ring_reduce": 2}}}


def write_rank(directory: Path, rank: int, jobs, logs, status=None) -> None:
    directory.mkdir(parents=True)
    rows = []
    for job in jobs:
        name = log_name(job)
        (directory / name).write_text(logs.get(job, "") if rank == 0 else "")
        rows.append(f"{job}\t{name}\t{(status or {}).get(job, 0)}")
    (directory / "jobs.tsv").write_text("\n".join(rows) + "\n")
    (directory / f"receipt.rank{rank}.123.c1.json").write_text(json.dumps(receipt(rank)))


def check(*argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = check_nccl_tests.main([str(a) for a in argv])
    return code, out.getvalue()


class CheckNcclTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="check-nccl-tests-")
        self.root = Path(self.temporary.name)
        self.logs = {ALL_REDUCE: sweep("all_reduce_perf", [8, 16, 32]),
                     ALLTOALL: sweep("alltoall_perf", [8, 16], in_wrong="N/A")}

    def tearDown(self):
        self.temporary.cleanup()

    def pair(self, jobs0, jobs1=None, status=None, logs=None):
        write_rank(self.root / "r0", 0, jobs0, logs or self.logs, status)
        write_rank(self.root / "r1", 1, jobs1 if jobs1 is not None else jobs0, logs or self.logs, status)
        return self.root / "r0", self.root / "r1"

    def test_complete_pair_passes_and_reports_alltoall_in_place_uncovered(self):
        lines = self.root / "lines.txt"
        lines.write_text(f"# the two lines\n{ALL_REDUCE}\n{ALLTOALL}\n")
        code, out = check(*self.pair([ALL_REDUCE, ALLTOALL]), "--expect-lines", lines)
        self.assertEqual(code, 0, out)
        self.assertIn("INFO 2 in-place rows N/A (alltoall_perf 2): not covered", out)
        self.assertNotIn("FAIL", out)

    def test_sendrecv_and_alltoallv_in_place_na_is_not_covered(self):
        logs = dict(self.logs)
        logs[SENDRECV] = sweep("sendrecv_perf", [8, 16, 32], in_wrong="N/A")
        logs[ALLTOALLV] = sweep("alltoallv_perf", [8, 16], in_wrong="N/A")
        code, out = check(*self.pair([ALL_REDUCE, ALLTOALL, SENDRECV, ALLTOALLV], logs=logs))
        self.assertEqual(code, 0, out)
        self.assertIn("PASS #wrong 0 on 10 rows", out)
        self.assertIn("INFO 7 in-place rows N/A (alltoall_perf 2, alltoallv_perf 2, sendrecv_perf 3): not covered",
                      out)

    def test_a_binary_named_by_path_keeps_its_in_place_exemption(self):
        job = "/g/nccl-tests/build/" + SENDRECV
        code, out = check(*self.pair([job], logs={job: sweep("sendrecv_perf", [8, 16, 32], in_wrong="N/A")}))
        self.assertEqual(code, 0, out)
        self.assertIn("INFO 3 in-place rows N/A (sendrecv_perf 3)", out)

    def test_nonzero_wrong_of_a_test_without_in_place_result_fails(self):
        for out_wrong, in_wrong in (("1", "N/A"), ("0", "2")):
            with self.subTest(out_wrong=out_wrong, in_wrong=in_wrong):
                logs = {SENDRECV: sweep("sendrecv_perf", [8, 16, 32], in_wrong=in_wrong, out_wrong=out_wrong)}
                self.root = Path(self.temporary.name) / f"out-{out_wrong}-in-{in_wrong.replace('/', '')}"
                code, out = check(*self.pair([SENDRECV], logs=logs))
                self.assertEqual(code, 1, out)
                self.assertIn("FAIL #wrong 0 on 3 rows", out)
                self.assertIn(f"('{log_name(SENDRECV)}', 8, '{out_wrong}', '{in_wrong}')", out)

    def test_missing_manifest_fails(self):
        r0, r1 = self.pair([ALL_REDUCE])
        (r1 / "jobs.tsv").unlink()
        code, out = check(r0, r1)
        self.assertEqual(code, 1)
        self.assertIn("FAIL job manifests in 1 of 2 directories", out)

    def test_partial_log_then_nonzero_exit_fails(self):
        logs = dict(self.logs)
        logs[ALL_REDUCE] = sweep("all_reduce_perf", [8], stop=True)
        code, out = check(*self.pair([ALL_REDUCE, ALLTOALL], status={ALL_REDUCE: 1}, logs=logs))
        self.assertEqual(code, 1)
        self.assertIn("FAIL every job exited 0", out)
        self.assertIn("FAIL 1 of 2 jobs completed their sweep", out)

    def test_job_missing_from_expected_lines_fails(self):
        lines = self.root / "lines.txt"
        lines.write_text(f"{ALL_REDUCE}\n{ALLTOALL}\n")
        code, out = check(*self.pair([ALL_REDUCE]), "--expect-lines", lines)
        self.assertEqual(code, 1)
        self.assertIn("not run: ['alltoall_perf", out)

    def test_ranks_that_ran_different_jobs_fail(self):
        code, out = check(*self.pair([ALL_REDUCE, ALLTOALL], [ALL_REDUCE]))
        self.assertEqual(code, 1)
        self.assertIn("the job lists differ", out)

    def test_in_place_na_of_a_test_with_an_in_place_result_is_wrong(self):
        logs = dict(self.logs)
        logs[ALL_REDUCE] = sweep("all_reduce_perf", [8, 16, 32], in_wrong="N/A")
        code, out = check(*self.pair([ALL_REDUCE], logs=logs))
        self.assertEqual(code, 1)
        self.assertIn("FAIL #wrong 0 on 3 rows", out)

    def test_in_place_na_of_hypercube_is_wrong(self):
        # nccl-tests v2.21.1's hypercube_perf reports an in-place #wrong, so N/A there is not an exemption.
        logs = {HYPERCUBE: sweep("hypercube_perf", [8, 16], in_wrong="N/A", redop="")}
        code, out = check(*self.pair([HYPERCUBE], logs=logs))
        self.assertEqual(code, 1)
        self.assertIn("FAIL #wrong 0 on 2 rows", out)
        self.assertNotIn("INFO", out)

    def test_rows_without_a_reduction_op_are_checked(self):
        # hypercube_perf leaves the reduction op column blank; its rows parse, complete the sweep and count.
        logs = {HYPERCUBE: sweep("hypercube_perf", [0, 8, 16], redop="")}
        code, out = check(*self.pair([HYPERCUBE], logs=logs))
        self.assertEqual(code, 0, out)
        self.assertIn("PASS 1 of 1 jobs completed their sweep", out)
        self.assertIn("PASS #wrong 0 on 3 rows", out)
        logs = {HYPERCUBE: sweep("hypercube_perf", [8, 16], redop="", out_wrong="4")}
        self.root = Path(self.temporary.name) / "wrong"
        code, out = check(*self.pair([HYPERCUBE], logs=logs))
        self.assertEqual(code, 1)
        self.assertIn(f"('{log_name(HYPERCUBE)}', 8, '4', '0')", out)

    def test_a_data_row_that_does_not_parse_fails(self):
        logs = dict(self.logs)
        logs[ALL_REDUCE] = sweep("all_reduce_perf", [8, 16, 32]).replace(
            "      32             8     float     sum      -1     9.0", "      32             8     float     sum")
        code, out = check(*self.pair([ALL_REDUCE], logs=logs))
        self.assertEqual(code, 1, out)
        self.assertIn("1 data rows not parsed", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)
