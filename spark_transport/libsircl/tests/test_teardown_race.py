#!/usr/bin/env python3
"""CPU checks of the teardown harness's verdicts, thread failures and process deadline."""
from concurrent.futures import ThreadPoolExecutor
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from emulation import teardown_race as race


class ConcurrentCloseTests(unittest.TestCase):
    def run_ranks(self, accepted, close_codes, *, after_close=4, worker_code=5):
        """Four ranks enqueue independently; a barrier publishes their actual accepted counts."""
        barrier = threading.Barrier(len(accepted), timeout=5)
        counts = [None] * len(accepted)

        def rank_run(rank):
            ready, closing, waiting = threading.Event(), threading.Event(), threading.Event()
            posted, destroyed = 0, False

            def enqueue():
                nonlocal posted
                if destroyed:
                    return after_close
                if posted < accepted[rank]:
                    posted += 1
                    if posted == accepted[rank]:
                        ready.set()
                    return 0
                waiting.set()
                if not closing.wait(5):
                    raise TimeoutError("test close did not begin")
                return worker_code

            def finalize():
                self.assertTrue(ready.wait(5))
                self.assertTrue(waiting.wait(5))
                closing.set()
                return close_codes[0]

            def destroy():
                nonlocal destroyed
                destroyed = True
                return close_codes[1]

            def exchange(count):
                counts[rank] = count
                barrier.wait()
                return list(counts)

            return race.close_concurrently(enqueue, finalize, destroy, lambda: None, exchange)

        with ThreadPoolExecutor(max_workers=len(accepted)) as pool:
            return list(pool.map(rank_run, range(len(accepted))))

    def test_equal_counts_require_success_from_both_close_calls(self):
        for codes in ((0, 0), (6, 0), (0, 6), (6, 6)):
            with self.subTest(codes=codes):
                entries = self.run_ranks([2, 2, 2, 2], codes)
                self.assertTrue(all(entry["ok"] == (codes == (0, 0)) for entry in entries))

    def test_unequal_counts_allow_only_success_or_remote_error(self):
        for codes in ((0, 0), (6, 0), (0, 6), (6, 6), (2, 6), (6, 5)):
            with self.subTest(codes=codes):
                entries = self.run_ranks([3, 19, 2, 4], codes)
                self.assertTrue(all(entry["accepted_counts"] == [3, 19, 2, 4] for entry in entries))
                self.assertTrue(all(entry["ok"] == all(code in (0, 6) for code in codes)
                                    for entry in entries))

    def test_post_close_calls_must_be_refused(self):
        for result in (0, 4, 5, 6):
            with self.subTest(result=result):
                entries = self.run_ranks([2, 2, 2, 2], (0, 0), after_close=result)
                self.assertTrue(all(entry["ok"] == (result in (4, 5)) for entry in entries))

    def test_worker_error_is_not_excused_by_unequal_counts(self):
        entries = self.run_ranks([3, 19, 2, 4], (6, 6), worker_code=6)
        self.assertTrue(all(not entry["ok"] for entry in entries))

    def test_first_call_must_succeed_before_close_starts(self):
        entry = race.close_concurrently(lambda: 5, lambda: 0, lambda: 0, lambda: None, lambda _: [0, 0])
        self.assertFalse(entry["ok"])

    def test_worker_exception_fails_even_if_close_succeeds(self):
        def prepare():
            raise RuntimeError("thread setup failed")

        entry = race.close_concurrently(lambda: 4, lambda: 0, lambda: 0, prepare, lambda _: [0, 0])
        self.assertFalse(entry["ok"])
        self.assertEqual(entry["worker_errors"], ["RuntimeError: thread setup failed"])

    def test_stuck_worker_is_a_failure(self):
        release = threading.Event()
        exited = threading.Event()
        calls = 0

        def enqueue():
            nonlocal calls
            calls += 1
            if calls == 1:
                return 0
            release.wait(5)
            exited.set()
            return 5

        try:
            with self.assertRaisesRegex(TimeoutError, "did not exit"):
                race.close_concurrently(enqueue, lambda: 0, lambda: 0, lambda: None,
                                        lambda _: self.fail("must not exchange unstable counts"), timeout=0.01)
        finally:
            release.set()
            self.assertTrue(exited.wait(5))

    def test_failed_process_cannot_pass_with_successful_json(self):
        class Process:
            pid = 99999999

            def __init__(self, timeout):
                self.timeout = timeout
                self.killed = False

            def wait(self, timeout=None):
                if self.timeout and not self.killed:
                    raise subprocess.TimeoutExpired("rank", timeout)
                return -9 if self.killed else 1

            def kill(self):
                self.killed = True

        for timeout in (False, True):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as work:
                Path(work, "rank0.json").write_text(json.dumps({
                    "rounds": [{"round": 0, "ok": True}],
                    "parent": {"exact": True, "async_error": 0}, "parent_destroy": 0,
                }))
                process = Process(timeout)
                with patch.object(race.subprocess, "Popen", return_value=process), \
                        contextlib.redirect_stdout(io.StringIO()) as output:
                    code = race.main(["--library", "unused.so", "--world", "1", "--rounds", "1",
                                      "--expect", "concurrent-close", "--work", work])
                self.assertEqual(code, 1)
                self.assertIn("rank 0: exit", output.getvalue())
                self.assertEqual(process.killed, timeout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
