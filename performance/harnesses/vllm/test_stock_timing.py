"""Tests for the semantics-preserving fixed-K4 stock timing recorder."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from . import stock_timing as timing


class _Stream:
    def __init__(self, cuda_stream: int) -> None:
        self.cuda_stream = cuda_stream


_STREAM = _Stream(1234)
_OTHER_STREAM = _Stream(5678)


class _Event:
    _clock = 0.0

    def __init__(self, *, enable_timing: bool) -> None:
        assert enable_timing
        self.timestamp = 0.0
        self.synchronized = False

    def record(self, stream: object) -> None:
        assert isinstance(stream, _Stream)
        self.timestamp = _Event._clock
        _Event._clock += 0.25

    def synchronize(self) -> None:
        self.synchronized = True

    def elapsed_time(self, other: "_Event") -> float:
        return other.timestamp - self.timestamp


class _Cuda:
    Event = _Event


class _Torch:
    cuda = _Cuda()


_ENV = {
    "SPARK_TP4_STOCK_TIMING": "1",
    "SPARK_TP4_STOCK_TIMING_ARM_PATH": "/tmp/arm",
    "RANK": "0",
}


class StockTimingTests(unittest.TestCase):
    def setUp(self) -> None:
        timing.reset_for_test()
        _Event._clock = 0.0

    def _observe_startup_q3(self) -> None:
        with patch.object(timing.Path, "read_text", side_effect=OSError):
            timing.time_original("query", 3, _STREAM, lambda: None, _Torch)

    def _arm_and_sample(
        self,
        family: str = "query",
        q: int = 5,
        *,
        stream: _Stream = _STREAM,
        run_id: str = "run-1",
    ) -> None:
        with patch.object(timing.Path, "read_text", return_value=run_id):
            timing.time_original(family, q, stream, lambda: None, _Torch)

    def test_disabled_path_only_calls_original(self) -> None:
        calls = []
        with patch.dict(os.environ, {}, clear=True):
            result = timing.time_original(
                "query", 5, _STREAM, lambda: calls.append(1) or "result", _Torch
            )
        self.assertEqual(result, "result")
        self.assertEqual(calls, [1])
        self.assertFalse(timing.snapshot_for_test()["armed"])

    def test_q3_only_marks_startup_and_does_not_arm_or_sample(self) -> None:
        with patch.dict(os.environ, _ENV, clear=True):
            self._observe_startup_q3()
        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["startup_q3_seen"])
        self.assertFalse(snapshot["armed"])
        self.assertEqual(sum(snapshot["counts"].values()), 0)

    def test_operator_arm_before_startup_q3_invalidates(self) -> None:
        with (
            patch.dict(os.environ, _ENV, clear=True),
            patch.object(timing.Path, "read_text", return_value="early-run"),
        ):
            timing.time_original("query", 3, _STREAM, lambda: None, _Torch)
        snapshot = timing.snapshot_for_test()
        self.assertFalse(snapshot["armed"])
        self.assertTrue(snapshot["invalid"])
        self.assertIn(
            "operator_arm_before_startup_q3", snapshot["invalid_reasons"]
        )

    def test_post_startup_marker_arms_and_samples(self) -> None:
        with patch.dict(os.environ, _ENV, clear=True):
            self._observe_startup_q3()
            self._arm_and_sample()
        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["armed"])
        self.assertEqual(snapshot["run_id"], "run-1")
        self.assertEqual(snapshot["counts"][("query", 5)], 1)

    def test_unexpected_q_invalidates_after_arm(self) -> None:
        with patch.dict(os.environ, _ENV, clear=True):
            self._observe_startup_q3()
            self._arm_and_sample()
            with patch.object(timing.Path, "read_text", return_value="run-1"):
                timing.time_original(
                    "query", 2, _STREAM, lambda: None, _Torch
                )
        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["invalid"])
        self.assertIn("unexpected_q2", snapshot["invalid_reasons"])

    def test_stream_change_invalidates(self) -> None:
        with patch.dict(os.environ, _ENV, clear=True):
            self._observe_startup_q3()
            self._arm_and_sample()
            self._arm_and_sample(stream=_OTHER_STREAM)
        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["invalid"])
        self.assertIn("stream_changed", snapshot["invalid_reasons"])

    def test_run_id_change_or_removal_invalidates(self) -> None:
        with patch.dict(os.environ, _ENV, clear=True):
            self._observe_startup_q3()
            self._arm_and_sample()
            with patch.object(timing.Path, "read_text", return_value="run-2"):
                timing.time_original(
                    "query", 5, _STREAM, lambda: None, _Torch
                )
        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["invalid"])
        self.assertIn(
            "run_id_changed_or_removed", snapshot["invalid_reasons"]
        )

    def test_bucket_overflow_invalidates_cross_round_inventory(self) -> None:
        with patch.dict(os.environ, _ENV, clear=True):
            self._observe_startup_q3()
            expected_calls = timing._EXPECTED[("vocab", 5)][0]
            for _ in range(expected_calls + 1):
                self._arm_and_sample("vocab", 5)
        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["invalid"])
        self.assertEqual(snapshot["overflow"][("vocab", 5)], 1)

    def test_exact_fixed_k4_inventory_reports_once(self) -> None:
        calls = 0

        def operation() -> None:
            nonlocal calls
            calls += 1

        with (
            patch.dict(os.environ, _ENV, clear=True),
            self.assertLogs(timing.__name__, level="WARNING") as logs,
        ):
            with patch.object(timing.Path, "read_text", side_effect=OSError):
                timing.time_original(
                    "query", 3, _STREAM, operation, _Torch
                )
            with patch.object(
                timing.Path, "read_text", return_value="exact-run"
            ):
                for (family, q), (expected_calls, _) in timing._EXPECTED.items():
                    for _ in range(expected_calls):
                        timing.time_original(
                            family, q, _STREAM, operation, _Torch
                        )
                timing.time_original(
                    "query", 5, _STREAM, operation, _Torch
                )

        snapshot = timing.snapshot_for_test()
        self.assertTrue(snapshot["reported"])
        self.assertFalse(snapshot["invalid"])
        self.assertEqual(snapshot["stream_id"], 1234)
        self.assertEqual(snapshot["run_id"], "exact-run")
        self.assertEqual(calls, 171)
        joined_logs = "\n".join(logs.output)
        self.assertIn(
            "wrapper_calls=169 logical_collectives=251", joined_logs
        )
        self.assertIn("run_id=exact-run", joined_logs)
        self.assertIn("stream=1234", joined_logs)
        self.assertIn("valid=true", joined_logs)


if __name__ == "__main__":
    unittest.main()
