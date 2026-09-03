"""Behavior tests for the opt-in live graph-status reporter."""

from __future__ import annotations

import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import spark_graph_status_reporter
import spark_collective_audit


class GraphStatusReporterTest(unittest.TestCase):
    def setUp(self) -> None:
        spark_collective_audit._reset_for_tests()

    def tearDown(self) -> None:
        spark_graph_status_reporter.stop_status_reporter()
        spark_collective_audit._reset_for_tests()

    def test_periodically_publishes_an_atomic_rank_status_envelope(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank2.json"
            sequence = 0

            def snapshot() -> dict[str, object]:
                nonlocal sequence
                sequence += 1
                return {"sequence": sequence}

            reporter = spark_graph_status_reporter.start_status_reporter(
                path,
                snapshot_provider=snapshot,
                interval_seconds=0.01,
                rank=2,
            )

            deadline = time.monotonic() + 2.0
            payload: dict[str, object] = {}
            while time.monotonic() < deadline:
                if path.exists():
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if payload["snapshot"]["sequence"] >= 2:
                        break
                time.sleep(0.01)
            reporter.stop()

            self.assertEqual(payload["schema_version"], 3)
            self.assertEqual(payload["rank"], 2)
            self.assertEqual(payload["pid"], os.getpid())
            self.assertIsInstance(payload["unix_ns"], int)
            self.assertLessEqual(
                payload["snapshot_start_unix_ns"],
                payload["snapshot_end_unix_ns"],
            )
            self.assertEqual(
                payload["unix_ns"],
                payload["snapshot_end_unix_ns"],
            )
            self.assertGreaterEqual(payload["snapshot"]["sequence"], 2)
            self.assertEqual(list(path.parent.glob("*.tmp")), [])

    def test_collects_retained_transport_families_in_one_snapshot(
        self,
    ) -> None:
        all_reduce = types.ModuleType("spark_tp4_backend")
        all_reduce.graph_q1_diagnostic_snapshot = lambda: {
            "sessions": {"2": {"completed_sequence": 9}},
            "events": {"captured_nodes": 169},
        }
        vocabulary = types.ModuleType(
            "spark_tp4_vocab_allgather_backend"
        )
        vocabulary.vocab_graph_diagnostic_snapshot = lambda: {
            "sessions": {"2": {"completed_sequence": 5}},
            "events": {"captured_nodes": 5},
        }
        with patch.dict(
            "sys.modules",
            {
                "spark_tp4_backend": all_reduce,
                "spark_tp4_vocab_allgather_backend": vocabulary,
            },
        ):
            snapshot = (
                spark_graph_status_reporter.collect_graph_status()
            )

        self.assertEqual(
            snapshot,
            {
                "all_reduce": {
                    "sessions": {
                        "2": {"completed_sequence": 9}
                    },
                    "events": {"captured_nodes": 169},
                },
                "vocabulary": {
                    "sessions": {
                        "2": {"completed_sequence": 5}
                    },
                    "events": {"captured_nodes": 5},
                },
                "stock_collectives": {
                    "capture": {},
                    "eager": {},
                    "capture_total": 0,
                    "eager_total": 0,
                },
            },
        )

    def test_replay_timing_is_retained_only_when_enabled(self) -> None:
        replay_timing = types.ModuleType("spark_cudagraph_replay_timing")
        replay_timing.graph_replay_timing_snapshot = lambda: {
            "enabled": True,
            "completed": 17,
        }
        with patch.dict(
            os.environ,
            {
                "SPARK_CUDAGRAPH_REPLAY_TIMING": "1",
                "SPARK_Q2R_PROBE": "1",
                "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "1",
                "SPARK_ADAPTIVE_MTP_CONTROL": "1",
            },
            clear=False,
        ), patch.dict(
            "sys.modules",
            {"spark_cudagraph_replay_timing": replay_timing},
        ):
            enabled_snapshot = (
                spark_graph_status_reporter.collect_graph_status()
            )

        self.assertEqual(
            enabled_snapshot["cudagraph_replay_timing"],
            {"enabled": True, "completed": 17},
        )

        with patch.dict(os.environ, {}, clear=True):
            disabled_snapshot = (
                spark_graph_status_reporter.collect_graph_status()
            )
        self.assertNotIn(
            "cudagraph_replay_timing",
            disabled_snapshot,
        )

    def test_opt_in_process_reporter_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank1.json"
            with patch.dict(
                os.environ,
                {"SPARK_TP4_GRAPH_STATUS_PATH": str(path)},
                clear=True,
            ):
                first = (
                    spark_graph_status_reporter.ensure_status_reporter(
                        rank=1,
                        interval_seconds=0.01,
                    )
                )
                second = (
                    spark_graph_status_reporter.ensure_status_reporter(
                        rank=1,
                        interval_seconds=0.01,
                    )
                )

            self.assertIs(first, second)
            deadline = time.monotonic() + 2.0
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(path.exists())
            # The reporter owns this path. Stop and join it before the
            # TemporaryDirectory removes that path; class tearDown runs
            # after the context manager and is therefore too late on Linux.
            spark_graph_status_reporter.stop_status_reporter()


    def test_transient_snapshot_failure_does_not_kill_reporter(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rank3.json"
            calls = 0

            def snapshot() -> dict[str, object]:
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise RuntimeError("not ready")
                return {"ready": True}

            reporter = spark_graph_status_reporter.start_status_reporter(
                path,
                snapshot_provider=snapshot,
                interval_seconds=0.01,
                rank=3,
            )
            deadline = time.monotonic() + 2.0
            while not path.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            reporter.stop()

            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertTrue(payload["snapshot"]["ready"])
            self.assertGreaterEqual(calls, 2)


if __name__ == "__main__":
    unittest.main()
