"""Low-rate atomic status files for live graph-native transport gates."""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

SnapshotProvider = Callable[[], dict[str, Any]]
_process_reporter: GraphStatusReporter | None = None
_process_reporter_lock = threading.Lock()


def collect_graph_status() -> dict[str, object]:
    from spark_collective_audit import stock_collective_snapshot
    from spark_tp4_backend import graph_q1_diagnostic_snapshot
    from spark_tp4_vocab_allgather_backend import (
        vocab_graph_diagnostic_snapshot,
    )

    snapshot: dict[str, object] = {
        "all_reduce": graph_q1_diagnostic_snapshot(),
        "vocabulary": vocab_graph_diagnostic_snapshot(),
        "stock_collectives": stock_collective_snapshot(),
    }
    if os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING") == "1":
        from spark_cudagraph_replay_timing import (
            graph_replay_timing_snapshot,
        )

        snapshot["cudagraph_replay_timing"] = (
            graph_replay_timing_snapshot()
        )
    return snapshot


class GraphStatusReporter:
    def __init__(
        self,
        path: Path,
        snapshot_provider: SnapshotProvider,
        interval_seconds: float,
        rank: int,
    ) -> None:
        self._path = path
        self._snapshot_provider = snapshot_provider
        self._interval_seconds = interval_seconds
        self._rank = rank
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"spark-graph-status-rank{rank}",
            daemon=True,
        )
        self._thread.start()

    def _publish(self) -> None:
        snapshot_start_unix_ns = time.time_ns()
        snapshot = self._snapshot_provider()
        snapshot_end_unix_ns = time.time_ns()
        payload = {
            "schema_version": 3,
            # Kept as the compatibility timestamp; v3 consumers should use
            # the explicit collection interval to bind before/after deltas.
            "unix_ns": snapshot_end_unix_ns,
            "snapshot_start_unix_ns": snapshot_start_unix_ns,
            "snapshot_end_unix_ns": snapshot_end_unix_ns,
            "pid": os.getpid(),
            "rank": self._rank,
            "snapshot": snapshot,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":")
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_name(
            f".{self._path.name}.{os.getpid()}.tmp"
        )
        temporary.write_text(encoded + "\n", encoding="utf-8")
        deadline = time.monotonic() + 0.25
        while True:
            try:
                os.replace(temporary, self._path)
                break
            except PermissionError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.001)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._publish()
            except Exception:
                pass
            self._stop.wait(self._interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self._interval_seconds * 2))


def start_status_reporter(
    path: str | Path,
    *,
    snapshot_provider: SnapshotProvider,
    interval_seconds: float = 0.25,
    rank: int,
) -> GraphStatusReporter:
    return GraphStatusReporter(
        Path(path), snapshot_provider, interval_seconds, rank
    )


def ensure_status_reporter(
    *,
    rank: int,
    interval_seconds: float = 0.25,
) -> GraphStatusReporter | None:
    path = os.getenv("SPARK_TP4_GRAPH_STATUS_PATH")
    if not path:
        return None
    global _process_reporter
    with _process_reporter_lock:
        if _process_reporter is None:
            _process_reporter = start_status_reporter(
                path,
                snapshot_provider=collect_graph_status,
                interval_seconds=interval_seconds,
                rank=rank,
            )
        return _process_reporter


def stop_status_reporter() -> None:
    global _process_reporter
    with _process_reporter_lock:
        reporter = _process_reporter
        _process_reporter = None
    if reporter is not None:
        reporter.stop()
