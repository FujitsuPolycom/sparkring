"""Arm-file-gated CUDA timing for vLLM FULL graph replays.

The serving thread only records CUDA events. Completion polling and elapsed
time calculation happen from the existing low-rate graph status reporter, so
the measured request is not host-synchronized by this diagnostic.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import os
import threading
from collections import defaultdict, deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SUPPORTED_RUN_FULLGRAPH_SHA256 = {
    (
        "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
        "fi25dd814.cu132.20260626"
    ): {
        "4d58b8ef1a5023af0c11eb7a659620faca15f8a0303b37774ed0d28f4a5919db"
    },
    # Public GLM-5.3 Flash operator image. Keep the source hash gate: the
    # version string alone does not prove that replay still has the semantics
    # this diagnostic wraps.
    "0.1.dev1+gd377796e8": {
        "af0da51c4ca27ed318af8d50baaaec3074de33820b2f1be4a6b75de1510a9bb3"
    },
}
_DEFAULT_SAMPLE_LIMIT = 512
_MAXIMUM_SAMPLE_LIMIT = 65536
_installed = False
_collector: ReplayTimingCollector | None = None
_timing_reporter: Any | None = None


@dataclass(frozen=True)
class _PendingSample:
    key: str
    start: Any
    end: Any


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


class ReplayTimingCollector:
    def __init__(
        self,
        *,
        event_factory: Callable[[], Any],
        arm_path: Path,
        sample_limit: int,
    ) -> None:
        if not 1 <= sample_limit <= _MAXIMUM_SAMPLE_LIMIT:
            raise ValueError(
                "sample_limit must be in "
                f"[1, {_MAXIMUM_SAMPLE_LIMIT}]"
            )
        self._event_factory = event_factory
        self._arm_path = arm_path
        self._sample_limit = sample_limit
        self._lock = threading.Lock()
        self._pending: deque[_PendingSample] = deque()
        self._durations: dict[str, list[float]] = defaultdict(list)
        self._reserved = 0
        self._dropped = 0
        self._errors = 0

    def measure(
        self,
        key: str,
        stream: Any,
        operation: Callable[[], Any],
    ) -> Any:
        if not self._arm_path.is_file():
            return operation()

        with self._lock:
            if self._reserved >= self._sample_limit:
                self._dropped += 1
                return operation()
            self._reserved += 1

        start = self._event_factory()
        end = self._event_factory()
        start.record(stream)
        try:
            return operation()
        finally:
            end.record(stream)
            with self._lock:
                self._pending.append(
                    _PendingSample(key=key, start=start, end=end)
                )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()

        completed_samples: list[tuple[str, float]] = []
        remaining: list[_PendingSample] = []
        errors = 0
        for index, sample in enumerate(pending):
            try:
                if not sample.end.query():
                    remaining = pending[index:]
                    break
                completed_samples.append(
                    (
                        sample.key,
                        float(sample.start.elapsed_time(sample.end)),
                    )
                )
            except Exception:
                errors += 1

        with self._lock:
            if remaining:
                self._pending = deque(remaining) + self._pending
            for key, elapsed_ms in completed_samples:
                self._durations[key].append(elapsed_ms)
            self._errors += errors
            durations = {
                key: list(values)
                for key, values in self._durations.items()
            }
            reserved = self._reserved
            pending_count = len(self._pending)
            dropped = self._dropped
            error_count = self._errors

        descriptors: dict[str, dict[str, float | int]] = {}
        total_ms = 0.0
        completed = 0
        for key, values in sorted(durations.items()):
            count = len(values)
            subtotal = sum(values)
            completed += count
            total_ms += subtotal
            descriptors[key] = {
                "count": count,
                "total_ms": round(subtotal, 6),
                "mean_ms": round(subtotal / count, 6),
                "p50_ms": round(_percentile(values, 0.50), 6),
                "p90_ms": round(_percentile(values, 0.90), 6),
                "max_ms": round(max(values), 6),
            }

        return {
            "enabled": True,
            "armed": self._arm_path.is_file(),
            "arm_path": str(self._arm_path),
            "sample_limit": self._sample_limit,
            "reserved": reserved,
            "completed": completed,
            "pending": pending_count,
            "dropped": dropped,
            "errors": error_count,
            "total_completed_ms": round(total_ms, 6),
            "descriptors": descriptors,
        }


def _sample_limit() -> int:
    text = os.getenv(
        "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES",
        str(_DEFAULT_SAMPLE_LIMIT),
    )
    try:
        value = int(text)
    except ValueError as error:
        raise RuntimeError(
            "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES must be an integer"
        ) from error
    if not 1 <= value <= _MAXIMUM_SAMPLE_LIMIT:
        raise RuntimeError(
            "SPARK_CUDAGRAPH_REPLAY_TIMING_SAMPLES must be in "
            f"[1, {_MAXIMUM_SAMPLE_LIMIT}]"
        )
    return value


def _arm_path() -> Path:
    value = os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING_ARM_PATH")
    if not value:
        raise RuntimeError(
            "SPARK_CUDAGRAPH_REPLAY_TIMING_ARM_PATH is required"
        )
    return Path(value)


def _descriptor_key(descriptor: Any) -> str:
    mode = getattr(getattr(descriptor, "cg_mode", None), "name", "unknown")
    fields = (
        ("num_tokens", getattr(descriptor, "num_tokens", None)),
        ("num_reqs", getattr(descriptor, "num_reqs", None)),
        (
            "uniform_token_count",
            getattr(descriptor, "uniform_token_count", None),
        ),
        ("max_query_len", getattr(descriptor, "max_query_len", None)),
        (
            "num_active_loras",
            getattr(descriptor, "num_active_loras", 0),
        ),
    )
    encoded = ",".join(
        f"{name}={'none' if value is None else int(value)}"
        for name, value in fields
    )
    return f"mode={mode},{encoded}"


def install() -> None:
    global _collector, _installed, _timing_reporter
    if _installed:
        return

    import torch
    import vllm
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    expected_hashes = _SUPPORTED_RUN_FULLGRAPH_SHA256.get(vllm.__version__)
    if expected_hashes is None:
        raise RuntimeError(
            "unsupported vLLM version for graph replay timing: "
            f"{vllm.__version__}"
        )
    original = CudaGraphManager.run_fullgraph
    actual_hash = hashlib.sha256(
        inspect.getsource(original).encode("utf-8")
    ).hexdigest()
    if actual_hash not in expected_hashes:
        raise RuntimeError(
            "unsupported CudaGraphManager.run_fullgraph source: "
            f"{actual_hash}"
        )
    if getattr(original, "_spark_replay_timing", False):
        _installed = True
        return

    _collector = ReplayTimingCollector(
        event_factory=lambda: torch.cuda.Event(enable_timing=True),
        arm_path=_arm_path(),
        sample_limit=_sample_limit(),
    )

    status_path = os.getenv("SPARK_CUDAGRAPH_REPLAY_TIMING_STATUS_PATH")
    if status_path:
        try:
            rank = int(os.environ["SPARKRING_NODE_RANK"])
        except (KeyError, ValueError) as error:
            raise RuntimeError(
                "SPARKRING_NODE_RANK must be an integer when the replay "
                "timing status path is enabled"
            ) from error
        from spark_graph_status_reporter import start_status_reporter

        _timing_reporter = start_status_reporter(
            status_path,
            snapshot_provider=lambda: {
                "cudagraph_replay_timing": graph_replay_timing_snapshot()
            },
            interval_seconds=0.25,
            rank=rank,
        )

    def timed_run_fullgraph(self: Any, descriptor: Any) -> Any:
        assert _collector is not None
        stream = torch.cuda.current_stream(self.device)
        return _collector.measure(
            _descriptor_key(descriptor),
            stream,
            lambda: original(self, descriptor),
        )

    timed_run_fullgraph._spark_replay_timing = True  # type: ignore[attr-defined]
    timed_run_fullgraph._spark_original = original  # type: ignore[attr-defined]
    CudaGraphManager.run_fullgraph = timed_run_fullgraph
    _installed = True


def graph_replay_timing_snapshot() -> dict[str, Any]:
    if _collector is None:
        return {"enabled": False}
    return _collector.snapshot()
