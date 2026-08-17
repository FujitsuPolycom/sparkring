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

_EXPECTED_VERSION = (
    "0.11.2.dev279+eldritch.final.fcc6141.b12x284a2ea."
    "fi25dd814.cu132.20260626"
)
_EXPECTED_RUN_FULLGRAPH_SHA256 = (
    "4d58b8ef1a5023af0c11eb7a659620faca15f8a0303b37774ed0d28f4a5919db"
)
_DEFAULT_SAMPLE_LIMIT = 512
_MAXIMUM_SAMPLE_LIMIT = 65536
_installed = False
_collector: ReplayTimingCollector | None = None


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
            while self._pending:
                sample = self._pending[0]
                try:
                    if not sample.end.query():
                        break
                    elapsed_ms = float(
                        sample.start.elapsed_time(sample.end)
                    )
                except Exception:
                    self._errors += 1
                    self._pending.popleft()
                    continue
                self._pending.popleft()
                self._durations[sample.key].append(elapsed_ms)

            descriptors: dict[str, dict[str, float | int]] = {}
            total_ms = 0.0
            completed = 0
            for key, values in sorted(self._durations.items()):
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
                "reserved": self._reserved,
                "completed": completed,
                "pending": len(self._pending),
                "dropped": self._dropped,
                "errors": self._errors,
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
    return (
        f"mode={mode},num_tokens={int(descriptor.num_tokens)},"
        f"num_reqs={int(descriptor.num_reqs)},"
        f"num_active_loras={int(descriptor.num_active_loras)}"
    )


def install() -> None:
    global _collector, _installed
    if _installed:
        return

    import torch
    import vllm
    from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

    if vllm.__version__ != _EXPECTED_VERSION:
        raise RuntimeError(
            "unsupported vLLM version for graph replay timing: "
            f"{vllm.__version__}"
        )
    original = CudaGraphManager.run_fullgraph
    actual_hash = hashlib.sha256(
        inspect.getsource(original).encode("utf-8")
    ).hexdigest()
    if actual_hash != _EXPECTED_RUN_FULLGRAPH_SHA256:
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
