"""Stable phase timing for one asynchronous SparkCache restore.

The restore data path spans filesystem verification, CPU reconstruction, and
GPU placement. A single elapsed time cannot distinguish those resources, so
this module defines a small machine-readable record without importing torch or
vLLM. Timing failures must never affect restore correctness.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

RESTORE_TIMING_PREFIX = "spark-context-cache-restore-timing:"
RESTORE_PHASES = (
    "metadata_preparation",
    "manifest_lookup",
    "prior_cuda_work",
    "restore_read",
    "reassembly_decode",
    "h2d_submit",
    "cuda_sync",
)


@dataclass
class RestoreTiming:
    request_id: str
    digest: str
    span_tokens: int
    storage_mode: str
    enqueued_ns: int
    service_started_ns: int = 0
    service_finished_ns: int = 0
    outcome: str = "pending"
    chunk_count: int = 0
    page_bytes: int = 0
    phase_ns: dict[str, int] = field(
        default_factory=lambda: {phase: 0 for phase in RESTORE_PHASES}
    )
    _observed_phases: set[str] = field(default_factory=set, repr=False)

    def start_service(self, at_ns: int | None = None) -> None:
        if self.service_started_ns:
            raise RuntimeError("restore service timing already started")
        self.service_started_ns = time.perf_counter_ns() if at_ns is None else at_ns

    def add(self, phase: str, elapsed_ns: int) -> None:
        if phase not in self.phase_ns:
            raise ValueError(f"unknown restore phase {phase!r}")
        if phase in self._observed_phases:
            raise ValueError(f"restore phase {phase!r} was already recorded")
        if isinstance(elapsed_ns, bool) or not isinstance(elapsed_ns, int):
            raise TypeError("restore phase duration must be an integer")
        self.phase_ns[phase] = max(0, elapsed_ns)
        self._observed_phases.add(phase)

    def observe(self, phase: str, elapsed_ns: int) -> None:
        """Record optional timing without allowing telemetry to affect restore."""

        try:
            self.add(phase, elapsed_ns)
        except (RuntimeError, TypeError, ValueError):
            return

    def finish(self, outcome: str, at_ns: int | None = None) -> None:
        if outcome not in {"verified", "recompute"}:
            raise ValueError("restore timing outcome is invalid")
        if not self.service_started_ns:
            raise RuntimeError("restore service timing has not started")
        if self.service_finished_ns:
            raise RuntimeError("restore service timing already finished")
        finished = time.perf_counter_ns() if at_ns is None else at_ns
        self.service_finished_ns = max(self.service_started_ns, finished)
        self.outcome = outcome

    @property
    def queue_wait_ns(self) -> int:
        if not self.service_started_ns:
            return 0
        return max(0, self.service_started_ns - self.enqueued_ns)

    @property
    def service_ns(self) -> int:
        if not self.service_finished_ns:
            return 0
        return max(0, self.service_finished_ns - self.service_started_ns)

    @property
    def end_to_end_ns(self) -> int:
        if not self.service_finished_ns:
            return 0
        return max(0, self.service_finished_ns - self.enqueued_ns)

    def render(self) -> str:
        if self.outcome == "pending":
            raise RuntimeError("cannot render unfinished restore timing")

        def milliseconds(value: int) -> float:
            return round(value / 1_000_000, 3)

        record = {
            "schema": "sparkcache-restore-timing/v1",
            "request_id": self.request_id,
            "digest": self.digest[:12],
            "span_tokens": self.span_tokens,
            "storage_mode": self.storage_mode,
            "outcome": self.outcome,
            "chunk_count": self.chunk_count,
            "page_bytes": self.page_bytes,
            "queue_wait_ms": milliseconds(self.queue_wait_ns),
            "service_ms": milliseconds(self.service_ns),
            "end_to_end_ms": milliseconds(self.end_to_end_ns),
            "phase_ms": {
                phase: milliseconds(self.phase_ns[phase])
                for phase in RESTORE_PHASES
            },
        }
        return RESTORE_TIMING_PREFIX + json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
        )

    def operator_lines(self) -> tuple[str, str]:
        """Render two compact INFO lines for an operator-facing restore event."""

        if self.outcome == "pending":
            raise RuntimeError("cannot summarize unfinished restore timing")

        total_ms = self.end_to_end_ns / 1_000_000
        token_rate = (
            self.span_tokens * 1_000_000_000 / self.end_to_end_ns
            if self.end_to_end_ns
            else 0.0
        )
        if token_rate >= 1_000_000:
            rate = f"{token_rate / 1_000_000:.2f}M"
        elif token_rate >= 1_000:
            rate = f"{token_rate / 1_000:.0f}K"
        else:
            rate = f"{token_rate:.0f}"
        payload = (
            f" bytes={self.page_bytes / 1024**2:.1f}MiB"
            if self.page_bytes
            else ""
        )
        result = (
            f"sparkcache: restore tokens={self.span_tokens} total={total_ms:.1f}ms"
            f" rate={rate} tok/s{payload}"
        )
        phases = (
            "sparkcache: phases"
            f" read={self.phase_ns['restore_read'] / 1_000_000:.1f}ms"
            f" place={self.phase_ns['h2d_submit'] / 1_000_000:.1f}ms"
            f" sync={self.phase_ns['cuda_sync'] / 1_000_000:.1f}ms"
            f" queue={self.queue_wait_ns / 1_000_000:.1f}ms"
        )
        return result, phases
