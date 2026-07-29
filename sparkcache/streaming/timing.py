"""Optional, fail-open timing trace for the final streaming snapshot batch."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field


TIMING_PREFIX = "[sparkcache-streaming-timing] "

STAGES = (
    "final_watermark",
    "ring_submit_begin",
    "ring_submit_end",
    "fence_ready_observed",
    "claim_begin",
    "claim_end",
    "writer_enqueued",
    "writer_start",
    "writer_end",
    "writer_completion_polled",
    "manifest_publish_begin",
    "manifest_publish_end",
    "runtime_commit",
    "adapter_observed",
)
_STAGE_SET = frozenset(STAGES)

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _TraceRecord:
    span_tokens: int
    marks_ns: dict[str, int] = field(default_factory=dict)
    emitted: bool = False


class StreamingTimingTrace:
    """Record one compact final-batch timeline without affecting serving."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        sink: Callable[[str], object] | None = None,
        clock_ns: Callable[[], int] | None = None,
    ) -> None:
        self._enabled = enabled
        self._sink = sink if sink is not None else _LOGGER.info
        self._clock_ns = clock_ns if clock_ns is not None else time.monotonic_ns
        self._lock = threading.Lock()
        self._final_by_request: dict[str, tuple[int, _TraceRecord]] = {}

    def register_final(
        self,
        request_id: str,
        batch_index: int,
        span_tokens: int,
    ) -> None:
        """Register the one batch eligible to produce a trace for a request."""

        if not self._enabled:
            return
        try:
            request_key = str(request_id)
            batch_key = int(batch_index)
            span_key = int(span_tokens)
            with self._lock:
                existing = self._final_by_request.get(request_key)
                if (
                    existing is not None
                    and existing[0] == batch_key
                    and existing[1].span_tokens == span_key
                ):
                    return
                self._final_by_request[request_key] = (
                    batch_key,
                    _TraceRecord(span_tokens=span_key),
                )
        except Exception:
            return

    def mark(
        self,
        request_id: str,
        batch_index: int,
        stage: str,
        *,
        at_ns: int | None = None,
    ) -> None:
        """Mark a stage; the registered final batch emits at adapter observation."""

        if not self._enabled:
            return
        try:
            if stage not in _STAGE_SET:
                return
            timestamp_ns = self._clock_ns() if at_ns is None else int(at_ns)
            line: str | None = None
            request_key = str(request_id)
            batch_key = int(batch_index)
            with self._lock:
                registered = self._final_by_request.get(request_key)
                if registered is None or registered[0] != batch_key:
                    return
                record = registered[1]
                if record.emitted:
                    return
                record.marks_ns.setdefault(stage, timestamp_ns)
                if stage == "adapter_observed":
                    record.emitted = True
                    line = self._render(
                        request_key,
                        batch_key,
                        record,
                    )
                    self._final_by_request.pop(request_key, None)
            if line is not None:
                try:
                    self._sink(line)
                except Exception:
                    return
        except Exception:
            return

    def mark_final(
        self,
        request_id: str,
        stage: str,
        *,
        at_ns: int | None = None,
    ) -> None:
        """Mark the scheduler-registered final batch without re-deriving its key."""

        if not self._enabled:
            return
        try:
            request_key = str(request_id)
            with self._lock:
                registered = self._final_by_request.get(request_key)
                if registered is None:
                    return
                batch_index = registered[0]
            self.mark(
                request_key,
                batch_index,
                stage,
                at_ns=at_ns,
            )
        except Exception:
            return

    @staticmethod
    def _render(
        request_id: str,
        batch_index: int,
        record: _TraceRecord,
    ) -> str:
        ordered_marks = [
            (stage, record.marks_ns[stage])
            for stage in STAGES
            if stage in record.marks_ns
        ]
        base_ns = ordered_marks[0][1]
        effective_ns = base_ns
        stage_delta_ns: dict[str, int] = {}
        for stage, marked_ns in ordered_marks:
            effective_ns = max(effective_ns, marked_ns)
            stage_delta_ns[stage] = effective_ns - base_ns
        total_ns = stage_delta_ns.get("adapter_observed", 0)
        payload = {
            "batch_index": batch_index,
            "request_id": request_id,
            "span_tokens": record.span_tokens,
            "stage_delta_ns": stage_delta_ns,
            "total_ns": total_ns,
        }
        return TIMING_PREFIX + json.dumps(
            payload,
            separators=(",", ":"),
        )
