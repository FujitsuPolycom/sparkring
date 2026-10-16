"""Background progress for one bounded CUDA manager-page capture ring."""

from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from sparkcache.spark_context_cache_hybrid import (
    PageLayout,
    encode_page_snapshot_header,
)
from sparkcache.streaming.manager_page_capture import (
    select_manager_page_extension_pages,
)

logger = logging.getLogger("vllm.spark_context_cache")
_CAPTURE_JOB_ID = re.compile(r"([0-9a-f]{32}):([1-9][0-9]{0,19})\Z")


def _format_token_rate(token_rate: float) -> str:
    if token_rate >= 1_000_000:
        return f"{token_rate / 1_000_000:.2f}M"
    if token_rate >= 1_000:
        return f"{token_rate / 1_000:.0f}K"
    return f"{token_rate:.0f}"


def _log_capture_observed(
    *,
    rank: int,
    digest: str,
    span_tokens: int,
    elapsed_ns: int,
    payload_bytes: int,
) -> None:
    """Report optional capture telemetry without affecting publication."""

    token_rate = (
        span_tokens * 1_000_000_000 / elapsed_ns if elapsed_ns else 0.0
    )
    try:
        logger.info(
            "sparkcache: capture rank=%d digest=%s tokens=%d observed=%.1fms"
            " rate=%s tok/s bytes=%.1fMiB",
            rank,
            digest[:12],
            span_tokens,
            elapsed_ns / 1_000_000,
            _format_token_rate(token_rate),
            payload_bytes / 1024**2,
        )
    except Exception:  # noqa: BLE001 - diagnostics never alter cache behavior
        return


@dataclass(slots=True)
class _PendingCapture:
    request_id: str
    context_sequence: int
    plan: Any
    ticket: Any
    capture_block_counts: tuple[int, ...]
    result_block_counts: tuple[int, ...]
    reused_pages_by_group: tuple[int, ...]
    submitted_ns: int
    abandoned: bool = False
    producer_ready: Any = None


@dataclass(slots=True)
class _QueuedCapture:
    plan: Any
    producer_ready: Any
    submitted_ns: int
    abandoned: bool = False


class PageSnapshotScatter:
    """Small header plus a claimed ring view with bounded read access."""

    def __init__(
        self,
        header: bytes,
        body: memoryview,
        release: Callable[[], None],
    ) -> None:
        self.header_bytes = bytes(header)
        self._body = body.toreadonly()
        self._release_callback = release
        self._released = False
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return len(self.header_bytes) + self._body.nbytes

    def read_range(self, start: int, end: int) -> bytes:
        if self._released:
            raise RuntimeError("page snapshot scatter view was released")
        if not 0 <= start <= end <= self.total_bytes:
            raise ValueError("page snapshot scatter range is invalid")
        header_length = len(self.header_bytes)
        parts = []
        if start < header_length:
            parts.append(self.header_bytes[start : min(end, header_length)])
        if end > header_length:
            body_start = max(0, start - header_length)
            body_end = end - header_length
            parts.append(self._body[body_start:body_end].tobytes())
        return b"".join(parts)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._release_callback()
            self._body.release()
            self._released = True


class PageExtensionScatter:
    """Claimed changed-page bytes plus their authenticated base geometry."""

    def __init__(
        self,
        body: memoryview,
        *,
        reused_pages_by_group: tuple[int, ...],
        result_block_counts: tuple[int, ...],
        release: Callable[[], None],
    ) -> None:
        self._body = body.toreadonly()
        self.reused_pages_by_group = reused_pages_by_group
        self.result_block_counts = result_block_counts
        self._release_callback = release
        self._released = False
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return self._body.nbytes

    def read_range(self, start: int, end: int) -> bytes:
        if self._released:
            raise RuntimeError("page extension scatter view was released")
        if not 0 <= start <= end <= self.total_bytes:
            raise ValueError("page extension scatter range is invalid")
        return self._body[start:end].tobytes()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._release_callback()
            self._body.release()
            self._released = True


class ManagerPageCaptureRuntime:
    """Poll native capture off the inference thread and hand off CPU bytes."""

    def __init__(
        self,
        connector: Any,
        *,
        ring: Any,
        progress_poll_seconds: float = 0.005,
        progress_thread_initializer: Callable[[], None],
        job_stream_factory: Callable[[Any], int] | None = None,
    ) -> None:
        if progress_poll_seconds <= 0:
            raise ValueError("progress_poll_seconds must be positive")
        layout = getattr(connector, "_page_layout", None)
        if not isinstance(layout, PageLayout):
            raise RuntimeError(
                "asynchronous manager-page capture requires a registered page layout"
            )
        self._connector = connector
        self._rank = int(connector._worker_rank())
        self._layout = layout
        self._ring = ring
        self._poll_seconds = float(progress_poll_seconds)
        self._thread_initializer = progress_thread_initializer
        self._cv = threading.Condition(threading.RLock())
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_sequence = 1
        self._pending: dict[str, _PendingCapture] = {}
        self._completed: set[str] = set()
        # vLLM supplies each finished request ID once, but permits a connector
        # to report its asynchronous completion on a later get_finished call.
        # Retain only IDs that this runtime currently owns, and retire each ID
        # with the corresponding completion notification.
        self._pending_finished_requests: set[str] = set()
        self._completed_started_ns: dict[str, int] = {}
        self._completed_manager_pages: dict[str, int] = {}
        self._closed = False
        self._fatal: BaseException | None = None
        self._queued_jobs: dict[str, _QueuedCapture] = {}
        self._submitting_jobs: dict[str, _QueuedCapture] = {}
        self._job_stream_factory = job_stream_factory
        self._submission_stream: Any = None
        self._failed_jobs: set[str] = set()
        self._draining_jobs: set[str] = set()
        self._job_epoch: str | None = None
        self._job_high_watermark = 0

    def _mark_completed_locked(
        self,
        request_id: str,
        retained_manager_pages: int,
        *,
        started_ns: int | None = None,
        capture_job_id: str = "",
    ) -> None:
        if capture_job_id:
            self._connector._capture_read_completed(capture_job_id)
            return
        self._completed.add(request_id)
        self._completed_started_ns.setdefault(
            request_id,
            time.perf_counter_ns() if started_ns is None else started_ns,
        )
        self._completed_manager_pages[request_id] = max(
            retained_manager_pages,
            self._completed_manager_pages.get(request_id, 0),
        )

    def submit(self, plan: Any, *, producer_stream: int, producer_ready: Any = None) -> bool:
        """Queue job-leased reads after a caller-recorded producer event."""
        job = getattr(plan, "capture_job_id", "")
        if not job:
            return self._submit_request(plan, producer_stream=producer_stream)
        if producer_ready is None:
            raise ValueError("job-leased capture requires a recorded producer event")
        request_id = str(plan.request_id)
        with self._cv:
            # Ownership outranks replay filtering. A lower sequence can still
            # name an active read after a higher sequence has been accepted.
            if self._job_read_is_owned_locked(job):
                if self._closed or self._fatal is not None:
                    return False
                return any(
                    getattr(capture.plan, "capture_job_id", "") == job
                    and str(capture.plan.request_id) == request_id
                    for capture in (
                        *self._queued_jobs.values(),
                        *self._submitting_jobs.values(),
                        *self._pending.values(),
                    )
                )
            match = _CAPTURE_JOB_ID.fullmatch(job) if isinstance(job, str) else None
            if match is None or int(match[2]) > 0xFFFFFFFFFFFFFFFF:
                raise ValueError("capture job must name a lowercase UUID epoch and positive uint64 sequence")
            epoch, sequence_text = match.groups()
            sequence = int(sequence_text)
            if self._closed or self._fatal is not None:
                self._connector._capture_read_completed(job)
                self._connector._abort_async_page_capture(plan.digest, "capture runtime unavailable")
                return False
            if (
                self._job_epoch is not None and epoch != self._job_epoch
            ) or sequence <= self._job_high_watermark:
                self._connector._capture_read_completed(job)
                self._connector._abort_async_page_capture(plan.digest, "capture job is outside the accepted epoch or increasing sequence")
                return False
            self._job_epoch = epoch
            self._job_high_watermark = sequence
            existing = self._queued_jobs.get(request_id) or self._submitting_jobs.get(request_id)
            if existing is None:
                existing = self._pending.get(request_id)
            if existing is not None:
                self._connector._capture_read_completed(job)
                self._connector._abort_async_page_capture(plan.digest, "request already has a capture read")
                return False
            self._queued_jobs[request_id] = _QueuedCapture(plan, producer_ready, time.perf_counter_ns())
            try:
                self._ensure_thread_locked()
            except Exception:
                self._queued_jobs.pop(request_id)
                self._thread = None
                self._connector._capture_read_completed(job)
                self._connector._abort_async_page_capture(plan.digest, "capture progress thread unavailable")
                return False
            self._wake.set()
            return True

    def _producer_stream_for_job(self, ready: Any) -> int:
        if self._job_stream_factory is not None:
            return self._job_stream_factory(ready)
        import torch
        if self._submission_stream is None:
            self._submission_stream = torch.cuda.Stream()
        self._submission_stream.wait_event(ready)
        return int(self._submission_stream.cuda_stream)

    def _submit_queued_job(self, queued: _QueuedCapture) -> None:
        """Submit and, on failure, drain without holding the callback lock."""
        plan = queued.plan
        job = plan.capture_job_id
        context_sequence = None
        entered_native = False
        try:
            with self._cv:
                if queued.abandoned or self._closed or self._fatal is not None:
                    self._connector._capture_read_completed(job)
                    self._connector._abort_async_page_capture(plan.digest, "capture cancelled before submission")
                    return
                context_sequence = self._next_sequence
                self._next_sequence += 1
            plan = self._connector._protect_capture_publication_base(plan)
            groups = self._connector._select_group_blocks_for_span(
                plan.group_block_ids, plan.span_tokens,
                recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
            )
            capture_groups = groups
            result_counts = tuple(len(group) for group in groups)
            reused = tuple(0 for _ in groups)
            logical_start = 0
            if plan.base_context_digest:
                capture_groups, _base, result_counts, reused = select_manager_page_extension_pages(
                    groups,
                    base_page_counts=self._connector._group_block_counts_for_span(plan.base_span_tokens),
                    logical_tokens_per_page=tuple(int(group["logical_tokens_per_block"]) for group in self._connector._group_topology),
                    reuse_policies=tuple(str(group["reuse_policy"]) for group in self._connector._group_topology),
                    base_boundary_tokens=plan.base_span_tokens,
                )
                logical_start = plan.base_span_tokens
            stream = self._producer_stream_for_job(queued.producer_ready)
            with self._cv:
                if queued.abandoned:
                    self._connector._capture_read_completed(job)
                    self._connector._abort_async_page_capture(plan.digest, "capture cancelled before submission")
                    return
            entered_native = True
            ticket = self._ring.submit(
                context_sequence=context_sequence, logical_start=logical_start,
                physical_pages_by_group=capture_groups, producer_stream=stream,
            )
            if ticket is None:
                self._connector._capture_read_completed(job)
                self._connector._abort_async_page_capture(plan.digest, "capture ring rejected optional work")
                return
            with self._cv:
                self._pending[str(plan.request_id)] = _PendingCapture(
                    str(plan.request_id), context_sequence, plan, ticket,
                    tuple(len(group) for group in capture_groups), result_counts,
                    reused, queued.submitted_ns, queued.abandoned, queued.producer_ready,
                )
        except Exception as error:
            retired = not entered_native
            if entered_native:
                try:
                    self._ring.drain_context(context_sequence)
                    retired = True
                except Exception as drain_error:
                    with self._cv:
                        self._failed_jobs.add(job)
                        self._fatal = drain_error
                    self._connector._capture_read_failed(job)
            if retired:
                self._connector._capture_read_completed(job)
            self._connector._abort_async_page_capture(plan.digest, f"capture submission failed: {error}")
        finally:
            with self._cv:
                self._submitting_jobs.pop(str(plan.request_id), None)
                self._cv.notify_all()

    def _submit_request(self, plan: Any, *, producer_stream: int) -> bool:
        request_id = str(plan.request_id)
        groups = self._connector._select_group_blocks_for_span(
            plan.group_block_ids,
            plan.span_tokens,
            recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
        )
        capture_groups = groups
        result_block_counts = tuple(len(group) for group in groups)
        reused_pages_by_group = tuple(0 for _ in groups)
        logical_start = 0
        if plan.base_context_digest:
            capture_groups, _base_counts, result_block_counts, reused_pages_by_group = (
                select_manager_page_extension_pages(
                    groups,
                    base_page_counts=self._connector._group_block_counts_for_span(
                        plan.base_span_tokens
                    ),
                    logical_tokens_per_page=tuple(
                        int(group["logical_tokens_per_block"])
                        for group in self._connector._group_topology
                    ),
                    reuse_policies=tuple(
                        str(group["reuse_policy"])
                        for group in self._connector._group_topology
                    ),
                    base_boundary_tokens=plan.base_span_tokens,
                )
            )
            logical_start = plan.base_span_tokens
        with self._cv:
            if self._closed:
                raise RuntimeError("manager-page capture runtime is closed")
            if self._fatal is not None:
                raise RuntimeError(
                    "manager-page capture ownership is uncertain"
                ) from self._fatal
            if request_id in self._pending:
                raise RuntimeError("request already has a manager-page capture")
            self._pending_finished_requests.discard(request_id)
            self._completed.discard(request_id)
            self._completed_started_ns.pop(request_id, None)
            self._completed_manager_pages.pop(request_id, None)
            context_sequence = self._next_sequence
            self._next_sequence += 1
            submitted_ns = time.perf_counter_ns()
            try:
                ticket = self._ring.submit(
                    context_sequence=context_sequence,
                    logical_start=logical_start,
                    physical_pages_by_group=capture_groups,
                    producer_stream=producer_stream,
                )
            except Exception:
                self._mark_completed_locked(
                    request_id,
                    sum(result_block_counts),
                    started_ns=submitted_ns,
                )
                self._cv.notify_all()
                raise
            if ticket is None:
                self._mark_completed_locked(
                    request_id,
                    sum(result_block_counts),
                    started_ns=submitted_ns,
                )
                self._connector._abort_async_page_capture(
                    plan.digest, "capture ring is busy or the payload was rejected"
                )
                self._cv.notify_all()
                return False
            self._pending[request_id] = _PendingCapture(
                request_id=request_id,
                context_sequence=context_sequence,
                plan=plan,
                ticket=ticket,
                capture_block_counts=tuple(len(group) for group in capture_groups),
                result_block_counts=result_block_counts,
                reused_pages_by_group=reused_pages_by_group,
                submitted_ns=submitted_ns,
            )
            self._ensure_thread_locked()
            self._wake.set()
            return True

    def preempt(self, request_id: str) -> None:
        with self._cv:
            queued = self._queued_jobs.get(request_id) or self._submitting_jobs.get(request_id)
            if queued is not None:
                queued.abandoned = True
                self._wake.set()
                return
            pending = self._pending.get(request_id)
            if pending is None:
                return
            if getattr(pending.plan, "capture_job_id", ""):
                # Connector-owned references protect the sources while the
                # background copy drains; preemption does not wait for cache I/O.
                pending.abandoned = True
                self._wake.set()
                return
            self._pending.pop(request_id)
            # vLLM may reuse every group page as soon as this callback returns.
            # The native drain synchronizes only this context's capture event.
            self._ring.drain_context(pending.context_sequence)
            self._connector._abort_async_page_capture(
                pending.plan.digest, "request was preempted"
            )
            self._mark_completed_locked(
                request_id,
                sum(pending.result_block_counts),
                started_ns=pending.submitted_ns,
            )
            self._cv.notify_all()

    def finish_failed_job(self, plan: Any) -> None:
        """Acknowledge rejected jobs only when no read has uncertain ownership."""
        job = getattr(plan, "capture_job_id", "")
        if not job:
            return
        with self._cv:
            if self._job_read_is_owned_locked(job):
                return
            self._connector._capture_read_completed(job)

    def _job_read_is_owned_locked(self, job: str) -> bool:
        """Include reads detached from the queue while shutdown drains them."""
        return (
            job in self._failed_jobs
            or job in self._draining_jobs
            or any(
                getattr(queued.plan, "capture_job_id", "") == job
                for queued in (*self._queued_jobs.values(), *self._submitting_jobs.values())
            )
            or any(
                getattr(capture.plan, "capture_job_id", "") == job
                for capture in self._pending.values()
            )
        )

    def finish_without_capture(
        self,
        request_id: str,
        *,
        retained_manager_pages: int = 0,
    ) -> None:
        """Report a terminal store outcome when no CUDA work was submitted.

        Scheduler-side asynchronous publication eligibility delays KV-block
        reclamation before workers decide whether they can accept the store.
        A worker that skips before submission therefore owes the same terminal
        completion as a drained capture. An in-flight request cannot use this
        path because its pages remain owned by the native completion fence.
        """

        if retained_manager_pages < 0:
            raise ValueError("retained manager pages cannot be negative")
        with self._cv:
            if self._closed:
                raise RuntimeError("manager-page capture runtime is closed")
            if request_id in self._pending:
                raise RuntimeError(
                    "cannot finish a manager-page capture before its fence drains"
                )
            self._mark_completed_locked(request_id, retained_manager_pages)
            self._cv.notify_all()

    def take_finished(self, finished_request_ids: set[str]) -> set[str]:
        with self._cv:
            # Read jobs retire through job acknowledgements, not request completion.
            owned = set(finished_request_ids) & (
                self._completed
                | {
                    request_id
                    for request_id, capture in self._pending.items()
                    if not getattr(capture.plan, "capture_job_id", "")
                }
            )
            self._pending_finished_requests.update(owned)
            if self._fatal is not None:
                # Withhold completion so vLLM retains every possibly owned
                # page, but allow the following connector-statistics call to
                # transport ownership_uncertain to logs and Prometheus.
                return set()
            ready = self._completed & self._pending_finished_requests
            self._completed.difference_update(ready)
            self._pending_finished_requests.difference_update(ready)
            for request_id in ready:
                self._completed_started_ns.pop(request_id, None)
                self._completed_manager_pages.pop(request_id, None)
            return ready

    def status(self) -> dict[str, int | float | bool]:
        """Describe request-owned pages awaiting a terminal acknowledgement."""

        with self._cv:
            delayed_ids = set(self._pending_finished_requests)
            timestamps = [
                self._pending[request_id].submitted_ns
                if request_id in self._pending
                else self._completed_started_ns[request_id]
                for request_id in delayed_ids
                if request_id in self._pending
                or request_id in self._completed_started_ns
            ]
            now_ns = time.perf_counter_ns()
            oldest_delayed_ms = (
                max(0, now_ns - min(timestamps)) / 1_000_000
                if timestamps
                else 0.0
            )
            retained_manager_pages = sum(
                sum(self._pending[request_id].result_block_counts)
                if request_id in self._pending
                else self._completed_manager_pages.get(request_id, 0)
                for request_id in delayed_ids
            )
            return {
                "pending_requests": len(self._pending) + len(self._queued_jobs) + len(self._submitting_jobs),
                "completed_notifications": len(self._completed),
                "delayed_requests": len(delayed_ids),
                "retained_manager_pages": retained_manager_pages,
                "oldest_delayed_ms": oldest_delayed_ms,
                "ownership_uncertain": self._fatal is not None,
            }

    def wait_idle(self, timeout: float | None = None) -> bool:
        with self._cv:
            return self._cv.wait_for(
                lambda: not self._pending and not self._queued_jobs and not self._submitting_jobs, timeout,
            )

    def shutdown(self) -> bool:
        self.quiesce()
        if int(getattr(self._ring, "active_ticket_count", 0)):
            return False
        self._ring.shutdown()
        return True

    def quiesce(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            for queued in (*self._queued_jobs.values(), *self._submitting_jobs.values()):
                queued.abandoned = True
            self._stop.set()
            self._wake.set()
            thread = self._thread
        # A submit may be inside native recovery. Join before collecting its
        # ticket or acknowledging a queued job that could otherwise start.
        if thread is not None and thread is not threading.current_thread():
            thread.join()
        with self._cv:
            pending = tuple(self._pending.values())
            self._draining_jobs.update(
                capture.plan.capture_job_id for capture in pending
                if getattr(capture.plan, "capture_job_id", "")
            )
            self._pending.clear()
            queued_jobs = tuple(self._queued_jobs.values())
            self._queued_jobs.clear()
        for queued in queued_jobs:
            self._connector._capture_read_completed(queued.plan.capture_job_id)
            self._connector._abort_async_page_capture(
                queued.plan.digest, "capture runtime shut down before submission"
            )
        for capture in pending:
            self._ring.drain_context(capture.context_sequence)
            job = getattr(capture.plan, "capture_job_id", "")
            if job:
                with self._cv:
                    self._connector._capture_read_completed(job)
                    self._draining_jobs.discard(job)
            self._connector._abort_async_page_capture(
                capture.plan.digest, "capture runtime shut down"
            )

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None:
            return
        thread = threading.Thread(
            target=self._progress_main,
            name="sparkcache-manager-page-capture",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _progress_main(self) -> None:
        try:
            self._thread_initializer()
            while not self._stop.is_set():
                self._wake.wait()
                if self._stop.is_set():
                    return
                self._progress_once()
                if self._stop.wait(self._poll_seconds):
                    return
        except Exception as error:  # noqa: BLE001 - background failure is isolated
            with self._cv:
                self._fatal = error
                for capture in self._pending.values():
                    self._connector._abort_async_page_capture(
                        capture.plan.digest,
                        f"background capture failed: {error}",
                    )
                    job = getattr(capture.plan, "capture_job_id", "")
                    if job:
                        self._failed_jobs.add(job)
                        self._connector._capture_read_failed(job)
                for queued in self._submitting_jobs.values():
                    job = queued.plan.capture_job_id
                    self._failed_jobs.add(job)
                    self._connector._capture_read_failed(job)
                for queued in self._queued_jobs.values():
                    self._connector._capture_read_completed(queued.plan.capture_job_id)
                    self._connector._abort_async_page_capture(
                        queued.plan.digest, f"capture progress unavailable: {error}"
                    )
                self._queued_jobs.clear()
                # Do not report finished_sending. vLLM must retain every page
                # until a later drain or worker termination proves ownership.
                self._cv.notify_all()

    def _progress_once(self) -> None:
        with self._cv:
            requests = tuple(self._queued_jobs)
        for request_id in requests:
            with self._cv:
                queued = self._queued_jobs.pop(request_id, None)
                if queued is None:
                    continue
                self._submitting_jobs[request_id] = queued
            self._submit_queued_job(queued)
        if self._fatal is not None:
            with self._cv:
                for capture in self._pending.values():
                    job = getattr(capture.plan, "capture_job_id", "")
                    if job:
                        self._failed_jobs.add(job)
                        self._connector._capture_read_failed(job)
                self._wake.clear()
            return
        ready: list[tuple[_PendingCapture, Any, int]] = []
        with self._cv:
            captures = tuple(self._pending.values())
        for capture in captures:
            if getattr(capture.plan, "capture_job_id", ""):
                # Job preemption only marks the read abandoned. It cannot
                # retire this ticket while polling runs outside the callback
                # lock; native recovery may hold the ring's separate lock.
                view = self._ring.poll(capture.ticket)
                if view is None:
                    continue
                claimed = self._ring.claim(capture.ticket)
            else:
                with self._cv:
                    if capture.request_id not in self._pending:
                        continue
                    view = self._ring.poll(capture.ticket)
                    if view is None:
                        continue
                    claimed = self._ring.claim(capture.ticket)
            with self._cv:
                if self._pending.pop(capture.request_id, None) is not None:
                    self._mark_completed_locked(
                        capture.request_id,
                        sum(capture.result_block_counts),
                        started_ns=capture.submitted_ns,
                        capture_job_id=getattr(capture.plan, "capture_job_id", ""),
                    )
                    ready.append((capture, claimed, time.perf_counter_ns()))
                if not self._pending and not self._queued_jobs and not self._submitting_jobs:
                    self._wake.clear()
                self._cv.notify_all()
        with self._cv:
            if not self._pending and not self._queued_jobs and not self._submitting_jobs:
                self._wake.clear()
        for capture, claimed, completed_ns in ready:
            scatter = None
            try:
                if capture.abandoned:
                    self._ring.release(capture.ticket)
                    self._connector._abort_async_page_capture(
                        capture.plan.digest, "request was preempted",
                    )
                    continue
                elapsed_ns = max(0, completed_ns - capture.submitted_ns)
                _log_capture_observed(
                    rank=self._rank,
                    digest=capture.plan.digest,
                    span_tokens=capture.plan.span_tokens,
                    elapsed_ns=elapsed_ns,
                    payload_bytes=claimed.payload.nbytes,
                )
                def release(ticket=capture.ticket) -> None:
                    self._ring.release(ticket)

                if capture.plan.base_context_digest:
                    scatter = PageExtensionScatter(
                        claimed.payload,
                        reused_pages_by_group=capture.reused_pages_by_group,
                        result_block_counts=capture.result_block_counts,
                        release=release,
                    )
                else:
                    header = encode_page_snapshot_header(
                        self._layout, capture.capture_block_counts
                    )
                    scatter = PageSnapshotScatter(
                        header,
                        claimed.payload,
                        release,
                    )
                self._connector._complete_async_page_capture(
                    capture.plan,
                    scatter,
                    capture.result_block_counts,
                )
            except Exception as error:  # noqa: BLE001 - serving continues
                if scatter is not None:
                    scatter.release()
                else:
                    self._ring.drain_context(capture.context_sequence)
                self._connector._abort_async_page_capture(
                    capture.plan.digest,
                    f"capture completion failed: {error}",
                )


__all__ = [
    "ManagerPageCaptureRuntime",
    "PageExtensionScatter",
    "PageSnapshotScatter",
]
