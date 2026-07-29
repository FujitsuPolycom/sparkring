"""Bounded ownership leases for asynchronous SparkCache snapshot gathers.

This module is deliberately independent of torch, CUDA and vLLM.  Production
code supplies a small completion-fence adapter (normally wrapping
``torch.cuda.Event``).  The state machine provides three properties:

* admission never waits: a full ring or overlapping block set returns ``None``;
* blocks remain leased until the submitted completion fence reports done;
* abort synchronizes every submitted fence before releasing its blocks.

The vLLM KV-Connector-V1 integration keeps a finished request's whole block
table allocated by returning ``True`` from ``request_finished()``.  Worker-side
``get_finished()`` can call :meth:`BlockLeaseRegistry.take_finished` and report
the request only after all of its fences complete.

During ordinary chunked prefill the request itself owns its blocks.  A lease is
therefore needed only from asynchronous gather submission until its completion,
and is normally gone long before request completion.  If the final tail is
still copying, the connector's delayed-free contract bridges that short gap.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol, runtime_checkable


@runtime_checkable
class CompletionFence(Protocol):
    """Minimal event surface needed by the lease registry."""

    def query(self) -> bool:
        """Return ``True`` only when the asynchronous read has completed."""

    def synchronize(self) -> None:
        """Wait until the asynchronous read can no longer touch its blocks."""


@dataclass(frozen=True, slots=True)
class LeaseCapacity:
    """Hard admission bounds; neither limit is allowed to queue inference."""

    max_active_leases: int
    max_leased_blocks: int

    def __post_init__(self) -> None:
        if self.max_active_leases <= 0:
            raise ValueError("max_active_leases must be positive")
        if self.max_leased_blocks <= 0:
            raise ValueError("max_leased_blocks must be positive")


class LeaseStateError(RuntimeError):
    """The caller violated the reserve -> submit -> complete state machine."""


class LeaseFenceError(RuntimeError):
    """Fence status is unknown, so releasing the associated blocks is unsafe."""


class _State(Enum):
    RESERVED = auto()
    SUBMITTING = auto()
    ARMED = auto()
    SUBMISSION_UNKNOWN = auto()


@dataclass(slots=True)
class _Lease:
    lease_id: int
    request_id: str
    block_ids: tuple[int, ...]
    state: _State = _State.RESERVED
    fence: CompletionFence | None = None


class LeaseHandle:
    """Single-use capability returned by :meth:`try_reserve`.

    Submit the gather through :meth:`submit`; never launch CUDA work first and
    publish its fence later. If a failure is known before any GPU work can
    reference the blocks, call :meth:`cancel_before_submit`.
    """

    __slots__ = ("_lease_id", "_registry")

    def __init__(self, registry: "BlockLeaseRegistry", lease_id: int) -> None:
        self._registry = registry
        self._lease_id = lease_id

    @property
    def lease_id(self) -> int:
        return self._lease_id

    def submit(self, submitter: Callable[[], CompletionFence]) -> None:
        """Atomically transition through CUDA submission to an armed fence.

        The registry serializes preemption against the callback. The callback
        must enqueue the gather, record its completion event, and return the
        event. Any callback exception is treated as unknown submission state:
        the lease remains held and the worker must fail closed.
        """

        self._registry._submit(self._lease_id, submitter)

    def cancel_before_submit(self) -> None:
        self._registry._cancel_before_submit(self._lease_id)


class BlockLeaseRegistry:
    """Thread-safe, bounded registry of block reads in flight."""

    def __init__(self, capacity: LeaseCapacity) -> None:
        self.capacity = capacity
        self.counters: Counter[str] = Counter()
        self._lock = threading.Lock()
        self._next_id = 1
        self._leases: dict[int, _Lease] = {}
        self._by_request: dict[str, set[int]] = defaultdict(set)
        self._block_owner: dict[int, int] = {}

    def try_reserve(
        self,
        request_id: str,
        block_ids: tuple[int, ...] | list[int],
    ) -> LeaseHandle | None:
        """Reserve a lease slot without blocking.

        Duplicate block IDs and invalid identifiers are programmer errors.
        Capacity pressure or overlap with an active lease is an expected
        fail-open cache miss and returns ``None``.
        """

        blocks = tuple(block_ids)
        if not request_id:
            raise ValueError("request_id must not be empty")
        if not blocks:
            raise ValueError("block_ids must not be empty")
        if len(set(blocks)) != len(blocks):
            raise ValueError("block_ids must be unique")
        if any(not isinstance(block_id, int) or block_id < 0 for block_id in blocks):
            raise ValueError("block_ids must contain non-negative integers")

        with self._lock:
            if len(self._leases) >= self.capacity.max_active_leases:
                self.counters["rejected_lease_capacity"] += 1
                return None
            if (
                len(self._block_owner) + len(blocks)
                > self.capacity.max_leased_blocks
            ):
                self.counters["rejected_block_capacity"] += 1
                return None
            if any(block_id in self._block_owner for block_id in blocks):
                self.counters["rejected_overlap"] += 1
                return None

            lease_id = self._next_id
            self._next_id += 1
            lease = _Lease(lease_id, request_id, blocks)
            self._leases[lease_id] = lease
            self._by_request[request_id].add(lease_id)
            for block_id in blocks:
                self._block_owner[block_id] = lease_id
            self.counters["reserved"] += 1
            return LeaseHandle(self, lease_id)

    def _submit(
        self,
        lease_id: int,
        submitter: Callable[[], CompletionFence],
    ) -> None:
        if not callable(submitter):
            raise TypeError("submitter must be callable")
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseStateError("lease is no longer active")
            if lease.state is not _State.RESERVED:
                raise LeaseStateError("lease has already entered submission")
            lease.state = _State.SUBMITTING
            self.counters["submitting"] += 1
            try:
                fence = submitter()
                if not isinstance(fence, CompletionFence):
                    raise TypeError(
                        "submitter must return a fence implementing"
                        " query() and synchronize()"
                    )
            except Exception as error:
                # The callback may have enqueued CUDA work before failing.
                # Without a published fence, no path can prove the read ended.
                lease.state = _State.SUBMISSION_UNKNOWN
                self.counters["submission_unknown"] += 1
                raise LeaseFenceError(
                    f"submission status unknown for lease {lease_id}"
                ) from error
            lease.fence = fence
            lease.state = _State.ARMED
            self.counters["armed"] += 1

    def _cancel_before_submit(self, lease_id: int) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseStateError("lease is no longer active")
            if lease.state is not _State.RESERVED:
                raise LeaseStateError(
                    "a submitted lease must be drained, never cancelled"
                )
            self._release_locked(lease)
            self.counters["cancelled_before_submit"] += 1

    def poll(self) -> int:
        """Release every armed lease whose fence has completed.

        A fence exception does *not* release blocks.  The caller must use
        :meth:`abort_request` (which synchronizes) or terminate the worker.
        """

        with self._lock:
            armed = tuple(
                lease
                for lease in self._leases.values()
                if lease.state is _State.ARMED
            )

        completed: list[int] = []
        for lease in armed:
            assert lease.fence is not None
            try:
                done = lease.fence.query()
            except Exception as error:
                self.counters["fence_query_failed"] += 1
                raise LeaseFenceError(
                    f"completion status unknown for lease {lease.lease_id}"
                ) from error
            if done:
                completed.append(lease.lease_id)

        with self._lock:
            released = 0
            for lease_id in completed:
                lease = self._leases.get(lease_id)
                if lease is None:
                    continue
                self._release_locked(lease)
                released += 1
            self.counters["completed"] += released
            return released

    def take_finished(self, finished_request_ids: set[str]) -> set[str]:
        """Return delayed-free requests whose last lease has completed.

        The caller must first intersect the request IDs vLLM supplied to
        worker-side ``get_finished()`` with the IDs owned by its streaming
        adapter.  IDs unknown to this registry *within that ownership domain*
        are intentionally considered ready: scheduler-side admission may have
        delayed a request even when this worker abandoned publication before
        submitting a gather.
        """

        self.poll()
        with self._lock:
            active_requests = set(self._by_request)
        ready = set(finished_request_ids) - active_requests
        self.counters["requests_finished"] += len(ready)
        return ready

    def abort_request(self, request_id: str) -> int:
        """Synchronize submitted work, then release every request lease.

        Reserved-but-unsubmitted leases are released immediately.  If a fence
        cannot be synchronized the registry retains that lease and raises
        :class:`LeaseFenceError`; reusing its blocks would be unsafe.
        """

        # Release RESERVED leases while holding the same lock used by submit().
        # submit() holds the lock across enqueue plus fence publication, so
        # preemption sees either RESERVED (nothing launched) or ARMED (a fence
        # it can synchronize), never the unsafe gap between those states.
        with self._lock:
            leases = tuple(
                self._leases[lease_id]
                for lease_id in self._by_request.get(request_id, ())
            )
            reserved = [lease for lease in leases if lease.state is _State.RESERVED]
            armed = [lease for lease in leases if lease.state is _State.ARMED]
            unknown = [
                lease
                for lease in leases
                if lease.state is _State.SUBMISSION_UNKNOWN
            ]
            for lease in reserved:
                self._release_locked(lease)
            released = len(reserved)
            if unknown:
                self.counters["submission_unknown_abort"] += len(unknown)
                raise LeaseFenceError(
                    "cannot safely abort lease(s) with unknown submission"
                    f" state: {[lease.lease_id for lease in unknown]}"
                )

        for lease in armed:
            assert lease.fence is not None
            try:
                lease.fence.synchronize()
            except Exception as error:
                self.counters["fence_synchronize_failed"] += 1
                raise LeaseFenceError(
                    f"cannot safely abort lease {lease.lease_id}"
                ) from error

        with self._lock:
            for lease in armed:
                current = self._leases.get(lease.lease_id)
                if current is None:
                    continue
                self._release_locked(current)
                released += 1
            self.counters["aborted_leases"] += released
            return released

    def abort_all(self) -> int:
        """Drain all requests, suitable for connector shutdown/rollback."""

        with self._lock:
            request_ids = tuple(self._by_request)
        return sum(self.abort_request(request_id) for request_id in request_ids)

    def has_pending(self, request_id: str | None = None) -> bool:
        with self._lock:
            if request_id is None:
                return bool(self._leases)
            return bool(self._by_request.get(request_id))

    def active_leases_for(self, request_id: str) -> int:
        """Return the exact number of leases still owned by one request."""

        if not isinstance(request_id, str) or not request_id:
            raise ValueError("request_id must be nonempty")
        with self._lock:
            return len(self._by_request.get(request_id, ()))

    @property
    def active_leases(self) -> int:
        with self._lock:
            return len(self._leases)

    @property
    def leased_blocks(self) -> int:
        with self._lock:
            return len(self._block_owner)

    def _release_locked(self, lease: _Lease) -> None:
        self._leases.pop(lease.lease_id)
        request_leases = self._by_request[lease.request_id]
        request_leases.remove(lease.lease_id)
        if not request_leases:
            del self._by_request[lease.request_id]
        for block_id in lease.block_ids:
            owner = self._block_owner.pop(block_id)
            if owner != lease.lease_id:
                raise AssertionError("block lease ownership corrupted")
