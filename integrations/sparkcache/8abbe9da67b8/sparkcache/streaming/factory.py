"""Verified model-serving assembly for SparkCache streaming snapshots.

Importing this module does not import the ctypes binding, allocate CUDA
memory, inspect cache tensors, or start a writer thread.  The worker adapter
performs those actions exactly once, after vLLM has registered its final KV
cache inventory. The scheduler adapter never imports the C++/CUDA binding.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Sequence

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_LOGGER = logging.getLogger("vllm.spark_context_cache.streaming")
_STATUS_PREFIX = "[sparkcache-streaming-status]"

NATIVE_LIBRARY_CONFIG_KEY = "spark_cache_streaming_native_library"
NATIVE_LIBRARY_ENVIRONMENT_KEY = (
    "SPARK_CONTEXT_CACHE_STREAMING_NATIVE_LIBRARY"
)
NATIVE_LIBRARY_SHA256_CONFIG_KEY = (
    "spark_cache_streaming_native_library_sha256"
)
NATIVE_LIBRARY_SHA256_ENVIRONMENT_KEY = (
    "SPARK_CONTEXT_CACHE_STREAMING_NATIVE_LIBRARY_SHA256"
)
VLLM_ROOT_CONFIG_KEY = "spark_cache_streaming_vllm_root"
VLLM_ROOT_ENVIRONMENT_KEY = "SPARK_CONTEXT_CACHE_STREAMING_VLLM_ROOT"
LEASE_CONTRACT_CONFIG_KEY = "spark_cache_streaming_lease_contract"
LEASE_CONTRACT_ENVIRONMENT_KEY = (
    "SPARK_CONTEXT_CACHE_STREAMING_LEASE_CONTRACT"
)
TIMING_CONFIG_KEY = "spark_cache_streaming_timing"
TIMING_ENVIRONMENT_KEY = "SPARK_CONTEXT_CACHE_STREAMING_TIMING"

BACKGROUND_PROGRESS_POLL_SECONDS = 0.005


def _deployment_path_is_absolute(path: Path) -> bool:
    text = str(path)
    # Windows CI parses a deployment path such as ``/opt/lib.so`` as a
    # drive-relative ``\opt\lib.so``.  A nonempty root is nevertheless an
    # absolute POSIX path in the Linux process that will consume this config.
    return (
        path.is_absolute()
        or bool(path.root)
        or PurePosixPath(text).is_absolute()
    )


def _set_background_cuda_device(device_ordinal: int) -> None:
    """Establish CUDA's thread-local device before C++/CUDA event polling."""

    import torch

    # CPU-only test environments cannot own a C++/CUDA ring; injected fake
    # rings still exercise the thread/lifecycle contract. A CUDA-enabled
    # serving environment exposes this binding, and set_device failures remain
    # sticky/fatal below.
    if not callable(getattr(torch._C, "_cuda_setDevice", None)):
        return
    torch.cuda.set_device(device_ordinal)


def _extra(connector: Any, key: str, environment: str, default: str = "") -> str:
    value = connector._kv_transfer_config.get_from_extra_config(
        key,
        os.environ.get(environment, default),
    )
    return str(value or "")


@dataclass(frozen=True, slots=True)
class ModelServingStreamingSettings:
    """Attested paths for the only admitted live snapshot profile."""

    native_library_path: Path
    native_library_sha256: str
    vllm_root: Path | None = None
    lease_contract_path: Path | None = None
    timing_enabled: bool = False

    def __post_init__(self) -> None:
        if (
            not _deployment_path_is_absolute(self.native_library_path)
            or _SHA256_RE.fullmatch(self.native_library_sha256) is None
        ):
            raise RuntimeError(
                "streaming snapshots require an absolute C++/CUDA library path "
                "and a 64-character lowercase SHA-256"
            )
        for name in ("vllm_root", "lease_contract_path"):
            value = getattr(self, name)
            if value is not None and not _deployment_path_is_absolute(value):
                raise RuntimeError(f"{name} must be an absolute path")

    @classmethod
    def from_connector(cls, connector: Any) -> "ModelServingStreamingSettings":
        library = Path(
            _extra(
                connector,
                NATIVE_LIBRARY_CONFIG_KEY,
                NATIVE_LIBRARY_ENVIRONMENT_KEY,
            )
        )
        sha256 = _extra(
            connector,
            NATIVE_LIBRARY_SHA256_CONFIG_KEY,
            NATIVE_LIBRARY_SHA256_ENVIRONMENT_KEY,
        )
        vllm_root_text = _extra(
            connector,
            VLLM_ROOT_CONFIG_KEY,
            VLLM_ROOT_ENVIRONMENT_KEY,
        )
        contract_text = _extra(
            connector,
            LEASE_CONTRACT_CONFIG_KEY,
            LEASE_CONTRACT_ENVIRONMENT_KEY,
        )
        timing_text = _extra(
            connector,
            TIMING_CONFIG_KEY,
            TIMING_ENVIRONMENT_KEY,
            "0",
        )
        if timing_text not in ("0", "1"):
            raise RuntimeError("streaming timing flag must be exactly 0 or 1")
        return cls(
            native_library_path=library,
            native_library_sha256=sha256,
            vllm_root=Path(vllm_root_text) if vllm_root_text else None,
            lease_contract_path=(
                Path(contract_text) if contract_text else None
            ),
            timing_enabled=timing_text == "1",
        )


class SchedulerStreamingSnapshotAdapter:
    """CUDA-free scheduler owner of conservative delayed-free bookkeeping."""

    def __init__(self, connector: Any) -> None:
        self._connector = connector
        self._eligible: set[str] = set()
        self._delayed: set[str] = set()
        self._closed = False
        self.counters: Counter[str] = Counter()

    def observe_metadata(self, metadata: Any) -> None:
        # A resumed request may be present in both fields. Clear ownership from
        # the completed attempt before the active offer establishes eligibility.
        for request_id in getattr(
            metadata,
            "preempted_request_ids",
            (),
        ):
            self._eligible.discard(request_id)
            self._delayed.discard(request_id)
        offers = getattr(metadata, "streaming_snapshot_offers", ())
        for offer in offers:
            request_id = getattr(offer, "request_id", None)
            if not isinstance(request_id, str) or not request_id:
                raise RuntimeError(
                    "streaming snapshot metadata has an invalid request ID"
                )
            self._eligible.add(request_id)
            self.counters["offers_observed"] += 1

    def request_finished(
        self,
        request_id: str,
        block_ids: Sequence[int],
    ) -> bool:
        del block_ids
        if request_id not in self._eligible:
            return False
        self._delayed.add(request_id)
        self.counters["requests_delayed"] += 1
        return True

    def take_finished(self, finished_request_ids: set[str]) -> set[str]:
        ready = self._delayed & set(finished_request_ids)
        self._delayed.difference_update(ready)
        self._eligible.difference_update(ready)
        self.counters["requests_released"] += len(ready)
        return ready

    def shutdown(self) -> None:
        self._eligible.clear()
        self._delayed.clear()
        self._closed = True

    def status(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "role": "scheduler",
            "closed": self._closed,
            "eligible_requests": len(self._eligible),
            "delayed_requests": len(self._delayed),
            "counters": dict(self.counters),
        }


class WorkerStreamingSnapshotAdapter:
    """Connector-facing, lazily bound owner of the live worker pipeline."""

    def __init__(
        self,
        connector: Any,
        *,
        settings: ModelServingStreamingSettings,
        ring_builder: Callable[..., Any] | None = None,
        progress_poll_seconds: float = BACKGROUND_PROGRESS_POLL_SECONDS,
        progress_thread_initializer: Callable[[int], None] | None = None,
    ) -> None:
        if progress_poll_seconds <= 0:
            raise ValueError("progress_poll_seconds must be positive")
        self._connector = connector
        from sparkcache.profile_adapters import resolve_streaming_profile_adapter

        profile_name = str(getattr(getattr(connector, "_profile", None), "name", ""))
        if not profile_name:
            raise RuntimeError(
                "streaming snapshots require an explicit model profile"
            )
        self._profile_adapter = resolve_streaming_profile_adapter(profile_name)
        self.settings = settings
        self._ring_builder = ring_builder
        self._progress_poll_seconds = float(progress_poll_seconds)
        self._progress_thread_initializer = (
            progress_thread_initializer or _set_background_cuda_device
        )
        self._progress_device_ordinal: int | None = None
        self._state_lock = threading.RLock()
        self._progress_wake = threading.Event()
        self._progress_stop = threading.Event()
        self._progress_thread: threading.Thread | None = None
        self._background_progress_error: str | None = None
        self._background_progress_fatal: BaseException | None = None
        self._runtime: Any = None
        self._ring: Any = None
        self._writer: Any = None
        self._planner: Any = None
        self._leases: Any = None
        self._timing_trace: Any = None
        self._retained_row_views: tuple[Any, ...] = ()
        self._seen_requests: set[str] = set()
        # vLLM presents each newly finished scheduler request to worker-side
        # get_finished() once, but permits the connector to report completion
        # in a later call. Retain only IDs owned by this adapter until their
        # final lease drains; never retain ordinary, unseen request IDs.
        self._pending_finished_requests: set[str] = set()
        self._admitted_requests: set[str] = set()
        self._suppressed_requests: set[str] = set()
        # Scheduler ownership and cache-publication ownership end at different
        # times. A request's blocks may become releasable before the writer
        # publishes its manifest, so retain terminal evidence until the
        # runtime reports committed or aborted.
        self._terminal_pending_requests: set[str] = set()
        self._contexts: dict[str, tuple[str, int, float]] = {}
        self._last_watermark_at: dict[str, float] = {}
        self._bound = False
        self._closed = False
        self._started_at = time.monotonic()
        self.counters: Counter[str] = Counter()

    def bind_kv_caches(self) -> None:
        """Build and attest the worker pipeline after final cache registration."""

        with self._state_lock:
            self._bind_kv_caches_locked()

    def _bind_kv_caches_locked(self) -> None:
        if self._closed:
            raise RuntimeError("streaming snapshot adapter is closed")
        if self._bound:
            return
        adapter = self._profile_adapter
        adapter.validate_connector(self._connector)

        # All imports below are worker-only and occur after vLLM has supplied
        # the final tensor inventory. NativeSnapshotRing.from_attested is the
        # first edge that hashes/loads the .so and allocates its CUDA arena.
        from .block_lease import BlockLeaseRegistry, LeaseCapacity
        from .native_ring import NativeRingConfig, NativeSnapshotRing
        from .planner import StreamingSnapshotCoordinator
        from .publisher import ManifestSnapshotJournalWriter
        from .runtime import (
            StreamingSnapshotRuntime,
            StreamingSnapshotRuntimeConfig,
        )

        inventory = adapter.build_source_inventory(self._connector)
        rank = int(self._connector._worker_rank())
        adapter.validate_rank(rank)
        config = adapter.build_ring_config(inventory.device_ordinal)
        builder = self._ring_builder
        if builder is None:
            def build_attested_ring(
                ring_config: NativeRingConfig,
                *,
                library_path: Path,
                expected_sha256: str,
            ) -> NativeSnapshotRing:
                return NativeSnapshotRing.from_attested(
                    ring_config,
                    library_path=library_path,
                    expected_sha256=expected_sha256,
                )

            builder = build_attested_ring

        ring = None
        writer = None
        runtime = None
        timing_trace = None
        try:
            ring = builder(
                config,
                library_path=self.settings.native_library_path,
                expected_sha256=self.settings.native_library_sha256,
            )
            ring.configure_sources(inventory.sources)
            connector_module = sys.modules.get(
                self._connector.__class__.__module__
            )
            if connector_module is None:
                raise RuntimeError(
                    "cannot resolve connector storage ABI module"
                )
            context_chunk_factory = getattr(
                connector_module,
                "ContextChunk",
                None,
            )
            state_record_type = getattr(
                connector_module,
                "StateRecord",
                None,
            )
            pack_positions_function = getattr(
                connector_module,
                "pack_positions",
                None,
            )
            if (
                not callable(context_chunk_factory)
                or state_record_type is None
                or not callable(pack_positions_function)
            ):
                raise RuntimeError(
                    "connector storage ABI exports are incomplete"
                )
            translator = adapter.build_translator(
                ring,
                dcp_rank=rank,
                context_chunk_factory=context_chunk_factory,
                state_record_type=state_record_type,
                pack_positions_function=pack_positions_function,
            )
            if self.settings.timing_enabled:
                from .timing import StreamingTimingTrace

                timing_trace = StreamingTimingTrace(
                    enabled=True,
                    sink=_LOGGER.info,
                )
            identity = self._connector._identity(rank)
            adapter.validate_identity(identity)
            writer = ManifestSnapshotJournalWriter(
                store=self._connector._store,
                identity=identity,
                translator=translator,
                max_pending_batches=adapter.ring_depth,
                timing_trace=timing_trace,
            )
            planner = StreamingSnapshotCoordinator(
                chunk_tokens=256,
                chunks_per_batch=adapter.chunks_per_batch,
                max_inflight_batches=adapter.ring_depth,
                max_sessions=adapter.max_sessions,
            )
            leases = BlockLeaseRegistry(
                LeaseCapacity(
                    max_active_leases=adapter.ring_depth,
                    max_leased_blocks=adapter.ring_depth
                    * (
                        (
                            adapter.max_rows
                            + int(self._connector._block_size)
                            - 1
                        )
                        // int(self._connector._block_size)
                    ),
                )
            )
            runtime = StreamingSnapshotRuntime(
                StreamingSnapshotRuntimeConfig(
                    enabled=True,
                    block_size=int(self._connector._block_size),
                    dcp_degree=adapter.dcp_degree,
                    dcp_rank=rank,
                ),
                planner=planner,
                leases=leases,
                ring=ring,
                writer=writer,
                timing_trace=timing_trace,
            )
        except BaseException:
            if runtime is not None:
                runtime.shutdown()
            elif ring is not None:
                ring.shutdown()
            if writer is not None:
                writer.shutdown()
            raise

        self._ring = ring
        self._writer = writer
        self._planner = planner
        self._leases = leases
        self._timing_trace = timing_trace
        self._runtime = runtime
        self._retained_row_views = inventory.retained_row_views
        self._progress_device_ordinal = inventory.device_ordinal
        self._bound = True
        self._start_background_progress_locked()
        self.counters["bound"] += 1
        self._emit_status("bound", dcp_rank=rank, **adapter.bound_status())

    def offer_completed(
        self,
        offer: Any,
        *,
        producer_stream: int,
    ) -> None:
        with self._state_lock:
            self._offer_completed_locked(
                offer,
                producer_stream=producer_stream,
            )

    def _offer_completed_locked(
        self,
        offer: Any,
        *,
        producer_stream: int,
    ) -> None:
        self._require_bound()
        request_id = str(offer.request_id)
        digest = str(offer.digest)
        span = int(offer.span_tokens)
        if (
            not request_id
            or _SHA256_RE.fullmatch(digest) is None
            or span <= 0
        ):
            raise RuntimeError("streaming snapshot offer is malformed")
        self._seen_requests.add(request_id)
        existing_context = self._contexts.get(request_id)
        if existing_context is not None and (
            existing_context[0] != digest or existing_context[1] != span
        ):
            if request_id in self._admitted_requests:
                self._runtime.cancel(
                    request_id,
                    reason="offer_identity_changed",
                )
            self._admitted_requests.discard(request_id)
            self._suppressed_requests.add(request_id)
            self.counters["offer_identity_changed"] += 1
            return
        if request_id in self._suppressed_requests:
            self.counters["offers_suppressed_after_abort"] += 1
            return
        self._contexts.setdefault(
            request_id,
            (digest, span, time.monotonic()),
        )
        if request_id not in self._admitted_requests:
            admitted = self._runtime.begin_context(
                request_id=request_id,
                context_digest=digest,
                span_tokens=span,
            )
            if not admitted:
                self.counters["begin_suppressed"] += 1
                self._suppressed_requests.add(request_id)
                return
            self._admitted_requests.add(request_id)
            self._terminal_pending_requests.add(request_id)
        self._last_watermark_at[request_id] = time.monotonic()
        completed_tokens = int(offer.completed_tokens)
        result = self._runtime.accept_completed_prefill(
            request_id=request_id,
            completed_tokens=completed_tokens,
            block_table=tuple(offer.block_ids),
            producer_stream=producer_stream,
        )
        self.counters["offers_completed"] += 1
        if result.aborted:
            self.counters["offers_aborted"] += 1
            self._admitted_requests.discard(request_id)
            self._suppressed_requests.add(request_id)
        self._wake_background_if_needed_locked()

    def poll(self) -> int:
        with self._state_lock:
            return self._poll_locked()

    def _poll_locked(self) -> int:
        self._require_bound()
        completed = int(self._runtime.poll())
        aborted = dict(self._runtime.take_aborted())
        for request_id, reason in sorted(aborted.items()):
            self._admitted_requests.discard(request_id)
            if reason == "preempted":
                self._suppressed_requests.discard(request_id)
            else:
                self._suppressed_requests.add(request_id)
            self.counters["runtime_aborted"] += 1
            self._emit_terminal(
                request_id,
                "aborted",
                reason=str(reason),
            )
            self._retire_terminal_request(request_id)
        committed = dict(self._runtime.take_committed())
        if committed:
            handoff = getattr(
                self._connector,
                "_handoff_streaming_commits",
                None,
            )
            if callable(handoff):
                advertised = set(handoff(committed))
            else:
                # Lightweight embedders predating bounded capacity have no
                # handoff hook. Keep their post-manifest behavior intact.
                advertised = set(committed)
                with self._connector._load_lock:
                    self._connector._held.update(advertised)
                    self._connector.counters["streaming_store_committed"] = (
                        self._connector.counters.get(
                            "streaming_store_committed",
                            0,
                        )
                        + len(advertised)
                    )
            for digest in sorted(committed):
                request_id = next(
                    (
                        candidate
                        for candidate in sorted(
                            self._terminal_pending_requests
                        )
                        if (
                            (context := self._contexts.get(candidate))
                            is not None
                            and context[0] == digest
                        )
                    ),
                    None,
                )
                if request_id is not None:
                    if self._timing_trace is not None:
                        self._timing_trace.mark_final(
                            request_id,
                            "adapter_observed",
                        )
                    self._admitted_requests.discard(request_id)
                    self._suppressed_requests.add(request_id)
                    self._emit_terminal(request_id, "committed")
                    self._retire_terminal_request(request_id)
            self.counters["manifests_handed_off"] += len(committed)
            self.counters["digests_advertised"] += len(advertised)
        ownership = self.status()
        if completed > 0 and all(
            int(ownership[name]) == 0
            for name in (
                "active_contexts",
                "active_leases",
                "leased_blocks",
                "active_tickets",
            )
        ):
            # An abort may retain a claimed writer-owned ring view after its
            # one terminal edge. Emit the later observable ownership boundary
            # so log-based checks do not mistake a stale abort snapshot for a
            # leak after the writer really drains.
            self._emit_status("drained", completed_batches=completed)
        return completed

    def handle_preemptions(self, metadata: Any) -> None:
        with self._state_lock:
            self._handle_preemptions_locked(metadata)

    def _handle_preemptions_locked(self, metadata: Any) -> None:
        self._require_bound()
        request_ids = getattr(metadata, "preempted_request_ids", None)
        if (
            not isinstance(request_ids, tuple)
            or request_ids != tuple(sorted(request_ids))
            or len(set(request_ids)) != len(request_ids)
            or any(
                not isinstance(request_id, str) or not request_id
                for request_id in request_ids
            )
        ):
            raise RuntimeError(
                "streaming preemption metadata must be a sorted unique tuple"
            )
        for request_id in request_ids:
            if self._runtime.preempt(request_id):
                self.counters["preempted"] += 1
                self._admitted_requests.discard(request_id)
                # A resumed request receives a replacement physical block
                # table and must be allowed to begin a fresh invisible
                # transaction.
                self._suppressed_requests.discard(request_id)
        # take_aborted() is drained through poll so each terminal edge emits
        # exactly one structured status record.
        self.poll()
        # SchedulerStreamingSnapshotAdapter retires the preempted delayed-free
        # generation as soon as it observes preemption. Mirror that boundary
        # here after poll has emitted the preempted transaction's terminal
        # evidence.
        # Otherwise an ID retained from an earlier get_finished() call can be
        # echoed after the scheduler has removed it, or can incorrectly finish
        # a resumed generation that reuses the same public request ID.
        for request_id in request_ids:
            self._retire_finished_request(request_id)

    def request_finished(
        self,
        request_id: str,
        block_ids: Sequence[int],
    ) -> bool:
        del block_ids
        with self._state_lock:
            return request_id in self._seen_requests

    def take_finished(self, finished_request_ids: set[str]) -> set[str]:
        with self._state_lock:
            return self._take_finished_locked(finished_request_ids)

    def _take_finished_locked(
        self,
        finished_request_ids: set[str],
    ) -> set[str]:
        self._require_bound()
        # Route through the public method so test/host adapters that supply a
        # compatible polling hook retain the pre-existing extension point.
        self.poll()
        # vLLM supplies every request that finished in the scheduler output,
        # not only requests whose blocks the streaming adapter asked it to
        # retain.  BlockLeaseRegistry deliberately considers an unknown
        # request ready so an admitted publication that abandoned before its
        # first lease can be released.  Apply that rule only inside this
        # adapter's ownership domain; echoing an entirely unseen request as a
        # finished async send makes Scheduler._update_from_kv_xfer_finished()
        # receive an ID it already removed from Scheduler.requests.
        owned_finished = set(finished_request_ids) & self._seen_requests
        self._pending_finished_requests.update(owned_finished)
        ready = self._leases.take_finished(self._pending_finished_requests)
        self._pending_finished_requests.difference_update(ready)
        for request_id in ready:
            self._retire_finished_request(request_id)
        return ready

    def _retire_finished_request(self, request_id: str) -> None:
        """Forget scheduler ownership, retaining any pending terminal record."""

        self._pending_finished_requests.discard(request_id)
        self._seen_requests.discard(request_id)
        self._admitted_requests.discard(request_id)
        self._suppressed_requests.discard(request_id)
        if request_id not in self._terminal_pending_requests:
            self._contexts.pop(request_id, None)
            self._last_watermark_at.pop(request_id, None)

    def _retire_terminal_request(self, request_id: str) -> None:
        """Forget publication evidence after its one terminal edge is emitted."""

        self._terminal_pending_requests.discard(request_id)
        self._contexts.pop(request_id, None)
        self._last_watermark_at.pop(request_id, None)

    def shutdown(self) -> None:
        with self._state_lock:
            if self._closed:
                return
            self._closed = True
            self._progress_stop.set()
            self._progress_wake.set()
            progress_thread = self._progress_thread
        if (
            progress_thread is not None
            and progress_thread is not threading.current_thread()
        ):
            progress_thread.join()
        with self._state_lock:
            if self._runtime is not None:
                self._runtime.shutdown()
            if self._writer is not None:
                self._writer.shutdown()
            self._timing_trace = None
            self._retained_row_views = ()
            self._pending_finished_requests.clear()
            self._terminal_pending_requests.clear()
            self.counters["shutdown"] += 1
            self._emit_status("shutdown")

    def _start_background_progress_locked(self) -> None:
        if self._progress_thread is not None:
            return
        thread = threading.Thread(
            target=self._background_progress_main,
            name="sparkcache-streaming-progress",
            daemon=True,
        )
        self._progress_thread = thread
        thread.start()

    def _wake_background_if_needed_locked(self) -> None:
        if bool(getattr(self._runtime, "needs_progress", False)):
            self._progress_wake.set()

    def _background_progress_main(self) -> None:
        device_ordinal = self._progress_device_ordinal
        if device_ordinal is None:
            self._record_background_fatal(
                RuntimeError("background progress has no CUDA device ordinal")
            )
            return
        try:
            self._progress_thread_initializer(device_ordinal)
        except Exception as error:
            self._record_background_fatal(error)
            return
        while not self._progress_stop.is_set():
            self._progress_wake.wait()
            if self._progress_stop.is_set():
                return
            while not self._progress_stop.is_set():
                try:
                    with self._state_lock:
                        if self._closed:
                            return
                        if not bool(
                            getattr(self._runtime, "needs_progress", False)
                        ):
                            self._progress_wake.clear()
                            break
                        self._poll_locked()
                        self.counters["background_progress_polls"] += 1
                        if not bool(
                            getattr(self._runtime, "needs_progress", False)
                        ):
                            self._progress_wake.clear()
                            break
                except Exception as error:
                    self._record_background_fatal(error)
                    return
                if self._progress_stop.wait(self._progress_poll_seconds):
                    return

    def _record_background_fatal(self, error: BaseException) -> None:
        with self._state_lock:
            self._background_progress_fatal = error
            self._background_progress_error = (
                f"{type(error).__name__}: {error}"
            )
            self.counters["background_progress_failed"] += 1
            self._progress_wake.clear()
        _LOGGER.error(
            "sparkcache streaming background progress stopped: %s",
            self._background_progress_error,
            exc_info=error,
        )

    def _require_bound(self) -> None:
        if not self._bound or self._runtime is None:
            raise RuntimeError(
                "streaming worker runtime is not bound to registered KV caches"
            )
        if self._closed:
            raise RuntimeError("streaming snapshot adapter is closed")
        if self._background_progress_fatal is not None:
            raise RuntimeError(
                "streaming background progress failed"
            ) from self._background_progress_fatal

    def _emit_terminal(
        self,
        request_id: str,
        event: str,
        *,
        reason: str | None = None,
    ) -> None:
        context = self._contexts.get(request_id)
        fields: dict[str, Any] = {"request_id": request_id}
        if context is not None:
            digest, span, started = context
            fields.update(
                context_digest=digest,
                context_digest_short=digest[:12],
                span_tokens=span,
                context_elapsed_ms=round(
                    (time.monotonic() - started) * 1000,
                    3,
                ),
            )
        if reason is not None:
            fields["reason"] = reason
        if event == "committed":
            watermark_at = self._last_watermark_at.get(request_id)
            fields.update(
                final_tail_ms=(
                    round(
                        (time.monotonic() - watermark_at) * 1000,
                        3,
                    )
                    if watermark_at is not None
                    else None
                ),
                visible_before_manifest=False,
                visible_after_manifest=True,
            )
            fields.update(self._manifest_evidence(context))
        elif event == "aborted":
            request_leases = (
                int(self._leases.active_leases_for(request_id))
                if self._leases is not None
                else 0
            )
            global_active_leases = (
                int(self._leases.active_leases)
                if self._leases is not None
                else 0
            )
            fields.update(
                visible_after_abort=False,
                # `leases_after_abort` remains the stable compatibility field, but its
                # contract is request-scoped. Other concurrently caching
                # requests may legitimately retain leases.
                leases_after_abort=request_leases,
                request_leases_after_abort=request_leases,
                global_active_leases_after_abort=global_active_leases,
            )
        self._emit_status(event, **fields)

    def _manifest_evidence(
        self,
        context: tuple[str, int, float] | None,
    ) -> dict[str, Any]:
        if context is None:
            return {}
        store = getattr(self._connector, "_store", None)
        if store is None:
            return {}
        digest = context[0]
        manifest_path_method = getattr(
            store,
            "_manifest_path",
            None,
        )
        if not callable(manifest_path_method):
            return {}
        try:
            path = Path(
                manifest_path_method(
                    self._connector._identity(
                        int(self._connector._worker_rank())
                    ),
                    digest,
                )
            )
            payload = path.read_bytes()
        except (OSError, TypeError, ValueError):
            return {}
        return {
            "manifest_path": str(path),
            "manifest_sha256": hashlib.sha256(payload).hexdigest(),
        }

    def _emit_status(self, event: str, **fields: Any) -> None:
        payload = {
            "event": event,
            "elapsed_ms": round(
                (time.monotonic() - self._started_at) * 1000,
                3,
            ),
            **fields,
            **self.status(),
        }
        _LOGGER.info(
            "%s %s",
            _STATUS_PREFIX,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        )

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            return self._status_locked()

    def _status_locked(self) -> dict[str, Any]:
        counters = Counter(self.counters)
        for dependency in (
            self._runtime,
            self._planner,
            self._leases,
        ):
            counters.update(getattr(dependency, "counters", {}) or {})
        return {
            "enabled": True,
            "bound": self._bound,
            "closed": self._closed,
            "background_progress_alive": bool(
                self._progress_thread is not None
                and self._progress_thread.is_alive()
            ),
            "background_progress_error": self._background_progress_error,
            "arena_mode": self._profile_adapter.arena_mode_name,
            "ring_depth": self._profile_adapter.ring_depth,
            "slot_bytes": self._profile_adapter.slot_bytes,
            "native_library_sha256": self.settings.native_library_sha256,
            "timing_enabled": self.settings.timing_enabled,
            "active_contexts": (
                int(getattr(self._runtime, "active_contexts", 0))
                if self._runtime is not None
                else 0
            ),
            "active_leases": (
                int(self._leases.active_leases)
                if self._leases is not None
                else 0
            ),
            "leased_blocks": (
                int(self._leases.leased_blocks)
                if self._leases is not None
                else 0
            ),
            "active_tickets": (
                int(self._ring.active_ticket_count)
                if self._ring is not None
                else 0
            ),
            "counters": dict(counters),
        }


def _default_vllm_root() -> Path:
    import vllm

    module_path = getattr(vllm, "__file__", None)
    if not module_path:
        raise RuntimeError("cannot resolve the installed vLLM source root")
    return Path(module_path).resolve().parent.parent


def verify_model_serving_lease_contract(
    settings: ModelServingStreamingSettings,
) -> tuple[Path, ...]:
    """Hash-attest the allocator/preemption sources before adapter creation."""

    from sparkcache.runtime_patches.verify_lease_contract import (
        verify_contract,
    )

    vllm_root = settings.vllm_root or _default_vllm_root()
    contract = settings.lease_contract_path or (
        Path(__file__).resolve().parents[1]
        / "runtime_patches"
        / "vllm-kv-block-lease-contract.json"
    )
    try:
        verified = verify_contract(vllm_root, contract)
    except Exception as error:
        raise RuntimeError(
            "streaming snapshot vLLM lease-contract attestation failed"
        ) from error
    return tuple(verified)


def make_model_serving_runtime_factory(
    role_name: str,
) -> Callable[[Any], Any]:
    """Return the builtin connector factory for one exact vLLM role."""

    normalized = str(role_name).strip().upper()
    if normalized not in {"SCHEDULER", "WORKER"}:
        raise ValueError("streaming runtime role must be SCHEDULER or WORKER")

    def factory(connector: Any) -> Any:
        settings = ModelServingStreamingSettings.from_connector(connector)
        verify_model_serving_lease_contract(settings)
        if normalized == "SCHEDULER":
            return SchedulerStreamingSnapshotAdapter(connector)
        return WorkerStreamingSnapshotAdapter(
            connector,
            settings=settings,
        )

    return factory
