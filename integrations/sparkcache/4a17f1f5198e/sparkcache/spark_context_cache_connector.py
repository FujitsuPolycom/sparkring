"""SparkCache persistent NVMe context-cache connector.

KVConnectorBase_V1 implementation that persists each DCP rank's token shard
of every registered cache layer to rank-local NVMe through a verified-or-recompute
ManifestStore, and restores it after a runtime restart. Supported degrees:
any tensor-parallel size and any decode-context-parallel size that divides
the profile's chunk length (DCP1 stores each rank's full span). The mapping
from a model's cache layers to persistent record families comes from a named
ModelProfile (``sparkcache.spark_context_cache_profiles``). Implemented profiles cover
GLM-5.2 per-token rows and DeepSeek-V4 opaque hybrid-memory-allocator (HMA)
pages.

Restore uses KV-Connector-V1's asynchronous-load contract. A cache hit parks
only that request, allocates private blocks, and queues verification,
assembly, H2D copy, and scatter on a background loader. The request is
reported finished only after its writes have completed; unrelated requests
remain schedulable while it restores.

Verified-or-recompute contract:
- Storage uses content-addressed chunks with per-record SHA-256 and an
  identity pinning model/quant/TP/DCP/shard-rank/chunk geometry.
- A failed asynchronous load publishes its invalid blocks and finished
  request id together. vLLM can then unpark it through the supported clean
  recompute path instead of exposing partially restored state.
- Store failures only log and count; they never fail a request.

Enabled explicitly via --kv-transfer-config naming this module. Without that
argument, vLLM does not instantiate SparkCache.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import math
import os
import re

import queue
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from dataclasses import dataclass, field, replace
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Mapping, Sequence

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
    KVConnectorRole,
    SupportsHMA,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
)
from vllm.logger import init_logger
from sparkcache.request_cache_scope import (
    UNSALTED_SCOPE,
    fingerprint,
    metadata_binding,
    valid_metadata,
    valid_scope,
)

from sparkcache.spark_context_cache_codec import (
    CHUNK_TOKENS,
    CodecError,
    MultimodalFeatureIdentity,
    build_layer_plans,
    chunk_prefix_digests,
    chunk_count,
    classify_layer,
    context_prefix_digest,
    local_slots_for_positions,
    owned_positions,
    pack_positions,
    pack_record,
    unpack_positions,
    unpack_record,
    validate_multimodal_features,
)
from sparkcache.spark_context_cache_config import (
    SHA256_RE,
    parse_connector_config,
)
from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
    encode_page_snapshot,
    materialize_page_extension_capture,
    plan_page_snapshot,
    split_snapshot,
)
from sparkcache.spark_context_cache_restore_timing import RestoreTiming
from sparkcache.request_attribution import RequestAttribution
from sparkcache.held_inventory import HeldInventory
from sparkcache.page_base_read_flights import (
    PageBaseReadEvidence,
    PageBaseReadFlightKey,
    PageBaseReadFlights,
    PageBaseReadResult,
)
from sparkcache.capture_read_leases import CaptureReadLeases
from sparkcache.spark_context_cache_store import (
    CacheIdentity,
    CapacityPolicy,
    ContextChunk,
    EntryKey,
    LookupResult,
    MaintenanceReport,
    ManifestStore,
    PageDeltaDepthExceeded,
    StateRecord,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.spark_context_cache")


def _debug_log(message: str, *args: Any) -> None:
    """Emit optional detail without requiring debug support from embedders."""

    debug = getattr(logger, "debug", None)
    if callable(debug):
        debug(message, *args)

def _operator_reason(value: object) -> str:
    return re.sub(r"\b[0-9a-f]{64}\b", lambda match: match[0][:12], " ".join(str(value).split()))


def _warning(message: str, *args: Any) -> None:
    """Keep complete rejection evidence at DEBUG and shorten operator digests."""
    _debug_log(message, *args)
    logger.warning(message, *tuple(_operator_reason(arg) if isinstance(arg, (str, BaseException))
                                  else arg for arg in args))


_CAPACITY_RETRY_SECONDS = 5.0
_CLEAR_ONCE_LOCK_TIMEOUT_SECONDS = 30.0
_QUORUM_REPORT_BATCH_SIZE = 64
_QUORUM_DELTA_HISTORY_SIZE = 64
_STARTUP_HANDSHAKE_MAX_DIGESTS = 8 * _QUORUM_REPORT_BATCH_SIZE
_QUORUM_PENDING_DELTA_LIMIT = 64
_QUORUM_RETIRED_GENERATION_LIMIT = 64
_QUORUM_DELTA_PROTOCOL = "sparkcache-quorum-delta-v1"
_DIAGNOSTIC_PUBLICATION_SCHEMAS = (
    "", "tail-cow-v1", "page-tail-cow-v1", "page-tail-cow-v2",
    "page-tail-cow-v3-fast", "page-tail-cow-v3-none",
)
_MAX_RESTORE_FLIGHT_FOLLOWERS = 16
_MAX_SHARED_PREFIX_LEASES = 2
_MAX_SHAREABLE_PREFIXES_PER_FLIGHT = 64
_MAX_PAGE_BASE_READ_FLIGHTS = 2
_MAX_PAGE_BASE_READ_MEMBERS = 16
_MAX_PAGE_BASE_BYTES_PER_FLIGHT = 1024**3
_MAX_PAGE_BASE_BYTES_TOTAL = 2 * 1024**3
_CONTEXT_DIGEST_NAMESPACE = "sparkcache-context-v3-request-scope"

# The runtime is deliberately supplied by the embedding process instead of
# importing or constructing the optional streaming-snapshot runtime here.
# A scheduler and each worker have distinct ownership, so factories are keyed
# by connector role.  Installing a factory is an explicit deployment action;
# setting the feature flag without one remains a startup error.
_STREAMING_RUNTIME_FACTORIES: dict[
    KVConnectorRole, Callable[["SparkContextCacheConnector"], Any]
] = {}
_ASYNC_PAGE_RUNTIME_FACTORY: Callable[
    ["SparkContextCacheConnector"], Any
] | None = None


def configure_streaming_snapshot_runtime(
    role: KVConnectorRole,
    factory: Callable[["SparkContextCacheConnector"], Any] | None,
) -> None:
    """Install or remove the explicitly injected streaming runtime factory.

    This is a narrow integration seam for an attested embedding runtime.  It
    does not create, import, or otherwise enable a C++/CUDA runtime by itself.
    ``factory`` receives the connector after its basic vLLM configuration is
    available and must return a role-appropriate adapter.
    """

    if not isinstance(role, KVConnectorRole):
        raise TypeError("streaming runtime role must be a KVConnectorRole")
    if factory is None:
        _STREAMING_RUNTIME_FACTORIES.pop(role, None)
        return
    if not callable(factory):
        raise TypeError("streaming runtime factory must be callable")
    _STREAMING_RUNTIME_FACTORIES[role] = factory


def configure_async_page_capture_runtime(
    factory: Callable[["SparkContextCacheConnector"], Any] | None,
) -> None:
    """Install a GPU-free test or attested manager-page runtime builder."""

    global _ASYNC_PAGE_RUNTIME_FACTORY
    if factory is not None and not callable(factory):
        raise TypeError("manager-page capture runtime factory must be callable")
    _ASYNC_PAGE_RUNTIME_FACTORY = factory


def _load_cuda_components() -> SimpleNamespace:
    """Load the optional SparkCache CUDA placement components lazily."""

    placement = importlib.import_module("sparkcache.spark_context_cache_cuda_placement")
    restore = importlib.import_module("sparkcache.spark_context_cache_cuda_restore")
    hybrid_restore = importlib.import_module(
        "sparkcache.spark_context_cache_cuda_hybrid_restore"
    )
    binding = importlib.import_module("sparkcache.spark_cache_cuda")
    return SimpleNamespace(
        CudaPlacementLibrary=placement.CudaPlacementLibrary,
        CudaPlacementAdapter=placement.CudaPlacementAdapter,
        ArenaMode=placement.ArenaMode,
        RecordKind=placement.RecordKind,
        execute_cuda_restore=restore.execute_cuda_restore,
        execute_cuda_hybrid_restore=hybrid_restore.execute_cuda_hybrid_restore,
        execute_cuda_hybrid_placement=(hybrid_restore.execute_cuda_hybrid_placement),
        bind_page_reference=binding.bind_page_reference,
        hybrid_page_cuda_capability=binding.CAP_HYBRID_PAGE_CUDA,
    )


# Private compatibility alias for embedders that imported the earlier helper.
_load_native_components = _load_cuda_components


@dataclass
class _ReqPlan:
    request_id: str
    digest: str
    span_tokens: int
    block_ids: tuple[int, ...]
    is_store: bool
    block_ids_by_group: tuple[tuple[int, ...], ...] = ()
    # Store plans retain the exact token prefix whose digest names the
    # manifest. Workers need these tokens only after a durable row snapshot
    # commit, when ManifestStore derives authenticated sparse prefix aliases.
    # Restore plans and opaque block-page snapshots leave this empty.
    token_ids: tuple[int, ...] = ()
    # Tail publication reuses the authenticated chunks ending at this exact
    # stored prefix. Empty fields select ordinary full-snapshot publication.
    base_context_digest: str = ""
    base_span_tokens: int = 0
    # vLLM may retain a recurrent replay-boundary page outside the request's
    # arithmetic block-table slot after later forward work advances the live
    # state. Each pair is an exact (group index, physical block id) override
    # proven by vLLM for this plan's span_tokens boundary.
    recurrent_boundary_blocks: tuple[tuple[int, int], ...] = ()
    # Authenticated row-prefix roots whose descriptors must match the leading
    # descriptors of this plan.  Workers validate these roots before the
    # leader's blocks may back a shorter shared-prefix lease.
    shared_segments: tuple[tuple[str, int], ...] = ()
    # True when digest includes a media content identity. Token-only prefix
    # alias and tail derivation cannot reproduce that identity and must not
    # publish alternative keys for this plan.
    has_multimodal_identity: bool = False
    capture_job_id: str = ""
    request_scope: str | None = None
    # Consistency check for the trusted scheduler/worker channel, not a MAC.
    scope_binding: str = ""

    def __post_init__(self) -> None:
        if not self.scope_binding:
            self.scope_binding = metadata_binding(self.request_scope, self.digest)

    @property
    def group_block_ids(self) -> tuple[tuple[int, ...], ...]:
        return self.block_ids_by_group or (self.block_ids,)


@dataclass
class _RestoreFlight:
    """One scheduler-owned restore whose result may satisfy peer requests."""

    digest: str
    span_tokens: int
    leader_request_id: str
    followers: set[str] = field(default_factory=set)
    restored_block_ids: frozenset[int] = frozenset()
    dispatched: bool = False
    workers_finished: bool = False
    leader_finished: bool = False
    # Every candidate is a full-quorum prefix digest of the leader request.
    # Row storage can prove that such a root is the leading descriptor graph
    # of the restored root. Opaque page snapshots deliberately leave this
    # empty because their recurrent boundary state cannot be truncated.
    shareable_prefixes: tuple[tuple[int, str], ...] = ()
    segment_digest: str | None = None
    segment_span_tokens: int = 0
    lease_digest: str | None = None
    lease_span_tokens: int = 0
    lease_published_at: float | None = None
    lease_expires_at: float | None = None
    started_ns: int = 0

    @property
    def lease_published(self) -> bool:
        return self.lease_expires_at is not None


@dataclass(frozen=True)
class _RestoreFollower:
    """One request waiting to attach a bounded verified prefix lease."""

    flight_digest: str
    lease_digest: str
    span_tokens: int


@dataclass(frozen=True)
class _QueuedLoad:
    plan: _ReqPlan
    timing: RestoreTiming
    prior_cuda_event: Any | None = None
    prior_cuda_error: str | None = None


@dataclass(frozen=True)
class _QueuedLoadBatch:
    """Requests awaiting background page-base metadata and cohort admission."""

    loads: tuple[_QueuedLoad, ...]


@dataclass(frozen=True)
class _StreamingSnapshotOffer:
    """Scheduler promise that becomes a worker completion after forward.

    ``completed_tokens`` is intentionally only a scheduler-side watermark.
    The worker must not hand it to the streaming runtime until
    :meth:`wait_for_save`, which vLLM calls after the corresponding forward.
    ``block_ids`` is the complete table known for that watermark, not merely
    the blocks allocated by the current cached-request step.
    """

    request_id: str
    digest: str
    span_tokens: int
    completed_tokens: int
    block_ids: tuple[int, ...]
    request_scope: str | None = None
    scope_binding: str = ""

    def __post_init__(self) -> None:
        if not self.scope_binding:
            object.__setattr__(self, "scope_binding",
                               metadata_binding(self.request_scope, self.digest))

    @property
    def span(self) -> int:
        return self.span_tokens

    @property
    def promised_completed_tokens(self) -> int:
        return self.completed_tokens

    @property
    def full_block_table(self) -> tuple[int, ...]:
        return self.block_ids


@dataclass
class _PendingStreamingCommit:
    """Durable streaming manifest awaiting capacity-safe advertisement."""

    receipt: Any
    accounted: bool = False


@dataclass(frozen=True)
class _StoreSnapshot:
    """CPU-owned immutable state required to commit one cache entry.

    No live KV tensor is retained after construction, so vLLM may reuse the
    source blocks as soon as ``wait_for_save`` returns.
    """

    plan: _ReqPlan
    rank: int
    identity: CacheIdentity
    positions: tuple[int, ...]
    layer_bytes: dict[str, bytes]
    layer_plans: tuple[Any, ...]
    record_kinds: tuple[str, ...]
    logical_start: int = 0


@dataclass(frozen=True)
class _HybridStoreSnapshot:
    plan: _ReqPlan
    rank: int
    identity: CacheIdentity
    positions: tuple[int, ...]
    encoded_pages: Any
    block_counts: tuple[int, ...]


class _HybridSnapshotChunks(Sequence[ContextChunk]):
    def __init__(self, snapshot: _HybridStoreSnapshot, chunk_tokens: int) -> None:
        self._snapshot = snapshot
        self._count = chunk_count(snapshot.plan.span_tokens, chunk_tokens)
        self._chunk_tokens = chunk_tokens
        self._parts = split_snapshot(snapshot.encoded_pages, self._count)

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self, index: int | slice
    ) -> ContextChunk | tuple[ContextChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        start = index * self._chunk_tokens
        end = (index + 1) * self._chunk_tokens
        return ContextChunk(
            logical_start=start,
            logical_end=end,
            records={
                StateRecord.LOGICAL_POSITIONS: pack_positions(range(start, end)),
                StateRecord.TARGET_CKV: self._parts[index],
            },
        )


class _SnapshotChunks(Sequence[ContextChunk]):
    """Build one encoded chunk at a time from an immutable CPU snapshot."""

    def __init__(
        self,
        snapshot: _StoreSnapshot,
        dcp_degree: int,
        chunk_tokens: int = CHUNK_TOKENS,
    ) -> None:
        if chunk_tokens <= 0 or chunk_tokens % dcp_degree:
            raise CodecError(
                f"chunk_tokens {chunk_tokens} is not divisible by"
                f" dcp_degree {dcp_degree}"
            )
        self._snapshot = snapshot
        self._dcp_degree = dcp_degree
        self._chunk_tokens = chunk_tokens
        self._count = chunk_count(snapshot.plan.span_tokens, chunk_tokens)
        if snapshot.logical_start:
            self._count -= chunk_count(snapshot.logical_start, chunk_tokens)
        self._rows_per_chunk = chunk_tokens // dcp_degree

    def __len__(self) -> int:
        return self._count

    def __getitem__(
        self, index: int | slice
    ) -> ContextChunk | tuple[ContextChunk, ...]:
        if isinstance(index, slice):
            return tuple(self[item] for item in range(*index.indices(self._count)))
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        row_slice = slice(
            index * self._rows_per_chunk,
            (index + 1) * self._rows_per_chunk,
        )
        records = {
            StateRecord.LOGICAL_POSITIONS: pack_positions(
                self._snapshot.positions[row_slice]
            )
        }
        for kind in self._snapshot.record_kinds:
            rows_map = {}
            for plan_entry in self._snapshot.layer_plans:
                if plan_entry.record_kind != kind:
                    continue
                width = plan_entry.bytes_per_token
                full = self._snapshot.layer_bytes[plan_entry.name]
                rows_map[plan_entry.name] = full[
                    row_slice.start * width : row_slice.stop * width
                ]
            records[StateRecord(kind)] = pack_record(
                self._snapshot.layer_plans,
                kind,
                rows_map,
                self._rows_per_chunk,
            )
        return ContextChunk(
            logical_start=self._snapshot.logical_start + index * self._chunk_tokens,
            logical_end=(
                self._snapshot.logical_start + (index + 1) * self._chunk_tokens
            ),
            records=records,
        )


@dataclass
class SparkCacheConnectorMetadata(KVConnectorMetadata):
    plans: list[_ReqPlan] = field(default_factory=list)
    streaming_snapshot_offers: list[_StreamingSnapshotOffer] = field(
        default_factory=list
    )
    preempted_request_ids: tuple[str, ...] = ()

    # Keep the transport field descriptive while making inspection by an
    # embedding runtime pleasantly direct.
    @property
    def offers(self) -> list[_StreamingSnapshotOffer]:
        return self.streaming_snapshot_offers


@dataclass
class SparkCacheReadCompletionMetadata(KVConnectorWorkerMetadata):
    """Distinct physical ranks whose capture reads finished for each job."""

    completed_reads: dict[str, set[int]] = field(default_factory=dict)
    uncertain_reads: dict[str, set[int]] = field(default_factory=dict)

    def aggregate(self, other: "KVConnectorWorkerMetadata"):
        if not isinstance(other, SparkCacheReadCompletionMetadata):
            raise TypeError("incompatible SparkCache read-completion metadata")
        for job, ranks in other.completed_reads.items():
            self.completed_reads.setdefault(job, set()).update(ranks)
        for job, ranks in other.uncertain_reads.items():
            self.uncertain_reads.setdefault(job, set()).update(ranks)
        return self


@dataclass
class SparkCacheHandshakeMetadata(KVConnectorHandshakeMetadata):
    """One worker's bounded startup manifest inventory.

    Each report contains at most ``_QUORUM_REPORT_BATCH_SIZE`` compatible
    digests and a separately bounded batch of incompatible-schema evidence.
    The engine transports this metadata once after every worker has
    registered its KV caches and before it accepts requests.
    """

    reports: tuple[dict[str, Any], ...] = ()


@dataclass
class SparkCacheStats(KVConnectorStats):
    """Per-rank "digests I can serve" reports, merged across workers.

    MUST be defined at module scope: these objects are pickled across the
    worker->engine shared-memory queue, and a class created inside a
    function is not picklable.
    """

    # vLLM replaces the stats accumulator after every logging interval.
    # Worker UUIDs identify a serving engine without mixing its counters
    # with other engines or retaining history across a worker restart.
    _log_history: ClassVar[OrderedDict[tuple, dict[str, int | float]]] = OrderedDict()

    def reset(self):
        self.data = {"reports": []}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        mine = list(self.data.get("reports", []))
        theirs = list(getattr(other, "data", {}).get("reports", []))
        merged: dict[int, dict[str, Any]] = {}
        for report in mine + theirs:
            if isinstance(report, dict) and isinstance(report.get("rank"), int):
                normalized = dict(report)
                if isinstance(report.get("held"), list):
                    normalized["held"] = list(report["held"])
                for field in (
                    "delta",
                    "checkpoint",
                    "streaming",
                    "capacity",
                    "async_capture",
                    "publication",
                    "publication_work",
                ):
                    if isinstance(report.get(field), dict):
                        normalized[field] = dict(report[field])
                merged[report["rank"]] = normalized
        self.data = {"reports": [merged[rank] for rank in sorted(merged)]}
        return self

    def reduce(self) -> dict[str, int | float]:
        reports = self.data.get("reports", [])
        reduced: dict[str, int | float] = {
            "sparkcache_ranks": len(reports),
            "sparkcache_entries": sum(
                int(r.get("held_count", len(r.get("held", [])))) for r in reports
            ),
        }
        counters: list[dict[str, Any]] = [
            report["counters"]
            for report in reports
            if isinstance(report.get("counters"), dict)
        ]
        if counters:
            keys = {key for status in counters for key in status}
            reduced.update(
                {
                    f"sparkcache_counter_{key}": sum(
                        int(status.get(key, 0)) for status in counters
                    )
                    for key in keys
                }
            )
        streaming = [
            report.get("streaming")
            for report in reports
            if isinstance(report.get("streaming"), dict)
        ]
        if streaming:
            active_contexts = sum(
                int(status.get("active_contexts", 0)) for status in streaming
            )
            active_leases = sum(
                int(status.get("active_leases", 0)) for status in streaming
            )
            active_tickets = sum(
                int(status.get("active_tickets", 0)) for status in streaming
            )
            if active_contexts:
                reduced["sparkcache_streams"] = active_contexts
            if active_leases:
                reduced["sparkcache_leases"] = active_leases
            if active_tickets:
                reduced["sparkcache_tickets"] = active_tickets
        capacity = [
            report.get("capacity")
            for report in reports
            if isinstance(report.get("capacity"), dict)
        ]
        if capacity:
            used_bytes = sum(int(status.get("bytes", 0)) for status in capacity)
            maximum_bytes = sum(
                int(status.get("max_bytes", 0)) for status in capacity
            )
            reduced.update(
                sparkcache_used_gib=round(used_bytes / 1024**3, 1),
                sparkcache_limit_gib=round(maximum_bytes / 1024**3, 1),
                sparkcache_healthy=int(
                    all(
                        bool(status.get("capacity_satisfied", False))
                        for status in capacity
                    )
                ),
            )
            alerts = {
                "sparkcache_evicted": sum(
                    int(status.get("manifests_evicted", 0)) for status in capacity
                ),
                "sparkcache_reclaimed_gib": round(
                    sum(
                        int(status.get("bytes_reclaimed", 0))
                        for status in capacity
                    )
                    / 1024**3,
                    1,
                ),
                "sparkcache_pending": sum(
                    int(status.get("pending_streaming_commits", 0))
                    for status in capacity
                ),
                "sparkcache_stream_evicted": sum(
                    int(status.get("streaming_store_evicted", 0))
                    for status in capacity
                ),
                "sparkcache_invalid_receipts": sum(
                    int(status.get("invalid_streaming_receipts", 0))
                    for status in capacity
                ),
                "sparkcache_dropped_commits": sum(
                    int(status.get("shutdown_dropped_streaming_commits", 0))
                    for status in capacity
                ),
            }
            reduced.update({key: value for key, value in alerts.items() if value})
            for name in ("deletion_attempts", "budget_exhausted", "skipped_cooldown"):
                reduced[f"sparkcache_maintenance_{name}"] = sum(
                    int(status.get(f"maintenance_{name}", 0)) for status in capacity
                )
        async_capture = [
            report.get("async_capture")
            for report in reports
            if isinstance(report.get("async_capture"), dict)
        ]
        if async_capture:
            delayed_rank_slots = sum(
                int(status.get("delayed_requests", 0))
                for status in async_capture
            )
            reduced.update(
                sparkcache_capture_pending_rank_slots=sum(
                    int(status.get("pending_requests", 0))
                    for status in async_capture
                ),
                sparkcache_capture_completed_rank_slots=sum(
                    int(status.get("completed_notifications", 0))
                    for status in async_capture
                ),
                sparkcache_capture_delayed_requests=max(
                    int(status.get("delayed_requests", 0))
                    for status in async_capture
                ),
                sparkcache_capture_delayed_rank_slots=delayed_rank_slots,
                sparkcache_capture_retained_pages=sum(
                    int(status.get("retained_manager_pages", 0))
                    for status in async_capture
                ),
                sparkcache_capture_oldest_delayed_ms=max(
                    float(status.get("oldest_delayed_ms", 0.0))
                    for status in async_capture
                ),
                sparkcache_capture_uncertain_ranks=sum(
                    int(bool(status.get("ownership_uncertain", False)))
                    for status in async_capture
                ),
                sparkcache_capture_busy_ranks=sum(
                    int(bool(status.get("store_inflight", False)))
                    for status in async_capture
                ),
            )
        publication_work = [
            report["publication_work"]
            for report in reports
            if isinstance(report.get("publication_work"), dict)
        ]
        if publication_work:
            reduced.update(
                sparkcache_publication_pending_rank_slots=sum(
                    int(status.get("pending", 0)) for status in publication_work
                ),
                sparkcache_publication_oldest_pending_ms=max(
                    float(status.get("oldest_pending_ms", 0.0)) for status in publication_work
                ),
                sparkcache_maintenance_active_ranks=sum(
                    int(bool(status.get("maintenance_active", False))) for status in publication_work
                ),
            )
        publication = [
            report.get("publication")
            for report in reports
            if isinstance(report.get("publication"), dict)
        ]
        if publication:
            fields = {
                "committed_publications": "sparkcache_publications",
                "logical_payload_bytes": "sparkcache_payload_bytes",
                "committed_unique_object_bytes": (
                    "sparkcache_unique_bytes"
                ),
                "staged_write_bytes": "sparkcache_staged_bytes",
                "deduplicated_bytes": "sparkcache_dedup_bytes",
                "aborted_staged_write_bytes": "sparkcache_aborted_bytes",
                "failed_staged_write_bytes": "sparkcache_failed_bytes",
            }
            reduced.update(
                {
                    metric: sum(int(status.get(field, 0)) for status in publication)
                    for field, metric in fields.items()
                }
            )
            failures = {
                "sparkcache_publication_aborted": sum(
                    int(status.get("aborted_publications", 0))
                    for status in publication
                ),
                "sparkcache_publication_failed": sum(
                    int(status.get("failed_publications", 0))
                    for status in publication
                ),
            }
            reduced.update({key: value for key, value in failures.items() if value})
        return reduced

    def format_log_lines(self) -> tuple[str, ...]:
        """Return short human-readable summaries without dropping raw metrics."""

        reduced = self.reduce()
        reports = self.data.get("reports", [])
        generations = tuple(sorted(
            (report["rank"], report["generation"]) for report in reports
            if type(report.get("rank")) is int
            and isinstance(report.get("generation"), str) and report["generation"]
        ))
        history_key = generations if generations and len(generations) == len(reports) else None
        previous = (
            self._log_history.get(history_key, {}) if history_key is not None
            else getattr(self, "_logged_metrics", {})
        )
        changed = {
            key: value for key, value in reduced.items()
            if value != previous.get(key, 0)
        }
        # Sparse gauges disappearing from reduce() must report their return to zero.
        changed.update({key: 0 for key, value in previous.items()
                        if value and key not in reduced})
        self._logged_metrics = dict(reduced)
        if history_key is not None:
            self._log_history[history_key] = dict(reduced)
            self._log_history.move_to_end(history_key)
            while len(self._log_history) > 128:
                self._log_history.popitem(last=False)
        if not changed:
            return ()
        return ("sparkcache: counters " + " ".join(
            f"{key.removeprefix('sparkcache_').removeprefix('counter_')}={value}"
            for key, value in sorted(changed.items())
        ),)

    def is_empty(self) -> bool:
        return not self.data.get("reports")


class SparkCachePromMetrics(KVConnectorPromMetrics):
    """Prometheus gauges for reported capture, publication, and maintenance work."""

    _GAUGES = {
        "sparkcache_publication_pending_rank_slots": (
            "vllm:sparkcache_publication_pending_rank_slots",
            "Pending saver admissions summed across physical ranks at their last worker reports.",
            1.0,
        ),
        "sparkcache_publication_oldest_pending_ms": (
            "vllm:sparkcache_publication_oldest_pending_seconds",
            "Oldest saver admission age in seconds at the last worker reports.",
            0.001,
        ),
        "sparkcache_maintenance_active_ranks": (
            "vllm:sparkcache_maintenance_active_ranks",
            "Physical ranks reporting an active capacity scan or reconciliation at their last worker reports.",
            1.0,
        ),
        "sparkcache_maintenance_deletion_attempts": (
            "vllm:sparkcache_maintenance_deletion_attempts",
            "Reported cumulative unlink attempts summed across physical ranks; resets with workers.",
            1.0,
        ),
        "sparkcache_maintenance_budget_exhausted": (
            "vllm:sparkcache_maintenance_budget_exhausted",
            "Reported cumulative passes deferring deletion work across physical ranks.",
            1.0,
        ),
        "sparkcache_maintenance_skipped_cooldown": (
            "vllm:sparkcache_maintenance_skipped_cooldown",
            "Reported cumulative cooldown skips across physical ranks.",
            1.0,
        ),
        "sparkcache_capture_delayed_requests": (
            "vllm:sparkcache_capture_delayed_requests",
            "Maximum delayed SparkCache capture requests on any physical rank.",
            1.0,
        ),
        "sparkcache_capture_delayed_rank_slots": (
            "vllm:sparkcache_capture_delayed_rank_slots",
            "Sum of delayed SparkCache request ownership records across ranks.",
            1.0,
        ),
        "sparkcache_capture_retained_pages": (
            "vllm:sparkcache_capture_retained_manager_pages",
            "Physical manager pages retained for asynchronous SparkCache capture.",
            1.0,
        ),
        "sparkcache_capture_oldest_delayed_ms": (
            "vllm:sparkcache_capture_oldest_delayed_seconds",
            "Age in seconds of the oldest asynchronous SparkCache capture ownership.",
            0.001,
        ),
        "sparkcache_capture_uncertain_ranks": (
            "vllm:sparkcache_capture_ownership_uncertain_ranks",
            "Physical ranks whose SparkCache capture-page ownership is uncertain.",
            1.0,
        ),
        "sparkcache_counter_restore_hit": (
            "vllm:sparkcache_restore_hits",
            "Cumulative verified external restores summed across physical worker ranks; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_restore_skip_local_prefix": (
            "vllm:sparkcache_restore_skip_local_prefix",
            "Cumulative restore lookups refused because the request already held locally cached prefix blocks; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_restore_skip_oversize": (
            "vllm:sparkcache_restore_skip_oversize",
            "Cumulative lookups whose aligned span exceeded the maximum span; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_restore_skip_backlog": (
            "vllm:sparkcache_restore_skip_backlog",
            "Cumulative lookups refused because pending restores reached the configured limit; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_quorum_incomplete": (
            "vllm:sparkcache_quorum_incomplete",
            "Cumulative lookups with no fully quorum-covered candidate manifest; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_multimodal_bypass": (
            "vllm:sparkcache_multimodal_bypass",
            "Cumulative requests skipped because a multimodal feature could not be proven; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_store_skipped_delayed_limit": (
            "vllm:sparkcache_store_skipped_delayed_limit",
            "Cumulative store plans dropped because delayed capture jobs reached the configured limit; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_load_verified": (
            "vllm:sparkcache_load_verified",
            "Cumulative worker restores whose bytes verified; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_load_failed": (
            "vllm:sparkcache_load_failed",
            "Cumulative worker restores that failed verification and degraded to recompute; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_attribution_requests_complete": (
            "vllm:sparkcache_attribution_requests_complete",
            "Cumulative requests whose attribution summary completed cleanly; resets with workers.",
            1.0,
        ),
        "sparkcache_counter_attribution_requests_incomplete": (
            "vllm:sparkcache_attribution_requests_incomplete",
            "Cumulative requests whose attribution summary was incomplete; resets with workers.",
            1.0,
        ),
    }

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )
        self._capture_gauges: dict[str, dict[int, Any]] = {}
        for key, (name, documentation, _scale) in self._GAUGES.items():
            definition = self._gauge_cls(
                name=name,
                documentation=documentation,
                labelnames=labelnames,
            )
            self._capture_gauges[key] = {
                engine_idx: definition.labels(*labelvalues)
                for engine_idx, labelvalues in per_engine_labelvalues.items()
            }

    def observe(
        self,
        transfer_stats_data: dict[str, Any],
        engine_idx: int = 0,
    ) -> None:
        reduced = SparkCacheStats(data=transfer_stats_data).reduce()
        for key, (_name, _documentation, scale) in self._GAUGES.items():
            self._capture_gauges[key][engine_idx].set(
                float(reduced.get(key, 0)) * scale
            )


def _multimodal_feature_identities(
    request: object,
    token_count: int,
) -> tuple[MultimodalFeatureIdentity, ...] | None:
    """Extract complete vLLM media identity, or None when it is unprovable.

    Current vLLM request and scheduler-transfer objects retain each feature's
    content identifier and placeholder range even when its tensor data is
    stripped after a local prefix-cache hit. Older ``mm_hashes``-only objects
    do not expose enough placeholder geometry for prefix-stable digests.
    """

    raw_features = getattr(request, "mm_features", None)
    if raw_features is not None:
        try:
            raw_features = tuple(raw_features)
        except TypeError:
            return None
    if raw_features:
        try:
            features = tuple(
                MultimodalFeatureIdentity(
                    modality=feature.modality,
                    identifier=feature.identifier,
                    offset=feature.mm_position.offset,
                    length=feature.mm_position.length,
                )
                for feature in raw_features
            )
            return validate_multimodal_features(features, token_count)
        except (AttributeError, CodecError, TypeError):
            return None
    if getattr(request, "mm_hashes", None):
        return None
    return ()


class SparkContextCacheConnector(KVConnectorBase_V1, SupportsHMA):
    """Store/restore each rank's DCP shard on rank-local NVMe."""

    # Capture/restore kernels use CUDA virtual addresses while the registered
    # tensors remain owned. No KV physical pages are registered for RDMA.
    supports_cuda_vmm = True

    @property
    def _held(self) -> HeldInventory:
        return self._held_inventory

    @_held.setter
    def _held(self, values: set[str] | HeldInventory) -> None:
        # In-place operators assign their result back through this setter.
        if values is not getattr(self, "_held_inventory", None):
            self._held_inventory = HeldInventory(values)

    # Exact-vLLM runtimes use this opt-in before pinning and exporting aligned
    # recurrent replay-boundary blocks. Synchronous capture detaches them in
    # wait_for_save. Explicit asynchronous page capture retains all groups until
    # its completion event or preemption drain proves CUDA stopped reading.
    supports_recurrent_boundary_blocks = True

    @property
    def requires_kv_delivery(self) -> bool:
        """Optional persistence failures remain cache misses, not request failures."""
        return False

    @property
    def recurrent_boundary_granularity(self) -> int:
        """Token boundary used for connector-owned recurrent publication."""

        return self._chunk_tokens

    def get_recurrent_publication_boundaries(
        self, request: "Request"
    ) -> tuple[int, ...]:
        """Propose the exact eligible store boundary without mutating state."""

        if (
            not self._cache_available
            or not self._store_enabled
            or self._streaming_snapshots_enabled
            or not self._recurrent_group_indexes
        ):
            return ()
        prompt_token_ids = getattr(request, "prompt_token_ids", None) or ()
        span = self._aligned_span(len(prompt_token_ids))
        if not self._min_span <= span <= self._max_span:
            return ()
        return (span,)

    configure_streaming_snapshot_runtime = staticmethod(
        configure_streaming_snapshot_runtime
    )
    configure_async_page_capture_runtime = staticmethod(
        configure_async_page_capture_runtime
    )

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._kv_cache_config = kv_cache_config
        config = parse_connector_config(
            vllm_config, self._kv_transfer_config, kv_cache_config
        )
        self._config = config
        self._trace_reuse_enabled = os.environ.get("SPARK_CONTEXT_CACHE_TRACE_REUSE") == "1"
        self.request_cache_events_enabled = (
            self._trace_reuse_enabled and role is KVConnectorRole.SCHEDULER
        )
        self._request_attribution: dict[str, RequestAttribution] = {}
        self._block_size = config.block_size
        self._tp_degree = config.tp_degree
        self._dcp_degree = config.dcp_degree
        self._cp_kv_cache_interleave_size = config.cp_kv_cache_interleave_size
        self._profile = config.profile
        self._storage_mode = config.storage_mode
        self._publication_schema = config.publication_schema
        self._group_topology = config.group_topology
        self._prefill_replay_tokens = config.prefill_replay_tokens
        self._prefix_state_alignment = config.prefix_state_alignment
        self._recurrent_group_indexes = frozenset(
            group_index
            for group_index, topology in enumerate(self._group_topology)
            if topology["reuse_policy"] == "recurrent_align"
        )
        self._chunk_tokens = config.chunk_tokens
        self._root = config.root
        self._store = ManifestStore(
            self._root,
            page_delta_stage_limit_count=config.page_delta_stage_limit_count,
            hot_page_snapshot_bytes=config.hot_page_snapshot_bytes,
        )
        self._clear_once_token = config.clear_once_token
        self._cache_available = True
        if self._clear_once_token:
            try:
                cleared = self._store.clear_once(
                    self._clear_once_token,
                    lock_timeout_seconds=_CLEAR_ONCE_LOCK_TIMEOUT_SECONDS,
                )
            except Exception as error:  # noqa: BLE001 - serving remains available
                self._cache_available = False
                _warning(
                    "sparkcache: clear-once did not complete;"
                    " persistent cache is disabled for role=%s: %s",
                    role.name,
                    error,
                )
            else:
                logger.debug(
                    "sparkcache: clear-once %s for role=%s",
                    "removed owned cache data" if cleared else "already completed",
                    role.name,
                )
        self._capacity_policy = config.capacity_policy
        self._min_span = config.min_span
        self._max_span = config.max_span
        self._access_mode = config.access_mode
        self._store_enabled = config.store_enabled and self._cache_available
        self._restore_enabled = config.restore_enabled and self._cache_available
        self._streaming_snapshots_enabled = (
            config.streaming_snapshots_enabled and self._cache_available
        )
        if config.async_page_capture_enabled and not self._cache_available:
            raise RuntimeError(
                "sparkcache: asynchronous manager-page capture cannot"
                " start while persistent cache initialization is unavailable"
            )
        self._async_page_capture_enabled = config.async_page_capture_enabled
        self._page_snapshot_interval_tokens = config.page_snapshot_interval_tokens
        self._async_page_capture_runtime: Any = None
        self._async_page_capture_settings: Any = None
        self._async_page_capture_eligible: set[str] = set()
        self._max_delayed_stores = config.max_delayed_stores
        self._async_page_capture_reservations: dict[str, int] = {}
        self._capture_read_leases: CaptureReadLeases | None = None
        self._capture_read_done: set[str] = set()
        self._capture_read_uncertain: set[str] = set()
        self._capture_read_lock = threading.Lock()
        self._streaming_runtime: Any = None
        if self._streaming_snapshots_enabled:
            # An explicit opt-in never falls back to end-of-prefill snapshots.
            # Preserve an explicitly injected test/deployment adapter. When
            # none exists, import the builtin factory only on this opt-in
            # path; default-off scheduler and worker processes never import
            # factory and C++/CUDA ring modules.
            if role not in _STREAMING_RUNTIME_FACTORIES:
                from sparkcache.streaming.factory import (
                    make_model_serving_runtime_factory,
                )

                _STREAMING_RUNTIME_FACTORIES[role] = make_model_serving_runtime_factory(
                    role.name
                )
            # Resolve the adapter before C++/CUDA import/allocation so a
            # partially configured deployment is rejected at startup.
            self._install_streaming_runtime(role)
        if self._async_page_capture_enabled:
            from sparkcache.streaming.manager_page_factory import (
                ManagerPageCaptureSettings,
                verify_manager_page_lease_contract,
            )

            settings = ManagerPageCaptureSettings.from_connector(self)
            verify_manager_page_lease_contract(settings)
            self._async_page_capture_settings = settings
        self._native_restore_enabled = (
            config.native_restore_enabled and self._cache_available
        )
        self._native_library_path = config.native_library_path
        self._native_library_sha256 = config.native_library_sha256
        self._native_arena_bytes = config.native_arena_bytes
        self._native_io_workers = config.native_io_workers
        self._identity_base = config.identity_base
        self._request_cache_scopes: dict[str, str | None] = {}
        # Legacy digests cannot reveal which request salt produced their bytes.
        # All requests, including unsalted text, therefore use a distinct
        # namespace. Media content/range identity remains bound independently.
        self._context_digest_salt = (
            f"{_CONTEXT_DIGEST_NAMESPACE}:"
            f"{config.build_identity(0, 0).storage_key}"
        )
        self._scheduler_probe = config.scheduler_probe
        self._shard_rank = 0
        self._capacity_estimated_bytes = 0
        self._capacity_status: dict[str, int | bool] = {
            "max_bytes": self._capacity_policy.max_bytes,
            "low_watermark_bytes": self._capacity_policy.low_watermark_bytes,
            "ttl_seconds": self._capacity_policy.ttl_seconds,
            "bytes": 0,
            "bytes_exact": False,
            "capacity_satisfied": True,
        }
        self._capacity_wakeup = threading.Event()
        self._capacity_stop = threading.Event()
        self._capacity_thread: threading.Thread | None = None
        # Filesystem maintenance and its survivor publication form one serial
        # capacity operation. This lock is never taken by inference callbacks;
        # streaming callbacks enqueue a receipt and wake the janitor instead.
        self._capacity_lock = threading.RLock()
        self._capacity_maintenance_depth = 0
        self._capacity_commit_queue: "queue.SimpleQueue[tuple[str, Any]]" = (
            queue.SimpleQueue()
        )
        self._capacity_handoff_cv = threading.Condition()
        self._streaming_capacity_pending: set[str] = set()
        # Worker state.
        self._layer_tensors: dict[str, torch.Tensor] = {}
        self._plans = None
        self._page_layout: PageLayout | None = None
        self._record_kinds: tuple[str, ...] = ()
        # Digests this rank can offer: either discovered from a structurally
        # valid manifest at startup or published by a durable ManifestStore
        # commit. Every restore remains the byte/hash integrity boundary.
        # Reported to the scheduler through bounded deltas and rolling
        # checkpoints so admission can require unanimity without making
        # each scheduler step proportional to the retained inventory.
        self._held = set()
        self._stats_observed_held: set[str] = set()
        self._stats_readvertise_held: set[str] = set()
        self._stats_observed_inventory: HeldInventory | None = None
        self._stats_observed_revision = -1
        self._stats_sequence = 0
        self._stats_delta_history: list[dict[str, Any]] = []
        self._stats_delta_cursor = 0
        self._stats_checkpoint_items: tuple[str, ...] = ()
        self._stats_checkpoint_sequence = 0
        self._stats_checkpoint_cycle = 1
        self._stats_checkpoint_index = 0
        self._incompatible_inventory: tuple[tuple[str, str], ...] = ()
        self._incompatible_cursor = 0
        self._pending_saves: dict[str, dict[str, torch.Tensor]] = {}
        self._load_errors: set[int] = set()
        self._load_lock = threading.Lock()
        self._load_cv = threading.Condition(self._load_lock)
        self._store_cv = threading.Condition(self._load_lock)
        self._store_queue: "queue.SimpleQueue[_StoreSnapshot | _HybridStoreSnapshot | None]" = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._store_inflight = 0
        self._store_pending_started_ns: int | None = None
        self._publication_base_pins: dict[str, EntryKey] = {}
        self._store_accepting = True
        self._load_queue: "queue.SimpleQueue[_QueuedLoad | _QueuedLoadBatch | None]" = queue.SimpleQueue()
        self._load_threads: list[threading.Thread] = []
        self._load_thread_limit = config.load_thread_limit
        if self._native_restore_enabled and self._storage_mode != "block_pages_v1":
            # One SparkCache CUDA adapter owns one transaction and two arenas.
            # Keep row restores serialized; CUDA page mode creates one adapter
            # per bounded load lane instead.
            self._load_thread_limit = 1
        self._inflight_load_reqs: set[str] = set()
        self._cancelled_load_reqs: set[str] = set()
        self._load_stop_requested = False
        self._load_stops_queued = False
        self._finished_load_reqs: set[str] = set()
        self._load_stream: Any = None
        self._native_adapter: Any = None
        self._native_adapters: list[Any] = []
        self._native_execute_restore: Any = None
        self._native_execute_hybrid_restore: Any = None
        self._native_execute_hybrid_placement: Any = None
        self._native_required_record_mask = 0
        self._page_base_reads = PageBaseReadFlights(
            max_flights=_MAX_PAGE_BASE_READ_FLIGHTS,
            max_members=_MAX_PAGE_BASE_READ_MEMBERS,
            max_bytes_per_flight=_MAX_PAGE_BASE_BYTES_PER_FLIGHT,
            max_bytes_total=_MAX_PAGE_BASE_BYTES_TOTAL,
        )
        self._page_base_plan_keys: dict[str, PageBaseReadFlightKey] = {}
        self._deferred_page_base_loads: dict[str, _QueuedLoad] = {}
        # Scheduler state.
        self._need_load: dict[str, tuple[str, int]] = {}
        # Bound on concurrently promised async restores. Enforced at
        # admission in get_num_new_matched_tokens; entries are consumed by
        # update_state_after_alloc / build_connector_meta and cleared by
        # request_finished, so the map cannot grow past in-flight requests.
        self._max_pending_restores = config.max_pending_restores
        self._shared_prefix_lease_ttl_seconds = (
            config.shared_prefix_lease_ttl_seconds
        )
        # Coalesce requests selecting the same persistent digest around one
        # restore. Followers own no restore blocks: they remain ordinary
        # waiting requests until vLLM publishes the leader's verified blocks
        # into its local prefix cache. Unique flights are bounded by the same
        # admission limit as SparkCache CUDA load lanes; excess unrelated work
        # recomputes immediately instead of joining a cache-side queue.
        self._restore_flights: dict[str, _RestoreFlight] = {}
        self._restore_flight_leaders: dict[str, str] = {}
        self._restore_flight_followers: dict[str, _RestoreFollower] = {}
        # Async loads are parked after block allocation, so they do not
        # appear in the next SchedulerOutput. Carry their allocated blocks
        # across that boundary explicitly.
        self._pending_async_loads: dict[
            str, tuple[str, int, tuple[tuple[int, ...], ...]]
        ] = {}
        # Digests this scheduler has admitted a restore for, so a reported
        # load failure on ANY rank can retire the entry cluster-wide.
        # request_id -> (digest, restored block ids). The block set lets a
        # load-error report retire exactly the failing entry. Entries are
        # removed on rescheduling, on retirement, or by request_finished.
        self._admitted: dict[str, tuple[str, frozenset[int]]] = {}
        # Quorum map: digest -> ranks that can offer a compatible manifest.
        # A restore is only offered at full quorum; every participating worker
        # then verifies its chunk bytes before writing private request blocks.
        self._quorum: dict[str, set[int]] = {}
        # A worker process gets a fresh generation on every connector
        # construction. The scheduler withdraws that physical rank's old
        # confirmations before accepting a report from a different generation,
        # so an isolated worker restart cannot inherit stale quorum state.
        self._stats_generation = uuid.uuid4().hex
        # This process-local value detects inconsistent reuse of one UUID. It
        # never orders generations because a host reboot resets its clock.
        self._stats_generation_epoch = time.monotonic_ns()
        self._worker_generations: dict[int, str] = {}
        self._worker_generation_epochs: dict[int, int] = {}
        self._worker_retired_generations: dict[int, list[str]] = {}
        self._worker_report_sequences: dict[int, int] = {}
        self._worker_held: dict[int, set[str]] = {}
        self._worker_pending_deltas: dict[int, dict[int, dict[str, Any]]] = {}
        self._worker_checkpoints: dict[int, dict[str, Any]] = {}
        self._worker_desynchronized: set[int] = set()
        self._worker_requires_checkpoint: set[int] = set()
        self._worker_publication_pending: set[int] = set()
        self._worker_incompatible: dict[int, set[tuple[str, str]]] = {}
        # Bind reusable digest tables to immutable token values, not a mutable
        # list's identity. Availability is checked afresh by each callback.
        # Request completion releases the token snapshot and digest table.
        self._prefix_digest_candidates: dict[
            str,
            tuple[
                tuple[tuple[int, ...], int, tuple[MultimodalFeatureIdentity, ...], str],
                tuple[tuple[int, str], ...],
            ],
        ] = {}
        self._store_progress: dict[str, tuple[str, int, int, list[list[int]]]] = {}
        # Kept separate from the long-standing progress tuple so scheduler
        # checkpoint/test adapters that seed that tuple remain compatible.
        self._store_token_ids: dict[str, tuple[int, ...]] = {}
        self._store_multimodal: set[str] = set()
        self._store_bases: dict[str, tuple[str, int]] = {}
        self._store_recurrent_boundaries: dict[str, tuple[tuple[int, int], ...]] = {}
        self.counters: dict[str, int] = {
            "store_committed": 0,
            "store_failed": 0,
            "store_evicted": 0,
            "store_skipped_busy": 0,
            "store_skipped_present": 0,
            "store_skipped_quorum": 0,
            "store_skipped_delayed_limit": 0,
            "multimodal_bypass": 0,
            "page_delta_compactions": 0,
            "recurrent_boundary_metadata_rejected": 0,
            "prefix_alias_publication_attempted": 0,
            "prefix_alias_publication_failed": 0,
            "prefix_aliases_published": 0,
            "prefix_alias_segments_published": 0,
            "prefix_aliases_discovered": 0,
            "prefix_alias_scheduler_probe_hit": 0,
            "prefix_alias_restore_hit": 0,
            "prefix_alias_advertisement_failed": 0,
            "prefix_alias_capacity_evicted": 0,
            "prefix_alias_segments_deleted": 0,
            "restore_hit": 0,
            "restore_miss_absent": 0,
            "restore_miss_corrupt": 0,
            "restore_miss_incompatible": 0,
            "restore_flights_started": 0,
            "restore_flights_joined": 0,
            "restore_flights_completed": 0,
            "restore_flights_failed": 0,
            "restore_flight_follower_overflow": 0,
            "restore_flight_leader_aborted": 0,
            "restore_segment_flights_joined": 0,
            "restore_segment_roots_verified": 0,
            "restore_segment_roots_rejected": 0,
            "page_base_flights_completed": 0,
            "page_base_flights_recomputed": 0,
            "page_base_flights_cancelled": 0,
            "page_base_flight_participants": 0,
            "page_base_physical_reads": 0,
            "page_base_reads_avoided": 0,
            "native_page_delta_load_verified": 0,
            "native_page_delta_base_bytes_skipped": 0,
            "shared_prefix_leases_published": 0,
            "shared_prefix_leases_attached": 0,
            "shared_prefix_leases_expired": 0,
            "shared_prefix_leases_evicted": 0,
            "shared_prefix_lease_rejected": 0,
            "prefix_digest_cache_hits": 0,
            "prefix_digest_cache_misses": 0,
            "quorum_generation_resets": 0,
            "load_verified": 0,
            "load_failed": 0,
            "capacity_runs": 0,
            "capacity_skipped_busy": 0,
            "capacity_failed": 0,
            "capacity_manifests_evicted": 0,
            "capacity_chunks_deleted": 0,
            "capacity_bytes_reclaimed": 0,
            "capacity_retries": 0,
            "streaming_store_committed": 0,
            "streaming_store_evicted": 0,
            "streaming_capacity_queued": 0,
            "streaming_capacity_retries": 0,
            "streaming_capacity_invalid_receipts": 0,
            "streaming_capacity_shutdown_dropped": 0,
            "restore_skip_local_prefix": 0,
            "restore_skip_oversize": 0,
            "restore_skip_backlog": 0,
            "quorum_incomplete": 0,
            "attribution_requests_complete": 0,
            "attribution_requests_incomplete": 0,
            "attribution_invalid_events": 0,
        }
        logger.info(
            "sparkcache: startup profile=%s groups=%d policies=%s identity=%s integrity=%s"
            " role=%s root=%s dcp=%d access_mode=%s store=%s restore=%s"
            " cuda_restore=%s max_span=%d",
            self._profile.name, len(self._group_topology),
            ",".join(sorted({group["reuse_policy"] for group in self._group_topology})),
            self._identity(self._shard_rank).storage_key[:12], config.payload_integrity,
            role.name, self._root, self._dcp_degree, self._access_mode,
            self._store_enabled, self._restore_enabled, self._native_restore_enabled,
            self._max_span,
        )
        logger.debug(
            "sparkcache: config role=%s root=%s dcp=%d access_mode=%s"
            " store=%s restore=%s"
            " cuda_restore=%s max_span=%d load_threads=%d max_bytes=%d"
            " low_bytes=%d ttl_seconds=%d shared_prefix_lease_ttl_seconds=%.3g",
            role.name,
            self._root,
            self._dcp_degree,
            self._access_mode,
            self._store_enabled,
            self._restore_enabled,
            self._native_restore_enabled,
            self._max_span,
            self._load_thread_limit,
            self._capacity_policy.max_bytes,
            self._capacity_policy.low_watermark_bytes,
            self._capacity_policy.ttl_seconds,
            self._shared_prefix_lease_ttl_seconds,
        )
        logger.debug(
            "sparkcache: topology profile=%s groups=%d policies=%s",
            self._profile.name,
            len(self._group_topology),
            ",".join(sorted({group["reuse_policy"] for group in self._group_topology})),
        )
        logger.debug(
            "sparkcache: group_topology %s",
            json.dumps(self._group_topology, default=dict, sort_keys=True, separators=(",", ":")),
        )

    def _install_streaming_runtime(self, role: KVConnectorRole) -> None:
        """Resolve the opt-in runtime without importing a C++/CUDA backend."""

        factory = _STREAMING_RUNTIME_FACTORIES.get(role)
        if factory is None:
            raise RuntimeError(
                "sparkcache: streaming snapshots requested but no "
                f"runtime is installed for {role.name.lower()} role"
            )
        try:
            runtime = factory(self)
        except Exception as error:
            raise RuntimeError(
                "sparkcache: streaming runtime installation was rejected"
            ) from error
        if runtime is None:
            raise RuntimeError(
                "sparkcache: streaming runtime installation was rejected"
            )
        if role is KVConnectorRole.SCHEDULER:
            required = (
                "observe_metadata",
                "request_finished",
                "take_finished",
                "shutdown",
            )
        else:
            required = (
                "bind_kv_caches",
                "offer_completed",
                "poll",
                "handle_preemptions",
                "request_finished",
                "take_finished",
                "shutdown",
            )
        missing = [
            name for name in required if not callable(getattr(runtime, name, None))
        ]
        if missing:
            raise RuntimeError(
                "sparkcache: streaming "
                f"{role.name.lower()} runtime is incomplete: " + ", ".join(missing)
            )
        self._streaming_runtime = runtime

    # ------------------------------------------------------------------
    # identity helpers
    # ------------------------------------------------------------------

    def _remember_request_scope(self, request: Any) -> str | None:
        """Latch scope from the original Request, never from incomplete output data."""
        scopes = self._request_cache_scopes
        request_id = request.request_id
        try:
            scope = fingerprint(request.cache_salt)
        except (AttributeError, ValueError, UnicodeError):
            scope = None
        if request_id in scopes and scopes[request_id] != scope:
            scope = None
        scopes[request_id] = scope
        if scope is None:
            self._remove_restore_follower(request_id)
            leader = self._restore_flight_leaders.get(request_id)
            flight = self._restore_flights.get(leader)
            if flight is not None and not flight.dispatched:
                self._retire_restore_flight(leader, outcome="cancelled")
            self._prefix_digest_candidates.pop(request_id, None)
            self._store_progress.pop(request_id, None)
            self._store_token_ids.pop(request_id, None)
            self._store_bases.pop(request_id, None)
            self._store_multimodal.discard(request_id)
            self._store_recurrent_boundaries.pop(request_id, None)
            # Keep pending/admitted allocation evidence until the ordinary
            # failed-load handoff returns its blocks for recomputation.
        return scope

    def _scope_salt(self, scope: str = UNSALTED_SCOPE) -> str:
        if not valid_scope(scope):
            raise ValueError("Missing or invalid request cache scope")
        return self._context_digest_salt + ":" + scope

    def _identity(
        self, shard_rank: int, tp_shard_rank: int | None = None
    ) -> CacheIdentity:
        """Build a CacheIdentity for the given DCP shard rank.

        tp_shard_rank defaults to the scheduler's shard rank (0) when
        None and the connector is on the scheduler side, or to the
        worker's physical TP rank when on the worker side.  This
        ensures persistent storage keys distinguish physical workers
        that share a DCP-local rank (e.g. TP0 and TP2 under DCP2).
        """
        if tp_shard_rank is None:
            tp_shard_rank = (
                self._shard_rank
                if self._role is KVConnectorRole.SCHEDULER
                else self._physical_rank()
            )
        return self._config.build_identity(shard_rank, tp_shard_rank)

    def _digest(
        self,
        token_ids: list[int],
        span: int,
        multimodal_features: Sequence[MultimodalFeatureIdentity] = (),
        *, request_scope: str = UNSALTED_SCOPE,
    ) -> str:
        # The salt must be identical on every role and rank: it names the
        # shared context, not this process's shard. Pin both shard fields to
        # zero at construction rather than letting the worker role substitute
        # its physical rank, which would fork the namespace per worker.
        return context_prefix_digest(
            token_ids,
            self._scope_salt(request_scope),
            token_count=span,
            multimodal_features=multimodal_features,
        )

    def _request_prefix_candidates(
        self,
        request_id: str,
        token_ids: Sequence[int],
        span_ceiling: int,
        multimodal_features: Sequence[MultimodalFeatureIdentity] = (),
        *, request_scope: str = UNSALTED_SCOPE,
    ) -> tuple[tuple[int, str], ...]:
        """Reuse exact digest analysis across scheduler and publication callbacks."""
        signature = (
            tuple(token_ids),
            span_ceiling,
            tuple(multimodal_features),
            request_scope,
        )
        cached = self._prefix_digest_candidates.get(request_id)
        if cached is not None and cached[0] == signature:
            self.counters["prefix_digest_cache_hits"] += 1
            return cached[1]
        first = (
            (self._min_span + self._chunk_tokens - 1) // self._chunk_tokens
        ) * self._chunk_tokens
        candidates = chunk_prefix_digests(
            signature[0],
            self._scope_salt(request_scope),
            boundaries=range(first, span_ceiling + 1, self._chunk_tokens),
            chunk_tokens=self._chunk_tokens,
            multimodal_features=multimodal_features,
        )
        self._prefix_digest_candidates[request_id] = (signature, candidates)
        self.counters["prefix_digest_cache_misses"] += 1
        return candidates

    def _publication_base(
        self,
        token_ids: Sequence[int],
        span_tokens: int,
        multimodal_features: Sequence[MultimodalFeatureIdentity] = (),
        *,
        request_id: str | None = None,
        request_scope: str = UNSALTED_SCOPE,
    ) -> tuple[str, int]:
        """Select the longest all-rank prefix eligible for tail publication."""

        if not self._publication_schema:
            return "", 0
        if any(feature.offset < span_tokens for feature in multimodal_features):
            # ManifestStore's extension proof derives token-only digests. An
            # exact multimodal snapshot is safe; a derived key is not.
            return "", 0
        first = (
            (self._min_span + self._chunk_tokens - 1) // self._chunk_tokens
        ) * self._chunk_tokens
        if span_tokens - self._chunk_tokens < first:
            return "", 0
        candidates = (
            self._request_prefix_candidates(
                request_id, token_ids, span_tokens, multimodal_features,
                request_scope=request_scope,
            )
            if request_id is not None
            else chunk_prefix_digests(
                token_ids,
                self._scope_salt(request_scope),
                boundaries=range(first, span_tokens, self._chunk_tokens),
                chunk_tokens=self._chunk_tokens,
                multimodal_features=multimodal_features,
            )
        )
        selected = next(
            (
                (digest, boundary)
                for boundary, digest in reversed(candidates)
                if boundary < span_tokens and self._has_full_quorum(digest)
            ),
            None,
        )
        return selected or ("", 0)

    def _lookup_reusable(
        self,
        identity: CacheIdentity,
        digest: str,
        *,
        verify_chunks: bool = True,
        verify_chunk_metadata: bool = False,
    ) -> tuple[LookupResult, bool]:
        """Look up one exact manifest or safe row-prefix alias.

        Exact metadata always wins, including an exact entry that is corrupt
        or incompatible. Prefix aliases are never consulted for opaque
        block-page state or any storage mode other than ``per_token_rows``.
        """

        lookup = self._store.lookup(
            identity,
            digest,
            verify_chunks=verify_chunks,
            verify_chunk_metadata=verify_chunk_metadata,
            storage_mode=(
                "per_token_rows" if self._storage_mode == "per_token_rows" else None
            ),
        )
        return lookup, lookup.root_kind == "prefix_alias"

    def _invalidate_reusable(
        self,
        identity: CacheIdentity,
        digest: str,
        *,
        is_alias: bool,
        verify_chunk_payloads: bool = False,
    ) -> bool:
        """Remove exactly the root selected by the preceding lookup.

        ManifestStore invalidation is exact-first. Once lookup selected an
        alias, deleting its path directly avoids a concurrent exact commit
        with the same digest being mistaken for the failed alias.
        """

        if not is_alias:
            return self._store.invalidate(
                identity,
                digest,
                verify_chunk_payloads=verify_chunk_payloads,
            )
        alias_path = (
            Path(self._root)
            / "prefix-aliases"
            / identity.storage_key
            / f"{digest}.json"
        )
        try:
            alias_path.unlink()
            return True
        except OSError:
            return False

    def _aligned_span(self, prompt_len: int) -> int:
        # A persistent boundary must be a multiple of the manager-group block LCM,
        # of every prefix-cacheable group's tokens_per_state (vLLM aligns
        # prefix hits to the same quanta), and — under DCP sharding — of every
        # sharded group's logical page width. It must also leave at least the
        # largest per-group prefill replay reserve to recompute, so restored
        # runs rebuild the request-private state a cache-off run rebuilds.
        # With no replay group the reserve is zero and the boundary is the
        # largest aligned span below the prompt end, as before.
        block_alignment = self._block_size
        if self._storage_mode == "block_pages_v1" and any(
            int(group.get("recurrent_state", {}).get("tokens_per_state", -1)) > 0
            for group in self._group_topology
        ):
            # Fixed-quantum recurrent groups publish explicit checkpoint
            # blocks at chunk boundaries, including partial attention pages.
            # Missing checkpoint metadata still refuses publication below.
            # Variable-quantum recurrent groups retain manager-page alignment.
            block_alignment = self._chunk_tokens
        alignment = math.lcm(
            block_alignment, self._prefix_state_alignment, self._chunk_tokens
        )
        if self._storage_mode == "block_pages_v1" and self._dcp_degree > 1:
            try:
                sharded_page_quanta = tuple(
                    int(topology["logical_tokens_per_block"])
                    for topology in self._group_topology
                    if int(topology["dcp_shard_count"]) > 1
                )
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(
                    "sparkcache: block-page DCP topology is incomplete"
                ) from error
            alignment = math.lcm(
                alignment,
                self._chunk_tokens,
                *(sharded_page_quanta or (self._block_size,)),
            )
        limit = prompt_len - max(1, self._prefill_replay_tokens)
        if limit <= 0:
            return 0
        return limit // alignment * alignment

    @staticmethod
    def _normalize_group_blocks(
        block_ids: Sequence[Sequence[int]],
        *,
        allow_empty_groups: bool = False,
    ) -> tuple[tuple[int, ...], ...]:
        groups = tuple(tuple(int(block) for block in group) for group in block_ids)
        if not groups or (
            not allow_empty_groups and any(not group for group in groups)
        ):
            raise RuntimeError("sparkcache: KV-cache block table is empty")
        return groups

    def _select_group_blocks_for_span(
        self,
        groups: tuple[tuple[int, ...], ...],
        span_tokens: int,
        *,
        recurrent_boundary_blocks: Sequence[tuple[int, int]] = (),
    ) -> tuple[tuple[int, ...], ...]:
        if len(groups) != len(self._group_topology):
            raise HybridCodecError("request block tables disagree with page groups")
        overrides: dict[int, int] = {}
        for entry in recurrent_boundary_blocks:
            if (
                not isinstance(entry, (list, tuple))
                or len(entry) != 2
                or any(type(value) is not int for value in entry)
            ):
                raise HybridCodecError("recurrent boundary override is malformed")
            group_index, block_id = entry
            if (
                not 0 <= group_index < len(self._group_topology)
                or group_index in overrides
                or block_id <= 0
                or self._group_topology[group_index]["reuse_policy"]
                != "recurrent_align"
            ):
                raise HybridCodecError("recurrent boundary override is incompatible")
            overrides[group_index] = block_id
        trimmed = []
        for group_index, (group, topology) in enumerate(
            zip(groups, self._group_topology)
        ):
            logical_tokens_per_block = int(topology["logical_tokens_per_block"])
            boundary_required = (
                span_tokens + logical_tokens_per_block - 1
            ) // logical_tokens_per_block
            policy = topology["reuse_policy"]
            window = topology["reuse_window_tokens"]
            selected = boundary_required
            minimum_table_blocks = boundary_required
            if policy in ("transient", "replay"):
                # vLLM rebuilds in-flight ring and request-private state at
                # every persistent boundary. These groups contribute no pages
                # to a store or restore; the aligned span already leaves the
                # replay reserve to recompute. Their request tables name only
                # the ring's own in-flight block or the live window tail, so
                # they need not cover the span.
                selected = 0
                minimum_table_blocks = 1
            elif policy == "sliding":
                if window is None:
                    raise HybridCodecError("sliding page group has no reuse window")
                # The window includes the token about to be computed. Only its
                # preceding window-1 KV positions must be resident.
                selected = min(
                    boundary_required,
                    (int(window) - 1 + logical_tokens_per_block - 1)
                    // logical_tokens_per_block,
                )
            elif policy == "recurrent_align":
                # Align mode retains a single recurrent checkpoint at the
                # persistent boundary. Earlier position-indexed table entries
                # are null after vLLM removes skipped state blocks.
                selected = 1
                minimum_table_blocks = 1
            elif policy != "full":
                raise HybridCodecError(f"unsupported page reuse policy {policy!r}")
            if len(group) < minimum_table_blocks:
                raise HybridCodecError(
                    "request block table is shorter than the aligned cache span"
                )
            # Full and sliding tables are position-indexed. Recurrent-align
            # tables instead retain one live checkpoint in their final slot;
            # earlier entries may be null padding or historical state. An
            # explicit boundary hand-off names a pinned publication checkpoint
            # and therefore takes precedence over the live tail.
            if policy == "recurrent_align" and group_index in overrides:
                chosen = (overrides[group_index],)
            elif policy == "recurrent_align" and len(group) < boundary_required:
                chosen = (group[-1],)
            else:
                chosen = group[
                    boundary_required - selected : boundary_required
                ]
            if any(block <= 0 for block in chosen):
                raise HybridCodecError(
                    "selected page window contains vLLM's null block"
                )
            if len(set(chosen)) != len(chosen):
                raise HybridCodecError(
                    "selected page window contains duplicate physical blocks"
                )
            trimmed.append(chosen)
        return tuple(trimmed)

    def _group_block_counts_for_span(self, span_tokens: int) -> tuple[int, ...]:
        """Return page counts without consulting disposable physical IDs."""

        counts = []
        for topology in self._group_topology:
            logical_tokens_per_block = int(topology["logical_tokens_per_block"])
            required = (
                span_tokens + logical_tokens_per_block - 1
            ) // logical_tokens_per_block
            policy = topology["reuse_policy"]
            if policy in ("transient", "replay"):
                # Non-persisted groups contribute zero pages; see
                # _select_group_blocks_for_span.
                required = 0
            elif policy == "sliding":
                window = topology["reuse_window_tokens"]
                if window is None:
                    raise HybridCodecError("sliding page group has no reuse window")
                required = min(
                    required,
                    (int(window) - 1 + logical_tokens_per_block - 1)
                    // logical_tokens_per_block,
                )
            elif policy == "recurrent_align":
                required = 1
            elif policy != "full":
                raise HybridCodecError(f"unsupported page reuse policy {policy!r}")
            counts.append(required)
        return tuple(counts)

    def _has_full_quorum(self, digest: str) -> bool:
        """Return whether every physical TP worker already offers this cache entry.

        Quorum requires all physical TP workers in range(tp_degree),
        not just DCP-local ranks.  Under TP4/DCP2, this means all four
        physical workers (TP0..TP3) must confirm, preventing a single
        DCP group from falsely satisfying quorum while the other
        physical workers lack data.
        """

        confirmed = self._quorum.get(digest, ())
        return all(rank in confirmed for rank in range(self._tp_degree))

    def _join_segment_restore_flight(
        self,
        request_id: str,
        candidates: Sequence[tuple[int, str]],
        *,
        selected_digest: str,
    ) -> bool:
        """Join one different-root flight through a common row-prefix root.

        Only one shorter trunk is retained per flight. It replaces the full
        root as that flight's hot lease, so divergent tails cannot expand GPU
        state or scheduler metadata without a fixed bound.
        """
        if self._storage_mode != "per_token_rows":
            return False
        by_digest = {digest: span for span, digest in candidates}
        matches: list[tuple[int, _RestoreFlight, str]] = []
        for flight in self._restore_flights.values():
            if (
                flight.digest == selected_digest
                or flight.leader_finished
                or flight.dispatched
            ):
                continue
            for span, digest in reversed(flight.shareable_prefixes):
                if by_digest.get(digest) != span:
                    continue
                if flight.segment_digest not in (None, digest):
                    continue
                matches.append((span, flight, digest))
                break
        if not matches:
            return False
        span, flight, digest = max(matches, key=lambda item: item[0])
        if len(flight.followers) >= _MAX_RESTORE_FLIGHT_FOLLOWERS:
            self.counters["restore_flight_follower_overflow"] += 1
            return False
        if flight.segment_digest is None:
            flight.segment_digest = digest
            flight.segment_span_tokens = span
            # Exact-root followers that arrived first can safely use the
            # shorter common trunk. Keeping one lease per restore preserves
            # the existing vLLM callback and the global two-lease bound.
            for follower_request_id, binding in tuple(
                self._restore_flight_followers.items()
            ):
                if binding.flight_digest == flight.digest:
                    self._restore_flight_followers[follower_request_id] = (
                        _RestoreFollower(flight.digest, digest, span)
                    )
        flight.followers.add(request_id)
        self._restore_flight_followers[request_id] = _RestoreFollower(
            flight.digest, digest, span
        )
        self.counters["restore_flights_joined"] += 1
        self.counters["restore_segment_flights_joined"] += 1
        return True

    # ------------------------------------------------------------------
    # scheduler side
    # ------------------------------------------------------------------

    def _retire_restore_flight(self, digest: str, *, outcome: str) -> None:
        flight = self._restore_flights.pop(digest, None)
        if flight is None:
            return
        self._restore_flight_leaders.pop(flight.leader_request_id, None)
        for request_id in flight.followers:
            self._restore_flight_followers.pop(request_id, None)
        counter = {
            "completed": "restore_flights_completed",
            "failed": "restore_flights_failed",
        }.get(outcome)
        if counter is not None:
            self.counters[counter] += 1
        if outcome == "failed":
            self._log_restore_outcome(flight, "failed")

    def _log_restore_outcome(self, flight: _RestoreFlight, outcome: str) -> None:
        # The scheduler owns the request-level result: a restore succeeds only
        # when every rank placed its shard, so per-rank worker lines stay DEBUG.
        elapsed_ms = (
            (time.perf_counter_ns() - flight.started_ns) / 1_000_000
            if flight.started_ns else 0.0
        )
        if outcome == "hit":
            logger.info("sparkcache: restore hit tokens=%d ms=%.1f",
                        flight.span_tokens, elapsed_ms)
        else:
            logger.info("sparkcache: restore failed tokens=%d ms=%.1f; recomputing",
                        flight.span_tokens, elapsed_ms)

    def _remove_restore_follower(self, request_id: str) -> None:
        follower = self._restore_flight_followers.pop(request_id, None)
        if follower is None:
            return
        flight = self._restore_flights.get(follower.flight_digest)
        if flight is not None:
            flight.followers.discard(request_id)
            segment_still_needed = any(
                binding.flight_digest == flight.digest
                and binding.lease_digest == follower.lease_digest
                for binding in self._restore_flight_followers.values()
            )
            if (
                follower.lease_digest == flight.segment_digest
                and not flight.lease_published
                and not segment_still_needed
            ):
                flight.segment_digest = None
                flight.segment_span_tokens = 0

    def _expire_shared_prefix_flights(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        for digest, flight in tuple(self._restore_flights.items()):
            expires_at = flight.lease_expires_at
            if expires_at is not None and expires_at <= now:
                self.counters["shared_prefix_leases_expired"] += 1
                self._retire_restore_flight(digest, outcome="completed")

    def _lease_publication_candidates(
        self, flight: _RestoreFlight
    ) -> tuple[tuple[str, int, float], ...]:
        if flight.segment_digest is not None:
            return (
                (
                    flight.segment_digest,
                    flight.segment_span_tokens,
                    self._shared_prefix_lease_ttl_seconds,
                ),
            )
        return (
            (
                flight.digest,
                flight.span_tokens,
                self._shared_prefix_lease_ttl_seconds,
            ),
        )

    def get_shared_prefix_lease_to_publish(
        self, request: "Request"
    ) -> tuple[str, int, float] | None:
        """Name a restore only after every worker has verified and finished it."""
        request_id = request.request_id
        digest = self._restore_flight_leaders.get(request_id)
        if digest is None:
            return None
        flight = self._restore_flights.get(digest)
        if (
            flight is None
            or not flight.workers_finished
            or flight.leader_finished
            or flight.lease_published
        ):
            return None
        return self._lease_publication_candidates(flight)[0]

    def shared_prefix_lease_published(self, request_id: str, lease_key: str) -> bool:
        """Acknowledge that vLLM pinned the verified multi-group block table."""
        digest = self._restore_flight_leaders.get(request_id)
        flight = self._restore_flights.get(digest) if digest is not None else None
        if flight is None or not flight.workers_finished:
            return False
        candidates = {
            key: span for key, span, _ in self._lease_publication_candidates(flight)
        }
        span = candidates.get(lease_key)
        if span is None:
            return False
        now = time.monotonic()
        flight.lease_digest = lease_key
        flight.lease_span_tokens = span
        flight.lease_published_at = now
        flight.lease_expires_at = now + self._shared_prefix_lease_ttl_seconds
        self.counters["shared_prefix_leases_published"] += 1

        published = sorted(
            (
                item
                for item in self._restore_flights.values()
                if item.lease_published_at is not None
            ),
            key=lambda item: item.lease_published_at or 0.0,
        )
        for stale in published[:-_MAX_SHARED_PREFIX_LEASES]:
            self.counters["shared_prefix_leases_evicted"] += 1
            self._retire_restore_flight(stale.digest, outcome="completed")
        return True

    def get_shared_prefix_lease_candidate(
        self, request: "Request"
    ) -> tuple[str, int] | None:
        """Return the longest live verified lease matching this request prefix."""
        request_scope = self._remember_request_scope(request)
        if request_scope is None:
            return None
        if not self._restore_enabled:
            return None
        self._expire_shared_prefix_flights()
        request_id = request.request_id

        follower_digest = self._restore_flight_followers.get(request_id)
        if follower_digest is not None:
            flight = self._restore_flights.get(follower_digest.flight_digest)
            if (
                flight is not None
                and flight.lease_published
                and follower_digest.lease_digest == flight.lease_digest
            ):
                return follower_digest.lease_digest, follower_digest.span_tokens
            return None

        prompt_token_ids = request.prompt_token_ids or ()
        token_ids = prompt_token_ids
        multimodal_features = _multimodal_feature_identities(
            request,
            len(token_ids),
        )
        if multimodal_features is None:
            self.counters["multimodal_bypass"] += 1
            return None
        aligned_span = self._aligned_span(len(token_ids))
        span_ceiling = min(
            aligned_span,
            self._max_span // self._chunk_tokens * self._chunk_tokens,
        )
        if span_ceiling < self._min_span:
            return None
        candidates = self._request_prefix_candidates(
            request_id, token_ids, span_ceiling, multimodal_features,
            request_scope=request_scope,
        )
        selected = next(
            (
                (self._restore_flights.get(candidate_digest), candidate_digest)
                for _, candidate_digest in reversed(candidates)
                if (
                    self._restore_flights.get(candidate_digest) is not None
                    and self._restore_flights[candidate_digest].lease_published
                    and candidate_digest
                    == self._restore_flights[candidate_digest].lease_digest
                )
            ),
            None,
        )
        if selected is None:
            for candidate_span, candidate_digest in reversed(candidates):
                selected = next(
                    (
                        (item, candidate_digest)
                        for item in self._restore_flights.values()
                        if item.lease_published
                        and candidate_digest == item.lease_digest
                    ),
                    None,
                )
                if selected is not None:
                    break
        flight, lease_digest = selected if selected is not None else (None, "")
        if flight is None or request_id == flight.leader_request_id:
            return None
        if len(flight.followers) >= _MAX_RESTORE_FLIGHT_FOLLOWERS:
            self.counters["restore_flight_follower_overflow"] += 1
            return None
        flight.followers.add(request_id)
        lease_span = flight.lease_span_tokens
        self._restore_flight_followers[request_id] = _RestoreFollower(
            flight.digest, lease_digest, lease_span
        )
        self.counters["restore_flights_joined"] += 1
        if lease_digest != flight.digest:
            self.counters["restore_segment_flights_joined"] += 1
        return lease_digest, lease_span

    def _trace_reuse(
        self,
        event: str,
        request_id: str,
        *,
        timing: RestoreTiming | None = None,
        **fields: Any,
    ) -> None:
        """Emit opt-in decisions without treating offers or leases as disk reads.

        Caller prefix tokens are the scheduler's block-aligned input, not an
        exact local hash-hit measurement. Worker completion is rank-local;
        only vLLM's all-rank receive outcome permits the request to resume.
        """
        if not getattr(self, "_trace_reuse_enabled", False):
            return
        # Diagnostics cannot change cache ownership or verified-or-recompute.
        with contextlib.suppress(Exception):
            worker = self._role is KVConnectorRole.WORKER
            record = {
                "schema": "sparkcache-reuse-trace/v1",
                "event": event,
                "request_id": request_id,
                "time_ns": time.time_ns(),
                "role": "worker" if worker else "scheduler",
                "rank": self._physical_rank() if worker else None,
                "dcp_rank": self._worker_rank() if worker else None,
                **fields,
            }
            if timing is not None:
                record.update(
                    queue_wait_ms=round(timing.queue_wait_ns / 1_000_000, 3),
                    service_ms=round(timing.service_ns / 1_000_000, 3),
                    end_to_end_ms=round(timing.end_to_end_ns / 1_000_000, 3),
                    page_bytes=timing.page_bytes,
                    phase_ms={
                        phase: round(value / 1_000_000, 3)
                        for phase, value in timing.phase_ns.items()
                    },
                )
            logger.debug(
                "sparkcache: reuse %s",
                json.dumps(record, sort_keys=True, separators=(",", ":")),
            )

    def record_request_cache_event(self, request: "Request", event: str, **fields) -> None:
        """Consume optional scheduler observations without changing serving.

        Matching offers and worker-local completion traces never call this
        accounting path. The runtime must report adopted prefixes, final
        receive outcomes, and accepted target-model execution explicitly.
        """
        if not getattr(self, "request_cache_events_enabled", False):
            return
        request_id = request.request_id
        state = self._request_attribution.get(request_id)
        try:
            if event == "finished":
                state = self._request_attribution.pop(request_id, None)
                if state is None:
                    return
                status = fields.get("status", getattr(request, "status", "unknown"))
                status = str(getattr(status, "name", status))
                summary = state.summary(status)
                category = "complete" if summary["attribution_complete"] else "incomplete"
                key = f"attribution_requests_{category}"
                self.counters[key] = self.counters.get(key, 0) + 1
                if summary["attribution_complete"]:
                    for field in ("local_tokens_reused", "external_tokens_reused",
                                  "prompt_tokens_computed"):
                        key = "attribution_completed_" + field
                        self.counters[key] = self.counters.get(key, 0) + summary[field]
                self._trace_reuse("request_cache_attribution", request_id, **summary)
                return
            if state is None:
                if event != "admitted":
                    # Late receive drains for retired requests do not allocate
                    # another ledger or credit an aborted response.
                    return
                prompt_tokens = getattr(request, "num_prompt_tokens", None)
                if prompt_tokens is None:
                    prompt_tokens = len(request.prompt_token_ids or ())
                state = RequestAttribution(prompt_tokens)
                self._request_attribution[request_id] = state
            state.record(event, **fields)
        except Exception:  # Diagnostics cannot interrupt verified-or-recompute.
            if state is not None:
                state.valid = False
            key = "attribution_invalid_events"
            self.counters[key] = self.counters.get(key, 0) + 1

    def shared_prefix_lease_attached(self, request_id: str, lease_key: str) -> None:
        follower = self._restore_flight_followers.get(request_id)
        if follower is not None and follower.lease_digest == lease_key:
            self.counters["shared_prefix_leases_attached"] += 1
            self._trace_reuse(
                "gpu_lease_attached", request_id,
                digest=lease_key[:12], lease_span_tokens=follower.span_tokens,
            )

    def shared_prefix_lease_rejected(self, request_id: str, lease_key: str) -> None:
        self.counters["shared_prefix_lease_rejected"] += 1
        follower = self._restore_flight_followers.get(request_id)
        if follower is not None and follower.lease_digest == lease_key:
            self._remove_restore_follower(request_id)
        flight_digest = self._restore_flight_leaders.get(request_id)
        flight = (
            self._restore_flights.get(flight_digest)
            if flight_digest is not None
            else None
        )
        if flight is not None:
            # Publication failed, so no request may continue waiting for this
            # scheduler-owned pin. The leader already owns ordinary request
            # references and can continue; followers resume normal lookup.
            self._retire_restore_flight(flight.digest, outcome="cancelled")

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        lookup_started_ns = time.perf_counter_ns()
        request_scope = self._remember_request_scope(request)
        if request_scope is None:
            return 0, False
        if not self._restore_enabled:
            return 0, False
        token_ids = request.prompt_token_ids or ()
        multimodal_features = _multimodal_feature_identities(
            request,
            len(token_ids),
        )
        if multimodal_features is None:
            # Missing content identity or placeholder geometry cannot prove
            # which media produced the KV state. Recompute without caching.
            self.counters["multimodal_bypass"] += 1
            return 0, False
        request_id = request.request_id
        follower_digest = self._restore_flight_followers.get(request_id)
        if follower_digest is not None:
            # Shared-prefix leases can only adopt an empty request block table.
            # A request with locally computed tokens already owns allocator
            # state and must continue from that state instead of waiting for a
            # lease it cannot attach.
            if num_computed_tokens > 0:
                self._remove_restore_follower(request_id)
                return 0, False
            flight = self._restore_flights.get(follower_digest.flight_digest)
            if flight is not None and num_computed_tokens < follower_digest.span_tokens:
                return None, False
            self._remove_restore_follower(request_id)
        leader_digest = self._restore_flight_leaders.get(request_id)
        if leader_digest is not None:
            flight = self._restore_flights.get(leader_digest)
            if flight is not None:
                if (
                    flight.workers_finished
                    and num_computed_tokens >= flight.span_tokens
                ):
                    # vLLM has promoted the leader and published its verified
                    # blocks into the local prefix cache. Only now may peers
                    # leave the flight and attach through ordinary lookup.
                    self._retire_restore_flight(leader_digest, outcome="completed")
                    return 0, False
                if flight.dispatched or request_id in self._pending_async_loads:
                    # The leader cannot legitimately re-enter lookup before
                    # worker completion, but waiting is safer than creating a
                    # second writer into its private blocks.
                    return None, False
                if num_computed_tokens == 0 and self._has_full_quorum(leader_digest):
                    self._trace_reuse(
                        "external_restore_offer", request_id,
                        digest=leader_digest[:12], selected_span_tokens=flight.span_tokens,
                        offered_external_tokens=flight.span_tokens - num_computed_tokens,
                        caller_block_aligned_prefix_tokens=num_computed_tokens,
                        repeated_offer=True,
                    )
                    return flight.span_tokens - num_computed_tokens, True
                self._need_load.pop(request_id, None)
                self._retire_restore_flight(leader_digest, outcome="cancelled")
        if num_computed_tokens > 0:
            # Whole-prefix placement has no suffix write mask. A local prefix
            # may share pages with another request, so preserve it and compute
            # the remaining tokens instead of offering a second writer.
            self.counters["restore_skip_local_prefix"] = (
                self.counters.get("restore_skip_local_prefix", 0) + 1
            )
            return 0, False
        aligned_span = self._aligned_span(len(token_ids))
        span_ceiling = min(
            aligned_span,
            self._max_span // self._chunk_tokens * self._chunk_tokens,
        )
        first_candidate = max(
            (
                (self._min_span + self._chunk_tokens - 1)
                // self._chunk_tokens
                * self._chunk_tokens
            ),
            (num_computed_tokens // self._chunk_tokens + 1) * self._chunk_tokens,
        )
        if span_ceiling < first_candidate:
            return 0, False
        if aligned_span > self._max_span:
            self.counters["restore_skip_oversize"] = (
                self.counters.get("restore_skip_oversize", 0) + 1
            )
        candidates = tuple(
            candidate
            for candidate in self._request_prefix_candidates(
                request_id, token_ids, span_ceiling, multimodal_features,
                request_scope=request_scope,
            )
            if candidate[0] >= first_candidate
        )
        selected = next(
            (
                (candidate_span, candidate_digest)
                for candidate_span, candidate_digest in reversed(candidates)
                if self._has_full_quorum(candidate_digest)
            ),
            None,
        )
        # Manifest-only probe: chunk payloads are re-read and re-hashed by
        # every worker at load time, and any worker failure degrades to a
        # rank-synchronous recompute, so hashing ~200 MB here would only
        # add scheduler latency without adding safety.
        if selected is None:
            # Not every rank can offer a compatible manifest (or none has
            # reported yet). Treat as a plain miss: the request re-prefills
            # and republishes, which is also how a corrupted entry retires.
            reason = self._schema_miss_reason(
                token_ids, candidates, multimodal_features, request_scope,
            )
            counter = "quorum_incomplete" if reason == "quorum" else f"restore_miss_{reason}"
            self.counters[counter] = self.counters.get(counter, 0) + 1
            logger.info("sparkcache: restore miss tokens=%d ms=%.1f rank=%d reason=%s",
                        len(token_ids), (time.perf_counter_ns() - lookup_started_ns) / 1_000_000,
                        self._shard_rank, reason)
            return 0, False
        span, digest = selected
        flight = self._restore_flights.get(digest)
        if flight is not None:
            if num_computed_tokens > 0:
                return 0, False
            if len(flight.followers) >= _MAX_RESTORE_FLIGHT_FOLLOWERS:
                self.counters["restore_flight_follower_overflow"] += 1
                return 0, False
            flight.followers.add(request_id)
            lease_digest = flight.segment_digest or digest
            lease_span = flight.segment_span_tokens or span
            self._restore_flight_followers[request_id] = _RestoreFollower(
                digest, lease_digest, lease_span
            )
            self.counters["restore_flights_joined"] += 1
            if lease_digest != digest:
                self.counters["restore_segment_flights_joined"] += 1
            return None, False
        if num_computed_tokens == 0 and self._join_segment_restore_flight(
            request_id,
            candidates,
            selected_digest=digest,
        ):
            return None, False
        active_restore_flights = sum(
            not existing.lease_published for existing in self._restore_flights.values()
        )
        if active_restore_flights >= self._max_pending_restores or (
            request_id not in self._need_load
            and len(self._need_load) >= self._max_pending_restores
        ):
            self.counters["restore_skip_backlog"] = (
                self.counters.get("restore_skip_backlog", 0) + 1
            )
            return 0, False
        if self._scheduler_probe == "tp0":
            lookup, is_alias = self._lookup_reusable(
                self._identity(self._shard_rank),
                digest,
                verify_chunks=False,
            )
            if not lookup.is_hit:
                self.counters[f"restore_miss_{lookup.reason}"] = (
                    self.counters.get(f"restore_miss_{lookup.reason}", 0) + 1
                )
                logger.info("sparkcache: restore miss tokens=%d ms=%.1f rank=%d reason=%s",
                            span, (time.perf_counter_ns() - lookup_started_ns) / 1_000_000,
                            self._shard_rank, lookup.reason)
                return 0, False
            if is_alias:
                self.counters["prefix_alias_scheduler_probe_hit"] += 1
        # Probe mode "none": quorum alone determines admission; every worker
        # validated its own manifest at discovery or commit time, and any
        # rank's load failure still degrades to a clean recompute.
        self.counters["restore_hit"] += 1
        logger.debug(
            "sparkcache: hit tokens=%d digest=%s",
            span,
            digest[:12],
        )
        self._restore_flights[digest] = _RestoreFlight(
            digest=digest,
            span_tokens=span,
            leader_request_id=request_id,
            started_ns=lookup_started_ns,
            shareable_prefixes=tuple(
                (candidate_span, candidate_digest)
                for candidate_span, candidate_digest in candidates
                if candidate_span < span and self._has_full_quorum(candidate_digest)
            )[-_MAX_SHAREABLE_PREFIXES_PER_FLIGHT:],
        )
        self._restore_flight_leaders[request_id] = digest
        self.counters["restore_flights_started"] += 1
        self._need_load[request_id] = (digest, span)
        self.record_request_cache_event(
            request, "admitted", local_tokens=num_computed_tokens,
            external_tokens=span - num_computed_tokens, preemptions=0,
        )
        self._trace_reuse(
            "external_restore_offer", request_id,
            digest=digest[:12], selected_span_tokens=span,
            offered_external_tokens=span - num_computed_tokens,
            caller_block_aligned_prefix_tokens=num_computed_tokens,
            repeated_offer=False,
        )
        return span - num_computed_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._remember_request_scope(request)
        request_id = request.request_id
        if num_external_tokens <= 0:
            entry = self._need_load.pop(request_id, None)
            if entry is not None:
                self._retire_restore_flight(entry[0], outcome="cancelled")
            return
        entry = self._need_load.pop(request_id, None)
        if entry is None:
            return
        digest, span = entry
        group_blocks = self._normalize_group_blocks(blocks.get_block_ids())
        self._pending_async_loads[request_id] = (
            digest,
            span,
            group_blocks,
        )

    @staticmethod
    def _append_streaming_snapshot_offer(
        meta: SparkCacheConnectorMetadata,
        *,
        request_id: str,
        digest: str,
        span: int,
        promised_completed_tokens: int,
        block_ids: Sequence[int],
        request_scope: str = UNSALTED_SCOPE,
    ) -> None:
        """Record a scheduler promise without claiming forward completed it."""

        completed_tokens = min(span, promised_completed_tokens)
        if completed_tokens <= 0:
            return
        meta.streaming_snapshot_offers.append(
            _StreamingSnapshotOffer(
                request_id=request_id,
                digest=digest,
                span_tokens=span,
                completed_tokens=completed_tokens,
                block_ids=tuple(block_ids),
                request_scope=request_scope,
            )
        )

    def _validated_recurrent_boundary_blocks(
        self,
        scheduler_output: "SchedulerOutput",
        request_id: str,
        boundary_tokens: int,
        *,
        latched: tuple[tuple[int, int], ...] = (),
    ) -> tuple[tuple[int, int], ...] | None:
        """Validate vLLM's exact recurrent replay-boundary block hand-off.

        vLLM may expose an earlier aligned checkpoint while the request is
        still advancing toward this store boundary, followed by a partial-tail
        CoW target on a later scheduler output. Valid older entries and absent
        per-request metadata therefore preserve ``latched`` and keep the store
        pending. None means supplied metadata is incomplete, malformed, ahead
        of the plan, or conflicts with an earlier same-boundary latch, so this
        publication attempt must be poisoned. SparkCache never derives a
        replacement from another request-table entry because it can name
        overwritten running or speculative state instead of vLLM's pinned CoW
        target.
        """

        def reject(reason: str) -> None:
            self.counters["recurrent_boundary_metadata_rejected"] += 1
            _warning(
                "sparkcache: recurrent boundary metadata rejected"
                " request=%s boundary=%d: %s",
                request_id,
                boundary_tokens,
                reason,
            )
            return None

        required_groups = self._recurrent_group_indexes
        if not required_groups:
            return ()
        raw = getattr(scheduler_output, "recurrent_boundary_blocks", None)
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        native_offers = raw is None and block_state is not None
        if native_offers:
            # Generic offers belong to this worker step. They do not grant the
            # request-lifetime lease required to reuse a latched offer later.
            latched = ()
            if self._streaming_snapshots_enabled or (
                self._async_page_capture_enabled
                and not self._uses_capture_job_leases()
            ):
                return reject("generic boundary offers require synchronous or job-leased capture")
            raw = getattr(block_state, "boundary_state_offloads", None)
            if raw is None:
                return reject("generic block state lacks boundary offers")
        if raw is None:
            return latched
        if not isinstance(raw, Mapping):
            return reject("top-level value is not a mapping")
        entries = raw.get(request_id)
        if entries is None:
            return latched
        if not isinstance(entries, (list, tuple)):
            return reject("request value is not a sequence")
        if not entries:
            return reject("request has no recurrent boundary entries")

        overrides: list[tuple[int, int]] = []
        seen_target_groups: set[int] = set()
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 3:
                return reject("entry is not a group, block, boundary triple")
            group_index, block_id, entry_boundary = entry
            if any(type(value) is not int for value in entry):
                return reject("entry values are not integers")
            if not 0 <= group_index < len(self._group_topology):
                return reject("group index is outside the registered topology")
            topology = self._group_topology[group_index]
            if topology["reuse_policy"] != "recurrent_align":
                return reject("group is not an aligned recurrent cache")
            if block_id <= 0:
                return reject("physical block is vLLM's null block")
            if entry_boundary < boundary_tokens:
                continue
            if entry_boundary > boundary_tokens:
                if native_offers:
                    # The generic producer offers all retained checkpoints;
                    # later positions are independent of this store boundary.
                    continue
                return reject(
                    "entry boundary is ahead of the store plan"
                    f" observed={entry_boundary} target={boundary_tokens}"
                    f" group={group_index} block={block_id}"
                )
            if group_index in seen_target_groups:
                return reject("multiple blocks claim the same recurrent group")
            seen_target_groups.add(group_index)
            overrides.append((group_index, block_id))
        if not overrides:
            return latched
        if seen_target_groups != required_groups:
            return reject("entries do not cover every aligned recurrent group")
        validated = tuple(sorted(overrides))
        if latched and validated != latched:
            return reject("entries conflict with the latched recurrent boundary")
        return validated

    def _current_group_blocks(
        self,
        scheduler_output: "SchedulerOutput",
        request_id: str,
    ) -> tuple[bool, tuple[tuple[int, ...], ...] | None]:
        """Read generic scheduler tables without reconstructing reused slots."""
        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if block_state is None:
            return False, None
        tables = getattr(block_state, "block_ids", None)
        groups = tables.get(request_id) if isinstance(tables, Mapping) else None
        if (
            not isinstance(groups, (list, tuple))
            or len(groups) != len(self._group_topology)
            or any(
                not isinstance(group, (list, tuple))
                or not group
                or any(type(block) is not int or block < 0 for block in group)
                for group in groups
            )
        ):
            self.counters["recurrent_boundary_metadata_rejected"] += 1
            return True, None
        return True, tuple(tuple(group) for group in groups)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = SparkCacheConnectorMetadata(
            preempted_request_ids=tuple(
                sorted(getattr(scheduler_output, "preempted_req_ids", None) or ())
            )
        )
        # vLLM releases request-lifetime recurrent boundary pins on preemption.
        # Keep token/table accumulation for a possible resume, but require a
        # fresh hash-proven hand-off before the resumed request may publish.
        for request_id in meta.preempted_request_ids:
            self._store_recurrent_boundaries.pop(request_id, None)
        for request_id, (
            digest,
            span,
            group_blocks,
        ) in self._pending_async_loads.items():
            request_scope = self._request_cache_scopes.get(request_id)
            # An allocated external hit must receive a completion even when
            # its scope was rejected. The worker rejects None before reading
            # KV and reports failed blocks through the existing async handoff.
            all_blocks = frozenset(block for group in group_blocks for block in group)
            self._admitted[request_id] = (digest, all_blocks)
            flight = self._restore_flights.get(digest)
            if flight is not None and flight.leader_request_id == request_id:
                flight.restored_block_ids = all_blocks
                flight.dispatched = True
            shared_segments = (
                ((flight.segment_digest, flight.segment_span_tokens),)
                if flight is not None and flight.segment_digest is not None
                else ()
            )
            meta.plans.append(
                _ReqPlan(
                    request_id,
                    digest,
                    span,
                    group_blocks[0],
                    is_store=False,
                    block_ids_by_group=group_blocks,
                    shared_segments=shared_segments,
                    request_scope=request_scope,
                )
            )
        self._pending_async_loads.clear()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = new_req.prompt_token_ids or ()
            req_id = new_req.req_id
            # Some adapters export the scope explicitly; Kraken instead supplies
            # it through update_state_after_alloc on the original Request.
            if hasattr(new_req, "cache_salt"):
                self._remember_request_scope(SimpleNamespace(
                    request_id=req_id, cache_salt=new_req.cache_salt,
                ))
            request_scope = self._request_cache_scopes.get(req_id)
            if request_scope is None:
                continue
            self._need_load.pop(req_id, None)
            multimodal_features = _multimodal_feature_identities(
                new_req,
                len(token_ids),
            )
            if multimodal_features is None:
                # Publication without a complete media content/range identity
                # would recreate the token-placeholder alias.
                self.counters["multimodal_bypass"] += 1
                continue
            scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            native_tables, current_groups = self._current_group_blocks(
                scheduler_output, req_id,
            )
            if native_tables and current_groups is None:
                continue
            group_blocks = (
                current_groups if native_tables
                else self._normalize_group_blocks(new_req.block_ids)
            )
            assert group_blocks is not None
            block_ids = group_blocks[0]
            span = self._aligned_span(len(token_ids))
            if self._store_enabled and self._min_span <= span <= self._max_span:
                has_multimodal_identity = any(
                    feature.offset < span for feature in multimodal_features
                )
                candidates = self._request_prefix_candidates(
                    req_id, token_ids, span, multimodal_features,
                    request_scope=request_scope,
                )
                digest = candidates[-1][1]
                admitted = self._admitted.get(req_id)
                if admitted is not None and admitted[0] == digest:
                    # The restored entry already exists on every rank. A
                    # failed load retires this admission before recompute,
                    # so only verified restores take this fast exit.
                    self._admitted.pop(req_id, None)
                    continue
                if self._has_full_quorum(digest):
                    self.counters["store_skipped_quorum"] += 1
                    continue
                exact_token_ids = tuple(token_ids[:span])
                base_digest, base_span = self._publication_base(
                    token_ids,
                    span,
                    multimodal_features,
                    request_id=req_id,
                    request_scope=request_scope,
                )
                already = new_req.num_computed_tokens + scheduled
                if self._streaming_snapshots_enabled:
                    self._append_streaming_snapshot_offer(
                        meta,
                        request_id=req_id,
                        digest=digest,
                        span=span,
                        promised_completed_tokens=already,
                        block_ids=block_ids,
                        request_scope=request_scope,
                    )
                    if already < span:
                        # CachedRequestData.new_block_ids only carries the
                        # current append. Keep the complete table required by
                        # each following offer, including after resume.
                        self._store_progress[req_id] = (
                            digest,
                            span,
                            already,
                            [list(group) for group in group_blocks],
                        )
                        self._store_token_ids[req_id] = exact_token_ids
                        if has_multimodal_identity:
                            self._store_multimodal.add(req_id)
                elif self._recurrent_group_indexes:
                    recurrent_boundary_blocks = (
                        self._validated_recurrent_boundary_blocks(
                            scheduler_output,
                            req_id,
                            span,
                        )
                    )
                    if recurrent_boundary_blocks is None:
                        continue
                    if native_tables and already >= span and recurrent_boundary_blocks:
                        # Fresh offers are consumed in this step's post-forward
                        # CPU snapshot, including requests that finish after one
                        # generated token and never enter scheduled_cached_reqs.
                        meta.plans.append(
                            _ReqPlan(
                                req_id, digest, span, group_blocks[0], True,
                                block_ids_by_group=group_blocks,
                                token_ids=exact_token_ids,
                                base_context_digest=base_digest,
                                base_span_tokens=base_span,
                                recurrent_boundary_blocks=recurrent_boundary_blocks,
                                has_multimodal_identity=has_multimodal_identity,
                                request_scope=request_scope,
                            )
                        )
                        continue
                    # Full-page proof and partial-tail CoW hand-offs can arrive
                    # after the prefill which began this store. Retain the
                    # complete request table and any early proof until a later
                    # cached step has both finished the span and proven every
                    # recurrent group.
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        already,
                        [list(group) for group in group_blocks],
                    )
                    self._store_token_ids[req_id] = exact_token_ids
                    if has_multimodal_identity:
                        self._store_multimodal.add(req_id)
                    if recurrent_boundary_blocks:
                        self._store_recurrent_boundaries[req_id] = (
                            recurrent_boundary_blocks
                        )
                    if base_digest:
                        self._store_bases[req_id] = (base_digest, base_span)
                elif already >= span:
                    meta.plans.append(
                        _ReqPlan(
                            req_id,
                            digest,
                            span,
                            block_ids,
                            True,
                            block_ids_by_group=group_blocks,
                            token_ids=exact_token_ids,
                            base_context_digest=base_digest,
                            base_span_tokens=base_span,
                            has_multimodal_identity=has_multimodal_identity,
                            request_scope=request_scope,
                        )
                    )
                else:
                    # Chunked prefill: CachedRequestData.new_block_ids only
                    # carries each step's appended blocks, so accumulate the
                    # full table here until the span completes.
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        already,
                        [list(group) for group in group_blocks],
                    )
                    self._store_token_ids[req_id] = exact_token_ids
                    if has_multimodal_identity:
                        self._store_multimodal.add(req_id)
                    if base_digest:
                        self._store_bases[req_id] = (base_digest, base_span)
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            request_scope = self._request_cache_scopes.get(req_id)
            if request_scope is None:
                continue
            if req_id not in self._store_progress:
                continue
            digest, span, done, blocks_by_group = self._store_progress[req_id]
            exact_token_ids = self._store_token_ids.get(req_id, ())
            has_multimodal_identity = req_id in self._store_multimodal
            base_digest, base_span = self._store_bases.get(req_id, ("", 0))
            if self._has_full_quorum(digest):
                del self._store_progress[req_id]
                self._store_token_ids.pop(req_id, None)
                self._store_multimodal.discard(req_id)
                self._store_bases.pop(req_id, None)
                self._store_recurrent_boundaries.pop(req_id, None)
                self.counters["store_skipped_quorum"] += 1
                continue
            recurrent_boundary_blocks = self._validated_recurrent_boundary_blocks(
                scheduler_output,
                req_id,
                span,
                latched=self._store_recurrent_boundaries.get(req_id, ()),
            )
            if recurrent_boundary_blocks is None:
                del self._store_progress[req_id]
                self._store_token_ids.pop(req_id, None)
                self._store_multimodal.discard(req_id)
                self._store_bases.pop(req_id, None)
                self._store_recurrent_boundaries.pop(req_id, None)
                continue
            if recurrent_boundary_blocks:
                self._store_recurrent_boundaries[req_id] = recurrent_boundary_blocks
            native_tables, current_groups = self._current_group_blocks(
                scheduler_output, req_id,
            )
            if native_tables and current_groups is None:
                self._store_recurrent_boundaries.pop(req_id, None)
                continue
            if native_tables:
                assert current_groups is not None
                blocks_by_group = [list(group) for group in current_groups]
            elif done < span or req_id in cached.resumed_req_ids:
                new_block_ids = cached.new_block_ids[index]
                appended = (
                    [
                        list(group)
                        for group in self._normalize_group_blocks(
                            new_block_ids,
                            allow_empty_groups=True,
                        )
                    ]
                    if new_block_ids is not None
                    else [[] for _ in blocks_by_group]
                )
                if len(appended) != len(blocks_by_group):
                    raise RuntimeError(
                        "sparkcache: KV-cache group count changed while"
                        " accumulating a store"
                    )
                if req_id in cached.resumed_req_ids:
                    blocks_by_group = appended
                else:
                    blocks_by_group = [
                        existing + added
                        for existing, added in zip(blocks_by_group, appended)
                    ]
            blocks = blocks_by_group[0]
            done = cached.num_computed_tokens[index] + (
                scheduler_output.num_scheduled_tokens.get(req_id, 0)
            )
            if self._streaming_snapshots_enabled:
                if done >= span:
                    del self._store_progress[req_id]
                    self._store_token_ids.pop(req_id, None)
                    self._store_multimodal.discard(req_id)
                    self._store_bases.pop(req_id, None)
                    self._store_recurrent_boundaries.pop(req_id, None)
                    if self._has_full_quorum(digest):
                        self.counters["store_skipped_quorum"] += 1
                        continue
                else:
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        done,
                        blocks_by_group,
                    )
                self._append_streaming_snapshot_offer(
                    meta,
                    request_id=req_id,
                    digest=digest,
                    span=span,
                    promised_completed_tokens=done,
                    block_ids=blocks,
                    request_scope=request_scope,
                )
            elif done >= span:
                if self._recurrent_group_indexes and not recurrent_boundary_blocks:
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        done,
                        blocks_by_group,
                    )
                    continue
                if exact_token_ids and not has_multimodal_identity:
                    # A durable worker commit can finish while the scheduler is
                    # idle. Its first quorum report then returns only after the
                    # next request has begun prefill, which is too late for the
                    # base choice made in scheduled_new_reqs above. Select from
                    # the all-rank view observed at the publication step.
                    base_digest, base_span = self._publication_base(
                        exact_token_ids,
                        span,
                        request_id=req_id,
                        request_scope=request_scope,
                    )
                del self._store_progress[req_id]
                self._store_token_ids.pop(req_id, None)
                self._store_multimodal.discard(req_id)
                self._store_bases.pop(req_id, None)
                self._store_recurrent_boundaries.pop(req_id, None)
                normalized = tuple(tuple(group) for group in blocks_by_group)
                meta.plans.append(
                    _ReqPlan(
                        req_id,
                        digest,
                        span,
                        normalized[0],
                        True,
                        block_ids_by_group=normalized,
                        token_ids=exact_token_ids,
                        base_context_digest=base_digest,
                        base_span_tokens=base_span,
                        recurrent_boundary_blocks=recurrent_boundary_blocks,
                        has_multimodal_identity=has_multimodal_identity,
                        request_scope=request_scope,
                    )
                )
            else:
                self._store_progress[req_id] = (
                    digest,
                    span,
                    done,
                    blocks_by_group,
                )
        runtime = self._streaming_runtime
        if runtime is not None and self._role is KVConnectorRole.SCHEDULER:
            # request_finished() runs on the scheduler side. Give its adapter
            # the exact offers sent to workers so it delays block reuse only
            # for requests whose final gather may still be in flight.
            runtime.observe_metadata(meta)
        if (
            getattr(scheduler_output, "kv_connector_block_state", None) is not None
            and not any(scheduler_output.num_scheduled_tokens.values())
        ):
            # V2 no-forward callbacks do not invoke wait_for_save.
            meta.plans[:] = [plan for plan in meta.plans if not plan.is_store]
        self._reserve_async_page_capture_plans(meta)
        return meta

    def _reserve_async_page_capture_plans(
        self,
        meta: SparkCacheConnectorMetadata,
    ) -> None:
        """Bound request-block retention before worker capture can begin."""

        if (
            not self._async_page_capture_enabled
            or self._role is not KVConnectorRole.SCHEDULER
        ):
            return
        if self._uses_capture_job_leases():
            leases = self._capture_read_leases
            if leases is None:
                raise RuntimeError("GPU block pool must be bound before capture dispatch")
            accepted = []
            for plan in meta.plans:
                if not plan.is_store:
                    accepted.append(plan)
                    continue
                # Exact recurrent offers and complete attention pages are
                # immutable sources. Reference counts alone cannot protect a
                # mutable partial-page or running recurrent slot from writes.
                if any(
                    group["reuse_policy"] != "recurrent_align"
                    and plan.span_tokens % int(group["logical_tokens_per_block"])
                    for group in self._group_topology
                ):
                    self.counters["store_skipped_delayed_limit"] += 1
                    continue
                selected = self._select_group_blocks_for_span(
                    plan.group_block_ids, plan.span_tokens,
                    recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
                )
                try:
                    job = leases.reserve(bid for group in selected for bid in group)
                except ValueError:
                    self.counters["recurrent_boundary_metadata_rejected"] += 1
                    continue
                if job is None:
                    self.counters["store_skipped_delayed_limit"] += 1
                    continue
                accepted.append(replace(plan, capture_job_id=job))
            meta.plans[:] = accepted
            return
        accepted: list[_ReqPlan] = []
        for plan in meta.plans:
            if not plan.is_store:
                accepted.append(plan)
                continue
            if plan.request_id not in self._async_page_capture_reservations:
                if (
                    len(self._async_page_capture_reservations)
                    >= self._max_delayed_stores
                ):
                    self.counters["store_skipped_delayed_limit"] += 1
                    continue
                self._async_page_capture_reservations[plan.request_id] = sum(
                    len(group) for group in plan.group_block_ids
                )
            self._async_page_capture_eligible.add(plan.request_id)
            accepted.append(plan)
        meta.plans[:] = accepted

    def _uses_capture_job_leases(self) -> bool:
        return (
            self._async_page_capture_enabled
            and getattr(
                getattr(self, "_async_page_capture_settings", None), "lease_mode", None,
            )
            == "connector-jobs"
        )

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        """Use the generic connector API to retain exact capture sources."""
        if self._capture_read_leases is not None:
            if self._capture_read_leases.pool is not gpu_block_pool:
                raise RuntimeError("capture source pool cannot change after binding")
            return
        self._capture_read_leases = CaptureReadLeases(
            gpu_block_pool, ranks=self._tp_degree, max_jobs=self._max_delayed_stores,
        )

    def has_pending_push_work(self) -> bool:
        # Capture-read completion releases source pages before the store
        # finishes. Keep no-forward steps running until each worker's final
        # publication inventory has reached the scheduler as well.
        return bool(self._capture_read_leases or self._worker_publication_pending)

    def _capture_read_completed(self, job: str) -> None:
        if not job:
            return
        with self._capture_read_lock:
            self._capture_read_done.add(job)

    def _capture_read_failed(self, job: str) -> None:
        with self._capture_read_lock:
            self._capture_read_uncertain.add(job)

    def build_connector_worker_meta(self):
        with self._capture_read_lock:
            completed = self._capture_read_done
            uncertain = self._capture_read_uncertain
            self._capture_read_done = set()
            self._capture_read_uncertain = set()
        if not completed and not uncertain:
            return None
        rank = self._physical_rank()
        return SparkCacheReadCompletionMetadata(
            completed_reads={job: {rank} for job in completed},
            uncertain_reads={job: {rank} for job in uncertain},
        )

    def _release_async_page_capture_reservations(
        self,
        request_ids: Sequence[str],
    ) -> None:
        for request_id in request_ids:
            self._async_page_capture_reservations.pop(request_id, None)
            self._async_page_capture_eligible.discard(request_id)

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata) -> None:
        runtime = self._streaming_runtime
        if runtime is not None:
            # vLLM calls this before it may overwrite preempted blocks.
            # This must remain synchronous: the runtime drains armed CUDA
            # fences or raises without releasing blocks of unknown status.
            runtime.handle_preemptions(kv_connector_metadata)
        runtime = self._async_page_capture_runtime
        if runtime is not None:
            for request_id in getattr(
                kv_connector_metadata, "preempted_request_ids", ()
            ):
                runtime.preempt(request_id)

    def _manager_page_view(
        self,
        name: str,
        tensor: torch.Tensor,
        page_bytes: int,
    ) -> torch.Tensor:
        """Expose one complete physical manager page on dimension zero."""

        manager_blocks = int(getattr(self._kv_cache_config, "num_blocks", 0) or 0)
        if manager_blocks <= 0:
            raise RuntimeError(
                "sparkcache: block-page storage has no manager blocks"
            )
        if tensor.dim() < 2:
            raise RuntimeError(
                f"sparkcache: layer {name} has unexpected"
                f" shape {tuple(tensor.shape)}"
            )
        kernel_rows = int(tensor.shape[0])
        if kernel_rows % manager_blocks:
            raise RuntimeError(
                "sparkcache: layer "
                f"{name} has {kernel_rows} kernel rows for {manager_blocks}"
                " manager blocks"
            )
        kernel_rows_per_page = kernel_rows // manager_blocks
        element_size = int(tensor.element_size())
        logical_page_bytes = (
            kernel_rows_per_page
            * math.prod(int(size) for size in tensor.shape[1:])
            * element_size
        )
        manager_page_stride = (
            int(tensor.stride(0)) * element_size * kernel_rows_per_page
        )
        if manager_page_stride <= 0 or not (
            logical_page_bytes <= page_bytes <= manager_page_stride
        ):
            raise RuntimeError(
                "sparkcache: layer "
                f"{name} page geometry is incompatible: logical={logical_page_bytes}"
                f" physical={page_bytes} stride={manager_page_stride}"
            )
        try:
            byte_view = tensor.view(torch.uint8)
            storage_offset = int(byte_view.storage_offset())
            required_storage_bytes = (
                storage_offset + (manager_blocks - 1) * manager_page_stride + page_bytes
            )
            storage_bytes = int(byte_view.untyped_storage().nbytes())
            if required_storage_bytes > storage_bytes:
                raise RuntimeError(
                    f"physical pages require {required_storage_bytes} bytes"
                    f" from storage of {storage_bytes} bytes"
                )
            return torch.as_strided(
                byte_view,
                size=(manager_blocks, 1, page_bytes),
                stride=(manager_page_stride, page_bytes, 1),
                storage_offset=storage_offset,
            )
        except RuntimeError as error:
            raise RuntimeError(
                f"sparkcache: layer {name} physical page view is invalid"
            ) from error

    @staticmethod
    def _layer_page_bytes(group_spec: Any, layer_name: str) -> int:
        """Resolve one layer's physical page size from its manager group."""

        layer_specs = getattr(group_spec, "kv_cache_specs", None)
        if isinstance(layer_specs, Mapping):
            layer_spec = layer_specs.get(layer_name)
            if layer_spec is None:
                raise RuntimeError(
                    "sparkcache: group has no cache specification for"
                    f" layer {layer_name}"
                )
        else:
            layer_spec = group_spec
        page_bytes = int(getattr(layer_spec, "page_size_bytes", 0) or 0)
        if page_bytes <= 0:
            raise RuntimeError(
                f"sparkcache: layer {layer_name} has no physical page size"
            )
        return page_bytes

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        registered_tensors = dict(kv_caches)
        manager_page_summary: dict[str, int] | None = None
        if self._storage_mode == "block_pages_v1":
            groups = tuple(getattr(self._kv_cache_config, "kv_cache_groups", ()) or ())
            manager_blocks = int(getattr(self._kv_cache_config, "num_blocks", 0) or 0)
            page_bytes_by_layer: dict[str, int] = {}
            for group in groups:
                spec = getattr(group, "kv_cache_spec", None)
                for name in getattr(group, "layer_names", ()) or ():
                    if name in page_bytes_by_layer:
                        raise RuntimeError(
                            "sparkcache: layer belongs to multiple"
                            f" KV-cache groups: {name}"
                        )
                    page_bytes_by_layer[name] = self._layer_page_bytes(spec, name)
            unassigned = set(registered_tensors) - set(page_bytes_by_layer)
            missing = set(page_bytes_by_layer) - set(registered_tensors)
            if unassigned or missing:
                raise RuntimeError(
                    "sparkcache: registered layers disagree with KV-cache"
                    f" groups: missing={sorted(missing)}, extra={sorted(unassigned)}"
                )
            split_kernel_layers = 0
            physical_tail_layers = 0
            physical_tail_bytes = 0
            for name, tensor in registered_tensors.items():
                if tensor.dim() < 2:
                    continue
                kernel_rows = int(tensor.shape[0])
                if manager_blocks > 0 and kernel_rows % manager_blocks == 0:
                    kernel_rows_per_page = kernel_rows // manager_blocks
                    split_kernel_layers += int(kernel_rows_per_page > 1)
                    logical_page_bytes = (
                        kernel_rows_per_page
                        * math.prod(int(size) for size in tensor.shape[1:])
                        * int(tensor.element_size())
                    )
                    tail_bytes = page_bytes_by_layer[name] - logical_page_bytes
                    if tail_bytes > 0:
                        physical_tail_layers += 1
                        physical_tail_bytes += tail_bytes
            registered_tensors = {
                name: self._manager_page_view(
                    name,
                    tensor,
                    page_bytes_by_layer[name],
                )
                for name, tensor in registered_tensors.items()
            }
            manager_page_summary = {
                "blocks": manager_blocks,
                "layers": len(registered_tensors),
                "split_kernel_layers": split_kernel_layers,
                "physical_tail_layers": physical_tail_layers,
                "physical_tail_bytes_per_page": physical_tail_bytes,
            }
            logger.debug(
                "sparkcache: manager_pages blocks=%d layers=%d"
                " split_kernel_layers=%d tail_layers=%d"
                " tail_bytes_per_page=%d",
                manager_page_summary["blocks"],
                manager_page_summary["layers"],
                manager_page_summary["split_kernel_layers"],
                manager_page_summary["physical_tail_layers"],
                manager_page_summary["physical_tail_bytes_per_page"],
            )
        self._layer_tensors = registered_tensors
        if self._storage_mode == "block_pages_v1":
            page_groups = []
            assigned: set[str] = set()
            for group in groups:
                spec = getattr(group, "kv_cache_spec", None)
                layers = []
                for name in sorted(getattr(group, "layer_names", ()) or ()):
                    if name not in registered_tensors:
                        raise RuntimeError(
                            f"sparkcache: group layer {name} was not"
                            " registered"
                        )
                    tensor = registered_tensors[name]
                    page_shape = tuple(int(size) for size in tensor.shape[1:])
                    layers.append(
                        PageLayer(
                            name=name,
                            dtype=str(tensor.dtype),
                            page_shape=page_shape,
                            bytes_per_page=(
                                math.prod(page_shape) * tensor.element_size()
                            ),
                        )
                    )
                    assigned.add(name)
                page_groups.append(
                    PageGroup(
                        block_size=int(getattr(spec, "block_size", 0) or 0),
                        layers=tuple(layers),
                        reuse_window_tokens=self._group_topology[len(page_groups)][
                            "reuse_window_tokens"
                        ],
                        reuse_policy=self._group_topology[len(page_groups)][
                            "reuse_policy"
                        ],
                    )
                )
            unassigned = set(registered_tensors) - assigned
            if unassigned:
                raise RuntimeError(
                    "sparkcache: registered layers lack a KV-cache"
                    " group: " + ", ".join(sorted(unassigned))
                )
            self._page_layout = PageLayout(tuple(page_groups))
        widths = {}
        for name, tensor in sorted(registered_tensors.items()):
            if tensor.dim() < 3:
                raise RuntimeError(
                    f"sparkcache: layer {name} has unexpected"
                    f" shape {tuple(tensor.shape)}"
                )
            row = tensor[0, 0]
            widths[name] = int(row.numel() * row.element_size())
        profile = self._profile
        draft_named = any(
            classify_layer(name, profile.classification_rules, profile.default_family)
            == "mtp_draft_kv"
            for name in widths
        )
        policy = self._identity_base["draft_kv_policy"]
        expects_named = profile.expects_draft_named_layers(policy)
        if draft_named != expects_named:
            raise RuntimeError(
                "sparkcache: draft-layer registration disagrees with"
                f" the draft policy: profile {profile.name} under"
                f" draft_kv_policy={policy} expects draft-classified layer"
                f" names: {expects_named}; observed: {draft_named}. Set"
                " spark_cache_draft_policy to match how this runtime"
                " registers its drafter layers"
            )
        self._plans = build_layer_plans(
            widths,
            allow_missing=profile.allow_missing(policy),
            required_families=profile.required_families,
            classification_rules=profile.classification_rules,
            default_family=profile.default_family,
        )
        self._record_kinds = tuple(sorted({plan.record_kind for plan in self._plans}))
        kinds: dict[str, int] = {}
        for plan in self._plans:
            kinds[plan.record_kind] = kinds.get(plan.record_kind, 0) + 1
        item_bytes: dict[int, int] = {}
        for width in widths.values():
            item_bytes[width] = item_bytes.get(width, 0) + 1
        layer_records = {plan.name: plan.record_kind for plan in self._plans}
        self._startup_layer_inventory = {
            "storage": self._storage_mode,
            "draft_policy": self._identity_base.get("draft_kv_policy", "separate"),
            "layer_count": len(self._plans),
            "record_counts": dict(sorted(kinds.items())),
            "item_byte_counts": {
                str(width): count for width, count in sorted(item_bytes.items())
            },
            "manager_pages": manager_page_summary,
            "layers": [
                {
                    "name": name,
                    "item_bytes": widths[name],
                    "record": layer_records[name],
                }
                for name in sorted(widths)
            ],
        }
        self.counters["startup_layers_registered"] = len(self._plans)
        self.counters["startup_record_kinds"] = len(kinds)
        self.counters["startup_item_byte_widths"] = len(item_bytes)
        if manager_page_summary is not None:
            self.counters["startup_manager_blocks"] = manager_page_summary["blocks"]
            self.counters["startup_split_kernel_layers"] = manager_page_summary[
                "split_kernel_layers"
            ]
            self.counters["startup_physical_tail_layers"] = manager_page_summary[
                "physical_tail_layers"
            ]
            self.counters["startup_physical_tail_bytes_per_page"] = (
                manager_page_summary["physical_tail_bytes_per_page"]
            )
        logger.debug(
            "sparkcache: layers total=%d storage=%s records=%s item_bytes=%s"
            " draft_policy=%s",
            len(self._plans),
            self._storage_mode,
            json.dumps(dict(sorted(kinds.items())), separators=(",", ":")),
            json.dumps(
                self._startup_layer_inventory["item_byte_counts"],
                separators=(",", ":"),
            ),
            self._identity_base.get("draft_kv_policy", "separate"),
        )
        logger.debug(
            "sparkcache: layer_inventory %s",
            json.dumps(
                self._startup_layer_inventory,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if self._native_restore_enabled and self._role is KVConnectorRole.WORKER:
            self._configure_native_restore()
        if self._streaming_snapshots_enabled and self._role is KVConnectorRole.WORKER:
            runtime = self._streaming_runtime
            if runtime is None:
                raise RuntimeError(
                    "sparkcache: streaming worker runtime vanished"
                )
            # The adapter closes over this connector. Bind only after the
            # complete tensor inventory and canonical layer plans exist.
            runtime.bind_kv_caches()
        if self._async_page_capture_enabled and self._role is KVConnectorRole.WORKER:
            factory = _ASYNC_PAGE_RUNTIME_FACTORY
            if factory is None:
                from sparkcache.streaming.manager_page_factory import (
                    build_manager_page_runtime,
                )

                settings = self._async_page_capture_settings
                if settings is None:
                    raise RuntimeError(
                        "sparkcache: manager-page capture settings vanished"
                    )
                runtime = build_manager_page_runtime(self, settings)
            else:
                runtime = factory(self)
            required = (
                "submit",
                "preempt",
                "finish_without_capture",
                "take_finished",
                "quiesce",
                "shutdown",
            )
            if self._uses_capture_job_leases():
                required += ("finish_failed_job",)
            missing = [
                name for name in required if not callable(getattr(runtime, name, None))
            ]
            if missing:
                raise RuntimeError(
                    "sparkcache: manager-page capture runtime is incomplete: "
                    + ", ".join(missing)
                )
            self._async_page_capture_runtime = runtime
        if (
            self._cache_available
            and self._capacity_policy.enabled
            and self._role is KVConnectorRole.WORKER
        ):
            report = self._maintain_capacity(force=True)
            self._ensure_capacity_thread()
            if (report is not None and (report.skipped_busy or report.skipped_cooldown)) or not bool(
                self._capacity_status["capacity_satisfied"]
            ):
                self._capacity_wakeup.set()
        if self._restore_enabled:
            self.discover_manifests()

    def _configure_native_restore(self) -> None:
        """Attest SparkCache CUDA placement after final CUDA inventory."""

        if self._native_adapter is not None:
            raise RuntimeError(
                "sparkcache: SparkCache CUDA placement is already configured"
            )
        if self._storage_mode == "block_pages_v1":
            self._configure_native_hybrid_restore()
            return
        assert self._plans is not None
        if not self._plans:
            raise RuntimeError(
                "sparkcache: SparkCache CUDA restore has no layer plans"
            )
        first_tensor = self._layer_tensors[self._plans[0].name]
        device = first_tensor.device
        if device.type != "cuda":
            raise RuntimeError(
                "sparkcache: SparkCache CUDA restore requires CUDA"
                " cache tensors"
            )
        device_ordinal = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        capacities = {
            int(tensor.shape[0]) * int(tensor.shape[1])
            for tensor in self._layer_tensors.values()
        }
        if len(capacities) != 1:
            # Restore addresses every layer through one shared slot-index
            # space; heterogeneous pool geometries would scatter into wrong
            # slots on the larger pools.
            raise RuntimeError(
                "sparkcache: registered layers disagree on pool"
                f" geometry (slot capacities {sorted(capacities)})"
            )
        slot_capacity = capacities.pop()
        local_rows_per_chunk = self._chunk_tokens // self._dcp_degree
        payload_floor = local_rows_per_chunk * (
            4 + sum(plan.bytes_per_token for plan in self._plans)
        )
        if payload_floor >= self._native_arena_bytes:
            # Per-rank chunk bytes scale inversely with dcp_degree (DCP1
            # stores the full chunk on every rank); an arena that cannot
            # hold one encoded chunk would fail every restore plan later.
            raise RuntimeError(
                "sparkcache: SparkCache CUDA placement arena"
                f" ({self._native_arena_bytes} bytes) cannot hold one"
                f" encoded chunk (floor {payload_floor} bytes) at"
                f" dcp_degree={self._dcp_degree}"
            )
        max_chunks_per_slab = min(
            4096,
            max(1, self._native_arena_bytes // max(payload_floor, 1) + 1),
        )
        adapter = None
        try:
            components = _load_cuda_components()
            library = components.CudaPlacementLibrary.load(
                self._native_library_path,
                expected_sha256=self._native_library_sha256,
            )
            adapter = components.CudaPlacementAdapter.create(
                library,
                arena_mode=components.ArenaMode.MAPPED_HOST,
                arena_bytes=self._native_arena_bytes,
                max_destinations=len(self._plans),
                max_slots=slot_capacity,
                max_chunks_per_slab=max_chunks_per_slab,
                device_ordinal=device_ordinal,
            )
            adapter.configure(self._plans, self._layer_tensors)
            required = 0
            ordinal_names = {
                "target_ckv": "TARGET_CKV",
                "sparse_indexer": "SPARSE_INDEXER",
                "mtp_draft_kv": "MTP_DRAFT_KV",
                "boundary_hidden": "BOUNDARY_HIDDEN",
            }
            for kind in self._record_kinds:
                ordinal = getattr(
                    components.RecordKind, ordinal_names.get(kind, ""), None
                )
                if ordinal is None:
                    raise RuntimeError(
                        f"sparkcache: record kind {kind} has no"
                        " SparkCache CUDA placement ordinal"
                    )
                required |= 1 << int(ordinal)
            native_execute = components.execute_cuda_restore
            if not callable(native_execute):
                raise TypeError("SparkCache CUDA restore orchestrator is not callable")
        except Exception as error:
            if adapter is not None:
                with contextlib.suppress(Exception):
                    adapter.close()
            raise RuntimeError(
                f"sparkcache: SparkCache CUDA restore configuration"
                f" was rejected: {error}"
            ) from error
        self._native_adapter = adapter
        self._native_execute_restore = native_execute
        self._native_required_record_mask = required
        self.counters["native_configured"] = 1
        logger.debug(
            "sparkcache: cuda_restore library=%s"
            " sha256=%s arena_mib=%d destinations=%d slots=%d"
            " max_chunks_per_slab=%d",
            self._native_library_path,
            self._native_library_sha256,
            self._native_arena_bytes // (1024 * 1024),
            len(self._plans),
            slot_capacity,
            max_chunks_per_slab,
        )

    def _configure_native_hybrid_restore(self) -> None:
        layout = self._page_layout
        if layout is None:
            raise RuntimeError("SparkCache CUDA restore has no page layout")
        first_layer = layout.groups[0].layers[0]
        first_tensor = self._layer_tensors[first_layer.name]
        device = first_tensor.device
        if device.type != "cuda":
            raise RuntimeError("SparkCache CUDA restore requires CUDA page tensors")
        device_ordinal = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        destination_count = sum(len(group.layers) for group in layout.groups)
        max_slots = 0
        for group in layout.groups:
            capacities = {
                int(self._layer_tensors[layer.name].shape[0]) for layer in group.layers
            }
            if len(capacities) != 1:
                raise RuntimeError(
                    "SparkCache CUDA page group layers disagree on capacity"
                )
            max_slots += capacities.pop()
        adapters = []
        try:
            components = _load_cuda_components()
            library = components.CudaPlacementLibrary.load(
                self._native_library_path,
                expected_sha256=self._native_library_sha256,
            )
            components.bind_page_reference(library.cdll)
            if (
                int(library.abi_info.capability_flags)
                & int(components.hybrid_page_cuda_capability)
                == 0
            ):
                raise RuntimeError(
                    "SparkCache CUDA placement library lacks page scatter"
                )
            for _lane in range(self._load_thread_limit):
                adapter = components.CudaPlacementAdapter.create(
                    library,
                    arena_mode=components.ArenaMode.MAPPED_HOST,
                    arena_bytes=self._native_arena_bytes,
                    max_destinations=destination_count,
                    max_slots=max_slots,
                    max_chunks_per_slab=4096,
                    device_ordinal=device_ordinal,
                )
                adapters.append(adapter)
                adapter.configure_pages(layout, self._layer_tensors)
            execute_restore = components.execute_cuda_hybrid_restore
            execute_placement = components.execute_cuda_hybrid_placement
            if not callable(execute_restore):
                raise TypeError("SparkCache CUDA restore orchestrator is not callable")
            if not callable(execute_placement):
                raise TypeError(
                    "SparkCache CUDA page-placement orchestrator is not callable"
                )
        except Exception as error:
            for adapter in adapters:
                with contextlib.suppress(Exception):
                    adapter.close()
            raise RuntimeError(
                "sparkcache: SparkCache CUDA restore configuration"
                f" did not complete: {error}"
            ) from error
        self._native_adapters = adapters
        self._native_adapter = adapters[0]
        self._native_execute_hybrid_restore = execute_restore
        self._native_execute_hybrid_placement = execute_placement
        self.counters["native_hybrid_configured"] = 1
        logger.debug(
            "sparkcache: cuda_restore"
            " destinations=%d max_slots=%d arena_mib=%d lanes=%d",
            destination_count,
            max_slots,
            self._native_arena_bytes // (1024 * 1024),
            len(adapters),
        )

    def _maintain_capacity(
        self,
        *,
        force: bool = False,
        wake_worker_on_unsatisfied: bool = True,
    ) -> MaintenanceReport | None:
        with self._capacity_lock:
            return self._maintain_capacity_locked(
                force=force,
                wake_worker_on_unsatisfied=wake_worker_on_unsatisfied,
            )

    def _maintain_capacity_locked(
        self,
        *,
        force: bool = False,
        wake_worker_on_unsatisfied: bool = False,
    ) -> MaintenanceReport | None:
        """Run one capacity pass while ``_capacity_lock`` is held."""

        policy = self._capacity_policy
        if not policy.enabled:
            return None
        if not force and (
            (policy.max_bytes == 0 or self._capacity_estimated_bytes <= policy.max_bytes)
            and not self._capacity_status.get("maintenance_work_pending", False)
        ):
            return None
        self._capacity_maintenance_depth = getattr(self, "_capacity_maintenance_depth", 0) + 1
        try:
            return self._perform_capacity_maintenance_locked(
                policy, force=force, wake_worker_on_unsatisfied=wake_worker_on_unsatisfied
            )
        finally:
            self._capacity_maintenance_depth -= 1

    def _perform_capacity_maintenance_locked(
        self,
        policy: CapacityPolicy,
        *,
        force: bool,
        wake_worker_on_unsatisfied: bool,
    ) -> MaintenanceReport | None:
        """Keep scans and survivor reconciliation within the maintenance activity gauge."""
        try:
            with self._store_cv:
                held_snapshot = (self._held, self._held.revision, set(self._held))
                protected = tuple(
                    key
                    for result_digest, base in getattr(self, "_publication_base_pins", {}).items()
                    for key in (base, EntryKey(base.storage_key, result_digest))
                )
            report = self._store.maintain(
                policy, **({"protected_entries": protected} if protected else {})
            )
        except Exception as error:  # noqa: BLE001 - maintenance is nonfatal
            self.counters["capacity_failed"] += 1
            self._capacity_status.update(
                bytes=self._capacity_estimated_bytes,
                bytes_exact=False,
                capacity_satisfied=False,
            )
            self._reconcile_held_capacity()
            if wake_worker_on_unsatisfied:
                self._capacity_wakeup.set()
            _warning(
                "sparkcache: capacity maintenance failed: %s", error
            )
            return None
        if report.skipped_cooldown:
            self.counters["capacity_skipped_cooldown"] = (
                self.counters.get("capacity_skipped_cooldown", 0) + 1
            )
            if wake_worker_on_unsatisfied:
                self._capacity_wakeup.set()
            return report
        if report.skipped_busy:
            self.counters["capacity_skipped_busy"] += 1
            self._capacity_status.update(
                bytes=self._capacity_estimated_bytes,
                bytes_exact=False,
                capacity_satisfied=False,
            )
            self._reconcile_held_capacity()
            if wake_worker_on_unsatisfied:
                self._capacity_wakeup.set()
            return report
        self._capacity_estimated_bytes = report.bytes_after
        self._capacity_status.update(
            bytes=report.bytes_after,
            bytes_exact=True,
            capacity_satisfied=report.capacity_satisfied,
            maintenance_work_pending=report.work_pending,
        )
        self.counters["capacity_deletion_attempts"] = (
            self.counters.get("capacity_deletion_attempts", 0) + report.deletion_attempts
        )
        self.counters["capacity_budget_exhausted"] = (
            self.counters.get("capacity_budget_exhausted", 0) + int(report.work_pending)
        )
        self.counters["capacity_runs"] += 1
        self.counters["capacity_manifests_evicted"] += report.manifests_evicted
        self.counters["capacity_chunks_deleted"] += report.chunks_deleted
        self.counters["capacity_bytes_reclaimed"] += report.bytes_reclaimed
        self.counters["prefix_alias_capacity_evicted"] += int(
            getattr(report, "aliases_evicted", 0)
        )
        self.counters["prefix_alias_segments_deleted"] += int(
            getattr(report, "segments_deleted", 0)
        )
        if report.surviving_entries is None:
            # A report without an inventory cannot authorize withdrawals.
            self._reconcile_held_capacity()
        elif held_snapshot[2]:
            identity = self._identity(self._worker_rank())
            surviving = {
                entry.context_digest for entry in report.surviving_entries
                if entry.storage_key == identity.storage_key
                and (entry.root_kind == "manifest" or self._storage_mode == "per_token_rows")
            }
            self._apply_held_capacity_survivors(held_snapshot, surviving)
        if (not report.capacity_satisfied or report.work_pending) and wake_worker_on_unsatisfied:
            self._capacity_wakeup.set()
        if report.manifests_evicted or report.bytes_reclaimed:
            logger.info("sparkcache: eviction entries=%d bytes=%d rank=%d",
                        report.manifests_evicted, report.bytes_reclaimed, self._physical_rank())
        if force or report.bytes_reclaimed:
            logger.debug(
                "sparkcache: capacity bytes=%d max=%d reclaimed=%d"
                " manifests=%d chunks=%d orphans=%d satisfied=%s"
                " deletion_attempts=%d work_pending=%s",
                report.bytes_after,
                policy.max_bytes,
                report.bytes_reclaimed,
                report.manifests_evicted,
                report.chunks_deleted,
                report.orphan_chunks_deleted,
                report.capacity_satisfied,
                report.deletion_attempts,
                report.work_pending,
            )
        return report

    def _post_commit_was_evicted_locked(
        self,
        identity: CacheIdentity,
        context_digest: str,
        *,
        force_maintenance: bool = False,
    ) -> bool:
        """Maintain capacity; read back only when the outcome is ambiguous."""

        policy = self._capacity_policy
        maintenance_required = policy.enabled and (
            force_maintenance
            or (
                policy.max_bytes > 0
                and self._capacity_estimated_bytes > policy.max_bytes
            )
        )
        report = self._maintain_capacity_locked(
            force=force_maintenance,
            wake_worker_on_unsatisfied=True,
        )
        if report is not None and not (report.skipped_busy or report.skipped_cooldown):
            exact_evicted = (
                EntryKey(
                    identity.storage_key,
                    context_digest,
                )
                in report.evicted_entries
            )
            if not exact_evicted:
                return False
            alias_evicted = (
                EntryKey(
                    identity.storage_key,
                    context_digest,
                    "prefix_alias",
                )
                in report.evicted_entries
            )
            alias_path = (
                Path(self._root)
                / "prefix-aliases"
                / identity.storage_key
                / f"{context_digest}.json"
            )
            if (
                self._storage_mode == "per_token_rows"
                and not alias_evicted
                and alias_path.exists()
            ):
                lookup, _is_alias = self._lookup_reusable(
                    identity,
                    context_digest,
                    verify_chunks=False,
                    verify_chunk_metadata=True,
                )
                return not lookup.is_hit
            return True
        if not maintenance_required:
            return False

        # A failed or concurrently skipped pass cannot prove which manifests
        # survived. Retain verified survivor readback for that ambiguous path.
        return not self._store.lookup(
            identity,
            context_digest,
            verify_chunks=False,
            verify_chunk_metadata=True,
        ).is_hit

    def _note_capacity_commit(self, encoded_bytes: int) -> None:
        with self._capacity_lock:
            self._note_capacity_commit_locked(encoded_bytes)

    def _note_capacity_commit_locked(self, encoded_bytes: int) -> None:
        if type(encoded_bytes) is not int or encoded_bytes < 0:
            raise ValueError("capacity commit bytes must be a non-negative integer")
        self._capacity_estimated_bytes += encoded_bytes
        max_bytes = self._capacity_policy.max_bytes
        previously_satisfied = bool(self._capacity_status["capacity_satisfied"])
        self._capacity_status.update(
            bytes=self._capacity_estimated_bytes,
            bytes_exact=False,
            capacity_satisfied=(
                previously_satisfied
                and (max_bytes == 0 or self._capacity_estimated_bytes <= max_bytes)
            ),
        )

    def _streaming_commit_allocated_bytes(
        self,
        context_digest: str,
        receipt: Any,
    ) -> int:
        """Validate the stable commit-receipt fields without class identity.

        Staged deployments load the storage engine through a compatibility
        shim, while the publisher imports its package path directly. The two
        receipts are structurally identical but may not share Python class
        identity, so this boundary deliberately validates the wire fields.
        """

        if SHA256_RE.fullmatch(context_digest) is None:
            raise RuntimeError("streaming commit context digest is invalid")
        manifest_digest = getattr(receipt, "manifest_digest", None)
        committed_tokens = getattr(receipt, "committed_tokens", None)
        encoded_bytes = getattr(receipt, "encoded_bytes", None)
        allocated_bytes = getattr(
            receipt,
            "allocated_bytes_upper_bound",
            None,
        )
        if (
            not isinstance(manifest_digest, str)
            or SHA256_RE.fullmatch(manifest_digest) is None
            or type(committed_tokens) is not int
            or committed_tokens <= 0
            or committed_tokens % self._chunk_tokens != 0
            or committed_tokens > self._max_span
            or type(encoded_bytes) is not int
            or encoded_bytes <= 0
            or type(allocated_bytes) is not int
            or allocated_bytes < encoded_bytes
        ):
            raise RuntimeError("streaming commit receipt is invalid")
        return allocated_bytes

    def _reject_streaming_capacity_receipts(self, count: int) -> None:
        """Drop invalid handoffs and request an exact background scan."""

        if count <= 0:
            return
        with self._capacity_handoff_cv:
            self.counters["streaming_capacity_invalid_receipts"] += count
            self._capacity_status.update(
                bytes_exact=False,
                capacity_satisfied=False,
            )
        _warning(
            "sparkcache: dropped %d invalid streaming commit"
            " receipt(s); scheduling exact capacity maintenance",
            count,
        )
        self._capacity_wakeup.set()

    def _handoff_streaming_commits(
        self,
        committed: Mapping[str, Any],
    ) -> set[str]:
        """Accept durable receipts without running maintenance on callbacks.

        Unbounded deployments advertise each durable receipt immediately.
        A bounded or TTL-managed deployment queues the receipt for the
        capacity worker; it returns no advertised digest until that worker has
        serialized accounting, maintenance, survivor verification, and the
        ``_held`` update.
        """

        if not self._capacity_policy.enabled:
            # Capacity accounting is disabled, so receipt byte fields have no
            # bearing on the established post-manifest advertisement path.
            advertised = {str(raw_digest) for raw_digest in committed}
            with self._store_cv:
                self._held.update(advertised)
                self.counters["streaming_store_committed"] += len(advertised)
            return advertised

        staged: list[tuple[str, Any, int]] = []
        invalid_receipts = 0
        for raw_digest, receipt in committed.items():
            try:
                digest = str(raw_digest)
                allocated_bytes = self._streaming_commit_allocated_bytes(
                    digest,
                    receipt,
                )
            except Exception:  # noqa: BLE001 - invalid metadata stays unadvertised
                invalid_receipts += 1
                continue
            staged.append((digest, receipt, allocated_bytes))
        self._reject_streaming_capacity_receipts(invalid_receipts)
        if not staged:
            return set()

        with self._store_cv:
            already_advertised = {
                digest for digest, _receipt, _bytes in staged if digest in self._held
            }
        queued = 0
        with self._capacity_handoff_cv:
            if self._capacity_stop.is_set():
                self.counters["streaming_capacity_shutdown_dropped"] += len(
                    {digest for digest, _receipt, _bytes in staged} - already_advertised
                )
                return already_advertised
            for digest, receipt, _allocated_bytes in staged:
                if (
                    digest in already_advertised
                    or digest in self._streaming_capacity_pending
                ):
                    continue
                self._streaming_capacity_pending.add(digest)
                self._capacity_commit_queue.put((digest, receipt))
                queued += 1
        if queued:
            self.counters["streaming_capacity_queued"] += queued
            self._capacity_wakeup.set()
        return already_advertised

    def wait_for_pending_capacity_commits(
        self,
        timeout: float | None = None,
    ) -> bool:
        """Wait for tests/offline callers; serving callbacks never call this."""

        with self._capacity_handoff_cv:
            return self._capacity_handoff_cv.wait_for(
                lambda: not self._streaming_capacity_pending,
                timeout,
            )

    def _finalize_streaming_capacity_commits(
        self,
        pending: Mapping[str, _PendingStreamingCommit],
    ) -> bool:
        """Resolve one serialized batch, or retain all entries for retry."""

        with self._capacity_lock:
            for digest, entry in pending.items():
                if entry.accounted:
                    continue
                allocated_bytes = self._streaming_commit_allocated_bytes(
                    digest,
                    entry.receipt,
                )
                self._note_capacity_commit_locked(allocated_bytes)
                entry.accounted = True

            policy = self._capacity_policy
            maintenance_required = not bool(
                self._capacity_status["capacity_satisfied"]
            ) or (
                policy.max_bytes > 0
                and self._capacity_estimated_bytes > policy.max_bytes
            )
            if maintenance_required:
                report = self._maintain_capacity_locked(force=True)
                if (
                    report is None
                    or (report.skipped_busy or report.skipped_cooldown)
                    or not report.capacity_satisfied
                ):
                    return False

            identity = self._identity(self._worker_rank())
            try:
                surviving = {
                    digest
                    for digest in pending
                    if self._store.lookup(
                        identity,
                        digest,
                        verify_chunks=False,
                        verify_chunk_metadata=True,
                    ).is_hit
                }
            except Exception as error:  # noqa: BLE001 - leave unverified and retry
                self.counters["capacity_failed"] += 1
                self._capacity_status.update(
                    bytes=self._capacity_estimated_bytes,
                    bytes_exact=False,
                    capacity_satisfied=False,
                )
                _warning(
                    "sparkcache: capacity survivor check failed: %s",
                    error,
                )
                return False

            evicted = set(pending) - surviving
            # This is the only point at which a bounded streamed commit may
            # become visible to worker stats. The capacity lock stays held
            # across the survivor check and this short in-memory update, so a
            # competing maintenance pass cannot evict then resurrect it.
            with self._store_cv:
                if self._capacity_stop.is_set():
                    return False
                self._held.update(surviving)
                self._held.difference_update(evicted)
                self.counters["streaming_store_committed"] += len(surviving)
                self.counters["streaming_store_evicted"] += len(evicted)
            return True

    def _apply_held_capacity_survivors(
        self,
        snapshot: tuple[HeldInventory, int, set[str]],
        surviving: set[str],
    ) -> None:
        inventory, revision, held = snapshot
        with self._store_cv:
            if self._held is not inventory or self._held.revision != revision:
                # A digest can be withdrawn and republished during a probe.
                # An earlier inventory cannot revoke that publication; a
                # stable pass reconciles it, and restores still verify bytes.
                self.counters["capacity_stale_inventory_snapshots"] = (
                    self.counters.get("capacity_stale_inventory_snapshots", 0) + 1
                )
                return
            self._held.difference_update(held - surviving)

    def _reconcile_held_capacity(self) -> None:
        with self._store_cv:
            snapshot = (self._held, self._held.revision, set(self._held))
        if not snapshot[2]:
            return
        rank = self._worker_rank()
        identity = self._identity(rank)
        surviving = set()
        for digest in snapshot[2]:
            lookup, _is_alias = self._lookup_reusable(
                identity,
                digest,
                verify_chunks=False,
                verify_chunk_metadata=True,
            )
            if lookup.is_hit:
                surviving.add(digest)
        self._apply_held_capacity_survivors(snapshot, surviving)

    def _ensure_capacity_thread(self) -> None:
        if self._capacity_thread is not None:
            return
        thread = threading.Thread(
            target=self._capacity_worker_main,
            name="spark-cache-capacity",
            daemon=True,
        )
        self._capacity_thread = thread
        try:
            thread.start()
        except BaseException:
            self._capacity_thread = None
            raise

    def _capacity_worker_main(self) -> None:
        ttl_seconds = self._capacity_policy.ttl_seconds
        # TTL expiry needs a clock-driven pass. With TTL disabled, startup
        # performs one exact scan and commit accounting wakes this worker on
        # pressure or retry; an idle timer would only rescan an unchanged tree.
        interval = min(60, max(1, ttl_seconds // 2)) if ttl_seconds else None
        pending: dict[str, _PendingStreamingCommit] = {}
        retry_unsatisfied = False
        while not self._capacity_stop.is_set():
            timeout = (
                _CAPACITY_RETRY_SECONDS if pending or retry_unsatisfied else interval
            )
            self._capacity_wakeup.wait(timeout=timeout)
            self._capacity_wakeup.clear()
            if self._capacity_stop.is_set():
                return
            while True:
                try:
                    digest, receipt = self._capacity_commit_queue.get_nowait()
                except queue.Empty:
                    break
                pending.setdefault(
                    digest,
                    _PendingStreamingCommit(receipt=receipt),
                )

            if pending:
                if self._finalize_streaming_capacity_commits(pending):
                    resolved = set(pending)
                    pending.clear()
                    with self._capacity_handoff_cv:
                        self._streaming_capacity_pending.difference_update(resolved)
                        self._capacity_handoff_cv.notify_all()
                    retry_unsatisfied = bool(
                        self._capacity_status.get("maintenance_work_pending", False)
                    )
                else:
                    self.counters["streaming_capacity_retries"] += 1
                    retry_unsatisfied = True
                continue

            report = self._maintain_capacity(
                force=True,
                wake_worker_on_unsatisfied=False,
            )
            retry_unsatisfied = bool(
                report is None
                or not report.capacity_satisfied
                or report.work_pending
                or not bool(self._capacity_status["capacity_satisfied"])
            )
            if retry_unsatisfied:
                self.counters["capacity_retries"] += 1

    def _schema_miss_reason(
        self,
        token_ids: Sequence[int],
        candidates: Sequence[tuple[int, str]],
        multimodal_features: Sequence[MultimodalFeatureIdentity],
        request_scope: str,
    ) -> str:
        """Diagnose another schema's exact prefix without admitting its bytes."""
        evidence = set().union(*self._worker_incompatible.values())
        compatible = dict(candidates)
        for schema in {schema for schema, _digest in evidence}:
            identity = replace(self._config.build_identity(0, 0), publication_schema=schema)
            salt = f"{_CONTEXT_DIGEST_NAMESPACE}:{identity.storage_key}:{request_scope}"
            alternate = chunk_prefix_digests(
                token_ids, salt, boundaries=[span for span, _digest in candidates],
                chunk_tokens=self._chunk_tokens, multimodal_features=multimodal_features,
            )
            if any(
                (schema, digest) in entries
                and compatible[span] not in self._worker_held.get(rank, ())
                for span, digest in alternate
                for rank, entries in self._worker_incompatible.items()
            ):
                return "integrity_schema_mismatch"
        return "quorum"

    def _discover_incompatible_manifests(self, identity: CacheIdentity) -> None:
        # Only publication schema may differ. Model, topology, request scope,
        # and geometry remain bound by identity and prefix digests. This
        # startup metadata is diagnostic evidence, never a restore offer.
        entries = []
        for schema in _DIAGNOSTIC_PUBLICATION_SCHEMAS:
            if schema == identity.publication_schema:
                continue
            alternate = replace(identity, publication_schema=schema)
            root = Path(self._root) / "manifests" / alternate.storage_key
            # Diagnostic failure must not suppress compatible inventory.
            with contextlib.suppress(OSError, ValueError, RuntimeError):
                for path in root.glob("*.json"):
                    if SHA256_RE.fullmatch(path.stem) is None:
                        continue
                    lookup = self._store.lookup(
                        alternate, path.stem, verify_chunks=False, verify_chunk_metadata=True,
                    )
                    if lookup.is_hit:
                        entries.append((schema, path.stem))
        self._incompatible_inventory = tuple(sorted(entries))
        self._incompatible_cursor = 0

    def _absorb_incompatible_manifests(self, rank: int, report: dict[str, Any]) -> None:
        entries = report.get("incompatible", ())
        if not isinstance(entries, (list, tuple)) or len(entries) > _QUORUM_REPORT_BATCH_SIZE:
            return
        for entry in entries:
            if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                continue
            schema, digest = entry
            if (
                schema in _DIAGNOSTIC_PUBLICATION_SCHEMAS
                and schema != self._publication_schema
                and isinstance(digest, str) and SHA256_RE.fullmatch(digest)
            ):
                self._worker_incompatible.setdefault(rank, set()).add((schema, digest))

    def discover_manifests(self) -> dict[str, int]:
        """Offer structurally valid exact entries and safe row aliases.

        Referenced chunks must exist at their declared sizes; that metadata
        check does not fault payload pages into memory. Restore remains the
        byte/hash integrity boundary. Same-size corruption discovered there
        revokes the offer and degrades the request to clean recompute.
        """

        checked = rejected = 0
        discovered: set[str] = set()
        # Serialize discovery's snapshot replacement with async store
        # admission/completion. A commit may publish its manifest while this
        # scan is in progress; its completion then adds the durable digest
        # after discovery releases this lock instead of having that offer
        # erased by the replacement below.
        with self._store_cv:
            try:
                rank = self._worker_rank()
                identity = self._identity(rank)
                self._discover_incompatible_manifests(identity)
                roots = [
                    (
                        Path(self._root) / "manifests" / identity.storage_key,
                        False,
                    )
                ]
                if self._storage_mode == "per_token_rows":
                    roots.append(
                        (
                            Path(self._root) / "prefix-aliases" / identity.storage_key,
                            True,
                        )
                    )
                for root, candidate_is_alias in roots:
                    if not root.is_dir():
                        continue
                    for manifest_path in root.glob("*.json"):
                        digest = manifest_path.stem
                        # An exact manifest with this digest was already
                        # considered and always shadows a same-key alias.
                        if candidate_is_alias and digest in discovered:
                            continue
                        checked += 1
                        lookup, is_alias = self._lookup_reusable(
                            identity,
                            digest,
                            verify_chunks=False,
                            verify_chunk_metadata=True,
                        )
                        if lookup.is_hit:
                            discovered.add(digest)
                            if is_alias:
                                self.counters["prefix_aliases_discovered"] += 1
                        else:
                            rejected += 1
                            # A rejected manifest is not an authority for
                            # deleting content-addressed chunks. Its
                            # descriptors may be corrupt, malicious, or name
                            # chunks shared by a healthy manifest.
                            removed = self._invalidate_reusable(
                                identity,
                                digest,
                                is_alias=(candidate_is_alias or is_alias),
                                verify_chunk_payloads=False,
                            )
                            if not removed:
                                try:
                                    manifest_path.unlink()
                                except OSError:
                                    pass
            except (OSError, ValueError, RuntimeError) as error:
                _warning(
                    "sparkcache: manifest discovery aborted: %s",
                    error,
                )
            self._held = discovered
        if checked:
            logger.debug(
                "sparkcache: manifest discovery checked=%d"
                " offered=%d rejected=%d",
                checked,
                len(discovered),
                rejected,
            )
        self.counters["discovery_checked"] = (
            self.counters.get("discovery_checked", 0) + checked
        )
        self.counters["discovery_rejected"] = (
            self.counters.get("discovery_rejected", 0) + rejected
        )
        return {
            "checked": checked,
            "offered": len(discovered),
            "rejected": rejected,
        }

    def sweep_integrity(self) -> dict[str, int]:
        """Verify every entry this rank owns and invalidate damaged ones.

        This explicit diagnostic is not part of startup or request completion.
        A manifest enters the offered set directly after ManifestStore's
        durable commit. Startup performs manifest-only discovery, and every
        load independently re-verifies bytes. Returns
        {"checked": n, "invalidated": m}.
        """
        checked = invalidated = 0
        verified: set[str] = set()
        with self._load_lock:
            held_at_start = set(self._held)
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            if not held_at_start:
                with self._load_lock:
                    self._held.difference_update(held_at_start)
                return {"checked": 0, "invalidated": 0}
            for digest in sorted(held_at_start):
                checked += 1
                # restore() is the single parallel read/hash/decode pass.
                lookup, is_alias = self._lookup_reusable(
                    identity,
                    digest,
                    verify_chunks=False,
                )
                payload_verified = False
                if lookup.is_hit:
                    try:
                        if lookup.root_kind in ("page_delta", "page_snapshot"):
                            if self._page_layout is None:
                                raise RuntimeError(
                                    "block-page layout was not registered"
                                )
                            manifest = lookup._manifest or {}
                            self._store.restore_page_snapshot(
                                lookup,
                                layout=self._page_layout,
                                result_block_counts=manifest.get(
                                    "result_block_counts", ()
                                ),
                                result_boundary_tokens=manifest["committed_tokens"],
                            )
                            payload_verified = True
                        else:
                            payload_verified = self._store.restore(lookup) is not None
                    except (KeyError, RuntimeError, ValueError):
                        payload_verified = False
                if payload_verified:
                    verified.add(digest)
                    continue
                with self._load_lock:
                    self._held.discard(digest)
                if self._invalidate_reusable(
                    identity,
                    digest,
                    is_alias=is_alias,
                    verify_chunk_payloads=True,
                ):
                    invalidated += 1
                    _warning(
                        "sparkcache: sweep invalidated %s (%s)",
                        digest[:12],
                        lookup.reason,
                    )
            with self._load_lock:
                # Remove only offers that existed when the sweep began and
                # failed to verify. Preserve durable entries published while
                # the diagnostic was reading older manifests.
                self._held.difference_update(held_at_start - verified)
                self._held.update(verified)
        except (OSError, ValueError, RuntimeError) as error:
            _warning("sparkcache: sweep aborted: %s", error)
        if checked:
            logger.debug(
                "sparkcache: sweep checked=%d invalidated=%d",
                checked,
                invalidated,
            )
        self.counters["sweep_checked"] = self.counters.get("sweep_checked", 0) + checked
        self.counters["sweep_invalidated"] = (
            self.counters.get("sweep_invalidated", 0) + invalidated
        )
        return {"checked": checked, "invalidated": invalidated}

    def _physical_rank(self) -> int:
        """Return this worker's physical tensor-parallel rank.

        This is unique across all TP ranks (0..tp_degree-1), unlike
        _worker_rank() which returns the DCP-local rank (0..dcp_degree-1).
        Under TP4/DCP2, TP0 and TP2 both have DCP-local rank 0 but
        physical ranks 0 and 2 respectively.

        Used for quorum admission/withdrawal and persistent namespace
        identity (tp_shard_rank).  Token-position ownership and DCP
        slicing still use _worker_rank() (DCP-local rank).
        """
        from vllm.distributed import get_tensor_model_parallel_rank

        return int(get_tensor_model_parallel_rank())

    def _worker_rank(self) -> int:
        from vllm.distributed import get_dcp_group

        if self._dcp_degree <= 1:
            return 0
        return get_dcp_group().rank_in_group

    def _rows_view(self, tensor: torch.Tensor) -> torch.Tensor:
        # view, not reshape: restore scatters through this result, so it must
        # alias the registered KV storage. reshape silently returns a copy for
        # non-viewable layouts, and the scatter would write into a discarded
        # temporary while checksums and completion reporting still succeed.
        # view raises RuntimeError instead, which the load path converts into
        # a clean miss and recompute.
        num_blocks = tensor.shape[0]
        page = tensor.shape[1]
        return tensor.view(num_blocks * page, -1)

    def _ensure_load_threads(self) -> None:
        with self._load_lock:
            while len(self._load_threads) < self._load_thread_limit:
                thread = threading.Thread(
                    target=self._load_worker_main,
                    args=(len(self._load_threads),),
                    name=f"spark-cache-load-{len(self._load_threads)}",
                    daemon=True,
                )
                thread.start()
                self._load_threads.append(thread)

    def _load_worker_main(self, lane_index: int = 0) -> None:
        while True:
            queued = self._load_queue.get()
            if queued is None:
                return
            if isinstance(queued, _QueuedLoadBatch):
                self._prepare_queued_load_batch(queued)
                continue
            plan = queued.plan
            timing = queued.timing
            with contextlib.suppress(Exception):
                timing.start_service()
            try:
                with self._load_lock:
                    if plan.request_id in self._cancelled_load_reqs:
                        raise RuntimeError("request finished before restore placement")
                prerequisite_started = time.perf_counter_ns()
                if queued.prior_cuda_error is not None:
                    raise RuntimeError(queued.prior_cuda_error)
                if queued.prior_cuda_event is not None:
                    queued.prior_cuda_event.synchronize()
                with contextlib.suppress(Exception):
                    timing.observe(
                        "prior_cuda_work",
                        time.perf_counter_ns() - prerequisite_started,
                    )
                with self._load_lock:
                    if plan.request_id in self._cancelled_load_reqs:
                        raise RuntimeError("request finished before restore placement")
                verified = self._load_one(
                    plan,
                    timing=timing,
                    native_lane=lane_index,
                )
            except Exception as error:  # noqa: BLE001
                _warning(
                    "sparkcache: restore rejected; recomputing"
                    " request=%s reason=%s",
                    queued.plan.request_id,
                    error,
                )
                verified = False
            with contextlib.suppress(Exception):
                timing.finish("verified" if verified else "recompute")
            self._trace_reuse(
                "worker_restore_completed", plan.request_id,
                timing=timing, digest=plan.digest[:12],
                requested_span_tokens=plan.span_tokens,
                verified_span_tokens=plan.span_tokens if verified else 0,
                outcome="verified" if verified else "recompute",
            )
            with self._load_cv:
                if verified:
                    self.counters["load_verified"] += 1
                    # Metrics export worker reports, not scheduler admission counters.
                    # All restore backends converge here after verified placement.
                    self.counters["restore_hit"] += 1
                else:
                    # Publish invalid blocks before the finished id while
                    # holding the same lock. vLLM collects finished ids then
                    # load errors in one connector-output pass.
                    self._load_errors.update(
                        block for group in plan.group_block_ids for block in group
                        if block != 0  # Shared null padding is not a restore destination.
                    )
                    self.counters["load_failed"] += 1
                self._finished_load_reqs.add(plan.request_id)
                self._load_cv.notify_all()
            try:
                with contextlib.suppress(Exception):
                    _debug_log("%s", timing.render())
                if verified:
                    with contextlib.suppress(Exception):
                        ttl_seconds = self._capacity_policy.ttl_seconds
                        self._store.touch(
                            self._identity(self._worker_rank()),
                            plan.digest,
                            minimum_interval_seconds=(
                                min(60, ttl_seconds // 2) if ttl_seconds else 60
                            ),
                        )
                    with contextlib.suppress(Exception):
                        logger.debug("%s", timing.operator_lines()[1])
                with contextlib.suppress(Exception):
                    logger.debug(
                        "sparkcache: restore %s tokens=%d ms=%.1f rank=%d",
                        "hit" if verified else "miss", plan.span_tokens,
                        timing.end_to_end_ns / 1_000_000, self._physical_rank(),
                    )
            finally:
                # Request completion may be published immediately after the
                # verified placement edge, but connector quiescence includes
                # every later filesystem/logging action owned by this work
                # item. Tests and shutdown can therefore remove the cache root
                # only after no loader can recreate or reopen its metadata.
                page_base_key = self._page_base_plan_keys.get(plan.request_id)
                self._page_base_reads.finish(plan.request_id)
                if page_base_key is not None:
                    self._release_page_base_deferred(
                        page_base_key,
                        promote_registered=True,
                    )
                self._emit_page_base_flight_summaries()
                with self._load_cv:
                    self._page_base_plan_keys.pop(plan.request_id, None)
                    self._inflight_load_reqs.discard(plan.request_id)
                    self._cancelled_load_reqs.discard(plan.request_id)
                    self._stop_idle_loaders_locked()
                    self._load_cv.notify_all()

    def _stop_idle_loaders_locked(self) -> None:
        """Place stop sentinels only after all metadata and load work drains."""
        if (
            self._load_stop_requested
            and not self._load_stops_queued
            and not self._inflight_load_reqs
        ):
            self._load_stops_queued = True
            for _ in self._load_threads:
                self._load_queue.put(None)

    def _load_write_context(self) -> Any:
        assert self._plans is not None
        device = self._layer_tensors[self._plans[0].name].device
        if device.type != "cuda":
            return contextlib.nullcontext(None)
        with self._load_lock:
            if self._load_stream is None:
                self._load_stream = torch.cuda.Stream(device=device)
        return torch.cuda.stream(self._load_stream)

    def _record_prior_cuda_work(self) -> tuple[Any | None, str | None]:
        """Capture model-runner CUDA work that must precede cache placement.

        vLLM calls ``start_load_kv`` on its model-runner thread after it has
        enqueued cache-block zeroing and copy-on-write operations. SparkCache
        performs restore writes on background threads and independent CUDA
        streams. Recording an event on the caller's stream gives every queued
        restore an explicit dependency on those earlier writes without making
        the serving thread wait.
        """

        if not self._layer_tensors:
            return None, None
        device = next(iter(self._layer_tensors.values())).device
        if getattr(device, "type", None) != "cuda":
            return None, None
        try:
            event = torch.cuda.Event(blocking=False, interprocess=False)
            event.record(torch.cuda.current_stream(device=device))
        except Exception as error:  # noqa: BLE001
            return (
                None,
                "prior model-runner CUDA work could not be ordered before"
                f" cache placement: {type(error).__name__}: {error}",
            )
        return event, None

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        load_plans = [plan for plan in metadata.plans if not plan.is_store]
        if not load_plans:
            return
        enqueued_ns = time.perf_counter_ns()
        prior_cuda_event, prior_cuda_error = self._record_prior_cuda_work()
        queued = {
            plan.request_id: _QueuedLoad(
                plan=plan,
                timing=RestoreTiming(
                    request_id=plan.request_id,
                    digest=plan.digest,
                    span_tokens=plan.span_tokens,
                    storage_mode=self._storage_mode,
                    enqueued_ns=enqueued_ns,
                ),
                prior_cuda_event=prior_cuda_event,
                prior_cuda_error=prior_cuda_error,
            )
            for plan in load_plans
        }
        with self._load_cv:
            if self._load_stop_requested:
                self._load_errors.update(
                    block for plan in load_plans
                    for group in plan.group_block_ids for block in group
                    if block != 0
                )
                self._finished_load_reqs.update(queued)
                self.counters["load_failed"] += len(queued)
                self._load_cv.notify_all()
                return
            self._inflight_load_reqs.update(queued)
        self._ensure_load_threads()
        if self._storage_mode == "block_pages_v1":
            # Cohort discovery reads the filesystem. Keep that work on the
            # loader queue while the producer event remains the ordering proof
            # captured on the model-runner thread.
            self._load_queue.put(_QueuedLoadBatch(tuple(queued.values())))
        else:
            for item in queued.values():
                self._load_queue.put(item)

    def _prepare_queued_load_batch(self, batch: _QueuedLoadBatch) -> None:
        started_ns = time.perf_counter_ns()
        with self._load_lock:
            cancelled = self._cancelled_load_reqs.intersection(
                item.plan.request_id for item in batch.loads
            )
        for item in batch.loads:
            if item.plan.request_id in cancelled:
                self._load_queue.put(item)
        batch = _QueuedLoadBatch(tuple(
            item for item in batch.loads if item.plan.request_id not in cancelled
        ))
        if not batch.loads:
            return
        queued = {item.plan.request_id: item for item in batch.loads}
        try:
            runnable, deferred, page_base_keys = self._prepare_page_base_read_cohorts(
                [item.plan for item in batch.loads]
            )
        except Exception as error:  # noqa: BLE001 - complete every rejected load
            for item in batch.loads:
                self._page_base_reads.cancel(item.plan.request_id)
                self._load_queue.put(replace(
                    item,
                    prior_cuda_error=f"page-base metadata preparation failed: {error}",
                ))
            return
        elapsed_ns = time.perf_counter_ns() - started_ns
        for item in batch.loads:
            item.timing.observe("metadata_preparation", elapsed_ns)
        with self._load_lock:
            cancelled = self._cancelled_load_reqs.intersection(queued)
            self._page_base_plan_keys.update(page_base_keys)
            self._deferred_page_base_loads.update(
                (plan.request_id, queued[plan.request_id]) for plan in deferred
            )
        for request_id in cancelled:
            self._page_base_reads.cancel(request_id)
        for plan in runnable:
            self._load_queue.put(queued[plan.request_id])
        for key in set(page_base_keys.values()):
            self._release_page_base_deferred(
                key,
                promote_registered=bool(cancelled),
            )

    def _page_base_flight_key(
        self,
        evidence: PageBaseReadEvidence,
    ) -> PageBaseReadFlightKey:
        return PageBaseReadFlightKey(
            worker_generation=self._stats_generation,
            storage_mode=self._storage_mode,
            evidence=evidence,
        )

    def _prepare_page_base_read_cohorts(
        self,
        load_plans: Sequence[_ReqPlan],
    ) -> tuple[
        list[_ReqPlan],
        list[_ReqPlan],
        dict[str, PageBaseReadFlightKey],
    ]:
        """Pre-register bases and defer followers until shared bytes are ready."""

        if (
            self._storage_mode != "block_pages_v1"
            or self._page_layout is None
        ):
            return list(load_plans), [], {}
        identity = self._identity(self._worker_rank())
        grouped: dict[PageBaseReadFlightKey, list[_ReqPlan]] = {}
        for plan in load_plans:
            try:
                lookup, is_alias = self._lookup_reusable(
                    identity,
                    plan.digest,
                    verify_chunks=False,
                    verify_chunk_metadata=True,
                )
                if is_alias or lookup.root_kind != "page_delta":
                    continue
                groups = self._select_group_blocks_for_span(
                    plan.group_block_ids,
                    plan.span_tokens,
                )
                evidence = self._store.page_delta_base_read_evidence(
                    lookup,
                    layout=self._page_layout,
                    result_block_counts=tuple(len(group) for group in groups),
                    result_boundary_tokens=plan.span_tokens,
                )
                if (
                    self._native_restore_enabled
                    and evidence.base_root_kind != "page_snapshot"
                ):
                    # SparkCache CUDA object sharing publishes one verified
                    # flat-base object set. Nested deltas keep their ordinary
                    # independent direct path instead of waiting on a flight
                    # whose representation they cannot consume.
                    continue
                key = self._page_base_flight_key(evidence)
            except (KeyError, OSError, RuntimeError, ValueError):
                continue
            grouped.setdefault(key, []).append(plan)

        registered: set[str] = set()
        priority: list[_ReqPlan] = []
        deferred: list[_ReqPlan] = []
        page_base_keys: dict[str, PageBaseReadFlightKey] = {}
        by_request = {plan.request_id: plan for plan in load_plans}
        for key, plans in grouped.items():
            registration = self._page_base_reads.register_cohort(
                key,
                (plan.request_id for plan in plans),
            )
            member_ids = set(registration.member_ids)
            registered.update(member_ids)
            page_base_keys.update((request_id, key) for request_id in member_ids)
            leader_id = registration.leader_request_id
            if registration.flight_state in {"ready", "error"}:
                priority.extend(plan for plan in plans if plan.request_id in member_ids)
                continue
            if registration.flight_state == "registered" and leader_id in member_ids:
                priority.append(by_request[leader_id])
                member_ids.remove(leader_id)
            deferred.extend(plan for plan in plans if plan.request_id in member_ids)
        independent = [plan for plan in load_plans if plan.request_id not in registered]
        return [*priority, *independent], deferred, page_base_keys

    def _release_page_base_deferred(
        self,
        key: PageBaseReadFlightKey,
        *,
        promote_registered: bool,
    ) -> None:
        """Queue ready followers or one replacement reader without occupying a lane."""

        state = self._page_base_reads.flight_state(key)
        with self._load_lock:
            matching = [
                request_id
                for request_id in self._deferred_page_base_loads
                if self._page_base_plan_keys.get(request_id) == key
            ]
            if state in {None, "ready", "error"}:
                selected = matching
            elif state == "registered" and promote_registered and matching:
                selected = matching[:1]
            else:
                selected = []
            queued = [
                self._deferred_page_base_loads.pop(request_id)
                for request_id in selected
            ]
        for item in queued:
            self._load_queue.put(
                replace(item, prior_cuda_error="page-base cohort is no longer available")
                if state is None else item
            )

    def _release_all_page_base_deferred(self) -> None:
        """Queue shutdown-cancelled followers so loader ownership can drain."""

        with self._load_lock:
            queued = list(self._deferred_page_base_loads.values())
            self._deferred_page_base_loads.clear()
        for item in queued:
            self._load_queue.put(item)

    def _restore_page_base_for_request(
        self,
        request_id: str,
        evidence: PageBaseReadEvidence,
        reader: Callable[[], bytes | bytearray | PageBaseReadResult],
        *,
        allow_independent: bool = True,
    ) -> bytes | PageBaseReadResult | None:
        key = self._page_base_flight_key(evidence)
        try:
            return self._page_base_reads.resolve(
                request_id, key, reader, allow_independent=allow_independent,
            )
        finally:
            self._release_page_base_deferred(
                key,
                promote_registered=False,
            )

    def _emit_page_base_flight_summaries(self) -> None:
        for summary in self._page_base_reads.take_summaries():
            outcome = str(summary["outcome"])
            counter = {
                "verified": "page_base_flights_completed",
                "recompute": "page_base_flights_recomputed",
                "cancelled": "page_base_flights_cancelled",
            }[outcome]
            with self._load_lock:
                self.counters[counter] += 1
                self.counters["page_base_flight_participants"] += int(
                    summary["participants"]
                )
                self.counters["page_base_physical_reads"] += int(
                    summary["physical_base_reads"]
                )
                self.counters["page_base_reads_avoided"] += int(
                    summary["avoided_base_reads"]
                )
            logger.debug(
                "sparkcache: page-base-flight %s",
                json.dumps(summary, sort_keys=True, separators=(",", ":")),
            )

    def _verify_shared_segment_roots(
        self,
        identity: CacheIdentity,
        root_lookup: LookupResult,
        plan: _ReqPlan,
    ) -> bool:
        """Prove each shorter row root is the restored descriptor-graph prefix."""
        if not plan.shared_segments:
            return True
        if self._storage_mode != "per_token_rows" or root_lookup._manifest is None:
            self.counters["restore_segment_roots_rejected"] += 1
            return False
        root_chunks = tuple(root_lookup._manifest.get("chunks", ()))
        for digest, span_tokens in plan.shared_segments:
            segment_lookup, is_alias = self._lookup_reusable(
                identity,
                digest,
                verify_chunks=False,
                verify_chunk_metadata=True,
            )
            expected_count = chunk_count(span_tokens, self._chunk_tokens)
            segment_chunks = (
                tuple(segment_lookup._manifest.get("chunks", ()))
                if segment_lookup._manifest is not None
                else ()
            )
            if (
                not segment_lookup.is_hit
                or span_tokens >= plan.span_tokens
                or len(segment_chunks) != expected_count
                or tuple(root_chunks[:expected_count]) != segment_chunks
            ):
                self.counters["restore_segment_roots_rejected"] += 1
                if segment_lookup.reason == "corrupt" or segment_lookup.is_hit:
                    self._invalidate_reusable(
                        identity,
                        digest,
                        is_alias=is_alias,
                        verify_chunk_payloads=False,
                    )
                    with self._load_lock:
                        self._held.discard(digest)
                _warning(
                    "sparkcache: shared trunk rejected digest=%s leader=%s",
                    digest[:12],
                    plan.digest[:12],
                )
                return False
            self.counters["restore_segment_roots_verified"] += 1
        return True

    def _load_one(
        self,
        plan: _ReqPlan,
        *,
        timing: RestoreTiming | None = None,
        native_lane: int = 0,
    ) -> bool:
        is_alias = False
        try:
            if not valid_metadata(plan):
                return False
            self._scope_salt(plan.request_scope)
            rank = self._worker_rank()
            identity = self._identity(rank)
            lookup_started = time.perf_counter_ns()
            try:
                if self._store.expired(
                    identity,
                    plan.digest,
                    self._capacity_policy.ttl_seconds,
                ):
                    with self._load_lock:
                        self._held.discard(plan.digest)
                    self._capacity_wakeup.set()
                    logger.debug(
                        "sparkcache: worker rank %d TTL miss for %s",
                        rank,
                        plan.digest[:12],
                    )
                    return False
                # Restore is the integrity boundary and re-hashes every chunk in
                # parallel. A full lookup verification here would read/hash/decode
                # the complete context twice before installation.
                lookup, is_alias = self._lookup_reusable(
                    identity,
                    plan.digest,
                    verify_chunks=False,
                )
            finally:
                if timing is not None:
                    timing.observe(
                        "manifest_lookup",
                        time.perf_counter_ns() - lookup_started,
                    )
            if not lookup.is_hit:
                _warning(
                    "sparkcache: worker rank %d miss (%s) for %s",
                    rank,
                    lookup.reason,
                    plan.digest[:12],
                )
                if lookup.reason == "corrupt":
                    self._invalidate_after_failure(
                        plan.digest,
                        is_alias=is_alias,
                    )
                return False
            if is_alias:
                self.counters["prefix_alias_restore_hit"] += 1
            if not self._verify_shared_segment_roots(identity, lookup, plan):
                return False
            if self._storage_mode == "block_pages_v1":
                return self._load_hybrid_pages(
                    lookup,
                    plan,
                    timing=timing,
                    native_lane=native_lane,
                )
            if self._native_restore_enabled:
                if (
                    self._native_adapter is None
                    or not callable(self._native_execute_restore)
                    or self._native_required_record_mask == 0
                ):
                    raise RuntimeError(
                        "SparkCache CUDA restore selected without a configured adapter"
                    )
                positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
                slots = local_slots_for_positions(
                    positions,
                    plan.block_ids,
                    self._block_size,
                    self._dcp_degree,
                )
                try:
                    result = self._native_execute_restore(
                        adapter=self._native_adapter,
                        request_id=plan.request_id,
                        lookup=lookup,
                        cache_root=self._root,
                        slots=slots,
                        expected_span_tokens=plan.span_tokens,
                        dcp_degree=self._dcp_degree,
                        dcp_rank=rank,
                        arena_bytes=self._native_arena_bytes,
                        required_data_record_mask=(self._native_required_record_mask),
                        io_workers=self._native_io_workers,
                    )
                except Exception as error:  # noqa: BLE001
                    # The SparkCache CUDA transaction may already have written some
                    # private restore blocks. Never enter Python assembly or
                    # retry those blocks: retire the entry and publish all of
                    # this request's blocks as invalid for clean recompute.
                    _warning(
                        "sparkcache: SparkCache CUDA restore rejected;"
                        " recomputing: %s",
                        error,
                    )
                    self._invalidate_after_failure(
                        plan.digest,
                        is_alias=is_alias,
                    )
                    return False
                if timing is not None:
                    timing.observe(
                        "restore_read",
                        int(result.read_and_hash_ms * 1_000_000),
                    )
                    timing.observe(
                        "h2d_submit",
                        int(result.parse_and_submit_ms * 1_000_000),
                    )
                    timing.observe(
                        "cuda_sync",
                        int(result.finish_ms * 1_000_000),
                    )
                self.counters["native_load_verified"] = (
                    self.counters.get("native_load_verified", 0) + 1
                )
                self.counters["native_chunks_verified"] = self.counters.get(
                    "native_chunks_verified", 0
                ) + int(result.verified_chunks)
                _debug_log(
                    "sparkcache: SparkCache CUDA restore verified %d chunks"
                    " (%d encoded bytes, %d slabs) read_hash=%.1f ms"
                    " parse_submit=%.1f ms finish=%.1f ms",
                    result.verified_chunks,
                    result.verified_encoded_bytes,
                    result.slabs,
                    result.read_and_hash_ms,
                    result.parse_and_submit_ms,
                    result.finish_ms,
                )
                return True
            restore_started = time.perf_counter_ns()
            chunks = self._store.restore(lookup)
            if timing is not None:
                timing.observe(
                    "restore_read",
                    time.perf_counter_ns() - restore_started,
                )
            if chunks is None or len(chunks) != chunk_count(
                plan.span_tokens, self._chunk_tokens
            ):
                self._invalidate_after_failure(
                    plan.digest,
                    is_alias=is_alias,
                )
                return False
            reassembly_started = time.perf_counter_ns()
            positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
            slots = local_slots_for_positions(
                positions, plan.block_ids, self._block_size, self._dcp_degree
            )
            per_chunk = self._chunk_tokens // self._dcp_degree
            assert self._plans is not None
            layer_rows: dict[str, list[bytes]] = {p.name: [] for p in self._plans}
            for index, chunk in enumerate(chunks):
                expected = positions[index * per_chunk : (index + 1) * per_chunk]
                stored = unpack_positions(chunk.records[StateRecord.LOGICAL_POSITIONS])
                if stored != tuple(expected):
                    raise CodecError("stored positions disagree with shard")
                for kind in self._record_kinds:
                    split = unpack_record(
                        self._plans,
                        kind,
                        chunk.records[StateRecord(kind)],
                        per_chunk,
                    )
                    for name, payload in split.items():
                        layer_rows[name].append(payload)
            if timing is not None:
                timing.chunk_count = len(chunks)
                timing.observe(
                    "reassembly_decode",
                    time.perf_counter_ns() - reassembly_started,
                )
            slot_tensor = torch.tensor(slots, dtype=torch.long)
            submit_started = time.perf_counter_ns()
            with self._load_write_context():
                for plan_entry in self._plans:
                    tensor = self._layer_tensors[plan_entry.name]
                    rows = self._rows_view(tensor)
                    payload = b"".join(layer_rows[plan_entry.name])
                    flat = torch.frombuffer(
                        bytearray(payload), dtype=torch.uint8
                    ).reshape(len(slots), plan_entry.bytes_per_token)
                    staged = flat.to(rows.device)
                    rows_u8 = rows.view(torch.uint8)
                    rows_u8[slot_tensor.to(rows.device)] = staged
            if timing is not None:
                timing.observe(
                    "h2d_submit",
                    time.perf_counter_ns() - submit_started,
                )
            if self._load_stream is not None:
                sync_started = time.perf_counter_ns()
                self._load_stream.synchronize()
                if timing is not None:
                    timing.observe(
                        "cuda_sync",
                        time.perf_counter_ns() - sync_started,
                    )
            return True
        except (CodecError, KeyError, RuntimeError, ValueError) as error:
            _warning(
                "sparkcache: restore rejected; recomputing reason=%s",
                error,
            )
            self._invalidate_after_failure(
                plan.digest,
                is_alias=is_alias,
            )
            return False

    def _load_hybrid_pages(
        self,
        lookup: Any,
        plan: _ReqPlan,
        *,
        timing: RestoreTiming | None = None,
        native_lane: int = 0,
    ) -> bool:
        layout = self._page_layout
        if layout is None:
            raise RuntimeError("block-page layout was not registered")
        groups = self._select_group_blocks_for_span(
            plan.group_block_ids, plan.span_tokens
        )
        if len(groups) != len(layout.groups):
            raise HybridCodecError("request block tables disagree with page groups")
        if self._native_restore_enabled:
            if not self._native_adapters or not callable(
                self._native_execute_hybrid_restore
            ):
                raise RuntimeError(
                    "SparkCache CUDA restore selected without a configured adapter"
                )
            if not 0 <= native_lane < len(self._native_adapters):
                raise RuntimeError("SparkCache CUDA placement lane is unavailable")
            try:
                result = self._native_execute_hybrid_restore(
                    adapter=self._native_adapters[native_lane],
                    request_id=plan.request_id,
                    lookup=lookup,
                    cache_root=self._root,
                    layout=layout,
                    group_slots=groups,
                    expected_span_tokens=plan.span_tokens,
                    arena_bytes=self._native_arena_bytes,
                    io_workers=self._native_io_workers,
                    dcp_degree=self._dcp_degree,
                    dcp_rank=self._worker_rank(),
                    base_reader=lambda evidence, reader: (
                        self._restore_page_base_for_request(
                            plan.request_id,
                            evidence,
                            reader,
                            allow_independent=False,
                        )
                    ),
                )
            except Exception as error:  # noqa: BLE001
                _warning(
                    "sparkcache: SparkCache CUDA placement rejected: %s",
                    error,
                )
                self._invalidate_after_failure(plan.digest)
                return False
            if timing is not None:
                timing.page_bytes = result.source_bytes
                timing.observe(
                    "restore_read",
                    int(result.read_and_hash_ms * 1_000_000),
                )
                timing.observe("reassembly_decode", 0)
                timing.observe(
                    "h2d_submit",
                    int(result.copy_and_submit_ms * 1_000_000),
                )
                timing.observe(
                    "cuda_sync",
                    int(result.finish_ms * 1_000_000),
                )
            self.counters["native_hybrid_load_verified"] = (
                self.counters.get("native_hybrid_load_verified", 0) + 1
            )
            if lookup.root_kind == "page_delta":
                self.counters["native_page_delta_load_verified"] = (
                    self.counters.get("native_page_delta_load_verified", 0) + 1
                )
                self.counters["native_page_delta_base_bytes_skipped"] = (
                    self.counters.get(
                        "native_page_delta_base_bytes_skipped",
                        0,
                    )
                    + int(getattr(result, "skipped_base_object_bytes", 0))
                )
            if getattr(result, "hot_snapshot_hit", False):
                self.counters["native_hot_snapshot_hits"] = (
                    self.counters.get("native_hot_snapshot_hits", 0) + 1
                )
            _debug_log(
                "sparkcache: SparkCache CUDA restore verified %d bytes"
                " read_source_bytes=%d skipped_base_bytes=%d"
                " slabs=%d read_hash=%.1f ms (planning=%.1f ms)"
                " final_hash=%.1f ms placement=%.1f ms"
                " (arena_wait=%.1f ms host_copy=%.1f ms submit_call=%.1f ms)"
                " finish=%.1f ms hot_snapshot=%d",
                result.source_bytes,
                int(getattr(result, "read_source_bytes", 0)),
                int(getattr(result, "skipped_base_object_bytes", 0)),
                result.slabs,
                result.read_and_hash_ms,
                float(getattr(result, "planning_ms", 0.0)),
                float(getattr(result, "final_hash_ms", 0.0)),
                result.copy_and_submit_ms,
                result.arena_wait_ms,
                result.host_copy_ms,
                result.submit_call_ms,
                result.finish_ms,
                int(bool(getattr(result, "hot_snapshot_hit", False))),
            )
            return True
        restore_started = time.perf_counter_ns()
        if lookup.root_kind in ("page_delta", "page_snapshot"):
            encoded_pages = self._store.restore_page_snapshot(
                lookup,
                layout=layout,
                result_block_counts=tuple(len(group) for group in groups),
                result_boundary_tokens=plan.span_tokens,
                base_reader=lambda evidence, reader: (
                    self._restore_page_base_for_request(
                        plan.request_id,
                        evidence,
                        reader,
                    )
                ),
            )
            restored_chunk_count = chunk_count(plan.span_tokens, self._chunk_tokens)
        else:
            chunks = self._store.restore(lookup)
            expected_chunks = chunk_count(plan.span_tokens, self._chunk_tokens)
            if chunks is None or len(chunks) != expected_chunks:
                raise HybridCodecError("hybrid snapshot chunk count differs")
            encoded_parts = []
            for index, chunk in enumerate(chunks):
                start = index * self._chunk_tokens
                end = (index + 1) * self._chunk_tokens
                stored = unpack_positions(chunk.records[StateRecord.LOGICAL_POSITIONS])
                if stored != tuple(range(start, end)):
                    raise HybridCodecError("hybrid snapshot positions differ")
                encoded_parts.append(chunk.records[StateRecord.TARGET_CKV])
            encoded_pages = b"".join(encoded_parts)
            restored_chunk_count = len(chunks)
        if timing is not None:
            timing.observe(
                "restore_read",
                time.perf_counter_ns() - restore_started,
            )
        reassembly_started = time.perf_counter_ns()
        page_plan = plan_page_snapshot(
            layout,
            encoded_pages,
            tuple(len(group) for group in groups),
        )
        if timing is not None:
            timing.chunk_count = restored_chunk_count
            timing.page_bytes = len(encoded_pages)
            timing.observe(
                "reassembly_decode",
                time.perf_counter_ns() - reassembly_started,
            )
        if self._native_restore_enabled:
            if not self._native_adapters or not callable(
                self._native_execute_hybrid_placement
            ):
                raise RuntimeError(
                    "SparkCache CUDA restore selected without a configured adapter"
                )
            if not 0 <= native_lane < len(self._native_adapters):
                raise RuntimeError("SparkCache CUDA placement lane is unavailable")
            try:
                result = self._native_execute_hybrid_placement(
                    adapter=self._native_adapters[native_lane],
                    request_id=plan.request_id,
                    encoded_pages=encoded_pages,
                    plan=page_plan,
                    group_slots=groups,
                )
            except Exception as error:  # noqa: BLE001
                _warning(
                    "sparkcache: SparkCache CUDA placement rejected: %s",
                    error,
                )
                self._invalidate_after_failure(plan.digest)
                return False
            if timing is not None:
                timing.observe(
                    "h2d_submit",
                    int(result.copy_and_submit_ms * 1_000_000),
                )
                timing.observe(
                    "cuda_sync",
                    int(result.finish_ms * 1_000_000),
                )
            self.counters["native_hybrid_load_verified"] = (
                self.counters.get("native_hybrid_load_verified", 0) + 1
            )
            logger.debug(
                "sparkcache: SparkCache CUDA placement verified %d bytes"
                " submit=%.1f ms finish=%.1f ms",
                result.source_bytes,
                result.copy_and_submit_ms,
                result.finish_ms,
            )
            return True
        payloads = decode_page_snapshot(
            layout,
            encoded_pages,
            tuple(len(group) for group in groups),
        )
        submit_started = time.perf_counter_ns()
        with self._load_write_context():
            for page_group, block_ids in zip(layout.groups, groups):
                if not block_ids:
                    # Non-persisted groups published no pages. Leave their
                    # freshly allocated manager blocks untouched: vLLM
                    # rebuilds this state at the persistent boundary.
                    continue
                for layer in page_group.layers:
                    tensor = self._layer_tensors[layer.name]
                    source = torch.frombuffer(
                        bytearray(payloads[layer.name]), dtype=tensor.dtype
                    ).reshape((len(block_ids), *layer.page_shape))
                    index_tensor = torch.tensor(
                        block_ids, dtype=torch.long, device=tensor.device
                    )
                    tensor[index_tensor] = source.to(tensor.device)
        if timing is not None:
            timing.observe(
                "h2d_submit",
                time.perf_counter_ns() - submit_started,
            )
        if self._load_stream is not None:
            sync_started = time.perf_counter_ns()
            self._load_stream.synchronize()
            if timing is not None:
                timing.observe(
                    "cuda_sync",
                    time.perf_counter_ns() - sync_started,
                )
        return True

    def _invalidate_after_failure(
        self,
        digest: str,
        *,
        is_alias: bool = False,
    ) -> None:
        """Withdraw a failed entry without re-reading payloads inline.

        The current request already fell back to recompute. A later publisher
        can atomically repair a corrupt content-addressed chunk while
        republishing the manifest; full payload inspection remains the
        explicit integrity sweep's responsibility.
        """
        with self._load_lock:
            self._held.discard(digest)
        removed = self._invalidate_reusable(
            self._identity(self._worker_rank()),
            digest,
            is_alias=is_alias,
            verify_chunk_payloads=False,
        )
        if removed:
            with self._load_lock:
                self.counters["invalidated"] = self.counters.get("invalidated", 0) + 1
            _warning(
                "sparkcache: invalidated damaged entry %s",
                digest[:12],
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self.record_request_cache_event(request, "finished")
        # A finished request id never recurs, so its scheduler-side tracking
        # state is dropped here. This is what keeps _need_load, _admitted,
        # and _store_progress bounded without evicting live entries.
        request_id = request.request_id
        if request_id in getattr(self, "_inflight_load_reqs", ()):
            with self._load_lock:
                self._cancelled_load_reqs.add(request_id)
        self._page_base_reads.cancel(request_id)
        self._emit_page_base_flight_summaries()
        follower_digest = self._restore_flight_followers.get(request_id)
        if follower_digest is not None:
            self._remove_restore_follower(request_id)
        leader_digest = self._restore_flight_leaders.get(request_id)
        if leader_digest is not None:
            flight = self._restore_flights.get(leader_digest)
            if flight is not None:
                if flight.dispatched:
                    status = getattr(request, "status", None)
                    status_name = getattr(status, "name", str(status or ""))
                    completed_normally = status_name in {
                        "FINISHED_STOPPED",
                        "FINISHED_LENGTH_CAPPED",
                        "FINISHED_REPETITION",
                    }
                    if flight.lease_published:
                        # The scheduler-owned lease, not the leader request,
                        # now retains the verified table. Keep the digest hot
                        # for late arrivals until its bounded expiry.
                        self._restore_flight_leaders.pop(request_id, None)
                        flight.leader_finished = True
                    elif flight.workers_finished and completed_normally:
                        # The leader could only generate after vLLM published
                        # its verified external blocks. Retiring here also
                        # wakes an otherwise all-deferred follower cohort.
                        self._retire_restore_flight(leader_digest, outcome="completed")
                    elif flight.workers_finished:
                        # The worker writes have already drained, so an abort
                        # needs no later completion edge to release followers.
                        self.counters["restore_flight_leader_aborted"] += 1
                        self._retire_restore_flight(leader_digest, outcome="cancelled")
                    else:
                        # Worker completion owns the drain edge. Keep aborted
                        # or errored leaders reserved until that signal arrives
                        # so no second restore can overlap outstanding writes.
                        flight.leader_finished = True
                        self.counters["restore_flight_leader_aborted"] += 1
                else:
                    self._retire_restore_flight(leader_digest, outcome="cancelled")
        self._need_load.pop(request_id, None)
        # Test adapters and older connector shims may not initialize the
        # optional scheduler-side digest cache. Request completion must remain
        # idempotent for those callers.
        getattr(self, "_prefix_digest_candidates", {}).pop(request_id, None)
        getattr(self, "_request_cache_scopes", {}).pop(request_id, None)
        self._pending_async_loads.pop(request_id, None)
        self._admitted.pop(request_id, None)
        self._store_progress.pop(request_id, None)
        self._store_token_ids.pop(request_id, None)
        store_multimodal = getattr(self, "_store_multimodal", None)
        if store_multimodal is not None:
            store_multimodal.discard(request_id)
        self._store_bases.pop(request_id, None)
        self._store_recurrent_boundaries.pop(request_id, None)
        runtime = self._streaming_runtime
        if runtime is None:
            return False, None
        delay_free = bool(runtime.request_finished(request_id, tuple(block_ids)))
        return delay_free, None

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Release scheduler bookkeeping for a hybrid-memory-allocator request.

        Synchronous stores do not retain request blocks after the CPU snapshot.
        Explicit asynchronous manager-page capture delays all-group cleanup
        until every worker reports its native read complete. Row-oriented
        streaming snapshots do not support multiple KV-cache groups.
        """
        if self._uses_capture_job_leases():
            # Per-job references survive ordinary request cleanup and are
            # released by read-completion metadata, independent of disk commit.
            return self.request_finished(request, [])
        if self._async_page_capture_enabled:
            request_id = request.request_id
            cleanup_delay, _ = self.request_finished(request, [])
            if cleanup_delay:
                raise RuntimeError(
                    "sparkcache: manager-page cleanup encountered"
                    " an incompatible row-streaming lease"
                )
            eligible = request_id in self._async_page_capture_eligible
            if eligible:
                self._async_page_capture_eligible.discard(request_id)
                return True, None
            return False, None
        if self._streaming_runtime is not None:
            raise RuntimeError(
                "sparkcache: streaming snapshots do not support"
                " multiple KV-cache groups"
            )
        return self.request_finished(request, [])

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        runtime = self._streaming_runtime
        if runtime is not None:
            finished_sending.update(runtime.take_finished(finished_req_ids))
        runtime = self._async_page_capture_runtime
        if runtime is not None:
            finished_sending.update(runtime.take_finished(finished_req_ids))
        with self._load_cv:
            finished_recving = set(self._finished_load_reqs)
            self._finished_load_reqs.clear()
        return finished_sending or None, finished_recving or None

    def get_finished_count(self) -> int:
        """Require terminal store acknowledgement from every physical rank.

        A headless multi-node executor can report a local world size of one to
        vLLM's completion aggregator even though every tensor-parallel rank
        owns request pages. Releasing scheduler blocks after the first rank's
        acknowledgement can strand later worker completions. SparkCache uses
        the same physical-rank count here as its restore-admission quorum.
        """

        return self._tp_degree

    def wait_for_pending_loads(self, timeout: float | None = None) -> bool:
        with self._load_cv:
            return self._load_cv.wait_for(lambda: not self._inflight_load_reqs, timeout)

    def shutdown(self):
        with self._load_cv:
            self._load_stop_requested = True
        self._page_base_reads.close()
        self._release_all_page_base_deferred()
        self._emit_page_base_flight_summaries()
        runtime = self._streaming_runtime
        if runtime is not None:
            # Drain/cancel snapshot leases before any C++/CUDA cache or staging
            # resource can be destroyed. A failure remains fatal and prevents
            # unsafe teardown.
            runtime.shutdown()
            self._streaming_runtime = None
        runtime = self._async_page_capture_runtime
        if runtime is not None:
            runtime.quiesce()
        capacity_thread = self._capacity_thread
        if capacity_thread is not None:
            # Give already handed-off durable commits one final background
            # capacity pass. A persistent busy/failure remains unadvertised:
            # shutdown drops only the in-memory handoff, never advertises it.
            self._capacity_wakeup.set()
            self.wait_for_pending_capacity_commits(timeout=5.0)
        # Serialize the terminal stop edge with the final survivor check and
        # _held publication. Once this section completes, a capacity worker
        # can only leave an unresolved handoff unadvertised.
        with self._store_cv:
            self._capacity_stop.set()
        self._capacity_wakeup.set()
        if capacity_thread is not None:
            capacity_thread.join(timeout=5.0)
            if not capacity_thread.is_alive():
                self._capacity_thread = None
        with self._capacity_handoff_cv:
            dropped = len(self._streaming_capacity_pending)
            self._streaming_capacity_pending.clear()
            self._capacity_handoff_cv.notify_all()
        if dropped:
            self.counters["streaming_capacity_shutdown_dropped"] += dropped
        with self._store_cv:
            self._store_accepting = False
        store_idle = self.wait_for_pending_stores(timeout=5.0)
        store_thread = self._store_thread
        if store_thread is not None:
            self._store_queue.put(None)
            store_thread.join(timeout=5.0 if store_idle else 0.0)
            if store_thread.is_alive():
                self.counters["store_shutdown_thread_live"] = (
                    self.counters.get("store_shutdown_thread_live", 0) + 1
                )
                _warning(
                    "sparkcache: shutdown left saver alive;"
                    " process teardown will reclaim its CPU snapshot"
                )
            else:
                self._store_thread = None
        runtime = self._async_page_capture_runtime
        if runtime is not None:
            if runtime.shutdown():
                self._async_page_capture_runtime = None
            else:
                self.counters["async_page_capture_shutdown_ring_retained"] = (
                    self.counters.get(
                        "async_page_capture_shutdown_ring_retained", 0
                    )
                    + 1
                )
                _warning(
                    "sparkcache: shutdown retained manager-page"
                    " capture ring for a live durable writer"
                )
        with self._load_cv:
            self._load_stop_requested = True
            self._stop_idle_loaders_locked()
        self.wait_for_pending_loads(timeout=5.0)
        deadline = time.monotonic() + 5.0
        for thread in self._load_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        live_threads = sum(thread.is_alive() for thread in self._load_threads)
        if (self._native_adapters or self._native_adapter is not None) and live_threads:
            # A SparkCache CUDA loader can still own the handle or mapped arenas.
            # Keep the adapter strongly reachable and let process teardown
            # reclaim it; destroying it here would be a use-after-close race.
            self.counters["native_shutdown_handle_leaked"] = (
                self.counters.get("native_shutdown_handle_leaked", 0) + 1
            )
            _warning(
                "sparkcache: shutdown left %d loader(s) alive;"
                " retaining SparkCache CUDA placement handle until process exit",
                live_threads,
            )
            return None
        adapters = self._native_adapters or (
            [self._native_adapter] if self._native_adapter is not None else []
        )
        for adapter in adapters:
            adapter.close()
        self._native_adapters = []
        self._native_adapter = None
        return None

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: Any,
        **kwargs: Any,
    ) -> None:
        return

    def _protect_capture_publication_base(self, plan: _ReqPlan) -> _ReqPlan:
        """Reserve an offered base or capture complete state without waiting.

        The single inflight saver admission bounds retention to one base and
        its result per rank. Maintenance shares the capacity lock, so it cannot select that
        root between the inventory check and registration. This path performs
        no filesystem operations on the model-runner thread.
        """
        if not plan.base_context_digest:
            return plan
        interval = getattr(self, "_page_snapshot_interval_tokens", 0)
        if interval and plan.span_tokens // interval > plan.base_span_tokens // interval:
            settings = getattr(self, "_async_page_capture_settings", None)
            layout = self._page_layout
            # Native capture stores only the raw page body in its slot. This
            # exact size check uses registered geometry, without GPU or disk
            # work, and rejects every oversized full capture before submit.
            full_bytes = (
                sum(count * sum(layer.bytes_per_page for layer in group.layers)
                    for count, group in zip(
                        self._group_block_counts_for_span(plan.span_tokens),
                        layout.groups, strict=True))
                if settings is not None and layout is not None else 0
            )
            if settings is None or full_bytes <= settings.slot_bytes:
                self.counters["publication_periodic_full_capture_selected"] = (
                    self.counters.get("publication_periodic_full_capture_selected", 0) + 1
                )
                return replace(plan, base_context_digest="", base_span_tokens=0)
            self.counters["publication_full_capture_slot_fallback"] = (
                self.counters.get("publication_full_capture_slot_fallback", 0) + 1
            )
        if self._capacity_lock.acquire(blocking=False):
            try:
                with self._store_cv:
                    if (
                        self._capacity_status["capacity_satisfied"]
                        and plan.base_context_digest in self._held
                    ):
                        self._publication_base_pins[plan.digest] = EntryKey(
                            self._identity(self._worker_rank()).storage_key,
                            plan.base_context_digest,
                        )
                        return plan
            finally:
                self._capacity_lock.release()
        self.counters["publication_base_full_capture_fallback"] = (
            self.counters.get("publication_base_full_capture_fallback", 0) + 1
        )
        return replace(plan, base_context_digest="", base_span_tokens=0)

    def _release_publication_base_pin(self, digest: str) -> None:
        with self._store_cv:
            removed = getattr(self, "_publication_base_pins", {}).pop(digest, None)
            retry = removed is not None and not self._capacity_status["capacity_satisfied"]
        if retry:
            self._capacity_wakeup.set()

    def wait_for_save(self) -> None:
        # Enforce the publication policy independently on every worker. The
        # scheduler normally omits store plans when publication is disabled,
        # but a stale or mismatched metadata object must not make a
        # restore-only worker capture or publish model state.
        if not self._store_enabled:
            return
        metadata = self._get_connector_metadata()
        if self._streaming_snapshots_enabled:
            # Scheduler offers are promises for the step vLLM has just
            # forwarded.  Convert them to completed watermarks only here;
            # doing it in build_connector_meta would let a writer read KV
            # rows before the producer stream has populated them.
            if self._role is KVConnectorRole.WORKER:
                runtime = self._streaming_runtime
                if runtime is None:
                    raise RuntimeError(
                        "sparkcache: streaming worker runtime vanished"
                    )
                if isinstance(metadata, SparkCacheConnectorMetadata):
                    offers = metadata.streaming_snapshot_offers
                    if offers:
                        producer_stream = int(torch.cuda.current_stream().cuda_stream)
                    for offer in offers:
                        if not valid_metadata(offer):
                            continue
                        runtime.offer_completed(
                            offer,
                            producer_stream=producer_stream,
                        )
                # Poll even for an empty metadata object so background
                # completion, backpressure, and failure state advance on
                # every post-forward callback.
                runtime.poll()
            # Never invoke the synchronous CPU snapshot/ManifestStore path under
            # this flag: an incomplete injected runtime must publish nothing, not
            # silently publish a synchronous end-of-prefill entry.
            return
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        for plan in metadata.plans:
            if not plan.is_store:
                continue
            # Admission is atomic and precedes the multi-GB CPU snapshot.
            # At most one snapshot may be queued or committing per rank; a
            # busy store is a cache miss opportunity, never a serving stall.
            skipped_before_submit = False
            with self._store_cv:
                if not valid_metadata(plan):
                    self.counters["store_skipped_invalid_scope"] = (
                        self.counters.get("store_skipped_invalid_scope", 0) + 1
                    )
                    skipped_before_submit = True
                elif plan.digest in self._held:
                    # A peer's failed restore withdraws the scheduler's whole
                    # quorum. Recompute must renew even unchanged local offers.
                    self._stats_readvertise_held.add(plan.digest)
                    self.counters["store_skipped_present"] += 1
                    logger.debug(
                        "sparkcache: store skipped; entry already"
                        " present digest=%s",
                        plan.digest[:12],
                    )
                    skipped_before_submit = True
                elif not self._store_accepting or self._store_inflight:
                    self.counters["store_skipped_busy"] += 1
                    logger.debug(
                        "sparkcache: store skipped while saver busy digest=%s",
                        plan.digest[:12],
                    )
                    skipped_before_submit = True
                else:
                    self._store_inflight = 1
                    self._store_pending_started_ns = time.perf_counter_ns()
                    self._store_pending_tokens = plan.span_tokens
            if skipped_before_submit:
                reason = ("invalid-scope" if not valid_metadata(plan) else
                          "present" if plan.digest in self._held else "busy")
                (logger.debug if reason == "present" else logger.info)(
                    "sparkcache: save skipped tokens=%d rank=%d reason=%s",
                    plan.span_tokens, self._physical_rank(), reason)
                if plan.capture_job_id:
                    self._capture_read_completed(plan.capture_job_id)
                    continue
                if self._async_page_capture_enabled:
                    runtime = self._async_page_capture_runtime
                    if runtime is None:
                        raise RuntimeError(
                            "manager-page capture runtime vanished before"
                            " a skipped store completed"
                        )
                    runtime.finish_without_capture(
                        plan.request_id,
                        retained_manager_pages=sum(
                            len(group) for group in plan.group_block_ids
                        ),
                    )
                continue
            if self._async_page_capture_enabled:
                runtime = self._async_page_capture_runtime
                if runtime is None:
                    if plan.capture_job_id:
                        self._capture_read_completed(plan.capture_job_id)
                    self._finish_store(
                        plan.digest,
                        committed=False,
                        error=RuntimeError(
                            "manager-page capture runtime is unavailable"
                        ),
                    )
                    continue
                try:
                    stream = torch.cuda.current_stream()
                    producer_stream = int(stream.cuda_stream)
                    if plan.capture_job_id:
                        # post_forward follows target and MTP state writes.
                        # The background submitter waits on this event before
                        # launching reads against connector-owned source pages.
                        ready = torch.cuda.Event()
                        ready.record(stream)
                        runtime.submit(
                            plan,
                            producer_stream=producer_stream,
                            producer_ready=ready,
                        )
                    else:
                        plan = self._protect_capture_publication_base(plan)
                        runtime.submit(plan, producer_stream=producer_stream)
                except Exception as error:  # noqa: BLE001 - serving continues
                    runtime.preempt(plan.request_id)
                    if plan.capture_job_id:
                        runtime.finish_failed_job(plan)
                    self._finish_store(
                        plan.digest,
                        committed=False,
                        error=error,
                    )
                continue
            snapshot_started = time.perf_counter()
            try:
                snapshot = self._snapshot_store(plan)
            except Exception as error:  # noqa: BLE001 - never kill serving
                self._finish_store(
                    plan.digest,
                    committed=False,
                    error=error,
                )
                continue
            try:
                self._ensure_store_thread()
                self._store_queue.put(snapshot)
                logger.debug(
                    "sparkcache: rank %d snapshotted %d tokens"
                    " digest=%s in %.1f ms; queued background commit",
                    snapshot.rank,
                    plan.span_tokens,
                    plan.digest[:12],
                    1e3 * (time.perf_counter() - snapshot_started),
                )
            except Exception as error:  # noqa: BLE001 - failed enqueue publishes nothing
                self._finish_store(
                    plan.digest,
                    committed=False,
                    error=error,
                )

    def _snapshot_store(self, plan: _ReqPlan) -> _StoreSnapshot | _HybridStoreSnapshot:
        """Synchronously detach source KV blocks into owned CPU bytes."""
        if not valid_metadata(plan):
            raise ValueError("Missing or inconsistent request cache scope metadata")
        self._scope_salt(plan.request_scope)

        if self._storage_mode == "block_pages_v1":
            return self._snapshot_hybrid_store(plan)

        rank = self._worker_rank()
        identity = self._identity(rank)
        logical_start = plan.base_span_tokens if plan.base_context_digest else 0
        if logical_start % self._chunk_tokens:
            raise RuntimeError("tail publication base is not chunk-aligned")
        positions = tuple(
            position
            for position in owned_positions(plan.span_tokens, self._dcp_degree, rank)
            if position >= logical_start
        )
        slots = local_slots_for_positions(
            positions, plan.block_ids, self._block_size, self._dcp_degree
        )
        assert self._plans is not None
        slot_tensor = torch.tensor(slots, dtype=torch.long)
        layer_bytes: dict[str, bytes] = {}
        for plan_entry in self._plans:
            tensor = self._layer_tensors[plan_entry.name]
            rows = self._rows_view(tensor).view(torch.uint8)
            gathered = rows[slot_tensor.to(rows.device)].cpu().contiguous()
            layer_bytes[plan_entry.name] = gathered.numpy().tobytes()
        return _StoreSnapshot(
            plan=plan,
            rank=rank,
            identity=identity,
            positions=positions,
            layer_bytes=layer_bytes,
            layer_plans=tuple(self._plans),
            record_kinds=self._record_kinds,
            logical_start=logical_start,
        )

    def _snapshot_hybrid_store(self, plan: _ReqPlan) -> _HybridStoreSnapshot:
        layout = self._page_layout
        if layout is None:
            raise RuntimeError("block-page layout was not registered")
        groups = self._select_group_blocks_for_span(
            plan.group_block_ids,
            plan.span_tokens,
            recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
        )
        if len(groups) != len(layout.groups):
            raise HybridCodecError("request block tables disagree with page groups")
        layer_payloads: dict[str, bytes] = {}
        for page_group, block_ids in zip(layout.groups, groups):
            index = torch.tensor(block_ids, dtype=torch.long)
            for layer in page_group.layers:
                tensor = self._layer_tensors[layer.name]
                gathered = tensor[index.to(tensor.device)].cpu().contiguous()
                layer_payloads[layer.name] = (
                    gathered.view(torch.uint8).numpy().tobytes()
                )
        encoded = encode_page_snapshot(
            layout,
            tuple(len(group) for group in groups),
            layer_payloads,
        )
        rank = self._worker_rank()
        return _HybridStoreSnapshot(
            plan=plan,
            rank=rank,
            identity=self._identity(rank),
            positions=tuple(range(plan.span_tokens)),
            encoded_pages=encoded,
            block_counts=tuple(len(group) for group in groups),
        )

    def _complete_async_page_capture(
        self,
        plan: _ReqPlan,
        encoded_pages: Any,
        block_counts: tuple[int, ...],
    ) -> None:
        """Validate completed native bytes and queue the existing commit path."""

        layout = self._page_layout
        if layout is None:
            self._abort_async_page_capture(
                plan.digest, "block-page layout is not registered"
            )
            return
        try:
            expected_groups = self._select_group_blocks_for_span(
                plan.group_block_ids,
                plan.span_tokens,
                recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
            )
            expected_counts = tuple(len(group) for group in expected_groups)
            if block_counts != expected_counts:
                raise HybridCodecError(
                    "native capture block counts disagree with the store plan"
                )
            reused_pages = getattr(
                encoded_pages, "reused_pages_by_group", None
            )
            if reused_pages is None:
                header = getattr(encoded_pages, "header_bytes", encoded_pages)
                total_bytes = getattr(encoded_pages, "total_bytes", None)
                plan_page_snapshot(
                    layout,
                    header,
                    block_counts,
                    total_bytes=total_bytes,
                )
            else:
                from sparkcache.streaming.manager_page_capture import (
                    select_manager_page_extension_pages,
                )

                if not plan.base_context_digest or plan.base_span_tokens <= 0:
                    raise HybridCodecError(
                        "native page extension has no authenticated base"
                    )
                _, _, selected_counts, expected_reused = (
                    select_manager_page_extension_pages(
                        expected_groups,
                        base_page_counts=self._group_block_counts_for_span(
                            plan.base_span_tokens
                        ),
                        logical_tokens_per_page=tuple(
                            int(group["logical_tokens_per_block"])
                            for group in self._group_topology
                        ),
                        reuse_policies=tuple(
                            str(group["reuse_policy"])
                            for group in self._group_topology
                        ),
                        base_boundary_tokens=plan.base_span_tokens,
                    )
                )
                if (
                    tuple(reused_pages) != expected_reused
                    or selected_counts != expected_counts
                    or tuple(
                        getattr(encoded_pages, "result_block_counts", ())
                    )
                    != expected_counts
                ):
                    raise HybridCodecError(
                        "native page extension geometry disagrees with the store plan"
                    )
                expected_bytes = sum(
                    (count - reused) * layer.bytes_per_page
                    for group, count, reused in zip(
                        layout.groups,
                        expected_counts,
                        expected_reused,
                        strict=True,
                    )
                    for layer in group.layers
                )
                if getattr(encoded_pages, "total_bytes", None) != expected_bytes:
                    raise HybridCodecError(
                        "native page extension payload length differs"
                    )
            snapshot = _HybridStoreSnapshot(
                plan=plan,
                rank=self._worker_rank(),
                identity=self._identity(self._worker_rank()),
                positions=(),
                encoded_pages=encoded_pages,
                block_counts=block_counts,
            )
            self._ensure_store_thread()
            self._store_queue.put(snapshot)
        except Exception as error:  # noqa: BLE001 - serving continues
            release = getattr(encoded_pages, "release", None)
            if callable(release):
                release()
            self._finish_store(plan.digest, committed=False, error=error)

    def _abort_async_page_capture(self, digest: str, reason: str) -> None:
        self.counters["async_page_capture_aborted"] = (
            self.counters.get("async_page_capture_aborted", 0) + 1
        )
        self._finish_store(
            digest,
            committed=False,
            error=RuntimeError(reason),
        )

    def _ensure_store_thread(self) -> None:
        with self._store_cv:
            if self._store_thread is not None:
                if not self._store_thread.is_alive():
                    raise RuntimeError("background saver exited unexpectedly")
                return
            if not self._store_accepting:
                raise RuntimeError("background saver is shutting down")
            thread = threading.Thread(
                target=self._store_worker_main,
                name="spark-cache-store",
                daemon=True,
            )
            thread.start()
            self._store_thread = thread

    def _store_worker_main(self) -> None:
        while True:
            snapshot = self._store_queue.get()
            if snapshot is None:
                return
            commit_started = time.perf_counter()
            try:
                chunks = (
                    None
                    if isinstance(snapshot, _HybridStoreSnapshot)
                    else _SnapshotChunks(snapshot, self._dcp_degree, self._chunk_tokens)
                )
                if snapshot.plan.base_context_digest and (
                    snapshot.plan.digest
                    != self._digest(
                        list(snapshot.plan.token_ids),
                        snapshot.plan.span_tokens,
                        request_scope=snapshot.plan.request_scope,
                    )
                ):
                    raise RuntimeError(
                        "tail publication result digest differs from request"
                    )
                if (
                    isinstance(snapshot, _HybridStoreSnapshot)
                    and snapshot.plan.base_context_digest
                ):
                    layout = self._page_layout
                    if layout is None:
                        raise RuntimeError("block-page layout was not registered")
                    result_snapshot = snapshot.encoded_pages
                    verified_base_snapshot = None
                    reused_pages = getattr(
                        result_snapshot, "reused_pages_by_group", None
                    )
                    if (
                        reused_pages is not None
                        and not self._publication_schema.startswith("page-tail-cow-v")
                    ):
                        base = self._store.lookup(
                            snapshot.identity,
                            snapshot.plan.base_context_digest,
                            # restore_page_snapshot authenticates the bytes it
                            # returns; a preceding payload read adds no proof.
                            verify_chunks=False,
                        )
                        if not base.is_hit:
                            raise RuntimeError(
                                f"page extension base is unavailable: {base.reason}"
                            )
                        base_counts = self._group_block_counts_for_span(
                            snapshot.plan.base_span_tokens
                        )
                        base_snapshot = self._store.restore_page_snapshot(
                            base,
                            layout=layout,
                            result_block_counts=base_counts,
                            result_boundary_tokens=snapshot.plan.base_span_tokens,
                        )
                        result_snapshot = materialize_page_extension_capture(
                            layout,
                            base_snapshot,
                            result_snapshot,
                            base_block_counts=base_counts,
                            result_block_counts=snapshot.block_counts,
                            reused_pages_by_group=reused_pages,
                        )
                        verified_base_snapshot = base_snapshot
                    try:
                        receipt = self._store.commit_page_extension(
                            identity=snapshot.identity,
                            base_context_digest=snapshot.plan.base_context_digest,
                            token_ids=snapshot.plan.token_ids,
                            identity_salt=self._scope_salt(snapshot.plan.request_scope),
                            layout=layout,
                            base_block_counts=self._group_block_counts_for_span(
                                snapshot.plan.base_span_tokens
                            ),
                            result_block_counts=snapshot.block_counts,
                            base_boundary_tokens=snapshot.plan.base_span_tokens,
                            result_boundary_tokens=snapshot.plan.span_tokens,
                            result_snapshot=result_snapshot,
                            verified_base_snapshot=verified_base_snapshot,
                        )
                    except PageDeltaDepthExceeded:
                        receipt = self._store.commit_page_snapshot(
                            identity=snapshot.identity,
                            context_digest=snapshot.plan.digest,
                            span_tokens=snapshot.plan.span_tokens,
                            snapshot=snapshot.encoded_pages,
                        )
                        self.counters["page_delta_compactions"] += 1
                elif snapshot.plan.base_context_digest:
                    assert chunks is not None
                    receipt = self._store.commit_extension(
                        identity=snapshot.identity,
                        base_context_digest=snapshot.plan.base_context_digest,
                        token_ids=snapshot.plan.token_ids,
                        identity_salt=self._scope_salt(snapshot.plan.request_scope),
                        tail_chunks=chunks,
                    )
                elif isinstance(snapshot, _HybridStoreSnapshot):
                    receipt = self._store.commit_page_snapshot(
                        identity=snapshot.identity,
                        context_digest=snapshot.plan.digest,
                        span_tokens=snapshot.plan.span_tokens,
                        snapshot=snapshot.encoded_pages,
                    )
                else:
                    assert chunks is not None
                    receipt = self._store.commit(
                        identity=snapshot.identity,
                        context_digest=snapshot.plan.digest,
                        chunks=chunks,
                        span_tokens=snapshot.plan.span_tokens,
                    )
                alias_digests = self._publish_row_prefix_aliases(snapshot)
                with self._capacity_lock:
                    # The committed result retains its complete dependency graph.
                    # Release under the maintenance lock before selecting victims.
                    self._release_publication_base_pin(snapshot.plan.digest)
                    self._note_capacity_commit_locked(
                        receipt.allocated_bytes_upper_bound
                    )
                    evicted = self._post_commit_was_evicted_locked(
                        snapshot.identity,
                        snapshot.plan.digest,
                        force_maintenance=bool(alias_digests),
                    )
                    logger.debug(
                        "sparkcache: rank %d committed %d tokens"
                        " digest=%s manifest=%s in %.1f ms",
                        snapshot.rank,
                        receipt.committed_tokens,
                        snapshot.plan.digest[:12],
                        receipt.manifest_digest[:12],
                        1e3 * (time.perf_counter() - commit_started),
                    )
                    if receipt.publication is not None:
                        logger.debug(receipt.publication.format_compact())
                    self._finish_store(
                        snapshot.plan.digest,
                        committed=not evicted,
                        evicted=evicted,
                        additional_digests=(
                            self._surviving_alias_digests(
                                snapshot.identity,
                                alias_digests,
                            )
                            if not evicted
                            else ()
                        ),
                    )
            except Exception as error:  # noqa: BLE001 - never kill serving
                self._finish_store(
                    snapshot.plan.digest,
                    committed=False,
                    error=error,
                )
                continue
            finally:
                encoded_pages = getattr(snapshot, "encoded_pages", None)
                release = getattr(encoded_pages, "release", None)
                if callable(release):
                    release()

    def _publish_row_prefix_aliases(
        self,
        snapshot: _StoreSnapshot | _HybridStoreSnapshot,
    ) -> set[str]:
        """Publish sparse aliases without changing exact commit success.

        Alias derivation is intentionally post-commit: its source must be a
        durable exact row manifest. Any alias-side error is isolated and the
        exact digest remains eligible for ordinary restore.
        """

        plan = snapshot.plan
        if (
            self._storage_mode != "per_token_rows"
            or self._streaming_snapshots_enabled
            or not plan.token_ids
            or plan.has_multimodal_identity
        ):
            return set()
        self.counters["prefix_alias_publication_attempted"] += 1
        try:
            receipt = self._store.publish_prefix_aliases(
                identity=snapshot.identity,
                source_context_digest=plan.digest,
                token_ids=plan.token_ids,
                identity_salt=self._scope_salt(plan.request_scope),
                storage_mode="per_token_rows",
            )
        except Exception as error:  # noqa: BLE001 - exact entry remains usable
            self.counters["prefix_alias_publication_failed"] += 1
            _warning(
                "sparkcache: prefix alias publication skipped for"
                " exact digest=%s: %s",
                plan.digest[:12],
                error,
            )
            return set()
        self.counters["prefix_aliases_published"] += int(receipt.aliases_published)
        self.counters["prefix_alias_segments_published"] += int(
            receipt.segments_published
        )
        return {entry.context_digest for entry in receipt.alias_keys}

    def _surviving_alias_digests(
        self,
        identity: CacheIdentity,
        digests: set[str],
    ) -> tuple[str, ...]:
        """Return aliases still structurally reusable after maintenance."""

        surviving: list[str] = []
        for digest in sorted(digests):
            try:
                lookup, _is_alias = self._lookup_reusable(
                    identity,
                    digest,
                    verify_chunks=False,
                    verify_chunk_metadata=True,
                )
            except Exception as error:  # noqa: BLE001 - exact commit stands
                self.counters["prefix_alias_advertisement_failed"] += 1
                _warning(
                    "sparkcache: prefix alias not advertised digest=%s: %s",
                    digest[:12],
                    error,
                )
                continue
            # The source-boundary alias shares the exact digest and is hidden
            # by that exact manifest. It remains safe to advertise that key.
            if lookup.is_hit:
                surviving.append(digest)
        return tuple(surviving)

    def _finish_store(
        self,
        digest: str,
        *,
        committed: bool,
        evicted: bool = False,
        additional_digests: Sequence[str] = (),
        error: BaseException | None = None,
    ) -> None:
        self._release_publication_base_pin(digest)
        started_ns = getattr(self, "_store_pending_started_ns", None)
        tokens = getattr(self, "_store_pending_tokens", 0)
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000 if started_ns else 0.0
        with contextlib.suppress(Exception):
            logger.info(
                "sparkcache: save %s tokens=%d ms=%.1f rank=%d reason=%s",
                "committed" if committed else "refused", tokens, elapsed_ms,
                self._physical_rank(), "ok" if committed else (
                    "evicted" if evicted else _operator_reason(error) if error else "error"),
            )
        with self._store_cv:
            if committed:
                # ManifestStore publishes each fsynced immutable chunk before
                # atomically publishing the fsynced manifest. Load re-verifies
                # every byte, so no completion-time readback or global sweep
                # is needed to make this digest eligible for quorum.
                self._held.add(digest)
                self._held.update(additional_digests)
                # Withdrawal and repair can occur between inventory reports.
                self._stats_readvertise_held.add(digest)
                self.counters["store_committed"] += 1
            else:
                self._held.discard(digest)
                self._held.difference_update(additional_digests)
                self.counters["store_evicted" if evicted else "store_failed"] += 1
            self._store_inflight = 0
            self._store_pending_started_ns = None
            self._store_cv.notify_all()
        if error is not None:
            _warning(
                "sparkcache: store failed (entry skipped): %s",
                error,
            )

    def wait_for_pending_stores(self, timeout: float | None = None) -> bool:
        with self._store_cv:
            return self._store_cv.wait_for(
                lambda: self._store_inflight == 0,
                timeout,
            )

    def _store_one(self, plan: _ReqPlan) -> None:
        """Compatibility helper for offline callers; not the request path."""

        snapshot = self._snapshot_store(plan)
        chunks = (
            None
            if isinstance(snapshot, _HybridStoreSnapshot)
            else _SnapshotChunks(snapshot, self._dcp_degree, self._chunk_tokens)
        )
        if plan.base_context_digest and (
            plan.digest != self._digest(list(plan.token_ids), plan.span_tokens,
                                       request_scope=plan.request_scope)
        ):
            raise RuntimeError("tail publication result digest differs from request")
        if isinstance(snapshot, _HybridStoreSnapshot) and plan.base_context_digest:
            layout = self._page_layout
            if layout is None:
                raise RuntimeError("block-page layout was not registered")
            try:
                receipt = self._store.commit_page_extension(
                    identity=snapshot.identity,
                    base_context_digest=plan.base_context_digest,
                    token_ids=plan.token_ids,
                    identity_salt=self._scope_salt(plan.request_scope),
                    layout=layout,
                    base_block_counts=self._group_block_counts_for_span(
                        plan.base_span_tokens
                    ),
                    result_block_counts=snapshot.block_counts,
                    base_boundary_tokens=plan.base_span_tokens,
                    result_boundary_tokens=plan.span_tokens,
                    result_snapshot=snapshot.encoded_pages,
                )
            except PageDeltaDepthExceeded:
                receipt = self._store.commit_page_snapshot(
                    identity=snapshot.identity,
                    context_digest=plan.digest,
                    span_tokens=plan.span_tokens,
                    snapshot=snapshot.encoded_pages,
                )
                self.counters["page_delta_compactions"] += 1
        elif plan.base_context_digest:
            assert chunks is not None
            receipt = self._store.commit_extension(
                identity=snapshot.identity,
                base_context_digest=plan.base_context_digest,
                token_ids=plan.token_ids,
                identity_salt=self._scope_salt(plan.request_scope),
                tail_chunks=chunks,
            )
        elif isinstance(snapshot, _HybridStoreSnapshot):
            receipt = self._store.commit_page_snapshot(
                identity=snapshot.identity,
                context_digest=plan.digest,
                span_tokens=plan.span_tokens,
                snapshot=snapshot.encoded_pages,
            )
        else:
            assert chunks is not None
            receipt = self._store.commit(
                identity=snapshot.identity,
                context_digest=plan.digest,
                chunks=chunks,
                span_tokens=plan.span_tokens,
            )
        alias_digests = self._publish_row_prefix_aliases(snapshot)
        with self._capacity_lock:
            self._note_capacity_commit_locked(receipt.allocated_bytes_upper_bound)
            evicted = self._post_commit_was_evicted_locked(
                snapshot.identity,
                plan.digest,
                force_maintenance=bool(alias_digests),
            )
            if evicted:
                with self._store_cv:
                    self._held.discard(plan.digest)
                    self._held.difference_update(alias_digests)
            else:
                surviving_aliases = self._surviving_alias_digests(
                    snapshot.identity,
                    alias_digests,
                )
                with self._store_cv:
                    self._held.add(plan.digest)
                    self._held.update(surviving_aliases)
        logger.debug(
            "sparkcache: rank %d committed %d tokens digest=%s manifest=%s",
            snapshot.rank,
            receipt.committed_tokens,
            plan.digest[:12],
            receipt.manifest_digest[:12],
        )

    def _withdraw_worker_quorum(self, rank: int) -> None:
        for digest, ranks in list(self._quorum.items()):
            ranks.discard(rank)
            if not ranks:
                self._quorum.pop(digest, None)

    def _publish_worker_quorum(self, rank: int) -> None:
        for digest in self._worker_held.get(rank, ()):
            self._quorum.setdefault(digest, set()).add(rank)

    def _mark_worker_desynchronized(self, rank: int) -> None:
        if rank not in self._worker_desynchronized:
            self._withdraw_worker_quorum(rank)
            self._worker_desynchronized.add(rank)

    def _replace_worker_held(self, rank: int, held: set[str]) -> None:
        self._withdraw_worker_quorum(rank)
        self._worker_held[rank] = held
        if rank not in self._worker_desynchronized:
            self._publish_worker_quorum(rank)

    def _begin_worker_generation(
        self, rank: int, generation: str, generation_epoch: int | None
    ) -> None:
        self._withdraw_worker_quorum(rank)
        self._worker_publication_pending.discard(rank)
        self._worker_incompatible.pop(rank, None)
        self._worker_generations[rank] = generation
        if generation_epoch is None:
            self._worker_generation_epochs.pop(rank, None)
        else:
            self._worker_generation_epochs[rank] = generation_epoch
        self._worker_report_sequences[rank] = 0
        self._worker_held[rank] = set()
        self._worker_pending_deltas.pop(rank, None)
        self._worker_checkpoints.pop(rank, None)
        self._worker_desynchronized.discard(rank)
        self._worker_requires_checkpoint.discard(rank)

    def _accept_worker_generation(
        self, rank: int, generation: str, generation_epoch: int | None
    ) -> bool:
        previous = self._worker_generations.get(rank)
        if previous == generation:
            expected_epoch = self._worker_generation_epochs.get(rank)
            return expected_epoch is None or generation_epoch == expected_epoch
        # Generation UUIDs have no cross-reboot ordering. Remember a bounded
        # set of retired UUIDs so delayed reports cannot replace the serving
        # process generation after one or more worker restarts.
        retired = self._worker_retired_generations.setdefault(rank, [])
        if generation in retired:
            return False
        if generation == "missing-generation-field" and previous is not None:
            return False
        if previous is not None:
            retired.append(previous)
            if len(retired) > _QUORUM_RETIRED_GENERATION_LIMIT:
                del retired[:-_QUORUM_RETIRED_GENERATION_LIMIT]
            self.counters["quorum_generation_resets"] += 1
        self._begin_worker_generation(rank, generation, generation_epoch)
        return True

    def _apply_worker_delta(
        self, rank: int, added: set[str], removed: set[str]
    ) -> None:
        held = self._worker_held.setdefault(rank, set())
        held.difference_update(removed)
        held.update(added)
        if rank in self._worker_desynchronized:
            return
        for digest in removed:
            ranks = self._quorum.get(digest)
            if ranks is None:
                continue
            ranks.discard(rank)
            if not ranks:
                self._quorum.pop(digest, None)
        for digest in added:
            self._quorum.setdefault(digest, set()).add(rank)

    def _drain_worker_deltas(self, rank: int) -> None:
        pending = self._worker_pending_deltas.setdefault(rank, {})
        sequence = self._worker_report_sequences.get(rank, 0)
        while True:
            delta = pending.get(sequence + 1)
            if delta is None or delta["base_sequence"] != sequence:
                break
            pending.pop(sequence + 1)
            self._apply_worker_delta(
                rank,
                set(delta["added"]),
                set(delta["removed"]),
            )
            sequence += 1
            self._worker_report_sequences[rank] = sequence
        if pending:
            self._mark_worker_desynchronized(rank)
            return
        if rank in self._worker_requires_checkpoint:
            self._mark_worker_desynchronized(rank)
            return
        if rank in self._worker_desynchronized:
            self._worker_desynchronized.discard(rank)
            self._publish_worker_quorum(rank)

    def _absorb_worker_delta(self, rank: int, delta: Any) -> None:
        if not isinstance(delta, dict):
            return
        sequence = delta.get("sequence")
        base_sequence = delta.get("base_sequence")
        added = delta.get("added")
        removed = delta.get("removed")
        if (
            type(sequence) is not int
            or type(base_sequence) is not int
            or sequence != base_sequence + 1
            or base_sequence < 0
            or not isinstance(added, list)
            or not isinstance(removed, list)
            or len(added) + len(removed) > _QUORUM_REPORT_BATCH_SIZE
        ):
            return
        added_set = {
            digest
            for digest in added
            if isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None
        }
        removed_set = {
            digest
            for digest in removed
            if isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None
        }
        if (
            len(added_set) != len(added)
            or len(removed_set) != len(removed)
            or added_set & removed_set
        ):
            return
        accepted = self._worker_report_sequences.get(rank, 0)
        if sequence <= accepted:
            return
        normalized = {
            "sequence": sequence,
            "base_sequence": base_sequence,
            "added": tuple(sorted(added_set)),
            "removed": tuple(sorted(removed_set)),
        }
        pending = self._worker_pending_deltas.setdefault(rank, {})
        duplicate = pending.get(sequence)
        if duplicate is not None:
            if duplicate != normalized:
                self._worker_requires_checkpoint.add(rank)
                self._mark_worker_desynchronized(rank)
            return
        if len(pending) >= _QUORUM_PENDING_DELTA_LIMIT:
            self._worker_requires_checkpoint.add(rank)
            self._mark_worker_desynchronized(rank)
            if sequence >= max(pending):
                return
            pending.pop(max(pending))
        pending[sequence] = normalized
        if base_sequence != accepted:
            self._mark_worker_desynchronized(rank)
        self._drain_worker_deltas(rank)

    def _absorb_worker_checkpoint(self, rank: int, checkpoint: Any) -> None:
        if not isinstance(checkpoint, dict):
            return
        state_sequence = checkpoint.get("state_sequence")
        cycle = checkpoint.get("cycle")
        index = checkpoint.get("index")
        count = checkpoint.get("count")
        held_count = checkpoint.get("held_count")
        held = checkpoint.get("held")
        if (
            type(state_sequence) is not int
            or type(cycle) is not int
            or type(index) is not int
            or type(count) is not int
            or type(held_count) is not int
            or state_sequence < 0
            or cycle < 1
            or count < 1
            or not 0 <= index < count
            or held_count < 0
            or not isinstance(held, list)
            or len(held) > _QUORUM_REPORT_BATCH_SIZE
        ):
            return
        expected_count = max(1, math.ceil(held_count / _QUORUM_REPORT_BATCH_SIZE))
        expected_chunk_size = min(
            _QUORUM_REPORT_BATCH_SIZE,
            max(0, held_count - index * _QUORUM_REPORT_BATCH_SIZE),
        )
        if count != expected_count or len(held) != expected_chunk_size:
            return
        held_chunk = tuple(
            digest
            for digest in held
            if isinstance(digest, str) and SHA256_RE.fullmatch(digest) is not None
        )
        if len(held_chunk) != len(held) or len(set(held_chunk)) != len(held_chunk):
            return
        accepted = self._worker_report_sequences.get(rank, 0)
        if state_sequence < accepted:
            return
        if (
            state_sequence == accepted
            and rank not in self._worker_desynchronized
            and rank not in self._worker_requires_checkpoint
        ):
            self._worker_checkpoints.pop(rank, None)
            return
        if state_sequence > accepted:
            self._mark_worker_desynchronized(rank)
        candidate_key = (state_sequence, cycle)
        active = self._worker_checkpoints.get(rank)
        if active is not None:
            active_key = (active["state_sequence"], active["cycle"])
            if candidate_key < active_key:
                return
            if candidate_key > active_key:
                active = None
        if active is None:
            active = {
                "state_sequence": state_sequence,
                "cycle": cycle,
                "count": count,
                "held_count": held_count,
                "chunks": {},
            }
            self._worker_checkpoints[rank] = active
        elif active["count"] != count or active["held_count"] != held_count:
            self._worker_requires_checkpoint.add(rank)
            self._mark_worker_desynchronized(rank)
            return
        chunks = active["chunks"]
        prior_chunk = chunks.get(index)
        if prior_chunk is not None and prior_chunk != held_chunk:
            self._worker_requires_checkpoint.add(rank)
            self._mark_worker_desynchronized(rank)
            return
        chunks[index] = held_chunk
        if len(chunks) != count:
            return
        reconstructed = {
            digest for chunk_index in range(count) for digest in chunks[chunk_index]
        }
        if len(reconstructed) != held_count:
            self._worker_requires_checkpoint.add(rank)
            self._mark_worker_desynchronized(rank)
            return
        self._worker_desynchronized.add(rank)
        self._replace_worker_held(rank, reconstructed)
        self._worker_report_sequences[rank] = state_sequence
        pending = self._worker_pending_deltas.setdefault(rank, {})
        for delta_sequence in tuple(pending):
            if delta_sequence <= state_sequence:
                pending.pop(delta_sequence, None)
        self._worker_requires_checkpoint.discard(rank)
        self._worker_checkpoints.pop(rank, None)
        self._drain_worker_deltas(rank)

    def _absorb_quorum(self, connector_output: Any) -> None:
        stats = getattr(connector_output, "kv_connector_stats", None)
        data = getattr(stats, "data", None)
        if not isinstance(data, dict):
            return
        payload = data.get("reports", data.get("spark_context_cache"))
        reports = payload if isinstance(payload, list) else [payload]
        for report in reports:
            if not isinstance(report, dict):
                continue
            rank = report.get("rank")
            generation = report.get("generation")
            if type(rank) is not int or not 0 <= rank < self._tp_degree:
                continue
            if not isinstance(generation, str) or not generation:
                # Reports without a generation field share one inert sentinel.
                # Connectors from this source always send a UUID generation, so
                # an unversioned report cannot reset a worker's UUID state.
                generation = "missing-generation-field"
            generation_epoch = report.get("generation_epoch")
            if type(generation_epoch) is not int or generation_epoch < 0:
                generation_epoch = None
            protocol = report.get("protocol")
            if protocol == _QUORUM_DELTA_PROTOCOL:
                if generation == "missing-generation-field" or generation_epoch is None:
                    continue
                if not self._accept_worker_generation(
                    rank, generation, generation_epoch
                ):
                    continue
                self._absorb_worker_delta(rank, report.get("delta"))
                self._absorb_worker_checkpoint(rank, report.get("checkpoint"))
                self._absorb_incompatible_manifests(rank, report)
                publication = report.get("publication_work")
                if isinstance(publication, dict):
                    sequence = report.get("inventory_sequence")
                    if publication.get("pending") or (
                        type(sequence) is int
                        and (
                            self._worker_report_sequences.get(rank, -1) < sequence
                            or rank in self._worker_desynchronized
                        )
                    ):
                        self._worker_publication_pending.add(rank)
                    else:
                        self._worker_publication_pending.discard(rank)
                continue
            held = report.get("held")
            if not isinstance(held, list):
                continue
            if not self._accept_worker_generation(rank, generation, generation_epoch):
                continue
            held_set = {digest for digest in held if isinstance(digest, str)}
            self._worker_pending_deltas.pop(rank, None)
            self._worker_checkpoints.pop(rank, None)
            self._worker_desynchronized.discard(rank)
            self._worker_requires_checkpoint.discard(rank)
            self._replace_worker_held(rank, held_set)

    def update_connector_output(self, connector_output: Any) -> None:
        self._absorb_quorum(connector_output)
        reads = getattr(connector_output, "kv_connector_worker_meta", None)
        if isinstance(reads, SparkCacheReadCompletionMetadata):
            leases = self._capture_read_leases
            if leases is None and reads.completed_reads:
                raise RuntimeError("capture completion arrived without a source pool")
            if leases is not None:
                for job, ranks in reads.uncertain_reads.items():
                    leases.quarantine(job, ranks)
                for job, ranks in reads.completed_reads.items():
                    leases.complete(job, ranks)
        self._release_async_page_capture_reservations(
            tuple(getattr(connector_output, "finished_sending", None) or ())
        )
        invalid = getattr(connector_output, "invalid_block_ids", None)
        invalid_blocks = set(invalid or ())
        # Async failure output reaches this callback before the parked
        # request is rescheduled, so retire the admission before its clean
        # recompute can republish the entry. Only admissions whose restored
        # blocks intersect the reported invalid blocks are retired: the
        # report carries no request id, and retiring every admission would
        # destroy unrelated healthy entries.
        for request_id, (digest, block_ids) in list(self._admitted.items()):
            if not (block_ids & invalid_blocks):
                continue
            # Each damaged worker retires its own entry. The merged block
            # errors do not identify a rank, so even a scheduler sharing TP0's
            # root must preserve that potentially healthy copy. Withdraw all
            # offers until recompute renews each worker's advertisement.
            self._quorum.pop(digest, None)
            flight = self._restore_flights.get(digest)
            if flight is not None and flight.segment_digest is not None:
                self._quorum.pop(flight.segment_digest, None)
            self._admitted.pop(request_id, None)
            self._retire_restore_flight(digest, outcome="failed")

        # A client may abort a leader after its plan is dispatched. Its
        # ordinary admission bookkeeping is already gone, but the flight
        # retains the block set until worker completion so a failed write can
        # still withdraw quorum and release followers to recompute.
        for digest, flight in list(self._restore_flights.items()):
            if flight.restored_block_ids & invalid_blocks:
                self._quorum.pop(digest, None)
                if flight.segment_digest is not None:
                    self._quorum.pop(flight.segment_digest, None)
                self._retire_restore_flight(digest, outcome="failed")

        for request_id in getattr(connector_output, "finished_recving", None) or ():
            digest = self._restore_flight_leaders.get(request_id)
            if digest is not None:
                flight = self._restore_flights.get(digest)
                if flight is None:
                    continue
                if flight.leader_finished:
                    # A finished/aborted leader is never published. Worker
                    # completion merely proves its destination blocks are no
                    # longer being written, so followers may recompute.
                    self._retire_restore_flight(digest, outcome="cancelled")
                else:
                    flight.workers_finished = True
                    self._log_restore_outcome(flight, "hit")

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return SparkCacheStats(data=data if data is not None else {})

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[Any], type[Any]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return SparkCachePromMetrics(
            vllm_config,
            metric_types,
            labelnames,
            per_engine_labelvalues,
        )

    def _build_quorum_report_locked(self) -> dict[str, Any]:
        inventory = self._held
        if (
            self._stats_observed_inventory is not inventory
            or self._stats_observed_revision != inventory.revision
            or self._stats_readvertise_held
        ):
            held = inventory.copy()
            added = (held - self._stats_observed_held) | (
                self._stats_readvertise_held & held
            )
            removed = self._stats_observed_held - held
            self._stats_readvertise_held.clear()
            if added or removed:
                base_sequence = self._stats_sequence
                self._stats_sequence += 1
                if len(added) + len(removed) <= _QUORUM_REPORT_BATCH_SIZE:
                    self._stats_delta_history.append(
                        {
                            "sequence": self._stats_sequence,
                            "base_sequence": base_sequence,
                            "added": sorted(added),
                            "removed": sorted(removed),
                        }
                    )
                    if len(self._stats_delta_history) > _QUORUM_DELTA_HISTORY_SIZE:
                        self._stats_delta_history = self._stats_delta_history[
                            -_QUORUM_DELTA_HISTORY_SIZE:
                        ]
                    self._stats_delta_cursor %= len(self._stats_delta_history)
                self._stats_observed_held = held
                self._stats_checkpoint_items = tuple(sorted(held))
                self._stats_checkpoint_sequence = self._stats_sequence
                self._stats_checkpoint_cycle += 1
                self._stats_checkpoint_index = 0
            self._stats_observed_inventory = inventory
            self._stats_observed_revision = inventory.revision

        checkpoint_count = max(
            1,
            math.ceil(len(self._stats_checkpoint_items) / _QUORUM_REPORT_BATCH_SIZE),
        )
        checkpoint_index = self._stats_checkpoint_index
        checkpoint_start = checkpoint_index * _QUORUM_REPORT_BATCH_SIZE
        checkpoint_held = list(
            self._stats_checkpoint_items[
                checkpoint_start : checkpoint_start + _QUORUM_REPORT_BATCH_SIZE
            ]
        )
        checkpoint = {
            "state_sequence": self._stats_checkpoint_sequence,
            "cycle": self._stats_checkpoint_cycle,
            "index": checkpoint_index,
            "count": checkpoint_count,
            "held_count": len(self._stats_checkpoint_items),
            "held": checkpoint_held,
        }
        self._stats_checkpoint_index += 1
        if self._stats_checkpoint_index >= checkpoint_count:
            self._stats_checkpoint_index = 0
            self._stats_checkpoint_cycle += 1

        report: dict[str, Any] = {
            "rank": self._physical_rank(),
            "protocol": _QUORUM_DELTA_PROTOCOL,
            "generation": self._stats_generation,
            "generation_epoch": self._stats_generation_epoch,
            "held_count": len(inventory),
            "inventory_sequence": self._stats_sequence,
            # A scheduler without delta-protocol support interprets this as a
            # withdrawal instead of retaining stale full-set confirmations.
            "held": [],
            "checkpoint": checkpoint,
        }
        if self._stats_delta_history:
            delta_index = self._stats_delta_cursor % len(self._stats_delta_history)
            report["delta"] = dict(self._stats_delta_history[delta_index])
            self._stats_delta_cursor = (delta_index + 1) % len(
                self._stats_delta_history
            )
        start = self._incompatible_cursor
        report["incompatible"] = self._incompatible_inventory[start:start + _QUORUM_REPORT_BATCH_SIZE]
        self._incompatible_cursor = (
            (start + _QUORUM_REPORT_BATCH_SIZE) % len(self._incompatible_inventory)
            if self._incompatible_inventory else 0
        )
        return report

    def get_kv_connector_stats(self):
        """Report this rank's manifest-offered digests to the scheduler.

        This is the admission quorum's transport: the scheduler will only
        offer a restore that every physical TP worker has confirmed.
        Load-time corruption withdraws the failing worker's offer and
        publishes invalid blocks so the request cleanly re-prefills
        instead of exposing restored state.

        The report uses the physical TP rank (unique across all workers),
        not the DCP-local rank, so TP0 and TP2 are not deduplicated
        despite sharing DCP-local rank 0 under TP4/DCP2.

        Each report carries at most one bounded state delta, one bounded
        checkpoint chunk, and one diagnostic batch for other schemas.
        Sequence gaps withdraw the rank's confirmations;
        replayed deltas or a complete rolling checkpoint restore them.
        """
        if self._role is not KVConnectorRole.WORKER:
            return None
        with self._load_lock:
            report = self._build_quorum_report_locked()
            pending = int(bool(self._store_inflight))
            started_ns = getattr(self, "_store_pending_started_ns", None)
            report["publication_work"] = {
                "pending": pending,
                "oldest_pending_ms": (
                    max(0, time.perf_counter_ns() - started_ns) / 1_000_000
                    if pending and started_ns is not None else 0.0
                ),
                # Reading the activity counter must not wait for the capacity
                # lock held by the scan being measured.
                "maintenance_active": bool(getattr(self, "_capacity_maintenance_depth", 0)),
            }
        runtime = self._streaming_runtime
        status = getattr(runtime, "status", None)
        if self._streaming_snapshots_enabled and callable(status):
            report["streaming"] = status()
        if self._capacity_policy.enabled:
            capacity = dict(self._capacity_status)
            with self._capacity_handoff_cv:
                pending_streaming_commits = len(self._streaming_capacity_pending)
            capacity.update(
                manifests_evicted=self.counters["capacity_manifests_evicted"],
                chunks_deleted=self.counters["capacity_chunks_deleted"],
                bytes_reclaimed=self.counters["capacity_bytes_reclaimed"],
                pending_streaming_commits=pending_streaming_commits,
                streaming_store_committed=self.counters["streaming_store_committed"],
                streaming_store_evicted=self.counters["streaming_store_evicted"],
                invalid_streaming_receipts=self.counters[
                    "streaming_capacity_invalid_receipts"
                ],
                shutdown_dropped_streaming_commits=self.counters[
                    "streaming_capacity_shutdown_dropped"
                ],
                maintenance_retries=self.counters["capacity_retries"],
                maintenance_deletion_attempts=self.counters.get("capacity_deletion_attempts", 0),
                maintenance_stale_inventory_snapshots=self.counters.get("capacity_stale_inventory_snapshots", 0),
                maintenance_budget_exhausted=self.counters.get("capacity_budget_exhausted", 0),
                maintenance_skipped_cooldown=self.counters.get("capacity_skipped_cooldown", 0),
            )
            report["capacity"] = capacity
        runtime = self._async_page_capture_runtime
        status = getattr(runtime, "status", None)
        if self._async_page_capture_enabled and callable(status):
            async_capture = dict(status())
            with self._store_cv:
                async_capture["store_inflight"] = bool(self._store_inflight)
            report["async_capture"] = async_capture
        report["publication"] = self._store.publication_telemetry_snapshot().as_dict()
        report["counters"] = dict(self.counters)
        return SparkCacheStats(data={"reports": [report]})

    def get_handshake_metadata(self) -> SparkCacheHandshakeMetadata | None:
        """Return a bounded manifest inventory at engine startup.

        vLLM collects connector handshake metadata after worker KV-cache
        registration and before the engine becomes ready. SparkCache uses that
        existing exchange to establish the same physical-rank quorum otherwise
        learned from post-request worker statistics. The handshake carries at
        most ``_STARTUP_HANDSHAKE_MAX_DIGESTS`` entries in each of the compatible
        and incompatible inventories so cache size cannot make the readiness
        exchange unbounded. Later bounded statistics report entries
        outside that startup subset.
        """

        if self._role is not KVConnectorRole.WORKER or not self._restore_enabled:
            return None
        with self._load_lock:
            held = tuple(sorted(self._held))[:_STARTUP_HANDSHAKE_MAX_DIGESTS]
            state_sequence = self._stats_sequence + int(
                self._held != self._stats_observed_held
            )
            checkpoint_count = max(
                1,
                math.ceil(len(held) / _QUORUM_REPORT_BATCH_SIZE),
            )
            reports = []
            incompatible = self._incompatible_inventory[:_STARTUP_HANDSHAKE_MAX_DIGESTS]
            report_count = max(checkpoint_count, math.ceil(len(incompatible) / _QUORUM_REPORT_BATCH_SIZE))
            for report_index in range(report_count):
                index = report_index % checkpoint_count
                start = index * _QUORUM_REPORT_BATCH_SIZE
                diagnostic_start = report_index * _QUORUM_REPORT_BATCH_SIZE
                checkpoint_held = list(
                    held[start : start + _QUORUM_REPORT_BATCH_SIZE]
                )
                reports.append(
                    {
                        "rank": self._physical_rank(),
                        "protocol": _QUORUM_DELTA_PROTOCOL,
                        "generation": self._stats_generation,
                        "generation_epoch": self._stats_generation_epoch,
                        "held_count": len(held),
                        "held": [],
                        "incompatible": incompatible[diagnostic_start:diagnostic_start + _QUORUM_REPORT_BATCH_SIZE],
                        "checkpoint": {
                            "state_sequence": state_sequence,
                            "cycle": self._stats_checkpoint_cycle,
                            "index": index,
                            "count": checkpoint_count,
                            "held_count": len(held),
                            "held": checkpoint_held,
                        },
                    }
                )
        return SparkCacheHandshakeMetadata(reports=tuple(reports))

    def set_xfer_handshake_metadata(
        self, metadata: dict[int, KVConnectorHandshakeMetadata]
    ) -> None:
        """Establish scheduler quorum from worker startup inventories.

        The transport rank must agree with each report's physical rank. A
        missing, malformed, or mismatched worker report contributes no quorum;
        requests continue with ordinary prompt computation until later bounded
        worker statistics establish unanimity.
        """

        if self._role is not KVConnectorRole.SCHEDULER:
            return
        for transport_rank, handshake in sorted(metadata.items()):
            if type(transport_rank) is not int or not isinstance(
                handshake, SparkCacheHandshakeMetadata
            ):
                continue
            for report in handshake.reports:
                if not isinstance(report, dict) or report.get("rank") != transport_rank:
                    continue
                self._absorb_quorum(
                    SimpleNamespace(
                        kv_connector_stats=SparkCacheStats(
                            data={"reports": [dict(report)]}
                        )
                    )
                )

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._load_lock:
            errors, self._load_errors = self._load_errors, set()
        return errors
