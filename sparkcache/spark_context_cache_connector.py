"""SparkCache persistent NVMe context-cache connector (DCP4-native).

KVConnectorBase_V1 implementation that persists each DCP rank's token shard
of every registered cache layer (target CKV incl. per-token NVFP4 scales,
sparse-indexer state, MTP draft KV) to rank-local NVMe through the
fail-closed ManifestStore, and restores it after a runtime restart.

Restore uses KV-Connector-V1's asynchronous-load contract. A cache hit parks
only that request, allocates private blocks, and queues verification,
assembly, H2D copy, and scatter on a background loader. The request is
reported finished only after its writes have completed; unrelated requests
remain schedulable while it restores.

Fail-closed contract:
- Storage uses content-addressed chunks with per-record SHA-256 and an
  identity pinning model/quant/TP/DCP/shard-rank/chunk geometry.
- A failed asynchronous load publishes its invalid blocks and finished
  request id together. vLLM can then unpark it through the supported clean
  recompute path instead of exposing partially restored state.
- Store failures only log and count; they never fail a request.

Enabled explicitly via --kv-transfer-config naming this module; without
that, serving behavior is byte-identical to v40.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import queue
import re
import threading
import time
from pathlib import Path
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Sequence

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorStats,
)
from vllm.logger import init_logger

from spark_context_cache_codec import (
    CHUNK_TOKENS,
    CodecError,
    build_layer_plans,
    chunk_count,
    classify_layer,
    context_digest,
    local_slots_for_positions,
    owned_positions,
    pack_positions,
    pack_record,
    unpack_positions,
    unpack_record,
)
from spark_context_cache_store import (
    CacheIdentity,
    ContextChunk,
    ManifestStore,
    StateRecord,
)
from sparkcache.streaming.feature_gate import (
    ENVIRONMENT_KEY as _STREAMING_SNAPSHOTS_ENV,
    EXTRA_CONFIG_KEY as _STREAMING_SNAPSHOTS_CONFIG,
    is_enabled as _streaming_snapshots_enabled,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.spark_context_cache")

_RECORD_KINDS = ("target_ckv", "sparse_indexer", "mtp_draft_kv")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NATIVE_ARENA_BYTES = frozenset(
    {64 * 1024 * 1024, 128 * 1024 * 1024, 256 * 1024 * 1024}
)

# The runtime is deliberately supplied by the embedding process instead of
# importing (or constructing) the experimental native implementation here.
# A scheduler and each worker have distinct ownership, so factories are keyed
# by connector role.  Installing a factory is an explicit deployment action;
# setting the feature flag without one remains a startup error.
_STREAMING_RUNTIME_FACTORIES: dict[
    KVConnectorRole, Callable[["SparkContextCacheConnector"], Any]
] = {}


def configure_streaming_snapshot_runtime(
    role: KVConnectorRole,
    factory: Callable[["SparkContextCacheConnector"], Any] | None,
) -> None:
    """Install or remove the explicitly injected streaming runtime factory.

    This is a narrow integration seam for an attested embedding runtime.  It
    does not create, import, or otherwise enable a native runtime by itself.
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


def _load_native_components() -> SimpleNamespace:
    """Lazy optional import: the Python restore path needs no native bundle."""

    placement = importlib.import_module("spark_context_cache_native_placement")
    restore = importlib.import_module("spark_context_cache_native_restore")
    return SimpleNamespace(
        NativePlacementLibrary=placement.NativePlacementLibrary,
        NativePlacementAdapter=placement.NativePlacementAdapter,
        ArenaMode=placement.ArenaMode,
        RecordKind=placement.RecordKind,
        execute_native_restore=restore.execute_native_restore,
    )


@dataclass
class _ReqPlan:
    request_id: str
    digest: str
    span_tokens: int
    block_ids: tuple[int, ...]
    is_store: bool


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

    @property
    def span(self) -> int:
        return self.span_tokens

    @property
    def promised_completed_tokens(self) -> int:
        return self.completed_tokens

    @property
    def full_block_table(self) -> tuple[int, ...]:
        return self.block_ids


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


class _SnapshotChunks(Sequence[ContextChunk]):
    """Build one encoded chunk at a time from an immutable CPU snapshot."""

    def __init__(self, snapshot: _StoreSnapshot, dcp_degree: int) -> None:
        self._snapshot = snapshot
        self._dcp_degree = dcp_degree
        self._count = chunk_count(snapshot.plan.span_tokens)
        self._rows_per_chunk = CHUNK_TOKENS // dcp_degree

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
            logical_start=index * CHUNK_TOKENS,
            logical_end=(index + 1) * CHUNK_TOKENS,
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
class SparkCacheStats(KVConnectorStats):
    """Per-rank "digests I can serve" reports, merged across workers.

    MUST be defined at module scope: these objects are pickled across the
    worker->engine shared-memory queue, and a class created inside a
    function is not picklable.
    """

    def reset(self):
        self.data = {"reports": []}

    def aggregate(self, other: "KVConnectorStats") -> "KVConnectorStats":
        mine = list(self.data.get("reports", []))
        theirs = list(getattr(other, "data", {}).get("reports", []))
        merged: dict[int, dict[str, Any]] = {}
        for report in mine + theirs:
            if isinstance(report, dict) and isinstance(report.get("rank"), int):
                normalized: dict[str, Any] = {
                    "rank": report["rank"],
                    "held": list(report.get("held", [])),
                }
                streaming = report.get("streaming")
                if isinstance(streaming, dict):
                    normalized["streaming"] = dict(streaming)
                merged[report["rank"]] = normalized
        self.data = {
            "reports": [merged[rank] for rank in sorted(merged)]
        }
        return self

    def reduce(self) -> dict[str, int | float]:
        reports = self.data.get("reports", [])
        reduced: dict[str, int | float] = {
            "spark_cache_ranks_reporting": len(reports),
            "spark_cache_digests_held": sum(len(r.get("held", [])) for r in reports),
        }
        streaming = [
            report.get("streaming")
            for report in reports
            if isinstance(report.get("streaming"), dict)
        ]
        if streaming:
            reduced.update(
                spark_cache_streaming_ranks_reporting=len(streaming),
                spark_cache_streaming_active_contexts=sum(
                    int(status.get("active_contexts", 0))
                    for status in streaming
                ),
                spark_cache_streaming_active_leases=sum(
                    int(status.get("active_leases", 0))
                    for status in streaming
                ),
                spark_cache_streaming_active_tickets=sum(
                    int(status.get("active_tickets", 0))
                    for status in streaming
                ),
            )
        return reduced

    def is_empty(self) -> bool:
        return not self.data.get("reports")


class SparkContextCacheConnector(KVConnectorBase_V1):
    """Store/restore each rank's DCP shard on rank-local NVMe."""

    configure_streaming_snapshot_runtime = staticmethod(
        configure_streaming_snapshot_runtime
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
        self._block_size = vllm_config.cache_config.block_size
        parallel = vllm_config.parallel_config
        self._tp_degree = max(1, getattr(parallel, "tensor_parallel_size", 1))
        self._dcp_degree = max(1, getattr(parallel, "decode_context_parallel_size", 1))
        extra = self._kv_transfer_config.get_from_extra_config
        self._root = extra(
            "spark_cache_root",
            os.environ.get("SPARK_CONTEXT_CACHE_ROOT", "/cache/context"),
        )
        self._min_span = int(
            extra(
                "spark_cache_min_span_tokens",
                os.environ.get("SPARK_CONTEXT_CACHE_MIN_SPAN", "1024"),
            )
        )
        model_max = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
        default_max_span = str(model_max if model_max > 0 else 1 << 30)
        self._max_span = int(
            extra(
                "spark_cache_max_span_tokens",
                os.environ.get("SPARK_CONTEXT_CACHE_MAX_SPAN", default_max_span),
            )
        )
        self._store_enabled = extra(
            "spark_cache_store",
            os.environ.get("SPARK_CONTEXT_CACHE_STORE", "1"),
        ) in (1, "1", True, "true")
        self._restore_enabled = extra(
            "spark_cache_restore",
            os.environ.get("SPARK_CONTEXT_CACHE_RESTORE", "1"),
        ) in (1, "1", True, "true")
        try:
            self._streaming_snapshots_enabled = _streaming_snapshots_enabled(
                extra(
                    _STREAMING_SNAPSHOTS_CONFIG,
                    os.environ.get(_STREAMING_SNAPSHOTS_ENV, "0"),
                )
            )
        except ValueError as error:
            raise RuntimeError(f"spark-context-cache: {error}") from error
        self._streaming_runtime: Any = None
        if self._streaming_snapshots_enabled:
            # An explicit opt-in never falls back to end-of-prefill snapshots.
            # Preserve an explicitly injected test/deployment adapter. When
            # none exists, import the builtin factory only on this opt-in
            # path; default-off scheduler and worker processes never import
            # factory/native-ring modules.
            if role not in _STREAMING_RUNTIME_FACTORIES:
                from sparkcache.streaming.factory import (
                    make_production_runtime_factory,
                )

                _STREAMING_RUNTIME_FACTORIES[role] = (
                    make_production_runtime_factory(role.name)
                )
            # Resolve the adapter before native import/allocation so a
            # partially configured deployment fails closed at startup.
            self._install_streaming_runtime(role)
        self._native_restore_enabled = extra(
            "spark_cache_native_restore",
            os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_RESTORE", "0"),
        ) in (1, "1", True, "true")
        self._native_library_path = str(
            extra(
                "spark_cache_native_library",
                os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_LIBRARY", ""),
            )
            or ""
        )
        self._native_library_sha256 = str(
            extra(
                "spark_cache_native_library_sha256",
                os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_LIBRARY_SHA256", ""),
            )
            or ""
        )
        native_arena_raw = str(
            extra(
                "spark_cache_native_arena_bytes",
                os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_ARENA_BYTES", ""),
            )
            or ""
        )
        try:
            self._native_arena_bytes = int(native_arena_raw) if native_arena_raw else 0
        except ValueError as error:
            if self._native_restore_enabled:
                raise RuntimeError(
                    "spark-context-cache: native restore requires an integer arena size"
                ) from error
            self._native_arena_bytes = 0
        native_workers_raw = extra(
            "spark_cache_native_io_workers",
            os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_IO_WORKERS", "8"),
        )
        try:
            self._native_io_workers = int(native_workers_raw)
        except (TypeError, ValueError) as error:
            if self._native_restore_enabled:
                raise RuntimeError(
                    "spark-context-cache: native restore requires an integer"
                    " IO worker count"
                ) from error
            self._native_io_workers = 8
        if self._native_restore_enabled:
            library_path = Path(self._native_library_path)
            if (
                not self._native_library_path
                or not library_path.is_absolute()
                or _SHA256_RE.fullmatch(self._native_library_sha256) is None
                or self._native_arena_bytes not in _NATIVE_ARENA_BYTES
            ):
                raise RuntimeError(
                    "spark-context-cache: native restore requires an"
                    " absolute library path, a 64-character lowercase"
                    " SHA-256, and arena bytes equal to 64, 128, or 256 MiB"
                )
            if not 1 <= self._native_io_workers <= 32:
                raise RuntimeError(
                    "spark-context-cache: native restore IO workers must be in [1, 32]"
                )
        draft_policy = extra(
            "spark_cache_draft_policy",
            os.environ.get("SPARK_CONTEXT_CACHE_DRAFT_POLICY", "colocated_target"),
        )
        target_id = str(
            extra(
                "spark_cache_target_checkpoint_sha256",
                os.environ.get(
                    "SPARK_CONTEXT_CACHE_TARGET_CHECKPOINT_SHA256",
                    "",
                ),
            )
            or ""
        )
        if _SHA256_RE.fullmatch(target_id) is None:
            raise RuntimeError(
                "spark-context-cache: target checkpoint identity must be a"
                " 64-character lowercase SHA-256; set"
                " spark_cache_target_checkpoint_sha256"
            )
        draft_id = str(
            extra(
                "spark_cache_draft_checkpoint_sha256",
                os.environ.get(
                    "SPARK_CONTEXT_CACHE_DRAFT_CHECKPOINT_SHA256",
                    "",
                ),
            )
            or ""
        )
        if str(draft_policy) == "colocated_target":
            if draft_id and draft_id != target_id:
                raise RuntimeError(
                    "spark-context-cache: colocated_target draft state must"
                    " use the target checkpoint identity; omit"
                    " spark_cache_draft_checkpoint_sha256"
                )
            draft_id = target_id
        elif _SHA256_RE.fullmatch(draft_id) is None:
            raise RuntimeError(
                "spark-context-cache: separate draft checkpoint identity must"
                " be a 64-character lowercase SHA-256; set"
                " spark_cache_draft_checkpoint_sha256"
            )
        self._identity_base = dict(
            target_checkpoint=target_id,
            draft_checkpoint=draft_id,
            quantization_layout="nvfp4_ds_mla-per-token-v1",
            rope_layout="glm52-rope-v1",
            tp_degree=parallel.tensor_parallel_size,
            dcp_degree=self._dcp_degree,
            chunk_tokens=CHUNK_TOKENS,
            boundary_hidden_policy="live_forward",
            draft_kv_policy=str(draft_policy),
        )
        # Scheduler-side lookups consult rank 0's shard entries (the driver
        # shares node 0's NVMe). Worker side re-resolves its own rank.
        self._shard_rank = 0
        self._store = ManifestStore(self._root)
        # Worker state.
        self._layer_tensors: dict[str, torch.Tensor] = {}
        self._plans = None
        self._record_kinds: tuple[str, ...] = ()
        # Digests this rank can offer: either discovered from a structurally
        # valid manifest at startup or published by a durable ManifestStore
        # commit. Every restore remains the byte/hash integrity boundary.
        # Reported to the scheduler each step so admission can require
        # unanimity.
        self._held: set[str] = set()
        self._pending_saves: dict[str, dict[str, torch.Tensor]] = {}
        self._load_errors: set[int] = set()
        self._load_lock = threading.Lock()
        self._load_cv = threading.Condition(self._load_lock)
        self._store_cv = threading.Condition(self._load_lock)
        self._store_queue: "queue.SimpleQueue[_StoreSnapshot | None]" = (
            queue.SimpleQueue()
        )
        self._store_thread: threading.Thread | None = None
        self._store_inflight = 0
        self._store_accepting = True
        self._load_queue: "queue.SimpleQueue[_ReqPlan | None]" = queue.SimpleQueue()
        self._load_threads: list[threading.Thread] = []
        self._load_thread_limit = min(
            2,
            max(
                1,
                int(
                    extra(
                        "spark_cache_load_threads",
                        os.environ.get("SPARK_CONTEXT_CACHE_LOAD_THREADS", "1"),
                    )
                ),
            ),
        )
        if self._native_restore_enabled:
            # One native adapter owns one transaction and two arenas. Keep
            # outer restores serialized; its bounded inner preadv/hash pool
            # supplies the desired storage parallelism.
            self._load_thread_limit = 1
        self._inflight_load_reqs: set[str] = set()
        self._finished_load_reqs: set[str] = set()
        self._load_stream: Any = None
        self._native_adapter: Any = None
        self._native_execute_restore: Any = None
        self._native_required_record_mask = 0
        # Scheduler state.
        self._need_load: dict[str, tuple[str, int]] = {}
        # Async loads are parked after block allocation, so they do not
        # appear in the next SchedulerOutput. Carry their allocated blocks
        # across that boundary explicitly.
        self._pending_async_loads: dict[str, tuple[str, int, tuple[int, ...]]] = {}
        # Digests this scheduler has admitted a restore for, so a reported
        # load failure on ANY rank can retire the entry cluster-wide.
        self._admitted: dict[str, str] = {}
        # Quorum map: digest -> ranks that can offer a compatible manifest.
        # A restore is only offered at full quorum; every participating worker
        # then verifies its chunk bytes before writing private request blocks.
        self._quorum: dict[str, set[int]] = {}
        self._store_progress: dict[str, tuple[str, int, int, list[int]]] = {}
        self.counters: dict[str, int] = {
            "store_committed": 0,
            "store_failed": 0,
            "store_skipped_busy": 0,
            "store_skipped_present": 0,
            "store_skipped_quorum": 0,
            "restore_hit": 0,
            "restore_miss_absent": 0,
            "restore_miss_corrupt": 0,
            "restore_miss_incompatible": 0,
            "load_verified": 0,
            "load_failed": 0,
        }
        logger.info(
            "spark-context-cache: role=%s root=%s dcp=%d store=%s restore=%s"
            " native_restore=%s max_span=%d load_threads=%d",
            role.name,
            self._root,
            self._dcp_degree,
            self._store_enabled,
            self._restore_enabled,
            self._native_restore_enabled,
            self._max_span,
            self._load_thread_limit,
        )

    def _install_streaming_runtime(self, role: KVConnectorRole) -> None:
        """Resolve the opt-in runtime without importing a native backend."""

        factory = _STREAMING_RUNTIME_FACTORIES.get(role)
        if factory is None:
            raise RuntimeError(
                "spark-context-cache: streaming snapshots requested but no "
                f"runtime is installed for {role.name.lower()} role"
            )
        try:
            runtime = factory(self)
        except Exception as error:
            raise RuntimeError(
                "spark-context-cache: streaming runtime installation failed closed"
            ) from error
        if runtime is None:
            raise RuntimeError(
                "spark-context-cache: streaming runtime installation failed closed"
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
                "spark-context-cache: streaming "
                f"{role.name.lower()} runtime is incomplete: "
                + ", ".join(missing)
            )
        self._streaming_runtime = runtime

    # ------------------------------------------------------------------
    # identity helpers
    # ------------------------------------------------------------------

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
        return CacheIdentity(
            dcp_shard_rank=shard_rank,
            tp_shard_rank=tp_shard_rank,
            **self._identity_base,
        )

    def _digest(self, token_ids: list[int], span: int) -> str:
        salt = self._identity(0).storage_key
        return context_digest(token_ids[:span], salt)

    def _aligned_span(self, prompt_len: int) -> int:
        span = (prompt_len - 1) // self._block_size * self._block_size
        return span // CHUNK_TOKENS * CHUNK_TOKENS

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

    # ------------------------------------------------------------------
    # scheduler side
    # ------------------------------------------------------------------

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not self._restore_enabled:
            return 0, False
        token_ids = list(request.prompt_token_ids or [])
        span = self._aligned_span(len(token_ids))
        if span < self._min_span or span <= num_computed_tokens:
            return 0, False
        if span > self._max_span:
            self.counters["restore_skip_oversize"] = (
                self.counters.get("restore_skip_oversize", 0) + 1
            )
            return 0, False
        digest = self._digest(token_ids, span)
        # Manifest-only probe: chunk payloads are re-read and re-hashed by
        # every worker at load time, and any worker failure degrades to a
        # rank-synchronous recompute, so hashing ~200 MB here would only
        # add scheduler latency without adding safety.
        if not self._has_full_quorum(digest):
            # Not every rank can offer a compatible manifest (or none has
            # reported yet). Treat as a plain miss: the request re-prefills
            # and republishes, which is also how a corrupted entry retires.
            self.counters["quorum_incomplete"] = (
                self.counters.get("quorum_incomplete", 0) + 1
            )
            return 0, False
        lookup = self._store.lookup(
            self._identity(self._shard_rank), digest, verify_chunks=False
        )
        if not lookup.is_hit:
            self.counters[f"restore_miss_{lookup.reason}"] = (
                self.counters.get(f"restore_miss_{lookup.reason}", 0) + 1
            )
            return 0, False
        self.counters["restore_hit"] += 1
        logger.info(
            "spark-context-cache: scheduler hit digest=%s span=%d (async)",
            digest[:12],
            span,
        )
        self._need_load[request.request_id] = (digest, span)
        while len(self._need_load) > 64:
            self._need_load.pop(next(iter(self._need_load)))
        return span - num_computed_tokens, True

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        request_id = request.request_id
        if num_external_tokens <= 0:
            self._need_load.pop(request_id, None)
            return
        entry = self._need_load.pop(request_id, None)
        if entry is None:
            return
        digest, span = entry
        self._pending_async_loads[request_id] = (
            digest,
            span,
            tuple(blocks.get_block_ids()[0]),
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
            )
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = SparkCacheConnectorMetadata(
            preempted_request_ids=tuple(
                sorted(getattr(scheduler_output, "preempted_req_ids", None) or ())
            )
        )
        for request_id, (digest, span, block_ids) in self._pending_async_loads.items():
            self._admitted[request_id] = digest
            while len(self._admitted) > 8:
                self._admitted.pop(next(iter(self._admitted)))
            meta.plans.append(
                _ReqPlan(request_id, digest, span, block_ids, is_store=False)
            )
        self._pending_async_loads.clear()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            req_id = new_req.req_id
            self._need_load.pop(req_id, None)
            scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            block_ids = tuple(new_req.block_ids[0])
            span = self._aligned_span(len(token_ids))
            if self._store_enabled and self._min_span <= span <= self._max_span:
                digest = self._digest(token_ids, span)
                if self._admitted.get(req_id) == digest:
                    # The restored entry already exists on every rank. A
                    # failed load retires this admission before recompute,
                    # so only verified restores take this fast exit.
                    self._admitted.pop(req_id, None)
                    continue
                if self._has_full_quorum(digest):
                    self.counters["store_skipped_quorum"] += 1
                    continue
                already = new_req.num_computed_tokens + scheduled
                if self._streaming_snapshots_enabled:
                    self._append_streaming_snapshot_offer(
                        meta,
                        request_id=req_id,
                        digest=digest,
                        span=span,
                        promised_completed_tokens=already,
                        block_ids=block_ids,
                    )
                    if already < span:
                        # CachedRequestData.new_block_ids only carries the
                        # current append. Keep the complete table required by
                        # each following offer, including after resume.
                        self._store_progress[req_id] = (
                            digest,
                            span,
                            already,
                            list(block_ids),
                        )
                elif already >= span:
                    meta.plans.append(_ReqPlan(req_id, digest, span, block_ids, True))
                else:
                    # Chunked prefill: CachedRequestData.new_block_ids only
                    # carries each step's appended blocks, so accumulate the
                    # full table here until the span completes.
                    self._store_progress[req_id] = (
                        digest,
                        span,
                        already,
                        list(block_ids),
                    )
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            if req_id not in self._store_progress:
                continue
            digest, span, done, blocks = self._store_progress[req_id]
            new_block_ids = cached.new_block_ids[index]
            appended = list(new_block_ids[0]) if new_block_ids is not None else []
            if req_id in cached.resumed_req_ids:
                blocks = appended
            else:
                blocks = blocks + appended
            done = cached.num_computed_tokens[index] + (
                scheduler_output.num_scheduled_tokens.get(req_id, 0)
            )
            if self._streaming_snapshots_enabled:
                if done >= span:
                    del self._store_progress[req_id]
                    if self._has_full_quorum(digest):
                        self.counters["store_skipped_quorum"] += 1
                        continue
                else:
                    self._store_progress[req_id] = (digest, span, done, blocks)
                self._append_streaming_snapshot_offer(
                    meta,
                    request_id=req_id,
                    digest=digest,
                    span=span,
                    promised_completed_tokens=done,
                    block_ids=blocks,
                )
            elif done >= span:
                del self._store_progress[req_id]
                if self._has_full_quorum(digest):
                    self.counters["store_skipped_quorum"] += 1
                    continue
                meta.plans.append(_ReqPlan(req_id, digest, span, tuple(blocks), True))
            else:
                self._store_progress[req_id] = (digest, span, done, blocks)
        runtime = self._streaming_runtime
        if runtime is not None and self._role is KVConnectorRole.SCHEDULER:
            # request_finished() runs on the scheduler side. Give its adapter
            # the exact offers sent to workers so it delays block reuse only
            # for requests whose final gather may still be in flight.
            runtime.observe_metadata(meta)
        return meta

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

    def handle_preemptions(
        self, kv_connector_metadata: KVConnectorMetadata
    ) -> None:
        runtime = self._streaming_runtime
        if runtime is not None:
            # vLLM calls this before it may overwrite preempted blocks.
            # This must remain synchronous: the runtime drains armed CUDA
            # fences or raises without releasing blocks of unknown status.
            runtime.handle_preemptions(kv_connector_metadata)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._layer_tensors = dict(kv_caches)
        widths = {}
        for name, tensor in sorted(kv_caches.items()):
            if tensor.dim() < 3:
                raise RuntimeError(
                    f"spark-context-cache: layer {name} has unexpected"
                    f" shape {tuple(tensor.shape)}"
                )
            row = tensor[0, 0]
            widths[name] = int(row.numel() * row.element_size())
        logger.info(
            "spark-context-cache: layer inventory (%d layers): %s",
            len(widths),
            {name: widths[name] for name in sorted(widths)},
        )
        draft_named = any(classify_layer(name) == "mtp_draft_kv" for name in widths)
        policy = self._identity_base["draft_kv_policy"]
        if draft_named and policy == "colocated_target":
            raise RuntimeError(
                "spark-context-cache: draft-named cache layers exist but"
                " draft_kv_policy=colocated_target; set"
                " SPARK_CONTEXT_CACHE_DRAFT_POLICY=separate"
            )
        if not draft_named and policy == "separate":
            raise RuntimeError(
                "spark-context-cache: no draft-named cache layers under"
                " draft_kv_policy=separate; GLM MTP drafter layers register"
                " in the shared pool - set"
                " SPARK_CONTEXT_CACHE_DRAFT_POLICY=colocated_target"
            )
        allow = (
            frozenset({"mtp_draft_kv"}) if policy == "colocated_target" else frozenset()
        )
        self._plans = build_layer_plans(widths, allow_missing=allow)
        self._record_kinds = tuple(sorted({plan.record_kind for plan in self._plans}))
        kinds: dict[str, int] = {}
        for plan in self._plans:
            kinds[plan.record_kind] = kinds.get(plan.record_kind, 0) + 1
        logger.info(
            "spark-context-cache: registered %d layers %s policy=%s",
            len(self._plans),
            kinds,
            self._identity_base.get("draft_kv_policy", "separate"),
        )
        if self._native_restore_enabled and self._role is KVConnectorRole.WORKER:
            self._configure_native_restore()
        if self._streaming_snapshots_enabled and self._role is KVConnectorRole.WORKER:
            runtime = self._streaming_runtime
            if runtime is None:
                raise RuntimeError(
                    "spark-context-cache: streaming worker runtime vanished"
                )
            # The adapter closes over this connector. Bind only after the
            # complete tensor inventory and canonical layer plans exist.
            runtime.bind_kv_caches()
        if self._restore_enabled:
            self.discover_manifests()

    def _configure_native_restore(self) -> None:
        """Attest and bind native placement only after final CUDA inventory."""

        if self._native_adapter is not None:
            raise RuntimeError(
                "spark-context-cache: native placement is already configured"
            )
        assert self._plans is not None
        if not self._plans:
            raise RuntimeError("spark-context-cache: native restore has no layer plans")
        first_tensor = self._layer_tensors[self._plans[0].name]
        device = first_tensor.device
        if device.type != "cuda":
            raise RuntimeError(
                "spark-context-cache: native restore requires CUDA cache tensors"
            )
        device_ordinal = (
            int(device.index)
            if device.index is not None
            else int(torch.cuda.current_device())
        )
        slot_capacity = min(
            int(tensor.shape[0]) * int(tensor.shape[1])
            for tensor in self._layer_tensors.values()
        )
        local_rows_per_chunk = CHUNK_TOKENS // self._dcp_degree
        payload_floor = local_rows_per_chunk * (
            4 + sum(plan.bytes_per_token for plan in self._plans)
        )
        max_chunks_per_slab = min(
            4096,
            max(1, self._native_arena_bytes // max(payload_floor, 1) + 1),
        )
        adapter = None
        try:
            components = _load_native_components()
            library = components.NativePlacementLibrary.load(
                self._native_library_path,
                expected_sha256=self._native_library_sha256,
            )
            adapter = components.NativePlacementAdapter.create(
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
            enum_by_kind = {
                "target_ckv": components.RecordKind.TARGET_CKV,
                "sparse_indexer": components.RecordKind.SPARSE_INDEXER,
                "mtp_draft_kv": components.RecordKind.MTP_DRAFT_KV,
            }
            for kind in self._record_kinds:
                required |= 1 << int(enum_by_kind[kind])
            native_execute = components.execute_native_restore
            if not callable(native_execute):
                raise TypeError("native restore orchestrator is not callable")
        except Exception as error:
            if adapter is not None:
                with contextlib.suppress(Exception):
                    adapter.close()
            raise RuntimeError(
                f"spark-context-cache: native restore configuration"
                f" failed closed: {error}"
            ) from error
        self._native_adapter = adapter
        self._native_execute_restore = native_execute
        self._native_required_record_mask = required
        self.counters["native_configured"] = 1
        logger.info(
            "spark-context-cache: native restore configured library=%s"
            " sha256=%s arena_mib=%d destinations=%d slots=%d"
            " max_chunks_per_slab=%d",
            self._native_library_path,
            self._native_library_sha256,
            self._native_arena_bytes // (1024 * 1024),
            len(self._plans),
            slot_capacity,
            max_chunks_per_slab,
        )

    def discover_manifests(self) -> dict[str, int]:
        """Offer structurally valid manifests without reading chunk payloads.

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
                root = Path(self._root) / "manifests" / identity.storage_key
                if root.is_dir():
                    for manifest_path in root.glob("*.json"):
                        digest = manifest_path.stem
                        checked += 1
                        lookup = self._store.lookup(
                            identity,
                            digest,
                            verify_chunks=False,
                            verify_chunk_metadata=True,
                        )
                        if lookup.is_hit:
                            discovered.add(digest)
                        else:
                            rejected += 1
                            # A rejected manifest is not an authority for
                            # deleting content-addressed chunks. Its
                            # descriptors may be corrupt, malicious, or name
                            # chunks shared by a healthy manifest.
                            removed = self._store.invalidate(
                                identity,
                                digest,
                                verify_chunk_payloads=False,
                            )
                            if not removed:
                                try:
                                    manifest_path.unlink()
                                except OSError:
                                    pass
            except (OSError, ValueError, RuntimeError) as error:
                logger.warning(
                    "spark-context-cache: manifest discovery aborted: %s",
                    error,
                )
            self._held = discovered
        if checked:
            logger.info(
                "spark-context-cache: manifest discovery checked=%d"
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
        New commits are published directly after ManifestStore's durable
        manifest commit, startup performs manifest-only discovery, and every
        later load independently re-verifies bytes. Returns
        {"checked": n, "invalidated": m}.
        """
        checked = invalidated = 0
        verified: set[str] = set()
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            root = Path(self._root) / "manifests" / identity.storage_key
            if not root.is_dir():
                with self._load_lock:
                    self._held = set()
                return {"checked": 0, "invalidated": 0}
            for manifest_path in sorted(root.glob("*.json")):
                digest = manifest_path.stem
                checked += 1
                # restore() is the single parallel read/hash/decode pass.
                lookup = self._store.lookup(identity, digest, verify_chunks=False)
                if lookup.is_hit and self._store.restore(lookup) is not None:
                    verified.add(digest)
                    continue
                with self._load_lock:
                    self._held.discard(digest)
                if self._store.invalidate(identity, digest):
                    invalidated += 1
                    logger.warning(
                        "spark-context-cache: sweep invalidated %s (%s)",
                        digest[:12],
                        lookup.reason,
                    )
            with self._load_lock:
                self._held = verified
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("spark-context-cache: sweep aborted: %s", error)
        if checked:
            logger.info(
                "spark-context-cache: sweep checked=%d invalidated=%d",
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
        num_blocks = tensor.shape[0]
        page = tensor.shape[1]
        return tensor.reshape(num_blocks * page, -1)

    def _ensure_load_threads(self) -> None:
        with self._load_lock:
            while len(self._load_threads) < self._load_thread_limit:
                thread = threading.Thread(
                    target=self._load_worker_main,
                    name=f"spark-cache-load-{len(self._load_threads)}",
                    daemon=True,
                )
                thread.start()
                self._load_threads.append(thread)

    def _load_worker_main(self) -> None:
        while True:
            plan = self._load_queue.get()
            if plan is None:
                return
            started = time.perf_counter()
            try:
                verified = self._load_one(plan)
            except Exception as error:  # noqa: BLE001
                logger.warning(
                    "spark-context-cache: load crashed fail-closed: %s", error
                )
                verified = False
            with self._load_cv:
                if verified:
                    self.counters["load_verified"] += 1
                else:
                    # Publish invalid blocks before the finished id while
                    # holding the same lock. vLLM collects finished ids then
                    # load errors in one connector-output pass.
                    self._load_errors.update(plan.block_ids)
                    self.counters["load_failed"] += 1
                self._finished_load_reqs.add(plan.request_id)
                self._inflight_load_reqs.discard(plan.request_id)
                self._load_cv.notify_all()
            if verified:
                logger.info(
                    "spark-context-cache: restored %d tokens async in %.1f ms",
                    plan.span_tokens,
                    1e3 * (time.perf_counter() - started),
                )

    def _load_write_context(self) -> Any:
        assert self._plans is not None
        device = self._layer_tensors[self._plans[0].name].device
        if device.type != "cuda":
            return contextlib.nullcontext(None)
        with self._load_lock:
            if self._load_stream is None:
                self._load_stream = torch.cuda.Stream(device=device)
        return torch.cuda.stream(self._load_stream)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        load_plans = [plan for plan in metadata.plans if not plan.is_store]
        if not load_plans:
            return
        self._ensure_load_threads()
        with self._load_lock:
            self._inflight_load_reqs.update(plan.request_id for plan in load_plans)
        for plan in load_plans:
            self._load_queue.put(plan)

    def _load_one(self, plan: _ReqPlan) -> bool:
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            # Restore is the integrity boundary and re-hashes every chunk in
            # parallel. A full lookup verification here would read/hash/decode
            # the complete context twice before installation.
            lookup = self._store.lookup(identity, plan.digest, verify_chunks=False)
            if not lookup.is_hit:
                logger.warning(
                    "spark-context-cache: worker rank %d miss (%s) for %s",
                    rank,
                    lookup.reason,
                    plan.digest[:12],
                )
                if lookup.reason == "corrupt":
                    self._invalidate_after_failure(plan.digest)
                return False
            if self._native_restore_enabled:
                if (
                    self._native_adapter is None
                    or not callable(self._native_execute_restore)
                    or self._native_required_record_mask == 0
                ):
                    raise RuntimeError(
                        "native restore selected without a configured adapter"
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
                    # The native transaction may already have written some
                    # private restore blocks. Never enter Python assembly or
                    # retry those blocks: retire the entry and publish all of
                    # this request's blocks as invalid for clean recompute.
                    logger.warning(
                        "spark-context-cache: native load failed closed: %s",
                        error,
                    )
                    self._invalidate_after_failure(plan.digest)
                    return False
                self.counters["native_load_verified"] = (
                    self.counters.get("native_load_verified", 0) + 1
                )
                self.counters["native_chunks_verified"] = self.counters.get(
                    "native_chunks_verified", 0
                ) + int(result.verified_chunks)
                logger.info(
                    "spark-context-cache: native restored %d chunks"
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
            chunks = self._store.restore(lookup)
            if chunks is None or len(chunks) != chunk_count(plan.span_tokens):
                self._invalidate_after_failure(plan.digest)
                return False
            positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
            slots = local_slots_for_positions(
                positions, plan.block_ids, self._block_size, self._dcp_degree
            )
            per_chunk = CHUNK_TOKENS // self._dcp_degree
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
            slot_tensor = torch.tensor(slots, dtype=torch.long)
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
            if self._load_stream is not None:
                self._load_stream.synchronize()
            return True
        except (CodecError, KeyError, RuntimeError, ValueError) as error:
            logger.warning("spark-context-cache: load failed fail-closed: %s", error)
            self._invalidate_after_failure(plan.digest)
            return False

    def _invalidate_after_failure(self, digest: str) -> None:
        """Self-heal: drop this rank's damaged entry so the next store can
        republish it. The current request already fell back to recompute."""
        with self._load_lock:
            self._held.discard(digest)
        removed = self._store.invalidate(self._identity(self._worker_rank()), digest)
        if removed:
            with self._load_lock:
                self.counters["invalidated"] = self.counters.get("invalidated", 0) + 1
            logger.warning(
                "spark-context-cache: invalidated damaged entry %s",
                digest[:12],
            )

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        runtime = self._streaming_runtime
        if runtime is None:
            return False, None
        delay_free = bool(
            runtime.request_finished(request.request_id, tuple(block_ids))
        )
        return delay_free, None

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        finished_sending: set[str] = set()
        runtime = self._streaming_runtime
        if runtime is not None:
            finished_sending.update(runtime.take_finished(finished_req_ids))
        with self._load_cv:
            finished_recving = set(self._finished_load_reqs)
            self._finished_load_reqs.clear()
        return finished_sending or None, finished_recving or None

    def wait_for_pending_loads(self, timeout: float | None = None) -> bool:
        with self._load_cv:
            return self._load_cv.wait_for(lambda: not self._inflight_load_reqs, timeout)

    def shutdown(self):
        runtime = self._streaming_runtime
        if runtime is not None:
            # Drain/cancel snapshot leases before any native cache or staging
            # resource can be destroyed. A failure remains fatal and prevents
            # unsafe teardown.
            runtime.shutdown()
            self._streaming_runtime = None
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
                logger.warning(
                    "spark-context-cache: shutdown left saver alive;"
                    " process teardown will reclaim its CPU snapshot"
                )
            else:
                self._store_thread = None
        self.wait_for_pending_loads(timeout=5.0)
        for _ in self._load_threads:
            self._load_queue.put(None)
        deadline = time.monotonic() + 5.0
        for thread in self._load_threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        live_threads = sum(thread.is_alive() for thread in self._load_threads)
        if self._native_adapter is not None and live_threads:
            # A native loader can still own the handle or mapped arenas.
            # Keep the adapter strongly reachable and let process teardown
            # reclaim it; destroying it here would be a use-after-close race.
            self.counters["native_shutdown_handle_leaked"] = (
                self.counters.get("native_shutdown_handle_leaked", 0) + 1
            )
            logger.warning(
                "spark-context-cache: shutdown left %d loader(s) alive;"
                " retaining native placement handle until process exit",
                live_threads,
            )
            return None
        if self._native_adapter is not None:
            self._native_adapter.close()
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

    def wait_for_save(self) -> None:
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
                        "spark-context-cache: streaming worker runtime vanished"
                    )
                if isinstance(metadata, SparkCacheConnectorMetadata):
                    offers = metadata.streaming_snapshot_offers
                    if offers:
                        producer_stream = int(
                            torch.cuda.current_stream().cuda_stream
                        )
                    for offer in offers:
                        runtime.offer_completed(
                            offer,
                            producer_stream=producer_stream,
                        )
                # Poll even for an empty metadata object so background
                # completion, backpressure, and failure state advance on
                # every post-forward callback.
                runtime.poll()
            # Never invoke the legacy CPU snapshot/ManifestStore path under
            # this flag: an incomplete injected runtime must fail closed, not
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
            with self._store_cv:
                if plan.digest in self._held:
                    self.counters["store_skipped_present"] += 1
                    logger.info(
                        "spark-context-cache: store skipped; entry already"
                        " present digest=%s",
                        plan.digest[:12],
                    )
                    continue
                if not self._store_accepting or self._store_inflight:
                    self.counters["store_skipped_busy"] += 1
                    logger.warning(
                        "spark-context-cache: store skipped while saver busy digest=%s",
                        plan.digest[:12],
                    )
                    continue
                self._store_inflight = 1
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
                logger.info(
                    "spark-context-cache: rank %d snapshotted %d tokens"
                    " digest=%s in %.1f ms; queued background commit",
                    snapshot.rank,
                    plan.span_tokens,
                    plan.digest[:12],
                    1e3 * (time.perf_counter() - snapshot_started),
                )
            except Exception as error:  # noqa: BLE001 - enqueue must fail closed
                self._finish_store(
                    plan.digest,
                    committed=False,
                    error=error,
                )

    def _snapshot_store(self, plan: _ReqPlan) -> _StoreSnapshot:
        """Synchronously detach source KV blocks into owned CPU bytes."""

        rank = self._worker_rank()
        identity = self._identity(rank)
        positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
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
                receipt = self._store.commit(
                    identity=snapshot.identity,
                    context_digest=snapshot.plan.digest,
                    chunks=_SnapshotChunks(snapshot, self._dcp_degree),
                )
                logger.info(
                    "spark-context-cache: rank %d committed %d tokens digest=%s"
                    " manifest=%s in %.1f ms",
                    snapshot.rank,
                    receipt.committed_tokens,
                    snapshot.plan.digest[:12],
                    receipt.manifest_digest[:12],
                    1e3 * (time.perf_counter() - commit_started),
                )
            except Exception as error:  # noqa: BLE001 - never kill serving
                self._finish_store(
                    snapshot.plan.digest,
                    committed=False,
                    error=error,
                )
                continue
            self._finish_store(snapshot.plan.digest, committed=True)

    def _finish_store(
        self,
        digest: str,
        *,
        committed: bool,
        error: BaseException | None = None,
    ) -> None:
        with self._store_cv:
            if committed:
                # ManifestStore publishes each fsynced immutable chunk before
                # atomically publishing the fsynced manifest. Load re-verifies
                # every byte, so no completion-time readback or global sweep
                # is needed to make this digest eligible for quorum.
                self._held.add(digest)
                self.counters["store_committed"] += 1
            else:
                self._held.discard(digest)
                self.counters["store_failed"] += 1
            self._store_inflight = 0
            self._store_cv.notify_all()
        if error is not None:
            logger.warning(
                "spark-context-cache: store failed (entry skipped): %s",
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
        receipt = self._store.commit(
            identity=snapshot.identity,
            context_digest=plan.digest,
            chunks=_SnapshotChunks(snapshot, self._dcp_degree),
        )
        logger.info(
            "spark-context-cache: rank %d committed %d tokens digest=%s manifest=%s",
            snapshot.rank,
            receipt.committed_tokens,
            plan.digest[:12],
            receipt.manifest_digest[:12],
        )

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
            held = report.get("held")
            if (
                type(rank) is not int
                or not 0 <= rank < self._tp_degree
                or not isinstance(held, list)
            ):
                continue
            held_set = {d for d in held if isinstance(d, str)}
            for digest in held_set:
                self._quorum.setdefault(digest, set()).add(rank)
            # a digest this rank no longer holds loses its confirmation
            for digest, ranks in list(self._quorum.items()):
                if digest not in held_set and rank in ranks:
                    ranks.discard(rank)
                    if not ranks:
                        self._quorum.pop(digest, None)

    def update_connector_output(self, connector_output: Any) -> None:
        self._absorb_quorum(connector_output)
        invalid = getattr(connector_output, "invalid_block_ids", None)
        if not invalid or not self._admitted:
            return
        # Async failure output reaches this callback before the parked
        # request is rescheduled, so retire the admission before its clean
        # recompute can republish the entry.
        for request_id, digest in list(self._admitted.items()):
            if self._store.invalidate(self._identity(self._shard_rank), digest):
                self.counters["scheduler_retired"] = (
                    self.counters.get("scheduler_retired", 0) + 1
                )
                logger.warning(
                    "spark-context-cache: retired entry %s after a rank"
                    " reported load errors (request %s)",
                    digest[:12],
                    request_id,
                )
            self._quorum.pop(digest, None)
            self._admitted.pop(request_id, None)

    @classmethod
    def build_kv_connector_stats(cls, data=None):
        return SparkCacheStats(data=data if data is not None else {})

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
        """
        if self._role is not KVConnectorRole.WORKER:
            return None
        with self._load_lock:
            held = sorted(self._held)
        report: dict[str, Any] = {
            "rank": self._physical_rank(),
            "held": held,
        }
        runtime = self._streaming_runtime
        status = getattr(runtime, "status", None)
        if self._streaming_snapshots_enabled and callable(status):
            report["streaming"] = status()
        return SparkCacheStats(
            data={
                "reports": [report]
            }
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        with self._load_lock:
            errors, self._load_errors = self._load_errors, set()
        return errors
