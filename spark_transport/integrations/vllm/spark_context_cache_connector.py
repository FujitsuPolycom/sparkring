"""SparkCache: SparkRing persistent NVMe context-cache connector (DCP4-native).

SparkCache is an original vLLM KV-Connector-V1 implementation (NOT the
LMCache project; it rides the same standard connector interface). Its
distinguishing design is per-rank DCP sharding: on a switchless four-Spark
DCP4 ring each rank persists only its own token shard to local NVMe, so a
restore is four parallel local reads with no cross-machine traffic.

KVConnectorBase_V1 implementation that persists each DCP rank's token shard
of every registered cache layer (target CKV incl. per-token NVFP4 scales,
sparse-indexer state, MTP draft KV) to rank-local NVMe through the
fail-closed ManifestStore, and restores it after a runtime restart.

Fail-closed contract:
- Storage uses content-addressed chunks with per-record SHA-256 and an
  identity pinning model/quant/TP/DCP/shard-rank/chunk geometry.
- Any worker-side verification failure at load reports the request's blocks
  through get_block_ids_with_load_errors, which the scheduler converts into
  a full recompute: the entry degrades to a clean, rank-synchronous miss.
- Store failures only log and count; they never fail a request.

Enabled explicitly via --kv-transfer-config naming this module; without
that, serving behavior is byte-identical to v40.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

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

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger("vllm.spark_context_cache")

_RECORD_KINDS = ("target_ckv", "sparse_indexer", "mtp_draft_kv")


@dataclass
class _ReqPlan:
    request_id: str
    digest: str
    span_tokens: int
    block_ids: tuple[int, ...]
    is_store: bool


@dataclass
class SparkCacheConnectorMetadata(KVConnectorMetadata):
    plans: list[_ReqPlan] = field(default_factory=list)


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
        merged: dict[int, list[str]] = {}
        for report in mine + theirs:
            if isinstance(report, dict) and isinstance(report.get("rank"), int):
                merged[report["rank"]] = list(report.get("held", []))
        self.data = {
            "reports": [
                {"rank": rank, "held": merged[rank]} for rank in sorted(merged)
            ]
        }
        return self

    def reduce(self) -> dict[str, int | float]:
        reports = self.data.get("reports", [])
        return {
            "spark_cache_ranks_reporting": len(reports),
            "spark_cache_digests_held": sum(
                len(r.get("held", [])) for r in reports
            ),
        }

    def is_empty(self) -> bool:
        return not self.data.get("reports")


class SparkContextCacheConnector(KVConnectorBase_V1):
    """Store/restore each rank's DCP shard on rank-local NVMe."""

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
        self._dcp_degree = max(
            1, getattr(parallel, "decode_context_parallel_size", 1)
        )
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
        self._store_enabled = extra(
            "spark_cache_store",
            os.environ.get("SPARK_CONTEXT_CACHE_STORE", "1"),
        ) in (1, "1", True, "true")
        self._restore_enabled = extra(
            "spark_cache_restore",
            os.environ.get("SPARK_CONTEXT_CACHE_RESTORE", "1"),
        ) in (1, "1", True, "true")
        target_id = extra(
            "spark_cache_target_id",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_TARGET_ID",
                vllm_config.model_config.model,
            ),
        )
        draft_id = extra(
            "spark_cache_draft_id",
            os.environ.get("SPARK_CONTEXT_CACHE_DRAFT_ID", "mtp-draft"),
        )
        draft_policy = extra(
            "spark_cache_draft_policy",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_DRAFT_POLICY", "colocated_target"
            ),
        )
        self._identity_base = dict(
            target_checkpoint=str(target_id),
            draft_checkpoint=str(draft_id),
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
        # Digests this rank has verified it can serve. Reported to the
        # scheduler each step so admission can require unanimity.
        self._held: set[str] = set()
        self._pending_saves: dict[str, dict[str, torch.Tensor]] = {}
        self._load_errors: set[int] = set()
        # Scheduler state.
        self._need_load: dict[str, tuple[str, int]] = {}
        # Digests this scheduler has admitted a restore for, so a reported
        # load failure on ANY rank can retire the entry cluster-wide.
        self._admitted: dict[str, str] = {}
        # Quorum map: digest -> set of ranks that have confirmed they hold a
        # verified copy. A restore is only offered at full quorum, so a rank
        # whose sweep invalidates an entry silently removes it from service
        # and the request becomes an ordinary miss that re-prefills.
        self._quorum: dict[str, set[int]] = {}
        self._store_progress: dict[str, tuple[str, int, int, list[int]]] = {}
        self.counters: dict[str, int] = {
            "store_committed": 0,
            "store_failed": 0,
            "restore_hit": 0,
            "restore_miss_absent": 0,
            "restore_miss_corrupt": 0,
            "restore_miss_incompatible": 0,
            "load_verified": 0,
            "load_failed": 0,
        }
        logger.info(
            "spark-context-cache: role=%s root=%s dcp=%d store=%s restore=%s",
            role.name,
            self._root,
            self._dcp_degree,
            self._store_enabled,
            self._restore_enabled,
        )

    # ------------------------------------------------------------------
    # identity helpers
    # ------------------------------------------------------------------

    def _identity(self, shard_rank: int) -> CacheIdentity:
        return CacheIdentity(dcp_shard_rank=shard_rank, **self._identity_base)

    def _digest(self, token_ids: list[int], span: int) -> str:
        salt = self._identity(0).storage_key
        return context_digest(token_ids[:span], salt)

    def _aligned_span(self, prompt_len: int) -> int:
        span = (prompt_len - 1) // self._block_size * self._block_size
        return span // CHUNK_TOKENS * CHUNK_TOKENS

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
        digest = self._digest(token_ids, span)
        # Manifest-only probe: chunk payloads are re-read and re-hashed by
        # every worker at load time, and any worker failure degrades to a
        # rank-synchronous recompute, so hashing ~200 MB here would only
        # add scheduler latency without adding safety.
        confirmed = self._quorum.get(digest, set())
        if len(confirmed) < self._dcp_degree:
            # Not every rank has confirmed a verified copy (or none has
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
            "spark-context-cache: scheduler hit digest=%s span=%d",
            digest[:12],
            span,
        )
        self._need_load[request.request_id] = (digest, span)
        return span - num_computed_tokens, False

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        if num_external_tokens <= 0:
            self._need_load.pop(request.request_id, None)

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = SparkCacheConnectorMetadata()
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            req_id = new_req.req_id
            scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
            block_ids = tuple(new_req.block_ids[0])
            if req_id in self._need_load:
                digest, span = self._need_load.pop(req_id)
                self._admitted[req_id] = digest
                while len(self._admitted) > 8:
                    self._admitted.pop(next(iter(self._admitted)))
                meta.plans.append(
                    _ReqPlan(req_id, digest, span, block_ids, is_store=False)
                )
                continue
            span = self._aligned_span(len(token_ids))
            if self._store_enabled and span >= self._min_span:
                digest = self._digest(token_ids, span)
                already = new_req.num_computed_tokens + scheduled
                if already >= span:
                    meta.plans.append(
                        _ReqPlan(req_id, digest, span, block_ids, True)
                    )
                else:
                    # Chunked prefill: CachedRequestData.new_block_ids only
                    # carries each step's appended blocks, so accumulate the
                    # full table here until the span completes.
                    self._store_progress[req_id] = (
                        digest, span, already, list(block_ids)
                    )
        cached = scheduler_output.scheduled_cached_reqs
        for index, req_id in enumerate(cached.req_ids):
            if req_id not in self._store_progress:
                continue
            digest, span, done, blocks = self._store_progress[req_id]
            new_block_ids = cached.new_block_ids[index]
            appended = (
                list(new_block_ids[0]) if new_block_ids is not None else []
            )
            if req_id in cached.resumed_req_ids:
                blocks = appended
            else:
                blocks = blocks + appended
            done = cached.num_computed_tokens[index] + (
                scheduler_output.num_scheduled_tokens.get(req_id, 0)
            )
            if done >= span:
                del self._store_progress[req_id]
                meta.plans.append(
                    _ReqPlan(req_id, digest, span, tuple(blocks), True)
                )
            else:
                self._store_progress[req_id] = (digest, span, done, blocks)
        return meta

    # ------------------------------------------------------------------
    # worker side
    # ------------------------------------------------------------------

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
        draft_named = any(
            classify_layer(name) == "mtp_draft_kv" for name in widths
        )
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
            frozenset({"mtp_draft_kv"})
            if policy == "colocated_target"
            else frozenset()
        )
        self._plans = build_layer_plans(widths, allow_missing=allow)
        self._record_kinds = tuple(
            sorted({plan.record_kind for plan in self._plans})
        )
        kinds: dict[str, int] = {}
        for plan in self._plans:
            kinds[plan.record_kind] = kinds.get(plan.record_kind, 0) + 1
        logger.info(
            "spark-context-cache: registered %d layers %s policy=%s",
            len(self._plans),
            kinds,
            self._identity_base.get("draft_kv_policy", "separate"),
        )
        if self._restore_enabled:
            self.sweep_integrity()

    def sweep_integrity(self) -> dict[str, int]:
        """Verify every entry this rank owns and invalidate damaged ones.

        Runs off the request path (startup and after each store), so disk
        damage is normally discovered and repaired before a request can
        select the entry. Returns {"checked": n, "invalidated": m}.
        """
        checked = invalidated = 0
        verified: set[str] = set()
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            root = Path(self._root) / "manifests" / identity.storage_key
            if not root.is_dir():
                self._held = set()
                return {"checked": 0, "invalidated": 0}
            for manifest_path in sorted(root.glob("*.json")):
                digest = manifest_path.stem
                checked += 1
                lookup = self._store.lookup(identity, digest)
                if lookup.is_hit and self._store.restore(lookup) is not None:
                    verified.add(digest)
                    continue
                self._held.discard(digest)
                if self._store.invalidate(identity, digest):
                    invalidated += 1
                    logger.warning(
                        "spark-context-cache: sweep invalidated %s (%s)",
                        digest[:12],
                        lookup.reason,
                    )
            self._held = verified
        except (OSError, ValueError, RuntimeError) as error:
            logger.warning("spark-context-cache: sweep aborted: %s", error)
        if checked:
            logger.info(
                "spark-context-cache: sweep checked=%d invalidated=%d",
                checked,
                invalidated,
            )
        self.counters["sweep_checked"] = (
            self.counters.get("sweep_checked", 0) + checked
        )
        self.counters["sweep_invalidated"] = (
            self.counters.get("sweep_invalidated", 0) + invalidated
        )
        return {"checked": checked, "invalidated": invalidated}

    def _worker_rank(self) -> int:
        from vllm.distributed import get_dcp_group

        if self._dcp_degree <= 1:
            return 0
        return get_dcp_group().rank_in_group

    def _rows_view(self, tensor: torch.Tensor) -> torch.Tensor:
        num_blocks = tensor.shape[0]
        page = tensor.shape[1]
        return tensor.reshape(num_blocks * page, -1)

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        for plan in metadata.plans:
            if plan.is_store:
                continue
            started = time.perf_counter()
            if not self._load_one(plan):
                self._load_errors.update(plan.block_ids)
                self.counters["load_failed"] += 1
            else:
                self.counters["load_verified"] += 1
                logger.info(
                    "spark-context-cache: restored %d tokens in %.1f ms",
                    plan.span_tokens,
                    1e3 * (time.perf_counter() - started),
                )

    def _load_one(self, plan: _ReqPlan) -> bool:
        try:
            rank = self._worker_rank()
            identity = self._identity(rank)
            lookup = self._store.lookup(identity, plan.digest)
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
            chunks = self._store.restore(lookup)
            if chunks is None or len(chunks) != chunk_count(plan.span_tokens):
                self._invalidate_after_failure(plan.digest)
                return False
            positions = owned_positions(
                plan.span_tokens, self._dcp_degree, rank
            )
            slots = local_slots_for_positions(
                positions, plan.block_ids, self._block_size, self._dcp_degree
            )
            per_chunk = CHUNK_TOKENS // self._dcp_degree
            assert self._plans is not None
            layer_rows: dict[str, list[bytes]] = {
                p.name: [] for p in self._plans
            }
            for index, chunk in enumerate(chunks):
                expected = positions[index * per_chunk:(index + 1) * per_chunk]
                stored = unpack_positions(
                    chunk.records[StateRecord.LOGICAL_POSITIONS]
                )
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
            return True
        except (CodecError, KeyError, RuntimeError, ValueError) as error:
            logger.warning(
                "spark-context-cache: load failed fail-closed: %s", error
            )
            self._invalidate_after_failure(plan.digest)
            return False

    def _invalidate_after_failure(self, digest: str) -> None:
        """Self-heal: drop this rank's damaged entry so the next store can
        republish it. The current request already fell back to recompute."""
        self._held.discard(digest)
        removed = self._store.invalidate(
            self._identity(self._worker_rank()), digest
        )
        if removed:
            self.counters["invalidated"] = (
                self.counters.get("invalidated", 0) + 1
            )
            logger.warning(
                "spark-context-cache: invalidated damaged entry %s",
                digest[:12],
            )

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
        if not isinstance(metadata, SparkCacheConnectorMetadata):
            return
        committed = False
        for plan in metadata.plans:
            if not plan.is_store:
                continue
            try:
                self._store_one(plan)
                self.counters["store_committed"] += 1
            except Exception as error:  # noqa: BLE001 - store must not kill serving
                self.counters["store_failed"] += 1
                logger.warning(
                    "spark-context-cache: store failed (entry skipped): %s",
                    error,
                )
            else:
                committed = True
        if committed and self._restore_enabled:
            self.sweep_integrity()

    def _store_one(self, plan: _ReqPlan) -> None:
        rank = self._worker_rank()
        identity = self._identity(rank)
        positions = owned_positions(plan.span_tokens, self._dcp_degree, rank)
        slots = local_slots_for_positions(
            positions, plan.block_ids, self._block_size, self._dcp_degree
        )
        assert self._plans is not None
        slot_tensor = torch.tensor(slots, dtype=torch.long)
        per_chunk = CHUNK_TOKENS // self._dcp_degree
        layer_bytes: dict[str, bytes] = {}
        for plan_entry in self._plans:
            tensor = self._layer_tensors[plan_entry.name]
            rows = self._rows_view(tensor).view(torch.uint8)
            gathered = rows[slot_tensor.to(rows.device)].cpu().contiguous()
            layer_bytes[plan_entry.name] = gathered.numpy().tobytes()
        chunks = []
        for index in range(chunk_count(plan.span_tokens)):
            row_slice = slice(index * per_chunk, (index + 1) * per_chunk)
            chunk_positions = positions[row_slice]
            records = {
                StateRecord.LOGICAL_POSITIONS: pack_positions(chunk_positions)
            }
            for kind in self._record_kinds:
                rows_map = {}
                for plan_entry in self._plans:
                    if plan_entry.record_kind != kind:
                        continue
                    width = plan_entry.bytes_per_token
                    full = layer_bytes[plan_entry.name]
                    rows_map[plan_entry.name] = full[
                        row_slice.start * width:row_slice.stop * width
                    ]
                records[StateRecord(kind)] = pack_record(
                    self._plans, kind, rows_map, per_chunk
                )
            chunks.append(
                ContextChunk(
                    logical_start=index * CHUNK_TOKENS,
                    logical_end=(index + 1) * CHUNK_TOKENS,
                    records=records,
                )
            )
        receipt = self._store.commit(
            identity=identity, context_digest=plan.digest, chunks=chunks
        )
        logger.info(
            "spark-context-cache: rank %d committed %d tokens digest=%s"
            " manifest=%s",
            rank,
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
            if not isinstance(rank, int) or not isinstance(held, list):
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
        # NOTE: the runtime finishes failed requests before this callback,
        # so entries are retained (bounded) rather than popped on finish.
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
        """Report this rank's verified-servable digests to the scheduler.

        This is the admission quorum's transport: the scheduler will only
        offer a restore that every rank has confirmed, so an entry damaged
        on any single rank stops being offered and the next request simply
        re-prefills instead of failing.
        """
        if not self._held:
            return None
        return SparkCacheStats(
            data={
                "reports": [
                    {
                        "rank": self._worker_rank(),
                        "held": sorted(self._held),
                    }
                ]
            }
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        errors, self._load_errors = self._load_errors, set()
        return errors
