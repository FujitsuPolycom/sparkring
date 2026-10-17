"""Streaming-snapshot adapter for the registered GLM-5.2 row layout."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterator, Mapping, Sequence

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ContextChunk,
    StateRecord,
)
from sparkcache.spark_context_cache_codec import CHUNK_TOKENS, pack_positions
from sparkcache.streaming.native_ring import (
    NativeRingConfig,
    NativeSnapshotRing,
    SnapshotSourceSpec,
    SnapshotView,
)
from sparkcache.streaming.publisher import SnapshotTranslationError

ARENA_MODE_MAPPED_HOST = 1
RING_DEPTH = 2
SLOT_BYTES = 64 * 1024 * 1024
MACRO_ROWS = 1024
DCP_DEGREE = 4
ROWS_PER_CHUNK = CHUNK_TOKENS // DCP_DEGREE
TARGET_LAYERS = 79
TARGET_BYTES_PER_TOKEN = 368
INDEXER_LAYERS = 22
INDEXER_BYTES_PER_TOKEN = 132
SOURCE_COUNT = TARGET_LAYERS + INDEXER_LAYERS
MACRO_PAYLOAD_BYTES = MACRO_ROWS * (
    TARGET_LAYERS * TARGET_BYTES_PER_TOKEN
    + INDEXER_LAYERS * INDEXER_BYTES_PER_TOKEN
)
CHUNKS_PER_BATCH = 16
MAX_SESSIONS = 8

_TARGET_KIND = 0
_INDEXER_KIND = 1


@dataclass(frozen=True, slots=True)
class Glm52LayerOrder:
    name: str
    record_kind: int
    source_ordinal: int
    bytes_per_token: int


# The profile adapter owns the exact C++/CUDA and persistent record order.
# Target sources contain colocated draft state, so no separate draft record is
# present in this layout.
LAYER_ORDER = tuple(
    Glm52LayerOrder(
        name=f"model.layers.{ordinal:02d}.mla",
        record_kind=_TARGET_KIND,
        source_ordinal=ordinal,
        bytes_per_token=TARGET_BYTES_PER_TOKEN,
    )
    for ordinal in range(TARGET_LAYERS)
) + tuple(
    Glm52LayerOrder(
        name=f"model.layers.{ordinal:02d}.indexer",
        record_kind=_INDEXER_KIND,
        source_ordinal=ordinal,
        bytes_per_token=INDEXER_BYTES_PER_TOKEN,
    )
    for ordinal in range(INDEXER_LAYERS)
)


class Glm52ReadyViewTranslator:
    """Translate the registered DCP4 row layout into canonical chunks."""

    def __init__(
        self,
        sources: Sequence[SnapshotSourceSpec],
        *,
        dcp_rank: int,
        context_chunk_factory: Callable[..., ContextChunk] = ContextChunk,
        state_record_type: type[StateRecord] = StateRecord,
        pack_positions_function: Callable[[Sequence[int]], bytes] = pack_positions,
    ) -> None:
        if not 0 <= dcp_rank < DCP_DEGREE:
            raise ValueError("dcp_rank must be in [0, 4)")
        self.dcp_degree = DCP_DEGREE
        self.dcp_rank = dcp_rank
        self._sources = tuple(sources)
        self._context_chunk_factory = context_chunk_factory
        self._state_record_type = state_record_type
        self._pack_positions = pack_positions_function
        self._validate_source_order()

    @classmethod
    def for_ring(
        cls,
        ring: NativeSnapshotRing,
        *,
        dcp_rank: int,
        context_chunk_factory: Callable[..., ContextChunk] = ContextChunk,
        state_record_type: type[StateRecord] = StateRecord,
        pack_positions_function: Callable[[Sequence[int]], bytes] = pack_positions,
    ) -> "Glm52ReadyViewTranslator":
        config = ring.config
        if (
            config.arena_mode != ARENA_MODE_MAPPED_HOST
            or config.slot_count != RING_DEPTH
            or config.slot_bytes != SLOT_BYTES
            or config.max_sources < SOURCE_COUNT
            or config.max_rows < MACRO_ROWS
        ):
            raise SnapshotTranslationError(
                "ring configuration does not match the GLM-5.2 streaming profile"
            )
        sources = ring.configured_sources
        if sources is None:
            raise SnapshotTranslationError(
                "ring source inventory must be configured before translation"
            )
        return cls(
            sources,
            dcp_rank=dcp_rank,
            context_chunk_factory=context_chunk_factory,
            state_record_type=state_record_type,
            pack_positions_function=pack_positions_function,
        )

    def batch_logical_tokens(self, view: SnapshotView) -> int:
        rows = view.ticket.row_count
        if rows <= 0 or rows % ROWS_PER_CHUNK:
            raise SnapshotTranslationError(
                "READY row count must cover complete 256-token chunks"
            )
        return rows * self.dcp_degree

    def iter_chunks(self, view: SnapshotView) -> Iterator[ContextChunk]:
        logical_tokens = self.batch_logical_tokens(view)
        logical_start = view.ticket.logical_start
        if logical_start % CHUNK_TOKENS:
            raise SnapshotTranslationError(
                "READY logical_start must be 256-token aligned"
            )
        logical_end = logical_start + logical_tokens
        if logical_end > 0x100000000:
            raise SnapshotTranslationError(
                "logical positions exceed the uint32 storage ABI"
            )

        rows = view.ticket.row_count
        expected_target = rows * TARGET_LAYERS * TARGET_BYTES_PER_TOKEN
        expected_indexer = rows * INDEXER_LAYERS * INDEXER_BYTES_PER_TOKEN
        if (
            view.record_mask != 0b011
            or view.record_lengths[_TARGET_KIND] != expected_target
            or view.record_lengths[_INDEXER_KIND] != expected_indexer
            or view.record_lengths[2:] != (0, 0)
        ):
            raise SnapshotTranslationError(
                "READY records do not match the GLM-5.2 streaming profile"
            )

        for batch_row_start in range(0, rows, ROWS_PER_CHUNK):
            chunk_start = logical_start + batch_row_start * self.dcp_degree
            chunk_end = chunk_start + CHUNK_TOKENS
            positions = tuple(
                range(
                    chunk_start + self.dcp_rank,
                    chunk_end,
                    self.dcp_degree,
                )
            )
            state_record = self._state_record_type
            records = {
                state_record.LOGICAL_POSITIONS: self._pack_positions(positions),
                state_record.TARGET_CKV: self._copy_chunk_record(
                    view,
                    record_kind=_TARGET_KIND,
                    layer_count=TARGET_LAYERS,
                    width=TARGET_BYTES_PER_TOKEN,
                    batch_rows=rows,
                    batch_row_start=batch_row_start,
                ),
                state_record.SPARSE_INDEXER: self._copy_chunk_record(
                    view,
                    record_kind=_INDEXER_KIND,
                    layer_count=INDEXER_LAYERS,
                    width=INDEXER_BYTES_PER_TOKEN,
                    batch_rows=rows,
                    batch_row_start=batch_row_start,
                ),
            }
            yield self._context_chunk_factory(
                logical_start=chunk_start,
                logical_end=chunk_end,
                records=records,
            )

    def _validate_source_order(self) -> None:
        if len(self._sources) != len(LAYER_ORDER):
            raise SnapshotTranslationError(
                "GLM-5.2 streaming inventory must contain 79 target and "
                "22 indexer layers"
            )
        for source, expected in zip(self._sources, LAYER_ORDER, strict=True):
            if (
                source.record_kind != expected.record_kind
                or source.source_layer_ordinal != expected.source_ordinal
                or source.bytes_per_token != expected.bytes_per_token
            ):
                raise SnapshotTranslationError(
                    f"source inventory disagrees at {expected.name}"
                )

    @staticmethod
    def _copy_chunk_record(
        view: SnapshotView,
        *,
        record_kind: int,
        layer_count: int,
        width: int,
        batch_rows: int,
        batch_row_start: int,
    ) -> bytes:
        record = view.record(record_kind)
        layer_bytes = batch_rows * width
        chunk_layer_bytes = ROWS_PER_CHUNK * width
        parts: list[memoryview] = []
        try:
            for layer in range(layer_count):
                start = layer * layer_bytes + batch_row_start * width
                parts.append(record[start : start + chunk_layer_bytes])
            return b"".join(parts)
        finally:
            for part in parts:
                part.release()
            record.release()


@dataclass(frozen=True, slots=True)
class Glm52SourceInventory:
    """C++/CUDA descriptors and row views retained for the ring lifetime."""

    sources: tuple[SnapshotSourceSpec, ...]
    retained_row_views: tuple[Any, ...]
    device_ordinal: int


def _is_contiguous(value: Any) -> bool:
    check = getattr(value, "is_contiguous", None)
    return bool(callable(check) and check())


def build_source_inventory(connector: Any) -> Glm52SourceInventory:
    """Validate registered profile tensors without allowing reshape copies."""

    plans = tuple(connector._plans or ())
    tensors: Mapping[str, Any] = connector._layer_tensors
    if len(plans) != SOURCE_COUNT or set(tensors) != {
        plan.name for plan in plans
    }:
        raise RuntimeError(
            "GLM-5.2 streaming requires exactly 101 registered cache layers"
        )
    grouped = {
        "target_ckv": tuple(
            sorted(
                (plan for plan in plans if plan.record_kind == "target_ckv"),
                key=lambda plan: plan.name,
            )
        ),
        "sparse_indexer": tuple(
            sorted(
                (
                    plan
                    for plan in plans
                    if plan.record_kind == "sparse_indexer"
                ),
                key=lambda plan: plan.name,
            )
        ),
    }
    if any(plan.record_kind not in grouped for plan in plans):
        raise RuntimeError(
            "GLM-5.2 streaming requires colocated draft state without a "
            "separate draft cache record"
        )
    if (
        len(grouped["target_ckv"]) != TARGET_LAYERS
        or {plan.bytes_per_token for plan in grouped["target_ckv"]}
        != {TARGET_BYTES_PER_TOKEN}
        or len(grouped["sparse_indexer"]) != INDEXER_LAYERS
        or {plan.bytes_per_token for plan in grouped["sparse_indexer"]}
        != {INDEXER_BYTES_PER_TOKEN}
    ):
        raise RuntimeError(
            "GLM-5.2 streaming requires 79x368-byte target and "
            "22x132-byte indexer sources"
        )

    sources: list[SnapshotSourceSpec] = []
    retained: list[Any] = []
    device_ordinal: int | None = None
    for record_kind, kind_name in enumerate(("target_ckv", "sparse_indexer")):
        for ordinal, plan in enumerate(grouped[kind_name]):
            tensor = tensors[plan.name]
            rows = connector._rows_view(tensor)
            tensor_device = getattr(tensor, "device", None)
            rows_device = getattr(rows, "device", None)
            shape = tuple(getattr(rows, "shape", ()))
            element_size_call = getattr(rows, "element_size", None)
            stride_call = getattr(rows, "stride", None)
            tensor_pointer_call = getattr(tensor, "data_ptr", None)
            rows_pointer_call = getattr(rows, "data_ptr", None)
            if (
                getattr(tensor_device, "type", None) != "cuda"
                or getattr(rows_device, "type", None) != "cuda"
                or getattr(tensor_device, "index", None)
                != getattr(rows_device, "index", None)
                or not _is_contiguous(tensor)
                or not _is_contiguous(rows)
                or len(shape) != 2
                or shape[0] < MACRO_ROWS
                or not callable(element_size_call)
                or not callable(stride_call)
                or not callable(tensor_pointer_call)
                or not callable(rows_pointer_call)
            ):
                raise RuntimeError(
                    f"layer {plan.name} is not a contiguous aliasing row view"
                )
            element_size = int(element_size_call())
            row_width = int(shape[1]) * element_size
            stride_bytes = int(stride_call(0)) * element_size
            tensor_pointer = int(tensor_pointer_call())
            rows_pointer = int(rows_pointer_call())
            if (
                element_size <= 0
                or row_width != plan.bytes_per_token
                or stride_bytes != row_width
                or rows_pointer != tensor_pointer
                or rows_pointer <= 0
            ):
                raise RuntimeError(
                    f"layer {plan.name} is not a contiguous aliasing row view"
                )
            index = getattr(rows_device, "index", None)
            current_device = 0 if index is None else int(index)
            if device_ordinal is None:
                device_ordinal = current_device
            elif current_device != device_ordinal:
                raise RuntimeError(
                    "streaming snapshot sources span multiple CUDA devices"
                )
            sources.append(
                SnapshotSourceSpec(
                    source_base=rows_pointer,
                    source_rows=int(shape[0]),
                    source_row_stride_bytes=stride_bytes,
                    bytes_per_token=plan.bytes_per_token,
                    record_kind=record_kind,
                    source_layer_ordinal=ordinal,
                )
            )
            # Retain the exact alias passed to C++/CUDA code. A later Python
            # collection must not invalidate a descriptor owned by the ring.
            retained.append(rows)
    assert device_ordinal is not None
    return Glm52SourceInventory(
        sources=tuple(sources),
        retained_row_views=tuple(retained),
        device_ordinal=device_ordinal,
    )


@dataclass(frozen=True, slots=True)
class Glm52StreamingProfileAdapter:
    """Factory inputs owned by the GLM-5.2 streaming profile."""

    ring_depth: int = RING_DEPTH
    chunks_per_batch: int = CHUNKS_PER_BATCH
    max_sessions: int = MAX_SESSIONS
    dcp_degree: int = DCP_DEGREE
    max_rows: int = MACRO_ROWS
    slot_bytes: int = SLOT_BYTES
    arena_mode_name: str = "mapped_host"

    def validate_connector(self, connector: Any) -> None:
        if (
            getattr(connector, "_dcp_degree", None) != DCP_DEGREE
            or getattr(connector, "_block_size", 0) <= 0
            or getattr(connector, "_identity_base", {}).get("draft_kv_policy")
            != "colocated_target"
        ):
            raise RuntimeError(
                "GLM-5.2 streaming requires DCP4 and colocated draft state"
            )

    def validate_identity(self, identity: CacheIdentity) -> None:
        if (
            identity.chunk_tokens != CHUNK_TOKENS
            or identity.dcp_degree != DCP_DEGREE
            or not 0 <= identity.dcp_shard_rank < DCP_DEGREE
            or identity.boundary_hidden_policy != "live_forward"
            or identity.draft_kv_policy != "colocated_target"
        ):
            raise RuntimeError(
                "GLM-5.2 streaming identity requires DCP4, 256-token chunks, "
                "live boundary state, and colocated draft state"
            )

    def validate_rank(self, rank: int) -> None:
        if not 0 <= rank < DCP_DEGREE:
            raise RuntimeError("streaming snapshot DCP rank is outside [0, 4)")

    def build_source_inventory(self, connector: Any) -> Glm52SourceInventory:
        return build_source_inventory(connector)

    def build_ring_config(self, device_ordinal: int) -> NativeRingConfig:
        return NativeRingConfig(
            arena_mode=ARENA_MODE_MAPPED_HOST,
            slot_bytes=SLOT_BYTES,
            slot_count=RING_DEPTH,
            max_sources=SOURCE_COUNT,
            max_rows=MACRO_ROWS,
            device_ordinal=device_ordinal,
        )

    def build_translator(
        self,
        ring: NativeSnapshotRing,
        *,
        dcp_rank: int,
        context_chunk_factory: Callable[..., ContextChunk],
        state_record_type: type[StateRecord],
        pack_positions_function: Callable[[Sequence[int]], bytes],
    ) -> Glm52ReadyViewTranslator:
        return Glm52ReadyViewTranslator.for_ring(
            ring,
            dcp_rank=dcp_rank,
            context_chunk_factory=context_chunk_factory,
            state_record_type=state_record_type,
            pack_positions_function=pack_positions_function,
        )

    def bound_status(self) -> dict[str, int]:
        """Describe the registered source layout for diagnostics."""

        return {
            "target_sources": TARGET_LAYERS,
            "target_bytes_per_token": TARGET_BYTES_PER_TOKEN,
            "indexer_sources": INDEXER_LAYERS,
            "indexer_bytes_per_token": INDEXER_BYTES_PER_TOKEN,
            "record_mask": 0b011,
            "max_rows": MACRO_ROWS,
        }


GLM52_STREAMING_ADAPTER = Glm52StreamingProfileAdapter()


__all__ = [
    "GLM52_STREAMING_ADAPTER",
    "Glm52ReadyViewTranslator",
    "Glm52SourceInventory",
    "Glm52StreamingProfileAdapter",
    "MACRO_PAYLOAD_BYTES",
    "build_source_inventory",
]
