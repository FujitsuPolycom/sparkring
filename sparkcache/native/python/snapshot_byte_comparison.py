"""GPU-free byte oracle for the SparkCache snapshot gather layout.

The reference side uses the same ``pack_record`` function as the current
Python snapshot path. The candidate side emulates the native gather kernel's
layer-major layout, including physical-slot indirection, padded source strides,
record-kind offsets, and 64-byte alignment.

This is not a CUDA correctness claim. It gives the live CUDA probe an exact
fixture and comparison oracle without importing torch, vLLM, or CUDA.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from sparkcache.spark_context_cache_codec import LayerPlan, pack_record


PAYLOAD_ALIGNMENT = 64
MAX_RECORD_KINDS = 4
RECORD_KIND_IDS = {
    "target_ckv": 0,
    "sparse_indexer": 1,
    "mtp_draft_kv": 2,
}


class SnapshotComparisonError(ValueError):
    """A fixture or candidate payload violates the snapshot layout contract."""


@dataclass(frozen=True, slots=True)
class SourceFamilyGeometry:
    """Storage geometry for one native snapshot record family."""

    record_kind: str
    source_count: int
    bytes_per_token: int
    row_stride_bytes: int

    def __post_init__(self) -> None:
        if self.record_kind not in RECORD_KIND_IDS:
            raise SnapshotComparisonError(
                f"unsupported record kind {self.record_kind!r}"
            )
        if self.source_count <= 0:
            raise SnapshotComparisonError("source_count must be positive")
        if self.bytes_per_token <= 0:
            raise SnapshotComparisonError("bytes_per_token must be positive")
        if self.row_stride_bytes < self.bytes_per_token:
            raise SnapshotComparisonError("source stride is smaller than row bytes")


@dataclass(frozen=True, slots=True)
class SnapshotPayloadGeometry:
    row_count: int
    record_mask: int
    record_offsets: tuple[int, ...]
    record_lengths: tuple[int, ...]
    used_bytes: int


GLM52_COLOCATED_SOURCE_FAMILIES = (
    SourceFamilyGeometry(
        record_kind="target_ckv",
        source_count=79,
        bytes_per_token=368,
        row_stride_bytes=368,
    ),
    SourceFamilyGeometry(
        record_kind="sparse_indexer",
        source_count=22,
        bytes_per_token=132,
        row_stride_bytes=132,
    ),
)


@dataclass(frozen=True, slots=True)
class SourceFixture:
    name: str
    record_kind: str
    layer_ordinal: int
    source_rows: int
    row_stride_bytes: int
    bytes_per_token: int
    storage: bytes

    def __post_init__(self) -> None:
        if not self.name:
            raise SnapshotComparisonError("source name must not be empty")
        if self.record_kind not in RECORD_KIND_IDS:
            raise SnapshotComparisonError(
                f"unsupported record kind {self.record_kind!r}"
            )
        if self.layer_ordinal < 0:
            raise SnapshotComparisonError("layer ordinal must be nonnegative")
        if self.source_rows <= 0 or self.bytes_per_token <= 0:
            raise SnapshotComparisonError("source dimensions must be positive")
        if self.row_stride_bytes < self.bytes_per_token:
            raise SnapshotComparisonError("source stride is smaller than row bytes")
        if len(self.storage) != self.source_rows * self.row_stride_bytes:
            raise SnapshotComparisonError("source storage length disagrees with geometry")

    @property
    def plan(self) -> LayerPlan:
        return LayerPlan(self.name, self.record_kind, self.bytes_per_token)


@dataclass(frozen=True, slots=True)
class SnapshotPayloadView:
    row_count: int
    record_mask: int
    record_offsets: tuple[int, ...]
    record_lengths: tuple[int, ...]
    used_bytes: int
    payload: bytes


@dataclass(frozen=True, slots=True)
class RecordComparison:
    record_kind: str
    expected_bytes: int
    expected_sha256: str
    candidate_sha256: str
    first_mismatch: int | None

    @property
    def is_equal(self) -> bool:
        return self.first_mismatch is None


def _align_up(value: int, alignment: int = PAYLOAD_ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


def calculate_payload_geometry(
    families: Sequence[SourceFamilyGeometry],
    *,
    row_count: int,
    slot_bytes: int,
) -> SnapshotPayloadGeometry:
    """Mirror the native layout calculation without allocating source bytes."""

    if not families:
        raise SnapshotComparisonError("source family table must not be empty")
    if row_count <= 0:
        raise SnapshotComparisonError("row_count must be positive")
    if slot_bytes <= 0:
        raise SnapshotComparisonError("slot_bytes must be positive")
    by_kind: dict[str, SourceFamilyGeometry] = {}
    for family in families:
        if family.record_kind in by_kind:
            raise SnapshotComparisonError(
                f"duplicate source family {family.record_kind}"
            )
        by_kind[family.record_kind] = family

    offsets = [0] * MAX_RECORD_KINDS
    lengths = [0] * MAX_RECORD_KINDS
    cursor = 0
    record_mask = 0
    for record_kind, kind_id in sorted(
        RECORD_KIND_IDS.items(), key=lambda item: item[1]
    ):
        family = by_kind.get(record_kind)
        if family is None:
            continue
        cursor = _align_up(cursor)
        record_bytes = (
            row_count * family.bytes_per_token * family.source_count
        )
        if cursor + record_bytes > slot_bytes:
            raise SnapshotComparisonError("candidate payload exceeds arena slot")
        offsets[kind_id] = cursor
        lengths[kind_id] = record_bytes
        record_mask |= 1 << kind_id
        cursor += record_bytes
    return SnapshotPayloadGeometry(
        row_count=row_count,
        record_mask=record_mask,
        record_offsets=tuple(offsets),
        record_lengths=tuple(lengths),
        used_bytes=cursor,
    )


def _validate_source_table(sources: Sequence[SourceFixture]) -> None:
    if not sources:
        raise SnapshotComparisonError("source table must not be empty")
    names: set[str] = set()
    next_ordinal = {kind: 0 for kind in RECORD_KIND_IDS}
    width: dict[str, int] = {}
    for source in sources:
        if source.name in names:
            raise SnapshotComparisonError(f"duplicate source name {source.name}")
        names.add(source.name)
        expected_ordinal = next_ordinal[source.record_kind]
        if source.layer_ordinal != expected_ordinal:
            raise SnapshotComparisonError(
                f"{source.record_kind} ordinal {source.layer_ordinal} "
                f"is not the next dense ordinal {expected_ordinal}"
            )
        next_ordinal[source.record_kind] += 1
        prior_width = width.setdefault(source.record_kind, source.bytes_per_token)
        if prior_width != source.bytes_per_token:
            raise SnapshotComparisonError(
                f"{source.record_kind} layers must share one row width"
            )


def gather_reference_records(
    sources: Sequence[SourceFixture],
    physical_slots: Sequence[int],
) -> Mapping[str, bytes]:
    """Gather through the current Python codec's canonical record packer."""

    _validate_source_table(sources)
    slots = tuple(physical_slots)
    if not slots:
        raise SnapshotComparisonError("physical slot list must not be empty")
    plans = tuple(sorted((source.plan for source in sources), key=lambda item: item.name))
    rows_by_layer: dict[str, bytes] = {}
    for source in sources:
        rows = bytearray()
        for slot in slots:
            if type(slot) is not int or not 0 <= slot < source.source_rows:
                raise SnapshotComparisonError(
                    f"physical slot {slot!r} is outside source {source.name}"
                )
            start = slot * source.row_stride_bytes
            rows.extend(source.storage[start : start + source.bytes_per_token])
        rows_by_layer[source.name] = bytes(rows)
    return {
        kind: pack_record(plans, kind, rows_by_layer, len(slots))
        for kind in RECORD_KIND_IDS
        if any(source.record_kind == kind for source in sources)
    }


def emulate_native_gather(
    sources: Sequence[SourceFixture],
    physical_slots: Sequence[int],
    *,
    slot_bytes: int,
) -> SnapshotPayloadView:
    """Emulate ``gather_snapshot_kernel`` and its payload-layout helper."""

    _validate_source_table(sources)
    slots = tuple(physical_slots)
    if not slots:
        raise SnapshotComparisonError("physical slot list must not be empty")
    if slot_bytes <= 0:
        raise SnapshotComparisonError("slot_bytes must be positive")

    layer_count = [0] * MAX_RECORD_KINDS
    width = [0] * MAX_RECORD_KINDS
    for source in sources:
        kind = RECORD_KIND_IDS[source.record_kind]
        layer_count[kind] += 1
        width[kind] = source.bytes_per_token
        if any(
            type(slot) is not int or not 0 <= slot < source.source_rows
            for slot in slots
        ):
            raise SnapshotComparisonError(
                f"physical slot is outside source {source.name}"
            )

    offsets = [0] * MAX_RECORD_KINDS
    lengths = [0] * MAX_RECORD_KINDS
    cursor = 0
    record_mask = 0
    for kind in range(MAX_RECORD_KINDS):
        if not layer_count[kind]:
            continue
        cursor = _align_up(cursor)
        record_bytes = len(slots) * width[kind] * layer_count[kind]
        if cursor + record_bytes > slot_bytes:
            raise SnapshotComparisonError("candidate payload exceeds arena slot")
        offsets[kind] = cursor
        lengths[kind] = record_bytes
        record_mask |= 1 << kind
        cursor += record_bytes

    # Alignment padding is deliberately nonzero. The native kernel does not
    # define those bytes, and comparisons must only inspect declared records.
    output = bytearray([0xA5]) * cursor
    for source in sources:
        kind = RECORD_KIND_IDS[source.record_kind]
        source_bytes = len(slots) * source.bytes_per_token
        output_base = offsets[kind] + source.layer_ordinal * source_bytes
        for output_row, physical_row in enumerate(slots):
            source_start = physical_row * source.row_stride_bytes
            destination_start = (
                output_base + output_row * source.bytes_per_token
            )
            output[
                destination_start : destination_start + source.bytes_per_token
            ] = source.storage[
                source_start : source_start + source.bytes_per_token
            ]
    return SnapshotPayloadView(
        row_count=len(slots),
        record_mask=record_mask,
        record_offsets=tuple(offsets),
        record_lengths=tuple(lengths),
        used_bytes=cursor,
        payload=bytes(output),
    )


def compare_ready_view(
    sources: Sequence[SourceFixture],
    physical_slots: Sequence[int],
    ready: SnapshotPayloadView,
) -> tuple[RecordComparison, ...]:
    """Compare candidate record slices against the current snapshot codec."""

    expected = gather_reference_records(sources, physical_slots)
    expected_mask = sum(1 << RECORD_KIND_IDS[kind] for kind in expected)
    if ready.row_count != len(physical_slots):
        raise SnapshotComparisonError("candidate row count disagrees with submission")
    if ready.record_mask != expected_mask:
        raise SnapshotComparisonError("candidate record mask disagrees with sources")
    if ready.used_bytes > len(ready.payload):
        raise SnapshotComparisonError("candidate used_bytes exceeds payload")

    results: list[RecordComparison] = []
    for kind, reference in expected.items():
        kind_id = RECORD_KIND_IDS[kind]
        offset = ready.record_offsets[kind_id]
        length = ready.record_lengths[kind_id]
        if length != len(reference) or offset + length > ready.used_bytes:
            raise SnapshotComparisonError(
                f"candidate {kind} range disagrees with reference geometry"
            )
        candidate = ready.payload[offset : offset + length]
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(zip(reference, candidate))
                if left != right
            ),
            None,
        )
        results.append(
            RecordComparison(
                record_kind=kind,
                expected_bytes=len(reference),
                expected_sha256=hashlib.sha256(reference).hexdigest(),
                candidate_sha256=hashlib.sha256(candidate).hexdigest(),
                first_mismatch=mismatch,
            )
        )
    return tuple(results)


def deterministic_source(
    *,
    name: str,
    record_kind: str,
    layer_ordinal: int,
    source_rows: int,
    row_stride_bytes: int,
    bytes_per_token: int,
) -> SourceFixture:
    """Build deterministic rows whose padding cannot alias useful bytes."""

    storage = bytearray([0xEE]) * (source_rows * row_stride_bytes)
    kind_id = RECORD_KIND_IDS[record_kind]
    for row in range(source_rows):
        start = row * row_stride_bytes
        for byte_index in range(bytes_per_token):
            storage[start + byte_index] = (
                kind_id * 71
                + layer_ordinal * 43
                + row * 17
                + byte_index * 29
            ) & 0xFF
    return SourceFixture(
        name=name,
        record_kind=record_kind,
        layer_ordinal=layer_ordinal,
        source_rows=source_rows,
        row_stride_bytes=row_stride_bytes,
        bytes_per_token=bytes_per_token,
        storage=bytes(storage),
    )


def default_fixture() -> tuple[SourceFixture, ...]:
    return (
        deterministic_source(
            name="model.layers.00.mla",
            record_kind="target_ckv",
            layer_ordinal=0,
            source_rows=2048,
            row_stride_bytes=512,
            bytes_per_token=368,
        ),
        deterministic_source(
            name="model.layers.01.mla",
            record_kind="target_ckv",
            layer_ordinal=1,
            source_rows=2048,
            row_stride_bytes=512,
            bytes_per_token=368,
        ),
        deterministic_source(
            name="model.layers.00.indexer",
            record_kind="sparse_indexer",
            layer_ordinal=0,
            source_rows=2048,
            row_stride_bytes=256,
            bytes_per_token=132,
        ),
        deterministic_source(
            name="model.layers.00.mtp",
            record_kind="mtp_draft_kv",
            layer_ordinal=0,
            source_rows=2048,
            row_stride_bytes=512,
            bytes_per_token=368,
        ),
    )


def default_slots(rank: int, row_count: int) -> tuple[int, ...]:
    if not 0 <= rank < 4:
        raise SnapshotComparisonError("fixture rank must be in [0, 4)")
    if row_count <= 0 or row_count > 1024:
        raise SnapshotComparisonError("fixture row_count must be in [1, 1024]")
    # 2039 is prime and below source_rows; each rank gets a deterministic
    # noncontiguous permutation without any out-of-range physical row.
    return tuple((11 + rank * 53 + row * 37) % 2039 for row in range(row_count))


def main() -> int:
    sources = default_fixture()
    summaries = []
    passed = True
    for rank in range(4):
        slots = default_slots(rank, 64)
        ready = emulate_native_gather(sources, slots, slot_bytes=2 << 20)
        comparisons = compare_ready_view(sources, slots, ready)
        passed = passed and all(item.is_equal for item in comparisons)
        summaries.append(
            {
                "rank": rank,
                "rows": len(slots),
                "record_mask": ready.record_mask,
                "record_offsets": ready.record_offsets,
                "record_lengths": ready.record_lengths,
                "used_bytes": ready.used_bytes,
                "records": [
                    {
                        "kind": item.record_kind,
                        "bytes": item.expected_bytes,
                        "sha256": item.expected_sha256,
                        "first_mismatch": item.first_mismatch,
                    }
                    for item in comparisons
                ],
            }
        )
    print(json.dumps({"passed": passed, "ranks": summaries}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
