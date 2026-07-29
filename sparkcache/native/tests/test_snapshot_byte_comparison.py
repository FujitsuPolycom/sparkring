from __future__ import annotations

import dataclasses

import pytest

from sparkcache.native.python.snapshot_byte_comparison import (
    GLM52_COLOCATED_SOURCE_FAMILIES,
    SnapshotComparisonError,
    calculate_payload_geometry,
    compare_ready_view,
    default_fixture,
    default_slots,
    emulate_native_gather,
    gather_reference_records,
)


@pytest.mark.parametrize(
    "row_count,offsets,lengths,used_bytes",
    [
        (
            64,
            (0, 1_860_608, 0, 0),
            (1_860_608, 185_856, 0, 0),
            2_046_464,
        ),
        (
            1024,
            (0, 29_769_728, 0, 0),
            (29_769_728, 2_973_696, 0, 0),
            32_743_424,
        ),
    ],
)
def test_production_colocated_glm52_geometry_is_exact(
    row_count: int,
    offsets: tuple[int, ...],
    lengths: tuple[int, ...],
    used_bytes: int,
) -> None:
    geometry = calculate_payload_geometry(
        GLM52_COLOCATED_SOURCE_FAMILIES,
        row_count=row_count,
        slot_bytes=32 << 20,
    )

    assert [
        (
            item.record_kind,
            item.source_count,
            item.bytes_per_token,
            item.row_stride_bytes,
        )
        for item in GLM52_COLOCATED_SOURCE_FAMILIES
    ] == [
        ("target_ckv", 79, 368, 368),
        ("sparse_indexer", 22, 132, 132),
    ]
    assert geometry.record_mask == 0b011
    assert geometry.record_offsets == offsets
    assert geometry.record_lengths == lengths
    assert geometry.used_bytes == used_bytes


def test_production_1024_row_fixture_has_only_792_kib_of_32_mib_headroom() -> None:
    geometry = calculate_payload_geometry(
        GLM52_COLOCATED_SOURCE_FAMILIES,
        row_count=1024,
        slot_bytes=32 << 20,
    )

    assert (32 << 20) - geometry.used_bytes == 811_008
    with pytest.raises(SnapshotComparisonError, match="exceeds arena slot"):
        calculate_payload_geometry(
            GLM52_COLOCATED_SOURCE_FAMILIES,
            row_count=1024,
            slot_bytes=geometry.used_bytes - 1,
        )


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("row_count", (1, 63, 64, 1024))
def test_all_record_families_match_at_scrambled_slots(
    rank: int,
    row_count: int,
) -> None:
    sources = default_fixture()
    slots = default_slots(rank, row_count)

    ready = emulate_native_gather(sources, slots, slot_bytes=2 << 20)
    comparisons = compare_ready_view(sources, slots, ready)

    assert [item.record_kind for item in comparisons] == [
        "target_ckv",
        "sparse_indexer",
        "mtp_draft_kv",
    ]
    assert all(item.is_equal for item in comparisons)
    assert ready.record_mask == 0b111
    assert ready.record_offsets[0] == 0
    assert ready.record_offsets[1] % 64 == 0
    assert ready.record_offsets[2] % 64 == 0


def test_padding_bytes_are_not_part_of_any_record_comparison() -> None:
    sources = default_fixture()
    slots = default_slots(0, 63)
    ready = emulate_native_gather(sources, slots, slot_bytes=2 << 20)
    references = gather_reference_records(sources, slots)

    claimed = sum(len(payload) for payload in references.values())
    assert ready.used_bytes > claimed
    comparisons = compare_ready_view(sources, slots, ready)
    assert all(item.is_equal for item in comparisons)


@pytest.mark.parametrize(
    "row_count,offsets,lengths,used_bytes",
    [
        (64, (0, 47_104, 55_552, 0), (47_104, 8_448, 23_552, 0), 79_104),
        (
            1024,
            (0, 753_664, 888_832, 0),
            (753_664, 135_168, 376_832, 0),
            1_265_664,
        ),
    ],
)
def test_documented_live_fixture_geometry_is_exact(
    row_count: int,
    offsets: tuple[int, ...],
    lengths: tuple[int, ...],
    used_bytes: int,
) -> None:
    ready = emulate_native_gather(
        default_fixture(),
        default_slots(0, row_count),
        slot_bytes=2 << 20,
    )

    assert ready.record_offsets == offsets
    assert ready.record_lengths == lengths
    assert ready.used_bytes == used_bytes


def test_one_corrupted_mtp_byte_reports_exact_record_offset() -> None:
    sources = default_fixture()
    slots = default_slots(2, 64)
    ready = emulate_native_gather(sources, slots, slot_bytes=2 << 20)
    payload = bytearray(ready.payload)
    corruption_offset = ready.record_offsets[2] + 17
    payload[corruption_offset] ^= 0x01
    corrupted = dataclasses.replace(ready, payload=bytes(payload))

    comparisons = compare_ready_view(sources, slots, corrupted)
    by_kind = {item.record_kind: item for item in comparisons}
    assert by_kind["target_ckv"].is_equal
    assert by_kind["sparse_indexer"].is_equal
    assert by_kind["mtp_draft_kv"].first_mismatch == 17


def test_candidate_geometry_mismatch_fails_before_byte_comparison() -> None:
    sources = default_fixture()
    slots = default_slots(0, 64)
    ready = emulate_native_gather(sources, slots, slot_bytes=2 << 20)

    with pytest.raises(SnapshotComparisonError, match="record mask"):
        compare_ready_view(
            sources,
            slots,
            dataclasses.replace(ready, record_mask=0b011),
        )
    with pytest.raises(SnapshotComparisonError, match="geometry"):
        lengths = list(ready.record_lengths)
        lengths[1] -= 1
        compare_ready_view(
            sources,
            slots,
            dataclasses.replace(ready, record_lengths=tuple(lengths)),
        )


def test_source_stride_padding_never_enters_packed_records() -> None:
    sources = default_fixture()
    slots = default_slots(3, 64)
    references = gather_reference_records(sources, slots)
    changed_padding = []
    for source in sources:
        storage = bytearray(source.storage)
        for row in range(source.source_rows):
            padding_start = row * source.row_stride_bytes + source.bytes_per_token
            padding_end = (row + 1) * source.row_stride_bytes
            storage[padding_start:padding_end] = b"\x11" * (
                padding_end - padding_start
            )
        changed_padding.append(dataclasses.replace(source, storage=bytes(storage)))

    assert gather_reference_records(changed_padding, slots) == references
    ready = emulate_native_gather(changed_padding, slots, slot_bytes=2 << 20)
    assert all(
        item.is_equal
        for item in compare_ready_view(changed_padding, slots, ready)
    )
