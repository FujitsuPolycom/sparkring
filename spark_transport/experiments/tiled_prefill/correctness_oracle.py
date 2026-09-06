"""Exact BF16 oracle for the research-only tiled-prefill probe.

The generated inputs are exactly representable integer BF16 values.  Their
seven-generation period differs from the eight-slot transport window, so a
physical slot cannot retain a valid-looking value when it is reused.  The two
tensor halves intentionally use different reduction trees and therefore have
different bit-exact expected results.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import struct


INPUT_GENERATION_PERIOD = 7
PAYLOAD_TILE_BYTES = 512 * 1024
MODEL_WIDTH = 6144
BF16_BYTES = 2
GUARD_BYTES = 4096
INPUT_GUARD_SENTINEL = 0xA5
OUTPUT_GUARD_SENTINEL = 0x5A
INACTIVE_INPUT_SENTINEL = 0xD3
INACTIVE_OUTPUT_SENTINEL = 0x6C


class TiledHalfAssociation(Enum):
    """The rank-pair order used for one tensor half."""

    XOR1_THEN_XOR3 = "xor1_then_xor3"
    XOR3_THEN_XOR1 = "xor3_then_xor1"


_RANK_BASE_BF16 = (0x3F80, 0x4000, 0x4040, 0x4380)  # 1, 2, 3, 256


@dataclass(frozen=True)
class PayloadGeometry:
    """Active and rounded allocation sizes for one probe operation."""

    active_bytes: int
    capacity_bytes: int
    inactive_bytes: int


@dataclass
class GuardedBuffers:
    """Input and output allocations including their prefix/suffix guards."""

    input: bytearray
    output: bytearray
    guard_bytes: int


@dataclass(frozen=True)
class OracleTile:
    """One active tile range and the physical-slot generation that owns it."""

    input_offset_bytes: int
    output_offset_bytes: int
    active_bytes: int
    generation: int


@dataclass(frozen=True)
class SentinelCounters:
    """Qualification ABI counters produced by inactive-capacity checks."""

    input_guard_corruptions: int
    output_guard_corruptions: int
    inactive_input_sentinel_corruptions: int
    inactive_output_sentinel_corruptions: int

    def as_receipt_fields(self) -> dict[str, int]:
        return {
            "input_guard_corruptions": self.input_guard_corruptions,
            "output_guard_corruptions": self.output_guard_corruptions,
            "inactive_input_sentinel_corruptions": (
                self.inactive_input_sentinel_corruptions
            ),
            "inactive_output_sentinel_corruptions": (
                self.inactive_output_sentinel_corruptions
            ),
        }


def payload_geometry(query_rows: int) -> PayloadGeometry:
    """Return the exact active and tile-rounded BF16 payload sizes."""

    if query_rows <= 0:
        raise ValueError("query_rows must be positive")
    active_bytes = query_rows * MODEL_WIDTH * BF16_BYTES
    tile_count = (
        active_bytes + PAYLOAD_TILE_BYTES - 1
    ) // PAYLOAD_TILE_BYTES
    capacity_bytes = tile_count * PAYLOAD_TILE_BYTES
    return PayloadGeometry(
        active_bytes=active_bytes,
        capacity_bytes=capacity_bytes,
        inactive_bytes=capacity_bytes - active_bytes,
    )


def initialize_guarded_buffers(
    geometry: PayloadGeometry, *, guard_bytes: int
) -> GuardedBuffers:
    """Fill guards and all payload capacity with distinct sentinels."""

    if guard_bytes <= 0:
        raise ValueError("guard_bytes must be positive")
    input_buffer = (
        bytearray([INPUT_GUARD_SENTINEL]) * guard_bytes
        + bytearray([INACTIVE_INPUT_SENTINEL]) * geometry.capacity_bytes
        + bytearray([INPUT_GUARD_SENTINEL]) * guard_bytes
    )
    output_buffer = (
        bytearray([OUTPUT_GUARD_SENTINEL]) * guard_bytes
        + bytearray([INACTIVE_OUTPUT_SENTINEL]) * geometry.capacity_bytes
        + bytearray([OUTPUT_GUARD_SENTINEL]) * guard_bytes
    )
    return GuardedBuffers(
        input=input_buffer,
        output=output_buffer,
        guard_bytes=guard_bytes,
    )


def _byte_mismatches(
    buffer: bytearray, expected: int, begin: int, end: int
) -> int:
    return end - begin - buffer.count(expected, begin, end)


def validate_sentinels(
    buffers: GuardedBuffers, geometry: PayloadGeometry
) -> SentinelCounters:
    """Count guard and inactive-capacity byte corruptions exactly."""

    guard_bytes = buffers.guard_bytes
    allocation_bytes = geometry.capacity_bytes + 2 * guard_bytes
    if len(buffers.input) != allocation_bytes or len(buffers.output) != (
        allocation_bytes
    ):
        raise ValueError("guarded buffer allocation size disagrees with geometry")
    payload_begin = guard_bytes
    inactive_begin = payload_begin + geometry.active_bytes
    payload_end = payload_begin + geometry.capacity_bytes
    input_guard_corruptions = _byte_mismatches(
        buffers.input, INPUT_GUARD_SENTINEL, 0, payload_begin
    ) + _byte_mismatches(
        buffers.input,
        INPUT_GUARD_SENTINEL,
        payload_end,
        allocation_bytes,
    )
    output_guard_corruptions = _byte_mismatches(
        buffers.output, OUTPUT_GUARD_SENTINEL, 0, payload_begin
    ) + _byte_mismatches(
        buffers.output,
        OUTPUT_GUARD_SENTINEL,
        payload_end,
        allocation_bytes,
    )
    return SentinelCounters(
        input_guard_corruptions=input_guard_corruptions,
        output_guard_corruptions=output_guard_corruptions,
        inactive_input_sentinel_corruptions=_byte_mismatches(
            buffers.input,
            INACTIVE_INPUT_SENTINEL,
            inactive_begin,
            payload_end,
        ),
        inactive_output_sentinel_corruptions=_byte_mismatches(
            buffers.output,
            INACTIVE_OUTPUT_SENTINEL,
            inactive_begin,
            payload_end,
        ),
    )


def _payload_capacity(buffers: GuardedBuffers) -> int:
    if len(buffers.input) != len(buffers.output):
        raise ValueError("guarded input and output allocation sizes disagree")
    capacity = len(buffers.input) - 2 * buffers.guard_bytes
    if capacity <= 0:
        raise ValueError("guarded buffers do not contain payload capacity")
    return capacity


def _validate_tile(tile: OracleTile, capacity_bytes: int) -> None:
    if (
        tile.active_bytes <= 0
        or tile.active_bytes % (2 * BF16_BYTES) != 0
        or tile.input_offset_bytes % BF16_BYTES != 0
        or tile.output_offset_bytes % BF16_BYTES != 0
        or tile.generation <= 0
    ):
        raise ValueError("tile has invalid BF16 geometry or generation")
    if (
        tile.input_offset_bytes + tile.active_bytes > capacity_bytes
        or tile.output_offset_bytes + tile.active_bytes > capacity_bytes
    ):
        raise ValueError("tile active range exceeds payload capacity")


def _store_bf16(buffer: bytearray, byte_offset: int, value: int) -> None:
    buffer[byte_offset] = value & 0xFF
    buffer[byte_offset + 1] = value >> 8


def _load_bf16(buffer: bytearray, byte_offset: int) -> int:
    return buffer[byte_offset] | (buffer[byte_offset + 1] << 8)


def fill_correctness_input(
    buffers: GuardedBuffers, tile: OracleTile, *, rank: int
) -> None:
    """Fill exactly one descriptor's active input with oracle values."""

    capacity_bytes = _payload_capacity(buffers)
    _validate_tile(tile, capacity_bytes)
    payload_begin = buffers.guard_bytes + tile.input_offset_bytes
    first_element = tile.input_offset_bytes // BF16_BYTES
    for local_element in range(tile.active_bytes // BF16_BYTES):
        _store_bf16(
            buffers.input,
            payload_begin + local_element * BF16_BYTES,
            input_bf16_bits(
                rank=rank,
                element=first_element + local_element,
                generation=tile.generation,
            ),
        )


def fill_expected_output(buffers: GuardedBuffers, tile: OracleTile) -> None:
    """Materialize the bit-exact result used by the active-output checker."""

    capacity_bytes = _payload_capacity(buffers)
    _validate_tile(tile, capacity_bytes)
    output_begin = buffers.guard_bytes + tile.output_offset_bytes
    first_element = tile.input_offset_bytes // BF16_BYTES
    active_elements = tile.active_bytes // BF16_BYTES
    lower_elements = active_elements // 2
    for local_element in range(active_elements):
        association = (
            TiledHalfAssociation.XOR1_THEN_XOR3
            if local_element < lower_elements
            else TiledHalfAssociation.XOR3_THEN_XOR1
        )
        _store_bf16(
            buffers.output,
            output_begin + local_element * BF16_BYTES,
            expected_output_bf16_bits(
                element=first_element + local_element,
                generation=tile.generation,
                association=association,
            ),
        )


def validate_active_output(
    buffers: GuardedBuffers, tiles: tuple[OracleTile, ...]
) -> int:
    """Return the qualification ABI's active-element mismatch count."""

    capacity_bytes = _payload_capacity(buffers)
    mismatches = 0
    for tile in tiles:
        _validate_tile(tile, capacity_bytes)
        output_begin = buffers.guard_bytes + tile.output_offset_bytes
        first_element = tile.input_offset_bytes // BF16_BYTES
        active_elements = tile.active_bytes // BF16_BYTES
        lower_elements = active_elements // 2
        for local_element in range(active_elements):
            association = (
                TiledHalfAssociation.XOR1_THEN_XOR3
                if local_element < lower_elements
                else TiledHalfAssociation.XOR3_THEN_XOR1
            )
            expected = expected_output_bf16_bits(
                element=first_element + local_element,
                generation=tile.generation,
                association=association,
            )
            actual = _load_bf16(
                buffers.output,
                output_begin + local_element * BF16_BYTES,
            )
            mismatches += actual != expected
    return mismatches


def _bf16_to_float(value: int) -> float:
    return struct.unpack(">f", struct.pack(">I", value << 16))[0]


def _float_to_bf16(value: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    bits += 0x7FFF + ((bits >> 16) & 1)
    return bits >> 16


def _add_bf16(left: int, right: int) -> int:
    return _float_to_bf16(_bf16_to_float(left) + _bf16_to_float(right))


def input_bf16_bits(*, rank: int, element: int, generation: int) -> int:
    """Return one rank's deterministic, generation-dependent input bits."""

    if rank not in range(4):
        raise ValueError("rank must be in [0, 3]")
    if element < 0:
        raise ValueError("element must be nonnegative")
    if generation <= 0:
        raise ValueError("generation must be positive")
    scale_exponent = (
        element % INPUT_GENERATION_PERIOD
        + generation % INPUT_GENERATION_PERIOD
    ) % INPUT_GENERATION_PERIOD
    return _RANK_BASE_BF16[rank] + (scale_exponent << 7)


def expected_output_bf16_bits(
    *,
    element: int,
    generation: int,
    association: TiledHalfAssociation,
) -> int:
    """Evaluate one explicit two-stage BF16 reduction tree."""

    values = tuple(
        input_bf16_bits(rank=rank, element=element, generation=generation)
        for rank in range(4)
    )
    if association is TiledHalfAssociation.XOR1_THEN_XOR3:
        first = _add_bf16(values[0], values[1])
        second = _add_bf16(values[2], values[3])
    elif association is TiledHalfAssociation.XOR3_THEN_XOR1:
        first = _add_bf16(values[0], values[3])
        second = _add_bf16(values[1], values[2])
    else:
        raise ValueError("unknown tiled-half association")
    return _add_bf16(first, second)
