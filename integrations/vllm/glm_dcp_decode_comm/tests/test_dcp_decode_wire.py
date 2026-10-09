"""The packed all-to-all's wire format against vLLM's records, on the host.

The staging model (:func:`reference.stage_send`) locates every pack with the index helpers the kernel calls,
so these tests check the kernel's index arithmetic; the combine model reads the wire format, and its result
equals the image's combine of vLLM's records bit for bit exactly when both read the same values.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from glm_dcp_decode_comm import layout as L
from glm_dcp_decode_comm import reference


@pytest.mark.parametrize("heads", (4, 8, 16, 32))
@pytest.mark.parametrize("rows", (1, 2, 3, 31))
def test_a_chunk_has_the_bytes_of_vllms_records(rows, heads):
    geometry = L.WireGeometry(rows, heads)
    assert geometry.chunk_bytes == L.wire_chunk_bytes(rows, heads) == rows * heads * 514 * 2
    assert geometry.data_packs * L.PACK_BYTES == L.wire_data_bytes(rows, heads)
    within = np.arange(geometry.data_packs)
    b, h, k = L.data_pack_source(within, geometry.row_shift, geometry.row_packs - 1, geometry.head_shift,
                                 L.HEAD_PACKS - 1)
    assert np.array_equal(b, within // geometry.row_packs)
    assert np.array_equal(h, (within % geometry.row_packs) // L.HEAD_PACKS)
    assert np.array_equal(k, within % L.HEAD_PACKS)
    lse = np.arange(geometry.lse_packs)
    lb, q = L.lse_pack_source(lse, geometry.lse_shift, geometry.lse_row_packs - 1)
    assert np.array_equal(lb, lse // geometry.lse_row_packs) and np.array_equal(q, lse % geometry.lse_row_packs)


@pytest.mark.parametrize("heads", (0, 2, 3, 6, 12, 24))
def test_head_counts_that_are_not_powers_of_two_of_at_least_four_are_refused(heads):
    assert not L.heads_supported(heads)
    with pytest.raises(ValueError, match="power-of-two count"):
        L.WireGeometry(1, heads)


def _inputs(world: int, rows: int, heads: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    outs = [torch.randint(-32768, 32768, (rows, world * heads, L.V_DIM), dtype=torch.int16, generator=g)
            .view(torch.bfloat16) for _ in range(world)]
    lses = [torch.randn((rows, world * heads), generator=g) for _ in range(world)]
    lses[0][0, 0] = float("-inf")
    lses[-1][rows - 1, heads - 1] = float("inf")
    lses[1 % world][0, heads] = float("nan")
    for lse in lses:                        # an empty row: every source -inf
        lse[rows - 1, 0] = float("-inf")
    return outs, lses


@pytest.mark.parametrize("world,heads,rows", [(2, 8, 1), (4, 8, 3), (4, 4, 2), (8, 8, 2), (4, 16, 5)])
def test_the_staged_chunks_are_the_wire_format(world, heads, rows):
    outs, lses = _inputs(world, rows, heads, seed=world * 100 + rows)
    for rank in range(world):
        send = reference.stage_send(outs[rank], lses[rank], world, rank)
        for dest in range(world):
            if dest == rank:
                assert not send[dest].any()                 # the own chunk never leaves the GPU
            else:
                assert torch.equal(send[dest], reference.wire_chunk(outs[rank], lses[rank], world, dest))


@pytest.mark.parametrize("base_e", (True, False), ids=("base-e", "base-2"))
@pytest.mark.parametrize("world,heads,rows", [(2, 8, 1), (4, 8, 3), (4, 4, 2), (8, 8, 2), (4, 16, 5)])
def test_the_fused_combine_equals_the_images_bit_for_bit(world, heads, rows, base_e):
    # Every rank's combine through the packed exchange (staging, exchange, wire combine with the own share in
    # place) against vLLM's (pack, all-to-all of records, unpack-combine), with random 16-bit output words and
    # LSEs that include -inf, +inf, NaN and a row whose every source is empty.
    outs, lses = _inputs(world, rows, heads, seed=world * 10 + heads + rows)
    sends = [reference.stage_send(outs[s], lses[s], world, s) for s in range(world)]
    packs = [reference.image_pack(outs[s], lses[s], world) for s in range(world)]
    for rank in range(world):
        fused = reference.wire_combine(reference.exchange(sends, rank), outs[rank], lses[rank], world, rank, base_e)
        image = reference.image_combine(torch.stack([packs[s][rank] for s in range(world)]), base_e)
        assert tuple(fused.shape) == (rows, heads, L.V_DIM)
        assert torch.equal(fused.view(torch.int16), image.view(torch.int16)), rank


def test_vllms_records_carry_the_lse_bits_in_two_slots():
    lse = torch.tensor([[1.5, float("-inf"), float("nan"), -3.25e-20]])
    out = torch.zeros((1, 4, L.V_DIM), dtype=torch.bfloat16)
    records = reference.image_pack(out, lse, 1)
    halves = records[0, 0, :, L.V_DIM:].contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    words = halves[:, 0] | (halves[:, 1] << 16)
    assert torch.equal(words, lse[0].view(torch.int32) & 0xFFFFFFFF)


def test_the_image_query_is_the_interleaved_rope_of_q_pe_after_ql_nope():
    g = torch.Generator().manual_seed(3)
    cache = torch.randn((8, L.ROPE_DIM), generator=g)
    positions = torch.tensor([5, 0], dtype=torch.int64)
    q_pe = torch.randn((2, 3, L.ROPE_DIM), generator=g).to(torch.bfloat16)
    ql_nope = torch.randn((2, 3, L.QL_NOPE_DIM), generator=g).to(torch.bfloat16)
    query = reference.image_query(positions, q_pe, cache, ql_nope, 2)
    assert tuple(query.shape) == (2, 3, L.HEAD_DIM) and torch.equal(query[:, :, :L.QL_NOPE_DIM], ql_nope)
    pair = 7
    cos, sin = cache[5, pair], cache[5, 32 + pair]
    x1, x2 = q_pe[0, 1, 2 * pair].float(), q_pe[0, 1, 2 * pair + 1].float()
    assert query[0, 1, L.QL_NOPE_DIM + 2 * pair] == (x1 * cos - x2 * sin).to(torch.bfloat16)
    assert query[0, 1, L.QL_NOPE_DIM + 2 * pair + 1] == (x2 * cos + x1 * sin).to(torch.bfloat16)
