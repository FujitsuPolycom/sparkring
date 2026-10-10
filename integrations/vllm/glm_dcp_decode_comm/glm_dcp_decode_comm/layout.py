"""Geometry of GLM-5.3's DCP decode tensors and of the packed all-to-all's wire format (pure Python).

The DCP query of one head is ``[ql_nope | RoPE(q_pe)]``: the 512 latent values
after the ``W_UK`` absorption and the 64 rotary values, 576 BF16 values in all.

The DCP combine exchanges, for every destination rank ``j`` and every token row
``b``, the attention output of the ``heads`` query heads that ``j`` keeps
(heads ``j * heads .. (j + 1) * heads - 1`` of this rank's
``out[rows, world * heads, 512]`` BF16) and their FP32 log-sum-exp
(``lse[rows, world * heads]``). vLLM's pack (``_dcp_a2a_pack_send_kernel`` of
``vllm/v1/attention/ops/dcp.py``) writes one 514-value BF16 record per row and
head, the LSE bits in the last two slots. The packed all-to-all
(``_scatter_pack_cute.py``) sends the same bytes in 16-byte packs instead:

    chunk j = [ data: rows x heads x 64 packs | lse: rows x heads / 4 packs ]

where data pack ``(b, h, k)`` holds output values ``8k .. 8k + 7`` of head
``j * heads + h`` in row ``b``, and LSE pack ``(b, q)`` holds the LSEs of heads
``j * heads + 4q .. 4q + 3`` in row ``b``. Both layouts carry
``rows * heads * 1028`` bytes per chunk, so an op of either has the same size.

The index helpers below split a pack index into its coordinates with shifts and
masks only, so the kernel calls them on device values while it is traced and
the CPU tests call them on Python integers.
"""

from __future__ import annotations

from dataclasses import dataclass

QL_NOPE_DIM = 512                    # latent query values per head (kv_lora_rank)
ROPE_DIM = 64                        # rotary query values per head (qk_rope_head_dim)
HEAD_DIM = QL_NOPE_DIM + ROPE_DIM    # one head of the DCP query
V_DIM = 512                          # attention output values per head before W_UV (the latent width)
PACK_BYTES = 16
LSE_BYTES = 4                        # one FP32 log-sum-exp per head
HEAD_PACKS = V_DIM * 2 // PACK_BYTES  # one head's BF16 output in 16-byte packs
RECORD_BYTES = V_DIM * 2 + LSE_BYTES  # one row and head of a chunk: 1028 bytes, vLLM's 514 BF16 values
LSE_PACK_HEADS = PACK_BYTES // LSE_BYTES  # heads per LSE pack


def wire_chunk_bytes(rows: int, heads: int) -> int:
    """Bytes of one destination's chunk: ``rows * heads`` records of 1028 bytes."""
    return int(rows) * int(heads) * RECORD_BYTES


def wire_data_bytes(rows: int, heads: int) -> int:
    """Bytes of a chunk's data region (the LSE region follows it)."""
    return int(rows) * int(heads) * V_DIM * 2


def heads_supported(heads: int) -> bool:
    """The packed all-to-all splits pack indices with shifts: a power of two of at least four heads."""
    heads = int(heads)
    return heads >= LSE_PACK_HEADS and heads & (heads - 1) == 0


@dataclass(frozen=True)
class WireGeometry:
    """Pack counts and index shifts of the chunks of one packed all-to-all."""

    rows: int
    heads: int

    def __post_init__(self) -> None:
        if int(self.rows) < 1:
            raise ValueError(f"the packed all-to-all needs at least one row, got {self.rows}")
        if not heads_supported(self.heads):
            raise ValueError(f"the packed all-to-all needs a power-of-two count of at least "
                             f"{LSE_PACK_HEADS} heads per rank, got {self.heads}")

    @property
    def row_packs(self) -> int:
        """Data packs per row of one chunk."""
        return int(self.heads) * HEAD_PACKS

    @property
    def lse_row_packs(self) -> int:
        """LSE packs per row of one chunk."""
        return int(self.heads) // LSE_PACK_HEADS

    @property
    def data_packs(self) -> int:
        return int(self.rows) * self.row_packs

    @property
    def lse_packs(self) -> int:
        return int(self.rows) * self.lse_row_packs

    @property
    def chunk_packs(self) -> int:
        return self.data_packs + self.lse_packs

    @property
    def chunk_bytes(self) -> int:
        return self.chunk_packs * PACK_BYTES

    @property
    def row_shift(self) -> int:
        return self.row_packs.bit_length() - 1

    @property
    def head_shift(self) -> int:
        return HEAD_PACKS.bit_length() - 1

    @property
    def lse_shift(self) -> int:
        return self.lse_row_packs.bit_length() - 1


def data_pack_source(within, row_shift, row_mask, head_shift, head_mask):
    """Row, head and pack-within-head ``(b, h, k)`` of data pack ``within`` of a chunk."""
    b = within >> row_shift
    hk = within & row_mask
    return b, hk >> head_shift, hk & head_mask


def lse_pack_source(within, lse_shift, lse_mask):
    """Row and pack-within-row ``(b, q)`` of LSE pack ``within`` of a chunk."""
    return within >> lse_shift, within & lse_mask


__all__ = [
    "HEAD_DIM",
    "HEAD_PACKS",
    "LSE_BYTES",
    "LSE_PACK_HEADS",
    "PACK_BYTES",
    "QL_NOPE_DIM",
    "RECORD_BYTES",
    "ROPE_DIM",
    "V_DIM",
    "WireGeometry",
    "data_pack_source",
    "heads_supported",
    "lse_pack_source",
    "wire_chunk_bytes",
    "wire_data_bytes",
]
