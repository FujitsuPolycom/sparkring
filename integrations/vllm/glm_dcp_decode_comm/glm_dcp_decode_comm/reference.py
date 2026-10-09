"""Torch references of the image's DCP decode computations and of this plugin's replacements.

Every function runs on any device with plain torch operations, so the CPU tests
can check data movement and layouts without Triton or a GPU:

- :func:`image_query`: the image's DCP query on the BF16 path, the RoPE that
  ``_fused_q_kernel`` program 0 applies to ``q_pe`` (interleaved pairs, the
  statements ``r1 = x1 * cos - x2 * sin`` and ``r2 = x2 * cos + x1 * sin`` in
  float32) and ``torch.cat`` with ``ql_nope``;
- :func:`image_pack` and :func:`image_combine`: ``_dcp_a2a_pack_send_kernel``
  (without the empty-shard mask, which decode batches without prefill context
  parallelism do not pass) and ``_dcp_a2a_unpack_combine_kernel``;
- :func:`wire_chunk` (the wire format by definition), :func:`stage_send` (the
  packed all-to-all's staging, through the index helpers the kernel uses),
  :func:`exchange` (what each rank receives) and :func:`wire_combine` (the
  combine of ``kernels.wire_combine``, own share from ``out`` and ``lse``).

The float32 arithmetic of :func:`image_combine` and :func:`wire_combine` is one
function (:func:`_combine`), so the two agree bit for bit exactly when they
read the same values: the CPU tests use that to check the wire format. The
GPU kernels' own bits are checked against the image's kernels on a GPU
(``tests/gpu_checks.py``), not against these references.
"""

from __future__ import annotations

import numpy as np
import torch

from . import layout as L


def image_query(positions: torch.Tensor, q_pe: torch.Tensor, cos_sin_cache: torch.Tensor,
                ql_nope: torch.Tensor, rows: int) -> torch.Tensor:
    """``torch.cat((ql_nope, RoPE(q_pe)), dim=-1)`` of the first ``rows`` rows: ``[rows, heads, 576]``."""
    half = cos_sin_cache.shape[-1] // 2
    pos = positions[:rows].long()
    cos = cos_sin_cache[pos, :half].float()[:, None, :]
    sin = cos_sin_cache[pos, half:2 * half].float()[:, None, :]
    x1 = q_pe[:rows, :, 0::2].float()
    x2 = q_pe[:rows, :, 1::2].float()
    r1 = x1 * cos - x2 * sin
    r2 = x2 * cos + x1 * sin
    rot = torch.stack((r1, r2), dim=-1).reshape(rows, q_pe.shape[1], 2 * half).to(q_pe.dtype)
    return torch.cat((ql_nope[:rows], rot), dim=-1)


def image_pack(out: torch.Tensor, lse: torch.Tensor, world: int) -> torch.Tensor:
    """vLLM's send buffer ``[world, rows, heads, 514]``: record ``(j, b, h)`` is head ``j * heads + h``."""
    rows, total, dim = out.shape
    heads = total // world
    records = torch.empty((world, rows, heads, dim + 2), dtype=out.dtype, device=out.device)
    halves = lse.float().contiguous().view(torch.int16).view(rows, total, 2).view(out.dtype)
    for j in range(world):
        records[j, :, :, :dim] = out[:, j * heads:(j + 1) * heads, :]
        records[j, :, :, dim:] = halves[:, j * heads:(j + 1) * heads, :]
    return records


def _clean(lse: torch.Tensor) -> torch.Tensor:
    return torch.where(torch.isnan(lse) | (lse == float("inf")), torch.full_like(lse, float("-inf")), lse)


def _combine(lses: list[torch.Tensor], partials: list[torch.Tensor], is_base_e: bool) -> torch.Tensor:
    """The unpack-combine kernel's float32 statements over sources in rank order."""
    exp = torch.exp if is_base_e else torch.exp2
    log = torch.log if is_base_e else torch.log2
    lses = [_clean(value.float()) for value in lses]
    lse_max = torch.full_like(lses[0], float("-inf"))
    for value in lses:
        lse_max = torch.maximum(lse_max, value)
    lse_max = torch.where(lse_max == float("-inf"), torch.zeros_like(lse_max), lse_max)
    lse_sum = torch.zeros_like(lse_max)
    for value in lses:
        lse_sum = lse_sum + exp(value - lse_max)
    global_lse = log(lse_sum) + lse_max
    acc = torch.zeros(partials[0].shape, dtype=torch.float32, device=partials[0].device)
    for value, partial in zip(lses, partials, strict=True):
        weight = exp(value - global_lse)
        weight = torch.where(torch.isnan(weight), torch.zeros_like(weight), weight)[..., None]
        part = partial.float()
        part = torch.where(weight == 0.0, torch.zeros_like(part), part)
        acc = acc + part * weight
    return acc


def image_combine(records: torch.Tensor, is_base_e: bool = True) -> torch.Tensor:
    """vLLM's combine of a receive buffer ``[world, rows, heads, 514]``: ``[rows, heads, 512]``."""
    world, rows, heads, width = records.shape
    dim = width - 2
    lses = [records[s, :, :, dim:].contiguous().view(torch.float32).view(rows, heads) for s in range(world)]
    partials = [records[s, :, :, :dim] for s in range(world)]
    return _combine(lses, partials, is_base_e).to(records.dtype)


def wire_chunk(out: torch.Tensor, lse: torch.Tensor, world: int, dest: int) -> torch.Tensor:
    """Destination ``dest``'s chunk by the wire format's definition (``layout``), as bytes."""
    heads = out.shape[1] // world
    data = out[:, dest * heads:(dest + 1) * heads, :].contiguous().view(torch.uint8).reshape(-1)
    lses = lse[:, dest * heads:(dest + 1) * heads].float().contiguous().view(torch.uint8).reshape(-1)
    return torch.cat((data, lses))


def stage_send(out: torch.Tensor, lse: torch.Tensor, world: int, rank: int) -> torch.Tensor:
    """The send slot ``[world, chunk bytes]`` the packed all-to-all stages, pack by pack (own chunk zero).

    Each pack is located with :func:`layout.data_pack_source` and
    :func:`layout.lse_pack_source`, the helpers the kernel calls.
    """
    rows, total, _ = out.shape
    heads = total // world
    geometry = L.WireGeometry(rows, heads)
    out_packs = out.contiguous().view(torch.uint8).view(rows, total, L.HEAD_PACKS, L.PACK_BYTES).cpu().numpy()
    lse_packs = (lse.float().contiguous().view(torch.uint8)
                 .view(rows, total // L.LSE_PACK_HEADS, L.PACK_BYTES).cpu().numpy())
    send = np.zeros((world, geometry.chunk_packs, L.PACK_BYTES), dtype=np.uint8)
    data = np.arange(geometry.data_packs)
    b, h, k = L.data_pack_source(data, geometry.row_shift, geometry.row_packs - 1, geometry.head_shift,
                                 L.HEAD_PACKS - 1)
    lse_index = np.arange(geometry.lse_packs)
    lb, q = L.lse_pack_source(lse_index, geometry.lse_shift, geometry.lse_row_packs - 1)
    for j in range(world):
        if j == rank:
            continue
        send[j, data] = out_packs[b, j * heads + h, k]
        send[j, geometry.data_packs + lse_index] = lse_packs[lb, j * heads // L.LSE_PACK_HEADS + q]
    return torch.from_numpy(send.reshape(world, geometry.chunk_bytes)).to(out.device)


def exchange(sends: list[torch.Tensor], rank: int) -> torch.Tensor:
    """What rank ``rank`` receives, ``[world, chunk bytes]``: row ``s`` is chunk ``rank`` of rank ``s``'s send
    slot; the own row stays zero (the packed all-to-all does not copy it)."""
    recv = torch.zeros_like(sends[rank])
    for source, send in enumerate(sends):
        if source != rank:
            recv[source] = send[rank]
    return recv


def wire_combine(recv: torch.Tensor, out: torch.Tensor, lse: torch.Tensor, world: int, rank: int,
                 is_base_e: bool = True) -> torch.Tensor:
    """The combine of ``kernels.wire_combine``: own share from ``out`` and ``lse``, peers' from ``recv``."""
    rows, total, dim = out.shape
    heads = total // world
    data_bytes = L.wire_data_bytes(rows, heads)
    lses, partials = [], []
    for source in range(world):
        if source == rank:
            lses.append(lse[:, rank * heads:(rank + 1) * heads])
            partials.append(out[:, rank * heads:(rank + 1) * heads, :])
        else:
            chunk = recv[source]
            partials.append(chunk[:data_bytes].view(out.dtype).view(rows, heads, dim))
            lses.append(chunk[data_bytes:].view(torch.float32).view(rows, heads))
    return _combine(lses, partials, is_base_e).to(out.dtype)


__all__ = ["exchange", "image_combine", "image_pack", "image_query", "stage_send", "wire_chunk", "wire_combine"]
