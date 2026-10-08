"""Host side of the ring session's scatter collectives: reduce-scatter and all-to-all.

Every function takes a session (``runtime.RoceOneshotAllReduce``) as its first
argument; the session's methods of the same names delegate here. A function
reads only the session's agreed settings and arena view: ``world_size``,
``rank``, ``device``, ``max_size``, ``lane_count``, ``spin_limit``,
``scatter_available``, ``relay_safe_bytes``, ``large_piece_bytes``,
``_threads``, ``_blocks``, ``_large_blocks``, ``_packs_per_thread``,
``_layout`` (``slots``, ``flag_stride``), ``_recv_base``,
``_flag_base``, ``_send_base``, ``_ctrl_base``, ``_slot_bytes``,
``_epoch_address``, ``_poison_address``, ``_lock``, ``_require_open``,
``check_health``, ``_counter_addresses``, ``_order_stream`` and
``_mark_stream``.

Contract (``sparkring_sircl.scatter_plan`` states the geometry rules):

- ``should_reduce_scatter`` and ``should_all_to_all`` decide from dtype,
  shape, contiguity and the byte arguments only, so every rank of a group
  routes a collective the same way; they return False when the session has no
  scatter collectives (``scatter_available``). Every message size is
  eligible: a message travels as ops of at most ``large_piece_bytes`` bytes
  (the ``W`` pieces of one op, :func:`op_bytes`), so a 64 MiB reduce-scatter
  is 16 ops of 4 MiB with 4 MiB pieces on four ranks;
- ``reduce_scatter`` stores chunk ``rank`` of the rank-ordered float32 sum,
  rounded once to the dtype, into ``out`` (one chunk; allocated when omitted:
  the chunk's shape for contiguous chunks of whole rows, else flat);
  ``all_to_all`` stores the chunk from rank ``s`` at ``s * dst_stride_bytes``
  of ``out`` unchanged. Input and output pointers must be 16-byte aligned;
- every op carries the same byte range of every chunk (a strided scatter),
  so the split changes no bit: each element is the rank-ordered sum of its
  own inputs, as in the session's all-reduce. With the session's relay-safe
  per-peer size set (forward windows off and lanes through relays), pieces
  also stay within it, so no relay hairpin queue holds more than its share;
- launchers that ``prepare`` did not compile raise inside a CUDA graph capture;
  a poisoned session raises; outside a capture the session's health is checked
  after the collective.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Sequence
from typing import Any, Optional

import torch

from .. import protocol as proto
from .. import scatter_plan as sp
from ..protocol import PACK_BYTES
from . import _scatter_cute

REDUCE_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _key(session: Any, mode: str, dtype: torch.dtype) -> tuple:
    name = _DTYPE_NAMES.get(dtype, "bytes") if mode == "reduce" else "bytes"
    return (mode, name, session.world_size, session.rank, session._threads, session._layout.slots,
            session._layout.flag_stride, session.lane_count, session.device.index)


def launcher(session: Any, mode: str, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
    """The compiled scatter launcher of ``mode`` (and ``dtype`` for ``reduce``); compiled eagerly only."""
    if mode not in sp.MODES:
        raise ValueError(f"unknown scatter mode {mode!r}")
    if mode == "reduce" and dtype not in REDUCE_DTYPES:
        raise ValueError(f"unsupported reduce-scatter dtype {dtype}")
    if not session.scatter_available:
        raise RuntimeError("the SIRCL scatter collectives are unavailable in this session")
    key = _key(session, mode, dtype)
    if capturing and not _scatter_cute.is_launcher_prepared(*key):
        what = f"reduce-scatter for {dtype}" if mode == "reduce" else "all-to-all"
        raise RuntimeError(f"SIRCL {what} was not prepared before CUDA graph capture; call prepare(scatter=True)")
    return _scatter_cute.get_launcher(*key)


def prepare(session: Any, dtypes: Sequence[torch.dtype]) -> None:
    """Compile the reduce-scatter launcher of every dtype and the all-to-all launcher."""
    for dtype in dtypes:
        launcher(session, "reduce", dtype, capturing=False)
    launcher(session, "copy", torch.uint8, capturing=False)


def _geometry(session: Any, inp: torch.Tensor, chunk_bytes: Optional[int],
              src_stride_bytes: Optional[int]) -> Optional[sp.ScatterGeometry]:
    if not session.scatter_available or not inp.is_cuda or inp.device != session.device:
        return None
    if inp.is_sparse or inp.dim() == 0 or not inp.is_contiguous():
        return None
    return sp.scatter_geometry(inp.numel() * inp.element_size(), session.world_size, None, chunk_bytes,
                               src_stride_bytes)


def should_reduce_scatter(session: Any, inp: torch.Tensor, *, chunk_bytes: Optional[int] = None,
                          src_stride_bytes: Optional[int] = None) -> bool:
    """Eligible: a float16, bfloat16 or float32 tensor whose scatter geometry the session carries."""
    session._require_open()
    return inp.dtype in REDUCE_DTYPES and _geometry(session, inp, chunk_bytes, src_stride_bytes) is not None


def should_all_to_all(session: Any, inp: torch.Tensor, *, chunk_bytes: Optional[int] = None,
                      src_stride_bytes: Optional[int] = None) -> bool:
    """Eligible: any plain dtype (bytes are copied) whose scatter geometry the session carries."""
    session._require_open()
    if inp.is_complex():
        return False
    return _geometry(session, inp, chunk_bytes, src_stride_bytes) is not None


def op_bytes(session: Any) -> int:
    """Largest whole message of one scatter op: the session's large-message piece, within one slot."""
    slot = int(session._slot_bytes)
    piece = getattr(session, "large_piece_bytes", None)
    if isinstance(piece, int) and piece >= PACK_BYTES:
        return min(piece, slot)
    return min(int(session.max_size), slot)


def piece_bytes(session: Any, chunk_bytes: int) -> int:
    """Per-peer bytes of one scatter op of chunks of ``chunk_bytes``."""
    return sp.piece_bytes(chunk_bytes, getattr(session, "relay_safe_bytes", None),
                          sp.op_peer_bytes(op_bytes(session), session.world_size))


def _launch(session: Any, mode: str, dtype: torch.dtype, input_address: int, output_address: int,
            chunk_bytes: int, src_stride: int, dst_stride: int, capturing: bool) -> None:
    run = launcher(session, mode, dtype, capturing)
    nbytes = chunk_bytes * session.world_size
    packs = nbytes // PACK_BYTES
    blocks = getattr(session, "_large_blocks", session._blocks)
    grid = proto.grid_blocks(packs, session._threads, blocks, session._packs_per_thread)
    stage, tail = session._counter_addresses(grid)
    session._order_stream(capturing)
    run(input_address, output_address, packs, nbytes, chunk_bytes // PACK_BYTES, src_stride, dst_stride,
        session._recv_base, session._flag_base, session._send_base, session._ctrl_base, session._slot_bytes,
        session._epoch_address, stage, tail, session._poison_address, session.spin_limit, grid)
    session._mark_stream(capturing)


def _ops(session: Any, mode: str, dtype: torch.dtype, inp: torch.Tensor, out: torch.Tensor,
         geometry: sp.ScatterGeometry, dst_stride: int, stream: object) -> None:
    context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
    with torch.cuda.device(session.device), context:
        capturing = torch.cuda.is_current_stream_capturing()
        piece = piece_bytes(session, geometry.chunk_bytes)
        for part in sp.scatter_plan(geometry.chunk_bytes, piece):
            _launch(session, mode, dtype, inp.data_ptr() + part.offset, out.data_ptr() + part.offset, part.nbytes,
                    geometry.src_stride_bytes, part.nbytes if mode == "reduce" else dst_stride, capturing)
    if not capturing:
        session.check_health()


def reduce_scatter(session: Any, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None, stream: object = None,
                   chunk_bytes: Optional[int] = None, src_stride_bytes: Optional[int] = None) -> torch.Tensor:
    """Sum chunk ``rank`` of every rank's ``inp`` into ``out`` (one chunk)."""
    with session._lock:
        session.check_health()
        if inp.dtype not in REDUCE_DTYPES:
            raise ValueError(f"unsupported SIRCL reduce-scatter dtype {inp.dtype}")
        geometry = _geometry(session, inp, chunk_bytes, src_stride_bytes)
        if geometry is None:
            raise ValueError("input is not eligible for the SIRCL reduce-scatter")
        if inp.data_ptr() % PACK_BYTES:
            raise ValueError("the SIRCL reduce-scatter needs a 16-byte aligned input pointer")
        chunk = geometry.chunk_bytes
        world = session.world_size
        if out is None:
            if chunk_bytes is None and inp.shape[0] % world == 0:
                out = torch.empty((inp.shape[0] // world, *inp.shape[1:]), dtype=inp.dtype, device=inp.device)
            else:
                out = torch.empty(chunk // inp.element_size(), dtype=inp.dtype, device=inp.device)
        if (out.dtype != inp.dtype or out.device != inp.device or not out.is_contiguous()
                or out.numel() * out.element_size() != chunk or out.data_ptr() % PACK_BYTES):
            raise ValueError("out must be a 16-byte aligned contiguous tensor of one chunk in the input's dtype")
        _ops(session, "reduce", inp.dtype, inp, out, geometry, chunk, stream)
        return out


def all_to_all(session: Any, inp: torch.Tensor, out: torch.Tensor, *, stream: object = None,
               chunk_bytes: Optional[int] = None, src_stride_bytes: Optional[int] = None,
               dst_stride_bytes: Optional[int] = None) -> torch.Tensor:
    """Send chunk ``p`` of ``inp`` to rank ``p``; place rank ``s``'s chunk at ``s * dst_stride_bytes`` of ``out``."""
    with session._lock:
        session.check_health()
        geometry = _geometry(session, inp, chunk_bytes, src_stride_bytes)
        if geometry is None or inp.is_complex():
            raise ValueError("input is not eligible for the SIRCL all-to-all")
        if (not out.is_cuda or out.device != inp.device or not out.is_contiguous()
                or inp.data_ptr() % PACK_BYTES or out.data_ptr() % PACK_BYTES):
            raise ValueError("input and out must be 16-byte aligned contiguous tensors on the session's device")
        try:
            dst_stride = sp.destination_stride(geometry.chunk_bytes, session.world_size,
                                               out.numel() * out.element_size(), dst_stride_bytes)
        except sp.ScatterError as error:
            raise ValueError(str(error)) from None
        _ops(session, "copy", torch.uint8, inp, out, geometry, dst_stride, stream)
        return out


__all__ = ["REDUCE_DTYPES", "all_to_all", "launcher", "op_bytes", "piece_bytes", "prepare", "reduce_scatter",
           "should_all_to_all", "should_reduce_scatter"]
