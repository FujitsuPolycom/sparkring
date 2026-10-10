"""Host side of the ring session's Swing all-reduce.

Every function takes a session (``runtime.RoceOneshotAllReduce``) as its first
argument and reads only its agreed settings and arena view: ``world_size``,
``rank``, ``device``, ``lane_count``, ``spin_limit``, ``max_size``,
``_available``, ``_threads``, ``_large_blocks``, ``_packs_per_thread``,
``_layout`` (``slots``, ``flag_stride``), ``_recv_base``, ``_flag_base``,
``_send_base``, ``_ctrl_base``, ``_slot_bytes``, ``_epoch_address``,
``_poison_address``, ``_counter_addresses``, ``_phase_counters``, and for
:func:`all_reduce` also ``_lock``, ``check_health``, ``_order_stream``,
``_mark_stream`` and ``_aligned_scratch``.

- :func:`launcher` and :func:`launch` are what the session's own all-reduce
  path needs: the compiled launcher of one dtype (refused inside a CUDA graph
  capture unless ``prepare`` compiled it) and one launch with the session's
  arena, counters and spin limit;
- :func:`all_reduce` is one complete Swing all-reduce of one op, eager or
  captured, with the session's locking, stream ordering and health checks;
- :func:`relay_safe_message_bytes` is the largest Swing message that keeps
  every relay hairpin queue of the session's layout within its share when no
  forward window paces the relayed lanes (``sparkring_sircl.swing_plan``).

The launch grid is the session's large-message grid (``_large_blocks``): every
block of a launch must be resident at once, as for the two-shot all-reduce.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

import torch

from .. import protocol as proto
from .. import routes as routes_mod
from .. import swing_plan
from ..protocol import PACK_BYTES
from . import _swing_cute

DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_DTYPE_NAMES = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


def available(session: Any) -> bool:
    """The session can run the Swing all-reduce: multi-phase ops and a power-of-two group."""
    return swing_plan.available(session.world_size, bool(getattr(session, "multi_phase", True)))


def _key(session: Any, dtype: torch.dtype) -> tuple:
    return (_DTYPE_NAMES[dtype], session.world_size, session.rank, session._threads, session._layout.slots,
            session._layout.flag_stride, session.lane_count, session.device.index)


def is_prepared(session: Any, dtype: torch.dtype) -> bool:
    """Whether the Swing launcher of ``dtype`` for this session is compiled."""
    return dtype in _DTYPE_NAMES and _swing_cute.is_launcher_prepared(*_key(session, dtype))


def launcher(session: Any, dtype: torch.dtype, capturing: bool) -> Callable[..., None]:
    """The compiled Swing launcher of ``dtype``; inside a capture only when ``prepare`` compiled it."""
    if dtype not in DTYPES:
        raise ValueError(f"unsupported Swing all-reduce dtype {dtype}")
    if not session._available.get("swing", False):
        raise RuntimeError("the SIRCL Swing all-reduce is unavailable in this session (it needs a power-of-two "
                           "group and multi-phase ops)")
    key = _key(session, dtype)
    if capturing and not _swing_cute.is_launcher_prepared(*key):
        raise RuntimeError(f"SIRCL Swing all-reduce for {dtype} was not prepared before CUDA graph capture; "
                           "call prepare()")
    return _swing_cute.get_launcher(*key)


def grid(session: Any, packs: int) -> int:
    return proto.grid_blocks(packs, session._threads, session._large_blocks, session._packs_per_thread)


def launch(session: Any, run: Callable[..., None], input_address: int, output_address: int, packs: int,
           nbytes: int, grid_blocks: int) -> None:
    """One Swing launch with the session's arena, counters and spin limit (no stream ordering)."""
    stage, tail = session._counter_addresses(grid_blocks)
    phase, phase_stride = session._phase_counters(grid_blocks)
    run(input_address, output_address, packs, nbytes, session._recv_base, session._flag_base, session._send_base,
        session._ctrl_base, session._slot_bytes, session._epoch_address, stage, phase, phase_stride, tail,
        session._poison_address, session.spin_limit, grid_blocks)


def all_reduce(session: Any, inp: torch.Tensor, *, out: Optional[torch.Tensor] = None,
               stream: object = None) -> torch.Tensor:
    """One Swing all-reduce of ``inp`` (contiguous, a multiple of 16 bytes, at most the capacity)."""
    with session._lock:
        session.check_health()
        nbytes = inp.numel() * inp.element_size()
        if (inp.dtype not in DTYPES or not inp.is_cuda or inp.device != session.device or not inp.is_contiguous()
                or not 0 < nbytes <= session.max_size or nbytes % PACK_BYTES):
            raise ValueError("input is not eligible for the SIRCL Swing all-reduce")
        if out is not None and (out.shape != inp.shape or out.dtype != inp.dtype or out.device != inp.device
                                or not out.is_contiguous()):
            raise ValueError("out must be a contiguous tensor on the input's device matching the input")
        context = torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()
        with torch.cuda.device(session.device), context:
            capturing = torch.cuda.is_current_stream_capturing()
            run = launcher(session, inp.dtype, capturing)
            if out is None:
                out = torch.empty_like(inp)
            src = inp
            if inp.data_ptr() % PACK_BYTES:
                src = session._aligned_scratch(0, inp)
                src.copy_(inp)
            dst = out if out.data_ptr() % PACK_BYTES == 0 else session._aligned_scratch(1, out)
            packs = nbytes // PACK_BYTES
            session._order_stream(capturing)
            launch(session, run, src.data_ptr(), dst.data_ptr(), packs, nbytes, grid(session, packs))
            if dst is not out:
                out.copy_(dst)
            session._mark_stream(capturing)
        if not capturing:
            session.check_health()
        return out


def relay_safe_message_bytes(session: Any, route_maps: Sequence[Mapping[int, Sequence[str]]]) -> Optional[int]:
    """Largest Swing message that keeps every relay queue within its share without forward windows."""
    layout = getattr(session, "_layout_identity_object", None)
    if layout is None or not available(session):
        return None
    return swing_plan.relay_safe_message_bytes(
        layout, route_maps, session.lane_count,
        queue_bytes=getattr(session, "hairpin_queue_bytes", routes_mod.DEFAULT_HAIRPIN_QUEUE),
        share=routes_mod.RELAY_QUEUE_SHARE)


__all__ = ["DTYPES", "all_reduce", "available", "grid", "is_prepared", "launch", "launcher",
           "relay_safe_message_bytes"]
