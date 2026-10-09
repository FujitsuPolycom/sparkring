"""Carry out a :class:`.planner.Plan` with one rank's ring session.

Each function issues the session ops the plan describes, on the caller's
current CUDA stream, in an order every rank shares. Pieces of a split message
are complete session ops on contiguous slices; outputs are allocated with
torch's caching allocator, so the same code runs eagerly and inside a CUDA
graph capture (the session itself checks its capture rules: one stream per
capture, nothing compiled inside it).
Nothing here synchronizes the host with the device.

A session that offers ``all_reduce_large`` or ``all_gather_large`` (the
optional large-message operations of :mod:`.sessionapi`) carries a split
message itself; the host loops below are the reference behaviour those
operations must reproduce bit for bit.

A column gather (an all-gather along a dimension with more than one row in
front of it) that the session would carry on its ring or chain as a
dimension-0 gather of the same shard runs as that gather into a staging
buffer plus one local copy (:class:`ColumnGather`, :func:`column_gather_route`).
"""

from __future__ import annotations

import math
import os
import threading
from collections.abc import Mapping, Sequence

import torch

from . import planner
from .planner import PACK, Plan, Policy, SessionLimits, TensorMeta

_ACCUMULATOR = {
    torch.float16: torch.float32, torch.bfloat16: torch.float32, torch.float32: torch.float32,
    torch.float64: torch.float64,
}


class PlanError(RuntimeError):
    """A plan cannot be carried out with the given tensors."""


COLUMN_GATHER_VARIABLE = "SIRCL_COLUMN_GATHER"
# Integer views of the column gather's local copy, widest first: each row moves in the widest view whose
# width divides the row's bytes. An integer copy moves the bits unchanged, whatever the tensor's dtype.
_ROW_VIEWS = ((8, torch.int64), (4, torch.int32), (2, torch.int16), (1, torch.uint8))


class ColumnGather:
    """Column gathers as one dimension-0 link all-gather and one local copy.

    A column gather is an all-gather along a dimension with more than one row in front of it
    (``prod(shape[:dim]) > 1``), such as a column-split projection's output gathered along its last
    dimension. Every rank's shard then lands in ``prod(shape[:dim])`` separate pieces of the output, which
    the session's ring and chain all-gathers (one contiguous piece per rank) do not take, so
    ``all_gather_large`` moves it in tiles. Where the session would carry a dimension-0 gather of the same
    shard on its ring or chain (:func:`column_gather_route`), :func:`all_gather` gathers the shard's bytes
    along dimension 0 into a staging buffer laid out as ``[world, *shard]`` and copies it once into the
    requested layout (:func:`interleave_rows`). Bytes move unchanged, so the output is bit-identical to the
    tiled gather's.

    ``enabled`` comes from ``SIRCL_COLUMN_GATHER`` (``1``, the default, or ``0``), which must be the same on
    every rank, like every dispatch setting. ``calls`` counts the column gathers each route carried (a CUDA
    graph capture counts once, its replays do not); ``last_route`` is the route of the latest
    :func:`all_gather` call made with this object, None when that call was not staged. One object serves the
    groups of one session, from the thread that issues their collectives.

    Memory: outside CUDA graph capture the object keeps one staging buffer, grown to the largest
    ``world * shard`` bytes it has served and reused by every later call, which takes a prefix of it
    (``staging_bytes``). A call inside a capture takes its staging from the caching allocator, like its
    output, so the graph's memory pool holds it; such a call leaves the kept buffer alone.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.calls: dict[str, int] = {}
        self.last_route: str | None = None
        self._buffer: torch.Tensor | None = None
        self._stream = None                     # the CUDA stream the kept buffer was allocated on
        self._lock = threading.Lock()

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> "ColumnGather":
        """The object ``SIRCL_COLUMN_GATHER`` of ``environ`` (default the process environment) describes."""
        raw = (os.environ if environ is None else environ).get(COLUMN_GATHER_VARIABLE, "").strip()
        if raw not in ("", "0", "1"):
            raise ValueError(f"{COLUMN_GATHER_VARIABLE} must be 0 or 1, got {raw!r}")
        return cls(raw != "0")

    @property
    def staging_bytes(self) -> int:
        """Bytes of the staging buffer kept for calls outside capture."""
        buffer = self._buffer
        return 0 if buffer is None else buffer.numel()

    def staging(self, nbytes: int, device) -> torch.Tensor:
        """A contiguous ``uint8`` tensor of ``nbytes`` on ``device`` that starts an allocation."""
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            return torch.empty(nbytes, dtype=torch.uint8, device=device)
        stream = torch.cuda.current_stream(device) if device.type == "cuda" else None
        buffer = self._buffer
        if buffer is None or buffer.device != device or buffer.numel() < nbytes:
            # The smaller buffer returns to the caching allocator, which hands it out again only after the
            # work queued on its own stream and on every stream named by record_stream below.
            buffer = torch.empty(nbytes, dtype=torch.uint8, device=device)
            self._buffer, self._stream = buffer, stream
        elif stream is not None and stream != self._stream:
            buffer.record_stream(stream)
        return buffer[:nbytes]

    def count(self, route: str) -> None:
        with self._lock:
            self.calls[route] = self.calls.get(route, 0) + 1

    def describe(self) -> str:
        """``on`` or ``off``: the receipt line's ``column_gather`` field."""
        return "on" if self.enabled else "off"

    def snapshot(self) -> dict[str, object]:
        """The receipt's ``column_gather_detail``: the switch, the calls per route and the staging bytes kept."""
        with self._lock:
            calls = dict(sorted(self.calls.items()))
        return {"enabled": self.enabled, "calls": calls, "staging_bytes": self.staging_bytes}


def column_gather_route(session, plan: Plan, inp: torch.Tensor, dim: int) -> str | None:
    """``ring`` or ``chain`` where :class:`ColumnGather` stages this all-gather, else None.

    The call is staged when all of these hold:

    - the plan hands it to the session's ``all_gather_large`` (method ``large``, or ``rows`` or ``tiles``
      on a session that offers the method), so it is more than one session op;
    - more than one row precedes ``dim`` (``prod(shape[:dim]) > 1``); with one row the shard is already one
      piece of the output and the session chooses its links itself;
    - the session reports that ``all_gather_large`` of a dimension-0 shard of the same bytes runs as one
      ring all-gather (``gather_uses_ring``: route ``ring``) or one chain all-gather
      (``gather_uses_chain``: route ``chain``), under its agreed schedule, minimums, tuning table and
      capture mode.

    Every input is the same on every rank: the plan, the shard's shape and element size and the session's
    agreed settings. Data pointers and contiguity are not read; :func:`all_gather` gives the session
    16-byte-aligned memory on every rank.
    """
    if not callable(getattr(session, "all_gather_large", None)):
        return None
    if not plan.method.endswith(("large", "rows", "tiles")):
        return None
    if inp.dim() == 0 or inp.numel() == 0 or math.prod(inp.shape[:dim % inp.dim()]) <= 1:
        return None
    probe = torch.empty((inp.numel() * inp.element_size(),), dtype=torch.uint8, device="meta")
    for route, question in (("ring", "gather_uses_ring"), ("chain", "gather_uses_chain")):
        ask = getattr(session, question, None)
        if callable(ask) and ask(probe, 0):
            return route
    return None


def interleave_rows(staging: torch.Tensor, out: torch.Tensor, world: int, outer: int, inner: int) -> torch.Tensor:
    """Copy ``world`` blocks of ``outer`` rows of ``inner`` bytes (``staging``, block after block) into
    ``out`` so that row ``o`` of ``out`` holds row ``o`` of every block in block order:
    ``out[o, w] = staging[w, o]``. Both are contiguous ``uint8`` tensors of ``world * outer * inner`` bytes
    that start 8-byte aligned; the copy moves the rows in the widest integer view whose width divides
    ``inner``."""
    width, view = next((width, view) for width, view in _ROW_VIEWS if inner % width == 0)
    source = staging.view(view).view(world, outer, inner // width)
    out.view(view).view(outer, world, inner // width).copy_(source.transpose(0, 1))
    return out


def _gather_columns(session, inp: torch.Tensor, dim: int, world: int, out_shape: list[int],
                    column_gather: ColumnGather, route: str) -> torch.Tensor:
    """One column gather: the shard's bytes gathered along dimension 0 into ``[world, *shard]`` staging,
    then row ``o`` of the ``[outer, inner]`` byte view of rank ``w``'s shard copied to columns
    ``w * inner`` to ``(w + 1) * inner`` of row ``o`` of the ``[outer, world * inner]`` byte view of the
    output (``outer`` rows precede ``dim``; ``inner`` is the bytes from ``dim`` on)."""
    src = inp.contiguous()
    if src.data_ptr() % PACK:
        # A fresh allocation is 16-byte aligned: the session's link kernels then read the shard in place
        # (the session would otherwise stage it itself, "Pointer alignment" in the package README).
        src = src.clone()
    nbytes = src.numel() * src.element_size()
    staging = column_gather.staging(world * nbytes, src.device)
    session.all_gather_large(src.reshape(-1).view(torch.uint8), dim=0, out=staging)
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    outer = math.prod(inp.shape[:dim])
    interleave_rows(staging, out.reshape(-1).view(torch.uint8), world, outer, nbytes // outer)
    column_gather.count(route)
    return out


def _padded_reduce(session, flat: torch.Tensor) -> torch.Tensor:
    """All-reduce a contiguous 1-D tensor whose bytes are not a multiple of 16."""
    item = flat.element_size()
    padded = -(-flat.numel() * item // PACK) * PACK // item
    staged = torch.zeros(padded, dtype=flat.dtype, device=flat.device)
    staged[: flat.numel()].copy_(flat)
    return session.all_reduce(staged)[: flat.numel()]


def all_reduce(session, plan: Plan, inp: torch.Tensor, limits: SessionLimits,
               policy: Policy, *, capturing: bool, out: torch.Tensor | None = None) -> torch.Tensor:
    """Sum ``inp`` over the group; write into ``out`` when given (it may alias ``inp``)."""
    if plan.method == "empty":
        return inp.clone() if out is None else out
    src = inp.contiguous()
    if plan.method == "direct":
        if out is not None and out.is_contiguous() and out.data_ptr() != src.data_ptr():
            return session.all_reduce(src, out=out)
        result = session.all_reduce(src)
    elif plan.method == "padded":
        result = _padded_reduce(session, src.reshape(-1)).view(src.shape)
    elif plan.method == "large" or (plan.method == "chunked"
                                    and callable(getattr(session, "all_reduce_large", None))):
        result = session.all_reduce_large(src)
    elif plan.method == "chunked":
        result = torch.empty_like(src)
        flat_in, flat_out = src.reshape(-1), result.view(-1)
        item = src.element_size()
        for offset, count in _ranges(flat_in.numel(), plan.piece // item):
            piece = flat_in[offset: offset + count]
            if (count * item) % PACK:
                flat_out[offset: offset + count].copy_(_padded_reduce(session, piece))
            else:
                session.all_reduce(piece, out=flat_out[offset: offset + count])
    elif plan.method == "gather_sum":
        result = _gather_sum(session, src, limits, policy, capturing=capturing)
    else:
        raise PlanError(f"unknown all-reduce method {plan.method}")
    if out is None:
        return result
    out.copy_(result)
    return out


def _gather_sum(session, src: torch.Tensor, limits: SessionLimits, policy: Policy,
                *, capturing: bool) -> torch.Tensor:
    flat = src.reshape(-1)
    meta = TensorMeta.of(flat)
    inner = planner.plan_all_gather(meta, 0, limits, policy.internal(),
                                    capturing=capturing)
    gathered = all_gather(session, inner, flat, 0, limits.world).view(limits.world, -1)
    accumulator = _ACCUMULATOR.get(src.dtype, torch.int64)
    total = torch.zeros(flat.shape, dtype=accumulator, device=src.device)
    for rank in range(limits.world):            # rank order, one rounding at the end
        total += gathered[rank].to(accumulator)
    return total.to(src.dtype).view(src.shape)


def _ranges(total: int, step: int) -> list[tuple[int, int]]:
    step = max(1, step)
    return [(offset, min(step, total - offset)) for offset in range(0, total, step)]


def all_gather(session, plan: Plan, inp: torch.Tensor, dim: int, world: int, *,
               column_gather: ColumnGather | None = None) -> torch.Tensor:
    """Concatenate every rank's ``inp`` along ``dim`` in rank order.

    With an enabled ``column_gather``, a column gather that :func:`column_gather_route` gives a route runs
    as one dimension-0 link all-gather plus one local copy; every other call runs as without it.
    """
    dim = dim % inp.dim()
    out_shape = list(inp.shape)
    out_shape[dim] *= world
    if plan.method == "empty":
        return torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    if column_gather is not None:
        route = column_gather_route(session, plan, inp, dim) if column_gather.enabled else None
        column_gather.last_route = route
        if route is not None:
            return _gather_columns(session, inp, dim, world, out_shape, column_gather, route)
    src = inp.contiguous()
    native = getattr(session, "all_gather_large", None)
    if plan.method.endswith("large") or (plan.method.endswith(("rows", "tiles")) and callable(native)):
        if not callable(native):
            raise PlanError(f"plan {plan.method} needs the session's all_gather_large")
        return native(src, dim=dim)
    outer = math.prod(src.shape[:dim])
    inner = math.prod(src.shape[dim:])
    method = plan.method
    view_dtype = None
    if method.startswith("bytes_"):
        method = method[len("bytes_"):]
        view_dtype = src.dtype
        item = src.element_size()
        src = src.reshape(-1).view(torch.uint8)
        inner *= item
    matrix = src.reshape(outer, inner)
    gathered = torch.empty((outer, world * inner), dtype=matrix.dtype, device=matrix.device)
    if outer == 1 and method in ("direct", "tiles"):
        flat_out = gathered.view(world, inner)
        if method == "direct":
            session.all_gather(matrix.view(-1), dim=0, out=gathered.view(-1))
        else:
            for offset, count in _ranges(inner, plan.piece):
                part = session.all_gather(matrix.view(-1)[offset: offset + count], dim=0)
                flat_out[:, offset: offset + count].copy_(part.view(world, count))
    elif method == "direct":
        session.all_gather(matrix, dim=-1, out=gathered)
    elif method == "rows":
        for offset, count in _ranges(outer, plan.piece):
            session.all_gather(matrix[offset: offset + count], dim=-1,
                               out=gathered[offset: offset + count])
    elif method == "tiles":
        blocks = gathered.view(outer, world, inner)
        for row in range(outer):
            for offset, count in _ranges(inner, plan.piece):
                part = session.all_gather(matrix[row, offset: offset + count].contiguous(), dim=0)
                blocks[row, :, offset: offset + count].copy_(part.view(world, count))
    else:
        raise PlanError(f"unknown all-gather method {plan.method}")
    if view_dtype is not None:
        gathered = gathered.view(-1).view(view_dtype)
    return gathered.reshape(out_shape)


def reduce_scatter(session, plan: Plan, inp: torch.Tensor, dim: int, rank: int,
                   limits: SessionLimits, policy: Policy, *, capturing: bool) -> torch.Tensor:
    """This rank's ``world``-th of the group sum along ``dim``."""
    world = limits.world
    dim = dim % inp.dim()
    chunk = inp.shape[dim] // world
    if plan.method == "empty":
        shape = list(inp.shape)
        shape[dim] = chunk
        return torch.empty(shape, dtype=inp.dtype, device=inp.device)
    if plan.method == "allreduce_slice":
        reduce_plan = planner.plan_all_reduce(TensorMeta.of(inp), limits,
                                              policy.internal(),
                                              capturing=capturing)
        summed = all_reduce(session, reduce_plan, inp, limits, policy, capturing=capturing)
        return summed.narrow(dim, rank * chunk, chunk).contiguous()
    if plan.method != "scatter":
        raise PlanError(f"unknown reduce-scatter method {plan.method}")
    moved = inp.movedim(dim, 0).contiguous()
    out = torch.empty((chunk,) + tuple(moved.shape[1:]), dtype=inp.dtype, device=inp.device)
    if plan.ops == 1:
        session.reduce_scatter(moved, out=out)
    else:
        item = moved.element_size()
        block = out.numel() * item
        flat_in, flat_out = moved.view(-1), out.view(-1)
        for offset, count in _ranges(block, plan.piece):
            session.reduce_scatter(flat_in[offset // item:],
                                   out=flat_out[offset // item: (offset + count) // item],
                                   chunk_bytes=count, src_stride_bytes=block)
    return out if dim == 0 else out.movedim(0, dim).contiguous()


def all_gatherv(session, plan: Plan, inp: torch.Tensor, sizes: Sequence[int] | None,
                rank: int, world: int) -> torch.Tensor:
    if not plan.method.startswith("padded_"):
        return all_gather(session, plan, inp, 0, world)
    assert sizes is not None
    longest = max(sizes)
    staged = torch.zeros((longest,) + tuple(inp.shape[1:]), dtype=inp.dtype, device=inp.device)
    staged[: inp.shape[0]].copy_(inp)
    inner = Plan(plan.collective, plan.backend, plan.method[len("padded_"):], plan.ops, plan.piece)
    gathered = all_gather(session, inner, staged, 0, world)
    return torch.cat([gathered[w * longest: w * longest + sizes[w]] for w in range(world)], dim=0)


def reduce_scatterv(session, plan: Plan, inp: torch.Tensor, dim: int, sizes: Sequence[int] | None,
                    rank: int, limits: SessionLimits, policy: Policy, *, capturing: bool) -> torch.Tensor:
    if sizes is None or len(set(sizes)) == 1:
        return reduce_scatter(session, plan, inp, dim, rank, limits, policy, capturing=capturing)
    reduce_plan = planner.plan_all_reduce(TensorMeta.of(inp), limits,
                                          policy.internal(), capturing=capturing)
    summed = all_reduce(session, reduce_plan, inp, limits, policy, capturing=capturing)
    dim = dim % inp.dim()
    offset = sum(sizes[:rank])
    return summed.narrow(dim, offset, sizes[rank]).contiguous()


def _gather_bytes(session, tensor: torch.Tensor, limits: SessionLimits, policy: Policy,
                  *, capturing: bool) -> torch.Tensor:
    """Every rank's bytes of ``tensor`` as a ``[world, nbytes]`` uint8 matrix."""
    flat = tensor.contiguous().reshape(-1).view(torch.uint8)
    inner = planner.plan_all_gather(TensorMeta.of(flat), 0, limits,
                                    policy.internal(), capturing=capturing)
    if inner.backend != planner.SIRCL:
        raise PlanError(f"the byte gather is not carried by the session: {inner.reason}")
    return all_gather(session, inner, flat, 0, limits.world).view(limits.world, -1)


def broadcast(session, plan: Plan, tensor: torch.Tensor, src: int, limits: SessionLimits,
              policy: Policy, *, capturing: bool) -> torch.Tensor:
    """Copy rank ``src``'s ``tensor`` into every rank's ``tensor`` (in place)."""
    if plan.method == "empty":
        return tensor
    rows = _gather_bytes(session, tensor, limits, policy, capturing=capturing)
    tensor.copy_(rows[src].view(tensor.dtype).view(tensor.shape))
    return tensor


def gather(session, plan: Plan, inp: torch.Tensor, dst: int, dim: int, rank: int,
           limits: SessionLimits, policy: Policy, *, capturing: bool) -> torch.Tensor | None:
    """Concatenation along ``dim`` on rank ``dst``; None on the other ranks."""
    world = limits.world
    if plan.method == "empty":
        shape = list(inp.shape)
        shape[dim % inp.dim()] *= world
        return torch.empty(shape, dtype=inp.dtype, device=inp.device) if rank == dst else None
    rows = _gather_bytes(session, inp, limits, policy, capturing=capturing)
    if rank != dst:
        return None
    parts = [rows[w].view(inp.dtype).view(inp.shape) for w in range(world)]
    return torch.cat(parts, dim=dim % inp.dim())


def all_to_all_single(session, plan: Plan, output: torch.Tensor, input_: torch.Tensor,
                      rank: int, limits: SessionLimits, policy: Policy, *,
                      capturing: bool) -> torch.Tensor:
    """Equal-split all-to-all of flat buffers: chunk ``p`` of ``input_`` to rank ``p``."""
    world = limits.world
    if plan.method == "empty":
        return output
    src = input_.contiguous()
    if plan.method == "scatter":
        if plan.ops == 1:
            session.all_to_all(src, output)
            return output
        item = src.element_size()
        block = src.numel() * item // world
        step = max(PACK, limits.scatter_op_bytes // world // PACK * PACK)
        flat_in, flat_out = src.view(-1), output.view(-1)
        for offset, count in _ranges(block, step):
            session.all_to_all(flat_in[offset // item:], flat_out[offset // item:],
                               chunk_bytes=count, src_stride_bytes=block, dst_stride_bytes=block)
        return output
    if plan.method != "gather_pick":
        raise PlanError(f"unknown all-to-all method {plan.method}")
    rows = _gather_bytes(session, src, limits, policy, capturing=capturing)   # [world, nbytes]
    per = rows.shape[1] // world
    target = output.reshape(-1).view(torch.uint8).view(world, per)
    for source in range(world):
        target[source].copy_(rows[source, rank * per: (rank + 1) * per])
    return output
