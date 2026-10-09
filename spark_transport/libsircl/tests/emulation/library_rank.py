#!/usr/bin/env python3
"""One rank of the library's GPU emulation: NCCL collectives through libsircl, checked bit for bit.

Started by ``run_library.py`` once per rank, every rank a process on the same GPU, with the library's
emulation transport (``LIBSIRCL_TRANSPORT=emulation``: the shared-memory verbs stand-in). The process
uses torch only to allocate tensors, create streams and capture CUDA graphs, as an engine's NCCL binding
does; the library itself is loaded with ctypes and called through its NCCL C API.

Every rank generates every rank's inputs from the case seed (SplitMix64, the same bits on every platform), so
it computes the reference itself: the float32 sum in rank order, rounded once to the dtype, which is
what SIRCL's one-shot and two-shot all-reduce produce. With ``--golden DIR`` it also compares each
output with the bytes the SIRCL Python session produced for the same inputs (``sircl_golden.py``), or
with their SHA-256 digests (``--golden FILE.json``, for hosts that do not hold the bytes). Under the
library's cycle plan (the receipt's ``cycle_plan``: the ring schedules from 8 MiB on a ring that closes over
cables) the cases the plan runs as a ring op are compared with the ring session's digests instead
(``--golden-ring FILE.json``, by default the ``-ring`` sibling of ``--golden``, ``w8.json`` ->
``w8-ring.json``). ``LIBRARY_RANK_EXPECT_CYCLE_PLAN=1`` (or ``0``) adds a check that the receipt's
``cycle_plan`` is that value.

Checks: ncclAllReduce of float16, bfloat16 and float32 sums at sizes from one element to several
large-message pieces (tails below 16 bytes, unaligned buffers, in place), consecutive calls alternating
between two streams with no synchronization between them; ncclAllGather (bytes of any dtype, tiles larger
than a slot, padded and unaligned shards, in place, ranks whose buffers differ in alignment) and
ncclReduceScatter (padded, unaligned, in place, several ops per call); CUDA graph capture of all three and
two replays with new inputs; ncclBroadcast with no send buffer on the other ranks, in-place ncclBcast and
ncclReduce to every root with the other ranks' buffers untouched. Through the fold pack: ncclAllReduce of
every datatype with every built-in op, ncclReduce and ncclReduceScatter of other datatypes and ops,
compared with the host model in ``fold_model.py``. ncclAlltoAll, ncclGather and ncclScatter (padded,
unaligned, in place, several ops per call, NULL buffers on the ranks that do not use them); one CUDA graph
of a fold all-reduce, an all-to-all, a gather and a scatter, replayed. ncclSend and ncclRecv: on two ranks
outside and inside groups, both directions, in-order matching, torch's all-to-all pattern and a graph; on
more ranks, a send to the rank itself carried and, without point-to-point channels (LIBSIRCL_P2P_CHANNELS
unset or off; p2p_channels.py tests them), a send to another rank refused. Rank-local staging: the
collectives of every selection point with one rank's buffers unaligned or in place, eager and captured
cold, launch and carry the same ops on every rank as with aligned buffers (receipt counts). On a pair,
one-way exchanges whose empty direction alternates between the ranks across link-slot wrap, each round
followed by two-way calls. An op that is not built in refused and counted; the receipt,
ncclCommGetAsyncError, finalize and destroy.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unique_id  # noqa: E402
from unique_id import UniqueId  # noqa: E402

NCCL_DTYPES = {"uint8": 1, "float16": 6, "float32": 7, "bfloat16": 9}




def splitmix64(seed: int, count: int):
    """`count` outputs of SplitMix64 from `seed`, in NumPy uint64 arithmetic: the same bits on every
    platform and library version, which SIRCL-session digests made on one host and checked on another
    need."""
    import numpy as np

    one = np.uint64
    with np.errstate(over="ignore"):
        z = one(seed % (1 << 64)) + (np.arange(1, count + 1, dtype=np.uint64) * one(0x9E3779B97F4A7C15))
        z = (z ^ (z >> one(30))) * one(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> one(27))) * one(0x94D049BB133111EB)
    return z ^ (z >> one(31))


def inputs_for(torch, world: int, count: int, dtype, seed: int):
    """Every rank's `count` elements for a case seed: uint8 bytes, or values uniform on [-4, 4) in steps
    of 2^-21 (exact in float32), rounded to nearest even into the dtype."""
    import numpy as np

    out = []
    for rank in range(world):
        bits = splitmix64(seed * 1009 + rank, count)
        if dtype == torch.uint8:
            out.append(torch.from_numpy((bits >> np.uint64(56)).astype(np.uint8)))
        else:
            steps = (bits >> np.uint64(40)).astype(np.int64) - (1 << 23)
            out.append(torch.from_numpy(steps.astype(np.float32) / np.float32(1 << 21)).to(dtype))
    return out


def inputs_digest(torch, inputs) -> str:
    """SHA-256 of every rank's input bytes in rank order: recorded beside the SIRCL session's outputs, so
    a host can tell different inputs from different outputs."""
    digest = hashlib.sha256()
    for tensor in inputs:
        digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


# The library's minimums (SIRCL's): the smallest all-reduce message, all-gather output and reduce-scatter
# input that auto runs as a chain op and that a ring schedule runs as a ring op.
CHAIN_MINS = {"reduce": 8 << 20, "gather": 8 << 20, "scatter": 4 << 20}
RING_MINS = {"reduce": 4 << 20, "gather": 8 << 20, "scatter": 4 << 20}


# The library's cycle plan (src/engine.c, CYCLE_RING_MIN_BYTES): a communicator of three or more ranks
# without SIRCL_LARGE_SCHEDULE, given a ring plan, whose ring closes over cables only, runs the ring
# schedules from this size (all-reduce message, all-gather output, reduce-scatter input), the pieces below.
CYCLE_RING_MIN_BYTES = 8 << 20
# The cycle plan as the library decided it, by communicator size, from a receipt's "cycle_plan"
# (note_cycle_plan, which the harnesses call after creating the communicator they check). Without a note,
# cycle_plan() models the decision from this process's settings, which is exact when every rank has the
# same settings (the emulation's runs); on a fabric each rank's forward windows differ, so the harnesses
# note the receipt's decision.
CYCLE_PLANS: dict = {}


def note_cycle_plan(receipt: dict, world: int) -> bool:
    CYCLE_PLANS[world] = bool(receipt.get("cycle_plan"))
    return CYCLE_PLANS[world]


def cycle_plan(world: int) -> bool:
    if world in CYCLE_PLANS:
        return CYCLE_PLANS[world]
    if world < 3 or os.environ.get("SIRCL_LARGE_SCHEDULE"):
        return False
    if not (os.environ.get("LIBSIRCL_RING_WINDOW") or os.environ.get("SIRCL_CCL_RING_WINDOW")):
        return False
    windows = os.environ.get("LIBSIRCL_FORWARD_WINDOWS") or os.environ.get("SIRCL_CCL_FORWARD_WINDOWS") or ""
    return not any(int(value or 0) for entry in windows.split(",") if "=" in entry
                   for value in entry.split("=", 1)[1].split("/"))


def setting(name: str, world: int = 0):
    """A schedule setting as the library runs it: the environment's value, or under the cycle plan its
    values for the unset ones (the ring schedules from CYCLE_RING_MIN_BYTES)."""
    value = os.environ.get(name)
    if value or not world or not cycle_plan(world):
        return value
    if name in ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE"):
        return "ring"
    if name == "SIRCL_RING_MIN_BYTES":
        return str(CYCLE_RING_MIN_BYTES)
    return value


def _minimum(name: str, collective: str, defaults, world: int = 0) -> int:
    text = setting(name, world)
    if text:
        return int(text)
    value = defaults[collective]
    if defaults is CHAIN_MINS and world and cycle_plan(world):
        # Under the cycle plan the chain minimums rise to the ring minimum: the pieces stay below it.
        value = max(value, _minimum("SIRCL_RING_MIN_BYTES", collective, RING_MINS, world))
    return value


def _schedule(name: str, nbytes: int, collective: str, world: int = 0) -> str:
    """The schedule variable `name` as the library runs it for a collective of `nbytes`: ring runs as auto
    without the ring plan (LIBSIRCL_RING_WINDOW) or below SIRCL_RING_MIN_BYTES."""
    schedule = setting(name, world) or "pieces"
    if schedule == "ring" and (not (os.environ.get("LIBSIRCL_RING_WINDOW") or os.environ.get("SIRCL_CCL_RING_WINDOW"))
                               or nbytes < _minimum("SIRCL_RING_MIN_BYTES", collective, RING_MINS, world)):
        return "auto"
    return schedule


# The library's pair default (SIRCL_LARGE_SCHEDULE unset, two ranks): all-reduces from this size run as one
# ring op on a cabled pair or one given a ring plan.
PAIR_RING_MIN_BYTES = 2 << 20


def pair_default(world: int):
    """The all-reduce schedule the library's pair default gives: "ring" for a pair joined by cables (no
    forward window in LIBSIRCL_FORWARD_WINDOWS) or given a ring plan (LIBSIRCL_RING_WINDOW), "auto" for a
    relayed pair without one; None for other groups or when SIRCL_LARGE_SCHEDULE is set."""
    if world != 2 or os.environ.get("SIRCL_LARGE_SCHEDULE"):
        return None
    ring_plan = os.environ.get("LIBSIRCL_RING_WINDOW") or os.environ.get("SIRCL_CCL_RING_WINDOW")
    windows = os.environ.get("LIBSIRCL_FORWARD_WINDOWS") or os.environ.get("SIRCL_CCL_FORWARD_WINDOWS") or ""
    relayed = any(int(value or 0) for entry in windows.split(",") if "=" in entry
                  for value in entry.split("=", 1)[1].split("/"))
    return "ring" if ring_plan or not relayed else "auto"


def large_plan(nbytes: int, world: int) -> tuple[str, int]:
    """The op the library's all-reduce of `nbytes` starts with, under SIRCL_LARGE_SCHEDULE: ("ring", the
    largest prefix of W equal chunks of whole packs), ("chain", the 16-byte-aligned body) or ("pieces", 0).
    Sizes and settings decide it, not the buffers' alignment."""
    body = nbytes // 16 * 16
    configured = setting("SIRCL_LARGE_SCHEDULE", world)
    pair = pair_default(world)
    if world < 2 or not body or (pair or configured or "pieces") == "pieces":
        return "pieces", 0
    if pair == "ring":
        ring_min = int(os.environ.get("SIRCL_RING_MIN_BYTES") or PAIR_RING_MIN_BYTES)
        schedule = "ring" if nbytes >= ring_min else "auto"
    else:
        schedule = pair or _schedule("SIRCL_LARGE_SCHEDULE", nbytes, "reduce", world)
    if schedule == "ring":
        ring_bytes = body // (16 * world) * 16 * world
        return ("ring", ring_bytes) if ring_bytes and ring_bytes // world < 1 << 31 else ("pieces", 0)
    if schedule == "chain" or body >= _minimum("SIRCL_CHAIN_MIN_BYTES", "reduce", CHAIN_MINS, world):
        return "chain", body
    return "pieces", 0


def scatter_path(chunk_bytes: int, world: int):
    """The link op the library's float16, bfloat16 or float32 reduce-scatter of `chunk_bytes` per rank runs
    as under SIRCL_SCATTER_SCHEDULE: "ring", "chain" or None (the scatter ops)."""
    if world < 2 or not chunk_bytes or chunk_bytes % 16:
        return None
    if pair_default(world) == "ring" and not os.environ.get("SIRCL_SCATTER_SCHEDULE"):
        # The pair plan: the ring from 4 MiB of input (SIRCL_RING_MIN_BYTES overrides it), pieces below.
        minimum = int(os.environ.get("SIRCL_RING_MIN_BYTES") or (4 << 20))
        return "ring" if chunk_bytes * world >= minimum else None
    schedule = _schedule("SIRCL_SCATTER_SCHEDULE", chunk_bytes * world, "scatter", world)
    if schedule in ("ring", "pieces"):
        return None if schedule == "pieces" else "ring"
    if schedule == "chain" or chunk_bytes * world >= _minimum("SIRCL_CHAIN_MIN_BYTES", "scatter", CHAIN_MINS, world):
        return "chain"
    return None


def chain_reduce_scatter(torch, inputs, order) -> list:
    """Every rank's chain reduce-scatter output (rank order): the owner at chain index j stores
    round((L + x) + R), L the per-hop-rounded partial of chain indices 0 .. j-1, R that of W-1 .. j+1."""
    world = len(inputs)
    chunk = inputs[0].numel() // world
    dtype = inputs[0].dtype
    outputs = [None] * world
    for index, owner in enumerate(order):
        part = [t[owner * chunk:(owner + 1) * chunk] for t in inputs]

        def fold(indices):
            total = None
            for i in indices:
                value = part[order[i]]
                total = value.clone() if total is None else (total.float() + value.float()).to(dtype)
            return total

        left, right = fold(range(index)), fold(range(world - 1, index, -1))
        total = part[owner].float()
        if left is not None:
            total = left.float() + total
        if right is not None:
            total = total + right.float()
        outputs[owner] = total.to(dtype)
    return outputs


def ring_reduce_scatter(torch, inputs, order) -> list:
    """Every rank's ring reduce-scatter output (rank order): the owner at ring index k gets the per-hop
    rounding of the values of ring indices k+1, k+2, ..., k added in that order."""
    world = len(inputs)
    chunk = inputs[0].numel() // world
    dtype = inputs[0].dtype
    outputs = [None] * world
    for index, owner in enumerate(order):
        part = [t[owner * chunk:(owner + 1) * chunk] for t in inputs]
        total = part[order[(index + 1) % world]].clone()
        for step in range(2, world + 1):
            total = (total.float() + part[order[(index + step) % world]].float()).to(dtype)
        outputs[owner] = total
    return outputs


def scatter_reference(torch, inputs, order=None) -> list:
    """Every rank's output of the library's reduce-scatter of `inputs` (W chunks each), in rank order."""
    world = len(inputs)
    chunk = inputs[0].numel() // world
    order = order or list(range(world))
    path = scatter_path(chunk * inputs[0].element_size(), world)
    if path == "ring":
        return ring_reduce_scatter(torch, inputs, order)
    if path == "chain":
        return chain_reduce_scatter(torch, inputs, order)
    total = reference(torch, inputs)
    return [total[r * chunk:(r + 1) * chunk] for r in range(world)]


def chain_sum(torch, parts, order):
    """The chain all-reduce of one op: half A (the first floor(packs / 2) packs) summed from chain index 0
    to W-1, half B from W-1 to 0, each hop the dtype rounding of the float32 sum of the partial and the
    rank's values; `order[i]` indexes `parts` at chain index i."""
    item = parts[0].element_size()
    half = parts[0].numel() * item // 16 // 2 * (16 // item)
    a = parts[order[0]][:half].clone()
    for k in order[1:]:
        a = (a.float() + parts[k][:half].float()).to(parts[0].dtype)
    b = parts[order[-1]][half:].clone()
    for k in reversed(order[:-1]):
        b = (b.float() + parts[k][half:].float()).to(parts[0].dtype)
    return torch.cat([a, b])


def allreduce_reference(torch, inputs, order=None):
    """The library's all-reduce of `inputs` (any buffer alignment): the ring or chain op its schedule starts
    with (large_plan; the ring op's chunk r is rank r's ring reduce-scatter output), then the rank-order sum
    of the rest, the zero-padded tail included; `order[i]` is the rank at chain index i."""
    world, item = len(inputs), inputs[0].element_size()
    kind, nbytes = large_plan(inputs[0].numel() * item, world)
    if kind == "pieces":
        return reference(torch, inputs)
    order = order or list(range(world))
    count = nbytes // item
    parts = [t[:count] for t in inputs]
    head = torch.cat(ring_reduce_scatter(torch, parts, order)) if kind == "ring" else chain_sum(torch, parts, order)
    if count == inputs[0].numel():
        return head
    return torch.cat([head, reference(torch, [t[count:] for t in inputs])])


def reference(torch, inputs):
    total = inputs[0].float().clone()
    for tensor in inputs[1:]:
        total += tensor.float()
    return total.to(inputs[0].dtype)


def same_bits(torch, a, b) -> bool:
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[a.element_size()]
    return bool(torch.equal(a.contiguous().view(view), b.contiguous().view(view)))


def cases(max_piece: int):
    """(name, dtype, element count, mode) where mode is eager, unaligned, inplace or streams."""
    sizes = [8, 16, 48, 4096, 65552, 131072, 131088, 2 << 20, max_piece, max_piece + 6,
             2 * max_piece + 2 * (1 << 20) + 18]
    result = []
    for dtype in ("bfloat16", "float16", "float32"):
        item = 4 if dtype == "float32" else 2
        for count in (1, 3):
            result.append((f"{dtype} {count * item} B", dtype, count, "eager"))
        for nbytes in sizes:
            count = max(1, nbytes // item)
            result.append((f"{dtype} {count * item} B", dtype, count, "eager"))
        for nbytes in (4096 + 2 * item, 131072 + 6 * item):
            result.append((f"{dtype} {nbytes // item * item} B unaligned", dtype, nbytes // item, "unaligned"))
        result.append((f"{dtype} 65536 B in place", dtype, 65536 // item, "inplace"))
        result.append((f"{dtype} 4 x 32 KiB alternating streams", dtype, 32768 // item, "streams"))
    return result


def gather_scatter_cases(max_piece: int):
    """(name, kind, dtype, count, mode): count is the shard of an all-gather and the received chunk of a
    reduce-scatter, in elements."""
    result = []
    for dtype in ("bfloat16", "uint8", "float32"):
        item = {"bfloat16": 2, "uint8": 1, "float32": 4}[dtype]
        for nbytes in (16, 4096, 65552, max_piece + 32, 2 * max_piece + 4096):
            result.append((f"all-gather {dtype} {nbytes} B shards", "all_gather", dtype, nbytes // item, "eager"))
        for nbytes in (item, 1000 // item * item, 3 * max_piece // 2 + 2 * item):
            result.append((f"all-gather {dtype} {nbytes} B shards (padded)", "all_gather", dtype, nbytes // item,
                           "eager"))
        result.append((f"all-gather {dtype} 8192 B shards unaligned", "all_gather", dtype, 8192 // item, "unaligned"))
        result.append((f"all-gather {dtype} 8192 B shards in place", "all_gather", dtype, 8192 // item, "inplace"))
    for dtype in ("bfloat16", "float16", "float32"):
        item = 4 if dtype == "float32" else 2
        for nbytes in (16, 4096, 65552, max_piece // 2 + 48, max_piece + 4096):
            result.append((f"reduce-scatter {dtype} {nbytes} B chunks", "reduce_scatter", dtype, nbytes // item,
                           "eager"))
        for nbytes in (item, 1000 // item * item + item):
            result.append((f"reduce-scatter {dtype} {nbytes} B chunks (padded)", "reduce_scatter", dtype,
                           nbytes // item, "eager"))
        result.append((f"reduce-scatter {dtype} 8192 B chunks unaligned", "reduce_scatter", dtype, 8192 // item,
                       "unaligned"))
        result.append((f"reduce-scatter {dtype} 8192 B chunks in place", "reduce_scatter", dtype, 8192 // item,
                       "inplace"))
    return result


def run_fold_and_routing_checks(torch, lib, comm, world, rank, max_piece, stream, seed_base, report) -> int:
    """The fold pack's reductions, ncclAlltoAll, ncclGather and ncclScatter, an all-gather whose ranks differ
    in buffer alignment, and a CUDA graph of the four; returns the number of calls made under capture.

    Fold outputs are compared with the host model in ``fold_model.py`` (SIRCL's session has no reductions
    other than the float16, bfloat16 and float32 sum); routing outputs are compared byte for byte with
    the inputs they move."""
    import numpy as np

    import fold_model as fm

    TRANSPORT_TYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}

    gather_tile = max_piece // 16 * 16
    scatter_tile = gather_tile // world // 16 * 16
    handle = ctypes.c_void_p(stream.cuda_stream)
    for name in ("ncclAlltoAll",):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p]
    for name in ("ncclGather", "ncclScatter"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p]

    def device(data: bytes, offset: int = 0):
        """A device copy of `data` starting `offset` bytes into a fresh allocation, and the allocation."""
        whole = torch.zeros(len(data) + offset + 16, dtype=torch.uint8, device="cuda")
        view = whole[offset:offset + len(data)]
        if data:
            view.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda())
        # The fill runs on torch's current stream and the library on another: order them.
        torch.cuda.synchronize()
        return view, whole

    def address(view) -> ctypes.c_void_p:
        return ctypes.c_void_p(view.data_ptr() if view is not None else 0)

    def host(view) -> bytes:
        return view.cpu().numpy().tobytes()

    def random_bytes(seed: int, source: int, size: int) -> bytes:
        return np.random.default_rng(seed * 1009 + source).integers(0, 256, size, dtype=np.uint8).tobytes()

    def checked(name, run):
        try:
            rc, ok, detail = run()
            if rc:
                report(name, False, f"result {rc}: {lib.ncclGetLastError(comm).decode()}")
            else:
                report(name, ok, detail[1] if ok else detail[0])
        except Exception as error:  # noqa: BLE001
            report(name, False, f"{type(error).__name__}: {error}")

    # -- all-reduce and reduce through the fold pack --------------------------------------------------
    fold_cases = [(t, op, 1027, "eager") for t in fm.TYPES for op in fm.OPS]
    fold_cases += [("int32", "sum", (2 * gather_tile + 4) // 4, "eager"), ("uint8", "max", 1, "eager"),
                   ("int64", "max", 4099, "unaligned"), ("bfloat16", "avg", 65539, "inplace"),
                   ("float8e4m3", "sum", 3, "eager"), ("float32", "avg", 70001, "eager")]
    for index, (tname, op, count, mode) in enumerate(fold_cases):
        enum, item, _ = fm.TYPES[tname]
        values = fm.inputs(torch, tname, op, world, count, seed_base + 1000 + index)
        want = fm.fold(torch, tname, op, [v for _, v in values], world)
        if op == "sum" and tname in TRANSPORT_TYPES:
            # The transport kernels carry these sums; under a chain or ring schedule they round per hop.
            tensors = [torch.frombuffer(bytearray(data), dtype=torch.uint8).view(TRANSPORT_TYPES[tname])
                       for data, _ in values]
            want = allreduce_reference(torch, tensors).view(torch.uint8).numpy().tobytes()

        def run(values=values, enum=enum, item=item, count=count, mode=mode, op=op, want=want):
            offset = item if mode == "unaligned" else 0
            x, _ = device(values[rank][0], offset)
            y = x if mode == "inplace" else device(b"\0" * len(want), offset)[0]
            rc = lib.ncclAllReduce(address(x), address(y), count, enum, fm.OPS[op], comm, handle)
            torch.cuda.synchronize()
            got = host(y)
            return rc, got == want, ("differs from the host model", "equals the host model")
        checked(f"all-reduce {tname} {op} {count * item} B {mode} (fold)", run)
    for index, (tname, op, count) in enumerate((("int64", "sum", 5000), ("float16", "max", 777))):
        enum, item, _ = fm.TYPES[tname]
        root = (index + 1) % world
        values = fm.inputs(torch, tname, op, world, count, seed_base + 1100 + index)
        want = fm.fold(torch, tname, op, [v for _, v in values], world)

        def run(values=values, enum=enum, count=count, op=op, want=want, root=root):
            x, _ = device(values[rank][0])
            y = device(b"\0" * len(want))[0] if rank == root else None
            rc = lib.ncclReduce(address(x), address(y), count, enum, fm.OPS[op], root, comm, handle)
            torch.cuda.synchronize()
            if rank != root:
                return rc, True, ("", "not the root; no receive buffer")
            return rc, host(y) == want, ("the root's result differs from the host model", "equals the host model")
        checked(f"reduce {tname} {op} {count} elements to rank {root} (fold)", run)

    # -- reduce-scatter through the fold pack (all-to-all, then fold) --------------------------------------
    scatter_cases = [("int32", "sum", 4099, "eager"), ("float64", "max", (2 * scatter_tile + 16) // 8, "eager"),
                     ("bfloat16", "avg", 4096, "unaligned"), ("float8e5m2", "sum", 1000, "inplace"),
                     ("uint64", "prod", 33, "eager")]
    for index, (tname, op, count, mode) in enumerate(scatter_cases):
        enum, item, _ = fm.TYPES[tname]
        values = fm.inputs(torch, tname, op, world, world * count, seed_base + 1200 + index)
        whole = fm.fold(torch, tname, op, [v for _, v in values], world)
        want = whole[rank * count * item:(rank + 1) * count * item]

        def run(values=values, enum=enum, item=item, count=count, mode=mode, op=op, want=want):
            offset = item if mode == "unaligned" else 0
            x, _ = device(values[rank][0], offset)
            y = x[rank * count * item:(rank + 1) * count * item] if mode == "inplace" else device(
                b"\0" * len(want), offset)[0]
            rc = lib.ncclReduceScatter(address(x), address(y), count, enum, fm.OPS[op], comm, handle)
            torch.cuda.synchronize()
            return rc, host(y) == want, ("differs from the host model", "equals the host model")
        checked(f"reduce-scatter {tname} {op} {count * item} B chunks {mode} (fold)", run)

    # -- ncclAlltoAll -------------------------------------------------------------------------------------
    alltoall_cases = [("uint8", 16, "eager"), ("bfloat16", 500, "eager"), ("float32", (2 * scatter_tile + 48) // 4,
                                                                           "eager"),
                      ("uint8", 8192, "inplace"), ("float16", 4096, "unaligned"),
                      # On a pair under the pair plan, one exchange each: staged in place and unaligned.
                      ("uint8", 1 << 20, "inplace"), ("float16", (1 << 20) // 2, "unaligned")]
    for index, (tname, count, mode) in enumerate(alltoall_cases):
        enum, item, _ = fm.TYPES[tname]
        chunk = count * item
        sent = [random_bytes(seed_base + 1300 + index, source, world * chunk) for source in range(world)]
        want = b"".join(sent[source][rank * chunk:(rank + 1) * chunk] for source in range(world))

        def run(sent=sent, enum=enum, item=item, count=count, mode=mode, want=want):
            offset = item if mode == "unaligned" else 0
            x, _ = device(sent[rank], offset)
            y = x if mode == "inplace" else device(b"\0" * len(want), offset)[0]
            rc = lib.ncclAlltoAll(address(x), address(y), count, enum, comm, handle)
            torch.cuda.synchronize()
            return rc, host(y) == want, ("differs from the chunks the ranks sent", "")
        checked(f"all-to-all {tname} {chunk} B chunks {mode}", run)

    # -- ncclGather and ncclScatter (other ranks pass NULL for the buffer they do not use) ----------------
    routing = [("bfloat16", 2048, "eager"), ("uint8", 1001, "eager"), ("float32", (gather_tile + 64) // 4, "eager"),
               ("uint8", 4096, "inplace")]
    for index, (tname, count, mode) in enumerate(routing):
        enum, item, _ = fm.TYPES[tname]
        shard = count * item
        root = index % world
        sent = [random_bytes(seed_base + 1400 + index, source, shard) for source in range(world)]

        def run(sent=sent, enum=enum, count=count, mode=mode, root=root, shard=shard):
            if rank == root:
                y, _ = device(b"\0" * (world * shard))
                if mode == "inplace":
                    x = y[root * shard:(root + 1) * shard]
                    x.copy_(torch.frombuffer(bytearray(sent[rank]), dtype=torch.uint8).cuda())
                    torch.cuda.synchronize()
                else:
                    x, _ = device(sent[rank])
            else:
                x, y = device(sent[rank])[0], None
            rc = lib.ncclGather(address(x), address(y), count, enum, root, comm, handle)
            torch.cuda.synchronize()
            return rc, rank != root or host(y) == b"".join(sent), ("the root's output differs from the shards", "")
        checked(f"gather {tname} {shard} B shards to rank {root} {mode}", run)
    scatter_routing = [("bfloat16", 2048, "eager"), ("uint8", 1001, "eager"),
                       ("float32", (2 * scatter_tile + 32) // 4, "eager"), ("uint8", 4096, "inplace")]
    for index, (tname, count, mode) in enumerate(scatter_routing):
        enum, item, _ = fm.TYPES[tname]
        chunk = count * item
        root = (index + 1) % world
        source = random_bytes(seed_base + 1500 + index, root, world * chunk)

        def run(source=source, enum=enum, count=count, mode=mode, root=root, chunk=chunk):
            if rank == root:
                x, _ = device(source)
                y = x[root * chunk:(root + 1) * chunk] if mode == "inplace" else device(b"\0" * chunk)[0]
            else:
                x, y = None, device(b"\0" * chunk)[0]
            rc = lib.ncclScatter(address(x), address(y), count, enum, root, comm, handle)
            torch.cuda.synchronize()
            return rc, host(y) == source[rank * chunk:(rank + 1) * chunk], ("differs from the root's chunk", "")
        checked(f"scatter {tname} {chunk} B chunks from rank {root} {mode}", run)

    # -- an all-gather whose ranks differ in alignment: every rank must split it into the same ops -------
    shard = (3 << 20)
    sent = [random_bytes(seed_base + 1600, source, shard) for source in range(world)]

    def mixed():
        x, _ = device(sent[rank], 2 if rank == 1 else 0)
        y, _ = device(b"\0" * (world * shard), 2 if rank == 0 else 0)
        rc = lib.ncclAllGather(address(x), address(y), shard // 2, 9, comm, handle)
        torch.cuda.synchronize()
        return rc, host(y) == b"".join(sent), ("differs from the shards", "")
    checked("all-gather bfloat16 3 MiB shards, rank 0's output and rank 1's input unaligned", mixed)

    # -- one CUDA graph: fold all-reduce, all-to-all, gather and scatter, replayed for new inputs --------
    try:
        fold_count, a2a_count, gather_count, scatter_count = 5000, 1024, 300, 100
        fx = torch.zeros(fold_count * 4, dtype=torch.uint8, device="cuda")
        fy = torch.zeros_like(fx)
        ax = torch.zeros(world * a2a_count * 4, dtype=torch.uint8, device="cuda")
        ay = torch.zeros_like(ax)
        gx = torch.zeros(gather_count * 2, dtype=torch.uint8, device="cuda")
        gy = torch.zeros(world * gather_count * 2, dtype=torch.uint8, device="cuda")
        sx = torch.zeros(world * scatter_count, dtype=torch.uint8, device="cuda")
        sy = torch.zeros(scatter_count, dtype=torch.uint8, device="cuda")
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        chandle = ctypes.c_void_p(capture.cuda_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            codes = [lib.ncclAllReduce(address(fx), address(fy), fold_count, 2, 2, comm, chandle),
                     lib.ncclAlltoAll(address(ax), address(ay), a2a_count, 7, comm, chandle),
                     lib.ncclGather(address(gx), address(gy if rank == 0 else None), gather_count, 9, 0, comm, chandle),
                     lib.ncclScatter(address(sx if rank == world - 1 else None), address(sy), scatter_count, 1,
                                     world - 1, comm, chandle)]
        report("graph capture of a fold all-reduce, all-to-all, gather and scatter", not any(codes),
               "" if not any(codes) else lib.ncclGetLastError(comm).decode())
        for replay, seed in enumerate((9301, 9302)):
            values = fm.inputs(torch, "int32", "max", world, fold_count, seed)
            a2a = [random_bytes(seed + 10, source, world * a2a_count * 4) for source in range(world)]
            gathered = [random_bytes(seed + 20, source, gather_count * 2) for source in range(world)]
            scattered = random_bytes(seed + 30, world - 1, world * scatter_count)
            for tensor, data in ((fx, values[rank][0]), (ax, a2a[rank]), (gx, gathered[rank]), (sx, scattered)):
                tensor.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda())
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            chunk = a2a_count * 4
            good = (host(fy) == fm.fold(torch, "int32", "max", [v for _, v in values], world)
                    and host(ay) == b"".join(a2a[s][rank * chunk:(rank + 1) * chunk] for s in range(world))
                    and (rank != 0 or host(gy) == b"".join(gathered))
                    and host(sy) == scattered[rank * scatter_count:(rank + 1) * scatter_count])
            report(f"graph replay {replay + 1} of a fold all-reduce, all-to-all, gather and scatter", good,
                   "" if good else "a replayed result differs")
    except Exception as error:  # noqa: BLE001
        report("graph of a fold all-reduce, all-to-all, gather and scatter", False, f"{type(error).__name__}: {error}")
    return 4


# Communicators this rank created (the parent and every split child): one receipt file each.
CREATED = {"communicators": 1}


def run_capture_staging_check(torch, lib, comm, world, rank, seed_base, report) -> None:
    """CUDA graph capture of an in-place ncclAlltoAll on a fresh split child. On a pair the overlapping
    buffers go through staging: captured cold, the call takes a graph allocation and is exact on replay
    (a driver or device without stream-ordered allocation refuses it with ncclInvalidUsage and the staging
    message, the same on every rank). After one eager call of the same shape the capture uses the
    communicator's grown staging buffer and two replays with new inputs are exact."""
    lib.ncclAlltoAll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p]
    name = "in-place all-to-all under graph capture on a fresh communicator, cold and after an eager call"
    child = ctypes.c_void_p()
    code = lib.ncclCommSplit(comm, 0, rank, ctypes.byref(child), None)
    if code:
        report(name, False, f"split: result {code}: {lib.ncclGetLastError(None).decode()}")
        return
    CREATED["communicators"] += 1
    # On a pair an all-to-all of 1 MiB blocks or more is one pair exchange, staged when its buffers overlap.
    chunk = (1 << 20) if world == 2 else 4096
    total = chunk * world

    def block(seed, source):
        generator = torch.Generator().manual_seed(seed * 1009 + source)
        return torch.randint(0, 256, (total,), generator=generator, dtype=torch.uint8)

    def want(seed):
        return torch.cat([block(seed, s)[rank * chunk:(rank + 1) * chunk] for s in range(world)])

    x = torch.zeros(total, dtype=torch.uint8, device="cuda")
    capture = torch.cuda.Stream()
    handle = ctypes.c_void_p(capture.cuda_stream)

    def captured_run(seeds):
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            rc = lib.ncclAlltoAll(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(x.data_ptr()), chunk, 1, child,
                                  handle)
        message = lib.ncclGetLastError(child).decode(errors="replace") if rc else ""
        exact = []
        if rc == 0:
            for seed in seeds:
                x.copy_(block(seed, rank).cuda())
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                exact.append(bool(torch.equal(x.cpu(), want(seed))))
        return rc, message, exact

    try:
        cold, cold_message, cold_exact = captured_run((seed_base + 1901,))
        if receipt_of(lib, child)["links"]["graph_staging"]:
            cold_ok = cold == 0 and len(cold_exact) == 1 and all(cold_exact)
        else:
            cold_ok = (cold == 5 and "staging" in cold_message) or (cold == 0 and all(cold_exact))
        x.copy_(block(seed_base + 1902, rank).cuda())
        torch.cuda.synchronize()
        eager = lib.ncclAlltoAll(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(x.data_ptr()), chunk, 1, child,
                                 handle)
        torch.cuda.synchronize()
        eager_ok = eager == 0 and bool(torch.equal(x.cpu(), want(seed_base + 1902)))
        warm, warm_message, warm_exact = captured_run((seed_base + 1903, seed_base + 1904))
        ok = cold_ok and eager_ok and warm == 0 and len(warm_exact) == 2 and all(warm_exact)
        report(name, ok, f"cold capture {cold} {cold_message[:120]!r} replays exact {cold_exact}; eager {eager} "
                         f"exact {eager_ok}; warm capture {warm} {warm_message[:120]!r} replays exact {warm_exact}")
    except Exception as error:  # noqa: BLE001
        report(name, False, f"{type(error).__name__}: {error}")
    finally:
        report("ncclCommDestroy of the capture-staging child", lib.ncclCommDestroy(child) == 0)


def run_split_checks(torch, lib, comm, world, rank, stream, seed_base, report) -> None:
    """ncclCommSplit with one color and keys that reverse the rank order: the child communicator's ranks
    are the parent's in reverse, it keeps each process's route-map position, and an all-reduce on it sums
    in the child's rank order."""
    lib.ncclCommSplit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_void_p]
    lib.ncclCommCount.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.ncclCommUserRank.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    child = ctypes.c_void_p()
    name = "ncclCommSplit reversing the rank order, then an all-reduce on the child"
    try:
        code = lib.ncclCommSplit(comm, 0, world - 1 - rank, ctypes.byref(child), None)
        if code:
            report(name, False, f"result {code}: {lib.ncclGetLastError(None).decode()}")
            return
        CREATED["communicators"] += 1
        count, child_rank = ctypes.c_int(-1), ctypes.c_int(-1)
        lib.ncclCommCount(child, ctypes.byref(count))
        lib.ncclCommUserRank(child, ctypes.byref(child_rank))
        values = inputs_for(torch, world, 4096, torch.bfloat16, seed_base + 1800)
        # Child rank c is parent rank W-1-c; the child's chain order follows the positions, the parent ranks.
        child_values = [values[world - 1 - c] for c in range(world)]
        want = allreduce_reference(torch, child_values, order=list(range(world - 1, -1, -1)))
        x = values[rank].cuda()
        y = torch.empty_like(x)
        torch.cuda.synchronize()
        code = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), 4096, 9, 0, child,
                                 ctypes.c_void_p(stream.cuda_stream))
        torch.cuda.synchronize()
        good = (code == 0 and count.value == world and child_rank.value == world - 1 - rank
                and same_bits(torch, y.cpu(), want))
        report(name, good, "" if good else f"result {code}, count {count.value}, rank {child_rank.value}")
    except Exception as error:  # noqa: BLE001
        report(name, False, f"{type(error).__name__}: {error}")
    finally:
        if child.value:
            report("ncclCommDestroy of the split communicator", lib.ncclCommDestroy(child) == 0)


def run_abort_with_queued_send_check(torch, lib, comm, world, rank, stream, report) -> None:
    """ncclCommAbort of a split child while another thread's group still holds a send queued on it (that
    thread left its group without ncclGroupEnd, as an exception would): the abort returns within seconds,
    and that thread's later ncclGroupEnd fails with ncclInvalidUsage instead of using the freed
    communicator. The send goes to the rank itself, so no peer takes part."""
    import threading

    lib.ncclCommAbort.argtypes = [ctypes.c_void_p]
    name = "ncclCommAbort with a send queued in another thread's open group: returns, and that group fails"
    child = ctypes.c_void_p()
    code = lib.ncclCommSplit(comm, 0, rank, ctypes.byref(child), None)
    if code:
        report(name, False, f"split: result {code}: {lib.ncclGetLastError(None).decode()}")
        return
    CREATED["communicators"] += 1
    x = torch.ones(1024, dtype=torch.uint8, device="cuda")
    torch.cuda.synchronize()
    queued, finish, results = threading.Event(), threading.Event(), {}

    def grouped():
        results["start"] = lib.ncclGroupStart()
        results["send"] = lib.ncclSend(ctypes.c_void_p(x.data_ptr()), 1024, 1, rank, child,
                                       ctypes.c_void_p(stream.cuda_stream))
        queued.set()
        finish.wait(60)
        results["end"] = lib.ncclGroupEnd()

    owner = threading.Thread(target=grouped, daemon=True)
    owner.start()
    queued.wait(30)
    aborter = threading.Thread(target=lambda: results.setdefault("abort", lib.ncclCommAbort(child)), daemon=True)
    begun = time.perf_counter()
    aborter.start()
    aborter.join(15)
    waited = time.perf_counter() - begun
    finish.set()
    owner.join(15)
    ok = (not aborter.is_alive() and results.get("abort") == 0 and results.get("send") == 0
          and results.get("end") == 5)
    report(name, ok, f"abort {results.get('abort')} after {waited:.2f} s, queued send {results.get('send')}, "
                     f"group end {results.get('end')}")


def run_point_to_point_checks(torch, lib, comm, world, rank, max_piece, stream, seed_base, report) -> int:
    """ncclSend and ncclRecv: between the ranks of a two-rank communicator, outside and inside groups
    (both directions with unequal sizes, several calls matched in issue order, sends to the rank itself
    beside the exchange, as torch's all-to-all issues them), and under CUDA graph capture; on a larger
    communicator, a send to another rank refused and a send to the rank itself carried. Returns the
    number of calls made under capture."""
    import numpy as np

    for name in ("ncclSend", "ncclRecv"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p]
    handle = ctypes.c_void_p(stream.cuda_stream)
    tile = max_piece // 16 * 16 // world // 16 * 16

    def payload(seed, source, size):
        return np.random.default_rng(seed * 1009 + source).integers(0, 256, size, dtype=np.uint8).tobytes()

    def device(data, offset=0):
        whole = torch.zeros(len(data) + offset + 16, dtype=torch.uint8, device="cuda")
        view = whole[offset:offset + len(data)]
        if data:
            view.copy_(torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda())
        # The fill runs on torch's current stream and the library on another: order them.
        torch.cuda.synchronize()
        return view

    def address(view):
        return ctypes.c_void_p(view.data_ptr())

    def host(view):
        return view.cpu().numpy().tobytes()

    def checked(name, run):
        try:
            codes, ok = run()
            if any(codes):
                report(name, False, f"results {codes}: {lib.ncclGetLastError(comm).decode()}")
            else:
                report(name, ok, "" if ok else "received bytes differ from the bytes sent")
        except Exception as error:  # noqa: BLE001
            report(name, False, f"{type(error).__name__}: {error}")

    if world != 2:
        def refused():
            x = device(b"\1" * 64)
            code = lib.ncclSend(address(x), 64, 1, (rank + 1) % world, comm, handle)
            return [0], code == 5
        if os.environ.get("LIBSIRCL_P2P_CHANNELS", "") != "on":
            checked(f"a send to another rank of a {world}-rank communicator without point-to-point channels "
                    "refused with ncclInvalidUsage", refused)

        def self_exchange():
            data = payload(seed_base + 1700, rank, 5000)
            x, y = device(data), device(b"\0" * 5000)
            codes = [lib.ncclGroupStart(), lib.ncclSend(address(x), 5000, 1, rank, comm, handle),
                     lib.ncclRecv(address(y), 5000, 1, rank, comm, handle), lib.ncclGroupEnd()]
            torch.cuda.synchronize()
            return codes, host(y) == data
        checked("a send to the rank itself, received in the same group", self_exchange)
        return 0

    peer = 1 - rank
    for index, size in enumerate((1, 4097, 2 * tile + 100, 1 << 20)):
        def one_way(size=size, index=index):
            data = payload(seed_base + 1710 + index, 0, size)
            if rank == 0:
                x = device(data, 1)
                codes = [lib.ncclSend(address(x), size, 1, 1, comm, handle)]
            else:
                y = device(b"\0" * size, 3)
                codes = [lib.ncclRecv(address(y), size, 1, 0, comm, handle)]
            torch.cuda.synchronize()
            return codes, rank == 0 or host(y) == data
        checked(f"send {size} B from rank 0 to rank 1 outside a group", one_way)

    def both_ways():
        sizes = (1000, 70000)
        mine, theirs = payload(seed_base + 1720, rank, sizes[rank]), payload(seed_base + 1720, peer, sizes[peer])
        x, y = device(mine), device(b"\0" * sizes[peer])
        calls = [lambda: lib.ncclSend(address(x), sizes[rank], 1, peer, comm, handle),
                 lambda: lib.ncclRecv(address(y), sizes[peer], 1, peer, comm, handle)]
        codes = [lib.ncclGroupStart()]
        codes += [call() for call in (calls if rank == 0 else calls[::-1])]
        codes.append(lib.ncclGroupEnd())
        torch.cuda.synchronize()
        return codes, host(y) == theirs
    checked("a group exchanging 1000 B and 70000 B in opposite directions", both_ways)

    def symmetric():
        size = 2 << 20
        mine, theirs = payload(seed_base + 1725, rank, size), payload(seed_base + 1725, peer, size)
        x, y = device(mine), device(b"\0" * size)
        codes = [lib.ncclGroupStart(), lib.ncclSend(address(x), size, 1, peer, comm, handle),
                 lib.ncclRecv(address(y), size, 1, peer, comm, handle), lib.ncclGroupEnd()]
        torch.cuda.synchronize()
        return codes, host(y) == theirs
    checked("a group exchanging 2 MiB each way (one pair exchange under the pair plan)", symmetric)

    def in_order():
        sizes = (16, 300, 65536 + 5)
        sent = [payload(seed_base + 1730 + k, 0, n) for k, n in enumerate(sizes)]
        codes = [lib.ncclGroupStart()]
        views = []
        for k, n in enumerate(sizes):
            if rank == 0:
                views.append(device(sent[k]))
                codes.append(lib.ncclSend(address(views[-1]), n, 1, 1, comm, handle))
            else:
                views.append(device(b"\0" * n))
                codes.append(lib.ncclRecv(address(views[-1]), n, 1, 0, comm, handle))
        codes.append(lib.ncclGroupEnd())
        torch.cuda.synchronize()
        return codes, rank == 0 or all(host(views[k]) == sent[k] for k in range(len(sizes)))
    checked("three sends in one group matched in issue order", in_order)

    def refused_in_group():
        size = 4096
        data = payload(seed_base + 1735, 0, size)
        codes = [lib.ncclGroupStart()]
        if rank == 0:
            x = device(data)
            codes.append(lib.ncclSend(address(x), size, 1, 1, comm, handle))
            invalid = lib.ncclSend(address(x), size, 1, 5, comm, handle)
        else:
            y = device(b"\0" * size)
            codes.append(lib.ncclRecv(address(y), size, 1, 0, comm, handle))
            invalid = 4
        codes.append(lib.ncclGroupEnd())
        torch.cuda.synchronize()
        return codes, invalid == 4 and (rank == 0 or host(y) == data)
    checked("inside a group, a send to peer 5 refused with ncclInvalidArgument; the group's valid send and "
            "receive still complete", refused_in_group)

    def receipt_now():
        return receipt_of(lib, comm)

    def all_to_all(count=3000, fused=False):  # bf16 elements per chunk
        chunks = [payload(seed_base + 1740 + source, source, world * count * 2) for source in range(world)]
        x = device(chunks[rank])
        y = device(b"\0" * (world * count * 2))
        before = receipt_now()
        codes = [lib.ncclGroupStart()]
        for other in range(world):
            part = count * 2
            codes.append(lib.ncclSend(ctypes.c_void_p(x.data_ptr() + other * part), count, 9, other, comm, handle))
            codes.append(lib.ncclRecv(ctypes.c_void_p(y.data_ptr() + other * part), count, 9, other, comm, handle))
        codes.append(lib.ncclGroupEnd())
        torch.cuda.synchronize()
        after = receipt_now()
        want = b"".join(chunks[source][rank * count * 2:(rank + 1) * count * 2] for source in range(world))
        got = host(y)
        ok = got == want
        if fused:
            # Under the pair plan one pair exchange whose kernel copies the own block: no separate local copy.
            copies = after["all_reduce"]["local_copies"] - before["all_reduce"]["local_copies"]
            exchanges = after["point_to_point"]["exchanges"] - before["point_to_point"]["exchanges"]
            blocks = [got[s * count * 2:(s + 1) * count * 2] == want[s * count * 2:(s + 1) * count * 2]
                      for s in range(world)]
            one = not after["pair_plan"] or (copies == 0 and exchanges == 1)
            report("torch's all-to-all pattern of 1 MiB blocks: one pair exchange under the pair plan, the own "
                   "block copied in its kernel", not any(codes) and ok and one,
                   "" if ok and one else f"results {codes}, blocks exact by source {blocks}, local copies "
                   f"{copies}, exchanges {exchanges}, pair plan {after['pair_plan']}")
        return codes, ok
    checked("torch's all-to-all pattern: sends and receives with every rank, itself included", all_to_all)
    if world == 2:
        all_to_all(1 << 19, fused=True)

    try:
        size = 12345
        x, y = device(b"\0" * size), device(b"\0" * size)
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        chandle = ctypes.c_void_p(capture.cuda_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            codes = [lib.ncclGroupStart(), lib.ncclSend(address(x), size, 1, peer, comm, chandle),
                     lib.ncclRecv(address(y), size, 1, peer, comm, chandle), lib.ncclGroupEnd()]
        report("graph capture of a send and a receive", not any(codes),
               "" if not any(codes) else lib.ncclGetLastError(comm).decode())
        for replay, seed in enumerate((9401, 9402)):
            x.copy_(torch.frombuffer(bytearray(payload(seed, rank, size)), dtype=torch.uint8).cuda())
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            good = host(y) == payload(seed, peer, size)
            report(f"graph replay {replay + 1} of a send and a receive", good, "" if good else "received bytes differ")
    except Exception as error:  # noqa: BLE001
        report("graph of a send and a receive", False, f"{type(error).__name__}: {error}")
    try:
        size = 1 << 20
        x, y = device(b"\0" * size), device(b"\0" * size)
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        chandle = ctypes.c_void_p(capture.cuda_stream)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            codes = [lib.ncclGroupStart(), lib.ncclSend(address(x), size, 1, peer, comm, chandle),
                     lib.ncclRecv(address(y), size, 1, peer, comm, chandle), lib.ncclGroupEnd()]
        report("graph capture of a 1 MiB send and receive", not any(codes),
               "" if not any(codes) else lib.ncclGetLastError(comm).decode())
        for replay, seed in enumerate((9411, 9412)):
            x.copy_(torch.frombuffer(bytearray(payload(seed, rank, size)), dtype=torch.uint8).cuda())
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            good = host(y) == payload(seed, peer, size)
            report(f"graph replay {replay + 1} of a 1 MiB send and receive", good,
                   "" if good else "received bytes differ")
    except Exception as error:  # noqa: BLE001
        report("graph of a 1 MiB send and receive", False, f"{type(error).__name__}: {error}")
    return 4


def run_alternating_grid_checks(torch, lib, comm, world, rank, stream, seed_base, report) -> int:
    """On a pair: all-gathers and all-to-alls of 1 MiB and 4 MiB blocks, which under the pair plan run as
    ring all-gathers and pair exchanges on the ring all-gather's tail word, six rounds, every output checked
    byte for byte. With LIBSIRCL_LINK_BLOCKS_CYCLE (for example 1,2,4) their grids alternate on that one
    kernel type. Returns the number of all-to-all calls."""
    import numpy as np

    if world != 2:
        return 0
    handle = ctypes.c_void_p(stream.cuda_stream)
    lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    lib.ncclAlltoAll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p]

    def data(seed, source, size):
        return np.random.default_rng(seed * 1013 + source).integers(0, 256, size, dtype=np.uint8).tobytes()

    def device(raw):
        view = torch.frombuffer(bytearray(raw), dtype=torch.uint8).cuda()
        torch.cuda.synchronize()
        return view

    calls, bad = 0, []
    for round_ in range(6):
        for step, (op, size) in enumerate((("gather", 1 << 20), ("alltoall", 4 << 20), ("gather", 4 << 20),
                                           ("alltoall", 1 << 20))):
            seed = seed_base + 1900 + round_ * 8 + step
            if op == "gather":
                shards = [data(seed, r, size) for r in range(world)]
                x, y = device(shards[rank]), torch.zeros(world * size, dtype=torch.uint8, device="cuda")
                torch.cuda.synchronize()
                rc = lib.ncclAllGather(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), size, 0, comm,
                                       handle)
                want = b"".join(shards)
            else:
                sent = [data(seed, r, world * size) for r in range(world)]
                x, y = device(sent[rank]), torch.zeros(world * size, dtype=torch.uint8, device="cuda")
                torch.cuda.synchronize()
                rc = lib.ncclAlltoAll(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), size, 0, comm,
                                      handle)
                calls += 1
                want = b"".join(sent[r][rank * size:(rank + 1) * size] for r in range(world))
            torch.cuda.synchronize()
            if rc or y.cpu().numpy().tobytes() != want:
                bad.append(f"round {round_} {op} {size} B: result {rc}")
    report("all-gathers and all-to-alls on one link kernel type, 24 calls checked", not bad,
           "; ".join(bad[:4]))
    return calls


# Receipt counters of the ops a rank launches and the transport carries. A collective whose shape, dtype and
# settings match on every rank must give every rank the same engine counts whatever its buffers' alignment,
# overlap or role; the native counts are compared per rank across runs of the same calls.
ENGINE_COUNTS = (("pair_exchange", "ops"), ("pair_exchange", "bytes"), ("link_blocks", "ops_by_blocks"),
                 ("all_reduce", "ops"), ("all_gather", "ops"), ("reduce_scatter", "ops"), ("all_to_all", "ops"),
                 ("chain", "ops"), ("chain", "bytes"), ("links", "ops"), ("links", "bytes"),
                 ("point_to_point", "ops"), ("point_to_point", "exchanges"))
NATIVE_COUNTS = (("links", "native_ops"), ("links", "native_items"), ("native", "ops_posted"),
                 ("native", "phases_posted"))
# Counters of work a rank does locally around those ops (staging copies, padded tiles).
LOCAL_COUNTS = (("links", "staged"), ("links", "graph_staged"), ("all_reduce", "unaligned_staged"),
                ("all_reduce", "local_copies"), ("all_gather", "padded"), ("reduce_scatter", "padded"),
                ("all_to_all", "padded"))


def receipt_of(lib, comm) -> dict:
    """The communicator's receipt. It is made at each call from live counters and may grow between the
    call that sizes it and the next (a counter gains a digit), so the read retries until the receipt it
    got was whole (sircl.h)."""
    room = 1 << 14
    while True:
        needed = ctypes.c_size_t(0)
        text = ctypes.create_string_buffer(room)
        lib.sirclGetReceipt(comm, text, room, ctypes.byref(needed))
        if needed.value <= room:
            return json.loads(text.value.decode())
        room = needed.value + 4096


def counts_of(receipt: dict, keys) -> dict:
    return {f"{a}.{b}": receipt.get(a, {}).get(b, 0) for a, b in keys}


def settled_receipt(lib, comm) -> dict:
    """The receipt once the transport's counts stop moving (the progress thread posts a kernel's last
    items after the kernel's own stream may already be idle)."""
    last = None
    for _ in range(60):
        receipt = receipt_of(lib, comm)
        now = counts_of(receipt, ENGINE_COUNTS + NATIVE_COUNTS)
        if now == last:
            return receipt
        last = now
        time.sleep(0.1)
    return receipt


def count_delta(after: dict, before: dict, keys) -> dict:
    """after - before for each counter; a dict counter (ops by kind) keeps its nonzero entries."""
    out = {}
    for key, value in counts_of(after, keys).items():
        old = counts_of(before, keys)[key]
        if isinstance(value, dict):
            entries = {k: value.get(k, 0) - (old or {}).get(k, 0) for k in value}
            out[key] = {k: v for k, v in sorted(entries.items()) if v}
        else:
            out[key] = value - old
    return out


class Call:
    """One collective prepared on device buffers: `launch(comm, stream_handle)` enqueues it and returns
    the result code, `restore()` refills its inputs (in-place calls overwrite them), `poison()` fills its
    outputs with 0x5A, `read()` returns the output bytes (b"" on a rank that keeps none)."""

    def __init__(self, torch, name, launch, inputs, outputs, want):
        self.torch, self.name, self._launch, self.inputs, self.outputs, self.want = (torch, name, launch, inputs,
                                                                                      outputs, want)

    def launch(self, comm, handle):
        return self._launch(comm, handle)

    def restore(self):
        for view, pristine in self.inputs:
            view.copy_(pristine)

    def poison(self):
        for view in self.outputs:
            if all(view.data_ptr() != v.data_ptr() for v, _ in self.inputs):
                view.fill_(0x5A)

    def read(self) -> bytes:
        return b"".join(view.cpu().numpy().tobytes() for view in self.outputs)


def staging_calls(torch, lib, world, rank, seed_base, mode):
    """The collectives of every selection point the library has (one-shot, pieces and padded tiles; the
    chain, ring and link ops; pair exchanges of every root; ncclAlltoAll and, on two ranks, torch's
    all-to-all and one-way point-to-point), each with rank 1's buffers in `mode` ("aligned", "unaligned":
    1 to 4 elements off 16-byte alignment, or "inplace": the output inside the input) and every other
    rank's aligned and apart. The shapes, dtypes and roots are the same in every mode."""
    import numpy as np

    mine = mode if rank == 1 else "aligned"
    BF16, F32, U8 = 9, 7, 1

    def raw(seed, source, size):
        return np.random.default_rng(seed * 1013 + source).integers(0, 256, size, dtype=np.uint8).tobytes()

    def tensor_bytes(t):
        return t.contiguous().view(torch.uint8).numpy().tobytes()

    def place(data: bytes, offset: int):
        whole = torch.zeros(len(data) + offset + 32, dtype=torch.uint8, device="cuda")
        view = whole[offset:offset + len(data)]
        pristine = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda() if data else view.clone()
        view.copy_(pristine)
        return view, pristine

    def ptr(view):
        return ctypes.c_void_p(view.data_ptr() if view is not None else 0)

    calls = []

    def all_reduce(tag, dtype, enum, count, seed):
        inputs = inputs_for(torch, world, count, dtype, seed)
        want = tensor_bytes(allreduce_reference(torch, inputs))
        item = inputs[0].element_size()
        data = tensor_bytes(inputs[rank])
        x, px = place(data, item if mine == "unaligned" else 0)
        y = x if mine == "inplace" else place(b"\0" * len(want), 3 * item if mine == "unaligned" else 0)[0]
        calls.append(Call(torch, f"all-reduce {tag}", lambda c, h: lib.ncclAllReduce(
            ptr(x), ptr(y), count, enum, 0, c, h), [(x, px)], [y], want))

    def all_gather(tag, dtype, enum, count, seed):
        inputs = inputs_for(torch, world, count, dtype, seed)
        want = b"".join(tensor_bytes(t) for t in inputs)
        item, shard = inputs[0].element_size(), count * inputs[0].element_size()
        data = tensor_bytes(inputs[rank])
        if mine == "inplace":
            y, _ = place(b"\0" * len(want), 0)
            x = y[rank * shard:(rank + 1) * shard]
            px = torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()
        else:
            x, px = place(data, item if mine == "unaligned" else 0)
            y, _ = place(b"\0" * len(want), 3 * item if mine == "unaligned" else 0)
        calls.append(Call(torch, f"all-gather {tag}", lambda c, h: lib.ncclAllGather(
            ptr(x), ptr(y), count, enum, c, h), [(x, px)], [y], want))

    def reduce_scatter(tag, dtype, enum, count, seed):
        inputs = inputs_for(torch, world, world * count, dtype, seed)
        want = tensor_bytes(scatter_reference(torch, inputs)[rank])
        item, chunk = inputs[0].element_size(), count * inputs[0].element_size()
        x, px = place(tensor_bytes(inputs[rank]), item if mine == "unaligned" else 0)
        if mine == "inplace":
            y = x[rank * chunk:(rank + 1) * chunk]
        else:
            y, _ = place(b"\0" * chunk, 3 * item if mine == "unaligned" else 0)
        calls.append(Call(torch, f"reduce-scatter {tag}", lambda c, h: lib.ncclReduceScatter(
            ptr(x), ptr(y), count, enum, 0, c, h), [(x, px)], [y], want))

    def all_to_all(tag, chunk, seed):
        sent = [raw(seed, s, world * chunk) for s in range(world)]
        want = b"".join(sent[s][rank * chunk:(rank + 1) * chunk] for s in range(world))
        x, px = place(sent[rank], 1 if mine == "unaligned" else 0)
        y = x if mine == "inplace" else place(b"\0" * len(want), 3 if mine == "unaligned" else 0)[0]
        calls.append(Call(torch, f"all-to-all {tag}", lambda c, h: lib.ncclAlltoAll(
            ptr(x), ptr(y), chunk, U8, c, h), [(x, px)], [y], want))

    def broadcast(tag, size, root, seed):
        data = raw(seed, root, size)
        if rank == root:
            x, px = place(data, 1 if mine == "unaligned" else 0)
            y = x if mine == "inplace" else place(b"\0" * size, 3 if mine == "unaligned" else 0)[0]
            inputs = [(x, px)]
        else:
            x, inputs = None, []
            y, _ = place(b"\0" * size, 3 if mine == "unaligned" else 0)
        calls.append(Call(torch, f"broadcast {tag} from rank {root}", lambda c, h: lib.ncclBroadcast(
            ptr(x), ptr(y), size, U8, root, c, h), inputs, [y], data))

    def reduce(tag, count, root, seed):
        inputs = inputs_for(torch, world, count, torch.bfloat16, seed)
        x, px = place(tensor_bytes(inputs[rank]), 2 if mine == "unaligned" else 0)
        if rank == root:
            want = tensor_bytes(allreduce_reference(torch, inputs))
            y = x if mine == "inplace" else place(b"\0" * len(want), 6 if mine == "unaligned" else 0)[0]
            outputs = [y]
        else:
            want, y, outputs = b"", None, []
        calls.append(Call(torch, f"reduce {tag} to rank {root}", lambda c, h: lib.ncclReduce(
            ptr(x), ptr(y), count, BF16, 0, root, c, h), [(x, px)], outputs, want))

    def gather(tag, shard, root, seed):
        sent = [raw(seed, s, shard) for s in range(world)]
        if rank == root:
            if mine == "inplace":
                y, _ = place(b"\0" * (world * shard), 0)
                x = y[root * shard:(root + 1) * shard]
                px = torch.frombuffer(bytearray(sent[rank]), dtype=torch.uint8).cuda()
            else:
                x, px = place(sent[rank], 1 if mine == "unaligned" else 0)
                y, _ = place(b"\0" * (world * shard), 3 if mine == "unaligned" else 0)
            outputs, want = [y], b"".join(sent)
        else:
            x, px = place(sent[rank], 1 if mine == "unaligned" else 0)
            y, outputs, want = None, [], b""
        calls.append(Call(torch, f"gather {tag} to rank {root}", lambda c, h: lib.ncclGather(
            ptr(x), ptr(y), shard, U8, root, c, h), [(x, px)], outputs, want))

    def scatter(tag, chunk, root, seed):
        source = raw(seed, root, world * chunk)
        if rank == root:
            x, px = place(source, 1 if mine == "unaligned" else 0)
            y = x[root * chunk:(root + 1) * chunk] if mine == "inplace" else place(
                b"\0" * chunk, 3 if mine == "unaligned" else 0)[0]
            inputs = [(x, px)]
        else:
            x, inputs = None, []
            y, _ = place(b"\0" * chunk, 3 if mine == "unaligned" else 0)
        calls.append(Call(torch, f"scatter {tag} from rank {root}", lambda c, h: lib.ncclScatter(
            ptr(x), ptr(y), chunk, U8, root, c, h), inputs, [y], source[rank * chunk:(rank + 1) * chunk]))

    def p2p_all_to_all(tag, part, seed):
        sent = [raw(seed, s, world * part) for s in range(world)]
        want = b"".join(sent[s][rank * part:(rank + 1) * part] for s in range(world))
        x, px = place(sent[rank], 2 if mine == "unaligned" else 0)
        y, _ = place(b"\0" * len(want), 6 if mine == "unaligned" else 0)

        def launch(c, h):
            codes = [lib.ncclGroupStart()]
            for other in range(world):
                codes.append(lib.ncclSend(ctypes.c_void_p(x.data_ptr() + other * part), part, U8, other, c, h))
                codes.append(lib.ncclRecv(ctypes.c_void_p(y.data_ptr() + other * part), part, U8, other, c, h))
            codes.append(lib.ncclGroupEnd())
            return next((code for code in codes if code), 0)
        calls.append(Call(torch, f"torch's all-to-all pattern {tag}", launch, [(x, px)], [y], want))

    def one_way(tag, size, sender, seed):
        data = raw(seed, sender, size)
        if rank == sender:
            x, px = place(data, 1 if mine == "unaligned" else 0)
            calls.append(Call(torch, f"send {tag} from rank {sender}", lambda c, h: lib.ncclSend(
                ptr(x), size, U8, 1 - sender, c, h), [(x, px)], [], b""))
        else:
            y, _ = place(b"\0" * size, 3 if mine == "unaligned" else 0)
            calls.append(Call(torch, f"send {tag} from rank {sender}", lambda c, h: lib.ncclRecv(
                ptr(y), size, U8, sender, c, h), [], [y], data))

    s = seed_base + 2100
    all_reduce("bfloat16 8 KiB", torch.bfloat16, BF16, 4096, s + 1)
    all_reduce("bfloat16 6 MiB", torch.bfloat16, BF16, 3 << 20, s + 2)
    all_reduce("float32 3 MiB + 20 B", torch.float32, F32, (3 << 20) // 4 + 5, s + 3)
    all_gather("bfloat16 3 MiB shards", torch.bfloat16, BF16, 3 << 19, s + 4)
    all_gather("bfloat16 6000 B shards", torch.bfloat16, BF16, 3000, s + 5)
    reduce_scatter("float32 2 MiB chunks", torch.float32, F32, (2 << 20) // 4, s + 6)
    reduce_scatter("float32 16 KiB chunks", torch.float32, F32, 4096, s + 7)
    all_to_all("1 MiB chunks", 1 << 20, s + 8)
    all_to_all("4 KiB chunks", 4096, s + 9)
    for root in (0, 1):
        broadcast("2 MiB", 2 << 20, root, s + 10 + root)
        reduce("bfloat16 1 MiB", 1 << 19, root, s + 12 + root)
        gather("1 MiB shards", 1 << 20, root, s + 14 + root)
        scatter("1 MiB chunks", 1 << 20, 1 - root, s + 16 + root)
    if world == 2:
        p2p_all_to_all("of 1 MiB blocks", 1 << 20, s + 18)
        one_way("1 MiB", 1 << 20, 0, s + 19)
        one_way("1 MiB", 1 << 20, 1, s + 20)
    return calls


def run_rank_local_staging_checks(torch, lib, comm, world, rank, seed_base, report) -> None:
    """A rank-local property of a call (its buffers' alignment, in place or apart, whether it keeps the
    output) must not change which ops any rank launches: otherwise the ranks' kernels wait for transfers
    that never come. On a fresh split child the collectives of staging_calls run three times eagerly: every
    rank aligned, rank 1 unaligned, rank 1 in place; then, on a second fresh child whose staging has never
    grown, they are captured cold into one CUDA graph with rank 1 unaligned and replayed twice. Each run's
    receipt counts of launched and carried ops equal the aligned run's on every rank, every rank's engine
    counts are the same, rank 1 staged locally, and every output is exact."""
    for name in ("ncclSend", "ncclRecv"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p]
    lib.ncclAlltoAll.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                 ctypes.c_void_p]
    for name in ("ncclGather", "ncclScatter"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p]
    child = ctypes.c_void_p()
    code = lib.ncclCommSplit(comm, 0, rank, ctypes.byref(child), None)
    title = "rank-local staging changes no rank's ops"
    if code:
        report(title, False, f"split: result {code}: {lib.ncclGetLastError(None).decode()}")
        return
    CREATED["communicators"] += 1
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    barrier = torch.zeros(16, dtype=torch.float32, device="cuda")

    def barrier_then_receipt():
        """Every rank past its previous calls (a small all-reduce), then this rank's settled receipt."""
        lib.ncclAllReduce(ctypes.c_void_p(barrier.data_ptr()), ctypes.c_void_p(barrier.data_ptr()), 16, 7, 0, child,
                          handle)
        torch.cuda.synchronize()
        return settled_receipt(lib, child)

    def prepare(calls):
        # Outputs first: an in-place call's input lies inside its output, or its output inside its input.
        for call in calls:
            call.poison()
        for call in calls:
            call.restore()
        torch.cuda.synchronize()

    def eager(calls):
        bad = []
        prepare(calls)
        for call in calls:
            rc = call.launch(child, handle)
            torch.cuda.synchronize()
            if rc:
                bad.append(f"{call.name}: result {rc}: {lib.ncclGetLastError(child).decode(errors='replace')[:160]}")
        return bad

    def exactness(calls, outputs):
        return [call.name for call, got in zip(calls, outputs) if got != call.want]

    runs = {}
    try:
        for label, mode in (("aligned", "aligned"), ("unaligned", "unaligned"), ("in place", "inplace")):
            calls = staging_calls(torch, lib, world, rank, seed_base, mode)
            before = barrier_then_receipt()
            bad = eager(calls)
            after = settled_receipt(lib, child)
            runs[label] = (count_delta(after, before, ENGINE_COUNTS), count_delta(after, before, NATIVE_COUNTS),
                           count_delta(after, before, LOCAL_COUNTS), bad + exactness(calls, [c.read() for c in calls]))
            del calls
        # Captured cold on a second fresh child: its staging buffer has never grown.
        cold = ctypes.c_void_p()
        code = lib.ncclCommSplit(comm, 0, rank, ctypes.byref(cold), None)
        if code:
            raise RuntimeError(f"second split: result {code}")
        CREATED["communicators"] += 1
        calls = staging_calls(torch, lib, world, rank, seed_base, "unaligned")
        prepare(calls)
        capture = torch.cuda.Stream()
        chandle = ctypes.c_void_p(capture.cuda_stream)
        before = settled_receipt(lib, cold)
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            codes = [call.launch(cold, chandle) for call in calls]
        captured = settled_receipt(lib, cold)
        capture_bad = [f"{call.name}: result {rc}: {lib.ncclGetLastError(cold).decode(errors='replace')[:160]}"
                       for call, rc in zip(calls, codes) if rc]
        replays = []
        if not capture_bad:
            for _ in range(2):
                prepare(calls)
                start = settled_receipt(lib, cold)
                graph.replay()
                torch.cuda.synchronize()
                replays.append((count_delta(settled_receipt(lib, cold), start, NATIVE_COUNTS),
                                exactness(calls, [c.read() for c in calls])))
        graph_staging = bool(captured["links"].get("graph_staging"))
        runs["captured"] = (count_delta(captured, before, ENGINE_COUNTS), None,
                            count_delta(captured, before, LOCAL_COUNTS), capture_bad)
        del graph, calls
        torch.cuda.synchronize()
        report("ncclCommDestroy of the cold capture child", lib.ncclCommDestroy(cold) == 0)

        # Exact outputs: every run against the references (the transport's arithmetic does not depend on
        # where a rank's buffers lie).
        aligned_engine, aligned_native, aligned_local, aligned_bad = runs["aligned"]
        report("rank-local staging: every call of the aligned run exact", not aligned_bad,
               "; ".join(aligned_bad[:6]))
        def comparable(counts):
            """Under LIBSIRCL_LINK_BLOCKS_CYCLE (a test hook giving successive link ops of a communicator the
            listed blocks per role in turn) the split of a run's link ops over block counts depends on where
            the rotation stood when the run began, which the runs before it on the same child moved; there the
            runs compare the number of link ops instead."""
            if not os.environ.get("LIBSIRCL_LINK_BLOCKS_CYCLE"):
                return counts
            return dict(counts, **{"link_blocks.ops_by_blocks": sum(counts.get("link_blocks.ops_by_blocks",
                                                                                {}).values())})

        for label in ("unaligned", "in place"):
            engine, native, local, bad = runs[label]
            report(f"rank-local staging: every call with rank 1 {label} exact", not bad, "; ".join(bad[:6]))
            same = comparable(engine) == comparable(aligned_engine) and native == aligned_native
            report(f"rank-local staging: with rank 1 {label}, this rank launches and carries the aligned run's ops",
                   same, "" if same else json.dumps({"aligned": [aligned_engine, aligned_native],
                                                     label: [engine, native]})[:1500])
            staged = sum(v for v in local.values() if isinstance(v, int)) > sum(
                v for v in aligned_local.values() if isinstance(v, int))
            report(f"rank-local staging: with rank 1 {label}, rank 1 staged locally", rank != 1 or staged,
                   json.dumps({"aligned": aligned_local, label: local}))
        engine, _, local, capture_bad = runs["captured"]
        report("rank-local staging: captured cold with rank 1 unaligned, every rank captures every call",
               not capture_bad and graph_staging, "; ".join(capture_bad[:4]) or
               ("" if graph_staging else "no stream-ordered allocation on this driver or device"))
        same = comparable(engine) == comparable(aligned_engine)
        report("rank-local staging: the captured calls launch the aligned run's ops", same,
               "" if same else json.dumps({"aligned": aligned_engine, "captured": engine})[:1500])
        # The cold child's staging buffer never grew: every staged call under capture is a graph allocation.
        report("rank-local staging: every call staged under the cold capture took a graph allocation",
               not graph_staging or local.get("links.graph_staged", 0) == local.get("links.staged", 0),
               json.dumps(local))
        for index, (native, wrong) in enumerate(replays):
            report(f"rank-local staging: replay {index + 1} of the cold capture exact, carrying the aligned run's "
                   f"transfers", not wrong and native == aligned_native,
                   "; ".join(wrong[:6]) + ("" if native == aligned_native else json.dumps(
                       {"aligned": aligned_native, "replay": native})))
        if not capture_bad and len(replays) != 2:
            report("rank-local staging: replays of the cold capture", False, f"{len(replays)} replays")

        # Every rank's engine counts of every run, compared across ranks (an all-gather of their digests).
        digests = b"".join(hashlib.sha256(json.dumps(runs[label][0], sort_keys=True).encode()).digest()
                           for label in ("aligned", "unaligned", "in place", "captured"))
        mine = torch.frombuffer(bytearray(digests), dtype=torch.uint8).cuda()
        every = torch.zeros(world * len(digests), dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        rc = lib.ncclAllGather(ctypes.c_void_p(mine.data_ptr()), ctypes.c_void_p(every.data_ptr()), len(digests),
                               NCCL_DTYPES["uint8"], child, handle)
        torch.cuda.synchronize()
        rows = every.cpu().numpy().tobytes()
        alike = rc == 0 and all(rows[r * len(digests):(r + 1) * len(digests)] == digests for r in range(world))
        report("rank-local staging: every rank's engine counts of every run are the same", alike,
               "" if alike else f"result {rc}; this rank's aligned counts {json.dumps(aligned_engine)[:800]}")
    except Exception as error:  # noqa: BLE001
        report(title, False, f"{type(error).__name__}: {error}")
    finally:
        torch.cuda.synchronize()
        report("ncclCommDestroy of the rank-local staging child", lib.ncclCommDestroy(child) == 0)


def run_staging_growth_check(torch, lib, comm, world, rank, seed_base, report) -> None:
    """Eager all-gathers of shards from 16 bytes to 4 MiB, doubling, with rank 1's buffers unaligned, on a
    fresh split child: where the schedule runs them as link ops (the chain or ring schedule from small
    sizes), rank 1's staging buffer grows at every size and keeps every earlier one, nineteen buffers in
    all, and no growth refuses a call its peers launch. Every output exact on every rank."""
    import numpy as np

    child = ctypes.c_void_p()
    code = lib.ncclCommSplit(comm, 0, rank, ctypes.byref(child), None)
    title = "all-gathers of doubling shards, rank 1 unaligned: the staging buffer grows at every size"
    if code:
        report(title, False, f"split: result {code}: {lib.ncclGetLastError(None).decode()}")
        return
    CREATED["communicators"] += 1
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    bad = []
    try:
        for k in range(19):
            shard = 16 << k
            sent = [np.random.default_rng((seed_base + 2500 + k) * 1031 + s).integers(0, 256, shard, dtype=np.uint8)
                    .tobytes() for s in range(world)]
            offset = 1 if rank == 1 else 0
            x = torch.zeros(shard + 32, dtype=torch.uint8, device="cuda")[offset:offset + shard]
            x.copy_(torch.frombuffer(bytearray(sent[rank]), dtype=torch.uint8).cuda())
            y = torch.zeros(world * shard + 32, dtype=torch.uint8, device="cuda")[3 * offset:3 * offset + world * shard]
            torch.cuda.synchronize()
            rc = lib.ncclAllGather(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), shard,
                                   NCCL_DTYPES["uint8"], child, handle)
            torch.cuda.synchronize()
            if rc or y.cpu().numpy().tobytes() != b"".join(sent):
                bad.append(f"{shard} B shards: result {rc} "
                           f"{lib.ncclGetLastError(child).decode(errors='replace')[:120] if rc else 'differs'}")
        links = receipt_of(lib, child)["links"]
        # Every call a staged link op (the chain or ring schedule from small sizes): one buffer per size.
        grown = rank != 1 or links["staged"] != 19 or links["stage_buffers"] == 19
        report(title, not bad and grown, "; ".join(bad[:4]) or f"{links['stage_buffers']} staging buffers, "
                                                                f"{links['staged']} staged ops")
    except Exception as error:  # noqa: BLE001
        report(title, False, f"{type(error).__name__}: {error}")
    finally:
        torch.cuda.synchronize()
        report("ncclCommDestroy of the staging growth child", lib.ncclCommDestroy(child) == 0)


def run_flags_only_checks(torch, lib, comm, world, rank, seed_base, report) -> None:
    """On a pair, the pair exchange of a call that sends one way marks the empty direction's items
    flags-only (the transport writes their flags and no bytes). Absent directions alternate between the
    ranks over consecutive one-way exchanges (broadcast, gather, scatter and reduce of both roots, sends
    each way) of sizes that wrap the link slots at shifting places, and after each round a two-way
    all-to-all and all-gather follow; every output is checked byte for byte. On a fresh split child, so
    the slot positions start at zero. Under the pair plan the receipt shows every call as one pair
    exchange; with a ring window (LIBSIRCL_RING_WINDOW) the lane carries them through its window."""
    if world != 2:
        return
    import numpy as np

    for name in ("ncclSend", "ncclRecv"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                                       ctypes.c_void_p]
    for name in ("ncclGather", "ncclScatter"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p]
    child = ctypes.c_void_p()
    code = lib.ncclCommSplit(comm, 0, rank, ctypes.byref(child), None)
    title = "flags-only directions alternating across slot wrap"
    if code:
        report(title, False, f"split: result {code}: {lib.ncclGetLastError(None).decode()}")
        return
    CREATED["communicators"] += 1
    stream = torch.cuda.Stream()
    h = ctypes.c_void_p(stream.cuda_stream)

    def raw(seed, source, size):
        return np.random.default_rng(seed * 1019 + source).integers(0, 256, size, dtype=np.uint8).tobytes()

    def dev(data):
        return torch.frombuffer(bytearray(data), dtype=torch.uint8).cuda()

    def poisoned(size):
        return torch.full((size,), 0x5A, dtype=torch.uint8, device="cuda")

    def p(t):
        return ctypes.c_void_p(t.data_ptr() if t is not None else 0)

    def host(t):
        return t.cpu().numpy().tobytes()

    # Sizes: whole 16-byte packs of 1 MiB or more per rank (the pair exchange's range under the pair plan),
    # 9 to 13 pieces each against 8 link slots, so every call wraps the slots and the wrap moves.
    sizes = [(3 << 19), (9 << 17) + 4096, (11 << 18) + 16, (5 << 18) - 4096, (13 << 17)]
    bad, exchanges, calls = [], 0, 0
    try:
        before = settled_receipt(lib, child)
        for round_ in range(3):
            for step in range(10):
                size = sizes[(round_ * 3 + step) % len(sizes)]
                seed = seed_base + 2300 + round_ * 16 + step
                # The rank whose direction is empty alternates: step even, rank 1; step odd, rank 0.
                empty = 1 if step % 2 == 0 else 0
                kind = ("broadcast", "gather", "scatter", "reduce", "send")[step // 2]
                rc, got, want = 0, b"", b""
                if kind == "broadcast":
                    root = 1 - empty
                    data = raw(seed, root, size)
                    x, y = (dev(data) if rank == root else None), poisoned(size)
                    torch.cuda.synchronize()
                    rc = lib.ncclBroadcast(p(x), p(y), size, 1, root, child, h)
                    want, out = data, y
                elif kind == "gather":
                    root = empty  # the root sends nothing to the other rank
                    sent = [raw(seed, s, size) for s in range(world)]
                    x, y = dev(sent[rank]), (poisoned(world * size) if rank == root else None)
                    torch.cuda.synchronize()
                    rc = lib.ncclGather(p(x), p(y), size, 1, root, child, h)
                    want, out = (b"".join(sent), y) if rank == root else (b"", None)
                elif kind == "scatter":
                    root = 1 - empty
                    source = raw(seed, root, world * size)
                    x, y = (dev(source) if rank == root else None), poisoned(size)
                    torch.cuda.synchronize()
                    rc = lib.ncclScatter(p(x), p(y), size, 1, root, child, h)
                    want, out = source[rank * size:(rank + 1) * size], y
                elif kind == "reduce":
                    root = empty
                    count = size // 2
                    values = inputs_for(torch, world, count, torch.bfloat16, seed)
                    x = values[rank].cuda()
                    y = torch.full_like(x, 3.0) if rank == root else None
                    torch.cuda.synchronize()
                    rc = lib.ncclReduce(p(x), p(y), count, 9, 0, root, child, h)
                    want = allreduce_reference(torch, values).contiguous().view(torch.uint8).numpy().tobytes() \
                        if rank == root else b""
                    out = y.view(torch.uint8) if y is not None else None
                else:
                    sender = 1 - empty
                    data = raw(seed, sender, size)
                    if rank == sender:
                        x = dev(data)
                        torch.cuda.synchronize()
                        rc = lib.ncclSend(p(x), size, 1, 1 - sender, child, h)
                        want, out = b"", None
                    else:
                        y = poisoned(size)
                        torch.cuda.synchronize()
                        rc = lib.ncclRecv(p(y), size, 1, sender, child, h)
                        want, out = data, y
                torch.cuda.synchronize()
                got = host(out) if out is not None else b""
                calls += 1
                exchanges += 1
                if rc or got != want:
                    bad.append(f"round {round_} {kind} {size} B, rank {empty}'s direction empty: result {rc}"
                               f"{'' if not rc else ' ' + lib.ncclGetLastError(child).decode(errors='replace')[:120]}")
            # Two-way: an all-to-all (one pair exchange, both directions full) and an all-gather (the ring).
            size = sizes[round_ % len(sizes)]
            sent = [raw(seed_base + 2390 + round_, s, world * size) for s in range(world)]
            x, y = dev(sent[rank]), poisoned(world * size)
            torch.cuda.synchronize()
            rc = lib.ncclAlltoAll(p(x), p(y), size, 1, child, h)
            torch.cuda.synchronize()
            if rc or host(y) != b"".join(sent[s][rank * size:(rank + 1) * size] for s in range(world)):
                bad.append(f"round {round_} two-way all-to-all {size} B chunks: result {rc}")
            shards = [raw(seed_base + 2395 + round_, s, size) for s in range(world)]
            x, y = dev(shards[rank]), poisoned(world * size)
            torch.cuda.synchronize()
            rc = lib.ncclAllGather(p(x), p(y), size, 1, child, h)
            torch.cuda.synchronize()
            if rc or host(y) != b"".join(shards):
                bad.append(f"round {round_} two-way all-gather {size} B shards: result {rc}")
            exchanges += 1
        after = settled_receipt(lib, child)
        report(f"{title}: {calls} one-way calls of both empty directions and roots 0 and 1, each round followed "
               f"by a two-way all-to-all and all-gather, every output exact", not bad, "; ".join(bad[:4]))
        ran = after["pair_exchange"]["ops"] - before["pair_exchange"]["ops"]
        report(f"{title}: under the pair plan every one-way call and all-to-all is one pair exchange",
               not after["pair_plan"] or ran == exchanges, f"{ran} pair exchanges for {exchanges} calls")
        windowed = after["forward_windows"]["ring_window_bytes"]
        chunks = after["forward_windows"]["ring_window_chunks"] - before["forward_windows"]["ring_window_chunks"]
        report(f"{title}: with a ring window the lane carried them through its window",
               not windowed or not after["pair_plan"] or chunks > 0,
               f"ring window {windowed} B, {chunks} window chunks")
    except Exception as error:  # noqa: BLE001
        report(title, False, f"{type(error).__name__}: {error}")
    finally:
        torch.cuda.synchronize()
        report("ncclCommDestroy of the flags-only child", lib.ncclCommDestroy(child) == 0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--id-file", help="a file through which rank 0 hands the unique id to the others "
                        "(ranks on one host)")
    parser.add_argument("--id-server", help="HOST:PORT where rank 0 serves the unique id over TCP (ranks on "
                        "several hosts, for example a cabled pair of Sparks)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--golden", default="")
    parser.add_argument("--golden-ring", default="", help="the ring session's digests, for the cases the cycle "
                        "plan runs as a ring op (default: the -ring sibling of --golden)")
    parser.add_argument("--seed-base", type=int, default=5000)
    args = parser.parse_args(argv)

    import torch

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")  # the primary context is current on this thread
    lib = ctypes.CDLL(args.library)
    lib.ncclGetErrorString.restype = ctypes.c_char_p
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, UniqueId, ctypes.c_int]
    lib.ncclAllGather.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_void_p,
                                  ctypes.c_void_p]
    lib.ncclReduceScatter.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                      ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclBroadcast.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclBcast.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
                              ctypes.c_void_p]
    lib.ncclReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
    for name in ("ncclCommDestroy", "ncclCommFinalize"):
        getattr(lib, name).argtypes = [ctypes.c_void_p]
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    lib.sirclGetReceipt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.POINTER(ctypes.c_size_t)]
    results: list[tuple[str, bool, str]] = []

    def report(name, ok, detail=""):
        results.append((name, bool(ok), detail))

    uid = unique_id.share(lib, args.rank, args.world, args.id_file or "", args.id_server or "")
    comm = ctypes.c_void_p()
    started = time.perf_counter()
    result = lib.ncclCommInitRank(ctypes.byref(comm), args.world, uid, args.rank)
    report("ncclCommInitRank", result == 0,
           f"{time.perf_counter() - started:.2f} s" if result == 0 else lib.ncclGetLastError(None).decode())
    if result != 0:
        Path(args.out).write_text(json.dumps({"rank": args.rank, "checks": results}))
        return 1
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    max_piece = int(os.environ.get("SIRCL_LARGE_PIECE_BYTES", str(4 << 20)))
    golden = Path(args.golden) if args.golden else None
    # A directory of the SIRCL session's output bytes, or a JSON file of their SHA-256 digests
    # ({"caseNNN/rank<r>": hex}, written by ``sircl_golden.py --digests``) that travels to other hosts.
    digests = json.loads(golden.read_text()) if golden is not None and golden.is_file() else None
    cycle = note_cycle_plan(receipt_of(lib, comm), args.world)
    expected_cycle = os.environ.get("LIBRARY_RANK_EXPECT_CYCLE_PLAN")
    if expected_cycle:
        report("the receipt's cycle plan", cycle == (expected_cycle == "1"), f"cycle_plan {cycle}")
    ring_golden = Path(args.golden_ring) if args.golden_ring else (
        golden.with_name(golden.stem + "-ring.json") if golden is not None and golden.is_file() else None)
    ring_digests = (json.loads(ring_golden.read_text())
                    if cycle and ring_golden is not None and ring_golden.is_file() else None)

    def all_reduce(src, dst, stream):
        dtype = NCCL_DTYPES[str(src.dtype).split(".")[-1]]
        return lib.ncclAllReduce(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()), src.numel(),
                                 dtype, 0, comm, ctypes.c_void_p(stream.cuda_stream))

    def check(name, got, want, case_key=None, inputs=None):
        ok = same_bits(torch, got.cpu(), want)
        detail = "" if ok else "differs from the rank-order reference"
        if ok and golden is not None and case_key is not None:
            mine = got.cpu().contiguous().view(torch.uint8).numpy().tobytes()
            table = digests
            if ring_digests is not None and inputs is not None and \
                    large_plan(inputs[0].numel() * inputs[0].element_size(), args.world)[0] == "ring":
                table = ring_digests  # the cycle plan runs this case as a ring op
            if table is not None:
                expected = table.get(f"{case_key}/rank{args.rank}")
                same = expected is not None and hashlib.sha256(mine).hexdigest() == expected
            else:
                path = golden / case_key / f"rank{args.rank}.bin"
                expected = path.read_bytes() if path.exists() else None
                same = expected is not None and mine == expected
            recorded = table.get(f"{case_key}/inputs") if table is not None else None
            if expected is None:
                detail = "no SIRCL session bytes for this case"
            elif not same and recorded and inputs is not None and inputs_digest(torch, inputs) != recorded:
                ok = False
                detail = "the inputs differ from those of the SIRCL session's bytes (another input generator)"
            else:
                ok = same
                detail = "equals the SIRCL session's bytes" if ok else "differs from the SIRCL session's bytes"
        report(name, ok, detail)

    for index, (name, dtype_name, count, mode) in enumerate(cases(max_piece)):
        dtype = getattr(torch, dtype_name)
        seed = args.seed_base + index
        inputs = inputs_for(torch, args.world, count, dtype, seed)
        want = allreduce_reference(torch, inputs)
        key = f"case{index:03d}"
        try:
            with torch.cuda.stream(stream_a):
                if mode == "eager":
                    x = inputs[args.rank].cuda()
                    y = torch.empty_like(x)
                    rc = all_reduce(x, y, stream_a)
                elif mode == "unaligned":
                    # Views 1 and 3 elements into fresh allocations: never 16-byte aligned.
                    raw = torch.empty(count + 8, dtype=dtype, device="cuda")
                    out = torch.empty(count + 8, dtype=dtype, device="cuda")
                    x = raw[1:1 + count]
                    y = out[3:3 + count]
                    x.copy_(inputs[args.rank].cuda())
                    rc = all_reduce(x, y, stream_a)
                elif mode == "inplace":
                    y = inputs[args.rank].cuda()
                    rc = all_reduce(y, y, stream_a)
                else:
                    xs = [inputs_for(torch, args.world, count, dtype, seed * 7 + k) for k in range(4)]
                    wants = [allreduce_reference(torch, values) for values in xs]
                    ins = [values[args.rank].cuda() for values in xs]
                    outs = [torch.empty_like(t) for t in ins]
                    torch.cuda.synchronize()
                    rc = 0
                    for k in range(4):
                        stream = stream_a if k % 2 == 0 else stream_b
                        rc = rc or all_reduce(ins[k], outs[k], stream)
                    torch.cuda.synchronize()
                    if rc:
                        report(name, False, lib.ncclGetLastError(comm).decode())
                    else:
                        good = all(same_bits(torch, outs[k].cpu(), wants[k]) for k in range(4))
                        report(name, good, "" if good else "an alternating-stream result differs")
                    continue
            if rc:
                report(name, False, f"result {rc}: {lib.ncclGetLastError(comm).decode()}")
                continue
            torch.cuda.synchronize()
            check(name, y, want, key if mode in ("eager", "inplace") else None, inputs)
        except Exception as error:  # noqa: BLE001
            report(name, False, f"{type(error).__name__}: {error}")

    def all_gather(src, dst, count, stream):
        dtype = NCCL_DTYPES[str(src.dtype).split(".")[-1]]
        return lib.ncclAllGather(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()), count, dtype,
                                 comm, ctypes.c_void_p(stream.cuda_stream))

    def reduce_scatter(src, dst, count, stream):
        dtype = NCCL_DTYPES[str(src.dtype).split(".")[-1]]
        return lib.ncclReduceScatter(ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()), count, dtype,
                                     0, comm, ctypes.c_void_p(stream.cuda_stream))

    world, rank = args.world, args.rank
    for index, (name, kind, dtype_name, count, mode) in enumerate(gather_scatter_cases(max_piece)):
        dtype = getattr(torch, dtype_name)
        seed = args.seed_base + 500 + index
        key = f"g{index:03d}"
        try:
            with torch.cuda.stream(stream_a):
                if kind == "all_gather":
                    inputs = inputs_for(torch, world, count, dtype, seed)
                    want = torch.cat(inputs)
                    if mode == "eager":
                        x = inputs[rank].cuda()
                        y = torch.empty(world * count, dtype=dtype, device="cuda")
                    elif mode == "unaligned":
                        x = torch.empty(count + 8, dtype=dtype, device="cuda")[1:1 + count]
                        x.copy_(inputs[rank].cuda())
                        y = torch.empty(world * count + 8, dtype=dtype, device="cuda")[3:3 + world * count]
                    else:
                        y = torch.empty(world * count, dtype=dtype, device="cuda")
                        x = y[rank * count:(rank + 1) * count]
                        x.copy_(inputs[rank].cuda())
                    rc = all_gather(x, y, count, stream_a)
                else:
                    inputs = inputs_for(torch, world, world * count, dtype, seed)
                    want = scatter_reference(torch, inputs)[rank]
                    if mode == "eager":
                        x = inputs[rank].cuda()
                        y = torch.empty(count, dtype=dtype, device="cuda")
                    elif mode == "unaligned":
                        x = torch.empty(world * count + 8, dtype=dtype, device="cuda")[1:1 + world * count]
                        x.copy_(inputs[rank].cuda())
                        y = torch.empty(count + 8, dtype=dtype, device="cuda")[3:3 + count]
                    else:
                        x = inputs[rank].cuda()
                        y = x[rank * count:(rank + 1) * count]
                    rc = reduce_scatter(x, y, count, stream_a)
            if rc:
                report(name, False, f"result {rc}: {lib.ncclGetLastError(comm).decode()}")
                continue
            torch.cuda.synchronize()
            check(name, y, want, key if mode in ("eager", "inplace") else None, inputs)
        except Exception as error:  # noqa: BLE001
            report(name, False, f"{type(error).__name__}: {error}")

    # Broadcast (non-roots pass no send buffer), in-place ncclBcast, and reduce to every root.
    # The last case, 2 MiB of whole packs, is one one-way pair exchange on a pair under the pair plan.
    for index, (dtype_name, count) in enumerate((("bfloat16", 1), ("bfloat16", 4096), ("uint8", 1000),
                                                 ("float32", 3 * max_piece // 8 + 5), ("bfloat16", 1 << 20))):
        dtype = getattr(torch, dtype_name)
        root = index % world
        values = inputs_for(torch, world, count, dtype, args.seed_base + 800 + index)
        name = f"broadcast {dtype_name} {count} elements from rank {root}"
        try:
            with torch.cuda.stream(stream_a):
                x = values[rank].cuda()
                y = torch.empty_like(x)
                rc = lib.ncclBroadcast(ctypes.c_void_p(x.data_ptr() if rank == root else 0),
                                       ctypes.c_void_p(y.data_ptr()), count, NCCL_DTYPES[dtype_name], root, comm,
                                       ctypes.c_void_p(stream_a.cuda_stream))
                z = values[rank].cuda()
                rc = rc or lib.ncclBcast(ctypes.c_void_p(z.data_ptr()), count, NCCL_DTYPES[dtype_name], root, comm,
                                         ctypes.c_void_p(stream_a.cuda_stream))
            torch.cuda.synchronize()
            if rc:
                report(name, False, lib.ncclGetLastError(comm).decode())
            else:
                good = same_bits(torch, y.cpu(), values[root]) and same_bits(torch, z.cpu(), values[root])
                report(name, good, "" if good else "differs from the root's bytes")
        except Exception as error:  # noqa: BLE001
            report(name, False, f"{type(error).__name__}: {error}")
    for index, (dtype_name, count) in enumerate((("bfloat16", 3), ("float16", 65536), ("float32", 70001),
                                                 ("float32", (max_piece + 4096) // 4))):
        dtype = getattr(torch, dtype_name)
        root = (index + 1) % world
        values = inputs_for(torch, world, count, dtype, args.seed_base + 900 + index)
        name = f"reduce {dtype_name} {count} elements to rank {root}"
        try:
            with torch.cuda.stream(stream_a):
                x = values[rank].cuda()
                y = torch.full_like(x, 7)
                rc = lib.ncclReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr() if rank == root else 0),
                                    count, NCCL_DTYPES[dtype_name], 0, root, comm,
                                    ctypes.c_void_p(stream_a.cuda_stream))
            torch.cuda.synchronize()
            if rc:
                report(name, False, lib.ncclGetLastError(comm).decode())
            else:
                # The root holds the all-reduce's bits under the same schedule (the chain or ring op's
                # per-hop rounding on larger groups); the other ranks' buffers stay untouched.
                good = same_bits(torch, y.cpu(), allreduce_reference(torch, values) if rank == root else torch.full_like(
                    values[rank], 7))
                report(name, good, "" if good else "the root's sum differs, or a non-root buffer changed")
        except Exception as error:  # noqa: BLE001
            report(name, False, f"{type(error).__name__}: {error}")

    # CUDA graph: all-gather and reduce-scatter (direct and padded) beside the all-reduces, replayed.
    try:
        gather_count, scatter_count = 3000, 4096
        gx = torch.zeros(gather_count, dtype=torch.bfloat16, device="cuda")
        gy = torch.empty(world * gather_count, dtype=torch.bfloat16, device="cuda")
        sx = torch.zeros(world * scatter_count, dtype=torch.float32, device="cuda")
        sy = torch.empty(scatter_count, dtype=torch.float32, device="cuda")
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            codes = [all_gather(gx, gy, gather_count, capture), reduce_scatter(sx, sy, scatter_count, capture)]
        report("graph capture of all-gather and reduce-scatter", not any(codes),
               "" if not any(codes) else lib.ncclGetLastError(comm).decode())
        for replay, seed in enumerate((9201, 9202)):
            gathered = inputs_for(torch, world, gather_count, torch.bfloat16, seed)
            scattered = inputs_for(torch, world, world * scatter_count, torch.float32, seed + 50)
            gx.copy_(gathered[rank].cuda())
            sx.copy_(scattered[rank].cuda())
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            good = same_bits(torch, gy.cpu(), torch.cat(gathered)) and same_bits(
                torch, sy.cpu(), scatter_reference(torch, scattered)[rank])
            report(f"graph replay {replay + 1} of all-gather and reduce-scatter", good,
                   "" if good else "a replayed result differs")
    except Exception as error:  # noqa: BLE001
        report("graph of all-gather and reduce-scatter", False, f"{type(error).__name__}: {error}")

    # CUDA graph: one-shot, two-shot, pieces and a padded tail in one graph, replayed for new inputs.
    graph_counts = [(torch.bfloat16, 2048), (torch.float32, (1 << 20) // 4), (torch.float16, (5 << 20) // 2 + 3)]
    try:
        xs = [torch.zeros(count, dtype=dtype, device="cuda") for dtype, count in graph_counts]
        ys = [torch.empty_like(x) for x in xs]
        graph = torch.cuda.CUDAGraph()
        capture = torch.cuda.Stream()
        torch.cuda.synchronize()
        with torch.cuda.graph(graph, stream=capture):
            codes = [all_reduce(x, y, capture) for x, y in zip(xs, ys)]
        report("graph capture", not any(codes), "" if not any(codes) else lib.ncclGetLastError(comm).decode())
        for replay, seed in enumerate((9101, 9102)):
            wants = []
            for k, (dtype, count) in enumerate(graph_counts):
                values = inputs_for(torch, args.world, count, dtype, seed + k)
                xs[k].copy_(values[args.rank].cuda())
                wants.append(allreduce_reference(torch, values))
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            good = all(same_bits(torch, y.cpu(), w) for y, w in zip(ys, wants))
            report(f"graph replay {replay + 1}", good, "" if good else "a replayed result differs")
    except Exception as error:  # noqa: BLE001
        report("graph", False, f"{type(error).__name__}: {error}")

    captured = len(graph_counts) + 2 + run_fold_and_routing_checks(torch, lib, comm, world, rank, max_piece, stream_a,
                                                                  args.seed_base, report)
    captured += run_point_to_point_checks(torch, lib, comm, world, rank, max_piece, stream_a, args.seed_base, report)
    alternating_alltoalls = run_alternating_grid_checks(torch, lib, comm, world, rank, stream_a, args.seed_base, report)
    run_split_checks(torch, lib, comm, world, rank, stream_a, args.seed_base, report)
    run_abort_with_queued_send_check(torch, lib, comm, world, rank, stream_a, report)
    run_capture_staging_check(torch, lib, comm, world, rank, args.seed_base, report)
    run_rank_local_staging_checks(torch, lib, comm, world, rank, args.seed_base, report)
    run_staging_growth_check(torch, lib, comm, world, rank, args.seed_base, report)
    run_flags_only_checks(torch, lib, comm, world, rank, args.seed_base, report)

    x = torch.ones(16, dtype=torch.int32, device="cuda")
    rc = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(x.data_ptr()), 16, 2, 7, comm,
                           ctypes.c_void_p(stream_a.cuda_stream))
    report("an all-reduce with op 7 (not built in) refused with ncclInvalidArgument", rc == 4,
           lib.ncclGetLastError(comm).decode())
    # A caller that ignores ncclRedOpCreatePreMulSum's refusal, its handle initialized to 0 (ncclSum), as
    # torch's is: the op written makes the all-reduce fail, never a plain sum of unequal inputs.
    lib.ncclRedOpCreatePreMulSum.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p, ctypes.c_int,
                                             ctypes.c_int, ctypes.c_void_p]
    premul, scale = ctypes.c_int(0), ctypes.c_float(0.5)
    created = lib.ncclRedOpCreatePreMulSum(ctypes.byref(premul), ctypes.byref(scale), 7, 1, comm)
    y = torch.full((16,), float(rank + 1), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    rc = lib.ncclAllReduce(ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(y.data_ptr()), 16, 7, premul.value, comm,
                           ctypes.c_void_p(stream_a.cuda_stream))
    torch.cuda.synchronize()
    report("ncclRedOpCreatePreMulSum refused; an all-reduce with the op it wrote refused with ncclInvalidArgument",
           created == 5 and premul.value == 5 and rc == 4 and bool((y == rank + 1).all()),
           f"create {created}, op {premul.value}, all-reduce {rc}")
    status = ctypes.c_int(-1)
    lib.ncclCommGetAsyncError(comm, ctypes.byref(status))
    report("ncclCommGetAsyncError", status.value == 0, f"status {status.value}")
    receipt = receipt_of(lib, comm)
    calls = receipt["all_reduce"]["calls"]
    ops = sum(receipt["all_reduce"]["ops"].values())
    # The fold ran for 57 (datatype, op) pairs: all 60 but the three sums the transport kernels reduce.
    report("receipt", calls > 0 and receipt["refused"]["op"] == 2 and receipt["forwarded"] == 0
           and receipt["all_reduce"]["captured_calls"] == captured and receipt["healthy"]
           and receipt["all_gather"]["calls"] > 0 and receipt["reduce_scatter"]["calls"] > 0
           and receipt["broadcast"]["calls"] == 10 and receipt["reduce"]["calls"] == 6
           and receipt["all_to_all"]["calls"] == 8 + alternating_alltoalls and receipt["gather"]["calls"] == 5
           and receipt["scatter"]["calls"] == 5 and len(receipt["fold"]["ops"]) == 57
           and receipt["point_to_point"]["sends"] + receipt["point_to_point"]["receives"] == (24 if world == 2 else 2)
           # One send to another rank refused on a larger communicator without point-to-point channels.
           and receipt["channels"]["refused"] == (1 if world != 2 and os.environ.get("LIBSIRCL_P2P_CHANNELS") != "on"
                                                  else 0),
           json.dumps({k: receipt[k] for k in ("refused", "channels", "all_reduce", "all_gather", "reduce_scatter",
                                               "all_to_all", "fold")}))
    if any((setting(name, world) or "pieces") != "pieces"
           for name in ("SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE")) or \
            setting("SIRCL_LARGE_SCHEDULE", world) == "ring" or pair_default(world) == "ring":
        links = receipt.get("links", {})
        report("link collectives ran", links.get("on") and links.get("ops") and links.get("native_ops", 0) > 0,
               json.dumps(links))
    report("ncclCommFinalize", lib.ncclCommFinalize(comm) == 0)
    report("ncclCommDestroy", lib.ncclCommDestroy(comm) == 0)
    Path(args.out).write_text(json.dumps({"rank": args.rank, "receipt": receipt, "ops": ops, "checks": results,
                                          "communicators": CREATED["communicators"]}))
    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
