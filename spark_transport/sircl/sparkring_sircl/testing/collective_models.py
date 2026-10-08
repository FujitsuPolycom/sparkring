"""CPU models of the scatter, Swing and fused-norm kernels, and numpy references (test support).

Two kinds of code live here, both torch-free (numpy only):

- **References**: what each collective must return, computed directly from
  every rank's input. :func:`rank_order_sum` is the bit pattern of the one-shot
  and two-shot all-reduce (float32 additions in rank order ``0 .. W-1``, one
  rounding); the reduce-scatter returns its rank's chunk of it
  (:func:`reduce_scatter_reference`); :func:`swing_reference` follows the Swing
  schedule from ``protocol.swing_peer`` and ``protocol.swing_chunk_owners``
  only, rounding to the dtype after every reduce-scatter step. The fused
  all-reduce + RMSNorm reference is ``sparkring_sircl.fused_norm._reference``.
- **Kernel models**: :func:`scatter_op`, :func:`swing_op` and
  :func:`fused_norm_op` play one rank's kernel part of the command ring for
  every rank of a :class:`sparkring_sircl.testing.fabric.LocalSession` (the
  real native layer over the in-memory verbs stand-in): they stage the same
  bytes into the same send-slot offsets, write the same op words, descriptors
  and doorbells, wait for the same flag lines and read the same receive-slot
  offsets as the CuTe kernels ``oneshot/_scatter_cute.py``,
  ``oneshot/_swing_cute.py`` and ``fused_norm/_kernel.py``, with the
  arithmetic of the references. A test that compares a model's outputs with a
  reference therefore checks the kernels' wire behaviour against the native
  layer; the GPU checks (``testing/kernel_gpu_checks.py``) run the kernels
  themselves.

Values are numpy arrays in their storage type: ``float32``, ``float16``, and
``uint16`` bit patterns for ``bfloat16``.
"""

from __future__ import annotations

import ctypes
import time
from collections.abc import Sequence

import numpy as np

from .. import protocol as proto
from .. import scatter_plan
from .. import swing_plan

DTYPES = ("float32", "float16", "bfloat16")
_STORAGE = {"float32": np.float32, "float16": np.float16, "bfloat16": np.uint16}


# -- dtype arithmetic ---------------------------------------------------------------


def storage(dtype: str) -> type:
    """The numpy storage type of ``dtype`` (``uint16`` bit patterns for bfloat16)."""
    return _STORAGE[dtype]


def to_f32(values: np.ndarray, dtype: str) -> np.ndarray:
    """Exact float32 values of a storage array."""
    if dtype == "bfloat16":
        return (np.asarray(values, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
    return np.asarray(values, dtype=_STORAGE[dtype]).astype(np.float32)


def from_f32(values: np.ndarray, dtype: str) -> np.ndarray:
    """Round float32 values to ``dtype`` (round to nearest, ties to even)."""
    values = np.asarray(values, dtype=np.float32)
    if dtype == "float32":
        return values.copy()
    if dtype == "float16":
        return values.astype(np.float16)
    bits = values.view(np.uint32).astype(np.uint64)
    rounded = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16).astype(np.uint16)
    return np.where(np.isnan(values), np.uint16(0x7FC0), rounded).astype(np.uint16)


def random_values(rng: np.random.Generator, dtype: str, count: int, scale: float = 1.0) -> np.ndarray:
    """``count`` normally distributed values of ``dtype``, as a storage array."""
    return from_f32((rng.standard_normal(count) * scale).astype(np.float32), dtype)


def add_rounded(a: np.ndarray, b: np.ndarray, dtype: str) -> np.ndarray:
    """``dtype(float32(a) + float32(b))``: one Swing reduce-scatter step."""
    return from_f32((to_f32(a, dtype) + to_f32(b, dtype)).astype(np.float32), dtype)


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and a.tobytes() == b.tobytes()


# -- references ------------------------------------------------------------------------


def rank_order_sum(inputs: Sequence[np.ndarray], dtype: str) -> np.ndarray:
    """Float32 additions in rank order ``0 .. W-1``, rounded once: the all-reduce bits."""
    total = to_f32(inputs[0], dtype).copy()
    for values in inputs[1:]:
        total = (total + to_f32(values, dtype)).astype(np.float32)
    return from_f32(total, dtype)


def chunk_of(flat: np.ndarray, geometry: scatter_plan.ScatterGeometry, chunk: int) -> np.ndarray:
    """Chunk ``chunk`` of one rank's flat scatter input (storage array)."""
    item = flat.dtype.itemsize
    start = geometry.chunk_offset(chunk) // item
    return flat[start:start + geometry.chunk_bytes // item]


def reduce_scatter_reference(inputs: Sequence[np.ndarray], dtype: str, geometry: scatter_plan.ScatterGeometry,
                             rank: int) -> np.ndarray:
    """Rank ``rank``'s reduce-scatter output: its chunk of the rank-ordered sum."""
    return rank_order_sum([chunk_of(flat, geometry, rank) for flat in inputs], dtype)


def all_to_all_reference(inputs: Sequence[np.ndarray], geometry: scatter_plan.ScatterGeometry,
                         rank: int) -> list[np.ndarray]:
    """Rank ``rank``'s all-to-all output chunks: chunk ``rank`` of every source, in source order."""
    return [chunk_of(flat, geometry, rank).copy() for flat in inputs]


def swing_reference(inputs: Sequence[np.ndarray], dtype: str) -> np.ndarray:
    """The Swing all-reduce result (identical on every rank).

    ``partial[r]`` starts as rank ``r``'s input; at step ``k`` both ranks of
    every pair ``(r, swing_peer(W, r, k))`` replace their partial sums with
    ``dtype(float32(own) + float32(peer's))``. Position ``i`` of the result is
    the partial sum that its owner (``swing_chunk_owners(W)[i]``) holds after
    the last step.
    """
    world = len(inputs)
    steps = world.bit_length() - 1
    partial = [np.asarray(values).copy() for values in inputs]
    for step in range(steps):
        partial = [add_rounded(partial[rank], partial[proto.swing_peer(world, rank, step)], dtype)
                   for rank in range(world)]
    count = partial[0].size
    item = partial[0].dtype.itemsize
    packs = count * item // proto.PACK_BYTES
    per_pack = proto.PACK_BYTES // item
    result = np.empty_like(partial[0])
    for position, owner in enumerate(proto.swing_chunk_owners(world)):
        lo, hi = swing_plan.position_packs(packs, world, position, position + 1)
        result[lo * per_pack:hi * per_pack] = partial[owner][lo * per_pack:hi * per_pack]
    return result


# -- arena access on a LocalSession --------------------------------------------------------


class Arena:
    """Byte access to every rank's arena of a ``LocalSession`` (ctypes, torch-free)."""

    def __init__(self, session) -> None:
        self.session = session
        self.layout = session.arena
        self.slot_bytes = session.slot_bytes
        self.lanes = session.lanes
        self.world = session.world

    def send_address(self, rank: int, seq: int) -> int:
        return self.session.addresses[rank] + self.layout.send_off + proto.slot_of(seq) * self.slot_bytes

    def recv_address(self, rank: int, source: int, seq: int) -> int:
        slot = proto.slot_of(seq)
        return (self.session.addresses[rank] + self.layout.recv_off
                + (source * proto.SLOTS + slot) * self.slot_bytes)

    def ctrl_address(self, rank: int, word: int) -> int:
        return self.session.addresses[rank] + self.layout.ctrl_off + 4 * int(word)

    @staticmethod
    def write(address: int, data: bytes) -> None:
        if data:
            ctypes.memmove(address, data, len(data))

    @staticmethod
    def read(address: int, nbytes: int) -> bytes:
        return ctypes.string_at(address, nbytes)

    def store(self, rank: int, word: int, value: int) -> None:
        ctypes.c_uint32.from_address(self.ctrl_address(rank, word)).value = value & 0xFFFFFFFF

    def release(self, rank: int, word: int, value: int) -> None:
        """A store after a full fence, as the kernels' system fence before a doorbell."""
        self.session.library.roce_store_release_u32(self.ctrl_address(rank, word), value & 0xFFFFFFFF)

    def wait(self, rank: int, source: int, seq: int, namespace: int, timeout: float = 20.0) -> None:
        """Wait for every lane flag of ``source`` in ``namespace`` at ``seq`` (the kernels' flag waits)."""
        deadline = time.monotonic() + timeout
        for lane in range(self.lanes):
            line = proto.flag_index(namespace, source, proto.slot_of(seq), lane, self.world, self.lanes)
            address = self.session.addresses[rank] + self.layout.flag_off + line * proto.FLAG_STRIDE
            while self.session.library.roce_load_acquire_u32(address) != seq & 0xFFFFFFFF:
                failed = [r for r, proxy in enumerate(self.session.proxies) if proxy.failed()]
                if failed:
                    raise RuntimeError(f"progress thread of rank {failed[0]} failed: "
                                       f"{self.session.proxies[failed[0]].error()}")
                if time.monotonic() > deadline:
                    raise TimeoutError(f"rank {rank} waited for rank {source} lane {lane} namespace "
                                       f"{namespace} at sequence {seq}")
                time.sleep(0.0001)

    def doorbell(self, rank: int, seq: int, op: int, nbytes: int, descriptor: int | None = None) -> None:
        """Byte count, the slot's op word (and the phase-0 descriptor), then the doorbell."""
        self.store(rank, proto.Ctrl.NBYTES, nbytes)
        self.store(rank, proto.Ctrl.OP_WORD + proto.slot_of(seq), proto.op_word(op, nbytes))
        if descriptor is not None:
            self.store(rank, proto.descriptor_word(0), descriptor)
        self.release(rank, proto.Ctrl.DOORBELL, seq)


# -- scatter kernel model --------------------------------------------------------------------


def scatter_op(session, seq: int, mode: str, inputs: Sequence[np.ndarray], dtype: str,
               geometry: scatter_plan.ScatterGeometry, *, offset: int = 0,
               dst_stride: int | None = None) -> list[np.ndarray]:
    """One scatter op of every rank; returns each rank's output chunks for this op.

    Op ``seq`` carries bytes ``[offset, offset + chunk_bytes)`` of every chunk
    of the flat inputs. Each rank stages chunk ``j != rank`` at bytes ``[j *
    c, (j + 1) * c)`` of ``send[seq & 1]`` (``c`` the op's chunk bytes), writes
    op code 3 and rings the doorbell; then waits for every peer's namespace-0
    flags and reads its own chunk ``[rank * c, (rank + 1) * c)`` of every
    peer's ``recv[peer][seq & 1]``, as the scatter kernel does. ``reduce``
    returns one array (the rank-ordered sum, rounded once); ``copy`` returns
    the ``W`` received chunks in source order (rank ``rank``'s from its input).
    """
    if mode not in scatter_plan.MODES:
        raise ValueError(f"unknown scatter mode {mode!r}")
    arena = Arena(session)
    world = session.world
    chunk = geometry.chunk_bytes
    nbytes = world * chunk
    if nbytes > session.slot_bytes:
        raise ValueError(f"a scatter op of {nbytes} bytes does not fit a slot of {session.slot_bytes}")
    raw = [np.ascontiguousarray(flat).view(np.uint8) for flat in inputs]

    def source_chunk(rank: int, j: int) -> bytes:
        start = geometry.chunk_offset(j) + offset
        return raw[rank][start:start + chunk].tobytes()

    for rank in range(world):
        send = arena.send_address(rank, seq)
        for j in range(world):
            if j != rank:
                arena.write(send + j * chunk, source_chunk(rank, j))
        arena.doorbell(rank, seq, proto.Op.SCATTER, nbytes)
    outputs = []
    for rank in range(world):
        received = []
        for source in range(world):
            if source == rank:
                received.append(source_chunk(rank, rank))
                continue
            arena.wait(rank, source, seq, namespace=0)
            received.append(arena.read(arena.recv_address(rank, source, seq) + rank * chunk, chunk))
        parts = [np.frombuffer(data, dtype=storage(dtype)) for data in received]
        if mode == "reduce":
            outputs.append(rank_order_sum(parts, dtype))
        else:
            outputs.append([part.copy() for part in parts])
    return outputs


def scatter_message(session, first_seq: int, mode: str, inputs: Sequence[np.ndarray], dtype: str,
                    geometry: scatter_plan.ScatterGeometry, piece: int | None = None) -> tuple[list, int]:
    """A whole scatter message in the ops of ``scatter_plan.scatter_plan``; returns (outputs, next sequence).

    Reduce outputs are flat arrays of one chunk; copy outputs are lists of the
    ``W`` received chunks, each reassembled from its pieces.
    """
    world = session.world
    item = np.dtype(storage(dtype)).itemsize
    plan = scatter_plan.scatter_plan(geometry.chunk_bytes, piece)
    seq = first_seq
    if mode == "reduce":
        results = [np.empty(geometry.chunk_bytes // item, dtype=storage(dtype)) for _ in range(world)]
    else:
        results = [[np.empty(geometry.chunk_bytes // item, dtype=storage(dtype)) for _ in range(world)]
                   for _ in range(world)]
    for part in plan:
        sub = scatter_plan.ScatterGeometry(part.nbytes, geometry.src_stride_bytes)
        outputs = scatter_op(session, seq, mode, inputs, dtype, sub, offset=part.offset)
        lo, hi = part.offset // item, (part.offset + part.nbytes) // item
        for rank in range(world):
            if mode == "reduce":
                results[rank][lo:hi] = outputs[rank]
            else:
                for source in range(world):
                    results[rank][source][lo:hi] = outputs[rank][source]
        seq += 1
    return results, seq


# -- Swing kernel model -----------------------------------------------------------------------


def swing_op(session, seq: int, inputs: Sequence[np.ndarray], dtype: str) -> list[np.ndarray]:
    """One Swing all-reduce op of every rank (op code 2); returns every rank's output.

    Per rank, as the Swing kernel: stage the positions of phase 0 from the
    input into ``send[seq & 1]``, write op code 2, the phase-0 descriptor and
    the doorbell. Network phase ``k`` of the reduce-scatter waits for the
    step's peer in namespace 0, adds that peer's partial sums of the kept
    positions (from ``recv[peer][seq & 1]``) to the own ones (the input at
    step 0, the send slot after it), rounds to the dtype and stores the result
    in the send slot (and in the output at the last step). An all-gather phase
    waits for the peer in namespace 1 and copies the peer's range into the
    output, and into the send slot while a later phase still sends it. After
    every phase but the last, the rank writes the next phase's descriptor and
    rings its doorbell. Ranks advance one phase at a time.
    """
    arena = Arena(session)
    world = session.world
    if not swing_plan.available(world):
        raise ValueError(f"the Swing all-reduce needs a power-of-two group, got {world} ranks")
    flats = [np.ascontiguousarray(values) for values in inputs]
    nbytes = flats[0].size * flats[0].dtype.itemsize
    if nbytes % proto.PACK_BYTES or nbytes > session.slot_bytes:
        raise ValueError(f"a Swing op of {nbytes} bytes does not fit")
    packs = nbytes // proto.PACK_BYTES
    phases = swing_plan.phase_count(world)
    steps = phases // 2
    plans = [swing_plan.swing_steps(world, rank) for rank in range(world)]
    descriptors = [swing_plan.descriptor_words(world, rank) for rank in range(world)]
    outputs = [np.zeros_like(flats[rank]) for rank in range(world)]

    def byte_range(first: int, end: int) -> tuple[int, int]:
        lo, hi = swing_plan.position_packs(packs, world, first, end)
        return lo * proto.PACK_BYTES, hi * proto.PACK_BYTES

    for rank in range(world):
        _, first, end = swing_plan.phase_range(world, rank, 0)
        lo, hi = byte_range(first, end)
        arena.write(arena.send_address(rank, seq) + lo, flats[rank].view(np.uint8)[lo:hi].tobytes())
        arena.doorbell(rank, seq, proto.Op.DESCRIBED, nbytes, descriptors[rank][0])
    for phase in range(phases):
        for rank in range(world):
            gather = phase >= steps
            step = plans[rank][phase if not gather else phases - 1 - phase]
            arena.wait(rank, step.peer, seq, namespace=1 if gather else 0)
            send = arena.send_address(rank, seq)
            theirs = arena.recv_address(rank, step.peer, seq)
            out_bytes = outputs[rank].view(np.uint8)
            if not gather:
                lo, hi = byte_range(*step.keep)
                mine = (flats[rank].view(np.uint8)[lo:hi].tobytes() if step.step == 0
                        else arena.read(send + lo, hi - lo))
                summed = add_rounded(np.frombuffer(mine, dtype=storage(dtype)),
                                     np.frombuffer(arena.read(theirs + lo, hi - lo), dtype=storage(dtype)), dtype)
                arena.write(send + lo, summed.tobytes())
                if step.step == steps - 1:
                    out_bytes[lo:hi] = np.frombuffer(summed.tobytes(), dtype=np.uint8)
            else:
                lo, hi = byte_range(*step.send)
                data = arena.read(theirs + lo, hi - lo)
                out_bytes[lo:hi] = np.frombuffer(data, dtype=np.uint8)
                if phase < phases - 1:
                    arena.write(send + lo, data)
            if phase < phases - 1:
                arena.store(rank, proto.descriptor_word(phase + 1), descriptors[rank][phase + 1])
                arena.release(rank, proto.phase_doorbell_word(phase + 1), seq)
    return outputs


# -- fused all-reduce + residual add + RMSNorm kernel model ------------------------------------


def fused_norm_op(session, seq: int, algorithm: str, partials: Sequence[np.ndarray], residuals: Sequence[np.ndarray],
                  weight: np.ndarray, eps: float, *, plain: Sequence[int] = ()) -> list:
    """One fused all-reduce + residual add + RMSNorm op of every rank (BF16 bit arrays ``[rows, hidden]``).

    Per rank, as the fused kernels of ``sparkring_sircl/fused_norm/_kernel.py``:
    one-shot stages the whole message and rings op code 0; each row (one CTA
    of the kernel) waits only for the peer stripes that ``_geometry.oneshot_wait``
    names and reduces its packs in rank order. Two-shot stages every pack
    outside the own chunk and rings op code 1; rows that hold own-chunk packs
    wait for the scatter stripes of ``twoshot_scatter_wait``, the own chunk is
    reduced in rank order into the send slot and the gather doorbell (phase
    1) rings; every row then waits for the gather stripes of
    ``twoshot_gather_wait`` and reads the reduced packs. The norm follows
    ``fused_norm._reference`` (correctly rounded reciprocal square root).

    Ranks listed in ``plain`` play the plain all-reduce kernels of the same
    algorithm: same staging, words and waits, and their result is the reduced
    message. Returns, per rank, ``(normed, new_residual)`` or the reduced
    message for plain ranks.
    """
    from ..fused_norm import _geometry as geo
    from ..fused_norm import _reference as ref

    arena = Arena(session)
    world = session.world
    lanes = session.lanes
    rows, hidden = partials[0].shape
    row_packs = hidden * 2 // proto.PACK_BYTES
    packs = rows * row_packs
    nbytes = packs * proto.PACK_BYTES
    geo.check_launch_geometry(rows, row_packs, world, lanes)
    raw = [np.ascontiguousarray(part).view(np.uint8).reshape(-1) for part in partials]
    pack_bytes = proto.PACK_BYTES

    def wait(rank: int, source: int, lane: int, namespace: int) -> None:
        deadline = time.monotonic() + 20.0
        line = proto.flag_index(namespace, source, proto.slot_of(seq), lane, world, lanes)
        address = session.addresses[rank] + arena.layout.flag_off + line * proto.FLAG_STRIDE
        while session.library.roce_load_acquire_u32(address) != seq & 0xFFFFFFFF:
            if time.monotonic() > deadline:
                raise TimeoutError(f"rank {rank} waited for rank {source} lane {lane} at sequence {seq}")
            time.sleep(0.0001)

    reduced: list[np.ndarray] = [np.empty(packs * 8, dtype=np.uint16) for _ in range(world)]
    if algorithm == geo.ONESHOT:
        for rank in range(world):
            arena.write(arena.send_address(rank, seq), raw[rank].tobytes())
            arena.doorbell(rank, seq, proto.Op.ONESHOT, nbytes)
        for rank in range(world):
            for row in range(rows):
                for peer in range(world):
                    for lane in range(lanes):
                        if geo.oneshot_wait(row, row_packs, packs, peer, lane, rank, lanes):
                            wait(rank, peer, lane, geo.SCATTER)
            sources = [raw[rank] if source == rank else
                       np.frombuffer(arena.read(arena.recv_address(rank, source, seq), nbytes), dtype=np.uint8)
                       for source in range(world)]
            reduced[rank] = ref.rank_order_sum([source.view(np.uint16) for source in sources])
    elif algorithm == geo.TWOSHOT:
        for rank in range(world):
            lo, hi = geo.chunk_bounds(packs, world, rank)
            send = arena.send_address(rank, seq)
            arena.write(send, raw[rank][:lo * pack_bytes].tobytes())
            arena.write(send + hi * pack_bytes, raw[rank][hi * pack_bytes:].tobytes())
            arena.doorbell(rank, seq, proto.Op.TWOSHOT, nbytes)
        for rank in range(world):
            lo, hi = geo.chunk_bounds(packs, world, rank)
            for row in range(rows):
                for peer in range(world):
                    for lane in range(lanes):
                        if geo.twoshot_scatter_wait(row, row_packs, packs, peer, lane, rank, world, lanes):
                            wait(rank, peer, lane, geo.SCATTER)
            parts = [raw[rank][lo * pack_bytes:hi * pack_bytes] if source == rank else
                     np.frombuffer(arena.read(arena.recv_address(rank, source, seq) + lo * pack_bytes,
                                              (hi - lo) * pack_bytes), dtype=np.uint8)
                     for source in range(world)]
            own = ref.rank_order_sum([part.view(np.uint16) for part in parts])
            arena.write(arena.send_address(rank, seq) + lo * pack_bytes, own.tobytes())
            arena.release(rank, proto.phase_doorbell_word(1), seq)
        for rank in range(world):
            for row in range(rows):
                for owner in range(world):
                    for lane in range(lanes):
                        if geo.twoshot_gather_wait(row, row_packs, packs, owner, lane, rank, world, lanes):
                            wait(rank, owner, lane, geo.GATHER)
            pieces = []
            for owner in range(world):
                lo, hi = geo.chunk_bounds(packs, world, owner)
                base = arena.send_address(rank, seq) if owner == rank else arena.recv_address(rank, owner, seq)
                pieces.append(arena.read(base + lo * pack_bytes, (hi - lo) * pack_bytes))
            reduced[rank] = np.frombuffer(b"".join(pieces), dtype=np.uint16).copy()
    else:
        raise ValueError(f"unknown fused-norm algorithm {algorithm!r}")
    results: list = []
    for rank in range(world):
        message = reduced[rank].reshape(rows, hidden)
        if rank in plain:
            results.append(message)
        else:
            results.append(ref.unfused_reference(message, residuals[rank], weight, eps))
    return results


__all__ = ["DTYPES", "Arena", "add_rounded", "all_to_all_reference", "chunk_of", "from_f32", "fused_norm_op",
           "random_values", "rank_order_sum", "reduce_scatter_reference", "same_bits", "scatter_message", "scatter_op",
           "storage", "swing_op", "swing_reference", "to_f32"]
