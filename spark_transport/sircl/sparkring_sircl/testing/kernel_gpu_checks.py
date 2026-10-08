"""GPU checks of the reduce-scatter, all-to-all and Swing kernels on an emulated group (test support).

Every rank of one group runs as a thread of one process on one CUDA device
over the simulator build of the native layer
(:class:`sparkring_sircl.testing.gpu_emulation.EmulatedGroup`). The checks run
the CuTe kernels ``oneshot/_scatter_cute.py`` and ``oneshot/_swing_cute.py``
on the group's sessions, through the session methods where the session class
offers the operation and otherwise through the host functions
(``oneshot/_scatter_ops.py``, ``oneshot/_swing_ops.py``), and compare every
rank's result bit for bit with the references of
:mod:`sparkring_sircl.testing.collective_models`:

- reduce-scatter of contiguous and strided chunks, every dtype: chunk ``rank``
  of the rank-ordered float32 sum, rounded once;
- all-to-all of contiguous and strided chunks with a destination stride:
  bytes unchanged;
- messages split into several ops by a small relay-safe per-peer size: the
  same bits as one op;
- large reduce-scatters of ``[rows, 4096]`` BF16 (``--large-reduce-scatter``,
  in MiB) far above the slot, carried in ops of the session's large-message
  piece: this rank's rows of the rank-ordered sum, bit for bit equal to the
  host reference and to the same rows of the session's all-reduce in ops of at
  most the capacity, eager and replayed from a CUDA graph, with the number of
  posted ops equal to the plan's. The difference from ``all_reduce_large`` is
  reported with its schedule: a chain op rounds once per hop and can differ in
  the last place;
- Swing all-reduce of power-of-two groups from one pack to the capacity,
  every dtype;
- the same collectives captured in a CUDA graph and replayed with new inputs;
- interleaved Swing, one-shot all-reduce and reduce-scatter ops on one session;
- fail-stop: with a short serving wait limit, ranks whose peer never joins a
  reduce-scatter time out, poison their sessions and name the missing rank.

A session class that does not offer the scatter collectives or Swing reports
them unavailable; :func:`enable` marks each such operation available on an
emulated session (multi-phase ops, and a power-of-two group for Swing), and
the checks of that operation then call its host functions.

Every input reaches the device before the ranks start. In one process, a rank
thread that copies from pageable host memory while other ranks' kernels wait
for its data can stall the verbs stand-in's delivery thread
(``testing/dcp_gpu_checks.py`` describes the coupling); on the ring every rank
is its own process.

Every rank's grid must be resident at once on the shared GPU: a rank whose
blocks wait for a free multiprocessor never stages its data, and the resident
ranks wait for it until their wait limit. The checks cap the large-message grid
(``SIRCL_LARGE_BLOCKS``, unless set) at :func:`emulation_large_blocks`: on an
RTX 5090 (170 multiprocessors) 16 blocks per rank for eight ranks, where the
default 32 leaves the BF16 reduce-scatter of 128 KiB per rank and more waiting
for multiprocessors.

``python -m sparkring_sircl.testing.kernel_gpu_checks [--layout ring:8] [--lanes 2] [--dtypes bfloat16,float32]``
prints one line per check. Requirements: CUDA, torch with CUDA, CUDA Python,
the CuTe DSL, and a GCC-compatible compiler for the simulator library; on a
discrete GPU set ``CUTE_DSL_ARCH`` (for example ``sm_120a``) when the DSL cannot
detect the architecture.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .. import protocol as proto
from .. import scatter_plan
from . import collective_models as models
from . import native_build

Check = tuple[str, bool, str]


def scatter_registered(session: Any) -> bool:
    """Whether the session class itself offers the scatter collectives wherever its geometry allows them."""
    return session.scatter_available == bool(session.multi_phase)


def swing_registered(session: Any) -> bool:
    """Whether the session class itself offers Swing wherever its group allows it."""
    from ..oneshot import _swing_ops

    return session._available.get("swing", False) == _swing_ops.available(session)


def registered(session: Any) -> bool:
    """Whether the session class itself offers both the scatter collectives and Swing."""
    return scatter_registered(session) and swing_registered(session)


def emulation_large_blocks(world: int, multiprocessors: int | None = None, default: int | None = None) -> int:
    """Large-message grid cap per rank that keeps every emulated rank's grid resident at once.

    The ranks of an emulated group share one GPU, and a collective's blocks
    spin until every rank has staged its data, so all ``world`` grids must be
    resident together. One block per multiprocessor fits any launch of the
    session's block size whatever its register use; the cap is the largest
    power of two of at most ``multiprocessors // world`` blocks per rank, and
    at most the session default (``runtime.DEFAULT_LARGE_BLOCKS`` unless
    ``default`` is given). On the ring every rank has its own GPU and keeps the
    default.
    """
    if default is None:
        from ..oneshot import runtime

        default = runtime.DEFAULT_LARGE_BLOCKS
    if multiprocessors is None:
        import torch

        multiprocessors = torch.cuda.get_device_properties(0).multi_processor_count
    per_rank = max(1, int(multiprocessors) // max(1, int(world)))
    return min(int(default), 1 << (per_rank.bit_length() - 1))


def enable(session: Any) -> tuple[bool, bool]:
    """Mark the scatter collectives and (on power-of-two groups) Swing available on ``session``.

    Returns ``(scatter, swing)``: True for an operation the session already
    offered (its own methods then run that operation's checks), False for one
    marked here (its checks then call the host functions of ``_scatter_ops``
    or ``_swing_ops`` directly).
    """
    from ..oneshot import _swing_ops

    scatter, swing = scatter_registered(session), swing_registered(session)
    if not scatter:
        session.scatter_available = bool(session.multi_phase)
    if not swing:
        session._available["swing"] = _swing_ops.available(session)
    return scatter, swing


def _torch_dtype(torch, name: str):
    return getattr(torch, name)


def _to_numpy(torch, tensor, name: str) -> np.ndarray:
    tensor = tensor.detach().cpu().contiguous().reshape(-1)
    if name == "bfloat16":
        return tensor.view(torch.int16).numpy().view(np.uint16).copy()
    return tensor.numpy().copy()


def _from_numpy(torch, array: np.ndarray, name: str):
    if name == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(array).view(np.int16)).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(array))


def _inputs(torch, world: int, shape: Sequence[int], name: str, seed: int) -> list:
    tensors = []
    for rank in range(world):
        generator = torch.Generator().manual_seed(seed * 7919 + rank)
        tensors.append(torch.randn(tuple(shape), generator=generator).to(_torch_dtype(torch, name)))
    return tensors


def _bits(torch, tensor) -> bytes:
    return tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()


class KernelChecks:
    """The checks of one emulated group; every method returns ``(name, ok, detail)``."""

    def __init__(self, group: Any, dtypes: Sequence[str], scatter_session: bool = False,
                 swing_session: bool = False) -> None:
        import torch

        self.torch = torch
        self.scatter_session = scatter_session
        self.swing_session = swing_session
        self.group = group
        self.world = group.world
        self.dtypes = tuple(dtypes)
        self.swing_available = all(session._available.get("swing", False) for session in group.sessions)
        self.seed = 1000

    # -- infrastructure -------------------------------------------------------------

    def _next_seed(self) -> int:
        self.seed += 1
        return self.seed

    def _reduce_scatter(self, session: Any, x: Any, **keywords: Any):
        from ..oneshot import _scatter_ops

        if self.scatter_session:
            return session.reduce_scatter(x, **keywords)
        return _scatter_ops.reduce_scatter(session, x, **keywords)

    def _all_to_all(self, session: Any, x: Any, out: Any, **keywords: Any):
        from ..oneshot import _scatter_ops

        if self.scatter_session:
            return session.all_to_all(x, out, **keywords)
        return _scatter_ops.all_to_all(session, x, out, **keywords)

    def _swing(self, session: Any, x: Any):
        from ..oneshot import _swing_ops

        if self.swing_session:
            return session.all_reduce(x, algorithm="swing")
        return _swing_ops.all_reduce(session, x)

    def _staged(self, tensors: Sequence[Any]) -> list:
        """Device copies of the host ``tensors``, one per rank, made from this thread before any rank runs."""
        from .gpu_emulation import wait_stream

        staged = []
        for rank, (session, tensor) in enumerate(zip(self.group.sessions, tensors)):
            stream = self.group.streams[rank]
            with self.torch.cuda.stream(stream):
                staged.append(tensor.to(session.device))
                wait_stream(stream)
        return staged

    def run(self, name: str, inputs: Sequence[Any], call: Callable[[int, Any, Any], Any],
            expected: Sequence[bytes]) -> Check:
        """``call(rank, session, x)`` on every rank with ``x`` on the device; compare each result's bytes."""
        torch = self.torch

        try:
            staged = self._staged(inputs)
            results = self.group.each(lambda rank, session: call(rank, session, staged[rank]))
        except Exception as error:  # noqa: BLE001 - reported as a failed check
            return name, False, f"{type(error).__name__}: {error}"
        wrong = [rank for rank, result in enumerate(results) if _bits(torch, result) != expected[rank]]
        return name, not wrong, f"ranks {wrong} differ" if wrong else ""

    def _reduce_reference(self, inputs: Sequence[Any], name: str, chunk: int, stride: int) -> list[bytes]:
        flats = [_to_numpy(self.torch, x, name) for x in inputs]
        geometry = scatter_plan.ScatterGeometry(chunk, stride)
        return [models.reduce_scatter_reference(flats, name, geometry, rank).tobytes() for rank in range(self.world)]

    def _copy_reference(self, inputs: Sequence[Any], chunk: int, stride: int, dst_stride: int,
                        out_bytes: int) -> list[bytes]:
        torch = self.torch
        flats = [x.contiguous().reshape(-1).view(torch.uint8).numpy() for x in inputs]
        geometry = scatter_plan.ScatterGeometry(chunk, stride)
        expected = []
        for rank in range(self.world):
            out = np.zeros(out_bytes, dtype=np.uint8)
            for source, part in enumerate(models.all_to_all_reference(flats, geometry, rank)):
                out[source * dst_stride:source * dst_stride + chunk] = part
            expected.append(out.tobytes())
        return expected

    # -- checks ---------------------------------------------------------------------------

    def reduce_scatter_contiguous(self, name: str, chunk: int) -> Check:

        item = self.torch.empty((), dtype=_torch_dtype(self.torch, name)).element_size()
        inputs = _inputs(self.torch, self.world, (self.world, chunk // item), name, self._next_seed())
        expected = self._reduce_reference(inputs, name, chunk, chunk)
        return self.run(f"reduce_scatter {name} {chunk} B per rank", inputs,
                        lambda rank, session, x: self._reduce_scatter(session, x), expected)

    def reduce_scatter_strided(self, name: str, rows: int, used: int, row_elems: int) -> Check:
        """Rows ``[0, used)`` of every source's ``rows`` rows (chunk ``used`` rows, stride ``rows`` rows)."""

        item = self.torch.empty((), dtype=_torch_dtype(self.torch, name)).element_size()
        chunk, stride = used * row_elems * item, rows * row_elems * item
        inputs = _inputs(self.torch, self.world, (self.world, rows, row_elems), name, self._next_seed())
        expected = self._reduce_reference(inputs, name, chunk, stride)
        return self.run(
            f"reduce_scatter {name} strided {used}/{rows} rows of {row_elems}", inputs,
            lambda rank, session, x: self._reduce_scatter(session, x, chunk_bytes=chunk, src_stride_bytes=stride),
            expected)

    def all_to_all(self, name: str, rows: int, used: int, row_elems: int, out_rows: int) -> Check:

        torch = self.torch
        dtype = _torch_dtype(torch, name)
        item = torch.empty((), dtype=dtype).element_size()
        chunk, stride, dst = used * row_elems * item, rows * row_elems * item, out_rows * row_elems * item
        inputs = _inputs(torch, self.world, (self.world, rows, row_elems), name, self._next_seed())
        out_bytes = self.world * dst
        expected = self._copy_reference(inputs, chunk, stride, dst, out_bytes)

        def call(rank, session, x):
            out = torch.zeros((self.world, out_rows, row_elems), dtype=dtype, device=session.device)
            return self._all_to_all(session, x, out, chunk_bytes=chunk, src_stride_bytes=stride,
                                    dst_stride_bytes=dst)

        return self.run(f"all_to_all {name} {used}/{rows} rows into {out_rows}", inputs, call, expected)

    def split_ops(self, name: str, piece: int) -> list[Check]:
        """Messages split into ops by a relay-safe per-peer size of ``piece`` bytes give the same bits."""
        sessions = self.group.sessions
        saved = [session.relay_safe_bytes for session in sessions]
        item = self.torch.empty((), dtype=_torch_dtype(self.torch, name)).element_size()
        cases = ((lambda: self.reduce_scatter_contiguous(name, 4 * piece + 16), 4 * piece + 16),
                 (lambda: self.reduce_scatter_strided(name, 9, 7, 64), 7 * 64 * item),
                 (lambda: self.all_to_all(name, 9, 7, 64, 8), 7 * 64 * item))
        for session in sessions:
            session.relay_safe_bytes = piece
        checks = []
        try:
            for case, chunk in cases:
                before = sessions[0].stats().get("ops_posted")
                label, ok, detail = case()
                after = sessions[0].stats().get("ops_posted")
                expected = len(scatter_plan.scatter_plan(chunk, scatter_plan.piece_bytes(chunk, piece)))
                if before is not None and after is not None and after - before != expected:
                    ok, detail = False, f"{after - before} ops posted, the plan has {expected}; {detail}"
                checks.append((f"split into {piece}-byte pieces ({expected} ops): {label}", ok, detail))
        finally:
            for session, value in zip(sessions, saved):
                session.relay_safe_bytes = value
        return checks

    def swing(self, name: str, nbytes: int) -> Check:
        item = self.torch.empty((), dtype=_torch_dtype(self.torch, name)).element_size()
        inputs = _inputs(self.torch, self.world, (nbytes // item,), name, self._next_seed())
        reference = models.swing_reference([_to_numpy(self.torch, x, name) for x in inputs], name)
        expected = [reference.tobytes()] * self.world
        return self.run(f"swing all_reduce {name} {nbytes} B", inputs,
                        lambda rank, session, x: self._swing(session, x), expected)

    def swing_graph(self, name: str, nbytes: int, seeds: Sequence[int] = (81, 82)) -> Check:
        """A Swing all-reduce of ``nbytes`` captured per rank and replayed with new inputs."""
        from .gpu_emulation import wait_stream

        torch = self.torch
        group = self.group
        dtype = _torch_dtype(torch, name)
        count = nbytes // torch.empty((), dtype=dtype).element_size()
        label = f"graph replay swing all_reduce {name} {nbytes} B"
        state: list[dict] = [dict() for _ in range(self.world)]
        try:
            for rank, session in enumerate(group.sessions):
                with torch.cuda.stream(group.streams[rank]):
                    s = torch.zeros(count, dtype=dtype, device=session.device)
                    graph = torch.cuda.CUDAGraph()
                    with session.capture():
                        with torch.cuda.graph(graph, stream=group.streams[rank]):
                            swung = self._swing(session, s)
                    wait_stream(group.streams[rank])
                state[rank].update(s=s, swung=swung, graph=graph)
            for seed in seeds:
                ss = _inputs(torch, self.world, (count,), name, seed)
                staged = self._staged(ss)

                def replay(rank: int, session: Any):
                    state[rank]["s"].copy_(staged[rank])
                    state[rank]["graph"].replay()

                group.each(replay)
                expected = models.swing_reference([_to_numpy(torch, s, name) for s in ss], name).tobytes()
                wrong = [rank for rank in range(self.world) if _bits(torch, state[rank]["swung"]) != expected]
                if wrong:
                    return label, False, f"seed {seed}: ranks {wrong} differ"
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"
        return label, True, f"{len(seeds)} replays"

    def large_reduce_scatter(self, mib: int, mode: str = "eager") -> Check:
        """A ``[rows, 4096]`` BF16 reduce-scatter of ``mib`` MiB per rank against the reference and the all-reduce."""
        from ..oneshot import _scatter_ops
        from .gpu_emulation import wait_stream

        torch = self.torch
        group = self.group
        world = self.world
        rows = (mib << 20) // (4096 * 2)
        label = f"large reduce_scatter bfloat16 [{rows}, 4096] ({mib} MiB) {mode}"
        if rows % world:
            return label, False, f"{rows} rows do not split over {world} ranks"
        share = rows // world
        inputs = _inputs(torch, world, (rows, 4096), "bfloat16", self._next_seed())
        total = inputs[0].float().clone()
        for tensor in inputs[1:]:
            total += tensor.float()
        reference = total.to(torch.bfloat16)
        session0 = group.sessions[0]
        chunk = share * 4096 * 2
        piece = _scatter_ops.piece_bytes(session0, chunk)
        expected_ops = len(scatter_plan.scatter_plan(chunk, piece))
        state: list[dict] = [dict() for _ in range(world)]
        try:
            for rank, session in enumerate(group.sessions):
                with torch.cuda.stream(group.streams[rank]):
                    x = inputs[rank].to(session.device)
                    out = torch.empty((share, 4096), dtype=torch.bfloat16, device=session.device)
                    graph = None
                    if mode == "graph":
                        graph = torch.cuda.CUDAGraph()
                        with session.capture():
                            with torch.cuda.graph(graph, stream=group.streams[rank]):
                                self._reduce_scatter(session, x, out=out)
                        wait_stream(group.streams[rank])
                    state[rank].update(x=x, out=out, graph=graph)
            before = session0.stats().get("ops_posted")

            def operation(rank: int, session: Any):
                if state[rank]["graph"] is not None:
                    state[rank]["graph"].replay()
                else:
                    self._reduce_scatter(session, state[rank]["x"], out=state[rank]["out"])
                return state[rank]["out"]

            outputs = group.each(operation)
            after = session0.stats().get("ops_posted")

            def plain(rank: int, session: Any):
                flat = state[rank]["x"].reshape(-1)
                result = torch.empty_like(flat)
                step = session.max_size // flat.element_size()
                for first in range(0, flat.numel(), step):
                    session.all_reduce(flat[first:first + step], out=result[first:first + step])
                return result.view(rows, 4096)

            summed = group.each(plain)
            large = group.each(lambda rank, session: session.all_reduce_large(state[rank]["x"]))
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"

        def rows_of(tensor, rank):
            return tensor[rank * share:(rank + 1) * share]

        wrong_reference = [rank for rank in range(world)
                           if _bits(torch, outputs[rank]) != _bits(torch, rows_of(reference, rank))]
        wrong_allreduce = [rank for rank in range(world)
                           if _bits(torch, outputs[rank]) != _bits(torch, rows_of(summed[rank], rank))]
        large_differs = [int((rows_of(large[rank], rank).cpu().view(torch.int16)
                              != outputs[rank].cpu().view(torch.int16)).sum().item()) for rank in range(world)]
        planner = getattr(session0, "large_reduce_plan", None)
        schedule = ("chain" if callable(planner) and any(getattr(part, "chain", False)
                                                         for part in planner(rows * 4096 * 2, aligned=True))
                    else "pieces")
        posted = None if before is None or after is None else after - before
        ok = not wrong_reference and not wrong_allreduce and posted in (None, expected_ops)
        detail = (f"{expected_ops} ops of {piece} bytes per peer, {posted} posted; differs from the reference on "
                  f"ranks {wrong_reference}, from the all-reduce rows on ranks {wrong_allreduce}; all_reduce_large "
                  f"({schedule}) differs in {large_differs} elements per rank")
        return label, ok, detail

    def graph(self, seeds: Sequence[int] = (71, 72)) -> Check:
        """Reduce-scatter, all-to-all and (when available) Swing captured per rank and replayed."""
        from .gpu_emulation import wait_stream

        torch = self.torch
        group = self.group
        name = self.dtypes[0]
        dtype = _torch_dtype(torch, name)
        rows, row_elems = 6, 64
        swing_elems = 4096
        state: list[dict] = [dict() for _ in range(self.world)]
        try:
            # Captures run one rank after another: a capture starts with a device
            # synchronization that would invalidate a concurrent capture.
            for rank, session in enumerate(group.sessions):
                with torch.cuda.stream(group.streams[rank]):
                    x = torch.zeros((self.world, rows, row_elems), dtype=dtype, device=session.device)
                    s = torch.zeros(swing_elems, dtype=dtype, device=session.device)
                    out = torch.zeros((self.world, rows, row_elems), dtype=dtype, device=session.device)
                    graph = torch.cuda.CUDAGraph()
                    with session.capture():
                        with torch.cuda.graph(graph, stream=group.streams[rank]):
                            reduced = self._reduce_scatter(session, x)
                            self._all_to_all(session, x, out)
                            swung = self._swing(session, s) if self.swing_available else None
                    wait_stream(group.streams[rank])
                state[rank].update(x=x, s=s, out=out, reduced=reduced, swung=swung, graph=graph)
            for seed in seeds:
                xs = _inputs(torch, self.world, (self.world, rows, row_elems), name, seed)
                ss = _inputs(torch, self.world, (swing_elems,), name, seed + 500)
                staged_x, staged_s = self._staged(xs), self._staged(ss)

                def replay(rank: int, session: Any):
                    state[rank]["x"].copy_(staged_x[rank])
                    state[rank]["s"].copy_(staged_s[rank])
                    state[rank]["graph"].replay()

                group.each(replay)
                chunk = rows * row_elems * xs[0].element_size()
                reduce_expected = self._reduce_reference(xs, name, chunk, chunk)
                copy_expected = self._copy_reference(xs, chunk, chunk, chunk, self.world * chunk)
                swing_expected = (models.swing_reference([_to_numpy(torch, s, name) for s in ss], name).tobytes()
                                  if self.swing_available else None)
                for rank in range(self.world):
                    if _bits(torch, state[rank]["reduced"]) != reduce_expected[rank]:
                        return "graph replay", False, f"seed {seed}: rank {rank} reduce-scatter differs"
                    if _bits(torch, state[rank]["out"]) != copy_expected[rank]:
                        return "graph replay", False, f"seed {seed}: rank {rank} all-to-all differs"
                    if self.swing_available and _bits(torch, state[rank]["swung"]) != swing_expected:
                        return "graph replay", False, f"seed {seed}: rank {rank} Swing differs"
        except Exception as error:  # noqa: BLE001
            return "graph replay", False, f"{type(error).__name__}: {error}"
        return "graph replay", True, f"{len(seeds)} replays of reduce-scatter, all-to-all" + (
            ", Swing" if self.swing_available else "")

    def mixed(self, rounds: int = 3) -> list[Check]:
        """Swing, one-shot all-reduce and reduce-scatter ops interleaved on one session."""
        checks = []
        for round_ in range(rounds):
            if self.swing_available:
                checks.append(self.swing("bfloat16", 16 * (1 + 211 * round_)))
            inputs = _inputs(self.torch, self.world, (64 * (round_ + 1),), "bfloat16", self._next_seed())
            flats = [_to_numpy(self.torch, x, "bfloat16") for x in inputs]
            expected = [models.rank_order_sum(flats, "bfloat16").tobytes()] * self.world
            checks.append(self.run(f"one-shot all_reduce bfloat16 {128 * (round_ + 1)} B", inputs,
                                   lambda rank, session, x: session.all_reduce(x, algorithm="oneshot"), expected))
            checks.append(self.reduce_scatter_contiguous("bfloat16", 16 * (round_ + 2)))
        return [(f"mixed: {name}", ok, detail) for name, ok, detail in checks]

    def fail_stop(self, wait_s: float = 0.5) -> Check:
        """Rank 0 never joins a reduce-scatter; every other rank times out, poisons and names rank 0."""
        from .gpu_emulation import wait_stream

        torch = self.torch
        for session in self.group.sessions:
            session.serving_wait_s = wait_s
            session.enter_serving()
        inputs = self._staged(_inputs(torch, self.world, (self.world, 64), "bfloat16", self._next_seed()))
        errors: list[str | None] = [None] * self.world

        def operation(rank: int, session: Any):
            if rank == 0:
                return None
            try:
                self._reduce_scatter(session, inputs[rank])
                wait_stream(torch.cuda.current_stream())
                session.check_health()
            except RuntimeError as error:
                errors[rank] = str(error)
            return None

        started = time.monotonic()
        try:
            self.group.each(operation)
        except Exception as error:  # noqa: BLE001
            return "fail-stop", False, f"{type(error).__name__}: {error}"
        elapsed = time.monotonic() - started
        poisoned = [session.poisoned for session in self.group.sessions]
        named = [rank for rank in range(1, self.world) if errors[rank] and "rank 0" in errors[rank]]
        ok = not poisoned[0] and all(poisoned[1:]) and len(named) == self.world - 1
        return "fail-stop", ok, f"{elapsed:.1f} s; poisoned {poisoned}; errors naming rank 0 on ranks {named}"


def load_modules(group: Any, dtypes: Sequence[str]) -> None:
    """Launch every prepared kernel of every rank once with the poison word set (loads its module)."""
    import torch

    from ..oneshot import _scatter_ops, _swing_ops
    from .gpu_emulation import wait_stream

    for rank, session in enumerate(group.sessions):
        with torch.cuda.stream(group.streams[rank]):
            poison = session._counter_layout.poison_word
            session._counters[poison] = 1
            for name in dtypes:
                dtype = _torch_dtype(torch, name)
                x = torch.zeros(group.world * (8 if dtype != torch.float32 else 4), dtype=dtype,
                                device=session.device)
                _scatter_ops.reduce_scatter(session, x)
                if session._available.get("swing", False) and _swing_ops.is_prepared(session, dtype):
                    _swing_ops.all_reduce(session, x)
            x = torch.zeros(group.world * 16, dtype=torch.uint8, device=session.device)
            _scatter_ops.all_to_all(session, x, torch.zeros_like(x))
            wait_stream(group.streams[rank])
            session._counters[poison] = 0
            wait_stream(group.streams[rank])


def run_checks(layout_text: str = "ring:8", lanes: int = 2, *, library: str | os.PathLike | None = None,
               max_size: int = 256 << 10, dtypes: Sequence[str] = ("bfloat16", "float32"),
               fail_stop: bool = True, large_reduce_scatter: Sequence[int] = (), only_large: bool = False,
               swing_sizes: Sequence[int] = (), only_swing: bool = False,
               report: Callable[[Check], None] | None = None) -> list[Check]:
    """Every check above on one emulated group.

    ``large_reduce_scatter`` lists the MiB sizes of the large reduce-scatter
    checks; ``only_large`` runs those checks alone (with ``prepare`` and the
    health check). ``swing_sizes`` replaces the Swing message sizes (default 16,
    48, 4,096 and 65,552 bytes and the capacity), and ``only_swing`` runs the
    Swing checks alone: every size eager per dtype, the largest captured and
    replayed, the health check and the native counters.
    """
    import torch

    from ..oneshot import _scatter_ops, _swing_ops
    from .gpu_emulation import EmulatedGroup

    from .. import routes as routes_mod

    if library is None:
        build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", Path.cwd() / ".build" / "sim"))
        library = native_build.build_shared_library(build)
    environment = {}
    if "SIRCL_LARGE_BLOCKS" not in os.environ:
        environment["SIRCL_LARGE_BLOCKS"] = str(emulation_large_blocks(routes_mod.Layout.parse(layout_text).world))
    group = EmulatedGroup(layout_text, lanes, max_size=max_size, max_gather_bytes=0, library=library,
                          environment=environment)
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        if report is not None:
            report(check)

    try:
        offered = [enable(session) for session in group.sessions]
        scatter_session = all(scatter for scatter, _ in offered)
        swing_session = all(swing for _, swing in offered)
        for session in group.sessions:
            # These checks exercise the scatter kernel: a session's chain reduce-scatter
            # (``scatter_schedule`` "auto" from ``chain_min_bytes`` on) rounds once per hop
            # and has checks of its own in the core's emulation suite.
            if hasattr(session, "scatter_schedule"):
                session.scatter_schedule = "pieces"
        started = time.perf_counter()
        # bfloat16 is always prepared: the interleaved checks use it. Every rank
        # compiles and loads every launcher before any collective, because a rank
        # that loads a module while a peer's kernel waits for it stalls the group.
        names = tuple(dict.fromkeys(("bfloat16", *dtypes)))
        types = tuple(_torch_dtype(torch, name) for name in names)
        # Compilation is not a collective; one rank after another keeps the compiler single-threaded.
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                swing_wanted = session._available.get("swing", False) and not only_large
                swing = ("swing",) if swing_wanted and swing_session else ()
                session.prepare(types, algorithms=("oneshot",) + swing, scatter=scatter_session)
                if not scatter_session:
                    _scatter_ops.prepare(session, types)
                if swing_wanted and not swing_session:
                    for dtype in types:
                        _swing_ops.launcher(session, dtype, capturing=False)
        group.load_modules(types)
        load_modules(group, names)
        swing_text = ("unavailable" if not group.sessions[0]._available["swing"] else
                      f"through {'the session methods' if swing_session else 'the host functions'}")
        add(("prepare", True, f"{time.perf_counter() - started:.1f} s; large grid {group.sessions[0]._large_blocks} "
             f"blocks per rank; scatter collectives through "
             f"{'the session methods' if scatter_session else 'the host functions'}; Swing {swing_text}"))
        suite = KernelChecks(group, dtypes, scatter_session, swing_session)
        if only_large:
            suite.swing_available = False
        world = group.world
        sizes = sorted(set(swing_sizes)) or sorted({16, 48, 4096, 65536 + 16, max_size})
        if only_swing:
            if not suite.swing_available:
                add(("swing", False, f"Swing is unavailable on a group of {world}"))
                return checks
            for name in dtypes:
                for nbytes in sizes:
                    add(suite.swing(name, nbytes))
            add(suite.swing_graph(dtypes[0], max(sizes)))
            healthy = [not session.poisoned for session in group.sessions]
            add(("health", all(healthy), "" if all(healthy) else f"healthy ranks {healthy}"))
            stats = group.sessions[0].stats()
            add(("native counters", True, f"ops {stats.get('ops_posted')}, later phases "
                 f"{stats.get('later_phases_posted')}, forward chunks {stats.get('forward_chunks_posted')}, "
                 f"most unacknowledged {stats.get('forward_max_unacked_bytes')} bytes"))
            return checks
        for mib in large_reduce_scatter:
            add(suite.large_reduce_scatter(mib))
        if large_reduce_scatter:
            add(suite.large_reduce_scatter(min(large_reduce_scatter), "graph"))
        if only_large:
            healthy = [not session.poisoned for session in group.sessions]
            add(("health", all(healthy), "" if all(healthy) else f"healthy ranks {healthy}"))
            return checks
        per_rank_max = max_size // world // proto.PACK_BYTES * proto.PACK_BYTES
        for name in dtypes:
            for chunk in sorted({16, 16 * 37, 8192, per_rank_max}):
                add(suite.reduce_scatter_contiguous(name, chunk))
            add(suite.reduce_scatter_strided(name, 9, 7, 64))
            add(suite.all_to_all(name, 5, 5, 64, 5))
            add(suite.all_to_all(name, 9, 7, 64, 8))
            if suite.swing_available:
                for nbytes in sizes:
                    add(suite.swing(name, nbytes))
        for check in suite.split_ops(dtypes[0], 1024):
            add(check)
        add(suite.graph())
        for check in suite.mixed():
            add(check)
        healthy = [not session.poisoned for session in group.sessions]
        add(("health", all(healthy), "" if all(healthy) else f"healthy ranks {healthy}"))
        stats = group.sessions[0].stats()
        add(("native counters", True, f"ops {stats.get('ops_posted')}, later phases "
             f"{stats.get('later_phases_posted')}, forward chunks {stats.get('forward_chunks_posted')}"))
        if fail_stop:
            add(suite.fail_stop())
    finally:
        group.close()
    return checks


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default="ring:8")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=256 << 10)
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--no-fail-stop", action="store_true")
    parser.add_argument("--large-reduce-scatter", default="",
                        help="MiB sizes of [rows, 4096] BF16 reduce-scatters, e.g. 8,32,64")
    parser.add_argument("--only-large", action="store_true", help="run the large reduce-scatter checks alone")
    parser.add_argument("--swing-sizes", default="", help="Swing message sizes in bytes, e.g. 1048576,2097152")
    parser.add_argument("--only-swing", action="store_true", help="run the Swing checks alone")
    args = parser.parse_args(argv)

    def show(check: Check) -> None:
        name, ok, detail = check
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    large = tuple(int(item) for item in args.large_reduce_scatter.split(",") if item.strip())
    swing = tuple(int(item) for item in args.swing_sizes.split(",") if item.strip())
    checks = run_checks(args.layout, args.lanes, max_size=args.max_size, dtypes=tuple(args.dtypes.split(",")),
                        fail_stop=not args.no_fail_stop, large_reduce_scatter=large, only_large=args.only_large,
                        swing_sizes=swing, only_swing=args.only_swing, report=show)
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"{len(checks)} checks, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
