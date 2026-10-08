"""In-process GPU emulation of one ring-session group (test support).

Every rank of a group runs as a thread of one process on one CUDA device.
The native layer is the simulator build of the library (the in-memory verbs
stand-in, ``fake_verbs``), so an RDMA write is a memory copy between two
ranks' pinned arenas; ``torch.distributed`` is replaced by a group of threads
(:class:`ThreadGroup`) for the setup exchange. The session class, its
kernels and its large-message paths run unchanged, on any GPU that addresses
pinned host memory at its host pointer (unified addressing).

Each rank has its own CUDA stream; ranks wait for their streams by polling
events with the interpreter lock released, so no rank blocks another rank's
launches or the stand-in's delivery thread. All ranks share one CUDA context,
where loading a kernel module waits for running kernels; a kernel waiting for
its peers would then hold back the very launch it waits for. Before any
collective, :meth:`EmulatedGroup.load_modules` therefore launches every
prepared kernel of every rank once with the session's poison word set (the
kernels return at once) and clears it again, and :func:`main` asks CUDA to
load every module when the context is created (``CUDA_MODULE_LOADING=EAGER``),
so library kernels used for the first time cannot stall a waiting group
either. Separate processes, as in serving, need none of this.

``python -m sparkring_sircl.testing.gpu_emulation [--layout path:0-3] [--lanes 2]``
runs :func:`run_checks` and prints one line per check. Requirements: CUDA,
torch with CUDA, CUDA Python, the CuTe DSL, and a GCC-compatible compiler for
the simulator library.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .. import references
from .. import routes as routes_mod
from . import native_build
from .fabric import FakeFabric


class ThreadGroup:
    """A stand-in for a CPU process group whose ranks are threads of this process."""

    def __init__(self, world: int) -> None:
        self.world = world
        self._barrier = threading.Barrier(world)
        self._slots: list[Any] = [None] * world
        self._local = threading.local()

    def bind(self, rank: int) -> None:
        self._local.rank = rank

    @property
    def rank(self) -> int:
        return self._local.rank

    def all_gather_object(self, out: list, obj: Any) -> None:
        self._slots[self.rank] = obj
        self._barrier.wait(timeout=600)
        out[:] = list(self._slots)
        self._barrier.wait(timeout=600)


class ThreadDist:
    """The ``torch.distributed`` functions a session calls, for :class:`ThreadGroup` groups."""

    @staticmethod
    def get_rank(group: ThreadGroup) -> int:
        return group.rank

    @staticmethod
    def get_world_size(group: ThreadGroup) -> int:
        return group.world

    @staticmethod
    def all_gather_object(out: list, obj: Any, group: ThreadGroup) -> None:
        group.all_gather_object(out, obj)


def device_name(rank: int, device: str) -> str:
    """The emulated device name of rank ``rank``'s function ``device``."""
    return f"n{rank}.{device}"


@contextlib.contextmanager
def emulated_device_roles(world: int):
    """Let the route checks recognise the emulation's per-rank device names."""
    added = {device_name(rank, role.device): role for rank in range(world) for role in routes_mod.ROLES}
    routes_mod._BY_DEVICE.update(added)
    try:
        yield
    finally:
        for name in added:
            routes_mod._BY_DEVICE.pop(name, None)


def wait_stream(stream, timeout: float = 120.0) -> None:
    """Wait for ``stream``'s work by polling an event (the interpreter lock stays free)."""
    import torch

    event = torch.cuda.Event()
    event.record(stream)
    deadline = time.monotonic() + timeout
    while not event.query():
        if time.monotonic() > deadline:
            raise TimeoutError("GPU work of an emulated rank did not finish")
        time.sleep(0.0002)


class EmulatedGroup:
    """Sessions of every rank of one group, connected through the verbs stand-in."""

    def __init__(self, layout_text: str, lanes: int = 2, *, max_size: int, max_gather_bytes: int,
                 library: str | os.PathLike, environment: dict[str, str] | None = None,
                 path_latency: tuple[int, int, int, int] | None = None) -> None:
        import torch

        from ..oneshot import _proxy, runtime

        self.torch = torch
        self.runtime = runtime
        self.layout_text = layout_text
        self.layout = routes_mod.Layout.parse(layout_text)
        self.world = self.layout.world
        derived = routes_mod.derive_routes(self.layout, lanes)
        os.environ["SIRCL_NATIVE_LIBRARY"] = str(library)
        for name, value in (environment or {}).items():
            os.environ[name] = value
        self.fabric = FakeFabric(_proxy.load(str(library)))
        self.fabric.reset()
        self.fabric.lib.fv_set_ideal(1)
        for rank in range(self.world):
            for index, role in enumerate(routes_mod.ROLES):
                self.fabric.add_device(device_name(rank, role.device), rank, role.port, int(role.secondary),
                                       FakeFabric.gid(rank, index))
        if path_latency is not None:
            # Every lane's relay count from the routes (device index = rank * roles + role index),
            # then the latency of each write and its completion and the queue pairs' sending rate.
            roles = list(routes_mod.ROLES)
            for rank in range(self.world):
                for peer in range(self.world):
                    if peer == rank:
                        continue
                    for lane in derived.lanes_to(rank, peer):
                        gid = (ctypes.c_uint8 * 16)(*FakeFabric.gid(peer, roles.index(lane.remote)))
                        self.fabric.lib.fv_set_dest_tag(rank * len(roles) + roles.index(lane.local), gid,
                                                        len(lane.relays))
            base, per_relay, rate, ack_delay = path_latency
            self.fabric.lib.fv_set_latency(int(base), int(per_relay))
            self.fabric.lib.fv_set_rate(int(rate))
            self.fabric.lib.fv_set_ack_delay(int(ack_delay))
        self.fabric.start(spin=path_latency is not None)
        self.streams = [torch.cuda.Stream() for _ in range(self.world)]
        self.sessions: list[Any] = [None] * self.world
        group = ThreadGroup(self.world)

        def construct(rank: int) -> None:
            group.bind(rank)
            peer_routes = {peer: tuple(device_name(rank, device) for device in devices)
                           for peer, devices in derived.route_map(rank).items()}
            self.sessions[rank] = runtime.AllReduce(
                exchange_group=group, device=torch.device("cuda", 0), max_size=max_size,
                max_gather_bytes=max_gather_bytes, peer_routes=peer_routes, layout=layout_text, gid_index=3,
                lane_check_ms=10000,
            )

        self._roles = emulated_device_roles(self.world)
        self._roles.__enter__()
        saved = runtime.dist
        runtime.dist = ThreadDist
        try:
            self._threads(construct)
        finally:
            runtime.dist = saved

    def _threads(self, body: Callable[[int], Any], timeout: float = 900.0) -> list[Any]:
        torch = self.torch
        results: list[Any] = [None] * self.world
        errors: list[BaseException | None] = [None] * self.world

        def run(rank: int) -> None:
            try:
                torch.cuda.set_device(0)
                with torch.cuda.stream(self.streams[rank]):
                    results[rank] = body(rank)
                    wait_stream(self.streams[rank])
            except BaseException as error:  # noqa: BLE001 - re-raised below with the rank
                errors[rank] = error
                traceback.print_exc()

        threads = [threading.Thread(target=run, args=(rank,), daemon=True) for rank in range(self.world)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("an emulated rank did not finish")
        for rank, error in enumerate(errors):
            if error is not None:
                raise RuntimeError(f"rank {rank}: {type(error).__name__}: {error}") from error
        return results

    def load_modules(self, dtypes: Sequence[Any]) -> None:
        """Launch every prepared kernel of every rank once, poisoned, so its module is loaded.

        A compiled launcher loads its module into the shared CUDA context on its first launch; a
        load while another rank's kernel spins waits for that kernel, which waits for this rank.
        """
        torch = self.torch
        for rank, session in enumerate(self.sessions):
            with torch.cuda.stream(self.streams[rank]):
                poison = session._counter_layout.poison_word
                session._counters[poison] = 1
                for dtype in dtypes:
                    x = torch.zeros(8 if dtype != torch.float32 else 4, dtype=dtype, device=session.device)
                    for name in ("oneshot", "twoshot"):
                        if session._available[name]:
                            session.all_reduce(x, algorithm=name)
                if session.max_gather_bytes > 0:
                    shard = torch.zeros((2, 8), dtype=torch.bfloat16, device=session.device)
                    session.all_gather(shard, dim=0)
                    session.all_gather_large(shard, dim=-1)
                if session.chain_available:
                    for dtype in dtypes:
                        x = torch.zeros(64, dtype=dtype, device=session.device)
                        session._launch_chain(session._chain_launcher(dtype, False), x, torch.empty_like(x), False)
                if session.link_available:
                    x = torch.zeros(64, dtype=torch.bfloat16, device=session.device)
                    if ("link-gather",) in session._launchers:
                        session._launch_gather_chain(x, torch.empty(64 * self.world, dtype=x.dtype,
                                                                    device=x.device), False)
                    for dtype in dtypes:
                        if ("link-scatter", dtype) in session._launchers:
                            y = torch.zeros(8 * self.world, dtype=dtype, device=session.device)
                            session._reduce_scatter_chain(y, None, None, None, None)
                    # Every compiled ring launcher, whatever its dtype.
                    for key in [key for key in session._launchers if key[0] == "link-ring"]:
                        dtype = torch.uint8 if key[2] == "bytes" else getattr(torch, key[2])
                        y = torch.zeros(16 * self.world, dtype=dtype, device=session.device)
                        out = torch.zeros(16 * self.world * self.world, dtype=dtype, device=session.device)
                        session._launch_ring(session._launchers[key], y, out, 16, 16, False)
                if session.scatter_available:
                    from ..oneshot import _scatter_cute, _scatter_ops

                    for dtype in dtypes:
                        if _scatter_cute.is_launcher_prepared(*_scatter_ops._key(session, "reduce", dtype)):
                            y = torch.zeros(8 * self.world, dtype=dtype, device=session.device)
                            _scatter_ops.reduce_scatter(session, y)
                    if _scatter_cute.is_launcher_prepared(*_scatter_ops._key(session, "copy", torch.uint8)):
                        y = torch.zeros(16 * self.world, dtype=torch.uint8, device=session.device)
                        _scatter_ops.all_to_all(session, y, torch.empty_like(y))
                wait_stream(self.streams[rank])
                session._counters[poison] = 0
                wait_stream(self.streams[rank])

    def each(self, operation: Callable[[int, Any], Any]) -> list[Any]:
        """``operation(rank, session)`` on every rank at once, on the rank's stream."""
        return self._threads(lambda rank: operation(rank, self.sessions[rank]))

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._threads(lambda rank: self.sessions[rank].close() if self.sessions[rank] else None)
        self.fabric.stop()
        self._roles.__exit__(None, None, None)


# -- checks -----------------------------------------------------------------------------------


def _inputs(torch, world: int, shape: Sequence[int], dtype, seed: int):
    tensors = []
    for rank in range(world):
        generator = torch.Generator().manual_seed(seed * 1009 + rank)
        tensors.append(torch.randn(tuple(shape), generator=generator).to(dtype))
    return tensors


def _sum(torch, inputs):
    total = inputs[0].float().clone()
    for tensor in inputs[1:]:
        total += tensor.float()
    return total.to(inputs[0].dtype)


def chain_reference(torch, inputs, order: Sequence[int], plan) -> object:
    """The result of ``all_reduce_large`` for ``plan`` (:func:`sparkring_sircl.references.large_all_reduce`)."""
    return references.large_all_reduce(torch, inputs, plan, order)


def _same_bits(torch, got, want) -> bool:
    if got.shape != want.shape or got.dtype != want.dtype:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[want.element_size()]
    return bool(torch.equal(got.contiguous().view(view), want.contiguous().view(view)))


def _collective(group: EmulatedGroup, name: str, inputs, call: Callable, reference) -> tuple[str, bool, str]:
    torch = group.torch

    def operation(rank: int, session):
        x = inputs[rank].to(session.device)
        return call(session, x)

    try:
        outputs = group.each(operation)
        outputs = [output.cpu() for output in outputs]
    except Exception as error:  # noqa: BLE001 - reported as a failed check
        return name, False, f"{type(error).__name__}: {error}"
    wrong = [rank for rank, output in enumerate(outputs) if not _same_bits(torch, output, reference)]
    return name, not wrong, f"ranks {wrong} differ" if wrong else ""


def _graph_check(group: EmulatedGroup, name: str, shape, dtype, call: Callable, combine: Callable,
                 seeds: Sequence[int], per_rank: bool = False) -> tuple[str, bool, str]:
    """Capture ``call`` once per rank, replay it for every seed, compare every replay (with
    ``per_rank``, ``combine`` returns one reference per rank)."""
    torch = group.torch
    state: list[dict] = [dict() for _ in range(group.world)]

    try:
        # Captures run one rank after another: nothing executes while capturing, and a capture
        # begins with a device synchronization that would invalidate a concurrent capture.
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                x = torch.zeros(tuple(shape), dtype=dtype, device=session.device)
                graph = torch.cuda.CUDAGraph()
                with session.capture():
                    with torch.cuda.graph(graph, stream=group.streams[rank]):
                        y = call(session, x)
            state[rank].update(x=x, y=y, graph=graph)
        for seed in seeds:
            inputs = _inputs(torch, group.world, shape, dtype, seed)

            def replay(rank: int, session):
                state[rank]["x"].copy_(inputs[rank].to(session.device))
                state[rank]["graph"].replay()
                return None

            group.each(replay)
            reference = combine(inputs)
            wrong = [rank for rank in range(group.world)
                     if not _same_bits(torch, state[rank]["y"].cpu(), reference[rank] if per_rank else reference)]
            if wrong:
                return name, False, f"seed {seed}: ranks {wrong} differ"
    except Exception as error:  # noqa: BLE001
        return name, False, f"{type(error).__name__}: {error}"
    return name, True, f"{len(seeds)} replays"


def _set_all(group: EmulatedGroup, **values) -> None:
    for session in group.sessions:
        for name, value in values.items():
            if name == "chain_chunk_bytes":
                session.set_chain_chunk_bytes(value)
            elif name == "link_chunk_bytes":
                session.set_link_chunk_bytes(value)
            else:
                setattr(session, name, value)


def _chain_checks(group: EmulatedGroup, types) -> list[tuple[str, bool, str]]:
    """all_reduce_large on a chain: automatic and forced chain ops, chunk sizes, capture."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.chain_available:
        return [("chain", True, f"not available here (schedule {session0.large_schedule})")]
    order = session0.chain_order
    results: list[tuple[str, bool, str]] = []
    seed = 900
    default_chunk = session0.chain_chunk_bytes
    cases = [("auto", None, (2 << 20) + 6), ("auto", None, (3 << 20) + 16), ("chain", None, 16),
             ("chain", None, 48), ("chain", None, 1 << 20), ("chain", 256 << 10, (5 << 20) + 32),
             ("chain", session0.chain_slot_bytes, (3 << 20) + 48), ("chain", 4096, 200 << 10)]
    for dtype in types:
        item = torch.empty((), dtype=dtype).element_size()
        for schedule, chunk, nbytes in cases:
            seed += 1
            _set_all(group, large_schedule=schedule, chain_chunk_bytes=chunk or default_chunk)
            plan = session0.large_reduce_plan(nbytes // item * item)
            inputs = _inputs(torch, group.world, (nbytes // item,), dtype, seed)
            label = (f"chain {dtype} {nbytes // item * item} B, {schedule}, chunks of "
                     f"{chunk or default_chunk} B ({'chain op' if any(p.chain for p in plan) else 'pieces'})")
            results.append(_collective(group, label, inputs, lambda session, x: session.all_reduce_large(x),
                                       chain_reference(torch, inputs, order, plan)))
    _set_all(group, large_schedule="chain", chain_chunk_bytes=default_chunk)
    results.append(_repeat_check(group, "chain all_reduce_large on the same inputs under 4 fabric orders",
                                 _inputs(torch, group.world, ((3 << 20) // 2,), torch.bfloat16, 999),
                                 lambda session, x: session.all_reduce_large(x)))
    _set_all(group, large_schedule="auto", chain_chunk_bytes=default_chunk)
    nbytes = (4 << 20) + 32
    plan = session0.large_reduce_plan(nbytes)
    results.append(_graph_check(group, "graph chain all_reduce_large", (nbytes // 2,), torch.bfloat16,
                                lambda session, x: session.all_reduce_large(x),
                                lambda inputs: chain_reference(torch, inputs, order, plan), (931, 932)))
    native = session0.stats()
    results.append(("chain counters", native.get("chain_ops", 0) > 0,
                    f"order {list(order)}, {native.get('chain_ops')} chain ops, {native.get('chain_chunks_posted')} "
                    f"chunks, {native.get('chain_credits_sent')} credits on rank 0"))
    return results


def _trace_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """The event trace of one chain all-reduce on every rank: every chunk a stream posted was traced
    ready, posted and done in that order; the kernel staged every chunk of both halves once; kernel
    times map onto the host clock (a staged chunk is seen ready after it was staged, within the
    clock probe's error and a millisecond of timer granularity); nothing was lost."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.event_trace or not session0.chain_available:
        return []
    from ..ring import trace as trace_mod

    saved = (session0.large_schedule, session0.chain_chunk_bytes)
    _set_all(group, large_schedule="chain", chain_chunk_bytes=65536)
    nbytes = (1 << 20) + 4096
    chunks = sum(-(-half // 65536) for half in proto_halves(nbytes))
    inputs = _inputs(torch, group.world, (nbytes // 2,), torch.bfloat16, 1601)
    try:
        group.each(lambda rank, session: session.event_trace_records())
        outputs = [output.cpu() for output in group.each(
            lambda rank, session: session.all_reduce_large(inputs[rank].to(session.device)))]
        traces = group.each(lambda rank, session: session.event_trace_records())
    except Exception as error:  # noqa: BLE001
        _set_all(group, large_schedule=saved[0], chain_chunk_bytes=saved[1])
        return [("event trace of a chain all-reduce", False, f"{type(error).__name__}: {error}")]
    _set_all(group, large_schedule=saved[0], chain_chunk_bytes=saved[1])
    problems, notes = [], []
    if any(not _same_bits(torch, output, outputs[0]) for output in outputs):
        problems.append("traced ranks differ")
    for rank, trace in enumerate(traces):
        records = trace["records"]
        if trace["lost"] != {"native": 0, "kernel": 0}:
            problems.append(f"rank {rank} lost {trace['lost']}")
        first: dict[tuple[str, int, int], int] = {}
        for ns, _source, event, stream, value in records:
            first.setdefault((event, stream, value), ns)
        staged = sum(1 for (event, _, _) in first if event == "KERNEL_READY")
        if staged != chunks:
            problems.append(f"rank {rank} staged {staged} chunks, the op has {chunks}")
        error = int(trace.get("offset_error_ns") or 0) + 1_000_000
        for (event, stream, tag), ns in first.items():
            if event != "POSTED":
                continue
            ready, done = first.get(("READY", stream, tag)), first.get(("DONE", stream, tag))
            if ready is None or done is None or not ready <= ns <= done:
                problems.append(f"rank {rank} stream {stream} chunk {tag}: ready {ready}, posted {ns}, done {done}")
                break
            kernel = first.get(("KERNEL_READY", stream, tag))
            if kernel is not None and ready < kernel - error:
                problems.append(f"rank {rank} stream {stream} chunk {tag} seen ready {kernel - ready} ns before "
                                f"it was staged (clock error {error} ns)")
                break
        summary = trace_mod.summary(records)
        notice = [values["notice"]["p50_us"] for values in summary.values() if "notice" in values]
        notes.append(f"rank {rank}: {len(records)} records, clock within {trace.get('offset_error_ns')} ns"
                     + (f", notice median {min(notice):.1f}-{max(notice):.1f} us" if notice else ""))
    return [("event trace of a chain all-reduce", not problems, "; ".join(problems[:4] or notes))]


def proto_halves(nbytes: int) -> tuple[int, int]:
    """Bytes of the two halves of a chain op of ``nbytes`` (``protocol.chain_halves`` in bytes)."""
    from .. import protocol

    a_packs, b_packs = protocol.chain_halves(nbytes // 16)
    return a_packs * 16, b_packs * 16


LATE_TAKE_US = 2000         # the late rank takes each link op this long after its doorbell


def _late_link_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """Link ops issued back to back on every rank while one rank's progress thread takes every link op
    ``LATE_TAKE_US`` after its doorbell (the native test hook ``roce_test_delay_link_ops``): that rank's
    kernel then finishes inbound items from its neighbors before their op is taken, and the progress thread
    must wait for the op. Two sequences, each once with the first and once with the last rank of the chain
    late: ops that switch between reduce-scatter, all-gather and (with a ring) the ring all-reduce, between
    the chain and ring schedules and between piece sizes; and runs of one op at a constant piece size, as a
    timed loop issues them. Every output against its reference."""
    session0 = group.sessions[0]
    if not session0.link_available:
        return [("link ops taken late", True, "not available here")]
    switching = [("scatter", "chain", 262144), ("scatter", "chain", 524288), ("gather", "chain", 262144),
                 ("gather", "chain", 4096), ("scatter", "chain", 4096)]
    constant = [("scatter", "chain", 524288)] * 6 + [("gather", "chain", 262144)] * 6
    if session0.ring_available:
        switching += [("reduce", "ring", 524288), ("scatter", "ring", 262144), ("gather", "ring", 524288),
                      ("reduce", "ring", 65536), ("scatter", "chain", 65536), ("gather", "ring", 4096)]
        constant += [("reduce", "ring", 524288)] * 6
    return (_late_link_sequence(group, "switching schedule and pieces", switching)
            + _late_link_sequence(group, "at constant pieces", constant))


def _late_link_sequence(group: EmulatedGroup, label: str, steps,
                        late_ranks: Sequence[int | None] | None = None) -> list[tuple[str, bool, str]]:
    """``steps`` (kind, schedule, piece bytes) back to back on every rank, once per entry of ``late_ranks``
    (default the first and the last rank of the chain) with that rank taking its link ops late (None: no
    rank late); every output against its reference."""
    torch = group.torch
    session0 = group.sessions[0]
    world, order = group.world, session0.chain_order
    bf16 = torch.bfloat16
    saved = (session0.large_schedule, session0.gather_schedule, session0.scatter_schedule,
             session0.link_chunk_bytes)
    inputs, expected = [], []
    for index, (kind, schedule, piece) in enumerate(steps):
        shape = {"scatter": (world * 128, 1024), "gather": (128, 1024), "reduce": ((1 << 20) // 2 + world * 8,)}[kind]
        step_inputs = _inputs(torch, world, shape, bf16, 1700 + index)
        inputs.append(step_inputs)
        if kind == "scatter":
            reference = references.chain_reduce_scatter if schedule == "chain" else references.ring_reduce_scatter
            expected.append([rows.reshape((shape[0] // world, *shape[1:]))
                             for rows in reference(torch, step_inputs, order)])
        elif kind == "gather":
            expected.append([torch.cat(step_inputs, dim=0)] * world)
        else:
            session0.large_schedule = schedule
            plan = session0.large_reduce_plan(step_inputs[0].numel() * 2)
            session0.large_schedule = saved[0]
            expected.append([references.large_all_reduce(torch, step_inputs, plan, order)] * world)

    def operation(rank: int, session):
        on_device = [step_inputs[rank].to(session.device) for step_inputs in inputs]
        outputs = []
        for (kind, schedule, piece), x in zip(steps, on_device):
            session.set_link_chunk_bytes(piece)
            if kind == "scatter":
                session.scatter_schedule = schedule
                outputs.append(session.reduce_scatter(x))
            elif kind == "gather":
                session.gather_schedule = schedule
                outputs.append(session.all_gather_large(x, dim=0))
            else:
                session.large_schedule = schedule
                outputs.append(session.all_reduce_large(x))
        return outputs

    results = []
    for late in (order[0], order[-1]) if late_ranks is None else late_ranks:
        session = group.sessions[late if late is not None else order[0]]
        delay = session._proxy._lib.roce_test_delay_link_ops
        delay.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        delay.restype = None
        name = ((f"link ops taken late on rank {late} ({LATE_TAKE_US} us after each doorbell)" if late is not None
                 else "link ops on time") + f": {len(steps)} ops back to back, {label}")
        try:
            delay(session._proxy._handle(), LATE_TAKE_US if late is not None else 0)
            try:
                outputs = [[output.cpu() for output in rank_outputs] for rank_outputs in group.each(operation)]
            finally:
                delay(session._proxy._handle(), 0)
        except Exception as error:  # noqa: BLE001 - reported as a failed check
            results.append((name, False, f"{type(error).__name__}: {error}"))
            break
        wrong = [f"op {index} ({steps[index][0]}, {steps[index][1]}, pieces of {steps[index][2]} B) rank {rank}"
                 for index in range(len(steps)) for rank in range(world)
                 if not _same_bits(torch, outputs[rank][index], expected[index][rank])]
        results.append((name, not wrong, "; ".join(wrong[:4])))
    _set_all(group, large_schedule=saved[0], gather_schedule=saved[1], scatter_schedule=saved[2],
             link_chunk_bytes=saved[3])
    return results


MIXED_PIECES = {"gather": 65536, "scatter": 4096, "reduce": 16384}


def _mixed_piece_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """Link collectives with pieces of their own (``MIXED_PIECES``: all-gather 64 KiB, reduce-scatter 4 KiB,
    ring all-reduce 16 KiB) while each step sets the session's piece to another value: back to back on time,
    where every rank's link items must equal the count the collectives' own pieces give
    (``protocol.link_rounds``, ``link_pieces``), and with the first and the last rank of the chain taking
    their link ops late (:func:`_late_link_sequence`); every output against its reference."""
    session0 = group.sessions[0]
    if not session0.link_available or not callable(getattr(session0, "link_chunk_for", None)):
        return [("link collectives with pieces of their own", True, "not available here")]
    from .. import protocol as proto

    world, order = group.world, session0.chain_order
    saved = {collective: dict(session0._link_chunks).get(collective, 0) for collective in MIXED_PIECES}
    for session in group.sessions:
        for collective, piece in MIXED_PIECES.items():
            session.set_link_chunk_bytes(piece, collective=collective)
    steps = [("gather", "chain", 524288), ("scatter", "chain", 262144), ("gather", "chain", 4096)]
    if session0.ring_available:
        steps += [("reduce", "ring", 524288), ("gather", "ring", 262144), ("scatter", "ring", 524288),
                  ("reduce", "ring", 4096), ("scatter", "chain", 65536)]
    ops = {("gather", "chain"): proto.LinkOp.ALL_GATHER, ("scatter", "chain"): proto.LinkOp.REDUCE_SCATTER,
           ("gather", "ring"): proto.LinkOp.RING_GATHER, ("scatter", "ring"): proto.LinkOp.RING_SCATTER,
           ("reduce", "ring"): proto.LinkOp.RING_REDUCE}
    reduce_bytes = ((1 << 20) // 2 + world * 8) * 2
    schedule = session0.large_schedule
    session0.large_schedule = "ring"
    ring_bytes = sum(piece.nbytes for piece in session0.large_reduce_plan(reduce_bytes) if piece.ring)
    session0.large_schedule = schedule
    per_rank = {"gather": 128 * 1024 * 2, "scatter": 128 * 1024 * 2, "reduce": ring_bytes // world}
    expected_items = [0] * world
    for kind, schedule, _ in steps:
        pieces = proto.link_pieces(per_rank[kind], MIXED_PIECES[kind])
        # Link 2 of a ring reduce-scatter or all-reduce runs the rounds of the session's stagger, link 3 of
        # a ring all-gather or all-reduce those of its all-gather stagger.
        rounds = ({2: proto.ring_rounds(pieces, world, session0.ring_stagger),
                   3: proto.ring_rounds(pieces, world, session0.ring_gather_stagger)} if schedule == "ring" else {})
        for rank in range(world):
            index = order.index(rank)
            expected_items[rank] += sum(rounds.get(link, pieces)
                                        * proto.link_rounds(ops[(kind, schedule)], world, index, link).out
                                        for link in range(proto.LINKS))
    results = []
    before = [session.stats()["link_items_posted"] for session in group.sessions]
    on_time = _late_link_sequence(group, "pieces of their own", steps, late_ranks=(None,))
    after = [session.stats()["link_items_posted"] for session in group.sessions]
    items = [b - a for a, b in zip(before, after)]
    name, ok, detail = on_time[0]
    counted = items == expected_items
    results.append((name, ok and counted, detail if not ok else
                    "" if counted else f"link items {items}, the pieces of their own give {expected_items}"))
    results += _late_link_sequence(group, "pieces of their own", steps)
    for session in group.sessions:
        for collective, piece in saved.items():
            session.set_link_chunk_bytes(piece, collective=collective)
    return results


RING_MIN_CHECK_BYTES = 256 << 10


def _ring_min_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """Ring schedules for every collective with the ring minimum at ``RING_MIN_CHECK_BYTES``: all-reduces,
    all-gathers and reduce-scatters just below, at and above it back to back. Below it the schedules run as
    auto does (here two-shot pieces, tiles and scatter ops: the chain minimum stays above), at and above it
    as ring ops, the same on every rank; once on time and once with the last rank of the chain taking its
    link ops late. Every output against the reference of the schedule the session chose."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.ring_available or not hasattr(session0, "ring_min_bytes"):
        return [("ring minimum", True, "not available here")]
    world, order = group.world, session0.chain_order
    bf16 = torch.bfloat16
    floor = RING_MIN_CHECK_BYTES
    saved = (session0.large_schedule, session0.gather_schedule, session0.scatter_schedule, session0.ring_min_bytes)
    _set_all(group, large_schedule="ring", gather_schedule="ring", scatter_schedule="ring", ring_min_bytes=floor)
    # (kind, bytes of the message, output or input). All-gather outputs and reduce-scatter inputs are whole
    # rows of 8 KiB per rank, so every world size divides them; ``at`` rows per rank reach the minimum.
    per = 8192 * world
    at = -(-floor // per)
    steps = [("reduce", floor - 16 * world), ("reduce", floor), ("gather", (at - 1) * per), ("gather", 2 * at * per),
             ("scatter", max(1, at // 2) * per), ("scatter", at * per), ("reduce", floor * 2),
             ("scatter", (at - 1) * per)]
    inputs, expected, picked = [], [], []
    for index, (kind, nbytes) in enumerate(steps):
        if kind == "reduce":
            shape = (nbytes // 2,)
        elif kind == "gather":
            shape = (nbytes // world // 8192, 4096)
        else:
            shape = (nbytes // 8192, 4096)
        step_inputs = _inputs(torch, world, shape, bf16, 1900 + index)
        inputs.append(step_inputs)
        probe = step_inputs[0].to(session0.device)
        if kind == "reduce":
            plan = session0.large_reduce_plan(nbytes)
            ring = any(piece.ring for piece in plan)
            expected.append([references.large_all_reduce(torch, step_inputs, plan, order)] * world)
        elif kind == "gather":
            ring = session0.gather_uses_ring(probe, 0)
            expected.append([torch.cat(step_inputs, dim=0)] * world)
        else:
            ring = session0.scatter_uses_ring(probe)
            if ring:
                rows = references.ring_reduce_scatter(torch, step_inputs, order)
            else:
                total = _sum(torch, [tensor.reshape(-1) for tensor in step_inputs])
                rows = list(total.reshape(world, -1))
            expected.append([part.reshape((shape[0] // world, *shape[1:])) for part in rows])
        picked.append(ring)
    wanted = [nbytes >= floor for _, nbytes in steps]
    results = [("ring minimum: the schedule each size runs", picked == wanted,
                "" if picked == wanted else f"ring ops {picked}, the sizes give {wanted}")]

    def operation(rank: int, session):
        on_device = [step_inputs[rank].to(session.device) for step_inputs in inputs]
        outputs = []
        for (kind, _), x in zip(steps, on_device):
            if kind == "reduce":
                outputs.append(session.all_reduce_large(x))
            elif kind == "gather":
                outputs.append(session.all_gather_large(x, dim=0))
            else:
                outputs.append(session.reduce_scatter(x))
        return outputs

    late = order[-1]
    delay = group.sessions[late]._proxy._lib.roce_test_delay_link_ops
    delay.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    delay.restype = None
    for label, delay_us in (("on time", 0), (f"rank {late} taking its link ops late", LATE_TAKE_US)):
        name = f"ring minimum of {floor} B, {label}: {len(steps)} collectives back to back below, at and above it"
        try:
            delay(group.sessions[late]._proxy._handle(), delay_us)
            try:
                outputs = [[output.cpu() for output in rank_outputs] for rank_outputs in group.each(operation)]
            finally:
                delay(group.sessions[late]._proxy._handle(), 0)
        except Exception as error:  # noqa: BLE001 - reported as a failed check
            results.append((name, False, f"{type(error).__name__}: {error}"))
            break
        wrong = [f"{steps[index][0]} of {steps[index][1]} B rank {rank}"
                 for index in range(len(steps)) for rank in range(world)
                 if not _same_bits(torch, outputs[rank][index], expected[index][rank])]
        results.append((name, not wrong, "; ".join(wrong[:4])))
    _set_all(group, large_schedule=saved[0], gather_schedule=saved[1], scatter_schedule=saved[2],
             ring_min_bytes=saved[3])
    return results


def _stagger_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """Ring all-gathers, reduce-scatters and all-reduces back to back under every stagger the link slots
    hold: the reduce-scatter's stagger D on link 2 (``set_ring_stagger``) and the all-gather's stagger D3
    on link 3 (``set_ring_gather_stagger``), the staggers, the piece and the collective changing between
    consecutive ops, the all-reduces with D and D3 apart: blocks of fewer pieces than a stagger adds
    rounds, of as many, and of more. Once on time, where every rank's link items must equal the count of
    the staggered rounds (``protocol.ring_rounds``: empty items included), and once with the last rank of
    the chain taking its link ops late. A stagger changes when items leave, not what they carry, so every
    output is checked against the ring references."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.ring_available:
        return [("ring stagger", True, "not available here")]
    from .. import protocol as proto

    world, order = group.world, session0.chain_order
    bf16 = torch.bfloat16
    staggers = [stagger for stagger in range(proto.MAX_RING_STAGGER + 1)
                if session0.link_slots >= proto.ring_stagger_slots(world, stagger)]

    def block(stagger: int, other: int) -> list[tuple[str, int, int, int, tuple[int, ...]]]:
        """(kind, D, D3, piece bytes, shape): a reduce-scatter and an all-gather of one piece per rank, of
        as many pieces as the stagger adds rounds (``(W - 2) D``, at least one) and of 64 KiB pieces, a
        ring all-reduce of 17 pieces per rank and one of a 96-byte piece (D3 ``other``)."""
        as_many = max(1, (world - 2) * stagger)
        return [("scatter", stagger, 0, 4096, (world * 2, 1024)),
                ("gather", 0, stagger, 4096, (2, 1024)),
                ("scatter", stagger, 0, 4096, (world * 2 * as_many, 1024)),
                ("gather", 0, stagger, 4096, (2 * as_many, 1024)),
                ("scatter", stagger, 0, 65536, (world * 128, 1024)),
                ("gather", 0, stagger, 65536, (128, 1024)),
                ("reduce", stagger, other, 16384, ((1 << 20) // 2 + world * 8,)),
                ("reduce", other, stagger, 4096, (world * 16 * 3,))]

    # The staggers change between every two consecutive ops.
    rotated = staggers[1:] + staggers[:1]
    blocks = [block(stagger, other) for stagger, other in zip(rotated, rotated[1:] + rotated[:1])]
    count = len(blocks[0])
    steps = [blocks[index % len(blocks)][index // len(blocks)] for index in range(count * len(blocks))]
    saved = (session0.large_schedule, session0.gather_schedule, session0.scatter_schedule, session0.ring_stagger,
             session0.ring_gather_stagger)
    _set_all(group, large_schedule="ring", gather_schedule="ring", scatter_schedule="ring")
    inputs, expected, per_rank = [], [], []
    for index, (kind, _, _, _, shape) in enumerate(steps):
        step_inputs = _inputs(torch, world, shape, bf16, 2100 + index)
        inputs.append(step_inputs)
        nbytes = step_inputs[0].numel() * step_inputs[0].element_size()
        if kind == "scatter":
            expected.append([rows.reshape((shape[0] // world, *shape[1:]))
                             for rows in references.ring_reduce_scatter(torch, step_inputs, order)])
            per_rank.append(nbytes // world)
        elif kind == "gather":
            expected.append([torch.cat(step_inputs, dim=0)] * world)
            per_rank.append(nbytes)
        else:
            plan = session0.large_reduce_plan(nbytes)
            expected.append([references.large_all_reduce(torch, step_inputs, plan, order)] * world)
            per_rank.append(sum(piece.nbytes for piece in plan if piece.ring) // world)
    ops = {"scatter": proto.LinkOp.RING_SCATTER, "gather": proto.LinkOp.RING_GATHER,
           "reduce": proto.LinkOp.RING_REDUCE}
    expected_items = [0] * world
    for (kind, stagger, gather_stagger, piece, _), nbytes in zip(steps, per_rank):
        pieces = proto.link_pieces(nbytes, piece)
        for rank in range(world):
            for link, d in ((2, stagger), (3, gather_stagger)):
                rounds = proto.link_rounds(ops[kind], world, order.index(rank), link)
                expected_items[rank] += proto.ring_rounds(pieces, world, d) * rounds.out

    def operation(rank: int, session):
        on_device = [step_inputs[rank].to(session.device) for step_inputs in inputs]
        outputs = []
        for (kind, stagger, gather_stagger, piece, _), x in zip(steps, on_device):
            session.set_ring_stagger(stagger)
            session.set_ring_gather_stagger(gather_stagger)
            session.set_link_chunk_bytes(piece, collective=kind)
            if kind == "scatter":
                outputs.append(session.reduce_scatter(x))
            elif kind == "gather":
                outputs.append(session.all_gather_large(x, dim=0))
            else:
                outputs.append(session.all_reduce_large(x))
        return outputs

    results = []
    late = order[-1]
    delay = group.sessions[late]._proxy._lib.roce_test_delay_link_ops
    delay.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    delay.restype = None
    for label, delay_us in (("on time", 0), (f"rank {late} taking its link ops late", LATE_TAKE_US)):
        name = (f"ring staggers {staggers}, {label}: {len(steps)} ring all-gathers, reduce-scatters and all-reduces "
                "back to back")
        before = [session.stats()["link_items_posted"] for session in group.sessions]
        try:
            delay(group.sessions[late]._proxy._handle(), delay_us)
            try:
                outputs = [[output.cpu() for output in rank_outputs] for rank_outputs in group.each(operation)]
            finally:
                delay(group.sessions[late]._proxy._handle(), 0)
        except Exception as error:  # noqa: BLE001 - reported as a failed check
            results.append((name, False, f"{type(error).__name__}: {error}"))
            break
        items = [session.stats()["link_items_posted"] - count for session, count in zip(group.sessions, before)]
        wrong = [f"{steps[index][0]} {index} (D {steps[index][1]}, D3 {steps[index][2]}, pieces of {steps[index][3]} B) "
                 f"rank {rank}"
                 for index in range(len(steps)) for rank in range(world)
                 if not _same_bits(torch, outputs[rank][index], expected[index][rank])]
        if items != expected_items:
            wrong.insert(0, f"link items {items}, the staggered rounds give {expected_items}")
        results.append((name, not wrong, "; ".join(wrong[:4])))
    for session in group.sessions:
        for collective in ("gather", "scatter", "reduce"):
            session.set_link_chunk_bytes(0, collective=collective)
    _set_all(group, large_schedule=saved[0], gather_schedule=saved[1], scatter_schedule=saved[2])
    for session in group.sessions:
        session.set_ring_stagger(saved[3])
        session.set_ring_gather_stagger(saved[4])
    return results


def _minimum_default_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """The per-collective default minimums (``DEFAULT_CHAIN_MINS``, ``DEFAULT_RING_MINS``) as the
    session's decisions apply them, just below and at each minimum, under ``auto`` and ``ring``; no
    collective runs. Every session returns to the emulation's minimums afterwards."""
    from ..oneshot import runtime as session_module

    torch = group.torch
    session0 = group.sessions[0]
    if not session0.chain_available or not session0.link_available:
        return [("default minimums", True, "not available here")]
    world = group.world
    saved = [({kind: session.chain_min_for(kind) for kind in session_module.MIN_COLLECTIVES},
              {kind: session.ring_min_for(kind) for kind in session_module.MIN_COLLECTIVES})
             for session in group.sessions]
    schedules = (session0.large_schedule, session0.gather_schedule, session0.scatter_schedule)
    wrong = []
    try:
        session0.set_chain_min_bytes(None)
        session0.set_ring_min_bytes(None)
        bf16 = torch.bfloat16
        for kind in session_module.MIN_COLLECTIVES:
            for name, minimum, schedule in (("chain", session_module.DEFAULT_CHAIN_MINS[kind], "auto"),
                                            ("ring", session_module.DEFAULT_RING_MINS[kind], "ring")):
                if schedule == "ring" and not session0.ring_available:
                    continue
                # Sizes whose shares are whole packs: the smallest at or above the minimum, and one less.
                step = 16 * world
                at = -(-minimum // step) * step
                for nbytes in (at - step, at):
                    if kind == "reduce":
                        session0.large_schedule = schedule
                        plan = session0.large_reduce_plan(nbytes)
                        taken = any(getattr(piece, name) for piece in plan)
                    elif kind == "gather":
                        session0.gather_schedule = schedule
                        shard = torch.empty((nbytes // world // 2,), dtype=bf16, device=session0.device)
                        taken = (session0.gather_uses_chain(shard, 0) if name == "chain"
                                 else session0.gather_uses_ring(shard, 0))
                    else:
                        session0.scatter_schedule = schedule
                        inp = torch.empty((nbytes // 2,), dtype=bf16, device=session0.device)
                        taken = (session0.scatter_uses_chain(inp) if name == "chain"
                                 else session0.scatter_uses_ring(inp))
                    if taken != (nbytes >= minimum):
                        wrong.append(f"{kind} of {nbytes} B under {schedule}: {name} {taken}")
    finally:
        session0.large_schedule, session0.gather_schedule, session0.scatter_schedule = schedules
        for session, (chain_mins, ring_mins) in zip(group.sessions, saved):
            for kind in session_module.MIN_COLLECTIVES:
                session.set_chain_min_bytes(chain_mins[kind], collective=kind)
                session.set_ring_min_bytes(ring_mins[kind], collective=kind)
    return [("default minimums: chain and ring from each collective's own size", not wrong, "; ".join(wrong[:4]))]


def _tuning_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """A tuning table on every session (``_tuning``, what ``SIRCL_TUNING_TABLE`` loads at setup) chooses
    each op: all-reduces from 16 B as two-shot ops in a grid of 2, from 4 KiB as one-shot ops in a grid of 1,
    from 64 KiB as two-shot ops in a grid of 4, and from 192 KiB a grid of 1024, which no session can launch
    (the rules then choose); all-gathers as tiles in a grid of 2. Eager and in CUDA graph replay: every
    output bit-exact, ``stats()["tuning"]`` counts each op under its choice or as unusable, and the grid
    caps return afterwards."""
    from .. import tuning as tuning_mod

    torch = group.torch
    session0 = group.sessions[0]
    if not session0._available.get("twoshot"):
        return [("tuning table", True, "no two-shot all-reduce here")]
    world = group.world

    def interval(start, choice):
        return {"from": start, "choice": choice, "nccl": False}

    reduce = [interval(16, {"algorithm": "twoshot", "grid": 2}), interval(4096, {"algorithm": "oneshot", "grid": 1}),
              interval(65536, {"algorithm": "twoshot", "grid": 4}),
              interval(196608, {"algorithm": "twoshot", "grid": 1024})]
    decisions = [{"collective": "all_reduce", "mode": mode, "intervals": reduce} for mode in ("eager", "graph")]
    decisions.append({"collective": "all_gather", "mode": "eager",
                      "intervals": [interval(16, {"schedule": "pieces", "grid": 2})]})
    table = tuning_mod.Table({"schema": tuning_mod.SCHEMA, "key": dict(session0.tuning_facts(), image=""),
                              "run_id": "emulation", "created": "", "measurements": [], "decisions": decisions})
    saved = [(session._tuning, session.large_blocks, session.blocks) for session in group.sessions]
    results = []
    bf16 = torch.bfloat16
    try:
        for session in group.sessions:
            session._tuning = table
            session._tuning_counts.clear()
            session._tuning_unusable.clear()
        wrong = []
        sizes = [nbytes for nbytes in (1024, 8192, 131072, 262144) if nbytes <= session0.max_size]
        for index, nbytes in enumerate(sizes):
            inputs = _inputs(torch, world, (nbytes // 2,), bf16, 300 + index)
            outs = group.each(lambda rank, session, inputs=inputs: session.all_reduce(inputs[rank].to(session.device)))
            torch.cuda.synchronize()
            expected = _sum(torch, inputs).to(bf16)
            wrong += [f"all-reduce of {nbytes} B, rank {rank}" for rank, out in enumerate(outs)
                      if not torch.equal(out.cpu(), expected)]
        gather_inputs = _inputs(torch, world, (64, 256), bf16, 310)
        outs = group.each(lambda rank, session: session.all_gather_large(gather_inputs[rank].to(session.device), dim=0))
        torch.cuda.synchronize()
        wrong += [f"all-gather, rank {rank}" for rank, out in enumerate(outs)
                  if not torch.equal(out.cpu(), torch.cat(gather_inputs, dim=0))]
        results.append(_graph_check(group, "tuning table: captured all-reduce", (32768,), bf16,
                                    lambda session, x: session.all_reduce(x), lambda inputs: _sum(torch, inputs),
                                    (320, 321)))
        stats = session0.stats()["tuning"]
        expected_counts = {"all_reduce/eager/twoshot grid 2": 1, "all_reduce/eager/oneshot grid 1": 1,
                           "all_gather/eager/pieces grid 2": 1}
        if 131072 in sizes:
            expected_counts["all_reduce/eager/twoshot grid 4"] = 1
        counted = {label: stats["decisions"].get(label) for label in expected_counts}
        if counted != expected_counts:
            wrong.append(f"decisions {stats['decisions']}")
        if 262144 in sizes and stats["unusable"].get("all_reduce/eager/twoshot grid 1024") != 1:
            wrong.append(f"unusable {stats['unusable']}")
        if not any(label.startswith("all_reduce/graph/") for label in stats["decisions"]):
            wrong.append("no captured decision counted")
        if stats["table"] != table.hash:
            wrong.append("stats name another table")
        for session, (_, large, blocks) in zip(group.sessions, saved):
            if (session.large_blocks, session.blocks) != (large, blocks):
                wrong.append(f"rank {session.rank} grid caps {session.large_blocks}, {session.blocks} after the ops")
        results.insert(0, ("tuning table: each op runs the table's choice (eager)", not wrong, "; ".join(wrong[:4])))
    finally:
        for session, (tuning, _, _) in zip(group.sessions, saved):
            session._tuning = tuning
    return results


def _tuning_forced_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """A tuning table that chooses, by size, chain, ring (both staggers at the most the link slots hold) and
    two-shot pieces for ``all_reduce_large``, ring and chain reduce-scatters and ring and chain all-gathers,
    alternating in one session with ops that force another schedule under ``untuned()``. Eager: every op
    runs twice on the same inputs, both outputs bit-exact against the reference of the ops that ran (the
    plan and link decisions the session gives under the same choice or forced settings), on every rank; the
    table's ops are counted under their choices and the forced ones not at all. Then one CUDA graph per
    rank holding a table op and a forced op, replayed twice."""
    from .. import protocol as proto
    from .. import tuning as tuning_mod

    torch = group.torch
    session0 = group.sessions[0]
    if not (session0.chain_available and session0.ring_available):
        return [("tuning table beside forced schedules", True, "no chain or ring here")]
    world, order = group.world, session0.chain_order
    bf16 = torch.bfloat16
    stagger = max(d for d in range(proto.MAX_RING_STAGGER + 1)
                  if session0.link_slots >= proto.ring_stagger_slots(world, d))
    # Messages above the capacity, whole packs for every world of 2, 3, 4 and 8 ranks.
    m_chain, m_ring, m_pieces = 384 << 10, 768 << 10, 1536 << 10
    rs_ring, rs_chain = (world * 32, 1024), (world * 128, 1024)
    ag_ring, ag_chain = (16, 1024), (64, 1024)

    def interval(start, choice):
        return {"from": int(start), "choice": choice, "nccl": False}

    ring_choice = {"schedule": "ring", "piece": 32768, "stagger": stagger, "gather_stagger": stagger}
    decisions = []
    for mode in ("eager", "graph"):
        decisions += [
            {"collective": "all_reduce", "mode": mode,
             "intervals": [interval(m_chain, {"schedule": "chain", "piece": 65536}),
                           interval(m_ring, ring_choice), interval(m_pieces, {"schedule": "pieces", "grid": 4})]},
            {"collective": "reduce_scatter", "mode": mode,
             "intervals": [interval(rs_ring[0] * 2048, {"schedule": "ring", "piece": 16384, "stagger": stagger}),
                           interval(rs_chain[0] * 2048, {"schedule": "chain", "piece": 65536})]},
            {"collective": "all_gather", "mode": mode,
             "intervals": [interval(ag_ring[0] * 2048, {"schedule": "ring", "piece": 16384,
                                                          "gather_stagger": stagger}),
                           interval(ag_chain[0] * 2048, {"schedule": "chain", "piece": 32768})]}]
    table = tuning_mod.Table({"schema": tuning_mod.SCHEMA, "key": dict(session0.tuning_facts(), image=""),
                              "run_id": "emulation", "created": "", "measurements": [], "decisions": decisions})
    # (kind, shape, forced settings or None for the table's choice)
    steps = [("reduce", (m_chain // 2,), None), ("reduce", (m_chain // 2,), {"large_schedule": "ring"}),
             ("reduce", (m_ring // 2,), None), ("reduce", (m_ring // 2,), {"large_schedule": "chain"}),
             ("reduce", (m_pieces // 2,), None), ("reduce", (m_pieces // 2,), {"large_schedule": "chain"}),
             ("reduce", (m_chain // 2,), {"large_schedule": "pieces"}),
             ("scatter", rs_ring, None), ("scatter", rs_ring, {"scatter_schedule": "chain"}),
             ("scatter", rs_chain, None), ("scatter", rs_chain, {"scatter_schedule": "ring"}),
             ("scatter", rs_chain, {"scatter_schedule": "pieces"}),
             ("gather", ag_ring, None), ("gather", ag_ring, {"gather_schedule": "pieces"}),
             ("gather", ag_chain, None), ("gather", ag_chain, {"gather_schedule": "ring"})]
    if not session0.scatter_available:
        # Scatter ops need the scatter kernels; the link reduce-scatters do not.
        steps = [step for step in steps if (step[2] or {}).get("scatter_schedule") != "pieces"]
    names = ("large_schedule", "scatter_schedule", "gather_schedule")

    class Forced:
        """``untuned()`` with the step's settings, restored afterwards."""

        def __init__(self, session, settings):
            self.session, self.settings = session, settings or {}

        def __enter__(self):
            self.saved = {name: getattr(self.session, name) for name in names}
            self.context = self.session.untuned() if self.settings else contextlib.nullcontext()
            self.context.__enter__()
            for name, value in self.settings.items():
                setattr(self.session, name, value)
            return self

        def __exit__(self, *exc):
            for name, value in self.saved.items():
                setattr(self.session, name, value)
            return self.context.__exit__(*exc)

    def reference(kind, shape, settings, inputs, mode="eager"):
        with Forced(session0, settings):
            if kind == "reduce":
                plan = session0.large_reduce_plan(inputs[0].numel() * 2, mode=mode)
                return [references.large_all_reduce(torch, inputs, plan, order)] * world, \
                    "ring" if any(piece.ring for piece in plan) else "chain" if any(piece.chain for piece in plan) \
                    else "pieces"
            if kind == "gather":
                probe = torch.empty(shape, dtype=bf16, device="meta")
                used = ("ring" if session0.gather_uses_ring(probe, 0, mode=mode) else
                        "chain" if session0.gather_uses_chain(probe, 0, mode=mode) else "pieces")
                return [torch.cat(inputs, dim=0)] * world, used
            probe = torch.empty(shape, dtype=bf16, device=session0.device)
            if session0.scatter_uses_ring(probe, mode=mode):
                rows = references.ring_reduce_scatter(torch, inputs, order)
                used = "ring"
            elif session0.scatter_uses_chain(probe, mode=mode):
                rows = references.chain_reduce_scatter(torch, inputs, order)
                used = "chain"
            else:
                summed = _sum(torch, inputs)
                rows = list(summed.reshape(world, -1).unbind(0))
                used = "pieces"
            return [row.reshape((shape[0] // world, *shape[1:])) for row in rows], used

    saved = [session._tuning for session in group.sessions]
    results = []
    try:
        for session in group.sessions:
            session._tuning = table
            session._tuning_counts.clear()
            session._tuning_unusable.clear()
        inputs = [_inputs(torch, world, shape, bf16, 2600 + index) for index, (_, shape, _) in enumerate(steps)]
        expected = [reference(kind, shape, settings, inputs[index]) for index, (kind, shape, settings)
                    in enumerate(steps)]

        def operation(rank: int, session):
            outputs = []
            for (kind, shape, settings), step_inputs in zip(steps, inputs):
                x = step_inputs[rank].to(session.device)
                pair = []
                for _ in range(2):
                    with Forced(session, settings):
                        if kind == "reduce":
                            pair.append(session.all_reduce_large(x))
                        elif kind == "scatter":
                            pair.append(session.reduce_scatter(x))
                        else:
                            pair.append(session.all_gather_large(x, dim=0))
                outputs.append(pair)
            return outputs

        outputs = group.each(operation)
        torch.cuda.synchronize()
        wrong = []
        for index, (kind, shape, settings) in enumerate(steps):
            want, used = expected[index]
            label = f"{kind} {shape} {'forced ' + str(settings) if settings else 'table'} ({used})"
            for rank in range(world):
                first, second = (out.cpu() for out in outputs[rank][index])
                if not _same_bits(torch, first, want[rank]):
                    wrong.append(f"{label} rank {rank}")
                if not _same_bits(torch, first, second):
                    wrong.append(f"{label} rank {rank} repeat")
        used = [expected[index][1] for index, (_, _, settings) in enumerate(steps) if settings is None]
        if used != ["chain", "ring", "pieces", "ring", "chain", "ring", "chain"]:
            wrong.insert(0, f"the table's ops ran {used}")
        stats = session0.stats()["tuning"]
        counted = sum(count for label, count in stats["decisions"].items() if "/eager/" in label)
        table_ops = sum(1 for _, _, settings in steps if settings is None)
        if counted != 2 * table_ops or stats["unusable"]:
            wrong.append(f"decisions {stats['decisions']}, unusable {stats['unusable']}")
        results.append((f"tuning table beside forced schedules: {len(steps)} ops, each twice, eager", not wrong,
                        "; ".join(wrong[:4])))

        def captured(session, x):
            table_out = session.all_reduce_large(x)
            with Forced(session, {"large_schedule": "chain"}):
                forced_out = session.all_reduce_large(x)
            return torch.cat([table_out, forced_out])

        def combine(step_inputs):
            table_ref = reference("reduce", (m_ring // 2,), None, step_inputs, mode="graph")[0][0]
            forced_ref = reference("reduce", (m_ring // 2,), {"large_schedule": "chain"}, step_inputs, mode="graph")[0][0]
            return torch.cat([table_ref, forced_ref])

        results.append(_graph_check(group, "tuning table beside forced schedules: one graph, ring by the table and "
                                    "chain forced", (m_ring // 2,), bf16, captured, combine, (2690, 2691)))
    finally:
        for session, tuning in zip(group.sessions, saved):
            session._tuning = tuning
    return results


def _gather_chain_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """all_gather_large as chain all-gathers: shard sizes, piece sizes, shapes, capture."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.link_available:
        return [("chain all-gather", True, f"not available here (schedule {session0.gather_schedule})")]
    results: list[tuple[str, bool, str]] = []
    default_piece = session0.link_chunk_bytes
    # (schedule, piece bytes, shard shape, dim, dtype): one-pack shards, a short last piece, many pieces,
    # rows along dimension 0 (the mHC gather), a leading dimension of size 1, float32 bytes.
    cases = [("chain", None, (8,), 0, torch.bfloat16), ("chain", None, (24,), 0, torch.bfloat16),
             ("chain", 4096, (2056,), 0, torch.bfloat16), ("chain", 16384, (64, 1040), 0, torch.bfloat16),
             ("auto", None, (512, 1024), 0, torch.bfloat16), ("auto", None, (1, 300, 2048), 1, torch.bfloat16),
             ("chain", None, (96, 4096 + 8), 0, torch.bfloat16), ("chain", 65536, (4100, 36), 0, torch.float32),
             ("pieces", None, (512, 1024), 0, torch.bfloat16)]
    seed = 1200
    for schedule, piece, shape, dim, dtype in cases:
        seed += 1
        _set_all(group, gather_schedule=schedule, link_chunk_bytes=piece or default_piece)
        inputs = _inputs(torch, group.world, shape, dtype, seed)
        chained = session0.gather_uses_chain(inputs[0], dim)
        nbytes = inputs[0].numel() * inputs[0].element_size()
        label = (f"chain all-gather {list(shape)} {dtype} dim {dim}, {nbytes} B per rank, {schedule}, pieces of "
                 f"{piece or default_piece} B ({'chain' if chained else 'tiles'})")
        results.append(_collective(group, label, inputs, lambda session, x, dim=dim: session.all_gather_large(x, dim=dim),
                                   torch.cat(inputs, dim=dim)))
    _set_all(group, gather_schedule="auto", link_chunk_bytes=default_piece)
    results.append(_graph_check(group, "graph chain all-gather", (512, 2048), torch.bfloat16,
                                lambda session, x: session.all_gather_large(x, dim=0),
                                lambda inputs: torch.cat(inputs, dim=0), (1231, 1232)))
    native = session0.stats()
    results.append(("chain all-gather counters", native.get("link_ops", 0) > 0,
                    f"order {list(session0.chain_order)}, {native.get('link_ops')} link ops, "
                    f"{native.get('link_items_posted')} items, {native.get('link_credits_sent')} credits on rank 0"))
    return results


def _per_rank(group: EmulatedGroup, name: str, inputs, call: Callable, expected: Sequence,
              compare: Sequence | None = None) -> tuple[str, bool, str]:
    """``call`` on every rank, each output against its own reference; ``compare`` (one tensor per rank)
    only counts the elements that differ from it."""
    torch = group.torch

    def operation(rank: int, session):
        return call(session, inputs[rank].to(session.device))

    try:
        outputs = [output.cpu() for output in group.each(operation)]
    except Exception as error:  # noqa: BLE001 - reported as a failed check
        return name, False, f"{type(error).__name__}: {error}"
    wrong = [rank for rank, output in enumerate(outputs) if not _same_bits(torch, output, expected[rank])]
    detail = f"ranks {wrong} differ" if wrong else ""
    if compare is not None and not wrong:
        view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[outputs[0].element_size()]
        differing = sum(int((output.contiguous().view(view) != other.contiguous().view(view)).sum().item())
                        for output, other in zip(outputs, compare))
        total = sum(output.numel() for output in outputs)
        detail = f"{differing} of {total} elements differ from the rank-ordered sum rounded once"
    return name, not wrong, detail


def _repeat_check(group: EmulatedGroup, name: str, inputs, call: Callable, seeds=(11, 23, 37, 41)):
    """``call`` on the same inputs once per seed of the fabric's scheduler, which executes posted writes
    in a random order across queue pairs (each queue pair in order): every run must give the same bits
    on every rank, whatever order the neighbors' partials arrive in."""
    torch = group.torch
    first = None
    try:
        for seed in seeds:
            group.fabric.lib.fv_progress(0, seed)

            def operation(rank: int, session):
                return call(session, inputs[rank].to(session.device))

            outputs = [output.cpu() for output in group.each(operation)]
            if first is None:
                first = outputs
                continue
            wrong = [rank for rank in range(group.world) if not _same_bits(torch, outputs[rank], first[rank])]
            if wrong:
                return name, False, f"fabric seed {seed}: ranks {wrong} differ from the first run"
    except Exception as error:  # noqa: BLE001
        return name, False, f"{type(error).__name__}: {error}"
    return name, True, f"{len(seeds)} runs"


def _scatter_chain_checks(group: EmulatedGroup, types) -> list[tuple[str, bool, str]]:
    """reduce_scatter as chain reduce-scatters: chunk and piece sizes, short last pieces, strided chunks,
    every dtype, capture; each rank's rows against ``references.chain_reduce_scatter``."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.link_available:
        return [("chain reduce-scatter", True, f"not available here (schedule {session0.scatter_schedule})")]
    world = group.world
    order = session0.chain_order
    results: list[tuple[str, bool, str]] = []
    default_piece = session0.link_chunk_bytes
    saved_schedule = session0.scatter_schedule
    bf16, fp16, fp32 = torch.bfloat16, torch.float16, torch.float32
    # Only the prepared dtypes run: a launcher compiled here would load its module while other
    # ranks' kernels spin.
    prepared = [dtype for dtype in types if all(("link-scatter", dtype) in session._launchers
                                                for session in group.sessions)]
    if not prepared:
        return [("chain reduce-scatter", False, "prepare(..., scatter=True) compiled no chain reduce-scatter")]
    first = bf16 if bf16 in prepared else prepared[0]
    # (schedule, piece bytes, input shape, dtype, chunk bytes, source stride bytes)
    cases = [("chain", None, (world * 8,), first, None, None),
             ("chain", None, (world * 24,), first, None, None),
             ("chain", 4096, (world * 2056,), first, None, None),
             ("chain", 16384, (world * 64, 1040), first, None, None),
             ("auto", None, (world * 256, 1024), first, None, None),
             ("chain", None, (world * 96, 4104), first, None, None),
             ("chain", 4096, ((world - 1) * 6144 + 4096 + 512,), first, 8192, 12288),
             ("pieces", None, (world * 256, 1024), first, None, None)]
    for dtype, piece, rows, columns in ((fp32, 65536, 4100, 36), (fp16, 32768, 512, 64)):
        if dtype in prepared:
            cases.append(("chain", piece, (world * rows, columns), dtype, None, None))
    seed = 1300
    for schedule, piece, shape, dtype, chunk_bytes, stride_bytes in cases:
        seed += 1
        _set_all(group, scatter_schedule=schedule, link_chunk_bytes=piece or default_piece)
        inputs = _inputs(torch, world, shape, dtype, seed)
        item = inputs[0].element_size()
        chained = session0.scatter_uses_chain(inputs[0].to(session0.device), chunk_bytes=chunk_bytes,
                                              src_stride_bytes=stride_bytes)
        chunk_elements = None if chunk_bytes is None else chunk_bytes // item
        stride_elements = None if stride_bytes is None else stride_bytes // item
        if chained:
            expected = references.chain_reduce_scatter(torch, inputs, order, chunk_elements=chunk_elements,
                                                       stride_elements=stride_elements)
        else:
            total = _sum(torch, [tensor.reshape(-1) for tensor in inputs])
            size = chunk_elements or total.numel() // world
            step = stride_elements or size
            expected = [total[rank * step:rank * step + size] for rank in range(world)]
        summed = _sum(torch, [tensor.reshape(-1) for tensor in inputs])
        size = chunk_elements or summed.numel() // world
        step = stride_elements or size
        oneshot_rows = [summed[rank * step:rank * step + size] for rank in range(world)]
        if chunk_bytes is None and shape[0] % world == 0:
            expected = [rows.reshape((shape[0] // world, *shape[1:])) for rows in expected]
            oneshot_rows = [rows.reshape((shape[0] // world, *shape[1:])) for rows in oneshot_rows]
        nbytes = inputs[0].numel() * item
        label = (f"chain reduce-scatter {list(shape)} {dtype}, {nbytes} B per rank"
                 + (f", chunks of {chunk_bytes} B every {stride_bytes} B" if chunk_bytes else "")
                 + f", {schedule}, pieces of {piece or default_piece} B ({'chain' if chained else 'scatter ops'})")
        results.append(_per_rank(
            group, label, inputs,
            lambda session, x, c=chunk_bytes, t=stride_bytes: session.reduce_scatter(x, chunk_bytes=c, src_stride_bytes=t),
            expected, oneshot_rows if chained else None))
    _set_all(group, scatter_schedule="chain", link_chunk_bytes=default_piece)
    inputs = _inputs(torch, world, (world * 512, 1024), first, 1399)
    results.append(_repeat_check(group, "chain reduce-scatter on the same inputs under 4 fabric orders", inputs,
                                 lambda session, x: session.reduce_scatter(x)))
    results.append(_graph_check(group, "graph chain reduce-scatter", (world * 256, 2048), first,
                                lambda session, x: session.reduce_scatter(x),
                                lambda inputs: [rows.reshape(256, 2048) for rows in
                                                references.chain_reduce_scatter(torch, inputs, order)],
                                (1331, 1332), per_rank=True))
    _set_all(group, scatter_schedule=saved_schedule, link_chunk_bytes=default_piece)
    native = session0.stats()
    results.append(("chain reduce-scatter counters", native.get("link_ops", 0) > 0,
                    f"order {list(order)}, {native.get('link_ops')} link ops, "
                    f"{native.get('link_items_posted')} items, {native.get('link_credits_sent')} credits on rank 0"))
    return results


def _ring_checks(group: EmulatedGroup, types) -> list[tuple[str, bool, str]]:
    """The ring collectives over the closed chain: all_reduce_large, reduce_scatter and all_gather_large
    with the ring schedule, against ``references.ring_all_reduce`` / ``ring_reduce_scatter`` and the
    concatenation; pieces of several sizes, sizes that leave a remainder, capture."""
    torch = group.torch
    session0 = group.sessions[0]
    if not session0.ring_available:
        return [("ring collectives", True, "not available here (the ring cannot close)")]
    world = group.world
    order = session0.chain_order
    bf16, fp16, fp32 = torch.bfloat16, torch.float16, torch.float32
    for rank, session in enumerate(group.sessions):
        with torch.cuda.stream(group.streams[rank]):
            for dtype in (bf16, fp16, fp32):
                session._ring_launcher("reduce", dtype, capturing=False)
                session._ring_launcher("scatter", dtype, capturing=False)
            session._ring_launcher("gather", None, capturing=False)
    # A launcher compiled here loads its module on its first launch, which would wait for the
    # other ranks' spinning kernels: load every module before the first ring op.
    group.load_modules(types)
    results: list[tuple[str, bool, str]] = []
    default_piece = session0.link_chunk_bytes
    saved = (session0.large_schedule, session0.gather_schedule, session0.scatter_schedule)
    _set_all(group, large_schedule="ring", gather_schedule="ring", scatter_schedule="ring")
    seed = 1500
    # All-reduce: a multiple of W packs, a remainder below W packs, a sub-pack tail, short last pieces.
    for dtype in types:
        item = torch.empty((), dtype=dtype).element_size()
        for nbytes, piece in ((world * 16, None), (world * 16 * 7 + 48 + 6, 4096), ((1 << 20) + world * 16, 65536),
                              ((3 << 20) + 32, None)):
            seed += 1
            _set_all(group, link_chunk_bytes=piece or default_piece)
            elements = nbytes // item
            inputs = _inputs(torch, world, (elements,), dtype, seed)
            plan = session0.large_reduce_plan(elements * item)
            expected = references.large_all_reduce(torch, inputs, plan, order)
            label = (f"ring all-reduce {dtype} {elements * item} B, pieces of {piece or default_piece} B "
                     f"({'ring op' if any(p.ring for p in plan) else 'pieces'})")
            results.append(_collective(group, label, inputs, lambda session, x: session.all_reduce_large(x),
                                       expected))
    _set_all(group, link_chunk_bytes=default_piece)
    # Reduce-scatter: contiguous rows, a strided chunk, every dtype.
    for shape, dtype, chunk_bytes, stride_bytes, piece in (((world * 256, 1024), bf16, None, None, None),
                                                           ((world * 64, 1040), fp16, None, None, 16384),
                                                           ((world * 4100, 36), fp32, None, None, 65536),
                                                           (((world - 1) * 6144 + 4096 + 512,), bf16, 8192, 12288,
                                                            4096)):
        seed += 1
        _set_all(group, link_chunk_bytes=piece or default_piece)
        inputs = _inputs(torch, world, shape, dtype, seed)
        item = inputs[0].element_size()
        chunk_elements = None if chunk_bytes is None else chunk_bytes // item
        stride_elements = None if stride_bytes is None else stride_bytes // item
        expected = references.ring_reduce_scatter(torch, inputs, order, chunk_elements=chunk_elements,
                                                  stride_elements=stride_elements)
        if chunk_bytes is None:
            expected = [rows.reshape((shape[0] // world, *shape[1:])) for rows in expected]
        label = (f"ring reduce-scatter {list(shape)} {dtype}, pieces of {piece or default_piece} B"
                 + (f", chunks of {chunk_bytes} B every {stride_bytes} B" if chunk_bytes else ""))
        results.append(_per_rank(
            group, label, inputs,
            lambda session, x, c=chunk_bytes, t=stride_bytes: session.reduce_scatter(x, chunk_bytes=c, src_stride_bytes=t),
            expected))
    _set_all(group, link_chunk_bytes=default_piece)
    # All-gather: rows along dimension 0, a leading dimension of size 1, float32 bytes.
    for shape, dim, dtype, piece in (((512, 1024), 0, bf16, None), ((1, 300, 2048), 1, bf16, 65536),
                                     ((4100, 36), 0, fp32, 4096), ((8,), 0, bf16, None)):
        seed += 1
        _set_all(group, link_chunk_bytes=piece or default_piece)
        inputs = _inputs(torch, world, shape, dtype, seed)
        label = f"ring all-gather {list(shape)} {dtype} dim {dim}, pieces of {piece or default_piece} B"
        results.append(_collective(group, label, inputs, lambda session, x, dim=dim: session.all_gather_large(x, dim=dim),
                                   torch.cat(inputs, dim=dim)))
    _set_all(group, link_chunk_bytes=default_piece)
    nbytes = (2 << 20) + world * 16
    plan = session0.large_reduce_plan(nbytes)
    results.append(_graph_check(group, "graph ring all-reduce", (nbytes // 2,), bf16,
                                lambda session, x: session.all_reduce_large(x),
                                lambda inputs: references.large_all_reduce(torch, inputs, plan, order), (1531, 1532)))
    results.append(_graph_check(group, "graph ring all-gather", (512, 2048), bf16,
                                lambda session, x: session.all_gather_large(x, dim=0),
                                lambda inputs: torch.cat(inputs, dim=0), (1533, 1534)))
    _set_all(group, large_schedule=saved[0], gather_schedule=saved[1], scatter_schedule=saved[2])
    stats = [session.stats() for session in group.sessions]
    windows = {rank: session.ring_window_bytes for rank, session in enumerate(group.sessions)
               if session.ring_window_bytes}
    chunks = [native.get("link_window_chunks_posted", 0) for native in stats]
    # Every rank ran link ops, and exactly the ranks whose ring lanes cross relays posted windowed chunks.
    healthy = (all(native.get("link_ops", 0) > 0 for native in stats)
               and all((count > 0) == (rank in windows) for rank, count in enumerate(chunks)))
    results.append(("ring counters", healthy,
                    f"order {list(order)}, windows {windows or 'none'} (rank: bytes), link ops "
                    f"{[native.get('link_ops') for native in stats]}, items "
                    f"{[native.get('link_items_posted') for native in stats]}, windowed chunks {chunks}"))
    return results


STARTUP_LAG_S = 3.0
SERVING_LAG_LIMIT_S = 0.5


def _lag_checks(group: EmulatedGroup) -> list[tuple[str, bool, str]]:
    """Rank 0 starts a collective late: the startup limit waits it out, the serving limit fails it."""
    torch = group.torch
    results = []
    for regime, lag in (("startup", STARTUP_LAG_S), ("serving", 4 * SERVING_LAG_LIMIT_S)):
        for session in group.sessions:
            session.enter_startup() if regime == "startup" else session.enter_serving()

        def operation(rank: int, session, lag=lag):
            x = torch.full((64,), float(rank + 1), dtype=torch.bfloat16, device=session.device)
            if rank == 0:
                time.sleep(lag)
            started = time.perf_counter()
            y = session.all_reduce(x)
            wait_stream(torch.cuda.current_stream())
            waited = time.perf_counter() - started
            try:
                session.check_health()
            except RuntimeError as error:
                return ("error", str(error), waited)
            return ("ok", y.cpu(), waited)

        try:
            outcomes = group.each(operation)
        except Exception as error:  # noqa: BLE001
            results.append((f"{regime} lag", False, f"{type(error).__name__}: {error}"))
            continue
        world = group.world
        expected = float(world * (world + 1) // 2)
        if regime == "startup":
            good = all(kind == "ok" and bool((value == expected).all()) for kind, value, _ in outcomes)
            slowest = max(waited for _, _, waited in outcomes)
            results.append((f"startup lag of {lag:g} s", good and slowest >= lag * 0.9,
                            f"peers waited up to {slowest:.2f} s and every sum is exact"
                            if good else f"outcomes {[o[:1] + o[2:] for o in outcomes]}"))
        else:
            peers = outcomes[1:]
            failed = [kind == "error" and f"wait limit {SERVING_LAG_LIMIT_S:g} s, serving regime" in value
                      for kind, value, _ in peers]
            waits = [waited for _, _, waited in peers]
            ok = all(failed) and max(waits) < lag
            results.append((f"serving limit of {SERVING_LAG_LIMIT_S:g} s against a {lag:g} s lag", ok,
                            f"peers poisoned after {min(waits):.2f}-{max(waits):.2f} s"
                            if ok else f"outcomes {[(o[0], str(o[1])[:120], round(o[2], 2)) for o in peers]}"))
    return results


class _Checks(list):
    """Check results, each also passed to ``report`` as it completes."""

    def __init__(self, report: Callable[[tuple[str, bool, str]], None] | None) -> None:
        super().__init__()
        self._report = report

    def append(self, check: tuple[str, bool, str]) -> None:
        super().append(check)
        if self._report is not None:
            self._report(check)


def run_checks(layout_text: str = "path:0-3", lanes: int = 2, *, library: str | os.PathLike | None = None,
               max_size: int = 256 << 10, max_gather_bytes: int = 64 << 10,
               dtypes: Sequence[str] = ("bfloat16", "float32"),
               report: Callable[[tuple[str, bool, str]], None] | None = None,
               event_trace: int = 0,
               path_latency: tuple[int, int, int, int] | None = None,
               column_gather_only: bool = False) -> list[tuple[str, bool, str]]:
    """Every collective of the session, eager and captured, against host references; with
    ``event_trace`` records, every session keeps an event trace (``SIRCL_EVENT_TRACE``, the traced
    chain kernel runs every chain op) and :func:`_trace_checks` checks one chain all-reduce's."""
    import torch

    if library is None:
        build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", Path.cwd() / ".build" / "sim"))
        library = native_build.build_shared_library(build)
    # Pieces of the capacity keep the large all-reduce at several ops; a short serving
    # limit lets the lag check below end in about a second. The checks run messages far below
    # the default minimums: ring schedules run ring ops at every size here (_ring_min_checks sets a
    # minimum of its own), auto runs chain ops from 2 MiB of every collective (the chain checks'
    # auto cases sit at 2 to 5 MiB), and _minimum_default_checks checks the defaults.
    environment = {"SIRCL_LARGE_PIECE_BYTES": str(max_size), "SIRCL_SERVING_WAIT_S": str(SERVING_LAG_LIMIT_S),
                   "SIRCL_RING_MIN_BYTES": "0", "SIRCL_CHAIN_MIN_BYTES": str(2 << 20)}
    if event_trace:
        environment["SIRCL_EVENT_TRACE"] = str(int(event_trace))
    group = EmulatedGroup(layout_text, lanes, max_size=max_size, max_gather_bytes=max_gather_bytes,
                          library=library, environment=environment, path_latency=path_latency)
    world = group.world
    checks = _Checks(report)
    try:
        types = [getattr(torch, name) for name in dtypes]
        started = time.perf_counter()
        # Compilation is not a collective; one rank after another keeps the compiler single-threaded.
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                # Every launcher a check uses is compiled here, so no rank compiles while its peers wait.
                session.prepare(tuple(types), padded_gather=True,
                                scatter=session.scatter_available or session.link_available, links=True)
        group.load_modules(types)
        checks.append(("prepare", True, f"{time.perf_counter() - started:.1f} s"))
        stats = group.sessions[0].stats()
        checks.append(("settings", True, f"algorithms {stats['algorithms_available']}, large pieces "
                       f"{stats['large_piece_bytes']}, gather pieces {stats['gather_piece_bytes']}, "
                       f"windows {stats['forward_windows']}"))
        if column_gather_only:
            from .column_gather_checks import column_gather_checks

            for check in column_gather_checks(group):
                checks.append(check)
            healthy = [not session.poisoned for session in group.sessions]
            checks.append(("health", all(healthy), "" if all(healthy) else f"poisoned ranks {healthy}"))
            return checks
        seed = 1
        for dtype in types:
            item = torch.empty((), dtype=dtype).element_size()
            for nbytes, algorithm in ((16, None), (4096, None), (131072, None), (131072 + 16, None),
                                      (max_size, None), (16, "twoshot"), (48, "twoshot"), (4096, "twoshot")):
                seed += 1
                inputs = _inputs(torch, world, (nbytes // item,), dtype, seed)
                label = f"all_reduce {dtype} {nbytes} B {algorithm or 'auto'}"
                checks.append(_collective(
                    group, label, inputs,
                    lambda session, x, algorithm=algorithm: session.all_reduce(x, algorithm=algorithm),
                    _sum(torch, inputs)))
            seed += 1
            elements = (3 * max_size + 6) // item
            inputs = _inputs(torch, world, (elements,), dtype, seed)
            checks.append(_collective(group, f"all_reduce_large {dtype} {elements * item} B", inputs,
                                      lambda session, x: session.all_reduce_large(x), _sum(torch, inputs)))
        bf16 = torch.bfloat16
        for shape, dim in (((4096,), 0), ((8, 96), -1), ((5, 7), -1), ((3, 40000), -1), ((200, 72), -1),
                           ((7, 333), -1), ((50000,), 0), ((4, 6, 10), 1), ((2, 3, 4096), 0)):
            seed += 1
            inputs = _inputs(torch, world, shape, bf16, seed)
            reference = torch.cat(inputs, dim=dim)
            nbytes = 2
            for extent in shape:
                nbytes *= extent
            if nbytes <= max_gather_bytes and dim in (0, -1, len(shape) - 1):
                checks.append(_collective(group, f"all_gather {list(shape)} dim {dim}", inputs,
                                          lambda session, x, dim=dim: session.all_gather(x, dim=dim), reference))
            checks.append(_collective(group, f"all_gather_large {list(shape)} dim {dim}", inputs,
                                      lambda session, x, dim=dim: session.all_gather_large(x, dim=dim), reference))
        for check in _chain_checks(group, types):
            checks.append(check)
        for check in _trace_checks(group):
            checks.append(check)
        for check in _gather_chain_checks(group):
            checks.append(check)
        for check in _scatter_chain_checks(group, types):
            checks.append(check)
        for check in _ring_checks(group, types):
            checks.append(check)
        for check in _late_link_checks(group):
            checks.append(check)
        for check in _mixed_piece_checks(group):
            checks.append(check)
        for check in _ring_min_checks(group):
            checks.append(check)
        for check in _minimum_default_checks(group):
            checks.append(check)
        for check in _stagger_checks(group):
            checks.append(check)
        for check in _tuning_checks(group):
            checks.append(check)
        for check in _tuning_forced_checks(group):
            checks.append(check)
        from .column_gather_checks import column_gather_checks

        for check in column_gather_checks(group):
            checks.append(check)
        checks.append(_graph_check(group, "graph all_reduce_large", ((2 * max_size + 32) // 2,), bf16,
                                   lambda session, x: session.all_reduce_large(x),
                                   lambda inputs: _sum(torch, inputs), (101, 102)))
        checks.append(_graph_check(group, "graph all_reduce two-shot", (max_size // 2,), bf16,
                                   lambda session, x: session.all_reduce(x),
                                   lambda inputs: _sum(torch, inputs), (103, 104)))
        checks.append(_graph_check(group, "graph all_gather_large", (6, 20000), bf16,
                                   lambda session, x: session.all_gather_large(x, dim=-1),
                                   lambda inputs: torch.cat(inputs, dim=-1), (105, 106)))
        checks.append(_graph_check(group, "graph all_gather_large padded", (9, 333), bf16,
                                   lambda session, x: session.all_gather_large(x, dim=-1),
                                   lambda inputs: torch.cat(inputs, dim=-1), (107,)))
        healthy = [not session.poisoned for session in group.sessions]
        checks.append(("health", all(healthy), "" if all(healthy) else f"poisoned ranks {healthy}"))
        native = group.sessions[0].stats()
        checks.append(("native counters", True,
                       f"ops {native.get('ops_posted')}, later phases {native.get('later_phases_posted')}, "
                       f"forward chunks {native.get('forward_chunks_posted')}, most unacknowledged "
                       f"{native.get('forward_max_unacked_bytes')} bytes, {native.get('poll_rate_per_s')} "
                       f"flag polls per second, wait limit {native.get('wait_limit_s')} s "
                       f"({native.get('wait_regime')})"))
        for check in _lag_checks(group):
            checks.append(check)
    finally:
        group.close()
    return checks


def _path_latency(text: str) -> tuple[int, int, int, int] | None:
    """``BASE_NS,RELAY_NS[,BYTES_PER_US[,ACK_DELAY_NS]]`` of ``--path-latency``, or None when empty."""
    if not text:
        return None
    values = [int(value, 0) for value in text.split(",")]
    if len(values) not in (2, 3, 4) or any(value < 0 for value in values):
        raise SystemExit("--path-latency takes BASE_NS,RELAY_NS[,BYTES_PER_US[,ACK_DELAY_NS]] of non-negative "
                         "integers")
    values += [0] * (4 - len(values))
    return values[0], values[1], values[2], values[3]


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default="path:0-3")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=256 << 10)
    parser.add_argument("--max-gather-bytes", type=int, default=64 << 10)
    parser.add_argument("--dtypes", default="bfloat16,float32")
    parser.add_argument("--path-latency", default="",
                        help="BASE_NS,RELAY_NS[,BYTES_PER_US[,ACK_DELAY_NS]]: every write and its completion "
                             "take BASE_NS plus RELAY_NS per relay of its lane, each queue pair sends at "
                             "BYTES_PER_US (default unlimited), and completions return ACK_DELAY_NS later")
    parser.add_argument("--event-trace", type=int, default=0,
                        help="records of every session's event trace (SIRCL_EVENT_TRACE); 0: no trace")
    parser.add_argument("--column-gather-only", action="store_true",
                        help="after preparing, run only the column-gather checks (the vLLM adapter's staged "
                             "dimension-0 link all-gather against all_gather_large along the column dimension)")
    args = parser.parse_args(argv)
    def show(check: tuple[str, bool, str]) -> None:
        name, ok, detail = check
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    checks = run_checks(args.layout, args.lanes, max_size=args.max_size, max_gather_bytes=args.max_gather_bytes,
                        dtypes=tuple(args.dtypes.split(",")), report=show, event_trace=args.event_trace,
                        path_latency=_path_latency(args.path_latency), column_gather_only=args.column_gather_only)
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"{len(checks)} checks, {failed} failed")
    sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
