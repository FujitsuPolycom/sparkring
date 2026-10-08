"""GPU checks of a TP8, DCP4 deployment's exchanges on one emulated world of eight ranks (test support).

The world's ranks run as threads of one process on one CUDA device over the
simulator build of the native layer; one in-memory verbs stand-in carries
every session of the world. Each rank holds the two sessions of a deployment
with tensor parallelism over the ring of eight and decode context parallel
(DCP) groups of four consecutive ranks (vLLM's ``--tensor-parallel-size 8
--decode-context-parallel-size 4``):

- a tensor-parallel session over the whole ring (``ring:8``), exchanging over
  all eight ranks;
- a DCP session over its group's path (``path:0-3`` for ranks 0-3,
  ``path:4-7`` for ranks 4-7), whose end ranks reach each other through the
  two middle ranks.

The checks compare every rank's output bit for bit with a host reference:

- the DCP all-to-all of vLLM's ``a2a`` combine
  (``vllm/v1/attention/ops/dcp.py``, ``dcp_a2a_lse_reduce``): a
  ``[W, rows, heads, 514]`` BF16 buffer, ``heads`` query heads per rank after
  the combine, each with 512 latent output values and its FP32 log-sum-exp in
  two BF16 slots. Chunk ``p`` goes to rank ``p`` and rank ``s``'s chunk lands
  at position ``s``, in both DCP groups at once, at decode row counts and at a
  prefill chunk of 8,192 rows. The payloads are random 16-bit words, so every
  bit pattern, NaN and infinity encodings included, must arrive unchanged;
- a decode step's exchanges on every rank in order, with fresh inputs per
  step, eagerly and replayed from one CUDA graph per rank: the DCP all-gather
  of the query (``[rows, heads, 576]`` BF16 along the head dimension), the
  DCP all-gather of the indexer's top-k candidates (``[rows, 2048, 2]``
  float32 along the candidate dimension), the DCP all-to-all, and the
  tensor-parallel all-reduce of ``[rows, hidden]`` BF16 on the ring session
  (the rank-ordered float32 sum of the eight inputs, rounded once);
- the health of every session afterwards.

A DCP session whose class does not offer the scatter collectives gets them
marked available (multi-phase posting), and the all-to-all then runs through
the host functions of ``oneshot/_scatter_ops.py``.

Host copies: every input reaches the device from the main thread before the
ranks start, and a rank thread only launches work and copies device to
device. In one process, a rank thread blocked in a copy from pageable host
memory can hold a driver lock that another rank's CUDA call waits for while
holding the interpreter lock; the verbs stand-in's delivery thread, a Python
thread, then cannot run, the peers' writes stop and every waiting kernel
times out with its flags written afterwards. Separate processes, as in
serving, have no such coupling.

``python -m sparkring_sircl.testing.dcp_gpu_checks [--decode-rows 1,8,64,128]
[--prefill-rows 8192] [--groups 2] [--no-tp]`` prints one line per check.
Requirements as for :mod:`.kernel_gpu_checks`; the 8,192-row prefill chunk
needs about 5 GB of device memory and 5 GB of host memory.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .. import routes as routes_mod
from . import kernel_gpu_checks, native_build
from .fabric import FakeFabric
from .gpu_emulation import EmulatedGroup, ThreadDist, ThreadGroup, device_name, emulated_device_roles, wait_stream

Check = tuple[str, bool, str]

LATENT = 512            # MLA latent width (kv_lora_rank): the attention output values per head
LSE_SLOTS = 2           # the FP32 log-sum-exp of a head packed into two BF16 slots
QUERY_WIDTH = 576       # MLA query per head: 512 latent values and 64 RoPE values
INDEXER_TOP_K = 2048    # DSA indexer candidates per row, each a (score, position) float32 pair
HEADS = 8               # query heads per rank at tensor-parallel size 8 (64 heads)
HIDDEN = 6144           # hidden width of the tensor-parallel all-reduce
DECODE_ROWS = (1, 2, 4, 8, 16, 32, 64, 128)
PREFILL_ROWS = (8192,)
STEP_PARTS = ("query", "indexer", "a2a", "tp")   # a decode step's exchanges, in order
_PART_NAMES = {"query": "query gather", "indexer": "indexer gather", "a2a": "all-to-all", "tp": "TP all-reduce"}


class SubGroup:
    """The sessions of one group of an emulated world, on the world ranks' streams.

    ``each``, ``load_modules`` and the thread runner are those of
    :class:`.gpu_emulation.EmulatedGroup`; they read ``world``, ``sessions``,
    ``streams`` and ``torch``.
    """

    _threads = EmulatedGroup._threads
    each = EmulatedGroup.each
    load_modules = EmulatedGroup.load_modules

    def __init__(self, torch: Any, runtime: Any, layout_text: str, members: Sequence[int], world_streams: list,
                 lanes: int, max_size: int, max_gather_bytes: int) -> None:
        self.torch = torch
        self.layout_text = layout_text
        self.layout = routes_mod.Layout.parse(layout_text)
        self.members = tuple(members)
        if self.layout.world != len(self.members):
            raise ValueError(f"layout {layout_text} has {self.layout.world} ranks, members {self.members}")
        self.world = len(self.members)
        self.streams = [world_streams[member] for member in self.members]
        self.sessions: list[Any] = [None] * self.world
        derived = routes_mod.derive_routes(self.layout, lanes)
        exchange = ThreadGroup(self.world)

        def construct(rank: int) -> None:
            exchange.bind(rank)
            node = self.members[rank]
            peer_routes = {peer: tuple(device_name(node, device) for device in devices)
                           for peer, devices in derived.route_map(rank).items()}
            self.sessions[rank] = runtime.AllReduce(
                exchange_group=exchange, device=torch.device("cuda", 0), max_size=max_size,
                max_gather_bytes=max_gather_bytes, peer_routes=peer_routes, layout=layout_text, gid_index=3,
                lane_check_ms=10000,
            )

        saved = runtime.dist
        runtime.dist = ThreadDist
        try:
            self._threads(construct)
        finally:
            runtime.dist = saved

    def rank_of(self, node: int) -> int:
        return self.members.index(node)


class EmulatedWorld:
    """Groups of one emulated world, every session on one in-memory verbs stand-in.

    ``groups`` lists ``(layout text, world ranks)``; a world rank may belong to
    several groups (one session each), and the devices of world rank ``n`` are
    named ``n<n>.<device>`` whatever group uses them.
    """

    def __init__(self, groups: Sequence[tuple[str, Sequence[int]]], *, lanes: int = 2, max_size: int,
                 max_gather_bytes: int, library: str | os.PathLike) -> None:
        import torch

        from ..oneshot import _proxy, runtime

        self.torch = torch
        self.size = max(member for _, members in groups for member in members) + 1
        os.environ["SIRCL_NATIVE_LIBRARY"] = str(library)
        self.fabric = FakeFabric(_proxy.load(str(library)))
        self.fabric.reset()
        self.fabric.lib.fv_set_ideal(1)
        for node in range(self.size):
            for index, role in enumerate(routes_mod.ROLES):
                self.fabric.add_device(device_name(node, role.device), node, role.port, int(role.secondary),
                                       FakeFabric.gid(node, index))
        self.fabric.start()
        self._roles = emulated_device_roles(self.size)
        self._roles.__enter__()
        self.streams = [torch.cuda.Stream() for _ in range(self.size)]
        self.groups: list[SubGroup] = []
        try:
            for layout_text, members in groups:
                self.groups.append(SubGroup(torch, runtime, layout_text, members, self.streams, lanes, max_size,
                                            max_gather_bytes))
        except BaseException:
            self.close()
            raise

    def run(self, body: Callable[[int], Any], nodes: Sequence[int] | None = None,
            timeout: float = 900.0) -> list[Any]:
        """``body(node)`` on every listed world rank at once, each on its own stream."""
        torch = self.torch
        nodes = tuple(range(self.size)) if nodes is None else tuple(nodes)
        results: list[Any] = [None] * self.size
        errors: list[BaseException | None] = [None] * self.size

        def run(node: int) -> None:
            try:
                torch.cuda.set_device(0)
                with torch.cuda.stream(self.streams[node]):
                    results[node] = body(node)
                    wait_stream(self.streams[node])
            except BaseException as error:  # noqa: BLE001 - re-raised below with the rank
                errors[node] = error
                traceback.print_exc()

        threads = [threading.Thread(target=run, args=(node,), daemon=True) for node in nodes]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout)
        if any(thread.is_alive() for thread in threads):
            raise TimeoutError("an emulated rank did not finish")
        for node, error in enumerate(errors):
            if error is not None:
                raise RuntimeError(f"rank {node}: {type(error).__name__}: {error}") from error
        return results

    def close(self) -> None:
        for group in reversed(self.groups):
            with contextlib.suppress(Exception):
                group._threads(lambda rank, group=group: group.sessions[rank].close()
                               if group.sessions[rank] else None)
        self.fabric.stop()
        self._roles.__exit__(None, None, None)


def a2a_shape(world: int, rows: int, heads: int) -> tuple[int, int, int, int]:
    """Shape of the ``a2a`` combine's send and receive buffers (BF16)."""
    return (world, rows, heads, LATENT + LSE_SLOTS)


def _random_words(torch, shape: Sequence[int], seed: int):
    """Random 16-bit words of ``shape`` (every bit pattern possible), as int16 on the host."""
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(-32768, 32768, tuple(shape), dtype=torch.int16, generator=generator)


def _randn(torch, shape: Sequence[int], dtype, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator).to(dtype)


def _same(torch, got, want) -> bool:
    """Bitwise equality of a device result and a host reference."""
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[want.element_size()]
    got = got.detach().cpu().contiguous()
    return got.shape == want.shape and bool(torch.equal(got.view(view), want.contiguous().view(view)))


def a2a_reference(torch, words: Sequence[Any], members: Sequence[int], rank: int):
    """Rank ``rank``'s receive buffer: chunk ``rank`` of every group member's send buffer, in group order."""
    return torch.stack([words[node][rank] for node in members])


class DcpChecks:
    """The checks of one emulated world; every method returns ``(name, ok, detail)``."""

    def __init__(self, world: EmulatedWorld, tp: SubGroup | None, dcp: Sequence[SubGroup], *,
                 scatter_session: bool, heads: int = HEADS, hidden: int = HIDDEN) -> None:
        import torch

        self.torch = torch
        self.world = world
        self.tp = tp
        self.dcp = tuple(dcp)
        self.scatter_session = scatter_session
        self.heads = heads
        self.hidden = hidden
        self.seed = 4000
        self.group_of = {node: group for group in self.dcp for node in group.members}

    def _next_seed(self) -> int:
        self.seed += 1
        return self.seed

    def _all_to_all(self, session: Any, x: Any, out: Any):
        from ..oneshot import _scatter_ops

        if self.scatter_session:
            return session.all_to_all(x, out)
        return _scatter_ops.all_to_all(session, x, out)

    @property
    def nodes(self) -> tuple[int, ...]:
        return tuple(sorted(self.group_of))

    def _dcp_session(self, node: int) -> tuple[SubGroup, int, Any]:
        group = self.group_of[node]
        rank = group.rank_of(node)
        return group, rank, group.sessions[rank]

    def _to_device(self, node: int, tensor: Any) -> Any:
        """``tensor`` on the device, copied on ``node``'s stream from this thread, before any rank runs."""
        stream = self.world.streams[node]
        with self.torch.cuda.stream(stream):
            copy = tensor.to(self.dcp[0].sessions[0].device)
            wait_stream(stream)
        return copy

    # -- the a2a combine -------------------------------------------------------------

    def all_to_all(self, rows: int) -> Check:
        """The ``a2a`` combine's all-to-all at ``rows`` rows, every DCP group at once."""
        torch = self.torch
        width = self.dcp[0].world
        shape = a2a_shape(width, rows, self.heads)
        seed = self._next_seed()
        words = {node: _random_words(torch, shape, seed * 131 + node) for node in self.nodes}
        nbytes = words[self.nodes[0]].numel() * 2
        label = f"DCP all-to-all {list(shape)} BF16 ({nbytes} bytes per rank), {len(self.dcp)} group(s) at once"
        # Inputs reach the device before the ranks start (see the module notes on host copies).
        inputs = {node: self._to_device(node, words[node]).view(torch.bfloat16) for node in self.nodes}
        outputs = {node: torch.empty_like(inputs[node]) for node in self.nodes}

        def body(node: int):
            group, rank, session = self._dcp_session(node)
            self._all_to_all(session, inputs[node], outputs[node])

        started = time.perf_counter()
        try:
            self.world.run(body, self.nodes)
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"
        elapsed = time.perf_counter() - started
        wrong = [node for node in self.nodes
                 if not _same(torch, outputs[node].view(torch.int16),
                              a2a_reference(torch, words, self.group_of[node].members,
                                            self.group_of[node].rank_of(node)))]
        return label, not wrong, f"{elapsed:.2f} s; ranks differing from the reference: {wrong}"

    # -- a decode step -----------------------------------------------------------------

    def _step_inputs(self, rows: int, steps: int) -> dict:
        torch = self.torch
        width = self.dcp[0].world
        seed = self._next_seed()
        inputs: dict[str, list[dict[int, Any]]] = {"query": [], "indexer": [], "a2a": [], "tp": []}
        for step in range(steps):
            base = (seed * 1000 + step) * 64
            inputs["query"].append({node: _random_words(torch, (rows, self.heads, QUERY_WIDTH), base + node)
                                    for node in self.nodes})
            inputs["indexer"].append({node: _random_words(torch, (rows, INDEXER_TOP_K, 2, 2), base + 16 + node)
                                      .view(torch.float32).view(rows, INDEXER_TOP_K, 2)
                                      for node in self.nodes})
            inputs["a2a"].append({node: _random_words(torch, a2a_shape(width, rows, self.heads), base + 32 + node)
                                  for node in self.nodes})
            if self.tp is not None:
                inputs["tp"].append({node: _randn(torch, (rows, self.hidden), torch.bfloat16, base + 48 + node)
                                     for node in self.tp.members})
        return inputs

    def _step_references(self, inputs: dict, steps: int) -> dict:
        torch = self.torch
        references: dict[str, list[dict[int, Any]]] = {"query": [], "indexer": [], "a2a": [], "tp": []}
        for step in range(steps):
            query, indexer, words = inputs["query"][step], inputs["indexer"][step], inputs["a2a"][step]
            references["query"].append({node: torch.cat([query[m] for m in self.group_of[node].members], dim=1)
                                        for node in self.nodes})
            references["indexer"].append({node: torch.cat([indexer[m] for m in self.group_of[node].members],
                                                          dim=1) for node in self.nodes})
            references["a2a"].append({node: a2a_reference(torch, words, self.group_of[node].members,
                                                          self.group_of[node].rank_of(node))
                                      for node in self.nodes})
            if self.tp is not None:
                tp = inputs["tp"][step]
                total = tp[self.tp.members[0]].float().clone()
                for node in self.tp.members[1:]:
                    total += tp[node].float()
                references["tp"].append(total.to(torch.bfloat16))
        return references

    def decode_step(self, rows: int, mode: str = "eager", steps: int = 3,
                    parts: Sequence[str] = STEP_PARTS) -> Check:
        """The exchanges ``parts`` of ``steps`` decode steps (:data:`STEP_PARTS`; ``tp`` needs the TP session)."""
        torch = self.torch
        width = self.dcp[0].world
        parts = tuple(part for part in parts if part != "tp" or self.tp is not None)
        label = (f"decode step {mode}, {rows} row(s): {', '.join(_PART_NAMES[part] for part in parts)}, "
                 f"{steps} steps")
        inputs = self._step_inputs(rows, steps)
        references = self._step_references(inputs, steps)
        state: dict[int, dict[str, Any]] = {}

        def buffers(node: int) -> dict[str, Any]:
            group, rank, session = self._dcp_session(node)
            device = session.device
            entry = {
                "query": torch.empty((rows, self.heads, QUERY_WIDTH), dtype=torch.bfloat16, device=device),
                "query_out": torch.empty((rows, width * self.heads, QUERY_WIDTH), dtype=torch.bfloat16,
                                         device=device),
                "indexer": torch.empty((rows, INDEXER_TOP_K, 2), dtype=torch.float32, device=device),
                "indexer_out": torch.empty((rows, width * INDEXER_TOP_K, 2), dtype=torch.float32, device=device),
                "a2a": torch.empty(a2a_shape(width, rows, self.heads), dtype=torch.bfloat16, device=device),
                "a2a_out": torch.empty(a2a_shape(width, rows, self.heads), dtype=torch.bfloat16, device=device),
            }
            if self.tp is not None:
                entry["tp"] = torch.empty((rows, self.hidden), dtype=torch.bfloat16, device=device)
                entry["tp_out"] = torch.empty((rows, self.hidden), dtype=torch.bfloat16, device=device)
            return entry

        def exchanges(node: int, entry: dict[str, Any]) -> None:
            group, rank, session = self._dcp_session(node)
            if "query" in parts:
                session.all_gather_large(entry["query"], dim=1, out=entry["query_out"])
            if "indexer" in parts:
                session.all_gather_large(entry["indexer"], dim=1, out=entry["indexer_out"])
            if "a2a" in parts:
                self._all_to_all(session, entry["a2a"], entry["a2a_out"])
            if "tp" in parts:
                tp_session = self.tp.sessions[self.tp.rank_of(node)]
                tp_session.all_reduce(entry["tp"], out=entry["tp_out"])

        # Every step's inputs reach the device before the ranks start; a step then
        # copies device to device on the rank's stream (see the module notes).
        staged = {node: [{key: self._to_device(node, inputs[key][step][node])
                          for key in ("query", "indexer", "a2a", "tp") if key in parts}
                         for step in range(steps)] for node in self.nodes}

        def load(node: int, entry: dict[str, Any], step: int) -> None:
            for key, value in staged[node][step].items():
                entry[key].copy_(value.view(entry[key].dtype))

        try:
            for node in self.nodes:
                with torch.cuda.stream(self.world.streams[node]):
                    state[node] = buffers(node)
                    wait_stream(self.world.streams[node])
            if mode == "graph":
                # Captures run one rank after another from this thread: a capture in
                # one thread forbids other threads' synchronizing calls.
                for node in self.nodes:
                    stream = self.world.streams[node]
                    group, rank, session = self._dcp_session(node)
                    tp_session = self.tp.sessions[self.tp.rank_of(node)] if self.tp is not None else None
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(stream):
                        with session.capture(), (tp_session.capture() if tp_session else contextlib.nullcontext()):
                            with torch.cuda.graph(graph, stream=stream):
                                exchanges(node, state[node])
                        wait_stream(stream)
                    state[node]["graph"] = graph

            results: dict[int, list[dict[str, Any]]] = {node: [] for node in self.nodes}

            def body(node: int):
                entry = state[node]
                for step in range(steps):
                    load(node, entry, step)
                    if mode == "graph":
                        entry["graph"].replay()
                    else:
                        exchanges(node, entry)
                    results[node].append({key: entry[key].clone() for key in entry
                                          if key.endswith("_out")})
                return None

            started = time.perf_counter()
            self.world.run(body, self.nodes)
            elapsed = time.perf_counter() - started
        except Exception as error:  # noqa: BLE001
            return label, False, f"{type(error).__name__}: {error}"

        wrong: dict[str, list[int]] = {}
        for node in self.nodes:
            for step in range(steps):
                got = results[node][step]
                checks = {"query": (got["query_out"].view(torch.int16),
                                    references["query"][step][node].view(rows, -1, QUERY_WIDTH)),
                          "indexer": (got["indexer_out"], references["indexer"][step][node]),
                          "a2a": (got["a2a_out"].view(torch.int16), references["a2a"][step][node])}
                if self.tp is not None:
                    checks["tp"] = (got["tp_out"], references["tp"][step])
                for name, (have, want) in checks.items():
                    if name not in parts:
                        continue
                    if not _same(torch, have, want):
                        wrong.setdefault(name, []).append(node)
        detail = f"{elapsed:.2f} s for {steps} steps; " + (
            "every output matches" if not wrong else
            "; ".join(f"{name} differs on ranks {sorted(set(nodes))}" for name, nodes in wrong.items()))
        return label, not wrong, detail

    def health(self) -> Check:
        groups = ([self.tp] if self.tp is not None else []) + list(self.dcp)
        poisoned = [(group.layout_text, rank) for group in groups for rank, session in enumerate(group.sessions)
                    if session.poisoned]
        return "health", not poisoned, "" if not poisoned else f"poisoned sessions {poisoned}"


def run_checks(*, groups: int = 2, tp: bool = True, lanes: int = 2, max_size: int = 2 << 20,
               max_gather_bytes: int = 2 << 20, decode_rows: Sequence[int] = DECODE_ROWS,
               prefill_rows: Sequence[int] = PREFILL_ROWS, step_rows: Sequence[int] = (1, 32, 128),
               step_parts: Sequence[str] = STEP_PARTS, step_modes: Sequence[str] = ("eager", "graph"),
               heads: int = HEADS, hidden: int = HIDDEN, library: str | os.PathLike | None = None,
               report: Callable[[Check], None] | None = None) -> list[Check]:
    """Every check above on one emulated world: ``groups`` DCP groups of four, with or without the TP session."""
    import torch

    from ..oneshot import _scatter_ops

    if groups not in (1, 2):
        raise ValueError("one or two DCP groups of four")
    if tp and groups != 2:
        raise ValueError("the tensor-parallel session spans the eight ranks of two DCP groups")
    if library is None:
        build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", Path.cwd() / ".build" / "sim"))
        library = native_build.build_shared_library(build)
    layouts = [("path:0-3", (0, 1, 2, 3)), ("path:4-7", (4, 5, 6, 7))][:groups]
    if tp:
        layouts = [("ring:8", tuple(range(8)))] + layouts
    world = EmulatedWorld(layouts, lanes=lanes, max_size=max_size, max_gather_bytes=max_gather_bytes,
                          library=library)
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        if report is not None:
            report(check)

    try:
        tp_group = world.groups[0] if tp else None
        dcp_groups = world.groups[1:] if tp else world.groups
        dcp_sessions = [session for group in dcp_groups for session in group.sessions]
        scatter_session = all(kernel_gpu_checks.scatter_registered(session) for session in dcp_sessions)
        if not scatter_session:
            for session in dcp_sessions:
                session.scatter_available = bool(session.multi_phase)
        started = time.perf_counter()
        types = (torch.bfloat16,)
        # Compilation is not a collective; one rank after another keeps the compiler single-threaded.
        for group in world.groups:
            for rank, session in enumerate(group.sessions):
                with torch.cuda.stream(group.streams[rank]):
                    is_dcp = group is not tp_group
                    session.prepare(types, padded_gather=True, scatter=is_dcp and scatter_session)
                    if is_dcp and not scatter_session:
                        _scatter_ops.prepare(session, types)
        for group in world.groups:
            group.load_modules(types)
        for group in dcp_groups:
            kernel_gpu_checks.load_modules(group, ("bfloat16",))
        add(("prepare", True, f"{time.perf_counter() - started:.1f} s; {len(dcp_groups)} DCP group(s) "
             f"{[group.layout_text for group in dcp_groups]}"
             + (f" and the TP session {tp_group.layout_text}" if tp_group else "")
             + f"; all-to-all through {'the session methods' if scatter_session else 'the host functions'}"))
        suite = DcpChecks(world, tp_group, dcp_groups, scatter_session=scatter_session, heads=heads, hidden=hidden)
        for rows in (*decode_rows, *prefill_rows):
            add(suite.all_to_all(rows))
        for rows in step_rows:
            for mode in step_modes:
                add(suite.decode_step(rows, mode, parts=step_parts))
        add(suite.health())
        stats = dcp_groups[0].sessions[0].stats()
        add(("native counters", True, f"DCP rank 0: ops {stats.get('ops_posted')}, later phases "
             f"{stats.get('later_phases_posted')}, forward chunks {stats.get('forward_chunks_posted')}"))
    finally:
        world.close()
    return checks


def _rows(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--groups", type=int, default=2, help="DCP groups of four: 1 (path:0-3) or 2 (and path:4-7)")
    parser.add_argument("--no-tp", action="store_true", help="leave out the tensor-parallel session over ring:8")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--max-size", type=int, default=2 << 20)
    parser.add_argument("--max-gather-bytes", type=int, default=2 << 20)
    parser.add_argument("--decode-rows", default=",".join(str(rows) for rows in DECODE_ROWS))
    parser.add_argument("--prefill-rows", default=",".join(str(rows) for rows in PREFILL_ROWS))
    parser.add_argument("--step-rows", default="1,32,128", help="row counts of the decode-step checks")
    parser.add_argument("--step-parts", default=",".join(STEP_PARTS),
                        help=f"exchanges of a decode step, from {','.join(STEP_PARTS)}")
    parser.add_argument("--step-modes", default="eager,graph")
    parser.add_argument("--heads", type=int, default=HEADS)
    parser.add_argument("--hidden", type=int, default=HIDDEN)
    args = parser.parse_args(argv)

    def show(check: Check) -> None:
        name, ok, detail = check
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    checks = run_checks(groups=args.groups, tp=not args.no_tp, lanes=args.lanes, max_size=args.max_size,
                        max_gather_bytes=args.max_gather_bytes, decode_rows=_rows(args.decode_rows),
                        prefill_rows=_rows(args.prefill_rows), step_rows=_rows(args.step_rows),
                        step_parts=tuple(part for part in args.step_parts.split(",") if part),
                        step_modes=tuple(mode for mode in args.step_modes.split(",") if mode), heads=args.heads,
                        hidden=args.hidden, report=show)
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"{len(checks)} checks, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
