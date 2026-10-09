"""GPU checks of glm_dcp_decode_comm on one GPU: its kernels against the image's, and its packed all-to-all on
emulated DCP groups of SIRCL sessions.

``python tests/gpu_checks.py [--parts kernels,dcp,register] [--layouts path:0-3,path:0-1] [--rows 1,2,3,8,16,31]
[--graph-rows 1,8,31] [--steps 3] [--check-timeout 300] [--setup-timeout 900]`` prints, as it goes, a ``START``
line before every check, ``STEP`` lines for its cases, and one ``PASS`` or ``FAIL`` line per check with its
duration; it ends with ``N checks, M failed`` and exits 1 when any check failed. Every comparison is word for word.

Bounds. A check that runs longer than ``--check-timeout`` seconds (a DCP group's setup and warm-up:
``--setup-timeout``) prints ``Timeout`` and every thread's Python stack (``faulthandler``), then the process
exits 1: a slow compile shows its compiler frames, a wait its waiting frame. Before any check the run requires
the compile caches it uses (``TRITON_CACHE_DIR``, ``CUTE_DSL_CACHE_DIR``, ``SIRCL_TEST_BUILD_DIR``,
``CUDA_CACHE_PATH``, where set) to be writable.

- ``kernels``: ``kernels.rope_cat`` against the image's ``fused_q`` (BF16 query path) followed by
  ``torch.cat`` with ``ql_nope``, for row counts 1 to 128, ``ql_nope`` laid out as the image's ``torch.bmm``
  leaves it and ``q_pe`` as a view of the ``q_b_proj`` output, with float32 and BF16 rotary caches; and
  ``kernels.wire_combine`` against the image's ``_dcp_a2a_unpack_combine`` of the records the image's
  ``_dcp_a2a_pack_send`` writes, for groups of 2, 4 and 8 ranks, 4 to 16 heads per rank, natural and base-2
  LSEs, output words of every bit pattern and LSEs with infinities, NaN and empty rows. The image's
  functions are compiled from the image's source files without importing vLLM (``dcp_decode_image_kernels.py``).
- ``dcp``: one emulated DCP group per layout (``path:0-3``: four ranks, the ends two relays apart;
  ``path:0-1``: two neighbours), every rank a thread on the one GPU over SIRCL's in-memory verbs stand-in
  (``sparkring_sircl.testing.dcp_gpu_checks.EmulatedWorld``). Eagerly per row count: the fused combine
  (``runtime.packed_all_to_all`` and ``kernels.wire_combine``) against the image's combine on SIRCL (vLLM's
  pack, the session's ``all_to_all``, vLLM's unpack-combine), and every received chunk against the wire format
  of the sender's tensors. In CUDA graphs: one graph per rank that forks onto a side stream as the overlap does,
  runs a query all-gather into a caller's buffer, the packed all-to-all, vLLM's pack and the session's
  all-to-all there, joins, and combines both on the capturing stream; each graph replays ``--steps`` times
  with new inputs. Then every session's health.
- ``register``: in the serving image, ``register()`` in a fresh process per flag set (every pin verified
  against the installed vLLM and b12x and the SIRCL tree on the path), the attention methods' edits and, with
  the overlap, the communicator's wrapped collectives. Each process imports vLLM's attention module, so it runs
  with lazy CUDA module loading and its own bound. Skipped when the installed vLLM is not the pinned build.

Requirements: CUDA, Triton, the CuTe DSL, a GCC-compatible compiler (SIRCL's simulator build) and the pinned SIRCL
(``sparkring_sircl``, with its ``testing`` package) on the path; ``SIRCL_TEST_BUILD_DIR`` names the
simulator build directory.
"""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

PROJECT = Path(__file__).resolve().parents[1]
for _path in (PROJECT, Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

Check = tuple[str, bool, str]
HEADS = 8                  # query heads per rank at tensor-parallel size 8 (64 heads)
OP_LIMIT = 1 << 20         # the DCP group's scatter op limit the fused combine fits (SIRCL's default on a path of 4)
CACHE_VARIABLES = ("TRITON_CACHE_DIR", "CUTE_DSL_CACHE_DIR", "SIRCL_TEST_BUILD_DIR", "CUDA_CACHE_PATH")


def _utc() -> str:
    return time.strftime("%H:%M:%S", time.gmtime())


def step(text: str) -> None:
    print(f"STEP {text}", flush=True)


class Watch:
    """A ``START`` line and a bound per check (``faulthandler``: every thread's stack, then exit 1)."""

    def __init__(self, limit_s: float) -> None:
        self.limit_s = float(limit_s)
        self.elapsed = 0.0

    @contextlib.contextmanager
    def check(self, name: str, limit_s: float | None = None):
        limit = self.limit_s if limit_s is None else float(limit_s)
        print(f"START {name} at {_utc()} (bound {limit:.0f} s)", flush=True)
        started = time.perf_counter()
        faulthandler.dump_traceback_later(limit, exit=True, file=sys.stderr)
        try:
            yield
        finally:
            faulthandler.cancel_dump_traceback_later()
            self.elapsed = time.perf_counter() - started


def _with_time(check: Check, seconds: float) -> Check:
    name, ok, detail = check
    return name, ok, (f"{detail}; " if detail else "") + f"{seconds:.1f} s"


def cache_check() -> Check:
    """Every compile cache directory the run uses exists or can be made, and takes a file."""
    used, problems = [], []
    for name in CACHE_VARIABLES:
        value = os.environ.get(name)
        if not value:
            continue
        path = Path(value)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / f".write-probe-{os.getpid()}"
            probe.write_bytes(b"probe")
            probe.unlink()
            used.append(f"{name}={value}")
        except OSError as error:
            problems.append(f"{name}={value}: {type(error).__name__}: {error}")
    unset = [name for name in CACHE_VARIABLES if not os.environ.get(name)]
    detail = "; ".join(problems) if problems else ("writable: " + ", ".join(used) if used else "none set")
    if unset:
        detail += f"; unset (defaults): {', '.join(unset)}"
    return "caches", not problems, detail


def _versions() -> str:
    found = []
    for module, label in (("torch", "torch"), ("triton", "Triton"), ("cutlass", "CuTe DSL")):
        try:
            found.append(f"{label} {getattr(__import__(module), '__version__', '?')}")
        except Exception as error:  # noqa: BLE001 - report what is missing
            found.append(f"{label} unavailable ({type(error).__name__})")
    return ", ".join(found)


def _words(torch, tensor):
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32}[tensor.element_size()]
    return tensor.detach().contiguous().view(view)


def _differing(torch, got, want) -> int:
    if tuple(got.shape) != tuple(want.shape) or got.dtype != want.dtype:
        return -1
    return int((_words(torch, got) != _words(torch, want)).sum())


def _random_words(torch, shape, generator):
    return torch.randint(-32768, 32768, tuple(shape), dtype=torch.int16, generator=generator).view(torch.bfloat16)


def _lse(torch, rows: int, total: int, generator, salt: int):
    lse = torch.randn((rows, total), generator=generator) * 4
    lse[0, salt % total] = float("-inf")
    lse[rows - 1, (salt + 1) % total] = float("inf")
    lse[0, (salt + 2) % total] = float("nan")
    lse[rows - 1, 0] = float("-inf")
    return lse


# -- kernels against the image's -------------------------------------------------------------------------------


def rope_check(torch, kernels, image, rows_list: Sequence[int], cache_dtype) -> Check:
    device = torch.device("cuda", 0)
    differing, cases = 0, 0
    for rows in rows_list:
        for scale in (1.0, 512.0):
            started = time.perf_counter()
            g = torch.Generator().manual_seed(rows * 7 + int(scale))
            tokens = rows + 3                     # fused_q runs on every token, the query on the rows
            q = (torch.randn((tokens, HEADS, 192), generator=g) * scale).to(torch.bfloat16).to(device)
            q_nope, q_pe = q.split([128, 64], dim=-1)
            w_uk_t = (torch.randn((HEADS, 128, 512), generator=g) * 0.05).to(torch.bfloat16).to(device)
            ql_nope = torch.bmm(q_nope.transpose(0, 1), w_uk_t).transpose(0, 1)
            cache = torch.randn((8192, 64), generator=g).to(cache_dtype).to(device)
            positions = torch.randint(0, 8192, (tokens,), generator=g, dtype=torch.int64).to(device)
            q_scale = torch.ones(1, dtype=torch.float32, device=device)
            _, _, mqa = image.fused_q(positions, q_pe, cache, None, None, ql_nope, q_scale, None, 0.0, 0.0,
                                      has_indexer=False, index_rope_interleave=False, quantize_mqa=False)
            want = torch.cat((ql_nope[:rows], mqa[:rows]), dim=-1)
            got = torch.empty((rows, HEADS, 576), dtype=torch.bfloat16, device=device)
            kernels.rope_cat(positions, q_pe, cache, ql_nope, got, rows)
            torch.cuda.synchronize()
            result = _differing(torch, got, want)
            differing += abs(result) if result >= 0 else 1
            cases += 1
            step(f"rope_cat {str(cache_dtype)[6:]} cache, {rows} rows, scale {scale:g}: "
                 f"{result} words differ ({time.perf_counter() - started:.1f} s)")
    return (f"rope_cat against the image's fused_q and torch.cat, {str(cache_dtype)[6:]} cache", differing == 0,
            f"{cases} cases (rows {list(rows_list)}, two scales), {differing} words differ; image kernels from "
            f"{image.source}")


COMBINE_CASES = ((2, 8, (1, 3, 31)), (4, 8, (1, 2, 3, 8, 31)), (8, 8, (1, 3, 16)), (4, 4, (1, 5)), (4, 16, (1, 7)))


def combine_check(torch, kernels, image, reference, world: int, heads: int, rows_list: Sequence[int],
                  base_e: bool) -> Check:
    device = torch.device("cuda", 0)
    differing, cases = 0, 0
    for rows in rows_list:
        started = time.perf_counter()
        g = torch.Generator().manual_seed(world * 1000 + heads * 10 + rows)
        outs = [_random_words(torch, (rows, world * heads, 512), g).to(device) for _ in range(world)]
        lses = [_lse(torch, rows, world * heads, g, s).to(device) for s in range(world)]
        sends = []
        for s in range(world):
            send = torch.empty((world, rows, heads, 514), dtype=torch.bfloat16, device=device)
            image.pack(outs[s], lses[s], send, world, heads, 512, 2)
            sends.append(send)
        worst = 0
        for rank in range(world):
            records = torch.stack([sends[s][rank] for s in range(world)])
            want = image.unpack(records, 512, 2, False, base_e)
            recv = torch.zeros((world, reference.L.wire_chunk_bytes(rows, heads)), dtype=torch.uint8, device=device)
            for s in range(world):
                if s != rank:
                    recv[s] = reference.wire_chunk(outs[s], lses[s], world, rank)
            got = kernels.wire_combine(recv, outs[rank], lses[rank], world, rank, heads, base_e)
            torch.cuda.synchronize()
            result = _differing(torch, got, want)
            differing += abs(result) if result >= 0 else 1
            worst = max(worst, abs(result) if result >= 0 else 1)
            cases += 1
        step(f"wire_combine {world} ranks, {heads} heads, {rows} rows, {'natural' if base_e else 'base-2'}: "
             f"{worst} words differ on the worst rank ({time.perf_counter() - started:.1f} s)")
    return (f"wire_combine against the image's unpack-combine, {world} ranks, {heads} heads, "
            f"{'natural' if base_e else 'base-2'} LSE", differing == 0,
            f"{cases} rank combines (rows {list(rows_list)}), {differing} words differ")


def kernel_checks(torch, watch: Watch, rope_rows: Sequence[int], report: Callable[[Check], None]) -> list[Check]:
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        report(check)

    try:
        with watch.check("load the image's kernels"):
            from glm_dcp_decode_comm import kernels, reference

            import dcp_decode_image_kernels as image_kernels

            image = image_kernels.load()
        add(_with_time(("load the image's kernels", True, image.source), watch.elapsed))
        for cache_dtype in (torch.float32, torch.bfloat16):
            name = f"rope_cat, {str(cache_dtype)[6:]} cache"
            with watch.check(name):
                check = rope_check(torch, kernels, image, rope_rows, cache_dtype)
            add(_with_time(check, watch.elapsed))
        for world, heads, rows_list in COMBINE_CASES:
            for base_e in (True, False):
                name = f"wire_combine, {world} ranks, {heads} heads, {'natural' if base_e else 'base-2'} LSE"
                with watch.check(name):
                    check = combine_check(torch, kernels, image, reference, world, heads, rows_list, base_e)
                add(_with_time(check, watch.elapsed))
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        add(("kernels", False, f"{type(error).__name__}: {error}"))
    return checks


# -- the packed all-to-all on emulated DCP groups --------------------------------------------------------------


class Group:
    """One emulated DCP group of SIRCL sessions, prepared for the packed all-to-all."""

    def __init__(self, torch, layout_text: str, library) -> None:
        from sparkring_sircl import routes
        from sparkring_sircl.oneshot import _scatter_ops
        from sparkring_sircl.testing import kernel_gpu_checks
        from sparkring_sircl.testing.dcp_gpu_checks import EmulatedWorld

        self.torch = torch
        self.layout_text = layout_text
        self.size = routes.Layout.parse(layout_text).world
        step(f"{layout_text}: building {self.size} sessions")
        self.world = EmulatedWorld([(layout_text, tuple(range(self.size)))], lanes=2, max_size=2 << 20,
                                   max_gather_bytes=2 << 20, library=library)
        self.group = self.world.groups[0]
        self.sessions = self.group.sessions
        bf16 = (torch.bfloat16,)
        scatter = all(kernel_gpu_checks.scatter_registered(session) for session in self.sessions)
        if not scatter:
            for session in self.sessions:
                session.scatter_available = bool(session.multi_phase)
        for rank, session in enumerate(self.sessions):
            step(f"{layout_text}: preparing rank {rank}")
            with torch.cuda.stream(self.group.streams[rank]):
                session.prepare(bf16, padded_gather=True, scatter=scatter)
                if not scatter:
                    _scatter_ops.prepare(session, bf16)
        step(f"{layout_text}: loading the session kernels' modules")
        self.group.load_modules(bf16)
        kernel_gpu_checks.load_modules(self.group, ("bfloat16",))
        self.notes = list(self.world.notes)

    def dcp(self, rank: int):
        """What ``runtime.scatter_pack_fits`` reads of a ``SirclDcpCollectives``."""
        return SimpleNamespace(_runtime=self.sessions[rank], _scatter_op_limit=lambda: OP_LIMIT)

    def warm(self, kernels, runtime, image, rows_list: Sequence[int]) -> None:
        """Compile and load the packed kernel and the Triton kernels of every rank before any rank runs.

        The packed kernel is launched once per rank with the poison word set, so it returns at once and only
        its module loads (as SIRCL's ``load_modules`` does for the session's kernels); the Triton kernels are
        compiled per rank and row count with zeros.
        """
        from sparkring_sircl.testing.gpu_emulation import wait_stream

        torch = self.torch
        device = self.sessions[0].device
        for rank, session in enumerate(self.sessions):
            stream = self.group.streams[rank]
            with torch.cuda.stream(stream):
                poison = session._counter_layout.poison_word
                session._counters[poison] = 1
                for rows in rows_list:
                    started = time.perf_counter()
                    out, lse, recv = self.buffers(rows)
                    runtime.packed_all_to_all(session, out, lse, recv, rows, HEADS)
                    kernels.wire_combine(recv, out, lse, self.size, rank, HEADS, True)
                    send = torch.zeros((self.size, rows, HEADS, 514), dtype=torch.bfloat16, device=device)
                    image.pack(out, lse, send, self.size, HEADS, 512, 2)
                    image.unpack(send, 512, 2, False, True)
                    wait_stream(stream)
                    step(f"{self.layout_text}: warmed rank {rank} at {rows} rows ({time.perf_counter() - started:.1f} s)")
                session._counters[poison] = 0
                wait_stream(stream)

    def buffers(self, rows: int):
        torch = self.torch
        device = self.sessions[0].device
        from glm_dcp_decode_comm import layout as L

        out = torch.zeros((rows, self.size * HEADS, L.V_DIM), dtype=torch.bfloat16, device=device)
        lse = torch.zeros((rows, self.size * HEADS), dtype=torch.float32, device=device)
        recv = torch.zeros((self.size, L.wire_chunk_bytes(rows, HEADS)), dtype=torch.uint8, device=device)
        return out, lse, recv

    def staged(self, rows: int, seed: int):
        """Every rank's inputs on the device, copied from this thread before any rank runs; and the host copies."""
        from sparkring_sircl.testing.gpu_emulation import wait_stream

        torch = self.torch
        g = torch.Generator().manual_seed(seed)
        host = [(_random_words(torch, (rows, self.size * HEADS, 512), g), _lse(torch, rows, self.size * HEADS, g, s),
                 _random_words(torch, (rows, HEADS * 576), g)) for s in range(self.size)]
        device = []
        for rank, item in enumerate(host):
            stream = self.group.streams[rank]
            with torch.cuda.stream(stream):
                device.append(tuple(t.to(self.sessions[0].device) for t in item))
                wait_stream(stream)
        return host, device

    def close(self) -> None:
        self.world.close()


def eager_check(torch, group: Group, kernels, runtime, image, reference, rows: int, seed: int) -> Check:
    size = group.size
    label = f"{group.layout_text} eager, {rows} row(s): packed all-to-all and wire_combine against the image's combine"
    fits = [runtime.scatter_pack_fits(group.dcp(rank), rows, HEADS) for rank in range(size)]
    if not all(fits):
        return label, False, f"one scatter op does not hold {rows} rows of {HEADS} heads ({fits})"
    host, device = group.staged(rows, seed)
    results: dict[int, Any] = {}

    def body(rank: int):
        session = group.sessions[rank]
        out, lse, _ = device[rank]
        _, _, recv = group.buffers(rows)
        launched = runtime.packed_all_to_all(session, out, lse, recv, rows, HEADS)
        fused = kernels.wire_combine(recv, out, lse, size, rank, HEADS, True)
        send = torch.empty((size, rows, HEADS, 514), dtype=torch.bfloat16, device=out.device)
        image.pack(out, lse, send, size, HEADS, 512, 2)
        received = torch.empty_like(send)
        session.all_to_all(send.view(-1), received.view(-1))
        stock = image.unpack(received, 512, 2, False, True)
        results[rank] = (launched, fused.clone(), stock.clone(), recv.clone())

    try:
        group.world.run(body)
    except Exception as error:  # noqa: BLE001
        return label, False, f"{type(error).__name__}: {error}"
    problems = []
    for rank in range(size):
        launched, fused, stock, recv = results[rank]
        if not launched:
            problems.append(f"rank {rank} declined")
        differing = _differing(torch, fused, stock)
        if differing:
            problems.append(f"rank {rank}: {differing} combined words differ")
        for source in range(size):
            if source != rank:
                want = reference.wire_chunk(host[source][0], host[source][1], size, rank)
                if not torch.equal(recv[source].cpu(), want):
                    problems.append(f"rank {rank}: the chunk from rank {source} differs from the wire format")
    return label, not problems, "; ".join(problems) or "every rank matches"


def graph_check(torch, group: Group, kernels, runtime, image, rows: int, steps: int, seed: int) -> Check:
    from sparkring_sircl.testing.gpu_emulation import wait_stream

    size = group.size
    label = (f"{group.layout_text} CUDA graph, {rows} row(s), {steps} replays: query gather, packed all-to-all and the "
             "image's all-to-all on a side stream, both combines joined")
    state: dict[int, dict] = {}
    try:
        for rank, session in enumerate(group.sessions):
            stream = group.group.streams[rank]
            with torch.cuda.stream(stream):
                out, lse, recv = group.buffers(rows)
                entry = {"out": out, "lse": lse, "recv": recv,
                         "query": torch.zeros((rows, HEADS * 576), dtype=torch.bfloat16, device=out.device),
                         "gathered": torch.zeros((rows, size * HEADS * 576), dtype=torch.bfloat16, device=out.device),
                         "send": torch.zeros((size, rows, HEADS, 514), dtype=torch.bfloat16, device=out.device),
                         "received": torch.zeros((size, rows, HEADS, 514), dtype=torch.bfloat16, device=out.device),
                         "side": torch.cuda.Stream()}
                wait_stream(stream)
            state[rank] = entry
        # Captures run one rank after another from this thread: a capture in one thread forbids other threads'
        # synchronizing calls.
        for rank, session in enumerate(group.sessions):
            entry, stream = state[rank], group.group.streams[rank]
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.stream(stream):
                with session.capture(), torch.cuda.graph(graph, stream=stream):
                    main = torch.cuda.current_stream()
                    side = entry["side"]
                    side.wait_stream(main)
                    with torch.cuda.stream(side):
                        session.all_gather(entry["query"], dim=-1, out=entry["gathered"])
                        entry["launched"] = runtime.packed_all_to_all(session, entry["out"], entry["lse"],
                                                                      entry["recv"], rows, HEADS)
                        image.pack(entry["out"], entry["lse"], entry["send"], size, HEADS, 512, 2)
                        session.all_to_all(entry["send"].view(-1), entry["received"].view(-1))
                    main.wait_stream(side)
                    entry["fused"] = kernels.wire_combine(entry["recv"], entry["out"], entry["lse"], size, rank,
                                                          HEADS, True)
                    entry["stock"] = image.unpack(entry["received"], 512, 2, False, True)
                wait_stream(stream)
            entry["graph"] = graph
            step(f"{group.layout_text}: captured rank {rank} at {rows} rows")
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        return label, False, f"capture: {type(error).__name__}: {error}"
    inputs = [group.staged(rows, seed + index) for index in range(steps)]
    outputs: dict[int, list] = {rank: [] for rank in range(size)}

    def body(rank: int):
        entry = state[rank]
        for index in range(steps):
            out, lse, query = inputs[index][1][rank]
            entry["out"].copy_(out)
            entry["lse"].copy_(lse)
            entry["query"].copy_(query)
            entry["graph"].replay()
            outputs[rank].append((entry["fused"].clone(), entry["stock"].clone(), entry["gathered"].clone()))

    try:
        group.world.run(body)
    except Exception as error:  # noqa: BLE001
        return label, False, f"replay: {type(error).__name__}: {error}"
    problems = [f"rank {rank} declined inside the capture" for rank in range(size) if not state[rank]["launched"]]
    for rank in range(size):
        for index in range(steps):
            fused, stock, gathered = outputs[rank][index]
            differing = _differing(torch, fused, stock)
            if differing:
                problems.append(f"rank {rank} step {index}: {differing} combined words differ")
            want = torch.cat([inputs[index][0][s][2] for s in range(size)], dim=-1)
            if not torch.equal(gathered.cpu().view(torch.int16), want.view(torch.int16)):
                problems.append(f"rank {rank} step {index}: the gathered query differs")
    return label, not problems, "; ".join(problems) or "every replay of every rank matches"


def dcp_checks(torch, watch: Watch, setup_s: float, layouts: Sequence[str], rows_list: Sequence[int],
               graph_rows: Sequence[int], steps: int, report: Callable[[Check], None]) -> list[Check]:
    checks: list[Check] = []

    def add(check: Check) -> None:
        checks.append(check)
        report(check)

    try:
        with watch.check("load the image's kernels and build SIRCL's simulator library", setup_s):
            from sparkring_sircl.testing import native_build

            from glm_dcp_decode_comm import kernels, reference, runtime

            import dcp_decode_image_kernels as image_kernels

            image = image_kernels.load()
            build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", Path.cwd() / ".build" / "sim"))
            library = native_build.build_shared_library(build)
        add(_with_time(("load and build", True, f"image kernels from {image.source}; simulator {library.name}"),
                       watch.elapsed))
    except Exception as error:  # noqa: BLE001
        traceback.print_exc()
        add(("load and build", False, f"{type(error).__name__}: {error}"))
        return checks
    for index, layout_text in enumerate(layouts):
        try:
            with watch.check(f"{layout_text} setup and warm-up", setup_s):
                group = Group(torch, layout_text, library)
                try:
                    group.warm(kernels, runtime, image, sorted(set(rows_list) | set(graph_rows)))
                except BaseException:
                    group.close()
                    raise
        except Exception as error:  # noqa: BLE001
            traceback.print_exc()
            add((f"{layout_text} setup", False, f"{type(error).__name__}: {error}"))
            continue
        try:
            add(_with_time((f"{layout_text} prepare", True, f"{group.size} ranks; emulation: "
                            f"{'; '.join(group.notes) or 'no caps'}"), watch.elapsed))
            for rows in rows_list:
                with watch.check(f"{layout_text} eager, {rows} row(s)"):
                    check = eager_check(torch, group, kernels, runtime, image, reference, rows,
                                        9000 + 100 * index + rows)
                add(_with_time(check, watch.elapsed))
            for rows in graph_rows:
                with watch.check(f"{layout_text} CUDA graph, {rows} row(s)"):
                    check = graph_check(torch, group, kernels, runtime, image, rows, steps, 7000 + 100 * index + rows)
                add(_with_time(check, watch.elapsed))
            poisoned = [rank for rank, session in enumerate(group.sessions) if session.poisoned]
            add((f"{layout_text} health", not poisoned, f"poisoned ranks {poisoned}" if poisoned else ""))
        finally:
            group.close()
    return checks


# -- registration in the serving image -------------------------------------------------------------------------

_REGISTER = r"""
import faulthandler, json, os, sys, time
# Every thread's stack, then exit, shortly before the parent's bound: a slow or stuck import shows where it is.
faulthandler.dump_traceback_later(float(os.environ["GLM_DCP_DECODE_CHECK_BOUND"]), exit=True, file=sys.stderr)
import glm_dcp_decode_comm as plugin
started = time.perf_counter()
plugin.register()
registered = time.perf_counter() - started
import vllm.models.deepseek_v32.attention as attention
imported = time.perf_counter() - started - registered
cls = attention.DeepseekV32Attention
sparse = cls.__dict__["_sparse_indexer_and_attn"]
inner = getattr(sparse, "__wrapped__", sparse)
found = {"forward": getattr(cls.forward, plugin.MARKER, None), "sparse": getattr(inner, plugin.MARKER, None),
         "wrapped_by_vllm": inner is not sparse, "wrapped": [],
         "seconds": {"register": round(registered, 1), "import vLLM's attention module": round(imported, 1)}}
if plugin.settings_from_env()["overlap"]:
    import sparkring_sircl.vllm.communicator as communicator
    from glm_dcp_decode_comm import runtime
    found["wrapped"] = [name for name in runtime._WRAPPED_METHODS
                        if getattr(getattr(communicator.SirclCudaCommunicator, name), runtime.WRAP_MARKER, False)]
print(json.dumps(found))
"""

_ALL_FIVE = ("GLM_DCP_DECODE_QUERY_PACK", "GLM_DCP_DECODE_OVERLAP", "GLM_DCP_DECODE_WK_OVERLAP",
             "GLM_DCP_DECODE_SELECTION_REUSE", "GLM_DCP_DECODE_A2A_FUSED")
REGISTER_CASES = (
    ("every flag off", {}, "1"),
    ("query pack, breakable CUDA graphs", {"GLM_DCP_DECODE_QUERY_PACK": "1"}, "1"),
    ("all five and audit, breakable CUDA graphs", {**{name: "1" for name in _ALL_FIVE}, "GLM_DCP_DECODE_AUDIT": "1"},
     "1"),
    ("all five, plain CUDA graphs", {name: "1" for name in _ALL_FIVE}, "0"),
)


def register_checks(watch: Watch, report: Callable[[Check], None]) -> list[Check]:
    import importlib.util

    import glm_dcp_decode_comm as plugin
    from glm_dcp_decode_comm import runtime

    spec = importlib.util.find_spec("vllm")
    attention = (Path(list(spec.submodule_search_locations)[0]) / plugin.ATTENTION_PATH
                 if spec is not None and spec.submodule_search_locations else None)
    if attention is None or not attention.exists() or plugin.digest(attention) != plugin.ATTENTION_SHA256:
        where = "vLLM is not installed here" if spec is None else f"the vLLM at {attention.parents[2]} is not the pinned build"
        check = ("register", True, f"SKIP: {where}; the registration checks run in the serving image")
        report(check)
        return [check]
    checks = []
    base = {key: value for key, value in os.environ.items() if not key.startswith("GLM_DCP_DECODE_")}
    base["PYTHONPATH"] = os.pathsep.join([str(PROJECT), base.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    base["CUDA_MODULE_LOADING"] = "LAZY"       # importing vLLM's model loads CUDA libraries; nothing is launched
    base["GLM_DCP_DECODE_CHECK_BOUND"] = str(max(10.0, watch.limit_s - 15.0))
    for label, flags, breakable in REGISTER_CASES:
        env = {**base, **flags, "VLLM_USE_BREAKABLE_CUDAGRAPH": breakable}
        env["GLM_DCP_DECODE_CHECK_BOUND"] = base["GLM_DCP_DECODE_CHECK_BOUND"]
        print(f"START register: {label} at {_utc()} (bound {watch.limit_s:.0f} s)", flush=True)
        started = time.perf_counter()
        try:
            done = subprocess.run([sys.executable, "-X", "faulthandler", "-c", _REGISTER], capture_output=True,
                                  text=True, env=env, timeout=watch.limit_s)
        except subprocess.TimeoutExpired as expired:
            tail = (expired.stderr or b"")[-1500:]
            tail = tail.decode("utf-8", "replace") if isinstance(tail, bytes) else tail
            check = (f"register: {label}", False, f"over the {watch.limit_s:.0f} s bound; stderr tail: {tail!r}")
        else:
            if done.returncode != 0:
                print(done.stderr[-4000:], file=sys.stderr, flush=True)
                check = (f"register: {label}", False, done.stderr.strip().splitlines()[-1][:300] if done.stderr else
                         f"exit {done.returncode}")
            else:
                found = json.loads(done.stdout.strip().splitlines()[-1])
                seconds = found.pop("seconds")
                on = bool(flags)
                want = {"forward": plugin.PATCHES[0].qualname if on else None,
                        "sparse": plugin.PATCHES[1].qualname if on else None,
                        "wrapped_by_vllm": breakable == "1",
                        "wrapped": list(runtime._WRAPPED_METHODS) if flags.get("GLM_DCP_DECODE_OVERLAP") else []}
                check = (f"register: {label}", found == want, json.dumps(found) if found != want else
                         f"{'edits installed' if on else 'nothing patched'}"
                         f"{', ' + str(len(found['wrapped'])) + ' collectives wrapped' if found['wrapped'] else ''}"
                         f"; vLLM's breakable-graph wrapper {'kept' if found['wrapped_by_vllm'] else 'absent'}"
                         f"; {', '.join(f'{k} {v} s' for k, v in seconds.items())}")
        check = _with_time(check, time.perf_counter() - started)
        checks.append(check)
        report(check)
    return checks


# -- main -----------------------------------------------------------------------------------------------------


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in text.split(",") if item.strip())


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parts", default="kernels,dcp,register")
    parser.add_argument("--layouts", default="path:0-3,path:0-1")
    parser.add_argument("--rows", default="1,2,3,8,16,31")
    parser.add_argument("--rope-rows", default="1,2,3,4,8,16,31,64,128")
    parser.add_argument("--graph-rows", default="1,8,31")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--check-timeout", type=float, default=300.0, help="seconds per check (setup: --setup-timeout)")
    parser.add_argument("--setup-timeout", type=float, default=900.0, help="seconds for a DCP group's setup and warm-up")
    args = parser.parse_args(argv)
    parts = {part for part in args.parts.split(",") if part}
    watch = Watch(args.check_timeout)
    checks: list[Check] = []

    def show(check: Check) -> None:
        name, ok, detail = check
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    check = cache_check()
    show(check)
    checks.append(check)
    if not check[1]:
        print("1 checks, 1 failed", flush=True)
        return 1
    import torch

    if parts & {"kernels", "dcp"}:
        show(("device", True, f"{torch.cuda.get_device_name(0)}; {_versions()}; CUDA_MODULE_LOADING="
                              f"{os.environ.get('CUDA_MODULE_LOADING')}"))
    if "kernels" in parts:
        checks += kernel_checks(torch, watch, _ints(args.rope_rows), show)
    if "dcp" in parts:
        checks += dcp_checks(torch, watch, args.setup_timeout, [item for item in args.layouts.split(",") if item],
                             _ints(args.rows), _ints(args.graph_rows), args.steps, show)
    if "register" in parts:
        checks += register_checks(watch, show)
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"{len(checks)} checks, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
