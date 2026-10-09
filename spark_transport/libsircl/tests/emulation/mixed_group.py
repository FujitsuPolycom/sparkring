#!/usr/bin/env python3
"""Mixed-implementation GPU emulation: libsircl's CUDA C++ kernels against SIRCL's CuTe DSL kernels.

Every rank of a group runs SIRCL's own session class as a thread of one process on one GPU over
SIRCL's in-process verbs stand-in (``sparkring_sircl.testing.gpu_emulation.EmulatedGroup``, used
unmodified from the SIRCL reference tree on ``PYTHONPATH``). Each session's launchers of the one-shot
and two-shot all-reduce, the all-gather (plain and tiled), the scatter ops (reduce-scatter and
all-to-all), the chain all-reduce and the link collectives (chain and ring all-gather and
reduce-scatter, ring all-reduce; whichever the session's schedules compiled) are then taken either from
SIRCL (CuTe DSL, compiled in process) or from libsircl's ahead-of-time kernel packs, launched from C
through ``libsirclkp_test.so`` with the session's own arena, counters and stream. The same group runs, back to back on the same sessions:

- ``dsl``: every rank on the DSL kernels (the SIRCL Python session's outputs);
- ``mixed``: even ranks on the CUDA C++ kernels, odd ranks on the DSL kernels;
- ``cxx``: every rank on the CUDA C++ kernels.

For every mode, collective, dtype and size, every rank's output must equal the host reference bit for
bit (the float32 sum in rank order rounded once for the reductions; for ``all_reduce_large`` the
reference of the session's plan, ``sparkring_sircl.references.large_all_reduce``, whose chain ops round
at every hop in chain order; the concatenation or exchange of bytes for the all-gather and all-to-all), and the CUDA C++ and mixed outputs must equal the DSL outputs
of the same inputs, eager and in CUDA graph replay. Reduce-scatters are checked against the reference
of the path the session takes: the rank-order sum, ``references.chain_reduce_scatter`` or
``references.ring_reduce_scatter``. Last, a serving-regime timeout: a C++ rank whose peer never
launches poisons its session within the wait limit and names the missing peer, lane and sequence in
the command ring.

``--suite sircl-links`` instead runs SIRCL's own link checks of its emulation harness
(``gpu_emulation``: chain all-reduce, chain all-gather and reduce-scatter, ring collectives, link ops
taken late by one rank's progress thread, pieces of their own, ring minimums and every ring stagger the
slots hold), in each mode, with SIRCL's harness settings (``SIRCL_RING_MIN_BYTES=0``,
``SIRCL_CHAIN_MIN_BYTES`` 2 MiB) and SIRCL's default schedules; each check compares with SIRCL's host
reference of the schedule the session chose.

Usage (inside WSL, under the GPU lock, ``CUDA_DEVICE_MAX_CONNECTIONS=32``):
  PYTHONPATH=<sircl reference>/spark_transport/sircl python mixed_group.py --kp-lib build/libsirclkp_test.so \
      --layout path:0-1 --lanes 1 [--suite cases|sircl-links]
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path

os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
# The cases suite runs pieces, tiles and scatter ops unless the environment selects a schedule
# (SIRCL_LARGE_SCHEDULE, SIRCL_GATHER_SCHEDULE, SIRCL_SCATTER_SCHEDULE: chain or ring); main() sets the
# defaults, which depend on the suite.
SCHEDULES = ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE")
SIRCL_DEFAULT_SCHEDULES = {"SIRCL_LARGE_SCHEDULE": "auto", "SIRCL_GATHER_SCHEDULE": "auto",
                           "SIRCL_SCATTER_SCHEDULE": "pieces"}

DTYPE_CODES = {"float32": 0, "float16": 1, "bfloat16": 2}
ONESHOT, TWOSHOT = 0, 1
# Link collective kinds of the link pack (SCCL_LINK_* and SCCL_RING_* in src/kernelpack.h), the roles of
# each kind's grid, and the most threads per block its entries are built for.
LINK_GATHER, LINK_SCATTER, RING_GATHER, RING_SCATTER, RING_REDUCE = range(5)
# The kp shim's kind of the two-pass ring all-reduce (LIBSIRCL_RING_REDUCE_PASSES=2 selects it).
RING_REDUCE_TWO_PASS = 5
LINK_ROLES = {LINK_GATHER: 4, LINK_SCATTER: 2, RING_GATHER: 2, RING_SCATTER: 2, RING_REDUCE: 3}
LINK_KIND_NAMES = {LINK_GATHER: "chain all-gather", LINK_SCATTER: "chain reduce-scatter",
                   RING_GATHER: "ring all-gather", RING_SCATTER: "ring reduce-scatter", RING_REDUCE: "ring all-reduce"}
LINK_MAX_THREADS = 512
LINK_MAX_WORLD = 8
LINK_KEYS = ("link-gather", "link-scatter", "link-ring")
u64, i64, i32, u32 = ctypes.c_uint64, ctypes.c_int64, ctypes.c_int32, ctypes.c_uint32


class ReduceArgs(ctypes.Structure):
    _fields_ = [("input", u64), ("output", u64), ("size_packs", i32), ("nbytes", i32), ("recv_base", u64),
                ("flag_base", u64), ("send_base", u64), ("ctrl_base", u64), ("slot_bytes", u64), ("epoch", u64),
                ("stage_counter", u64), ("phase_counter", u64), ("tail_counter", u64), ("poison", u64),
                ("arrival", u64), ("spin_limit", u32), ("rank", i32), ("lanes", i32), ("one_block", i32)]


class GatherArgs(ctypes.Structure):
    _fields_ = [("input", u64), ("output", u64), ("shard_packs", i32), ("nbytes", i32), ("tile_cols", i32),
                ("reserved", i32), ("in_row_stride", i64), ("out_row_stride", i64), ("out_src_stride", i64),
                ("recv_base", u64), ("flag_base", u64), ("send_base", u64), ("ctrl_base", u64), ("slot_bytes", u64),
                ("epoch", u64), ("stage_counter", u64), ("tail_counter", u64), ("poison", u64), ("arrival", u64),
                ("spin_limit", u32), ("rank", i32), ("lanes", i32), ("one_block", i32)]


class ScatterArgs(ctypes.Structure):
    _fields_ = [("input", u64), ("output", u64), ("size_packs", i32), ("nbytes", i32), ("chunk_packs", i32),
                ("reserved", i32), ("src_stride", i64), ("dst_stride", i64), ("recv_base", u64), ("flag_base", u64),
                ("send_base", u64), ("ctrl_base", u64), ("slot_bytes", u64), ("epoch", u64), ("stage_counter", u64),
                ("tail_counter", u64), ("poison", u64), ("spin_limit", u32), ("rank", i32), ("lanes", i32)]


class ChainArgs(ctypes.Structure):
    _fields_ = [("input", u64), ("output", u64), ("a_packs", i32), ("b_packs", i32), ("chunk_packs", i32),
                ("reserved", i32), ("chain_base", u64), ("counters", u64), ("ctrl_base", u64), ("poison", u64),
                ("spin_limit", u32), ("trace_capacity", u32), ("trace_base", u64), ("world", i32), ("index", i32),
                ("prev", i32), ("next", i32), ("rank", i32), ("lanes", i32), ("slots", i32),
                ("blocks_per_role", i32), ("slot_bytes", u64)]


class LinkArgs(ctypes.Structure):
    _fields_ = [("input", u64), ("output", u64), ("scratch", u64), ("chunk_packs", i32), ("stride_packs", i32),
                ("piece_packs", i32), ("stagger", i32), ("gather_stagger", i32), ("world", i32), ("index", i32),
                ("prev", i32), ("next", i32), ("rank", i32), ("lanes", i32), ("slots", i32),
                ("blocks_per_role", i32), ("reserved", i32), ("link_base", u64), ("counters", u64),
                ("piece_counters", u64), ("ctrl_base", u64), ("poison", u64), ("trace_base", u64),
                ("slot_bytes", u64), ("spin_limit", u32), ("trace_capacity", u32), ("order", i32 * LINK_MAX_WORLD)]


assert ctypes.sizeof(LinkArgs) == 176, "LinkArgs must match sccl_link_args"


class KernelPack:
    def __init__(self, path: str) -> None:
        lib = self.lib = ctypes.CDLL(path)
        lib.sirclkp_load.restype = ctypes.c_int
        lib.sirclkp_error.restype = ctypes.c_char_p
        lib.sirclkp_hash.restype = ctypes.c_char_p
        lib.sirclkp_launch.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                       ctypes.c_uint64, ctypes.POINTER(ReduceArgs)]
        lib.sirclkp_launch_allgather.argtypes = [ctypes.c_int, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint64,
                                                 ctypes.POINTER(GatherArgs)]
        lib.sirclkp_launch_scatter.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                               ctypes.c_uint64, ctypes.POINTER(ScatterArgs)]
        lib.sirclkp_launch_chain.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                             ctypes.c_uint64, ctypes.POINTER(ChainArgs)]
        lib.sirclkp_links_hash.restype = ctypes.c_char_p
        lib.sirclkp_launch_link.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_uint, ctypes.c_uint,
                                            ctypes.c_uint64, ctypes.POINTER(LinkArgs)]
        self.launches = 0
        self.link_launches = {kind: 0 for kind in LINK_ROLES}

    def load(self) -> None:
        if self.lib.sirclkp_load() != 0:
            raise RuntimeError(f"kernel pack load failed: {self.lib.sirclkp_error().decode()}")

    @property
    def hash(self) -> str:
        return self.lib.sirclkp_hash().decode()

    def _check(self, result: int) -> None:
        if result != 0:
            raise RuntimeError(f"kernel pack launch failed: {result} {self.lib.sirclkp_error().decode()}")
        self.launches += 1

    @staticmethod
    def _stream() -> int:
        import torch

        return torch.cuda.current_stream().cuda_stream

    def reduce_launcher(self, algorithm: int, dtype_name: str, session):
        """A launcher with the call signature of SIRCL's DSL all-reduce launcher of the same algorithm."""
        dtype, world, rank, lanes = DTYPE_CODES[dtype_name], session.world_size, session.rank, session.lane_count
        threads, one_block = session._threads, 1 if session._one_block_polls else 0

        def launch(args: ReduceArgs, grid: int) -> None:
            self._check(self.lib.sirclkp_launch(algorithm, dtype, world, grid, threads, self._stream(),
                                                ctypes.byref(args)))

        if algorithm == ONESHOT:
            def run(input_address, output_address, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base,
                    slot_bytes, epoch_address, stage_counter, tail_counter, poison_address, spin_limit, grid_blocks,
                    trace_ring=0, captured=0, arrival_address=0):
                launch(ReduceArgs(input_address, output_address, size_packs, nbytes, recv_base, flag_base, send_base,
                                  ctrl_base, slot_bytes, epoch_address, stage_counter, 0, tail_counter,
                                  poison_address, arrival_address, spin_limit, rank, lanes, one_block), grid_blocks)
        else:
            def run(input_address, output_address, size_packs, nbytes, recv_base, flag_base, send_base, ctrl_base,
                    slot_bytes, epoch_address, stage_counter, phase_counter, tail_counter, poison_address,
                    spin_limit, grid_blocks, trace_ring=0, arrival_address=0):
                launch(ReduceArgs(input_address, output_address, size_packs, nbytes, recv_base, flag_base, send_base,
                                  ctrl_base, slot_bytes, epoch_address, stage_counter, phase_counter, tail_counter,
                                  poison_address, arrival_address, spin_limit, rank, lanes, one_block), grid_blocks)
        return run

    def gather_launchers(self, session):
        """(plain, tiled) launchers with the call signatures of SIRCL's all-gather launchers."""
        world, rank, lanes = session.world_size, session.rank, session.lane_count
        threads, one_block = session._threads, 1 if session._one_block_polls else 0

        def tiled(input_address, output_address, shard_packs, nbytes, tile_cols, in_row_stride, out_row_stride,
                  out_src_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_address,
                  stage_counter, tail_counter, poison_address, spin_limit, grid_blocks, arrival_address=0):
            args = GatherArgs(input_address, output_address, shard_packs, nbytes, tile_cols, 0, in_row_stride,
                              out_row_stride, out_src_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes,
                              epoch_address, stage_counter, tail_counter, poison_address, arrival_address,
                              spin_limit, rank, lanes, one_block)
            self._check(self.lib.sirclkp_launch_allgather(world, grid_blocks, threads, self._stream(),
                                                          ctypes.byref(args)))

        def plain(input_address, output_address, shard_packs, nbytes, row_packs, recv_base, flag_base, send_base,
                  ctrl_base, slot_bytes, epoch_address, stage_counter, tail_counter, poison_address, spin_limit,
                  grid_blocks, arrival_address=0):
            tiled(input_address, output_address, shard_packs, nbytes, row_packs, row_packs, world * row_packs,
                  row_packs, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_address, stage_counter,
                  tail_counter, poison_address, spin_limit, grid_blocks, arrival_address=arrival_address)

        return plain, tiled

    def chain_launcher(self, dtype_name: str, session):
        """A launcher with the call signature of SIRCL's chain all-reduce launcher."""
        dtype = DTYPE_CODES[dtype_name]
        blocks, threads, unroll = session.chain_blocks, session._threads, session.chain_unroll

        threads = min(threads, LINK_MAX_THREADS)

        def run(input_address, output_address, a_packs, b_packs, chunk_packs, chain_base, counters, ctrl_base,
                poison_address, spin_limit, trace_address=0):
            args = ChainArgs(input_address, output_address, a_packs, b_packs, chunk_packs, 0, chain_base, counters,
                             ctrl_base, poison_address, spin_limit, session.event_trace if trace_address else 0,
                             trace_address, session.world_size, session.chain_index, session._chain_prev,
                             session._chain_next, session.rank, session.lane_count, session.chain_slots, blocks,
                             session.chain_slot_bytes)
            self._check(self.lib.sirclkp_launch_chain(dtype, unroll, 4 * blocks, threads, self._stream(),
                                                      ctypes.byref(args)))
        return run

    def _launch_link(self, kind: int, dtype: int, session, **fields) -> None:
        """One link collective op of the session: its chain position, rank order and link geometry."""
        args = LinkArgs(world=session.world_size, index=session.chain_index, prev=session._chain_prev,
                        next=session._chain_next, rank=session.rank, lanes=session.lane_count,
                        slots=session.link_slots, blocks_per_role=session.link_blocks,
                        slot_bytes=session.link_slot_bytes, **fields)
        args.order = (i32 * LINK_MAX_WORLD)(*session.chain_order)
        threads = min(session._threads, LINK_MAX_THREADS)
        entry = RING_REDUCE_TWO_PASS if kind == RING_REDUCE and os.environ.get("LIBSIRCL_RING_REDUCE_PASSES") == "2" \
            else kind
        self._check(self.lib.sirclkp_launch_link(entry, dtype, session.link_unroll, LINK_ROLES[kind] * session.link_blocks,
                                                 threads, self._stream(), ctypes.byref(args)))
        self.link_launches[kind] += 1

    def link_launcher(self, key: tuple, session):
        """A launcher with the call signature of SIRCL's launcher of session._launchers[key]: ("link-gather",),
        ("link-scatter", dtype) or ("link-ring", mode, dtype name or "bytes")."""
        if key[0] == "link-gather":
            def gather(input_address, output_address, shard_packs, piece_packs, link_base, counters, ctrl_base,
                       poison_address, spin_limit):
                self._launch_link(LINK_GATHER, 0, session, input=input_address, output=output_address,
                                  chunk_packs=shard_packs, stride_packs=shard_packs, piece_packs=piece_packs,
                                  link_base=link_base, counters=counters, ctrl_base=ctrl_base, poison=poison_address,
                                  spin_limit=spin_limit)
            return gather
        if key[0] == "link-scatter":
            code = DTYPE_CODES[str(key[1]).split(".")[-1]]

            def scatter(input_address, output_address, scratch_address, chunk_packs, stride_packs, piece_packs,
                        link_base, counters, piece_counters, ctrl_base, poison_address, spin_limit):
                self._launch_link(LINK_SCATTER, code, session, input=input_address, output=output_address,
                                  scratch=scratch_address, chunk_packs=chunk_packs, stride_packs=stride_packs,
                                  piece_packs=piece_packs, link_base=link_base, counters=counters,
                                  piece_counters=piece_counters, ctrl_base=ctrl_base, poison=poison_address,
                                  spin_limit=spin_limit)
            return scatter
        mode, name = key[1], key[2]
        kind = {"gather": RING_GATHER, "scatter": RING_SCATTER, "reduce": RING_REDUCE}[mode]
        code = 0 if name == "bytes" else DTYPE_CODES[name]

        def ring(input_address, output_address, chunk_packs, stride_packs, piece_packs, link_base, counters,
                 ctrl_base, poison_address, spin_limit, trace_address=0, stagger=0, gather_stagger=0):
            self._launch_link(kind, code, session, input=input_address, output=output_address,
                              chunk_packs=chunk_packs, stride_packs=stride_packs, piece_packs=piece_packs,
                              stagger=stagger, gather_stagger=gather_stagger, link_base=link_base,
                              counters=counters, ctrl_base=ctrl_base, poison=poison_address, spin_limit=spin_limit,
                              trace_base=trace_address, trace_capacity=session.event_trace if trace_address else 0)
        return ring

    def scatter_launcher(self, dtype_name: str | None, session):
        """A launcher with the call signature of SIRCL's scatter launcher (dtype None: the all-to-all)."""
        world, rank, lanes, threads = session.world_size, session.rank, session.lane_count, session._threads
        code = -1 if dtype_name is None else DTYPE_CODES[dtype_name]

        def run(input_address, output_address, size_packs, nbytes, chunk_packs, src_stride, dst_stride, recv_base,
                flag_base, send_base, ctrl_base, slot_bytes, epoch_address, stage_counter, tail_counter,
                poison_address, spin_limit, grid_blocks):
            args = ScatterArgs(input_address, output_address, size_packs, nbytes, chunk_packs, 0, src_stride,
                               dst_stride, recv_base, flag_base, send_base, ctrl_base, slot_bytes, epoch_address,
                               stage_counter, tail_counter, poison_address, spin_limit, rank, lanes)
            self._check(self.lib.sirclkp_launch_scatter(code, world, grid_blocks, threads, self._stream(),
                                                        ctypes.byref(args)))
        return run


def slots_preview(group, types):
    """Per rank, whether its session prepared chain launchers for the dtypes."""
    return [any(("chain", dtype) in session._launchers for dtype in types) for session in group.sessions]


def link_keys(session) -> list:
    """The session's compiled link collective launchers (keys of session._launchers)."""
    return sorted((key for key in session._launchers if key[0] in LINK_KEYS), key=str)


SIRCL_LINK_CHECKS = ("_chain_checks", "_gather_chain_checks", "_scatter_chain_checks", "_ring_checks",
                     "_late_link_checks", "_mixed_piece_checks", "_ring_min_checks", "_stagger_checks")


class _Done(Exception):
    """The suite finished early (``--suite sircl-links``)."""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--kp-lib", required=True)
    parser.add_argument("--layout", default="path:0-1")
    parser.add_argument("--lanes", type=int, default=1)
    parser.add_argument("--max-size", type=int, default=256 << 10)
    parser.add_argument("--dtypes", default="bfloat16,float16,float32")
    parser.add_argument("--json", default="", help="write the check results to this file")
    parser.add_argument("--no-timeout-check", action="store_true")
    parser.add_argument("--suite", choices=("cases", "sircl-links"), default="cases")
    args = parser.parse_args(argv)
    for name in SCHEDULES:
        os.environ.setdefault(name, "pieces" if args.suite == "cases" else SIRCL_DEFAULT_SCHEDULES[name])

    import torch
    from sparkring_sircl import references as sircl_references
    from sparkring_sircl.oneshot import _scatter_cute
    from sparkring_sircl.testing import gpu_emulation as ge
    from sparkring_sircl.testing import native_build

    pack = KernelPack(args.kp_lib)
    build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", "/tmp/sircl-ccl-work/sim"))
    library = native_build.build_shared_library(build)
    environment = {"SIRCL_LARGE_PIECE_BYTES": str(args.max_size), "SIRCL_SERVING_WAIT_S": "0.5"}
    if args.suite == "sircl-links":
        # SIRCL's emulation harness settings (gpu_emulation.run_checks).
        environment.update({"SIRCL_SERVING_WAIT_S": str(ge.SERVING_LAG_LIMIT_S), "SIRCL_RING_MIN_BYTES": "0",
                            "SIRCL_CHAIN_MIN_BYTES": str(2 << 20)})
    group = ge.EmulatedGroup(args.layout, args.lanes, max_size=args.max_size, max_gather_bytes=64 << 10,
                             library=library, environment=environment)
    world = group.world
    types = [getattr(torch, name) for name in args.dtypes.split(",")]
    names = {dtype: str(dtype).split(".")[-1] for dtype in types}
    results: list[tuple[str, bool, str]] = []

    def report(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    def scatter_key(session, mode: str, dtype_name: str):
        return _scatter_cute.launcher_key(mode, dtype_name, session.world_size, session.rank, session._threads,
                                          session._layout.slots, session._layout.flag_stride, session.lane_count,
                                          session.device.index)

    try:
        started = time.perf_counter()
        for rank, session in enumerate(group.sessions):
            with torch.cuda.stream(group.streams[rank]):
                session.prepare(tuple(types), padded_gather=True,
                                scatter=session.scatter_available or session.link_available,
                                links=args.suite == "sircl-links")
        group.load_modules(types)
        pack.load()
        session0 = group.sessions[0]
        chains = sum(1 for entry in slots_preview(group, types) if entry)
        report("prepare", True, f"{time.perf_counter() - started:.1f} s; kernel pack {pack.hash[:16]}; "
               f"link pack {pack.lib.sirclkp_links_hash().decode()[:16]}; chain launchers {chains}; "
               f"link launchers {sum(len(link_keys(s)) for s in group.sessions)} "
               f"({', '.join(str(k[:2] if k[0] != 'link-scatter' else (k[0], names[k[1]])) for k in link_keys(session0))}); "
               f"schedules {session0.large_schedule}/{session0.gather_schedule}/{session0.scatter_schedule}; "
               f"ring {session0.ring_available}; chain order {session0.chain_order}; "
               f"one-shot limit {session0.oneshot_max_bytes} B; threads {session0._threads}; "
               f"flag pollers {session0.flag_pollers}; scatter {session0.scatter_available}")
        # Per rank: the DSL's and the pack's launchers for every swapped slot.
        slots: list[dict] = []
        for session in group.sessions:
            entry = {}
            for dtype in types:
                name = names[dtype]
                for algorithm, label in ((ONESHOT, "oneshot"), (TWOSHOT, "twoshot")):
                    entry[("launchers", (label, dtype))] = (session._launchers[(label, dtype)],
                                                            pack.reduce_launcher(algorithm, name, session))
                key = scatter_key(session, "reduce", name)
                entry[("scatter", key)] = (_scatter_cute._LAUNCHERS[key], pack.scatter_launcher(name, session))
                if ("chain", dtype) in session._launchers:
                    entry[("launchers", ("chain", dtype))] = (session._launchers[("chain", dtype)],
                                                              pack.chain_launcher(name, session))
            key = scatter_key(session, "copy", "bytes")
            entry[("scatter", key)] = (_scatter_cute._LAUNCHERS[key], pack.scatter_launcher(None, session))
            plain, tiled = pack.gather_launchers(session)
            entry[("gather", "plain")] = (session._gather_launcher, plain)
            entry[("launchers", ("gather-tiles",))] = (session._launchers[("gather-tiles",)], tiled)
            for key in link_keys(session):
                entry[("launchers", key)] = (session._launchers[key], pack.link_launcher(key, session))
            slots.append(entry)
        prepared_keys = [set(session._launchers) for session in group.sessions]

        def launchers_unchanged() -> tuple[bool, str]:
            """No launcher was compiled after the swap (it would run the DSL on a C++ rank)."""
            added = [sorted(map(str, set(session._launchers) - prepared_keys[rank]))
                     for rank, session in enumerate(group.sessions)]
            return not any(added), "" if not any(added) else f"compiled after the swap: {added}"

        # Tail-arrival words of the chain counters (word 3) and the link counters (words 5, 6, 7, 12, 13).
        chain_tails, link_tails = [3], [5, 6, 7, 12, 13]

        def use(mode: str) -> None:
            # Every session is idle here: the libsircl pack's last block returns its kernel type's tail word
            # to 0, SIRCL's kernels may count every grid-th arrival, so the words restart at 0 for both.
            torch.cuda.synchronize()
            for session in group.sessions:
                for name, words in (("_chain_counters", chain_tails), ("_link_counters", link_tails)):
                    counters = getattr(session, name, None)
                    if counters is not None:
                        for word in words:
                            if word < counters.numel():
                                counters[word] = 0
            torch.cuda.synchronize()
            for rank, session in enumerate(group.sessions):
                mine = mode == "cxx" or (mode == "mixed" and rank % 2 == 0)
                for (where, key), (theirs, ours) in slots[rank].items():
                    chosen = ours if mine else theirs
                    if where == "launchers":
                        session._launchers[key] = chosen
                    elif where == "scatter":
                        _scatter_cute._LAUNCHERS[key] = chosen
                    else:
                        session._gather_launcher = chosen

        if args.suite == "sircl-links":
            for mode in ("dsl", "mixed", "cxx"):
                use(mode)
                before = dict(pack.link_launches)
                for check in SIRCL_LINK_CHECKS:
                    for name, ok, detail in getattr(ge, check)(group, types) if check in (
                            "_chain_checks", "_scatter_chain_checks", "_ring_checks") else getattr(ge, check)(group):
                        report(f"{mode} {name}", ok, detail)
                ok, detail = launchers_unchanged()
                report(f"{mode} launchers as swapped", ok, detail)
                counts = {LINK_KIND_NAMES[kind]: pack.link_launches[kind] - before[kind] for kind in LINK_ROLES}
                expected = mode != "dsl"
                report(f"{mode} C++ link launches", expected == any(counts.values()),
                       ", ".join(f"{name} {count}" for name, count in counts.items()))
            healthy = [not session.poisoned for session in group.sessions]
            report("health after every mode", all(healthy), "" if all(healthy) else f"poisoned {healthy}")
            raise _Done()

        max_size = args.max_size
        bf16 = torch.bfloat16
        cases = []   # (label, inputs factory args, call, reference)
        for dtype in types:
            item = torch.empty((), dtype=dtype).element_size()
            for nbytes, algorithm in ((16, "oneshot"), (16, "twoshot"), (48, "twoshot"), (4096, None),
                                      (4096, "twoshot"), (65536 + 16, "oneshot"), (131072, None),
                                      (131072 + 16, None), (max_size, None), (max_size - 16, "twoshot")):
                cases.append((f"all_reduce {names[dtype]} {nbytes} B {algorithm or 'auto'}", dtype,
                              (nbytes // item,), lambda s, x, a=algorithm: s.all_reduce(x, algorithm=a), "sum"))
            cases.append((f"all_reduce_large {names[dtype]} {(3 * max_size + 6) // item * item} B", dtype,
                          ((3 * max_size + 6) // item,), lambda s, x: s.all_reduce_large(x), "large"))
            if session0.chain_available:
                # Several chunks per half and more chunks than slots, so slots are reused within one op.
                chunk = session0.chain_chunk_bytes
                for nbytes in (2 * chunk + 16, 2 * session0.chain_slots * chunk + 4 * chunk + 48,
                               10 * session0.chain_slots * chunk // 3 // 16 * 16):
                    cases.append((f"all_reduce_large chain {names[dtype]} {nbytes} B", dtype, (nbytes // item,),
                                  lambda s, x: s.all_reduce_large(x), "large"))
            for chunk_bytes in (16, 4096, 65536 + 16, (2 * max_size // world) // 16 * 16 + 48):
                elements = chunk_bytes // item * world
                cases.append((f"reduce_scatter {names[dtype]} {chunk_bytes // item * item} B chunks", dtype,
                              (elements,), lambda s, x: s.reduce_scatter(x), "scatter"))
        for shape, dim in (((4096,), 0), ((8, 96), -1), ((5, 7), -1), ((3, 4096), 0), ((200, 72), -1)):
            cases.append((f"all_gather {list(shape)} dim {dim}", bf16, shape,
                          lambda s, x, d=dim: s.all_gather(x, dim=d), ("cat", dim)))
        for shape, dim in (((3, 40000), -1), ((50000,), 0), ((7, 333), -1)):
            cases.append((f"all_gather_large {list(shape)} dim {dim}", bf16, shape,
                          lambda s, x, d=dim: s.all_gather_large(x, dim=d), ("cat", dim)))
        for chunk_bytes in (16, 4096, (2 * max_size // world) // 16 * 16):
            cases.append((f"all_to_all {chunk_bytes} B chunks", torch.uint8, (chunk_bytes * world,),
                          lambda s, x: s.all_to_all(x, torch.empty_like(x)), "exchange"))

        def inputs_for(dtype, shape, seed):
            if dtype == torch.uint8:
                return [torch.randint(0, 256, shape, generator=torch.Generator().manual_seed(seed * 1009 + r),
                                      dtype=torch.uint8) for r in range(world)]
            return ge._inputs(torch, world, shape, dtype, seed)

        def large(inputs):
            """all_reduce_large of the inputs under the session's plan (chain ops round at every hop)."""
            nbytes = inputs[0].numel() * inputs[0].element_size()
            plan = session0.large_reduce_plan(nbytes)
            return sircl_references.large_all_reduce(torch, inputs, plan, session0.chain_order)

        def scatter_reference(inputs):
            """Every rank's reduce-scatter output on the path the session takes for the input's geometry."""
            probe = torch.empty(inputs[0].shape, dtype=inputs[0].dtype, device=session0.device)
            if session0.scatter_uses_ring(probe):
                return sircl_references.ring_reduce_scatter(torch, inputs, session0.chain_order)
            if session0.scatter_uses_chain(probe):
                return sircl_references.chain_reduce_scatter(torch, inputs, session0.chain_order)
            return list(ge._sum(torch, inputs).chunk(world))

        def references(kind, inputs):
            """One reference per rank."""
            if kind == "large":
                return [large(inputs)] * world
            if kind == "sum":
                total = ge._sum(torch, inputs)
                return [total] * world
            if kind == "scatter":
                return scatter_reference(inputs)
            if kind == "exchange":
                chunks = [t.chunk(world) for t in inputs]
                return [torch.cat([chunks[s][r] for s in range(world)]) for r in range(world)]
            return [torch.cat(inputs, dim=kind[1])] * world

        outputs: dict[str, dict[int, list]] = {}
        for mode in ("dsl", "mixed", "cxx"):
            use(mode)
            outputs[mode] = {}
            for index, (label, dtype, shape, call, kind) in enumerate(cases):
                inputs = inputs_for(dtype, shape, 7000 + index)
                wanted = references(kind, inputs)
                name = f"{mode} {label}"
                try:
                    got = group.each(lambda rank, session, call=call: call(session, inputs[rank].to(session.device)))
                    got = [tensor.cpu() for tensor in got]
                except Exception as error:  # noqa: BLE001
                    report(name, False, f"{type(error).__name__}: {error}")
                    continue
                wrong = [rank for rank in range(world) if not ge._same_bits(torch, got[rank], wanted[rank])]
                outputs[mode][index] = got
                detail = f"ranks {wrong} differ from the reference" if wrong else ""
                if not wrong and mode != "dsl" and index in outputs["dsl"]:
                    differ = [rank for rank in range(world)
                              if not ge._same_bits(torch, got[rank], outputs["dsl"][index][rank])]
                    if differ:
                        wrong = differ
                        detail = f"ranks {differ} differ from the DSL outputs"
                report(name, not wrong, detail)
            # CUDA graph capture: every kind in one graph per rank, replayed for two seeds.
            for dtype in types:
                item = torch.empty((), dtype=dtype).element_size()
                small, size = 4096 // item, (2 * max_size + 32) // item
                chunk = 8192 // item
                gather_rows = 6
                total = 2 * small + size + world * chunk

                def captured(session, x, small=small, chunk=chunk, size=size):
                    a = session.all_reduce(x[:small], algorithm="oneshot")
                    b = session.all_reduce(x[small:2 * small], algorithm="twoshot")
                    c = session.all_reduce_large(x[2 * small:2 * small + size])
                    d = session.reduce_scatter(x[2 * small + size:])
                    e = session.all_gather(x[:gather_rows * 16].view(gather_rows, 16), dim=0)
                    return torch.cat([a, b, c, d, e.reshape(-1)])

                def combine(inputs, small=small, chunk=chunk, size=size):
                    total = ge._sum(torch, inputs)
                    segment = large([t[2 * small:2 * small + size] for t in inputs])
                    scattered = scatter_reference([t[2 * small + size:] for t in inputs])
                    per_rank = []
                    gathered = torch.cat([t[:gather_rows * 16] for t in inputs])
                    for r in range(world):
                        per_rank.append(torch.cat([total[:2 * small], segment, scattered[r], gathered]))
                    return per_rank

                name, ok, detail = ge._graph_check(group, f"{mode} graph all-reduce + reduce-scatter + all-gather "
                                                   f"{names[dtype]}", (total,), dtype, captured, combine,
                                                   (8101, 8102), per_rank=True)
                report(name, ok, detail)
        healthy = [not session.poisoned for session in group.sessions]
        report("health after every mode", all(healthy), "" if all(healthy) else f"poisoned {healthy}")
        ok, detail = launchers_unchanged()
        report("launchers as swapped", ok, detail)
        report("kernel pack launches", pack.launches > 0,
               f"{pack.launches} C launches; " + ", ".join(f"{LINK_KIND_NAMES[kind]} {count}"
                                                        for kind, count in pack.link_launches.items()))
        if not args.no_timeout_check:
            use("cxx")
            for session in group.sessions:
                session.enter_serving()

            def lagging(rank: int, session):
                x = torch.ones(64, dtype=torch.bfloat16, device=session.device)
                if rank != 0:
                    return None
                begun = time.perf_counter()
                try:
                    session.all_reduce(x, algorithm="oneshot")
                    ge.wait_stream(torch.cuda.current_stream())
                except RuntimeError as error:
                    # Some SIRCL trees raise once the wait poisons the session; the check reads the same words.
                    if "poisoned" not in str(error):
                        raise
                    torch.cuda.synchronize()
                waited = time.perf_counter() - begun
                ctrl = [int(v) & 0xFFFFFFFF for v in session._ctrl_np[:9]]
                return waited, ctrl, session.poisoned

            outcomes = group.each(lagging)
            waited, ctrl, poisoned = outcomes[0]
            # Word 2: the sequence; 3: the missing peer; 6: the missing lane.
            ok = poisoned and 0.4 < waited < 3.0 and ctrl[2] != 0 and ctrl[3] == 1 and ctrl[6] == 0
            report("C++ rank 0 alone, serving limit 0.5 s: poisons and names peer 1 lane 0", ok,
                   f"waited {waited:.2f} s, error sequence {ctrl[2]}, peer {ctrl[3]}, lane {ctrl[6]}")
    except _Done:
        pass
    finally:
        group.close()
    failed = sum(1 for _, ok, _ in results if not ok)
    print(f"{len(results)} checks, {failed} failed", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"layout": args.layout, "lanes": args.lanes, "suite": args.suite,
                                               "kernel_pack": pack.hash,
                                               "link_pack": pack.lib.sirclkp_links_hash().decode(),
                                               "checks": results}, indent=1))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
