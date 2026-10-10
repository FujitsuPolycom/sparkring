#!/usr/bin/env python3
"""One rank of an all-reduce timing sweep through libsircl's NCCL API, in the manner of nccl-tests.

For every dtype and message size (``--min`` to ``--max`` bytes, multiplied by ``--factor``) the rank
checks one out-of-place ``ncclAllReduce`` (sum) bit for bit against the library's own result for its
schedule (sizes up to ``--check-max``; library_rank.allreduce_reference, which reads the schedule settings
from the environment as the library does): the rank-order float32 sum rounded once for the pieces
schedule, the per-hop rounding in ring order for a ring op and in chain order for a chain op. The ring and
chain order is LIBSIRCL_CHAIN_ORDER read as ranks (every rank at its own position, as the gates and
RUNBOOK.md place them), or the ranks in order. The check row also records whether the output equals the
rank-order reference, which differs from a ring or chain op's on more than two ranks. Then the rank times
the call two ways on one stream with CUDA events:

- eager: ``--warmup`` calls, then ``--iters`` calls back to back; time per call;
- graph: ``--graph`` calls captured in one CUDA graph, the graph replayed until ``--iters`` calls have
  run; time per call.

A device synchronization, a one-element all-reduce and another synchronization precede every timed
loop, so the ranks start each loop together and no replay overlaps another call. Each row reports the time per call, the algorithm bandwidth (bytes / time) and the bus
bandwidth (algorithm bandwidth x 2(W-1)/W, nccl-tests' all-reduce convention). Rank 0 prints the table;
every rank writes its rows, its receipt and its checks to ``--out``.

Usage, one process per rank:
  python perf_rank.py --library build/libsircl.so --world 2 --rank R --id-server HOST:PORT --out r.json
In GPU emulation the ranks share one GPU by time slicing, so the times there measure nothing about a
fabric; only the checks and the tool itself are exercised.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import unique_id  # noqa: E402

NCCL_DTYPES = {"bfloat16": 9, "float16": 6, "float32": 7}


def sizes(low: int, high: int, factor: int):
    size = low
    while size <= high:
        yield size
        size *= factor


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--id-file", default="")
    parser.add_argument("--id-server", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--dtypes", default="bfloat16,float16,float32")
    parser.add_argument("--min", type=int, default=8)
    parser.add_argument("--max", type=int, default=256 << 20)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--graph", type=int, default=20, help="calls per CUDA graph; 0 skips graph timing")
    parser.add_argument("--check-max", type=int, default=64 << 20)
    args = parser.parse_args(argv)

    import torch

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    lib = ctypes.CDLL(args.library)
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, unique_id.UniqueId,
                                     ctypes.c_int]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.sirclGetReceipt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                       ctypes.POINTER(ctypes.c_size_t)]
    uid = unique_id.share(lib, args.rank, args.world, args.id_file, args.id_server)
    comm = ctypes.c_void_p()
    if lib.ncclCommInitRank(ctypes.byref(comm), args.world, uid, args.rank) != 0:
        raise SystemExit(f"ncclCommInitRank: {lib.ncclGetLastError(None).decode()}")
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    token = torch.zeros(1, dtype=torch.float32, device="cuda")

    def all_reduce(x, y, dtype, stream_handle=handle):
        rc = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), x.numel(), dtype, 0,
                               comm, stream_handle)
        if rc:
            raise RuntimeError(f"ncclAllReduce result {rc}: {lib.ncclGetLastError(comm).decode()}")

    def together():
        # Everything queued finishes first: a graph replay of this communicator must never overlap its
        # other calls (the library orders eager calls across streams, not replays).
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            all_reduce(token, token, 7)
        torch.cuda.synchronize()

    from library_rank import allreduce_reference, note_cycle_plan, receipt_of, reference

    note_cycle_plan(receipt_of(lib, comm), args.world)

    position = os.environ.get("LIBSIRCL_POSITION")
    if position and int(position) != args.rank:
        raise SystemExit(f"LIBSIRCL_POSITION={position} on rank {args.rank}: the check models the chain order for "
                         "ranks placed at their own position")
    chain = os.environ.get("LIBSIRCL_CHAIN_ORDER")
    order = [int(item) for item in chain.split(",")] if chain else list(range(args.world))
    rows, checks = [], []
    bus = 2.0 * (args.world - 1) / args.world
    for dtype_name in args.dtypes.split(","):
        dtype = getattr(torch, dtype_name)
        item = torch.empty((), dtype=dtype).element_size()
        for nbytes in sizes(args.min, args.max, args.factor):
            count = max(1, nbytes // item)
            x = torch.empty(count, dtype=dtype, device="cuda")
            y = torch.empty_like(x)
            row = {"dtype": dtype_name, "bytes": count * item, "count": count}
            if count * item <= args.check_max:
                values = [torch.randn(count, generator=torch.Generator().manual_seed(7919 * count + r)).to(dtype)
                          for r in range(args.world)]
                want = allreduce_reference(torch, values, order)
                x.copy_(values[args.rank].cuda())
                together()
                with torch.cuda.stream(stream):
                    all_reduce(x, y, NCCL_DTYPES[dtype_name])
                torch.cuda.synchronize()
                got = y.cpu()
                good = torch.equal(got.view(torch.uint8), want.view(torch.uint8))
                row["correct"] = bool(good)
                row["rank_order_sum"] = bool(torch.equal(got.view(torch.uint8), reference(torch, values).view(torch.uint8)))
                checks.append((f"all-reduce {dtype_name} {count * item} B", bool(good)))
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            together()
            with torch.cuda.stream(stream):
                for _ in range(args.warmup):
                    all_reduce(x, y, NCCL_DTYPES[dtype_name])
            together()
            with torch.cuda.stream(stream):
                start.record(stream)
                for _ in range(args.iters):
                    all_reduce(x, y, NCCL_DTYPES[dtype_name])
                end.record(stream)
            end.synchronize()
            row["eager_us"] = start.elapsed_time(end) * 1000.0 / args.iters
            if args.graph:
                graph = torch.cuda.CUDAGraph()
                capture = torch.cuda.Stream()
                chandle = ctypes.c_void_p(capture.cuda_stream)
                torch.cuda.synchronize()
                with torch.cuda.graph(graph, stream=capture):
                    for _ in range(args.graph):
                        all_reduce(x, y, NCCL_DTYPES[dtype_name], chandle)
                replays = max(1, args.iters // args.graph)
                with torch.cuda.stream(stream):
                    graph.replay()
                together()
                with torch.cuda.stream(stream):
                    start.record(stream)
                    for _ in range(replays):
                        graph.replay()
                    end.record(stream)
                end.synchronize()
                row["graph_us"] = start.elapsed_time(end) * 1000.0 / (replays * args.graph)
                del graph
            seconds = row["eager_us"] / 1e6
            row["algbw_GBps"] = row["bytes"] / seconds / 1e9
            row["busbw_GBps"] = row["algbw_GBps"] * bus
            rows.append(row)
            if args.rank == 0:
                print(f"{dtype_name:>9} {row['bytes']:>11} B  eager {row['eager_us']:10.2f} us  "
                      f"graph {row.get('graph_us', float('nan')):10.2f} us  busbw {row['busbw_GBps']:8.3f} GB/s  "
                      f"{'' if 'correct' not in row else 'ok' if row['correct'] else 'WRONG'}", flush=True)
            del x, y
    from library_rank import receipt_of

    receipt = receipt_of(lib, comm)
    lib.ncclCommDestroy(comm)
    Path(args.out).write_text(json.dumps({"rank": args.rank, "world": args.world, "rows": rows, "checks": checks,
                                          "receipt": receipt, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}))
    wrong = [name for name, ok in checks if not ok]
    if args.rank == 0:
        print(f"{len(checks)} sizes checked, {len(wrong)} wrong")
    return 1 if wrong else 0


if __name__ == "__main__":
    sys.exit(main())
