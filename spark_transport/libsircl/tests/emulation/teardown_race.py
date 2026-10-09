#!/usr/bin/env python3
"""Communicator teardown right after a ring collective, in GPU emulation, with one rank's writes delayed.

Started without --rank, the script is the runner: it starts one process per rank on the one GPU over the
emulation transport, with every rank's environment equal except that the slow rank (``--slow-rank``)
executes each of its RDMA writes no earlier than ``--slow-ns`` after posting it (the stand-in's
SIRCL_EMU_LATENCY_NS). Every rank then repeats, ``--rounds`` times: ncclCommSplit with keys that reverse the
rank order, one bfloat16 ncclAllReduce on the child over the ring schedule (links 2 and 3, the all-gather
half forwarding every received item), a bit-exact check, and ncclCommDestroy of the child as soon as its own
result is right. A final all-reduce on the parent and ncclCommGetAsyncError check that the parent survives.

What it exercises: a rank's kernel completes once it has every inbound item, while its progress thread may
still owe a forward to its successor (an outbound item gated by the successor's credit). The slow rank
releases its inbound items, and so posts the credits its predecessor needs, late, so the predecessor
finishes first. Teardown is correct when no rank's destroy strands such a forward and no write lands on a
queue pair or region a peer has already freed: every round's child all-reduce is exact on every rank and
the parent stays healthy. A rank that waits for a stranded item fails after the startup wait limit
(SIRCL_STARTUP_WAIT_S, 20 s here) and its child's async error says why.

With --expect close-error the rounds check the close's error paths instead: each round finalizes the child
twice, attempts an all-reduce after the close and destroys it, without asking for its async error first, and
passes when the close reported an error (the first ncclCommFinalize or the destroy), the second finalize
returned the same result and the all-reduce after the close was refused. The close fails when the ranks'
flag waits time out (--late-rank R --late-s S: rank R starts its all-reduce S seconds late; with
SIRCL_STARTUP_WAIT_S below S) or when releasing the transport fails (--env SIRCL_EMU_FAIL_DEREG=1: the
stand-in fails every deregistration, and destroy keeps the arena allocated).

With --rank (and --id-server HOST:PORT, as library_rank.py takes it) one rank runs alone, on any transport
and host: started on every rank of a fabric group with that group's route-map settings, the ranks run the
same rounds without a delayed rank, and each prints one summary line and writes its JSON result.

Status: research-only test infrastructure (the delay models a late consumer, not a cable).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

RING = {
    "SIRCL_LARGE_SCHEDULE": "ring", "SIRCL_GATHER_SCHEDULE": "ring", "SIRCL_SCATTER_SCHEDULE": "ring",
    "SIRCL_RING_MIN_BYTES": "0", "LIBSIRCL_RING_WINDOW": "0", "SIRCL_STARTUP_WAIT_S": "20",
}


def run_rank(args) -> int:
    import torch

    import unique_id
    from library_rank import allreduce_reference, inputs_for, same_bits

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    lib = ctypes.CDLL(args.library)
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, unique_id.UniqueId,
                                     ctypes.c_int]
    lib.ncclCommSplit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_void_p]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    lib.ncclCommFinalize.argtypes = [ctypes.c_void_p]
    lib.ncclCommGetAsyncError.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int)]
    world, rank = args.world, args.rank
    rounds: list[dict] = []
    out = {"rank": rank, "rounds": rounds}

    def save():
        Path(args.out).write_text(json.dumps(out))

    uid = unique_id.share(lib, rank, world, args.id_file, args.id_server)
    comm = ctypes.c_void_p()
    code = lib.ncclCommInitRank(ctypes.byref(comm), world, uid, rank)
    if code:
        out["init"] = f"result {code}: {lib.ncclGetLastError(None).decode()}"
        save()
        return 1
    stream = torch.cuda.Stream()
    count = args.count
    for k in range(args.rounds):
        started = time.perf_counter()
        child = ctypes.c_void_p()
        entry: dict = {"round": k}
        code = lib.ncclCommSplit(comm, 0, world - 1 - rank, ctypes.byref(child), None)
        if code:
            entry.update(ok=False, detail=f"split: result {code}: {lib.ncclGetLastError(None).decode()}")
            rounds.append(entry)
            break
        values = inputs_for(torch, world, count, torch.bfloat16, 9100 + 37 * k)
        want = allreduce_reference(torch, [values[world - 1 - c] for c in range(world)],
                                   order=list(range(world - 1, -1, -1)))
        x = values[rank].cuda()
        y = torch.empty_like(x)
        torch.cuda.synchronize()
        if rank == args.late_rank:
            time.sleep(args.late_s)
        code = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), count, 9, 0, child,
                                 ctypes.c_void_p(stream.cuda_stream))
        torch.cuda.synchronize()
        exact = code == 0 and same_bits(torch, y.cpu(), want)
        if args.expect == "close-error":
            first = lib.ncclCommFinalize(child)
            error = lib.ncclGetLastError(child).decode(errors="replace") if first else ""
            second = lib.ncclCommFinalize(child)
            after = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), count, 9, 0,
                                      child, ctypes.c_void_p(stream.cuda_stream))
            destroyed = lib.ncclCommDestroy(child)
            if destroyed and not error:
                error = lib.ncclGetLastError(None).decode(errors="replace")
            entry.update(ok=(first != 0 or destroyed != 0) and second == first and after != 0, exact=exact,
                         finalize=[first, second], after_close=after, destroy=destroyed, error=error,
                         seconds=round(time.perf_counter() - started, 3))
            rounds.append(entry)
            save()
            continue
        state = ctypes.c_int(0)
        lib.ncclCommGetAsyncError(child, ctypes.byref(state))
        if not exact or state.value:
            entry["async_error"] = state.value
            entry["error"] = lib.ncclGetLastError(child).decode(errors="replace")
        destroyed = lib.ncclCommDestroy(child)
        entry.update(ok=exact and not state.value and destroyed == 0, exact=exact, destroy=destroyed,
                     seconds=round(time.perf_counter() - started, 3))
        if destroyed:
            entry["destroy_error"] = lib.ncclGetLastError(None).decode(errors="replace")
        rounds.append(entry)
        save()
    # A split in which no rank takes a color: its bootstrap rounds bring the ranks together again (a late
    # rank's rounds end seconds after the others') before the parent's all-reduce, whose wait limit may be short.
    nocolor = ctypes.c_void_p()
    lib.ncclCommSplit(comm, -1, 0, ctypes.byref(nocolor), None)
    values = inputs_for(torch, world, count, torch.bfloat16, 9099)
    x = values[rank].cuda()
    y = torch.empty_like(x)
    code = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), count, 9, 0, comm,
                             ctypes.c_void_p(stream.cuda_stream))
    torch.cuda.synchronize()
    state = ctypes.c_int(0)
    lib.ncclCommGetAsyncError(comm, ctypes.byref(state))
    out["parent"] = {"exact": code == 0 and same_bits(torch, y.cpu(), allreduce_reference(torch, values)),
                     "async_error": state.value}
    out["parent_destroy"] = lib.ncclCommDestroy(comm)
    save()
    failed = [entry["round"] for entry in rounds if not entry.get("ok")]
    # Under SIRCL_EMU_FAIL_DEREG the parent's destroy fails as well, by design.
    healthy = (out["parent"]["exact"] and not out["parent"]["async_error"]
               and (not out["parent_destroy"] or args.expect == "close-error"))
    print(f"rank {rank}: {len(rounds)} of {args.rounds} rounds ran, failed rounds {failed}, parent "
          f"{'exact and healthy' if healthy else out['parent']}, destroy {out['parent_destroy']}")
    return 0 if not failed and len(rounds) == args.rounds and healthy else 1


def run_group(args) -> int:
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="sircl-teardown-"))
    work.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ)
    base.update({
        "LIBSIRCL_TRANSPORT": "emulation", "SIRCL_EMU_FABRIC": f"/sircl-emu-teardown-{os.getpid()}",
        "LIBSIRCL_EMU_LANES": str(args.lanes), "CUDA_MODULE_LOADING": "EAGER",
        "CUDA_DEVICE_MAX_CONNECTIONS": base.get("CUDA_DEVICE_MAX_CONNECTIONS", "32"),
        "LIBSIRCL_RECEIPT": str(work / "receipt"),
    })
    base.update(RING)
    for item in args.env:
        name, _, value = item.partition("=")
        base[name] = value
    started = time.perf_counter()
    processes = []
    for rank in range(args.world):
        env = dict(base)
        if rank == args.slow_rank and args.slow_ns:
            env["SIRCL_EMU_LATENCY_NS"] = str(args.slow_ns)
        command = [args.python, str(Path(__file__).resolve()), "--rank", str(rank), "--world", str(args.world),
                   "--library", args.library, "--rounds", str(args.rounds), "--count", str(args.count),
                   "--expect", args.expect, "--late-rank", str(args.late_rank), "--late-s", str(args.late_s),
                   "--id-file", str(work / "unique-id"), "--out", str(work / f"rank{rank}.json")]
        log = open(work / f"rank{rank}.log", "w")
        processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    deadline = time.monotonic() + args.timeout
    codes = []
    for process, log in processes:
        try:
            codes.append(process.wait(timeout=max(1.0, deadline - time.monotonic())))
        except subprocess.TimeoutExpired:
            process.kill()
            codes.append(process.wait())
        log.close()
    # Segments the ranks left (a rank that ended before destroy, or a destroy that kept its memory after a
    # failed release): every rank process has ended, so its segments are removed.
    for process, _ in processes:
        for segment in Path("/dev/shm").glob(f"sircl-emu-seg-{process.pid}-*"):
            segment.unlink(missing_ok=True)
    fabric = Path("/dev/shm") / base["SIRCL_EMU_FABRIC"].lstrip("/")
    if fabric.exists():
        fabric.unlink()
    failed_rounds, problems = set(), []
    for rank in range(args.world):
        path = work / f"rank{rank}.json"
        if not path.exists():
            problems.append(f"rank {rank}: no result (exit {codes[rank]}); log {work / f'rank{rank}.log'}")
            continue
        data = json.loads(path.read_text())
        if "init" in data:
            problems.append(f"rank {rank}: ncclCommInitRank {data['init']}")
        done = data["rounds"]
        if len(done) < args.rounds:
            problems.append(f"rank {rank}: {len(done)} of {args.rounds} rounds ran")
        for entry in done:
            if not entry.get("ok") and args.expect == "close-error":
                failed_rounds.add(entry["round"])
                problems.append(f"rank {rank} round {entry['round']}: the close did not fail as expected: "
                                f"finalize {entry.get('finalize')}, all-reduce after the close "
                                f"{entry.get('after_close')}, destroy {entry.get('destroy')}: {entry.get('error')}")
            elif not entry.get("ok"):
                failed_rounds.add(entry["round"])
                problems.append(f"rank {rank} round {entry['round']}: exact {entry.get('exact')}, destroy "
                                f"{entry.get('destroy')}, async error {entry.get('async_error', 0)}: "
                                f"{entry.get('error', entry.get('detail', ''))} {entry.get('destroy_error', '')}")
        parent = data.get("parent")
        if not parent or not parent["exact"] or parent["async_error"] or (
                data.get("parent_destroy") and args.expect != "close-error"):
            problems.append(f"rank {rank}: parent after the rounds {parent}, destroy {data.get('parent_destroy')}")
    for line in problems:
        print(f"FAIL {line}")
    elapsed = time.perf_counter() - started
    print(f"teardown race: {args.rounds} rounds over {args.world} ranks (rank {args.slow_rank} writes delayed "
          f"{args.slow_ns} ns), {len(failed_rounds)} rounds failed, {len(problems)} problems, {elapsed:.1f} s; "
          f"work {work}")
    return 1 if problems else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--count", type=int, default=4096, help="bfloat16 elements per rank")
    parser.add_argument("--slow-rank", type=int, default=0)
    parser.add_argument("--slow-ns", type=int, default=5_000_000)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--expect", choices=("exact", "close-error"), default="exact")
    parser.add_argument("--late-rank", type=int, default=-1, help="the rank that starts each all-reduce late")
    parser.add_argument("--late-s", type=float, default=0.0)
    parser.add_argument("--env", action="append", default=[], help="NAME=VALUE for every rank")
    parser.add_argument("--work", default="")
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--id-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--id-server", default="", help="with --rank: HOST:PORT where rank 0 serves the unique id")
    parser.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run_rank(args) if args.rank >= 0 else run_group(args)


if __name__ == "__main__":
    sys.exit(main())
