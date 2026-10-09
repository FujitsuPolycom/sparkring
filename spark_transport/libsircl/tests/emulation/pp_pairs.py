#!/usr/bin/env python3
"""Pipeline stage pairs of a ring of eight in GPU emulation: two-rank point-to-point through relayed lanes.

The shape TP4 x PP2 gives eight Sparks on a ring: tensor-parallel groups of positions 0-3 and 4-7 and
pipeline pairs of positions i and i + 4, which no cable joins (every lane of the pair crosses three
relays). Started without --rank, the script is the runner: one process per rank on the one GPU over the
emulation transport, every rank with the settings SIRCL's route planner gives its position of ring:8 with
two lanes (tests/data/site_routes_ring8_l2.json, written by ``tools/site_routes.py --layout ring:8 --lanes 2
--json``: its position, chain order, the forward window of every lane toward a rank it reaches through
relays, and the ring plan LIBSIRCL_RING_WINDOW=0 of a ring whose next ranks are cables).

Every rank creates the eight-rank communicator, splits it into its tensor-parallel group (color rank // 4)
and its pipeline pair (color rank % 4; split children keep their parent's positions), runs one all-reduce
on its tensor-parallel group, then several microbatches on its pair at a pace of its own: the first stage
sends a tensor dictionary (2 MiB and 4 MiB bfloat16 tensors, a 6 KiB and a 1,000,003-byte uint8 tensor)
to the second as torch's batch_isend_irecv issues it (one group of sends, one group of the matching
receives), the second stage sends one tensor back, and every third microbatch both stages exchange in one
group (each sends and receives). Every received byte is checked. On the receipt of every pair: the ring
lanes toward the peer keep a window (LIBSIRCL_RING_WINDOW is 0, the peer is reached through relays), the
pair exchanges went through it (window chunks posted) and the smaller transfers through the forward
windows (forward chunks posted). With ``--no-ring-plan`` every rank runs without LIBSIRCL_RING_WINDOW (a
relayed pair without a ring plan): no pair exchange, every transfer through the forward windows.

Status: research-only test infrastructure (the emulation gives every pair a shared-memory lane; the
relays are modeled by the forward windows' chunked posting, not by relay queues).
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
ROUTES = HERE.parent / "data" / "site_routes_ring8_l2.json"
WORLD = 8


def run_rank(args) -> int:
    import numpy as np
    import torch

    import unique_id
    from library_rank import allreduce_reference, inputs_for, receipt_of, same_bits

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    lib = ctypes.CDLL(args.library)
    lib.ncclGetLastError.restype = ctypes.c_char_p
    lib.ncclGetLastError.argtypes = [ctypes.c_void_p]
    lib.ncclCommInitRank.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int, unique_id.UniqueId,
                                     ctypes.c_int]
    lib.ncclCommSplit.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
                                  ctypes.c_void_p]
    lib.ncclAllReduce.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p]
    for name in ("ncclSend", "ncclRecv"):
        getattr(lib, name).argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                                       ctypes.c_void_p, ctypes.c_void_p]
    lib.sirclGetReceipt.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t,
                                    ctypes.POINTER(ctypes.c_size_t)]
    lib.ncclCommDestroy.argtypes = [ctypes.c_void_p]
    rank = args.rank
    checks: list = []
    out = {"rank": rank, "checks": checks}

    def report(name, ok, detail=""):
        checks.append((name, bool(ok), detail))

    def save():
        partial = Path(args.out + ".partial")
        partial.write_text(json.dumps(out))
        os.replace(partial, args.out)

    uid = unique_id.share(lib, rank, WORLD, args.id_file, "")
    parent = ctypes.c_void_p()
    code = lib.ncclCommInitRank(ctypes.byref(parent), WORLD, uid, rank)
    if code:
        report("ncclCommInitRank of the ring of eight", False, lib.ncclGetLastError(None).decode())
        save()
        return 1
    tp, pp = ctypes.c_void_p(), ctypes.c_void_p()
    code = lib.ncclCommSplit(parent, rank // 4, rank, ctypes.byref(tp), None) or \
        lib.ncclCommSplit(parent, rank % 4, rank, ctypes.byref(pp), None)
    report("ncclCommSplit into the tensor-parallel group and the pipeline pair", code == 0,
           "" if code == 0 else lib.ncclGetLastError(None).decode())
    if code:
        save()
        return 1
    stream = torch.cuda.Stream()
    handle = ctypes.c_void_p(stream.cuda_stream)
    stage, peer = rank // 4, 1 - rank // 4  # rank within the pair (keys order the pair by parent rank)

    # One all-reduce on the tensor-parallel group (positions 0-3 or 4-7: a path inside the ring).
    tp_rank = rank % 4
    values = inputs_for(torch, 4, 1 << 20, torch.bfloat16, 7000 + rank // 4)
    x = values[tp_rank].cuda()
    y = torch.empty_like(x)
    torch.cuda.synchronize()
    code = lib.ncclAllReduce(ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(y.data_ptr()), x.numel(), 9, 0, tp,
                             handle)
    stream.synchronize()
    report("tensor-parallel all-reduce, bfloat16 2 MiB", code == 0 and same_bits(torch, y.cpu(),
                                                                                 allreduce_reference(torch, values)))

    def payload(seed, source, size):
        return torch.from_numpy(np.random.default_rng(seed * 1031 + source).integers(0, 256, size, dtype=np.uint8))

    sizes = {"hidden": 2 << 20, "residual": 4 << 20, "small": 6 << 10, "odd": 1000003}
    before = receipt_of(lib, pp)
    rng = np.random.default_rng(100 + rank % 4)  # one pace per pair
    bad = []
    for mb in range(args.microbatches):
        time.sleep(float(rng.uniform(0, 0.05)))
        seed = 8000 + 37 * mb + rank % 4
        # Forward: the first stage's dictionary, as batch_isend_irecv issues it (one group of operations).
        sent = {k: payload(seed, 0, n).cuda() for k, n in sizes.items()}
        got = {k: torch.zeros(n, dtype=torch.uint8, device="cuda") for k, n in sizes.items()}
        torch.cuda.synchronize()
        codes = [lib.ncclGroupStart()]
        for key, n in sizes.items():
            if stage == 0:
                codes.append(lib.ncclSend(ctypes.c_void_p(sent[key].data_ptr()), n, 1, peer, pp, handle))
            else:
                codes.append(lib.ncclRecv(ctypes.c_void_p(got[key].data_ptr()), n, 1, peer, pp, handle))
        codes.append(lib.ncclGroupEnd())
        stream.synchronize()
        if any(codes) or (stage == 1 and any(not torch.equal(got[k].cpu(), sent[k].cpu()) for k in sizes)):
            bad.append(f"microbatch {mb} forward: results {codes}")
        # Backward: one tensor from the second stage to the first.
        back = payload(seed, 1, sizes["hidden"]).cuda()
        into = torch.zeros(sizes["hidden"], dtype=torch.uint8, device="cuda")
        torch.cuda.synchronize()
        code = (lib.ncclSend(ctypes.c_void_p(back.data_ptr()), back.numel(), 1, peer, pp, handle) if stage == 1
                else lib.ncclRecv(ctypes.c_void_p(into.data_ptr()), into.numel(), 1, peer, pp, handle))
        stream.synchronize()
        if code or (stage == 0 and not torch.equal(into.cpu(), back.cpu())):
            bad.append(f"microbatch {mb} backward: result {code}")
        if mb % 3 == 2:
            # Both ways in one group: each stage sends and receives.
            mine = payload(seed, 10 + stage, sizes["residual"]).cuda()
            theirs = payload(seed, 10 + peer, sizes["residual"])
            recv = torch.zeros(sizes["residual"], dtype=torch.uint8, device="cuda")
            torch.cuda.synchronize()
            codes = [lib.ncclGroupStart(),
                     lib.ncclSend(ctypes.c_void_p(mine.data_ptr()), mine.numel(), 1, peer, pp, handle),
                     lib.ncclRecv(ctypes.c_void_p(recv.data_ptr()), recv.numel(), 1, peer, pp, handle),
                     lib.ncclGroupEnd()]
            stream.synchronize()
            if any(codes) or not torch.equal(recv.cpu(), theirs):
                bad.append(f"microbatch {mb} both ways: results {codes}")
    after = receipt_of(lib, pp)
    report(f"pipeline pair {rank % 4}: {args.microbatches} microbatches, every received byte exact", not bad,
           "; ".join(bad[:4]))
    windows = after["forward_windows"]
    ring_chunks = windows["ring_window_chunks"] - before["forward_windows"]["ring_window_chunks"]
    fwd_chunks = windows["chunks_posted"] - before["forward_windows"]["chunks_posted"]
    exchanges = after["pair_exchange"]["ops"] - before["pair_exchange"]["ops"]
    detail = (f"pair plan {after['pair_plan']}, ring window {windows['ring_window_bytes']} B, {exchanges} pair "
              f"exchanges, {ring_chunks} ring window chunks, {fwd_chunks} forward chunks, receipt world "
              f"{after['world']}")
    if args.ring_plan:
        report("pipeline pair through relays: pair exchanges through the ring lanes' window, the rest through "
               "the forward windows", after["pair_plan"] and windows["ring_window_bytes"] > 0 and exchanges > 0
               and ring_chunks > 0 and fwd_chunks > 0, detail)
    else:
        report("pipeline pair through relays without a ring plan: every transfer through the forward windows",
               not after["pair_plan"] and exchanges == 0 and fwd_chunks > 0, detail)
    report("pipeline pair healthy", after["healthy"], json.dumps(after.get("error")))
    for name, comm in (("pipeline pair", pp), ("tensor-parallel group", tp), ("ring of eight", parent)):
        report(f"ncclCommDestroy of the {name}", lib.ncclCommDestroy(comm) == 0,
               lib.ncclGetLastError(None).decode(errors="replace"))
    save()
    return 0 if all(ok for _, ok, _ in checks) else 1


def run_group(args) -> int:
    routes = json.loads(ROUTES.read_text())
    assert routes["layout"] == "ring:8" and len(routes["ranks"]) == WORLD
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix="sircl-pp-pairs-"))
    work.mkdir(parents=True, exist_ok=True)
    base = dict(os.environ)
    base.update({
        "LIBSIRCL_TRANSPORT": "emulation", "SIRCL_EMU_FABRIC": f"/sircl-emu-pp-{os.getpid()}",
        "LIBSIRCL_EMU_LANES": str(routes["lanes"]), "CUDA_MODULE_LOADING": "EAGER",
        "CUDA_DEVICE_MAX_CONNECTIONS": base.get("CUDA_DEVICE_MAX_CONNECTIONS", "32"),
        "LIBSIRCL_RECEIPT": str(work / "receipt"),
    })
    started = time.perf_counter()
    processes = []
    for rank in range(WORLD):
        env = dict(base)
        for name, value in routes["ranks"][rank]["env"].items():
            if name == "SIRCL_PEER_ROUTES":
                continue  # device names of the verbs transport; the emulation names its own devices
            if name == "LIBSIRCL_RING_WINDOW" and not args.ring_plan:
                continue
            env[name] = value
        command = [args.python, str(Path(__file__).resolve()), "--rank", str(rank), "--library", args.library,
                   "--microbatches", str(args.microbatches), "--id-file", str(work / "unique-id"),
                   "--out", str(work / f"rank{rank}.json")] + ([] if args.ring_plan else ["--no-ring-plan"])
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
    for process, _ in processes:
        for segment in Path("/dev/shm").glob(f"sircl-emu-seg-{process.pid}-*"):
            segment.unlink(missing_ok=True)
    fabric = Path("/dev/shm") / base["SIRCL_EMU_FABRIC"].lstrip("/")
    if fabric.exists():
        fabric.unlink()
    total = failed = 0
    for rank in range(WORLD):
        path = work / f"rank{rank}.json"
        if not path.exists():
            print(f"FAIL rank {rank}: no result (exit {codes[rank]}); {work / f'rank{rank}.log'}")
            failed += 1
            continue
        for name, ok, detail in json.loads(path.read_text())["checks"]:
            total += 1
            failed += not ok
            if not ok or rank in (0, 4):
                print(f"{'PASS' if ok else 'FAIL'} rank {rank} {name}{': ' + detail if detail else ''}")
    print(f"pipeline pairs of ring:8{'' if args.ring_plan else ' without a ring plan'}: {total} checks over "
          f"{WORLD} ranks, {failed} failed, {time.perf_counter() - started:.1f} s; work {work}")
    return 1 if failed else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--library", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--microbatches", type=int, default=6)
    parser.add_argument("--no-ring-plan", dest="ring_plan", action="store_false")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--work", default="")
    parser.add_argument("--rank", type=int, default=-1, help=argparse.SUPPRESS)
    parser.add_argument("--id-file", default="", help=argparse.SUPPRESS)
    parser.add_argument("--out", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    return run_rank(args) if args.rank >= 0 else run_group(args)


if __name__ == "__main__":
    sys.exit(main())
