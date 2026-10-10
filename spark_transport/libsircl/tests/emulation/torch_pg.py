#!/usr/bin/env python3
"""PyTorch ProcessGroupNCCL on libsircl in GPU emulation: torch.distributed unmodified, through LD_PRELOAD.

``torch_pg.py --launch --library build/libsircl.so --world 2`` starts one process per rank on the same
GPU with ``LD_PRELOAD`` set to the library and its emulation transport, so torch's ``libtorch_cuda.so``
binds its NCCL calls to libsircl (SONAME ``libnccl.so.2``). Each rank calls
``init_process_group("nccl", device_id=...)`` (eager communicator creation), then all-reduce of bf16, fp16
and fp32 (sum), of int64 (sum), fp32 (max) and bf16 (avg), ``all_gather_object``,
``broadcast_object_list``, an all-reduce on ``new_group([0, 1])``, ``send``/``recv``,
``batch_isend_irecv`` and ``all_to_all_single`` (two ranks), ``barrier``, ``all_gather_into_tensor``,
``reduce_scatter_tensor``,
``broadcast``, ``reduce``, a CUDA graph of an all-reduce replayed twice, and ``destroy_process_group``.
Every result is
compared with the host reference bit for bit; the runtime NCCL version torch reports must be libsircl's
22705 and the process must have libsircl mapped. Prints one line per check and rank.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def rank_main(args) -> int:
    import torch
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    results = []

    def check(name, got, want):
        same = got.shape == want.shape and torch.equal(got.cpu().contiguous().view(torch.uint8),
                                                       want.contiguous().view(torch.uint8))
        results.append((name, bool(same), "" if same else "differs from the host reference"))

    def inputs(count, dtype, seed):
        return [torch.randn(count, generator=torch.Generator().manual_seed(seed * 1009 + r)).to(dtype)
                for r in range(world)]

    def total(values):
        acc = values[0].float().clone()
        for value in values[1:]:
            acc += value.float()
        return acc.to(values[0].dtype)

    started = time.perf_counter()
    dist.init_process_group("nccl", device_id=device)
    results.append(("init_process_group(nccl, device_id)", True, f"{time.perf_counter() - started:.2f} s"))
    import ctypes

    # The process's global symbol scope (LD_PRELOAD first), which libtorch_cuda's NCCL references bind to.
    code = ctypes.c_int(0)
    ctypes.CDLL(None).ncclGetVersion(ctypes.byref(code))
    maps = Path("/proc/self/maps").read_text()
    results.append(("libsircl mapped and bound", "libsircl" in maps and code.value == 22705,
                    f"ncclGetVersion in the global scope reports {code.value}"))
    for index, (dtype, count) in enumerate(((torch.bfloat16, 4096), (torch.float16, 300001), (torch.float32, 7),
                                            (torch.bfloat16, (5 << 20) // 2))):
        values = inputs(count, dtype, 100 + index)
        x = values[rank].to(device)
        dist.all_reduce(x)
        check(f"all_reduce {dtype} {count}", x, total(values))
    # Other datatypes and ops, which libsircl reduces through its fold pack.
    ints = [torch.randint(-(1 << 40), 1 << 40, (5000,), generator=torch.Generator().manual_seed(150 + r),
                          dtype=torch.int64) for r in range(world)]
    x = ints[rank].to(device)
    dist.all_reduce(x)
    check("all_reduce int64 5000 (sum)", x, sum(ints[1:], ints[0].clone()))
    values = inputs(3001, torch.float32, 151)
    x = values[rank].to(device)
    dist.all_reduce(x, op=dist.ReduceOp.MAX)
    check("all_reduce float32 3001 (max)", x, torch.stack(values).amax(dim=0))
    values = inputs(4096, torch.bfloat16, 152)
    x = values[rank].to(device)
    dist.all_reduce(x, op=dist.ReduceOp.AVG)
    acc = values[0].float().clone()
    for value in values[1:]:
        acc += value.float()
    check("all_reduce bfloat16 4096 (avg)", x, (acc / world).to(torch.bfloat16))
    gathered = [None] * world
    dist.all_gather_object(gathered, {"rank": rank, "text": "x" * (100 + rank)})
    results.append(("all_gather_object", gathered == [{"rank": r, "text": "x" * (100 + r)} for r in range(world)],
                    ""))
    objects = [{"from": world - 1, "data": list(range(50))}] if rank == world - 1 else [None]
    dist.broadcast_object_list(objects, src=world - 1)
    results.append(("broadcast_object_list", objects == [{"from": world - 1, "data": list(range(50))}], ""))
    # Point-to-point: with eager initialization (device_id) torch carries send, receive and all-to-all
    # on the group's own communicator, which libsircl carries between ranks when it has two ranks.
    # new_group splits the group's communicator (ncclCommSplit).
    pair = dist.new_group([0, 1])
    if rank in (0, 1):
        values = inputs(2048, torch.bfloat16, 159)
        x = values[rank].to(device)
        dist.all_reduce(x, group=pair)
        check("all_reduce bf16 2048 on new_group([0, 1])", x, total(values[:2]))
    try:
        splits = dist.distributed_c10d._get_default_group()._get_backend(device).comm_split_count()
    except Exception as error:  # noqa: BLE001
        splits = f"unavailable ({type(error).__name__})"
    results.append(("new_group's communicator", True, f"torch's comm_split_count {splits}"))
    if world == 2:
        values = inputs(1000, torch.float32, 160)
        if rank == 0:
            dist.send(values[0].to(device), dst=1)
        else:
            x = torch.empty(1000, dtype=torch.float32, device=device)
            dist.recv(x, src=0)
            check("send and recv fp32 1000 from rank 0 to rank 1", x, values[0])
        values = inputs(4099, torch.bfloat16, 161)
        x = values[rank].to(device)
        y = torch.empty_like(x)
        other = 1 - rank
        for work in dist.batch_isend_irecv([dist.P2POp(dist.isend, x, other), dist.P2POp(dist.irecv, y, other)]):
            work.wait()
        check("batch_isend_irecv bf16 4099 between ranks 0 and 1", y, values[other])
        values = inputs(world * 2048, torch.float16, 162)
        out = torch.empty(world * 2048, dtype=torch.float16, device=device)
        dist.all_to_all_single(out, values[rank].to(device))
        want = torch.cat([values[source][rank * 2048:(rank + 1) * 2048] for source in range(world)])
        check("all_to_all_single fp16 2048 per rank", out, want)
    dist.barrier()
    results.append(("barrier", True, ""))
    values = inputs(3000, torch.bfloat16, 200)
    out = torch.empty(world * 3000, dtype=torch.bfloat16, device=device)
    dist.all_gather_into_tensor(out, values[rank].to(device))
    check("all_gather_into_tensor bf16", out, torch.cat(values))
    values = inputs(world * 4096, torch.float32, 201)
    out = torch.empty(4096, dtype=torch.float32, device=device)
    dist.reduce_scatter_tensor(out, values[rank].to(device))
    check("reduce_scatter_tensor fp32", out, total(values)[rank * 4096:(rank + 1) * 4096])
    values = inputs(1000, torch.float16, 202)
    x = values[rank].to(device)
    dist.broadcast(x, src=world - 1)
    check("broadcast fp16 from the last rank", x, values[world - 1])
    values = inputs(2048, torch.bfloat16, 203)
    x = values[rank].to(device)
    dist.reduce(x, dst=0)
    if rank == 0:
        check("reduce bf16 to rank 0", x, total(values))
    static = torch.zeros(8192, dtype=torch.bfloat16, device=device)
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        dist.all_reduce(static)   # warm-up outside the capture, as torch's graph recipes do
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        dist.all_reduce(static)
    for replay, seed in enumerate((300, 301)):
        values = inputs(8192, torch.bfloat16, seed)
        static.copy_(values[rank].to(device))
        graph.replay()
        torch.cuda.synchronize()
        check(f"CUDA graph all_reduce replay {replay + 1}", static, total(values))
    dist.destroy_process_group()
    results.append(("destroy_process_group", True, ""))
    Path(args.out).write_text(json.dumps({"rank": rank, "checks": results}))
    return 0 if all(ok for _, ok, _ in results) else 1


def launch(args) -> int:
    work = Path(tempfile.mkdtemp(prefix="sircl-ccl-torch-"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    processes = []
    for rank in range(args.world):
        env = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(args.world), LOCAL_RANK=str(rank),
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), LD_PRELOAD=str(Path(args.library).resolve()),
                   LIBSIRCL_TRANSPORT="emulation", SIRCL_EMU_FABRIC=f"/sircl-emu-torch-{os.getpid()}",
                   LIBSIRCL_RECEIPT=str(work / "receipt"), TORCH_NCCL_ASYNC_ERROR_HANDLING="1",
                   CUDA_MODULE_LOADING="EAGER")
        log = open(work / f"rank{rank}.log", "w")
        processes.append((subprocess.Popen([sys.executable, __file__, "--rank-main", "--out",
                                            str(work / f"rank{rank}.json")], env=env, stdout=log,
                                           stderr=subprocess.STDOUT), log))
    failed = 0
    for rank, (process, log) in enumerate(processes):
        try:
            process.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()
        path = work / f"rank{rank}.json"
        if not path.exists():
            print(f"FAIL rank {rank}: no result; log follows\n{(work / f'rank{rank}.log').read_text()[-4000:]}")
            failed += 1
            continue
        for name, ok, detail in json.loads(path.read_text())["checks"]:
            failed += 0 if ok else 1
            print(f"{'PASS' if ok else 'FAIL'} rank {rank} {name}{': ' + detail if detail else ''}")
    fabric = Path("/dev/shm") / f"sircl-emu-torch-{os.getpid()}"
    if fabric.exists():
        fabric.unlink()
    for receipt in sorted(work.glob("receipt.rank*.json")):
        print(f"receipt {receipt.name}: {receipt.read_text().strip()[:600]}")
    print(f"{failed} failed; work {work}")
    return 1 if failed else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--rank-main", action="store_true")
    parser.add_argument("--library", default="build/libsircl.so")
    parser.add_argument("--world", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--out", default="")
    options = parser.parse_args()
    sys.exit(rank_main(options) if options.rank_main else launch(options))
