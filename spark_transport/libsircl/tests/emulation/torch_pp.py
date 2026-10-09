#!/usr/bin/env python3
"""vLLM's pipeline-parallel exchange through torch.distributed on libsircl, in GPU emulation.

vLLM 0.19.1's V1 worker sends a pipeline stage's intermediate tensors with
``GroupCoordinator.isend_tensor_dict`` and receives them with ``irecv_tensor_dict``: one
``torch.distributed.isend`` per tensor to the next stage and one ``irecv`` from the previous, on the
pipeline group's device group, a ``new_group`` of a default group initialized without ``device_id``. torch
creates communicators lazily there: a group's own on its first collective, and for an unbatched send or
receive on a group of more than two ranks a two-rank communicator of the pair. ``torch_pp.py --launch
--library build/libsircl.so`` starts one process per rank on the one GPU with libsircl through
``LD_PRELOAD`` and the emulation transport, in one of two shapes:

- ``--shape chain`` (default, ``--world`` ranks, 4 by default): one pipeline group of every rank; stage r
  receives from r - 1 and sends to r + 1 for several steps, each stage pair at its own pace; every rank at
  its own position (``LIBSIRCL_POSITION``) with the chain order of all ranks (``LIBSIRCL_CHAIN_ORDER``).
- ``--shape tp4pp2``: eight ranks with exactly the settings ``tools/site_routes.py --layout ring:8 --lanes
  2`` emits for their positions (tests/data/site_routes_ring8_l2.json: position, route map, chain order,
  forward windows, ring plan), as vLLM lays out TP 4 x PP 2: an all-reduce on the default group (the
  eight-rank communicator), tensor-parallel groups ``new_group([0, 1, 2, 3])`` and ``new_group([4, 5, 6,
  7])``, and pipeline groups ``new_group([i, i + 4])`` for i 0-3, whose ranks no cable joins. Several
  microbatches: the first stage's tensor-parallel all-reduce, its sends of a tensor dictionary to its
  pipeline peer, the second stage's receives and its own tensor-parallel all-reduce.
- ``--shape eager`` (``--world`` ranks, 4 by default): the default group initialized eagerly
  (``device_id``), so torch creates its communicator at ``init_process_group``, with libsircl's
  point-to-point channels on (``LIBSIRCL_P2P_CHANNELS=on``). An all-reduce; ``batch_isend_irecv`` of a ring
  exchange (send to r + 1, receive from r - 1), which torch issues on the group's communicator; each rank
  sending to every other rank in one batch; and unbatched ``isend`` and ``irecv`` along the chain, which
  torch issues on whichever communicator its version selects for a pair.

Passes when every all-reduce and every received tensor is exact and every rank exits 0. For ``chain`` and
``tp4pp2``, every libsircl communicator that carried point-to-point traffic has two ranks (its receipt);
under ring:8, each pipeline pair's ring lanes keep a window, since its lanes cross relays. For ``eager``,
every rank's communicator of ``--world`` ranks carried point-to-point traffic as channel items (its
receipt's channel counts), and every two-rank communicator that carried some did so as pair exchanges.

Status: research-only test infrastructure.
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

ROUTES = Path(__file__).resolve().parent.parent / "data" / "site_routes_ring8_l2.json"


def rank_main(args) -> int:
    import torch
    import torch.distributed as dist

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(0)
    if args.shape == "eager":
        dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    else:
        dist.init_process_group("nccl")  # lazily, as vLLM's init_distributed_environment does
    checks = []

    def check(name, ok):
        checks.append({"name": name, "ok": bool(ok)})

    def filled(size, value):
        return torch.full((size,), value % 251, dtype=torch.uint8, device="cuda")

    if args.shape == "eager":
        x = torch.full((4096,), float(rank + 1), dtype=torch.float32, device="cuda")
        dist.all_reduce(x)
        check("all-reduce on the eagerly initialized default group", bool((x == float(world * (world + 1) // 2)).all()))
        nxt, prv = (rank + 1) % world, (rank - 1) % world
        for step in range(args.steps):
            size = (3 << 20) + 4096 * step + (7 if step % 2 else 0)
            got = torch.empty(size, dtype=torch.uint8, device="cuda")
            ops = [dist.P2POp(dist.isend, filled(size, rank + step), nxt), dist.P2POp(dist.irecv, got, prv)]
            for request in dist.batch_isend_irecv(ops):
                request.wait()
            check(f"batched ring exchange {step}", torch.equal(got.cpu(), filled(size, prv + step).cpu()))
            sizes = {peer: (1 << 20) + 977 * (rank + peer + step) for peer in range(world)}
            inbound = {peer: torch.empty((1 << 20) + 977 * (rank + peer + step), dtype=torch.uint8, device="cuda")
                       for peer in range(world) if peer != rank}
            ops = [dist.P2POp(dist.isend, filled(sizes[peer], 3 * rank + peer + step), peer)
                   for peer in range(world) if peer != rank]
            ops += [dist.P2POp(dist.irecv, inbound[peer], peer) for peer in range(world) if peer != rank]
            for request in dist.batch_isend_irecv(ops):
                request.wait()
            check(f"batched exchange with every rank {step}",
                  all(torch.equal(inbound[peer].cpu(), filled(inbound[peer].numel(), 3 * peer + rank + step).cpu())
                      for peer in inbound))
            time.sleep(0.03 * ((rank * 7 + step * 3) % 5))
            chain = (1 << 20) + 13 * step
            if rank > 0:
                y = torch.empty(chain, dtype=torch.uint8, device="cuda")
                dist.irecv(y, src=rank - 1).wait()
                check(f"unbatched step {step} from rank {rank - 1}", torch.equal(y.cpu(), filled(chain, rank - 1 + step).cpu()))
            if rank < world - 1:
                dist.isend(filled(chain, rank + step), dst=rank + 1).wait()
    elif args.shape == "chain":
        pp = dist.new_group(list(range(world)), backend="nccl")
        for step in range(args.steps):
            time.sleep(0.05 * ((rank * 7 + step * 3) % 5))  # each stage pair at its own pace
            size = (1 << 20) + 4096 * step + (13 if step % 2 else 0)
            if rank > 0:
                x = torch.empty(size, dtype=torch.uint8, device="cuda")
                dist.irecv(x, src=rank - 1, group=pp).wait()
                check(f"step {step} from rank {rank - 1}", torch.equal(x.cpu(), filled(size, rank - 1 + step).cpu()))
            if rank < world - 1:
                dist.isend(filled(size, rank + step), dst=rank + 1, group=pp).wait()
    else:
        x = torch.full((4096,), float(rank + 1), dtype=torch.float32, device="cuda")
        dist.all_reduce(x)
        check("all-reduce on the default group", bool((x == float(world * (world + 1) // 2)).all()))
        tp_groups = [dist.new_group([0, 1, 2, 3], backend="nccl"), dist.new_group([4, 5, 6, 7], backend="nccl")]
        pp_groups = [dist.new_group([i, i + 4], backend="nccl") for i in range(4)]
        tp, pp, stage, peer = tp_groups[rank // 4], pp_groups[rank % 4], rank // 4, (rank + 4) % 8
        sizes = {"hidden": 2 << 20, "residual": 4 << 20, "small": 6 << 10, "odd": 1000003}
        for mb in range(args.steps):
            time.sleep(0.05 * ((rank % 4 * 5 + mb * 3) % 5))  # each pipeline pair at its own pace
            if stage == 0:
                y = torch.full((1 << 20,), float(rank % 4 + mb), dtype=torch.bfloat16, device="cuda")
                dist.all_reduce(y, group=tp)
                check(f"microbatch {mb} first-stage all-reduce", bool((y == float(sum(r + mb for r in range(4)))).all()))
                handles = [dist.isend(filled(n, k + mb + rank), dst=peer, group=pp)
                           for k, n in enumerate(sizes.values())]
                for h in handles:
                    h.wait()
            else:
                got = [torch.empty(n, dtype=torch.uint8, device="cuda") for n in sizes.values()]
                handles = [dist.irecv(t, src=peer, group=pp) for t in got]
                for h in handles:
                    h.wait()
                check(f"microbatch {mb} received from rank {peer}",
                      all(torch.equal(t.cpu(), filled(n, k + mb + peer).cpu())
                          for k, (t, n) in enumerate(zip(got, sizes.values()))))
                y = torch.full((1 << 20,), float(rank % 4 + 2 * mb), dtype=torch.bfloat16, device="cuda")
                dist.all_reduce(y, group=tp)
                check(f"microbatch {mb} second-stage all-reduce",
                      bool((y == float(sum(r + 2 * mb for r in range(4)))).all()))
    torch.cuda.synchronize()
    dist.destroy_process_group()
    Path(args.out).write_text(json.dumps({"rank": rank, "checks": checks}))
    return 0


def launch(args) -> int:
    work = Path(args.work) if args.work else Path(tempfile.mkdtemp(prefix=f"sircl-torch-pp-{args.shape}-"))
    work.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    world = 8 if args.shape == "tp4pp2" else args.world
    channels = {"LIBSIRCL_P2P_CHANNELS": "on"} if args.shape == "eager" else {}
    routes = json.loads(ROUTES.read_text()) if args.shape == "tp4pp2" else None
    processes = []
    for rank in range(world):
        env = dict(os.environ, RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK="0",
                   MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), LD_PRELOAD=str(Path(args.library).resolve()),
                   LIBSIRCL_TRANSPORT="emulation", SIRCL_EMU_FABRIC=f"/sircl-emu-torch-pp-{os.getpid()}",
                   LIBSIRCL_RECEIPT=str(work / "receipt"),
                   CUDA_DEVICE_MAX_CONNECTIONS=os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "32"), **channels)
        if routes:
            env["LIBSIRCL_EMU_LANES"] = str(routes["lanes"])
            env.update(routes["ranks"][rank]["env"])  # every setting as the route planner emits it
        else:
            env.update(LIBSIRCL_EMU_LANES="1", LIBSIRCL_POSITION=str(rank),
                       LIBSIRCL_CHAIN_ORDER=",".join(str(r) for r in range(world)))
        log = open(work / f"rank{rank}.log", "w")
        command = [sys.executable, __file__, "--rank-main", "--shape", args.shape, "--steps", str(args.steps),
                   "--out", str(work / f"rank{rank}.json")]
        processes.append((subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT), log))
    codes = []
    for process, log in processes:
        try:
            codes.append(process.wait(timeout=args.timeout))
        except subprocess.TimeoutExpired:
            process.kill()
            codes.append(process.wait())
        log.close()
    for process, _ in processes:
        for segment in Path("/dev/shm").glob(f"sircl-emu-seg-{process.pid}-*"):
            segment.unlink(missing_ok=True)
    (Path("/dev/shm") / f"sircl-emu-torch-pp-{os.getpid()}").unlink(missing_ok=True)
    problems = [f"rank {r} exit {c}; {work / f'rank{r}.log'}" for r, c in enumerate(codes) if c]
    total = 0
    for rank in range(world):
        path = work / f"rank{rank}.json"
        if not path.exists():
            problems.append(f"rank {rank}: no result")
            continue
        for item in json.loads(path.read_text())["checks"]:
            total += 1
            if not item["ok"]:
                problems.append(f"rank {rank}: {item['name']} differs")
    p2p_comms = 0
    channel_ranks = set()
    for path in work.glob("receipt.rank*.json"):
        receipt = json.loads(path.read_text())
        traffic = receipt["point_to_point"]["sends"] + receipt["point_to_point"]["receives"]
        if args.shape == "eager":
            if traffic and receipt["world"] == world:
                if sum(v["messages"] for v in receipt["channels"].get("sent", {}).values()) and receipt["healthy"]:
                    channel_ranks.add(receipt["rank"])
            elif traffic and receipt["world"] != 2:
                problems.append(f"{path.name}: point-to-point on a communicator of {receipt['world']} ranks")
            elif traffic:
                p2p_comms += 1
            continue
        if traffic:
            p2p_comms += 1
            if receipt["world"] != 2:
                problems.append(f"{path.name}: point-to-point on a communicator of {receipt['world']} ranks")
            # A pipeline pair of ring:8 crosses relays: its ring lanes must keep a window although the
            # layout's ring plan (LIBSIRCL_RING_WINDOW=0) names cables.
            if args.shape == "tp4pp2" and receipt["pair_plan"] and not receipt["forward_windows"]["ring_window_bytes"]:
                problems.append(f"{path.name}: a relayed pair's ring lanes without a window")
    pairs = p2p_comms // 2
    if args.shape == "eager":
        if channel_ranks != set(range(world)):
            problems.append(f"channel items on the {world}-rank communicator of ranks {sorted(channel_ranks)} only")
    else:
        want_pairs = 4 if args.shape == "tp4pp2" else world - 1
        if pairs != want_pairs:
            problems.append(f"{pairs} two-rank communicators carried point-to-point, not {want_pairs}")
    for line in problems:
        print(f"FAIL {line}")
    print(f"torch pipeline exchange, shape {args.shape}, {world} ranks: {total} checks, {pairs} two-rank "
          f"point-to-point communicators, {len(problems)} problems; work {work}")
    return 1 if problems else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--rank-main", action="store_true")
    parser.add_argument("--library", default="")
    parser.add_argument("--shape", choices=("chain", "tp4pp2", "eager"), default="chain")
    parser.add_argument("--world", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--work", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)
    if args.rank_main:
        return rank_main(args)
    return launch(args)


if __name__ == "__main__":
    sys.exit(main())
