#!/usr/bin/env python3
"""Validate the switchless NCCL-IB bridge used by TP4/DCP2.

The probe exercises both communication scopes that a TP4/DCP2 vLLM process
creates:

* one four-rank TP communicator, forced to the physical ring; and
* two adjacent two-rank DCP communicators, [0, 1] and [2, 3].

Every row checks values as well as latency.  A fast result is not accepted if
the rank-major all-gather layout differs from torch.distributed semantics.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    iterations: int


CASES = (
    Case("owner_topk_q1", (1, 2, 2048), torch.int32, 500),
    Case("query_q1", (1, 16, 576), torch.bfloat16, 500),
    Case("lse_q1", (1, 32), torch.float32, 500),
    Case("query_q40", (40, 16, 576), torch.bfloat16, 100),
    Case("query_q4096", (4096, 16, 576), torch.bfloat16, 10),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def timed_pair_all_gather(
    case: Case,
    *,
    rank: int,
    pair_ranks: list[int],
    pair_group: dist.ProcessGroup,
) -> dict[str, object]:
    source = torch.full(case.shape, rank + 1, dtype=case.dtype, device="cuda")
    output_shape = (len(pair_ranks) * case.shape[0], *case.shape[1:])
    output = torch.empty(output_shape, dtype=case.dtype, device="cuda")

    for _ in range(min(20, case.iterations)):
        dist.all_gather_into_tensor(output, source, group=pair_group)
    torch.cuda.synchronize()
    dist.barrier(group=pair_group)

    samples_us: list[float] = []
    for _ in range(case.iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_gather_into_tensor(output, source, group=pair_group)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    expected = torch.cat(
        [
            torch.full(case.shape, peer + 1, dtype=case.dtype, device="cuda")
            for peer in pair_ranks
        ],
        dim=0,
    )
    correct = bool(torch.equal(output, expected))
    return {
        "scope": f"dcp_pair_{pair_ranks[0]}_{pair_ranks[1]}",
        "case": case.name,
        "bytes_per_rank": source.numel() * source.element_size(),
        "iterations": case.iterations,
        "p50_us": statistics.median(samples_us),
        "p99_us": percentile(samples_us, 0.99),
        "correct": correct,
    }


def timed_world_all_reduce(rank: int) -> dict[str, object]:
    elements = 6144
    tensor = torch.full(
        (elements,), rank + 1, dtype=torch.bfloat16, device="cuda"
    )
    for _ in range(20):
        tensor.fill_(rank + 1)
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    dist.barrier()

    samples_us: list[float] = []
    for _ in range(500):
        tensor.fill_(rank + 1)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        dist.all_reduce(tensor)
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    correct = bool(torch.all(tensor == 10).item())
    return {
        "scope": "tp4_ring",
        "case": "all_reduce_6144_bf16",
        "bytes_per_rank": tensor.numel() * tensor.element_size(),
        "iterations": 500,
        "p50_us": statistics.median(samples_us),
        "p99_us": percentile(samples_us, 0.99),
        "correct": correct,
    }


def graph_pair_all_gather(
    case: Case,
    *,
    rank: int,
    pair_ranks: list[int],
    pair_group: dist.ProcessGroup,
    replays: int = 2_000,
) -> dict[str, object]:
    source = torch.full(case.shape, rank + 1, dtype=case.dtype, device="cuda")
    output_shape = (len(pair_ranks) * case.shape[0], *case.shape[1:])
    output = torch.empty(output_shape, dtype=case.dtype, device="cuda")
    capture_stream = torch.cuda.Stream()

    # Initialize all lazy NCCL state before capture.
    with torch.cuda.stream(capture_stream):
        for _ in range(20):
            dist.all_gather_into_tensor(output, source, group=pair_group)
    torch.cuda.synchronize()
    dist.barrier(group=pair_group)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        dist.all_gather_into_tensor(output, source, group=pair_group)
    torch.cuda.synchronize()
    dist.barrier(group=pair_group)

    started_ns = time.perf_counter_ns()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter_ns() - started_ns) / 1000.0

    expected = torch.cat(
        [
            torch.full(case.shape, peer + 1, dtype=case.dtype, device="cuda")
            for peer in pair_ranks
        ],
        dim=0,
    )
    correct = bool(torch.equal(output, expected))
    return {
        "scope": f"dcp_pair_{pair_ranks[0]}_{pair_ranks[1]}",
        "case": f"{case.name}_graph",
        "bytes_per_rank": source.numel() * source.element_size(),
        "iterations": replays,
        "mean_us": elapsed_us / replays,
        "correct": correct,
    }


def emit(row: dict[str, object], rank: int) -> None:
    fields = " ".join(f"{key}={value}" for key, value in row.items())
    print(f"RESULT rank={rank} {fields}", flush=True)
    if not row["correct"]:
        raise RuntimeError(f"collective validation failed: {row}")


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "4"))
    if world_size != 4:
        raise ValueError(f"this probe requires WORLD_SIZE=4, got {world_size}")

    head_ip = os.environ.get("HEAD_IP")
    if not head_ip:
        raise RuntimeError(
            "HEAD_IP must be set to rank 0's control-plane IP address"
        )

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl",
        init_method=(
            f"tcp://{head_ip}:"
            f"{os.environ.get('MASTER_PORT', '29641')}"
        ),
        rank=rank,
        world_size=world_size,
    )

    # Every rank must create groups in the same global order.
    groups = [
        (pair, dist.new_group(pair, backend="nccl"))
        for pair in ([0, 1], [2, 3])
    ]
    pair_ranks, pair_group = next(
        (pair, group) for pair, group in groups if rank in pair
    )

    if rank == 0:
        print(
            "CONFIG"
            f" nccl_version={torch.cuda.nccl.version()}"
            f" NCCL_NET={os.environ.get('NCCL_NET')}"
            f" NCCL_ALGO={os.environ.get('NCCL_ALGO')}"
            f" NCCL_SKIP_TREE_CONNECT="
            f"{os.environ.get('NCCL_SKIP_TREE_CONNECT')}"
            f" NCCL_IB_HCA={os.environ.get('NCCL_IB_HCA')}"
            f" NCCL_IB_GID_INDEX={os.environ.get('NCCL_IB_GID_INDEX')}",
            flush=True,
        )

    emit(timed_world_all_reduce(rank), rank)
    for case in CASES:
        emit(
            timed_pair_all_gather(
                case,
                rank=rank,
                pair_ranks=pair_ranks,
                pair_group=pair_group,
            ),
            rank,
        )
    for case in CASES[:3]:
        emit(
            graph_pair_all_gather(
                case,
                rank=rank,
                pair_ranks=pair_ranks,
                pair_group=pair_group,
            ),
            rank,
        )

    dist.barrier()
    if rank == 0:
        print("PASS switchless TP4/DCP2 NCCL-IB bridge", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
