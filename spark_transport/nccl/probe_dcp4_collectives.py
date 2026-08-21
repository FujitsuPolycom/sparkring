#!/usr/bin/env python3
"""Validate the switchless NCCL-IB bridge for TP4/DCP4.

The probe exercises the four-rank communicator and the collective families
used by GLM-5.2's DCP4 attention path:

* all-reduce for the remaining TP fallback;
* all-gather for owner top-k, query, and LSE payloads; and
* reduce-scatter for the attention output.

Every case validates values. Representative decode and prefill shapes are
also captured into CUDA graphs and replayed before the bridge is admitted.
"""

from __future__ import annotations

import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class GatherCase:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    iterations: int


@dataclass(frozen=True)
class ReduceScatterCase:
    name: str
    q: int
    head_dimension: int
    iterations: int


GATHER_CASES = (
    GatherCase("owner_topk_q1", (1, 2, 2048), torch.int32, 300),
    GatherCase("query_q1", (1, 16, 576), torch.bfloat16, 300),
    GatherCase("lse_q1", (1, 64), torch.float32, 300),
    GatherCase("query_q40", (40, 16, 576), torch.bfloat16, 100),
    GatherCase("query_q512", (512, 16, 576), torch.bfloat16, 30),
    GatherCase("query_q4096", (4096, 16, 576), torch.bfloat16, 5),
)

REDUCE_SCATTER_CASES = (
    ReduceScatterCase("output_q1_d256", 1, 256, 300),
    ReduceScatterCase("output_q5_d256", 5, 256, 300),
    ReduceScatterCase("output_q40_d256", 40, 256, 100),
    ReduceScatterCase("output_q512_d512", 512, 512, 20),
)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _time_cuda(
    operation: Callable[[], None],
    *,
    warmups: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmups):
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    samples_us: list[float] = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    return samples_us


def timed_world_all_reduce(rank: int) -> dict[str, object]:
    tensor = torch.full(
        (6144,), rank + 1, dtype=torch.bfloat16, device="cuda"
    )

    def operation() -> None:
        tensor.fill_(rank + 1)
        dist.all_reduce(tensor)

    samples_us = _time_cuda(operation, warmups=20, iterations=300)
    return {
        "scope": "dcp4_world",
        "operation": "all_reduce",
        "case": "tp_6144_bf16",
        "bytes_per_rank": tensor.numel() * tensor.element_size(),
        "iterations": len(samples_us),
        "p50_us": statistics.median(samples_us),
        "p99_us": percentile(samples_us, 0.99),
        "correct": bool(torch.all(tensor == 10).item()),
    }


def _gather_tensors(
    case: GatherCase, rank: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    source = torch.full(
        case.shape, rank + 1, dtype=case.dtype, device="cuda"
    )
    output = torch.empty(
        (4 * case.shape[0], *case.shape[1:]),
        dtype=case.dtype,
        device="cuda",
    )
    expected = torch.cat(
        [
            torch.full(
                case.shape, peer + 1, dtype=case.dtype, device="cuda"
            )
            for peer in range(4)
        ],
        dim=0,
    )
    return source, output, expected


def timed_world_all_gather(
    case: GatherCase, rank: int
) -> dict[str, object]:
    source, output, expected = _gather_tensors(case, rank)

    def operation() -> None:
        dist.all_gather_into_tensor(output, source)

    samples_us = _time_cuda(
        operation,
        warmups=min(20, case.iterations),
        iterations=case.iterations,
    )
    return {
        "scope": "dcp4_world",
        "operation": "all_gather",
        "case": case.name,
        "bytes_per_rank": source.numel() * source.element_size(),
        "iterations": len(samples_us),
        "p50_us": statistics.median(samples_us),
        "p99_us": percentile(samples_us, 0.99),
        "correct": bool(torch.equal(output, expected)),
    }


def _reduce_scatter_tensors(
    case: ReduceScatterCase, rank: int
) -> tuple[torch.Tensor, torch.Tensor]:
    del rank
    output_shape = (16, case.q, case.head_dimension)
    input_tensor = torch.empty(
        (64, case.q, case.head_dimension),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = torch.empty(output_shape, dtype=torch.bfloat16, device="cuda")
    return input_tensor, output


def timed_world_reduce_scatter(
    case: ReduceScatterCase, rank: int
) -> dict[str, object]:
    input_tensor, output = _reduce_scatter_tensors(case, rank)

    def operation() -> None:
        input_tensor.fill_(rank + 1)
        dist.reduce_scatter_tensor(output, input_tensor)

    samples_us = _time_cuda(
        operation,
        warmups=min(20, case.iterations),
        iterations=case.iterations,
    )
    return {
        "scope": "dcp4_world",
        "operation": "reduce_scatter",
        "case": case.name,
        "bytes_per_rank": input_tensor.numel() * input_tensor.element_size(),
        "iterations": len(samples_us),
        "p50_us": statistics.median(samples_us),
        "p99_us": percentile(samples_us, 0.99),
        "correct": bool(torch.all(output == 10).item()),
    }


def _graph_replay(
    operation: Callable[[], None],
    *,
    replays: int,
) -> float:
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(20):
            operation()
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        operation()
    torch.cuda.synchronize()
    dist.barrier()

    started_ns = time.perf_counter_ns()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started_ns) / 1000.0 / replays


def graph_world_all_gather(
    case: GatherCase, rank: int, *, replays: int
) -> dict[str, object]:
    source, output, expected = _gather_tensors(case, rank)

    def operation() -> None:
        dist.all_gather_into_tensor(output, source)

    mean_us = _graph_replay(operation, replays=replays)
    return {
        "scope": "dcp4_world",
        "operation": "all_gather_graph",
        "case": case.name,
        "bytes_per_rank": source.numel() * source.element_size(),
        "iterations": replays,
        "mean_us": mean_us,
        "correct": bool(torch.equal(output, expected)),
    }


def graph_world_reduce_scatter(
    case: ReduceScatterCase, rank: int, *, replays: int
) -> dict[str, object]:
    input_tensor, output = _reduce_scatter_tensors(case, rank)
    input_tensor.fill_(rank + 1)

    def operation() -> None:
        dist.reduce_scatter_tensor(output, input_tensor)

    mean_us = _graph_replay(operation, replays=replays)
    return {
        "scope": "dcp4_world",
        "operation": "reduce_scatter_graph",
        "case": case.name,
        "bytes_per_rank": input_tensor.numel() * input_tensor.element_size(),
        "iterations": replays,
        "mean_us": mean_us,
        "correct": bool(torch.all(output == 10).item()),
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
            f"{os.environ.get('MASTER_PORT', '29642')}"
        ),
        rank=rank,
        world_size=world_size,
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
    for case in GATHER_CASES:
        emit(timed_world_all_gather(case, rank), rank)
    for case in REDUCE_SCATTER_CASES:
        emit(timed_world_reduce_scatter(case, rank), rank)

    graph_gather_replays = {
        "owner_topk_q1": 2_000,
        "query_q1": 2_000,
        "lse_q1": 2_000,
        "query_q40": 500,
        "query_q512": 100,
    }
    for case in GATHER_CASES:
        replays = graph_gather_replays.get(case.name)
        if replays is not None:
            emit(
                graph_world_all_gather(case, rank, replays=replays),
                rank,
            )

    graph_scatter_replays = {
        "output_q1_d256": 2_000,
        "output_q40_d256": 500,
        "output_q512_d512": 100,
    }
    for case in REDUCE_SCATTER_CASES:
        replays = graph_scatter_replays.get(case.name)
        if replays is not None:
            emit(
                graph_world_reduce_scatter(case, rank, replays=replays),
                rank,
            )

    dist.barrier()
    if rank == 0:
        print("PASS switchless TP4/DCP4 NCCL-IB bridge", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
