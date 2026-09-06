#!/usr/bin/env python3
"""Measure isolated NCCL CUDA-graph all-reduce residency by query size.

The probe creates one four-rank NCCL communicator, captures one BF16
all-reduce for each requested ``[Q, hidden]`` tensor, and records one CUDA
event pair around every graph replay. A zero-valued operand remains stable
across in-place replays, so input-reset work is excluded from the measured
region. A separate nonzero collective validates reduction correctness.

Every rank writes its own receipt. Compare ranks by taking the slowest-rank
statistic for each query size; CUDA event clocks are not synchronized across
hosts.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "sparkring-nccl-graph-allreduce/v1"


def parse_query_rows(value: str) -> tuple[int, ...]:
    """Parse a strictly increasing comma-separated query-row list."""

    try:
        rows = tuple(int(piece) for piece in value.split(",") if piece)
    except ValueError as error:
        raise ValueError("query rows must be comma-separated integers") from error
    if (
        not rows
        or any(row <= 0 for row in rows)
        or tuple(sorted(set(rows))) != rows
    ):
        raise ValueError("query rows must be positive, unique, and increasing")
    return rows


def nearest_rank(samples: Sequence[float], fraction: float) -> float:
    """Return an observed nearest-rank percentile from nonempty samples."""

    if not samples:
        raise ValueError("percentile samples must not be empty")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("percentile fraction must lie in (0, 1]")
    ordered = sorted(samples)
    rank = max(1, math.ceil(fraction * len(ordered)))
    return ordered[rank - 1]


def summarize(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize one rank's individual replay observations in microseconds."""

    if not samples:
        raise ValueError("timing samples must not be empty")
    return {
        "count": len(samples),
        "min_us": min(samples),
        "p50_us": statistics.median(samples),
        "p95_us": nearest_rank(samples, 0.95),
        "p99_us": nearest_rank(samples, 0.99),
        "max_us": max(samples),
    }


def measure_case(
    *,
    torch_module: Any,
    dist_module: Any,
    rank: int,
    query_rows: int,
    hidden_size: int,
    warmups: int,
    iterations: int,
    all_reduce: Callable[[Any, Any | None, Any | None], Any],
) -> dict[str, Any]:
    """Capture, validate, and time one NCCL graph all-reduce shape."""

    torch = torch_module
    dist = dist_module
    validation = torch.full(
        (query_rows, hidden_size),
        rank + 1,
        dtype=torch.bfloat16,
        device="cuda",
    )
    validation_result = all_reduce(validation, None, None)
    validation_correct = bool(torch.all(validation_result == 10).item())
    if not validation_correct:
        raise RuntimeError(f"NCCL validation failed at Q={query_rows}")

    operand = torch.zeros_like(validation)
    graph_output = torch.empty_like(operand)
    capture_stream = torch.cuda.Stream()
    with torch.cuda.stream(capture_stream):
        for _ in range(20):
            all_reduce(operand, graph_output, capture_stream)
    capture_stream.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        graph_result = all_reduce(operand, graph_output, capture_stream)
    torch.cuda.synchronize()
    dist.barrier()

    with torch.cuda.stream(capture_stream):
        for _ in range(warmups):
            graph.replay()
    capture_stream.synchronize()
    dist.barrier()

    event_pairs = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(iterations)
    ]
    samples_us: list[float] = []
    with torch.cuda.stream(capture_stream):
        for start, stop in event_pairs:
            start.record(capture_stream)
            graph.replay()
            stop.record(capture_stream)
            stop.synchronize()
            samples_us.append(float(start.elapsed_time(stop)) * 1000.0)

    graph_correct = bool(torch.all(graph_result == 0).item())
    if not graph_correct:
        raise RuntimeError(f"NCCL graph replay changed the zero operand at Q={query_rows}")
    dist.barrier()
    payload_bytes = query_rows * hidden_size * 2
    return {
        "query_rows": query_rows,
        "hidden_size": hidden_size,
        "dtype": "bfloat16",
        "payload_bytes": payload_bytes,
        "warmups": warmups,
        "timing": summarize(samples_us),
        "raw_samples_us": samples_us,
        "validation_correct": validation_correct,
        "graph_correct": graph_correct,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, default=int(os.getenv("RANK", "0")))
    parser.add_argument(
        "--world-size",
        type=int,
        default=int(os.getenv("WORLD_SIZE", "4")),
    )
    parser.add_argument("--head-ip", default=os.getenv("HEAD_IP"))
    parser.add_argument(
        "--master-port",
        type=int,
        default=int(os.getenv("MASTER_PORT", "12170")),
    )
    parser.add_argument("--query-rows", default="8,16,32,64,128")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--implementation",
        choices=("torch", "pynccl"),
        default="torch",
    )
    parser.add_argument(
        "--nccl-library",
        default=os.getenv("VLLM_NCCL_SO_PATH"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.world_size != 4 or not 0 <= args.rank < args.world_size:
        raise SystemExit("the NCCL graph probe requires ranks 0-3 at world size 4")
    if not args.head_ip:
        raise SystemExit("--head-ip or HEAD_IP is required")
    if args.hidden_size <= 0 or args.warmups < 0 or args.iterations <= 0:
        raise SystemExit("hidden size and iterations must be positive")
    try:
        query_rows = parse_query_rows(args.query_rows)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    import torch
    import torch.distributed as dist

    torch.cuda.set_device(0)
    dist.init_process_group(
        backend="nccl" if args.implementation == "torch" else "gloo",
        init_method=f"tcp://{args.head_ip}:{args.master_port}",
        rank=args.rank,
        world_size=args.world_size,
    )
    pynccl = None
    if args.implementation == "torch":
        def all_reduce(inp: Any, out: Any | None, stream: Any | None) -> Any:
            del out, stream
            dist.all_reduce(inp)
            return inp
    else:
        if not args.nccl_library:
            raise SystemExit("--nccl-library is required for pynccl")
        from vllm.distributed.device_communicators.pynccl import (
            PyNcclCommunicator,
        )

        pynccl = PyNcclCommunicator(
            group=dist.group.WORLD,
            device=torch.device("cuda:0"),
            library_path=args.nccl_library,
        )

        def all_reduce(inp: Any, out: Any | None, stream: Any | None) -> Any:
            return pynccl.all_reduce(inp, out_tensor=out, stream=stream)

    try:
        cases = [
            measure_case(
                torch_module=torch,
                dist_module=dist,
                rank=args.rank,
                query_rows=row,
                hidden_size=args.hidden_size,
                warmups=args.warmups,
                iterations=args.iterations,
                all_reduce=all_reduce,
            )
            for row in query_rows
        ]
        document = {
            "schema": SCHEMA,
            "rank": args.rank,
            "world_size": args.world_size,
            "head_ip": args.head_ip,
            "master_port": args.master_port,
            "implementation": args.implementation,
            "nccl_version": list(torch.cuda.nccl.version()),
            "environment": {
                name: os.getenv(name)
                for name in (
                    "NCCL_NET",
                    "NCCL_ALGO",
                    "NCCL_PROTO",
                    "NCCL_IB_HCA",
                    "NCCL_IB_GID_INDEX",
                    "NCCL_IB_SUBNET_AWARE_ROUTING",
                    "NCCL_IB_MERGE_NICS",
                    "NCCL_CROSS_NIC",
                    "NCCL_MIN_NCHANNELS",
                    "NCCL_MAX_NCHANNELS",
                    "LD_PRELOAD",
                    "VLLM_NCCL_SO_PATH",
                )
            },
            "timing_scope": "CUDA events around one synchronized graph replay",
            "cases": cases,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        for case in cases:
            timing = case["timing"]
            print(
                "NCCL_GRAPH_ALLREDUCE"
                f" rank={args.rank} q={case['query_rows']}"
                f" bytes={case['payload_bytes']}"
                f" p50_us={timing['p50_us']:.3f}"
                f" p95_us={timing['p95_us']:.3f}"
                f" correct={str(case['graph_correct']).lower()}",
                flush=True,
            )
    finally:
        if pynccl is not None:
            pynccl.destroy()
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
