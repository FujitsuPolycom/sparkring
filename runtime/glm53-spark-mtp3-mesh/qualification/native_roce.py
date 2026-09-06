"""Collect correctness, timing, and per-QP evidence for TP4 RoCE all-reduce.

Gloo carries metadata and evidence; RoCEnante carries tensor payloads.
Correctness probes and completion retirement execute outside timed intervals.
Short timing samples are diagnostics, not serving-performance qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from typing import Any

import torch
import torch.distributed as dist


def timed_eager(
    runtime: object,
    inp: torch.Tensor,
    out: torch.Tensor,
    samples: int,
) -> list[float]:
    """Return per-call CUDA-event latency in microseconds."""

    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        runtime.all_reduce(inp, out=out)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0)
    return values


def capture(
    runtime: object,
    inp: torch.Tensor,
    out: torch.Tensor,
    operations: int,
) -> torch.cuda.CUDAGraph:
    """Capture sequential collectives with stable addresses."""

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):
            runtime.all_reduce(inp, out=out)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(operations):
            runtime.all_reduce(inp, out=out)
    torch.cuda.synchronize()
    dist.barrier()
    return graph


def _tensor_sha256(value: torch.Tensor) -> str:
    """Hash exact BF16 storage bytes without converting their values."""

    raw = value.detach().contiguous().view(torch.uint16).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _pattern(
    rank: int,
    numel: int,
    variant: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Build rank-specific integer BF16 inputs and their exact TP4 sum."""

    index = torch.arange(numel, dtype=torch.int32, device=device)
    if variant == 0:
        base = index.remainder(17) - 8
        local = (rank + 1) * base + rank
        expected = 10 * base + 6
        local_formula = "((rank + 1) * ((index % 17) - 8)) + rank"
        sum_formula = "10 * ((index % 17) - 8) + 6"
    elif variant == 1:
        base = (index * 5 + 3).remainder(19) - 9
        local = (4 - rank) * base - rank
        expected = 10 * base - 6
        local_formula = "((4 - rank) * (((index * 5 + 3) % 19) - 9)) - rank"
        sum_formula = "10 * (((index * 5 + 3) % 19) - 9) - 6"
    else:
        raise ValueError(f"unsupported correctness-pattern variant {variant}")
    return (
        local.to(torch.bfloat16),
        expected.to(torch.bfloat16),
        local_formula,
        sum_formula,
    )


def _check_case(
    *,
    name: str,
    mode: str,
    inp: torch.Tensor,
    out: torch.Tensor,
    expected: torch.Tensor,
    input_formula: str,
    expected_formula: str,
) -> dict[str, Any]:
    """Require byte-exact output and return an explicit correctness record."""

    passed = torch.equal(out, expected)
    record = {
        "name": name,
        "mode": mode,
        "input_formula": input_formula,
        "expected_formula": expected_formula,
        "input_sha256": _tensor_sha256(inp),
        "expected_sha256": _tensor_sha256(expected),
        "output_sha256": _tensor_sha256(out),
        "passed": passed,
    }
    if not passed:
        mismatch_count = int(torch.count_nonzero(out != expected).item())
        raise AssertionError(
            f"{name} produced {mismatch_count} incorrect BF16 elements: {record}"
        )
    return record


def _run_eager_case(
    runtime: object,
    *,
    name: str,
    inp: torch.Tensor,
    new_input: torch.Tensor,
    out: torch.Tensor,
    expected: torch.Tensor,
    input_formula: str,
    expected_formula: str,
) -> dict[str, Any]:
    inp.copy_(new_input)
    out.fill_(321)
    runtime.all_reduce(inp, out=out)
    torch.cuda.synchronize()
    runtime.check_health()
    return _check_case(
        name=name,
        mode="eager",
        inp=inp,
        out=out,
        expected=expected,
        input_formula=input_formula,
        expected_formula=expected_formula,
    )


def _run_graph_case(
    runtime: object,
    graph: torch.cuda.CUDAGraph,
    *,
    name: str,
    inp: torch.Tensor,
    new_input: torch.Tensor,
    out: torch.Tensor,
    expected: torch.Tensor,
    poison: int,
    input_formula: str,
    expected_formula: str,
) -> dict[str, Any]:
    inp.copy_(new_input)
    out.fill_(poison)
    graph.replay()
    torch.cuda.synchronize()
    runtime.check_health()
    return _check_case(
        name=name,
        mode="graph-replay",
        inp=inp,
        out=out,
        expected=expected,
        input_formula=input_formula,
        expected_formula=expected_formula,
    )


def _retire_and_validate(
    runtime: object,
    *,
    payload_bytes: int,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    """Wait for every signaled flag completion and validate all local paths."""

    torch.cuda.synchronize()
    dist.barrier()
    deadline = time.monotonic() + timeout_seconds
    while True:
        runtime.check_health()
        stats = runtime.stats()
        paths = runtime.benchmark_counters()
        errors = sum(int(path["completion_errors"]) for path in paths)
        retired = (
            len(paths) == 6
            and int(stats["ops_posted"]) == int(stats["epoch"])
            and int(stats["last_seq"]) == int(stats["epoch"])
            and errors == 0
            and int(stats["writes_completed"])
            == int(stats["ops_posted"]) * len(paths)
            and all(
                int(path["send_completions"]) == int(path["flag_writes"])
                == int(stats["ops_posted"])
                for path in paths
            )
        )
        if retired:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "RoCE QP completion retirement did not converge before "
                f"{timeout_seconds:.3f}s: stats={stats} paths={paths}"
            )
        time.sleep(0.001)

    operations = int(stats["ops_posted"])
    # The bundled proxy posts one payload WQE per stripe and has no tiling knob.
    tile_bytes = 0
    packs = payload_bytes // 16
    stripe_bytes = (((packs + 1) // 2) * 16, payload_bytes - ((packs + 1) // 2) * 16)
    for path in paths:
        path_index = int(path["path_index"])
        expected_bytes = operations * stripe_bytes[path_index]
        path_bytes = stripe_bytes[path_index]
        chunks = (
            0
            if path_bytes == 0
            else 1
            if tile_bytes == 0
            else 1 + (path_bytes - 1) // tile_bytes
        )
        expected_writes = operations * chunks
        if int(path["payload_bytes"]) != expected_bytes:
            raise AssertionError(
                f"rank path payload-byte mismatch: expected {expected_bytes}, path={path}"
            )
        if int(path["payload_writes"]) != expected_writes:
            raise AssertionError(
                f"rank path payload-WQE mismatch: expected {expected_writes}, path={path}"
            )
        if int(path["flag_writes"]) != operations:
            raise AssertionError(f"rank path flag mismatch: expected {operations}, path={path}")
        expected_hop_bytes = expected_bytes * int(path["physical_hops"])
        if int(path["physical_hop_payload_bytes"]) != expected_hop_bytes:
            raise AssertionError(
                "rank path physical-hop byte mismatch: "
                f"expected {expected_hop_bytes}, path={path}"
            )
    if int(stats["writes_completed"]) != operations * len(paths):
        raise AssertionError(
            f"aggregate completion mismatch: expected {operations * len(paths)}, stats={stats}"
        )
    expected = {
        "operations_per_rank": operations,
        "paths_per_rank": len(paths),
        "flags_per_path": operations,
        "payload_bytes_path_0": operations * stripe_bytes[0],
        "payload_bytes_path_1": operations * stripe_bytes[1],
    }
    return stats, paths, expected


def main() -> None:
    """Run correctness probes, unchanged timing loops, and QP reconciliation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--bytes", type=int, default=65536)
    parser.add_argument("--warmups", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--graph-ops", type=int, default=10)
    parser.add_argument("--retire-timeout", type=float, default=10.0)
    args = parser.parse_args()

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world = dist.get_world_size()
    if world != 4:
        raise RuntimeError(f"virtual-diagonal evidence requires four ranks, got {world}")
    if args.bytes <= 0 or args.bytes % 16:
        raise ValueError("--bytes must be a positive multiple of 16")
    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)

    import b12x.comm
    # Select only the image-bundled transport, without installing vLLM hooks.
    b12x.comm.__path__.insert(0, "/opt/spark-sircl/b12x_overlay/b12x/comm")
    from b12x.comm import roce

    runtime = roce.AllReduce.from_exchange_group(
        exchange_group=dist.group.WORLD,
        device=device,
        max_size=2 << 20,
        max_gather_bytes=2 << 20,
    )
    runtime.prepare((torch.bfloat16,))
    numel = args.bytes // 2
    inp = torch.empty(numel, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(inp)
    cases: list[dict[str, Any]] = []

    first_input, first_expected, first_formula, first_sum = _pattern(rank, numel, 0, device)
    inp.copy_(first_input)
    cases.append(
        _run_eager_case(
            runtime,
            name="rank-specific-index-pattern-a",
            inp=inp,
            new_input=first_input,
            out=out,
            expected=first_expected,
            input_formula=first_formula,
            expected_formula=first_sum,
        )
    )
    inp.fill_(rank + 1)
    out.fill_(-321)
    runtime.all_reduce(inp, out=out)
    torch.cuda.synchronize()
    constant_expected = torch.full_like(out, 10)
    cases.append(
        _check_case(
            name="constant-timing-reference",
            mode="eager",
            inp=inp,
            out=out,
            expected=constant_expected,
            input_formula="rank + 1",
            expected_formula="10",
        )
    )

    for _ in range(args.warmups):
        runtime.all_reduce(inp, out=out)
    torch.cuda.synchronize()
    dist.barrier()
    eager = timed_eager(runtime, inp, out, args.samples)

    graph = capture(runtime, inp, out, args.graph_ops)
    cases.append(
        _run_graph_case(
            runtime,
            graph,
            name="graph-input-mutation-a",
            inp=inp,
            new_input=first_input,
            out=out,
            expected=first_expected,
            poison=257,
            input_formula=first_formula,
            expected_formula=first_sum,
        )
    )
    second_input, second_expected, second_formula, second_sum = _pattern(
        rank, numel, 1, device
    )
    cases.append(
        _run_graph_case(
            runtime,
            graph,
            name="graph-input-mutation-b",
            inp=inp,
            new_input=second_input,
            out=out,
            expected=second_expected,
            poison=-257,
            input_formula=second_formula,
            expected_formula=second_sum,
        )
    )
    if cases[-1]["output_sha256"] == cases[-2]["output_sha256"]:
        raise AssertionError("graph mutation probes produced the same output hash")

    inp.fill_(rank + 1)
    out.fill_(-321)
    for _ in range(args.warmups):
        graph.replay()
    torch.cuda.synchronize()
    dist.barrier()
    graph_values = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        graph_values.append(start.elapsed_time(end) * 1000.0 / args.graph_ops)
    runtime.check_health()
    cases.append(
        _check_case(
            name="constant-timing-output",
            mode="graph-replay",
            inp=inp,
            out=out,
            expected=constant_expected,
            input_formula="rank + 1",
            expected_formula="10",
        )
    )

    stats, paths, expected_counters = _retire_and_validate(
        runtime,
        payload_bytes=args.bytes,
        timeout_seconds=args.retire_timeout,
    )
    local = {
        "rank": rank,
        "eager_samples_us": eager,
        "graph_samples_us": graph_values,
        "eager_median_us": statistics.median(eager),
        "graph_median_us": statistics.median(graph_values),
        "correctness_cases": cases,
        "stats": stats,
        "path_counters": paths,
        "expected_counters": expected_counters,
    }
    gathered: list[object] = [None] * world
    dist.all_gather_object(gathered, local)
    if sum(len(row["path_counters"]) for row in gathered) != 24:  # type: ignore[index]
        raise AssertionError("four-rank evidence must contain exactly 24 origin-QP paths")
    if rank == 0:
        eager_max = max(float(row["eager_median_us"]) for row in gathered)  # type: ignore[index]
        graph_max = max(float(row["graph_median_us"]) for row in gathered)  # type: ignore[index]
        print(
            f"RESULT bytes={args.bytes} eager_max_rank_median_us={eager_max:.3f} "
            f"graph_max_rank_median_us={graph_max:.3f}",
            flush=True,
        )
        print(
            "EVIDENCE_JSON "
            + json.dumps(
                {
                    "schema": "b12x.rocenante-virtual-diagonal-evidence/v1",
                    "status": "research-only",
                    "payload_bytes": args.bytes,
                    "warmups": args.warmups,
                    "samples": args.samples,
                    "graph_operations_per_replay": args.graph_ops,
                    "ranks": gathered,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
    runtime.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
