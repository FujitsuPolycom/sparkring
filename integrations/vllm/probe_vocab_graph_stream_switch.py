"""Model-free four-rank probe for vocabulary graph stream switching.

The probe imports the production adapter, gives it a tiny stand-in
GroupCoordinator, and uses the real CUDA/native direct-cable graph session.
It reproduces the startup ordering that failed in live vLLM:

1. noncapturing vocabulary warmup on stream A;
2. CUDA graph capture on stream B;
3. another noncapturing speculator warmup on stream C;
4. repeated graph replay and byte-exact result validation.
"""

from __future__ import annotations

import argparse
import os
import sys
import types


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, choices=range(4), required=True)
    parser.add_argument("--iterations", type=int, default=32)
    parser.add_argument("--mtp-tokens", type=int, choices=(4, 5), default=4)
    parser.add_argument("--submit-cpu", type=int, default=10)
    parser.add_argument("--tp-progress-cpu", type=int, default=11)
    return parser.parse_args()


def _install_dependency_seams(
    torch_module: types.ModuleType,
    submit_cpu: int,
    tp_progress_cpu: int,
) -> tuple[type, list[tuple[str, bool, str]]]:
    class GroupCoordinator:
        def __init__(self, rank: int) -> None:
            self.unique_name = "tp:0"
            self.world_size = 4
            self.rank_in_group = rank
            self.original_streams: list[int] = []

        def _all_gather_out_place(self, input_tensor, dim: int):
            normalized_dim = dim + input_tensor.ndim if dim < 0 else dim
            if normalized_dim != 1:
                raise ValueError("probe stock gather only supports dim 1")
            self.original_streams.append(
                int(torch_module.cuda.current_stream().cuda_stream)
            )
            return torch_module.cat([input_tensor] * 4, dim=1)

    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.GroupCoordinator = GroupCoordinator
    distributed = types.ModuleType("vllm.distributed")
    distributed.parallel_state = parallel_state
    vllm = types.ModuleType("vllm")
    vllm.distributed = distributed

    audit_events: list[tuple[str, bool, str]] = []
    audit = types.ModuleType("spark_collective_audit")
    audit.record_stock = lambda family, capturing, reason: audit_events.append(
        (family, capturing, reason)
    )

    tp_backend = types.ModuleType("spark_tp4_backend")
    tp_backend._graph_preflight = lambda: (
        submit_cpu,
        tp_progress_cpu,
    )
    reporter = types.ModuleType("spark_graph_status_reporter")
    reporter.ensure_status_reporter = lambda rank: None

    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.distributed": distributed,
            "vllm.distributed.parallel_state": parallel_state,
            "spark_collective_audit": audit,
            "spark_tp4_backend": tp_backend,
            "spark_graph_status_reporter": reporter,
        }
    )
    return GroupCoordinator, audit_events


def _input(torch_module, rank: int, query_rows: int):
    row = torch_module.arange(
        query_rows, dtype=torch_module.int32, device="cuda"
    ).reshape(query_rows, 1)
    return (
        (row * 16 + rank)
        .to(dtype=torch_module.bfloat16)
        .expand(query_rows, 38720)
        .contiguous()
    )


def _expected(torch_module, query_rows: int):
    row = torch_module.arange(
        query_rows, dtype=torch_module.int32, device="cuda"
    ).reshape(query_rows, 1)
    blocks = [
        (row * 16 + rank).to(dtype=torch_module.bfloat16).expand(query_rows, 38720)
        for rank in range(4)
    ]
    return torch_module.cat(blocks, dim=1).contiguous()


def main() -> int:
    args = _arguments()
    if args.iterations <= 0:
        raise ValueError("iterations must be positive")
    if args.submit_cpu == args.tp_progress_cpu:
        raise ValueError("submit and TP progress CPUs must differ")

    import torch

    torch.cuda.set_device(0)
    group_type, audit_events = _install_dependency_seams(
        torch, args.submit_cpu, args.tp_progress_cpu
    )
    import spark_tp4_vocab_allgather_backend as adapter

    adapter.install()
    group = group_type(args.rank)
    pattern = [args.mtp_tokens + 1] + [1] * args.mtp_tokens
    inputs = [_input(torch, args.rank, query_rows) for query_rows in pattern]
    expected = [_expected(torch, query_rows) for query_rows in pattern]
    torch.cuda.synchronize()
    stream_a = torch.cuda.Stream()
    stream_b = torch.cuda.Stream()
    stream_c = torch.cuda.Stream()
    stream_pointers = {
        int(stream_a.cuda_stream),
        int(stream_b.cuda_stream),
        int(stream_c.cuda_stream),
    }
    if len(stream_pointers) != 3:
        raise RuntimeError("CUDA returned duplicate stream A/B/C handles")

    with torch.cuda.stream(stream_a):
        warmup_outputs = [
            group._all_gather_out_place(input_tensor, -1) for input_tensor in inputs
        ]
    stream_a.synchronize()
    warmup_mismatches = 0
    for actual, reference in zip(warmup_outputs, expected, strict=True):
        warmup_mismatches += int(
            torch.count_nonzero(
                actual.view(torch.uint8) != reference.view(torch.uint8)
            ).item()
        )

    state = group._spark_tp4_vocab_native
    if state._session is None:
        raise RuntimeError("graph-mode warmup did not prepare eager native session")
    if state._graph_session is None:
        raise RuntimeError("graph-mode warmup did not prepare graph session")
    eager_session = state._session

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream_b):
        captured_outputs = [
            group._all_gather_out_place(input_tensor, -1) for input_tensor in inputs
        ]

    # The eager native session must survive a caller-stream change after a
    # backbone graph exists. Its CUDA-event handoff preserves ordering.
    with torch.cuda.stream(stream_c):
        post_capture_warmup = [
            group._all_gather_out_place(input_tensor, -1) for input_tensor in inputs
        ]
    stream_c.synchronize()
    post_capture_mismatches = 0
    for actual, reference in zip(
        post_capture_warmup, expected, strict=True
    ):
        post_capture_mismatches += int(
            torch.count_nonzero(
                actual.view(torch.uint8) != reference.view(torch.uint8)
            ).item()
        )
    eager_session_reused = state._session is eager_session

    with torch.cuda.stream(stream_b):
        for _ in range(args.iterations):
            graph.replay()
    stream_b.synchronize()

    mismatches = 0
    for actual, reference in zip(captured_outputs, expected, strict=True):
        mismatches += int(
            torch.count_nonzero(
                actual.view(torch.uint8) != reference.view(torch.uint8)
            ).item()
        )

    status = state._graph_session.graph_status()
    nodes = len(pattern)
    expected_sequence = nodes * args.iterations
    passed = (
        len(group.original_streams) == 0
        and len(audit_events) == 0
        and eager_session_reused
        and warmup_mismatches == 0
        and post_capture_mismatches == 0
        and int(status["captured_nodes"]) == nodes
        and int(status["published_sequence"]) == expected_sequence
        and int(status["consumed_sequence"]) == expected_sequence
        and int(status["completed_sequence"]) == expected_sequence
        and int(status["overflow_sequence"]) == 0
        and mismatches == 0
    )
    fields = {
        "rank": args.rank,
        "pattern": ",".join(str(item) for item in pattern),
        "stream_a": hex(int(stream_a.cuda_stream)),
        "stream_b": hex(int(stream_b.cuda_stream)),
        "stream_c": hex(int(stream_c.cuda_stream)),
        "stock_warmups": len(audit_events),
        "warmup_mismatches": warmup_mismatches,
        "post_capture_mismatches": post_capture_mismatches,
        "captured_nodes": int(status["captured_nodes"]),
        "published": int(status["published_sequence"]),
        "consumed": int(status["consumed_sequence"]),
        "completed": int(status["completed_sequence"]),
        "overflow": int(status["overflow_sequence"]),
        "mismatches": mismatches,
        "eager_native_created": state._session is not None,
        "eager_session_reused": eager_session_reused,
        "passed": passed,
    }
    print(
        "TP4_VOCAB_STREAM_SWITCH "
        + " ".join(
            f"{key}={str(value).lower() if isinstance(value, bool) else value}"
            for key, value in fields.items()
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    os.environ.setdefault("VLLM_SPARK_TP4_VOCAB_MODE", "custom")
    os.environ.setdefault("VLLM_SPARK_TP4_GRAPH_Q1", "1")
    raise SystemExit(main())
