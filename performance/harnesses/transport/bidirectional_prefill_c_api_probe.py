"""Four-rank direct C-ABI qualification for bidirectional TP4 prefill."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import time
from pathlib import Path


class NativeConfig(ctypes.Structure):
    _fields_ = [
        ("rank", ctypes.c_uint32),
        ("peer0", ctypes.c_char_p),
        ("peer1", ctypes.c_char_p),
        ("device0", ctypes.c_char_p),
        ("device1", ctypes.c_char_p),
        ("gid0", ctypes.c_uint8),
        ("gid1", ctypes.c_uint8),
        ("control_port0", ctypes.c_uint16),
        ("control_port1", ctypes.c_uint16),
        ("payload_bytes", ctypes.c_size_t),
        ("graph_submit_cpu_plus_one", ctypes.c_uint32),
        ("graph_progress_cpu_plus_one", ctypes.c_uint32),
    ]


class NativeConfigV2(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("base", NativeConfig),
        ("elements_per_row", ctypes.c_uint32),
        ("bytes_per_row", ctypes.c_uint32),
    ]


class BidirectionalPrefillConfigV1(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("primary", NativeConfigV2),
        ("rail_count", ctypes.c_uint32),
        ("query_rows", ctypes.c_uint32),
        ("secondary_peer0", ctypes.c_char_p),
        ("secondary_peer1", ctypes.c_char_p),
        ("secondary_device0", ctypes.c_char_p),
        ("secondary_device1", ctypes.c_char_p),
        ("secondary_gid0", ctypes.c_uint8),
        ("secondary_gid1", ctypes.c_uint8),
        ("secondary_control_port0", ctypes.c_uint16),
        ("secondary_control_port1", ctypes.c_uint16),
        ("timeout_seconds", ctypes.c_uint32),
    ]


def bind(library: ctypes.CDLL):
    create = library.spark_tp4_bidirectional_prefill_create
    create.argtypes = [
        ctypes.POINTER(BidirectionalPrefillConfigV1),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    create.restype = ctypes.c_void_p
    all_reduce = library.spark_tp4_bidirectional_prefill_all_reduce
    all_reduce.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    all_reduce.restype = ctypes.c_int
    destroy = library.spark_tp4_bidirectional_prefill_destroy
    destroy.argtypes = [ctypes.c_void_p]
    destroy.restype = None
    return create, all_reduce, destroy


def invoke(
    all_reduce,
    handle: int,
    input_tensor,
    output_tensor,
    *,
    stream_zero: bool = False,
) -> float:
    import torch

    error = ctypes.create_string_buffer(512)
    stream = torch.cuda.current_stream(device=input_tensor.device)
    start = time.perf_counter_ns()
    result = all_reduce(
        handle,
        ctypes.c_void_p(input_tensor.data_ptr()),
        ctypes.c_void_p(output_tensor.data_ptr()),
        ctypes.c_void_p(0 if stream_zero else stream.cuda_stream),
        error,
        len(error),
    )
    elapsed_us = (time.perf_counter_ns() - start) / 1000.0
    if result != 0:
        raise RuntimeError(error.value.decode(errors="replace"))
    return elapsed_us


def validate_noninteger(output, query_rows: int) -> tuple[float, int]:
    import torch

    count = query_rows * 4096
    index = torch.arange(count, device="cuda", dtype=torch.int32)
    expected = torch.zeros(count, device="cuda", dtype=torch.float32)
    for rank in range(4):
        centered = ((index * 37 + rank * 53).remainder(257) - 128).float()
        values = centered * ((rank + 1) / 128.0)
        values += ((index + 3 * rank).remainder(11)).float() / 512.0
        expected += values.to(torch.bfloat16).float()
    expected = expected.to(torch.bfloat16).float()
    actual = output.reshape(-1).float()
    difference = (actual - expected).abs()
    tolerance = 0.0625 + 0.02 * expected.abs()
    return float(difference.max().item()), int((difference > tolerance).sum().item())


def run_shape(args, query_rows: int, library: ctypes.CDLL) -> dict[str, object]:
    import torch
    from spark_tp4_port_namespace import (
        bidirectional_prefill_control_ports,
        bidirectional_prefill_secondary_control_ports,
    )

    ports = bidirectional_prefill_control_ports(query_rows)
    secondary_ports = bidirectional_prefill_secondary_control_ports(query_rows)
    payload_bytes = query_rows * 4096 * 2
    base = NativeConfig(
        rank=args.rank,
        peer0=args.peer0.encode(),
        peer1=args.peer1.encode(),
        device0=args.device0.encode(),
        device1=args.device1.encode(),
        gid0=3,
        gid1=3,
        control_port0=ports[0],
        control_port1=ports[1],
        payload_bytes=payload_bytes,
        graph_submit_cpu_plus_one=0,
        graph_progress_cpu_plus_one=0,
    )
    primary = NativeConfigV2(
        struct_size=ctypes.sizeof(NativeConfigV2),
        base=base,
        elements_per_row=4096,
        bytes_per_row=8192,
    )
    config = BidirectionalPrefillConfigV1(
        struct_size=ctypes.sizeof(BidirectionalPrefillConfigV1),
        primary=primary,
        rail_count=2,
        query_rows=query_rows,
        secondary_peer0=args.secondary_peer0.encode(),
        secondary_peer1=args.secondary_peer1.encode(),
        secondary_device0=args.secondary_device0.encode(),
        secondary_device1=args.secondary_device1.encode(),
        secondary_gid0=3,
        secondary_gid1=3,
        secondary_control_port0=secondary_ports[0],
        secondary_control_port1=secondary_ports[1],
        timeout_seconds=120,
    )
    create, all_reduce, destroy = bind(library)
    error = ctypes.create_string_buffer(512)
    handle = create(ctypes.byref(config), error, len(error))
    if not handle:
        raise RuntimeError(error.value.decode(errors="replace"))
    exact_times: list[float] = []
    noninteger_times: list[float] = []
    exact_mismatches = 0
    noninteger_mismatches = 0
    noninteger_max_abs = 0.0
    stream_zero_exact_mismatches = 0
    stream_zero_noninteger_mismatches = 0
    try:
        exact_input = torch.full(
            (query_rows, 4096),
            float(args.rank + 1),
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = torch.empty_like(exact_input)
        for _ in range(args.repeats):
            exact_times.append(invoke(all_reduce, handle, exact_input, output))
            exact_mismatches += int((output.float() != 10.0).sum().item())

        index = torch.arange(
            query_rows * 4096, device="cuda", dtype=torch.int32
        )
        centered = (
            (index * 37 + args.rank * 53).remainder(257) - 128
        ).float()
        noninteger_input = (
            centered * ((args.rank + 1) / 128.0)
            + ((index + 3 * args.rank).remainder(11)).float() / 512.0
        ).to(torch.bfloat16).reshape(query_rows, 4096)
        for _ in range(args.repeats):
            noninteger_times.append(
                invoke(all_reduce, handle, noninteger_input, output)
            )
            maximum, mismatches = validate_noninteger(output, query_rows)
            noninteger_max_abs = max(noninteger_max_abs, maximum)
            noninteger_mismatches += mismatches
        if args.include_stream_zero:
            torch.cuda.synchronize()
            invoke(
                all_reduce,
                handle,
                exact_input,
                output,
                stream_zero=True,
            )
            stream_zero_exact_mismatches = int(
                (output.float() != 10.0).sum().item()
            )
            torch.cuda.synchronize()
            invoke(
                all_reduce,
                handle,
                noninteger_input,
                output,
                stream_zero=True,
            )
            maximum, stream_zero_noninteger_mismatches = validate_noninteger(
                output, query_rows
            )
            noninteger_max_abs = max(noninteger_max_abs, maximum)
    finally:
        destroy(handle)

    return {
        "query_rows": query_rows,
        "payload_bytes": payload_bytes,
        "ports": list(ports),
        "secondary_ports": list(secondary_ports),
        "rail_count": 2,
        "repeats_per_pattern": args.repeats,
        "exact_mismatches": exact_mismatches,
        "noninteger_mismatches": noninteger_mismatches,
        "noninteger_max_abs": noninteger_max_abs,
        "stream_zero_exact_mismatches": stream_zero_exact_mismatches,
        "stream_zero_noninteger_mismatches": (
            stream_zero_noninteger_mismatches
        ),
        "atol": 0.0625,
        "rtol": 0.02,
        "exact_host_wall_us": exact_times,
        "noninteger_host_wall_us": noninteger_times,
        "passed": (
            exact_mismatches == 0
            and noninteger_mismatches == 0
            and stream_zero_exact_mismatches == 0
            and stream_zero_noninteger_mismatches == 0
        ),
    }


def main() -> int:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--peer0", required=True)
    parser.add_argument("--peer1", required=True)
    parser.add_argument("--device0", required=True)
    parser.add_argument("--device1", required=True)
    parser.add_argument("--secondary-peer0", required=True)
    parser.add_argument("--secondary-peer1", required=True)
    parser.add_argument("--secondary-device0", required=True)
    parser.add_argument("--secondary-device1", required=True)
    parser.add_argument("--query-rows", default="2048,8192")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--library", required=True)
    parser.add_argument("--include-stream-zero", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0 <= args.rank < 4 or args.repeats < 2:
        raise SystemExit("rank must be 0..3 and repeats must be at least two")
    rows = [int(value) for value in args.query_rows.split(",")]
    if any(value not in {1024, 2048, 4096, 8192} for value in rows):
        raise SystemExit("unsupported query rows")

    os.environ["VLLM_SPARK_TP4_MODE"] = "custom"
    os.environ["VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL"] = "1"
    os.environ["VLLM_SPARK_TP4_BIDIRECTIONAL_PREFILL_RAIL_MODE"] = "dual"
    library = ctypes.CDLL(args.library)
    qualification_stream = torch.cuda.Stream()
    with torch.cuda.stream(qualification_stream):
        cases = [run_shape(args, value, library) for value in rows]
    qualification_stream.synchronize()
    document = {
        "schema": "sparkring-tp4-bidirectional-prefill-c-api/v1",
        "rank": args.rank,
        "library": args.library,
        "cases": cases,
        "passed": all(bool(case["passed"]) for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print("TP4_BIDIRECTIONAL_PREFILL_C_API " + json.dumps(document))
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
