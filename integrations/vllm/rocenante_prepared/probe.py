"""Bounded two/four-rank prepared RoCE correctness and graph qualification.

Adapts the pinned B12X tests/comm/test_roce_oneshot_gpu.py methodology
(a83336581a3a907076e60797df69ab66df5a2ff1, Apache-2.0). Run with torchrun
only in an explicitly reserved idle-hardware window. No model is loaded.
"""

from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import time


def path_payload_deltas(before, after, hca_count):
    """Aggregate the adaptive proxy's per-origin-QP counters by local HCA."""

    def index(rows):
        result = {}
        for row in rows:
            key = (row["peer_rank"], row["path_index"])
            if key in result:
                raise ValueError("duplicate RoCE peer/path counter")
            hca = row["local_hca_index"]
            if type(hca) is not int or not 0 <= hca < hca_count:
                raise ValueError("RoCE counter names an invalid local HCA")
            if type(row["payload_bytes"]) is not int or row["payload_bytes"] < 0:
                raise ValueError("RoCE payload counter must be nonnegative")
            if row["completion_errors"] != 0:
                raise ValueError("RoCE path reports a failed completion")
            result[key] = row
        if not result:
            raise ValueError("RoCE path counter inventory is empty")
        return result

    initial, final = index(before), index(after)
    if initial.keys() != final.keys():
        raise ValueError("RoCE path inventory changed during the probe")
    totals = [0] * hca_count
    for key, row in final.items():
        original = initial[key]
        if row["local_hca_index"] != original["local_hca_index"]:
            raise ValueError("RoCE path changed its local HCA")
        delta = row["payload_bytes"] - original["payload_bytes"]
        if delta <= 0:
            raise ValueError("RoCE path produced no positive payload traffic")
        totals[row["local_hca_index"]] += delta
    return totals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="execute real distributed GPU collectives"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    args = parser.parse_args()
    if not args.run:
        parser.error("real hardware execution requires --run")
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4):
        parser.error("requires torchrun WORLD_SIZE=2 or 4")

    import torch
    import torch.distributed as dist
    import sparkring_transport_selector as selector
    from b12x.comm import roce
    from b12x.comm.roce import _preparation
    from b12x._lib.runtime_control import (
        kernel_resolution_guard,
        kernel_resolution_frozen,
    )
    from b12x.preparation import PreparationSession, PreparedCall

    if os.environ.get(selector.PROFILE_ENV) != "tp2-rocenante-adaptive-prepared":
        raise RuntimeError("probe requires the explicit prepared adaptive profile")
    selector.require_active()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 1):
        raise RuntimeError("probe requires a GB10/SM121 GPU")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    dist.init_process_group("nccl", timeout=timedelta(seconds=args.timeout_seconds))
    control = dist.new_group(
        backend="gloo", timeout=timedelta(seconds=args.timeout_seconds)
    )
    rank = dist.get_rank()
    started = time.monotonic()
    records = []
    runtime = None
    args.output_dir.mkdir(parents=True, exist_ok=True)

    def record(name, **values):
        records.append(dict(case=name, **values))
        print(json.dumps(dict(rank=rank, case=name, status="passed")), flush=True)

    def tolerance(dtype):
        if dtype == torch.float32:
            return 1e-5, 1e-5 * world
        if dtype == torch.bfloat16:
            return 1e-2, 2e-2 * world
        return 2e-3, 4e-3 * world

    try:
        if not roce.is_supported(device):
            raise RuntimeError(
                "selected RoCE runtime reports unsupported hardware or RDMA"
            )
        runtime = roce.AllReduce.from_exchange_group(
            exchange_group=control,
            device=device,
            max_size=1 << 20,
            max_gather_bytes=1 << 20,
        )
        query = roce.query_from_runtime(
            runtime,
            surface="AllReduce.all_reduce",
            call={"dtypes": ("bfloat16", "float16", "float32")},
            topology="roce_rdma",
            peer_hosts=tuple(f"rank-{r}" for r in range(world)),
        )
        declaration = roce.plan(query, runtime=runtime)
        seeds = [
            torch.zeros(16 // dtype.itemsize, dtype=dtype, device=device)
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
        ]

        def prepare(state):
            calls = [_preparation.prepared_call(state, inp=seed) for seed in seeds]
            calls.append(_preparation.prepared_gather_call(state, inp=seeds[0]))
            return PreparedCall(run=lambda: [call.run() for call in calls])

        with PreparationSession(
            device=device, autotune=False, compile_workers=2
        ) as session:
            result = session.prepare(
                (declaration.request(name="adaptive-roce-probe", prepare_call=prepare),)
            )
            try:
                before_paths = runtime.benchmark_counters()
                for dtype in (torch.bfloat16, torch.float16, torch.float32):
                    for nbytes in (16, 4096, 1 << 20):
                        torch.manual_seed(704 + rank)
                        inp = torch.randn(
                            nbytes // dtype.itemsize, dtype=dtype, device=device
                        )
                        expected = inp.clone()
                        dist.all_reduce(expected)
                        with kernel_resolution_guard("prepared RoCE eager reduce"):
                            out = runtime.all_reduce(inp, plan=declaration)
                        torch.cuda.synchronize()
                        torch.testing.assert_close(
                            out,
                            expected,
                            rtol=tolerance(dtype)[0],
                            atol=tolerance(dtype)[1],
                        )
                        peer_outputs = [torch.empty_like(out) for _ in range(world)]
                        dist.all_gather(peer_outputs, out)
                        assert all(torch.equal(out, peer) for peer in peer_outputs)
                        runtime.check_health()
                        record(
                            "reduce",
                            dtype=str(dtype),
                            bytes=nbytes,
                            rank_identical=True,
                        )

                for shape, dimension in (((4, 32), 0), ((4, 32), -1), ((5, 3), -1)):
                    inp = torch.full(
                        shape, rank + 1, dtype=torch.bfloat16, device=device
                    )
                    peer_inputs = [torch.empty_like(inp) for _ in range(world)]
                    dist.all_gather(peer_inputs, inp)
                    expected = torch.cat(peer_inputs, dim=dimension)
                    with kernel_resolution_guard("prepared RoCE eager gather"):
                        out = runtime.all_gather(inp, dim=dimension, plan=declaration)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, expected, rtol=0, atol=0)
                    record("gather", shape=shape, dimension=dimension)

                # An unaligned but pack-sized input uses shared staging without
                # padding gaps. Its returned output must still be independently owned.
                backing = torch.full(
                    (33,), rank + 1, dtype=torch.bfloat16, device=device
                )
                unaligned = backing[1:]
                other_backing = torch.full(
                    (33,), 2 * (rank + 1), dtype=torch.bfloat16, device=device
                )
                other_unaligned = other_backing[1:]
                assert unaligned.data_ptr() % 16 != 0
                with kernel_resolution_guard("prepared RoCE output ownership"):
                    first = runtime.all_gather(unaligned, dim=0, plan=declaration)
                    saved = first.clone()
                    second = runtime.all_gather(
                        other_unaligned, dim=0, plan=declaration
                    )
                torch.cuda.synchronize()
                torch.testing.assert_close(first, saved, rtol=0, atol=0)
                assert (
                    first.untyped_storage().data_ptr()
                    != runtime._gather_buffers[1].untyped_storage().data_ptr()
                )
                torch.testing.assert_close(second, saved * 2, rtol=0, atol=0)
                record("gather-output-ownership")

                small = torch.zeros(2048, dtype=torch.bfloat16, device=device)
                large = torch.zeros(32768, dtype=torch.bfloat16, device=device)
                odd = torch.zeros((5, 3), dtype=torch.bfloat16, device=device)
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    runtime.all_reduce(small, plan=declaration)
                    runtime.all_reduce(large, plan=declaration)
                    runtime.all_gather(odd, dim=-1, plan=declaration)
                    runtime.all_gather(unaligned, dim=0, plan=declaration)
                    runtime.all_gather(other_unaligned, dim=0, plan=declaration)
                torch.cuda.current_stream().wait_stream(stream)
                torch.cuda.synchronize()
                dist.barrier(group=control)
                graph = torch.cuda.CUDAGraph()
                try:
                    with kernel_resolution_guard(
                        "prepared RoCE alternating-grid graph"
                    ):
                        assert kernel_resolution_frozen()
                        with (
                            torch.cuda.graph(graph, stream=stream),
                            runtime.capture(stream=stream),
                        ):
                            reduced_small = runtime.all_reduce(small, plan=declaration)
                            reduced_large = runtime.all_reduce(large, plan=declaration)
                            gathered = runtime.all_gather(odd, dim=-1, plan=declaration)
                            gathered_unaligned = runtime.all_gather(
                                unaligned, dim=0, plan=declaration
                            )
                            gathered_other = runtime.all_gather(
                                other_unaligned, dim=0, plan=declaration
                            )
                            reduced_again = runtime.all_reduce(
                                reduced_small, plan=declaration
                            )
                        graph_outputs = (
                            reduced_small,
                            reduced_large,
                            gathered,
                            gathered_unaligned,
                            gathered_other,
                            reduced_again,
                        )
                        pointers = tuple(t.data_ptr() for t in graph_outputs)
                        for replay in range(4):
                            small.fill_(rank + replay + 1)
                            large.fill_(rank + replay + 2)
                            odd.fill_(rank + replay + 3)
                            unaligned.fill_(rank + replay + 4)
                            other_unaligned.fill_(rank + replay + 7)
                            expected_small = small.clone()
                            expected_large = large.clone()
                            dist.all_reduce(expected_small)
                            dist.all_reduce(expected_large)
                            peer_inputs = [torch.empty_like(odd) for _ in range(world)]
                            dist.all_gather(peer_inputs, odd)
                            expected_gather = torch.cat(peer_inputs, dim=-1)
                            expected_unaligned = torch.cat(
                                [
                                    torch.full_like(unaligned, r + replay + 4)
                                    for r in range(world)
                                ]
                            )
                            expected_other = torch.cat(
                                [
                                    torch.full_like(other_unaligned, r + replay + 7)
                                    for r in range(world)
                                ]
                            )
                            torch.cuda.synchronize()
                            allocations = torch.cuda.memory_stats(device)[
                                "allocation.all.allocated"
                            ]
                            graph.replay()
                            torch.cuda.synchronize()
                            assert (
                                allocations
                                == torch.cuda.memory_stats(device)[
                                    "allocation.all.allocated"
                                ]
                            )
                            assert pointers == tuple(
                                t.data_ptr() for t in graph_outputs
                            )
                            torch.testing.assert_close(
                                reduced_small, expected_small, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                reduced_large, expected_large, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                gathered, expected_gather, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                gathered_unaligned, expected_unaligned, rtol=0, atol=0
                            )
                            torch.testing.assert_close(
                                gathered_other, expected_other, rtol=0, atol=0
                            )
                            assert (
                                gathered_unaligned.data_ptr()
                                != gathered_other.data_ptr()
                            )
                            torch.testing.assert_close(
                                reduced_again, expected_small * world, rtol=0, atol=0
                            )
                            runtime.check_health()
                    record(
                        "alternating-grid-frozen-graph",
                        replays=4,
                        stable_addresses=True,
                        replay_allocations=0,
                    )
                finally:
                    graph.reset()
                after_paths = runtime.benchmark_counters()
                byte_delta = path_payload_deltas(
                    before_paths, after_paths, len(runtime.hca_names)
                )
                active_hcas = set(
                    index
                    for peer in runtime._peer_hca_map
                    for index in peer
                    if index >= 0
                )
                assert all(byte_delta[index] > 0 for index in active_hcas)
                assert all(
                    value == 0
                    for index, value in enumerate(byte_delta)
                    if index not in active_hcas
                )
                record(
                    "selected-path-traffic",
                    active_hca_indices=sorted(active_hcas),
                    bytes_per_hca=byte_delta,
                    path_counters=after_paths,
                )
                dist.barrier(group=control)
            finally:
                result.close()
        outcome = dict(
            status="passed",
            rank=rank,
            world_size=world,
            elapsed_seconds=time.monotonic() - started,
            profile=os.environ[selector.PROFILE_ENV],
            manifest_sha256=os.environ[selector.DIGEST_ENV],
            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            cases=records,
            stats=runtime.stats(),
        )
        (args.output_dir / f"rank-{rank}.json").write_text(
            json.dumps(outcome, indent=2) + "\n"
        )
    except BaseException as error:
        (args.output_dir / f"rank-{rank}.json").write_text(
            json.dumps(
                dict(
                    status="failed",
                    rank=rank,
                    world_size=world,
                    error_type=type(error).__name__,
                    error=str(error),
                    completed_cases=records,
                ),
                indent=2,
            )
            + "\n"
        )
        raise
    finally:
        if runtime is not None:
            runtime.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
