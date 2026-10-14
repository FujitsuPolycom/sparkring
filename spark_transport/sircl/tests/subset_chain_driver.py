"""Exercise subset policy through real runtime setup, with CUDA imports mocked."""

# CUDA stand-ins must be installed before importing the runtime.
# ruff: noqa: E402

import os
import sys

from teardown_driver import _CudaStandIn

sys.meta_path.insert(0, _CudaStandIn())

import torch

from plan_driver import configured
from sparkring_sircl import routes, tuning
from sparkring_sircl.oneshot import runtime


def main():
    layout = "ring:8:0,1,2,3,4,5"
    for windows in ("0", "131072"):
        os.environ["SIRCL_FORWARD_WINDOW_BYTES"] = windows
        os.environ["SIRCL_SUBSET_RING"] = "0"
        off = configured(runtime, routes, layout, 0)
        os.environ["SIRCL_SUBSET_RING"] = "1"
        on = configured(runtime, routes, layout, 0)
        assert on._forward_table == off._forward_table, "pieces windows changed"
        assert on.forward_chunk_bytes == off.forward_chunk_bytes, "transport chunk changed"
        assert on.gather_piece_bytes == off.gather_piece_bytes, "gather pieces changed"
        assert on.large_piece_bytes == off.large_piece_bytes, "reduce pieces changed"
        assert on.relay_safe_bytes == off.relay_safe_bytes
        assert on.link_available and not on.chain_available and not on.ring_available
        assert on.ring_window_bytes == 0
        for collective in ("all_reduce", "reduce_scatter", "all_gather"):
            assert not on._apply_choice(collective, tuning.Choice(schedule="ring"), 8 << 20)
            if collective != "all_gather":
                assert not on._apply_choice(collective, tuning.Choice(schedule="chain"), 8 << 20)
        assert not on._apply_choice("all_gather", tuning.Choice(schedule="chain"), 1 << 20)
        assert on._apply_choice("all_gather", tuning.Choice(schedule="chain"), 4 << 20)
        for schedule in ("auto", "pieces", "chain", "ring"):
            on.large_schedule = on.scatter_schedule = on.gather_schedule = schedule
            for size in (16, 1 << 20, 8 << 20, 256 << 20):
                assert on.large_reduce_plan(size) == off.large_reduce_plan(size)
                assert on._schedule(schedule, size, "scatter") == "pieces"
            for shape, dim, expected in (((2 << 20,), 0, schedule != "pieces"),
                                         ((8 << 20,), -1, schedule != "pieces"),
                                         ((524288,), 0, False), ((1024, 2048), 1, False),
                                         ((1, 2 << 20), 0, False), ((2048, 1024), 0, False),
                                         ((3, 7), 1, False)):
                inp = torch.empty(shape, dtype=torch.bfloat16)
                assert on._gather_uses_chain(inp, dim) == expected, (schedule, shape)
                assert not on._gather_uses_ring(inp, dim)
    # Constructor requests and full-fabric opt-ins use the same policy as per-op overrides.
    os.environ["SIRCL_FORWARD_WINDOW_BYTES"] = "131072"
    for schedule in ("auto", "pieces", "chain", "ring"):
        for key in ("SIRCL_LARGE_SCHEDULE", "SIRCL_GATHER_SCHEDULE", "SIRCL_SCATTER_SCHEDULE"):
            os.environ[key] = schedule
        for rank in range(6):
            s = configured(runtime, routes, layout, rank)
            assert not s.ring_available and not s.chain_available
            assert s.link_available == (schedule != "pieces")
            assert s.large_schedule == s.scatter_schedule == "pieces"
            assert s.gather_schedule == ("auto" if schedule == "ring" else schedule)
        full = "ring:8:0,1,2,3,4,5,6,7"
        os.environ["SIRCL_SUBSET_RING"] = "0"
        off = configured(runtime, routes, full, 0)
        os.environ["SIRCL_SUBSET_RING"] = "1"
        on = configured(runtime, routes, full, 0)
        assert not on.subset_ring
        assert (off.large_schedule, off.gather_schedule, off.scatter_schedule, off._forward_table,
                off.ring_window_bytes, off.chain_available, off.ring_available) == (
                    on.large_schedule, on.gather_schedule, on.scatter_schedule, on._forward_table,
                    on.ring_window_bytes, on.chain_available, on.ring_available)
    print("subset schedule and OFF pieces parity passed")


if __name__ == "__main__":
    main()
