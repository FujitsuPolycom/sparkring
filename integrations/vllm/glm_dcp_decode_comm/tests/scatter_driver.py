"""The packed all-to-all's host side in a process of its own, with CUDA Python and the CuTe DSL stood in.

``python scatter_driver.py <mode> <sircl package dir>`` installs SIRCL's CUDA stand-in
(``tests/teardown_driver._CudaStandIn`` of the SIRCL tree: mocks of the ``cuda`` and ``cutlass`` packages),
imports the real ``glm_dcp_decode_comm._scatter_pack_cute`` and ``sparkring_sircl.oneshot._scatter_ops``
over them, and prints one JSON object:

- ``kernel``: the kernel object's validation and compile-time geometry, and ``get_launcher``'s compile-once
  cache and launch-argument checks (the compile call recorded, not run);
- ``session``: ``runtime.scatter_pack_fits`` against the session's op and piece rules, and
  ``runtime.packed_all_to_all``'s launch on a recording session: lock, tuned op, health checks, grid,
  counters, stream rules and the launch arguments, eager and inside a capture.
"""

from __future__ import annotations

import contextlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def _stand_in(sircl: Path) -> None:
    sys.path.insert(0, str(sircl.parent / "tests"))
    sys.path.insert(0, str(sircl.parent))
    from teardown_driver import _CudaStandIn

    sys.meta_path.insert(0, _CudaStandIn())


def kernel() -> dict:
    from glm_dcp_decode_comm import _scatter_pack_cute as K

    out: dict = {"errors": {}}
    for label, args in (("one rank", (1, 0, 512, 2, 128, 2, 8)), ("few threads", (4, 0, 4, 2, 128, 2, 8)),
                        ("slots", (4, 0, 512, 3, 128, 2, 8)), ("heads", (4, 0, 512, 2, 128, 2, 6))):
        try:
            K.ScatterPackLaunch(*args)
            out["errors"][label] = None
        except ValueError as error:
            out["errors"][label] = str(error)
    launch = K.ScatterPackLaunch(4, 1, 512, 2, 128, 2, 8)
    out["geometry"] = [launch._row_packs, launch._row_shift, launch._head_shift, launch._lse_row_packs,
                       launch._lse_shift]
    compiles, launches = [], []

    def compile_launcher(launch, *args, name, cache_key):
        compiles.append([type(launch).__name__, len(args), name, list(cache_key)])
        return lambda *values: launches.append([value if isinstance(value, (int, str)) else list(value)
                                                for value in values])

    K.compile_launcher = compile_launcher
    K.make_pointer = lambda address, assumed_align=16: ["pointer", int(address)]
    K.current_cuda_stream = lambda: "stream"
    key = (4, 1, 512, 2, 128, 2, 8, 0)
    out["prepared_before"] = K.is_prepared(*key)
    run = K.get_launcher(*key)
    out["same_launcher"] = K.get_launcher(*key) is run
    out["prepared_after"] = [K.is_prepared(*key), K.is_prepared(4, 1, 512, 2, 128, 2, 16, 0)]
    rows = 3
    chunk_packs = rows * 8 * 1028 // 16
    run(16, 32, 48, 4 * chunk_packs, 4 * chunk_packs * 16, chunk_packs, rows, 1, 2, 3, chunk_packs * 16, 5, 6, 7, 8,
        9, 10, 11, 12, 13, 14, 4)
    try:
        run(16, 32, 48, 4 * chunk_packs, 0, chunk_packs - 1, rows, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 4)
        out["bad_chunk"] = None
    except ValueError as error:
        out["bad_chunk"] = str(error)
    out["compiles"], out["launches"] = compiles, launches
    return out


class _Lock:
    def __init__(self, events: list) -> None:
        self.events = events

    def __enter__(self):
        self.events.append("lock")

    def __exit__(self, *exc):
        self.events.append("unlock")


def session() -> dict:
    import torch

    state = {"capturing": False}
    torch.cuda.device = lambda device=None: contextlib.nullcontext()
    torch.cuda.is_current_stream_capturing = lambda: state["capturing"]

    from glm_dcp_decode_comm import _scatter_pack_cute as K
    from glm_dcp_decode_comm import layout as L
    from glm_dcp_decode_comm import runtime
    from sparkring_sircl import protocol as proto

    events: list = []

    class Session:
        world_size, rank, device = 4, 1, torch.device("cuda", 0)
        _threads, _blocks, _large_blocks, _packs_per_thread = 512, 8, 32, 2
        _layout = SimpleNamespace(slots=2, flag_stride=128)
        lane_count, spin_limit, scatter_available = 2, 4321, True
        max_size, _slot_bytes, large_piece_bytes, relay_safe_bytes = 2 << 20, 2 << 20, 4 << 20, None
        _recv_base, _flag_base, _send_base, _ctrl_base = 0x10000, 0x20000, 0x30000, 0x40000
        _epoch_address, _poison_address = 0x50000, 0x50040
        _lock = _Lock(events)

        @contextlib.contextmanager
        def _tuned_op(self, collective, nbytes):
            events.append(["tuned", collective, nbytes])
            yield None
            events.append("untuned")

        def check_health(self):
            events.append("health")

        def _counter_addresses(self, grid):
            events.append(["counters", grid])
            return 0x60000 + grid, 0x70000 + grid

        def _order_stream(self, capturing):
            events.append(["order", capturing])

        def _mark_stream(self, capturing):
            events.append(["mark", capturing])

    rt = Session()
    out: dict = {"fits": {}}
    for rows, heads, limit, relay in ((31, 8, 1 << 20, None), (32, 8, 1 << 20, None), (31, 8, 1 << 20, 131072),
                                      (15, 8, 1 << 20, 131072), (16, 8, 1 << 20, 131072), (1, 6, 1 << 20, None),
                                      (63, 4, 1 << 20, None), (64, 4, 1 << 20, None), (1, 8, 16384, None)):
        rt.relay_safe_bytes = relay
        dcp = SimpleNamespace(_runtime=rt, _scatter_op_limit=lambda limit=limit: limit)
        out["fits"][f"{rows},{heads},{limit},{relay}"] = runtime.scatter_pack_fits(dcp, rows, heads)
    rt.relay_safe_bytes = None
    rt.scatter_available = False
    out["fits_unavailable"] = runtime.scatter_pack_fits(SimpleNamespace(_runtime=rt, _scatter_op_limit=lambda: 1 << 20),
                                                        1, 8)
    rt.scatter_available = True

    prepared = {"value": True}
    K.is_prepared = lambda *key: prepared["value"]
    K.get_launcher = lambda *key: (events.append(["launcher", list(key)]) or
                                   (lambda *args: events.append(["launch", list(args)])))
    rows, heads = 3, 8
    out_t = torch.zeros((rows, 4 * heads, L.V_DIM), dtype=torch.bfloat16)
    lse_t = torch.zeros((rows, 4 * heads), dtype=torch.float32)
    recv = torch.zeros((4, L.wire_chunk_bytes(rows, heads)), dtype=torch.uint8)
    out["eager"] = runtime.packed_all_to_all(rt, out_t, lse_t, recv, rows, heads)
    out["eager_events"] = list(events)
    out["pointers"] = [out_t.data_ptr(), lse_t.data_ptr(), recv.data_ptr()]
    chunk_packs = L.wire_chunk_bytes(rows, heads) // 16
    out["grid"] = proto.grid_blocks(4 * chunk_packs, 512, 32, 2)
    events.clear()
    state["capturing"] = True
    out["captured"] = runtime.packed_all_to_all(rt, out_t, lse_t, recv, rows, heads)
    out["captured_events"] = list(events)
    events.clear()
    prepared["value"] = False
    out["unprepared"] = runtime.packed_all_to_all(rt, out_t, lse_t, recv, rows, heads)
    out["unprepared_events"] = list(events)
    return out


def main() -> None:
    mode, sircl = sys.argv[1], Path(sys.argv[2])
    _stand_in(sircl)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    print(json.dumps({"kernel": kernel, "session": session}[mode]()))


if __name__ == "__main__":
    main()
