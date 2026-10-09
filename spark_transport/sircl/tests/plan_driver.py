"""Every rank's plan of a large all-reduce, from the session's real configuration steps, in a process of its own.

``python plan_driver.py <layout> <bytes,...>`` configures every rank's ring session of ``layout`` the way the
constructor does after the setup exchange (``_configure``, ``_plan_relays``, ``_plan_chain``) on objects built
without ``__init__``, with CUDA Python and the CuTe DSL mocked for this process (``teardown_driver._CudaStandIn``;
none of these steps calls them) and a recording stand-in for the native context. For every rank, size and pointer
alignment of the input and the output and mode (eager, graph), it prints what ``large_reduce_staging`` answers:
the ops (kind, offset, bytes) and whether the rank stages its input and its output. One JSON object:
``{"<rank>": {"<bytes>": {"<mode>,<in>,<out>": [ops, stage input, stage output]}}}`` with ``in`` and ``out`` 1
for an aligned pointer.

``python plan_driver.py large <layout> <bytes,...>`` runs the real ``all_reduce_large`` of rank 0 on host tensors
(BF16 and FP32, each message as an aligned tensor and as views one element past alignment for the input, the
output and both), with every launch replaced by a copy of the op's input into its output, as one rank's sum, and
prints one JSON object ``{"<dtype>,<bytes>,<in>,<out>": [ops, output elements, input elements, equal]}``.
"""

from __future__ import annotations

import json
import sys


class _Native:
    """The native calls the chain and link planning and the health check make."""

    def set_chain(self, *args) -> None:
        return None

    def set_links(self, *args, **kwargs) -> None:
        return None

    def failed(self) -> bool:
        return False


def configured(runtime, routes, layout_text: str, rank: int):
    layout = routes.Layout.parse(layout_text)
    derived = routes.derive_routes(layout, 2)
    s = runtime.RoceOneshotAllReduce.__new__(runtime.RoceOneshotAllReduce)
    s.rank, s.world_size = rank, layout.world
    s.startup_wait_s, s.serving_wait_s = runtime.DEFAULT_STARTUP_WAIT_S, runtime.DEFAULT_SERVING_WAIT_S
    for name in ("chain_order", "chain_index", "relay_safe_bytes", "ring_problem", "poll_rate_per_s"):
        setattr(s, name, None)
    s.wait_regime = "startup"
    s.chain_available = s.link_available = s.ring_available = False
    s._chain_offset = s._link_offset = 0
    s.ring_window_bytes = 0
    s._proxy = _Native()
    peer_routes = derived.route_map(rank)
    devices = sorted({device for lanes in peer_routes.values() for device in lanes})
    s._configure(2 << 20, 64 << 10, devices, peer_routes, 3, None, None, None, None, None, None, None, layout_text)
    route_maps = [derived.route_map(other) for other in range(layout.world)]
    s._plan_relays(route_maps)
    s._plan_chain(route_maps)
    return s


def main(layout_text: str, sizes: list[int]) -> None:
    from teardown_driver import _CudaStandIn

    sys.meta_path.insert(0, _CudaStandIn())

    from sparkring_sircl import routes
    from sparkring_sircl.oneshot import runtime

    world = routes.Layout.parse(layout_text).world
    out: dict = {}
    for rank in range(world):
        session = configured(runtime, routes, layout_text, rank)
        per_size = out.setdefault(str(rank), {})
        for nbytes in sizes:
            combos = per_size.setdefault(str(nbytes), {})
            for mode in ("eager", "graph"):
                for input_aligned in (True, False):
                    for output_aligned in (True, False):
                        ops, stage_in, stage_out = session.large_reduce_staging(nbytes, input_aligned,
                                                                                output_aligned, mode=mode)
                        kinds = [["ring" if op.ring else "chain" if op.chain else "padded" if op.padded
                                  else "pieces", op.offset, op.nbytes] for op in ops]
                        combos[f"{mode},{int(input_aligned)},{int(output_aligned)}"] = [kinds, stage_in, stage_out]
    print(json.dumps(out))


def large(layout_text: str, sizes: list[int]) -> None:
    """``all_reduce_large`` on host tensors with recorded launches (see the module docstring)."""
    from teardown_driver import _CudaStandIn

    sys.meta_path.insert(0, _CudaStandIn())

    import contextlib
    import threading

    import numpy as np
    import torch

    from sparkring_sircl import routes
    from sparkring_sircl.oneshot import runtime

    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.device = lambda device: contextlib.nullcontext()

    class Device(torch.Tensor):
        """A host tensor that passes the session's device checks."""

        @property
        def is_cuda(self) -> bool:
            return True

    session = configured(runtime, routes, layout_text, 0)
    session.device = torch.device("cpu")
    session._lock, session._profile, session._closed = threading.Lock(), None, False
    session._ctrl_np = np.zeros(32, dtype=np.int32)
    session._region = torch.zeros(1 << 16, dtype=torch.uint8)
    session._align_buffers = None
    ops: list[str] = []

    def copy(name):
        def launch(*args):
            inp, out = args[1], args[2]
            out.copy_(inp)
            ops.append(name)
        return launch

    session._ring_launcher = lambda *args: None
    session._chain_launcher = lambda *args: None
    session._reduce_launcher = lambda *args: None
    session._launch_ring = copy("ring")
    session._launch_chain = copy("chain")
    session._launch_reduce = lambda name, launcher, inp, out, capturing: copy("pieces")(name, inp, out)

    def offset(tensor):
        base = torch.zeros(tensor.numel() + 8, dtype=tensor.dtype)
        view = base[1:tensor.numel() + 1]
        view.copy_(tensor)
        return view

    out: dict = {}
    for dtype in (torch.bfloat16, torch.float32):
        for nbytes in sizes:
            count = nbytes // torch.empty((), dtype=dtype).element_size()
            values = torch.arange(count, dtype=torch.float32).remainder(251).to(dtype)
            for input_aligned in (True, False):
                for output_aligned in (True, False):
                    ops.clear()
                    src = (values.clone() if input_aligned else offset(values)).as_subclass(Device)
                    dst = None if output_aligned else torch.zeros(count + 8, dtype=dtype)[1:count + 1].as_subclass(Device)
                    try:
                        result = session.all_reduce_large(src, out=dst)
                        found = [list(ops), int(result.numel()), count, bool(torch.equal(result.as_subclass(torch.Tensor),
                                                                                            values))]
                    except Exception as exc:  # noqa: BLE001 - reported
                        found = [list(ops), f"{type(exc).__name__}: {exc}", count, False]
                    out[f"{str(dtype).split('.')[-1]},{nbytes},{int(input_aligned)},{int(output_aligned)}"] = found
    print(json.dumps(out))


if __name__ == "__main__":
    if sys.argv[1] == "large":
        large(sys.argv[2], [int(item) for item in sys.argv[3].split(",")])
    else:
        main(sys.argv[1], [int(item) for item in sys.argv[2].split(",")])
