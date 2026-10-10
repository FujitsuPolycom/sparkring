"""The direct all-reduce of a configured cycle-of-eight session with a raised capacity, in a process of its own.

``python direct_driver.py <capacity bytes> <message bytes>`` configures rank 0 of a ``ring:8`` session the way the
constructor does after the setup exchange (``_configure``, ``_plan_relays``, ``_plan_chain``) on an object built
without ``__init__``, with CUDA Python and the CuTe DSL mocked for this process (``teardown_driver._CudaStandIn``)
and a stand-in for the native context, so the session takes the cycle of eight's built-in plan. It then calls the
real ``all_reduce`` on a message stand-in, with the launch steps recorded instead of run, and prints one JSON
object: the algorithm launched, the decisions the session counted, and the ops ``large_reduce_plan`` names for the
same size.
"""

from __future__ import annotations

import json
import sys
import threading


class _Native:
    def set_chain(self, *args) -> None:
        return None

    def set_links(self, *args, **kwargs) -> None:
        return None

    def failed(self) -> bool:
        return False


def main(capacity: int, nbytes: int) -> None:
    from teardown_driver import _CudaStandIn

    sys.meta_path.insert(0, _CudaStandIn())

    import numpy as np
    import torch

    from sparkring_sircl import routes
    from sparkring_sircl.oneshot import runtime

    torch.cuda.current_device = lambda: 0
    torch.cuda.is_current_stream_capturing = lambda: False
    torch.empty_like = lambda tensor, **kwargs: tensor

    layout = routes.Layout.parse("ring:8")
    derived = routes.derive_routes(layout, 2)
    s = runtime.RoceOneshotAllReduce.__new__(runtime.RoceOneshotAllReduce)
    s.rank, s.world_size, s.device = 0, layout.world, torch.device("cuda", 0)
    s.startup_wait_s, s.serving_wait_s = runtime.DEFAULT_STARTUP_WAIT_S, runtime.DEFAULT_SERVING_WAIT_S
    for name in ("chain_order", "chain_index", "relay_safe_bytes", "ring_problem", "poll_rate_per_s"):
        setattr(s, name, None)
    s.wait_regime = "startup"
    s.chain_available = s.link_available = s.ring_available = False
    s._chain_offset = s._link_offset = 0
    s.ring_window_bytes = 0
    s._proxy = _Native()
    s._closed, s._lock, s._profile = False, threading.Lock(), None
    s._ctrl_np = np.zeros(32, dtype=np.int32)
    peer_routes = derived.route_map(0)
    devices = sorted({device for lanes in peer_routes.values() for device in lanes})
    s._configure(capacity, 64 << 10, devices, peer_routes, 3, None, None, None, None, None, None, None, "ring:8")
    route_maps = [derived.route_map(rank) for rank in range(layout.world)]
    s._plan_relays(route_maps)
    s._plan_chain(route_maps)

    launched: list[str] = []
    s._allreduce_eligible = lambda inp, limit: inp.numel() * inp.element_size() <= limit
    s._reduce_launcher = lambda name, dtype, capturing: name
    s._launch_reduce = lambda name, launcher, inp, out, capturing: launched.append(name)

    class Message:
        dtype = torch.bfloat16

        def numel(self) -> int:
            return nbytes // 2

        def element_size(self) -> int:
            return 2

    s.all_reduce(Message())
    plan = s.large_reduce_plan(nbytes, mode="eager")
    print(json.dumps({"launched": launched, "decisions": s._tuning_stats()["decisions"],
                      "large": ["ring" if op.ring else "chain" if op.chain else "pieces" for op in plan]}))


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]))
