"""Which launchers ``prepare`` leaves ready for a CUDA graph capture at which blocks per role, on one rank of a
configured session, in a process of its own.

``python blocks_driver.py <layout>`` configures rank 0 of ``layout`` as ``plan_driver.configured`` does (the
session's real configuration steps on an object built without ``__init__``, CUDA Python and the CuTe DSL mocked
by ``teardown_driver._CudaStandIn``), replaces the CuTe launcher getters of the chain and link kernels with
recorders, and prints one JSON object:

- ``available``: whether the session's chain, chain links and ring can run, and its own blocks per role;
- ``before``: after ``prepare`` as the ring harness's setup calls it (BF16, padded gather, links; then with the
  scatter collectives), every kernel and blocks per role 1, 2 and 4 whose capture-time launcher lookup
  (``capturing=True``, after ``set_op_blocks``) raises, with its message;
- ``after``: the same lookups after ``prepare(..., op_blocks=...)`` naming every kernel at 1, 2 and 4;
- ``compiles`` and ``launchers``: the getter calls and the distinct compiled launchers of these kernels;
- ``op_blocks``: the session's per-op blocks after ``prepare`` (prepare restores them);
- ``refused``: the ValueError messages of malformed ``op_blocks``.
"""

from __future__ import annotations

import contextlib
import json
import sys

COUNTS = (1, 2, 4)


def main(layout_text: str) -> None:
    from teardown_driver import _CudaStandIn

    sys.meta_path.insert(0, _CudaStandIn())

    import torch

    from plan_driver import configured
    from sparkring_sircl import routes
    from sparkring_sircl.oneshot import _chain_cute, _links_cute, _scatter_ops, runtime

    torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.device = lambda device: contextlib.nullcontext()
    calls: list[str] = []

    def getter(kind: str):
        def get(*args, **kwargs):
            calls.append(kind)
            return object()
        return get

    _chain_cute.get_launcher = getter("chain")
    _links_cute.get_gather_launcher = getter("link-gather")
    _links_cute.get_scatter_launcher = getter("link-scatter")
    _links_cute.get_ring_launcher = getter("link-ring")
    _scatter_ops.prepare = lambda session, dtypes: None

    s = configured(runtime, routes, layout_text, 0)
    s.device = torch.device("cpu")
    s._launchers, s._tuning, s.poll_rate_per_s, s.event_trace, s.scatter_available = {}, None, 1.0, 0, True
    s._region = torch.zeros(64, dtype=torch.uint8)
    for name in ("_reduce_launcher", "_all_gather_launcher", "_tiled_gather_launcher", "_aligned_scratch",
                 "_kernel_trace_address", "_gather_scratch"):
        setattr(s, name, lambda *args, **kwargs: None)
    kernels = (*s.link_blocks, "chain_reduce")
    bf16 = torch.bfloat16

    def lookup(kernel: str) -> None:
        if kernel == "chain_reduce":
            s._chain_launcher(bf16, True)
        elif kernel == "chain_gather":
            s._gather_chain_launcher(True)
        elif kernel == "chain_scatter":
            s._scatter_chain_launcher(bf16, True)
        else:
            mode = kernel.split("_", 1)[1]
            s._ring_launcher(mode, None if mode == "gather" else bf16, True)

    def unprepared() -> dict[str, str]:
        missing = {}
        for kernel in kernels:
            for blocks in COUNTS:
                s.set_op_blocks(kernel, blocks)
                try:
                    lookup(kernel)
                except RuntimeError as error:
                    missing[f"{kernel} {blocks}"] = str(error)
                finally:
                    s.set_op_blocks(kernel, None)
        return missing

    out: dict = {"available": {"chain": s.chain_available, "links": s.link_available, "ring": s.ring_available,
                               "blocks": {**s.link_blocks, "chain_reduce": s.chain_blocks}}}
    s.prepare((bf16,), padded_gather=True, links=True)
    s.prepare((bf16,), scatter=True, links=True)
    out["before"] = unprepared()
    s.prepare((bf16,), scatter=True, links=True, op_blocks={kernel: list(COUNTS) for kernel in kernels})
    out["after"] = unprepared()
    out["compiles"] = len(calls)
    out["launchers"] = sum(1 for key in s._launchers if key[0] in ("chain", "link-gather", "link-scatter", "link-ring"))
    out["op_blocks"] = dict(s._op_blocks)
    refused = []
    for bad in ({"chain": [1]}, {"chain_reduce": [0]}, {"ring_reduce": [65]}, {"chain_reduce": [True]},
                {"ring_gather": 2}):
        try:
            s.prepare((bf16,), op_blocks=bad)
            refused.append(None)
        except ValueError as error:
            refused.append(str(error))
    out["refused"] = refused
    print(json.dumps(out))


if __name__ == "__main__":
    main(sys.argv[1])
