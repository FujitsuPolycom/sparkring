"""One rank of the ring harness, inside the serving image.

``python -m sparkring_sircl.ring.worker --plan <plan.json> --global-rank N``

The worker joins the configuration's control exchange (gloo over the wired
LAN, ``GLOO_SOCKET_IFNAME``), creates one process subgroup per group of the
configuration, constructs its group's SIRCL session on GPU 0 with the plan's
route map and layout, prepares the kernels, and then runs, for the all-reduce
and the all-gather, eagerly and in CUDA graph replay, every planned size:

1. checked calls: fresh BF16 inputs per call, generated from a seed on every
   rank for every rank, so each rank computes the reference on the host (the
   float32 sum in rank order 0..W-1 rounded once to BF16, or the
   concatenation) and compares the output bit for bit;
2. timed calls: back-to-back calls with CUDA events around each one;
3. counters: the Spark's RDMA and Ethernet error counters before and after.

A global barrier starts every phase, so groups of one configuration run at
the same time; untimed calls after every barrier absorb the ranks' different
wake-up times. With the plan option ``large``, the cases continue with
two-shot all-reduces up to the capacity, ``all_reduce_large`` above it and
``all_gather_large``, each with its algorithm and bus bandwidth. With the
option ``large_reduce_scatter_sizes`` they add reduce-scatters of
``[rows, 4096]`` BF16 along dimension 0 in ops of the large-message piece
(the session's ``reduce_scatter``, or the host functions of
``oneshot/_scatter_ops.py`` when the session class does not offer the
scatter collectives); every checked call compares this rank's rows bit for bit
with the host reference and with the same rows of the session's all-reduce of
the same input in ops of at most the capacity (the rank-ordered float32 sum,
rounded once), and records how many elements differ from ``all_reduce_large``
with its schedule (a chain op rounds once per hop, so it can differ in the
last place).

With the options ``dcp_decode_rows`` and ``dcp_prefill_rows`` (the ``dcp4``
configuration) the cases continue with a decode context parallel group's
exchanges per row count: the all-to-all of vLLM's ``a2a`` combine on the
group's session (``[W * rows, heads * 514]`` BF16, chunk ``p`` to rank ``p``;
the session's ``all_to_all`` or the host functions as above) and, with
``dcp_gathers``, the query and indexer all-gathers (``all_gather_large`` along
dimension 1); decode row counts run eagerly and in CUDA graph replay, prefill
row counts eagerly. With ``world_session`` every rank also holds a session over
the whole ring (``world_layout`` and ``world_peer_routes`` of the plan), as the
tensor-parallel group around the DCP groups, and at decode row counts the
cases add its all-reduce of ``[rows, tp_hidden]`` BF16, checked against the
rank-ordered float32 sum of every world rank's input and timed after a
barrier of the whole world.

With the option ``swing_sizes`` (the ``ring-swing`` configuration) the cases
end with, per size, the Swing all-reduce (collective ``all_reduce_swing``:
``all_reduce(..., algorithm="swing")`` on a session that offers Swing, else the
host functions of ``oneshot/_swing_ops.py``), checked against the Swing
reference of ``testing/collective_models.py`` (partial sums rounded after every
step, identical on every rank), and the session's default all-reduce of the
same message, eagerly and in CUDA graph replay. Swing records carry the
traffic schedule ``swing`` for the summary's bounds.

With the options ``latency_sizes`` and ``post_orders`` (the ``ring-latency``
configuration) the session is built with the first posting order, and the
cases end with, per size, the one-shot and the two-shot all-reduce
(``all_reduce_oneshot`` and ``all_reduce_twoshot``: ``all_reduce(...,
algorithm=...)``), each with every posting order back to back, eagerly and in
CUDA graph replay, checked against the rank-ordered sum. A case switches the
order with the session's ``set_post_order`` before its first call; a session
without it runs the first order only and the result lists the others under
``latency.skipped``. Each record names its order (``post_order``) and this
rank's peers in it (``post_order_peers``) and, where the session counts its
posting time, the lanes posted during the timed calls and their mean posting
time (``posting``), for the summary's latency model.

Placement: before torch loads, the worker pins itself to the performance
cores the CPU policy names (:mod:`sparkring_sircl.cpus`) and gives the native
progress thread a performance core of its own. Eager calls are bound by the
launching thread and small graph replays nearly so, so a launching thread on
an efficiency core slows the whole group; every case records the CPU of the
launching thread before and after its timed calls and the progress thread's
CPU and migrations.

The result is ``rank-N.json`` next to the plan. A transport error ends the
rank (exit code 1) after writing the error; a bit mismatch is recorded and the
run continues (exit code 2); a watchdog ends a rank that does not finish in
time (exit code 3). The process exits with ``os._exit`` so that no teardown
can hang after a failure.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import math
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

from .. import cpus
from . import counters

CANONICAL_DEVICES = ("rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1")


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile."""
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summarize_times(times: list[float]) -> dict[str, float]:
    if not times:
        return {}
    return {"p50_us": round(percentile(times, 0.5), 2), "p90_us": round(percentile(times, 0.9), 2),
            "p99_us": round(percentile(times, 0.99), 2), "min_us": round(min(times), 2),
            "max_us": round(max(times), 2)}


def bandwidths(collective: str, nbytes: int, world: int, p50_us: float | None) -> dict[str, float]:
    """Algorithm and bus bandwidth in GB/s (10^9 bytes per second) at the median time.

    All-reduce (also the world session's ``tp_all_reduce``): the message bytes,
    bus factor ``2 (W - 1) / W``. All-gather: the gathered bytes (``W``
    shards), bus factor ``(W - 1) / W``. Reduce-scatter and all-to-all: the
    input bytes, bus factor ``(W - 1) / W``. Transport ops (one payload to
    every peer): the payload bytes, bus factor ``W - 1``.
    """
    if not p50_us or world < 2:
        return {}
    if collective.startswith("all_reduce") or collective == "tp_all_reduce":
        moved, factor = nbytes, 2 * (world - 1) / world
    elif collective in ("reduce_scatter", "all_to_all"):
        moved, factor = nbytes, (world - 1) / world
    elif collective.startswith("all_gather"):
        moved, factor = nbytes * world, (world - 1) / world
    else:
        moved, factor = nbytes, world - 1
    algbw = moved / (p50_us * 1e-6) / 1e9
    return {"algbw_gbps": round(algbw, 3), "busbw_gbps": round(algbw * factor, 3)}


def _write(path: Path, result: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(result, indent=1), encoding="utf-8")
    os.replace(temporary, path)


def link_collective(collective: str) -> str | None:
    """The link collective (``gather``, ``scatter``, ``reduce``) whose piece a case's link ops use."""
    if collective == "all_gather_large":
        return "gather"
    if collective == "reduce_scatter":
        return "scatter"
    if collective == "all_reduce_large":
        return "reduce"
    return None


class Harness:
    """The rank's run: session construction, cases and records."""

    def __init__(self, plan: dict, global_rank: int, result: dict,
                 placement: cpus.Placement | None = None) -> None:
        import torch
        import torch.distributed as dist

        from .. import routes
        from ..oneshot import AllReduce

        self.torch, self.dist = torch, dist
        self.plan = plan
        self.options = plan["options"]
        self.me = plan["ranks"][global_rank]
        self.global_rank = global_rank
        self.result = result
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        result["environment"] = {
            "hostname": socket.gethostname(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
        }
        dist.init_process_group("gloo", init_method=plan["rendezvous"], rank=global_rank,
                                world_size=plan["world"], timeout=datetime.timedelta(seconds=600))
        handles = [dist.new_group(ranks=list(group["global_ranks"]), backend="gloo") for group in plan["groups"]]
        self.group = plan["groups"][self.me["group"]]
        self.group_index = self.me["group"]
        self.process_group = handles[self.group_index]
        started = time.perf_counter()
        if self.options.get("forward_window") is not None:
            os.environ["SIRCL_FORWARD_WINDOW_BYTES"] = str(self.options["forward_window"])
        if self.options.get("large_piece_bytes"):
            os.environ["SIRCL_LARGE_PIECE_BYTES"] = str(self.options["large_piece_bytes"])
        for name, value in self.options.get("session_env", ()):
            os.environ[str(name)] = str(value)
        if self.options.get("eager_profile"):
            from .plan import EAGER_PROFILE_CALLS

            os.environ["SIRCL_CALL_PROFILE"] = str(EAGER_PROFILE_CALLS)
        self._adapter = None
        self._adapter_times: list[tuple[int, int, str]] = []
        if self.options.get("tune"):
            # The link slots hold the largest swept piece, and every swept stagger beside the session's default
            # slot count, unless the plan sets them.
            given = {str(name) for name, _ in self.options.get("session_env", ())}
            from .. import protocol

            if "SIRCL_LINK_SLOT_BYTES" not in given:
                largest = max(self.options.get("tune_pieces") or (0,))
                if largest > 512 << 10:
                    os.environ["SIRCL_LINK_SLOT_BYTES"] = str(largest)
            if "SIRCL_LINK_SLOTS" not in given:
                world = len(self.group["global_ranks"])
                needed = max([protocol.ring_stagger_slots(world, d) for d in self.options.get("tune_staggers") or (0,)]
                             + [protocol.default_link_slots(world)])
                os.environ["SIRCL_LINK_SLOTS"] = str(needed)
        os.environ["SIRCL_STARTUP_WAIT_S"] = str(self.options.get("startup_wait_s", 300.0))
        os.environ["SIRCL_SERVING_WAIT_S"] = str(self.options.get("serving_wait_s", 20.0))
        progress = placement.progress_cpu_list if placement is not None else None
        # The world session: every rank of the configuration, over the whole ring
        # (the tensor-parallel group around the DCP groups). Every rank builds it
        # before its group session, the order in which vLLM builds the
        # tensor-parallel group and its DCP groups, so the setup exchanges of all
        # ranks stay in one order.
        self.tp_session = None
        self.tp_world = plan["world"]
        if plan.get("world_layout"):
            self.tp_session = AllReduce(
                exchange_group=dist.group.WORLD, device=self.device, max_size=plan["capacity"],
                max_gather_bytes=0, peer_routes=routes.parse_peer_routes(self.me["world_peer_routes"]),
                layout=plan["world_layout"], spin_limit=self.options["spin_limit"],
                lane_check_ms=self.options["lane_check_ms"], gid_index=plan.get("gid_index"),
                progress_cpu=progress,
            )
            self.tp_session.prepare((torch.bfloat16,))
        self.session = AllReduce(
            exchange_group=self.process_group, device=self.device, max_size=plan["capacity"],
            max_gather_bytes=plan["gather_capacity"], peer_routes=routes.parse_peer_routes(self.me["peer_routes"]),
            layout=self.group["layout"], spin_limit=self.options["spin_limit"],
            lane_check_ms=self.options["lane_check_ms"], gid_index=plan.get("gid_index"),
            progress_cpu=progress, post_order=self._construction_order(),
        )
        self.session.prepare((torch.bfloat16,), padded_gather=True, links=True)
        self.scatter_through = None
        tuned_scatter = self.options.get("tune") and ({"reduce_scatter", "all_to_all"}
                                                      & set(self.options.get("tune_collectives") or ()))
        if (self.options.get("large_reduce_scatter_sizes") or self.options.get("dcp_decode_rows")
                or self.options.get("dcp_prefill_rows") or tuned_scatter):
            self.scatter_through = self._prepare_reduce_scatter()
        self.swing_through = self._prepare_swing() if self.options.get("swing_sizes") else None
        self.latency_orders = self._latency_orders()
        result["setup_seconds"] = round(time.perf_counter() - started, 3)
        stats = self.session.stats()
        result["session"] = {key: stats[key] for key in (
            "world_size", "rank", "devices", "gid_indices", "peer_routes", "lane_count", "max_size",
            "dispatch_limit_bytes", "max_gather_bytes", "slot_bytes", "spin_limit", "algorithms_available",
            "large_piece_bytes", "gather_piece_bytes", "relay_safe_bytes", "forward_windows",
            "forward_chunk_bytes", "large_blocks", "startup_wait_s", "serving_wait_s", "poll_rate_per_s",
            "spin_limit_s", "chain_order", "link_available", "link_chunk_bytes", "gather_schedule",
            "scatter_schedule", "link_slots", "link_slot_bytes", "chain_slot_bytes",
        ) if key in stats}
        self.world = self.session.world_size
        self.rank = self.session.rank
        self.devices = [d for d in CANONICAL_DEVICES if (counters.SYSFS / d).exists()]
        self.mismatches = 0
        self.nccl = self._prepare_nccl() if self.options.get("baseline") == "nccl" else None
        self._nccl_done: set[tuple] = set()
        if self.scatter_through is not None:
            result["session"]["reduce_scatter"] = {"through": self.scatter_through}
        if self.swing_through is not None:
            result["session"]["swing"] = {"through": self.swing_through}
        if self.tp_session is not None:
            stats = self.tp_session.stats()
            result["world_session"] = {key: stats[key] for key in (
                "world_size", "rank", "devices", "peer_routes", "lane_count", "max_size", "slot_bytes",
                "algorithms_available", "large_piece_bytes", "relay_safe_bytes", "forward_windows",
            ) if key in stats}

    def _prepare_nccl(self):
        """The NCCL communicator of the baseline rows. The container started with NCCL's environment
        (``nccl.environment``, the image's library preloaded); rank 0's unique id travels over the group's
        gloo group. After one warm-up all-reduce, NCCL's log names its transport; a rank with RoCE devices
        refuses a run whose NCCL connected over sockets."""
        from pathlib import Path

        from .. import routes
        from . import nccl

        layout = routes.Layout.parse(self.group["layout"])
        if nccl.baseline_kind(layout.fabric.kind, self.world) is None:
            raise RuntimeError(f"the NCCL baseline runs on pairs and whole cycles, not on {self.group['layout']}")
        values = nccl.environment(self.plan["lan_interface"], self.plan.get("gid_index"),
                                  self.options.get("nccl_library") or "",
                                  [tuple(item) for item in self.options.get("nccl_env", ())])
        preloaded = os.environ.get("LD_PRELOAD", "")
        os.environ.update({name: value for name, value in values.items() if name != "LD_PRELOAD"})
        library = nccl.Library(values["LD_PRELOAD"])
        token = [library.unique_id() if self.rank == 0 else None]
        self.dist.broadcast_object_list(token, src=self.group["global_ranks"][0], group=self.process_group)
        communicator = library.communicator(self.world, token[0], self.rank)
        self.nccl_version = library.version()
        torch = self.torch
        probe = torch.zeros(4096, dtype=torch.bfloat16, device=self.device)
        communicator.all_reduce(probe.data_ptr(), probe.data_ptr(), probe.numel(), "bfloat16",
                                torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        log_path = os.environ.get("NCCL_DEBUG_FILE", "")
        log_text = Path(log_path).read_text(errors="replace") if log_path and Path(log_path).is_file() else ""
        used = nccl.transport(log_text)
        maps = Path("/proc/self/maps")
        loaded = nccl.loaded_libraries(maps.read_text()) if maps.is_file() else []
        self.nccl_transport = used or "transport unknown"
        self.result["nccl"] = {"library": library.path, "version": self.nccl_version, "transport": used,
                               "preloaded": preloaded, "loaded_libraries": loaded, "environment": values}
        if self.devices and used != "NET/IB":
            raise RuntimeError(f"NCCL {self.nccl_version} connected over {used or 'an unknown transport'} "
                               f"(log {log_path or 'none'}) although this Spark has RoCE devices {self.devices}; "
                               "its rows would not be a valid baseline")
        return communicator

    def _nccl_after(self, collective: str, mode: str, shape: tuple[int, ...], dim: int, case: int,
                    large: bool) -> None:
        """With the NCCL baseline, NCCL's row of a SIRCL case's family, size and mode, once per run."""
        if getattr(self, "nccl", None) is None:
            return
        family = ("all_reduce" if collective in ("all_reduce", "all_reduce_large")
                  else "all_gather" if collective in ("all_gather", "all_gather_large") and dim == 0 else None)
        key = (family, tuple(shape), mode)
        if family is None or key in self._nccl_done:
            return
        self._nccl_done.add(key)
        before = counters.snapshot(self.devices)
        record = self._run_nccl_case(family, mode, tuple(shape), case, large)
        after = counters.snapshot(self.devices)
        record["counters"] = counters.key_deltas(counters.delta(before, after))
        self._record(record, None)

    def _run_nccl_case(self, family: str, mode: str, shape: tuple[int, ...], case: int, large: bool) -> dict:
        """NCCL's all-reduce or all-gather (along dimension 0) of BF16 ``shape``, timed like a SIRCL case.
        All-gathers are checked bit for bit; all-reduces against the float32 sum within the rounding of a
        sum in ``W`` BF16 steps, since NCCL's ring adds in another order than the rank order."""
        torch = self.torch
        reduce = family == "all_reduce"
        out_shape = list(shape)
        if not reduce:
            out_shape[0] *= self.world
        nbytes = math.prod(shape) * 2
        record = {"collective": f"nccl_{family}", "mode": mode, "dtype": "bfloat16", "shape": list(shape), "dim": 0,
                  "bytes": nbytes, "checked": 0, "mismatched_calls": 0, "mismatched_elements": 0,
                  "algorithm": f"NCCL {self.nccl_version} {self.nccl_transport}", "baseline": "nccl"}
        x = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        y = torch.empty(out_shape, dtype=torch.bfloat16, device=self.device)
        count = x.numel()

        def call() -> None:
            stream = torch.cuda.current_stream().cuda_stream
            if reduce:
                self.nccl.all_reduce(x.data_ptr(), y.data_ptr(), count, "bfloat16", stream)
            else:
                self.nccl.all_gather(x.data_ptr(), y.data_ptr(), count, "bfloat16", stream)

        graph = None
        if mode == "graph":
            graph = torch.cuda.CUDAGraph()
            capture_stream = torch.cuda.Stream()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=capture_stream):
                call()
            torch.cuda.synchronize()

        def once() -> None:
            if graph is not None:
                graph.replay()
            else:
                call()

        def wrong_elements(inputs) -> int:
            if not reduce:
                return self._differs(y, torch.cat(inputs, dim=0))
            total = inputs[0].float().clone()
            for tensor in inputs[1:]:
                total += tensor.float()
            got = y.detach().float().cpu()
            # W steps of BF16 rounding (2^-8 relative each) on partial sums of up to about W times the inputs.
            tolerance = self.world * 2.0 ** -8 * (total.abs() + self.world)
            return int(((got - total).abs() > tolerance).sum().item())

        inputs = self._inputs(1 if reduce else 2, shape, case, 0)
        x.copy_(inputs[self.rank])
        once()
        torch.cuda.synchronize()
        record["checked"] = 1
        wrong = wrong_elements(inputs)
        if wrong:
            record["mismatched_calls"] += 1
            record["mismatched_elements"] += wrong
        warmup = 2 if large else self.options["warmup_iterations"]
        for _ in range(warmup):
            once()
        torch.cuda.synchronize()
        self.dist.barrier(group=self.process_group)
        if large:
            count_calls = self.options.get("large_iterations", 20)
        else:
            count_calls = self.options["eager_iterations"] if mode == "eager" else self.options["graph_iterations"]
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count_calls)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count_calls)]
        for index in range(count_calls):
            starts[index].record()
            once()
            ends[index].record()
        torch.cuda.synchronize()
        times = [round(start.elapsed_time(end) * 1000.0, 2) for start, end in zip(starts, ends)]
        record["correct"] = record["mismatched_calls"] == 0
        record["times_us"] = times
        record.update(summarize_times(times))
        record.update(bandwidths(family, nbytes, self.world, record.get("p50_us")))
        self.mismatches += record["mismatched_calls"]
        return record

    def _construction_order(self) -> str | None:
        """The posting order the session is built with: the first planned order, ``farthest`` as this
        rank's explicit peer list on the plan's layout (every session class builds explicit lists)."""
        self._built_order_peers: tuple[int, ...] = ()
        orders = self.options.get("post_orders") or ()
        if not orders:
            return None
        if orders[0] != "farthest":
            return orders[0]
        from .. import posting, routes

        layout = routes.Layout.parse(self.group["layout"])
        self._built_order_peers = posting.resolve("farthest", self.me["group_rank"], layout.world, layout,
                                                  routes.parse_peer_routes(self.me["peer_routes"]))
        return ",".join(str(peer) for peer in self._built_order_peers)

    def _latency_orders(self) -> tuple[str, ...]:
        """The posting orders the latency cases run: every planned order when the session switches
        orders at run time, else the first (the others are listed in ``latency.skipped``)."""
        orders = tuple(self.options.get("post_orders") or ())
        if not self.options.get("latency_sizes") or not orders:
            return ()
        runnable = orders if callable(getattr(self.session, "set_post_order", None)) else orders[:1]
        self.result["latency"] = {
            "orders": list(runnable),
            "skipped": [{"order": order, "reason": "the session class has no set_post_order"}
                        for order in orders if order not in runnable],
        }
        return runnable

    def _prepare_reduce_scatter(self) -> str:
        """Compile the BF16 reduce-scatter and the all-to-all; returns whether calls go through the session or
        the host functions."""
        torch = self.torch
        from ..oneshot import _scatter_ops

        if self.session.scatter_available:
            self.session.prepare((torch.bfloat16,), scatter=True, links=True)
            return "session"
        if not self.session.multi_phase:
            raise RuntimeError("the reduce-scatter needs multi-phase posting (at most two lanes per peer)")
        # A session class without the scatter collectives but with multi-phase
        # posting: mark them available (every rank does the same) and call the
        # host functions of ``_scatter_ops``.
        self.session.scatter_available = True
        _scatter_ops.prepare(self.session, (torch.bfloat16,))
        return "host functions"

    def _prepare_swing(self) -> str:
        """Compile the BF16 Swing all-reduce; returns whether calls go through the session or the host functions."""
        torch = self.torch
        from ..oneshot import _swing_ops

        if self.session._available.get("swing", False):
            self.session.prepare((torch.bfloat16,), algorithms=("swing",))
            return "session"
        if not _swing_ops.available(self.session):
            raise RuntimeError(f"the Swing all-reduce needs a power-of-two group with multi-phase posting; this "
                               f"group has {self.session.world_size} ranks")
        # A session class without Swing: mark it available (every rank does the
        # same) and call the host functions of ``_swing_ops``. The session's own
        # algorithm choice never selects Swing (SIRCL_SWING_ABOVE_BYTES is 0).
        self.session._available["swing"] = True
        _swing_ops.launcher(self.session, torch.bfloat16, capturing=False)
        return "host functions"

    def _swing_all_reduce(self, x, y):
        if self.swing_through == "session":
            return self.session.all_reduce(x, out=y, algorithm="swing")
        from ..oneshot import _swing_ops

        return _swing_ops.all_reduce(self.session, x, out=y)

    def _swing_reference(self, inputs):
        """The Swing result of BF16 ``inputs`` (``testing.collective_models.swing_reference``), every rank alike."""
        torch = self.torch
        import numpy as np

        from ..testing import collective_models

        words = [tensor.contiguous().view(torch.int16).numpy().view(np.uint16) for tensor in inputs]
        result = collective_models.swing_reference(words, "bfloat16")
        return torch.from_numpy(result.view(np.int16).copy()).view(torch.bfloat16).view(inputs[0].shape)

    def _allreduce_rows(self, x, share: int):
        """This rank's rows of the session's all-reduce of ``x`` in ops of at most the capacity.

        Every op is the one-shot or two-shot all-reduce: the rank-ordered
        float32 sum rounded once, the bits a reduce-scatter must reproduce.
        """
        torch = self.torch
        flat = x.reshape(-1)
        out = torch.empty_like(flat)
        step = self.session.max_size // flat.element_size()
        for first in range(0, flat.numel(), step):
            self.session.all_reduce(flat[first:first + step], out=out[first:first + step])
        torch.cuda.synchronize()
        return out.view(x.shape)[self.rank * share:(self.rank + 1) * share]

    def _reduce_scatter(self, x, y):
        if self.scatter_through == "session":
            return self.session.reduce_scatter(x, out=y)
        from ..oneshot import _scatter_ops

        return _scatter_ops.reduce_scatter(self.session, x, out=y)

    def _all_to_all(self, x, y):
        if self.scatter_through == "session":
            return self.session.all_to_all(x, y)
        from ..oneshot import _scatter_ops

        return _scatter_ops.all_to_all(self.session, x, y)

    # -- inputs and references -------------------------------------------------------

    def _seed(self, label: int, rank: int, case: int, iteration: int) -> int:
        return (self.options["seed"] + 1_000_003 * self.group_index + 100_003 * label + 10_007 * rank
                + 101 * case + iteration) % (1 << 62)

    def _inputs(self, label: int, shape: tuple[int, ...], case: int, iteration: int):
        torch = self.torch
        tensors = []
        for rank in range(self.world):
            generator = torch.Generator().manual_seed(self._seed(label, rank, case, iteration))
            tensors.append(torch.randn(shape, generator=generator).to(torch.bfloat16))
        return tensors

    def _world_inputs(self, label: int, shape: tuple[int, ...], case: int, iteration: int):
        """Every world rank's input, the same on every rank whatever its group (the world session's cases)."""
        torch = self.torch
        tensors = []
        for rank in range(self.tp_world):
            seed = (self.options["seed"] + 100_003 * label + 10_007 * rank + 101 * case + iteration) % (1 << 62)
            generator = torch.Generator().manual_seed(seed)
            tensors.append(torch.randn(shape, generator=generator).to(torch.bfloat16))
        return tensors

    def _reference(self, collective: str, inputs, dim: int, mode: str | None = None):
        torch = self.torch
        if collective == "all_reduce_large":
            from .. import references

            nbytes = inputs[0].numel() * inputs[0].element_size()
            return references.large_all_reduce(torch, inputs, self.session.large_reduce_plan(nbytes, mode=mode),
                                               self.session.chain_order)
        if collective == "all_reduce_swing":
            return self._swing_reference(inputs)
        if collective == "all_to_all":
            # Chunk s of the output is chunk ``rank`` of rank s's input.
            return torch.cat([tensor.reshape(self.world, -1)[self.rank] for tensor in inputs]).view(inputs[0].shape)
        if collective.startswith("all_reduce") or collective in ("reduce_scatter", "tp_all_reduce"):
            total = inputs[0].float().clone()
            for tensor in inputs[1:]:
                total += tensor.float()
            summed = total.to(torch.bfloat16)
            if collective == "reduce_scatter":
                share = summed.shape[0] // self.world
                return summed[self.rank * share:(self.rank + 1) * share]
            return summed
        return torch.cat(inputs, dim=dim)

    def _expected(self, collective: str, inputs, dim: int, chained_scatter: bool, out_shape, mode: str | None = None):
        """This rank's expected output: the chain or ring reduce-scatter's reference when the case ran as
        one, else :meth:`_reference` (the ops of ``mode``)."""
        if chained_scatter:
            from .. import references

            build = references.ring_reduce_scatter if chained_scatter == "ring" else references.chain_reduce_scatter
            return build(self.torch, inputs, self.session.chain_order)[self.rank].reshape(out_shape)
        return self._reference(collective, inputs, dim, mode)

    def _differs(self, output, reference) -> int:
        got = output.detach().cpu().contiguous().view(self.torch.int16)
        want = reference.contiguous().view(self.torch.int16)
        return int((got != want).sum().item()) if got.shape == want.shape else int(want.numel())

    # -- one case ---------------------------------------------------------------------

    def _call(self, collective: str, x, y, dim: int):
        if self.options.get("eager_path") == "adapter" and collective in PROFILED:
            return self._adapter_call(collective, x, y, dim)
        if collective == "all_reduce":
            return self.session.all_reduce(x, out=y)
        if collective == "all_reduce_large":
            return self.session.all_reduce_large(x, out=y)
        if collective == "all_gather_large":
            return self.session.all_gather_large(x, dim=dim, out=y)
        if collective == "reduce_scatter":
            return self._reduce_scatter(x, y)
        if collective == "all_to_all":
            return self._all_to_all(x, y)
        if collective == "all_reduce_swing":
            return self._swing_all_reduce(x, y)
        if collective in ("all_reduce_oneshot", "all_reduce_twoshot"):
            return self.session.all_reduce(x, out=y, algorithm=collective[len("all_reduce_"):])
        if collective == "tp_all_reduce":
            return self.tp_session.all_reduce(x, out=y)
        return self.session.all_gather(x, dim=dim, out=y)

    def _adapter_call(self, collective: str, x, y, dim: int):
        """One collective as the vLLM adapter issues it: the planner's plan, then the executor's session ops
        (its outputs allocated as the adapter allocates them), copied into ``y``. Eager calls record the
        planner's and the executor's host times."""
        from ..vllm import executor, planner

        if self._adapter is None:
            limits = planner.SessionLimits.of(self.session, reduce_dtypes=("bfloat16",))
            self._adapter = (limits, planner.Policy(None, "never", "sircl"),
                             executor.ColumnGather.from_environment())
        limits, policy, column_gather = self._adapter
        capturing = self.torch.cuda.is_current_stream_capturing()
        started = time.perf_counter_ns()
        meta = planner.TensorMeta.of(x)
        if collective.startswith("all_reduce"):
            plan = planner.plan_all_reduce(meta, limits, policy, capturing=capturing)
            planned = time.perf_counter_ns()
            result = executor.all_reduce(self.session, plan, x, limits, policy, capturing=capturing)
        else:
            plan = planner.plan_all_gather(meta, dim, limits, policy, capturing=capturing)
            planned = time.perf_counter_ns()
            result = executor.all_gather(self.session, plan, x, dim, self.world, column_gather=column_gather)
        done = time.perf_counter_ns()
        method = plan.method
        if not collective.startswith("all_reduce") and column_gather.last_route is not None:
            # A column gather staged on the session's links (SIRCL_COLUMN_GATHER), e.g. large+column-ring.
            method += f"+column-{column_gather.last_route}"
        if not capturing:
            self._adapter_times.append((planned - started, done - planned, method))
        y.copy_(result)
        return y

    def _progress(self, session=None) -> dict:
        stats = (session or self.session).stats()
        return {"proxy_cpu": stats.get("proxy_cpu"), "migrations": stats.get("proxy_cpu_migrations", 0),
                "post_ns": stats.get("post_ns_total"), "post_lanes": stats.get("post_lanes_total"),
                "forward_waits": stats.get("forward_waits"), "forward_wait_ns": stats.get("forward_wait_ns"),
                "forward_proven_bytes": stats.get("forward_proven_bytes")}

    def run_case(self, collective: str, mode: str, shape: tuple[int, ...], dim: int, case: int,
                 large: bool = False, variant: dict | None = None) -> dict:
        """One case; ``variant`` sets the large-message schedule and chain chunk for its duration. A
        variant that fixes a setting (:func:`forced`) runs with the session's tuning table suspended."""
        session = self.session
        if forced(variant) and callable(getattr(session, "untuned", None)):
            with session.untuned():
                record = self._run_variant(collective, mode, shape, dim, case, large, variant)
            if getattr(session, "_tuning", None) is not None:
                record["tuning"] = "suspended: the row forces its own settings"
            return record
        return self._run_variant(collective, mode, shape, dim, case, large, variant)

    def _run_variant(self, collective: str, mode: str, shape: tuple[int, ...], dim: int, case: int,
                     large: bool, variant: dict | None) -> dict:
        session = self.session
        saved = (session.large_schedule, session.chain_chunk_bytes, getattr(session, "gather_schedule", None),
                 getattr(session, "link_chunk_bytes", None), getattr(session, "scatter_schedule", None))
        # The case's link piece is its collective's own (link_chunk_for); a session without per-collective
        # pieces takes it as the session's piece.
        kind = link_collective(collective)
        own_pieces = callable(getattr(session, "link_chunk_for", None)) and kind is not None
        saved_piece = dict(getattr(session, "_link_chunks", {})).get(kind) if own_pieces else None
        # A ring row measures the ring at its size: the session's ring minimum does not apply to it.
        ring_row = bool(variant) and "ring" in (variant.get("schedule"), variant.get("gather_schedule"),
                                                variant.get("scatter_schedule"))
        saved_ring_mins = None
        if ring_row and callable(getattr(session, "set_ring_min_bytes", None)):
            saved_ring_mins = {kind: session.ring_min_for(kind) for kind in ("reduce", "gather", "scatter")}
            session.set_ring_min_bytes(0)
        if variant and variant.get("post_order") and callable(getattr(session, "set_post_order", None)):
            session.set_post_order(variant["post_order"])
        saved_grid = None
        if variant and variant.get("grid"):
            saved_grid = (session.large_blocks, session.blocks)
            session.set_large_blocks(variant["grid"])
            session.set_blocks(variant["grid"])
        saved_stagger = None
        if variant and variant.get("stagger") is not None:
            saved_stagger = session.ring_stagger
            session.set_ring_stagger(variant["stagger"])
        saved_gather_stagger = None
        if variant and variant.get("gather_stagger") is not None:
            saved_gather_stagger = session.ring_gather_stagger
            session.set_ring_gather_stagger(variant["gather_stagger"])
        if variant:
            session.large_schedule = variant.get("schedule", saved[0])
            if variant.get("chunk"):
                session.set_chain_chunk_bytes(variant["chunk"])
            if variant.get("gather_schedule"):
                session.gather_schedule = variant["gather_schedule"]
            if variant.get("scatter_schedule"):
                session.scatter_schedule = variant["scatter_schedule"]
            if variant.get("link_chunk"):
                if own_pieces:
                    session.set_link_chunk_bytes(variant["link_chunk"], collective=kind)
                else:
                    session.set_link_chunk_bytes(variant["link_chunk"])
        try:
            return self._run_case(collective, mode, shape, dim, case, large)
        finally:
            session.large_schedule = saved[0]
            session.set_chain_chunk_bytes(saved[1])
            if saved[2] is not None:
                session.gather_schedule = saved[2]
                session.set_link_chunk_bytes(saved[3])
            if own_pieces:
                session.set_link_chunk_bytes(saved_piece or 0, collective=kind)
            if saved_ring_mins is not None:
                for kind, nbytes in saved_ring_mins.items():
                    session.set_ring_min_bytes(nbytes, collective=kind)
            if saved[4] is not None:
                session.scatter_schedule = saved[4]
            if saved_grid is not None:
                session.set_large_blocks(saved_grid[0])
                session.set_blocks(saved_grid[1])
            if saved_stagger is not None:
                session.set_ring_stagger(saved_stagger)
            if saved_gather_stagger is not None:
                session.set_ring_gather_stagger(saved_gather_stagger)

    def _link_piece(self, collective: str) -> int | None:
        """The link piece the session uses for ``collective``'s link ops."""
        kind = link_collective(collective)
        pick = getattr(self.session, "link_chunk_for", None)
        if callable(pick) and kind is not None:
            return pick(kind)
        return getattr(self.session, "link_chunk_bytes", None)

    def _run_case(self, collective: str, mode: str, shape: tuple[int, ...], dim: int, case: int,
                  large: bool = False) -> dict:
        torch = self.torch
        tp = collective == "tp_all_reduce"
        session = self.tp_session if tp else self.session
        world, rank = (self.tp_world, self.global_rank) if tp else (self.world, self.rank)
        reduce = collective.startswith("all_reduce") or tp
        scatter = collective == "reduce_scatter"
        exchange = collective == "all_to_all"
        label = 5 if tp else 4 if exchange else 3 if scatter else 1 if reduce else 2
        out_shape = list(shape)
        if scatter:
            out_shape[0] //= self.world
        elif not reduce and not exchange:
            out_shape[dim] *= self.world
        nbytes = math.prod(shape) * 2
        record = {"collective": collective, "mode": mode, "dtype": "bfloat16", "shape": list(shape), "dim": dim,
                  "bytes": nbytes, "checked": 0, "mismatched_calls": 0, "mismatched_elements": 0}
        chained_scatter = False
        if tp:
            record["world"] = world
            record["algorithm"] = f"{session.select_algorithm(nbytes, mode=mode)} on the world session of {world} ranks"
        elif exchange:
            from ..oneshot import _scatter_ops

            piece = _scatter_ops.piece_bytes(self.session, nbytes // self.world)
            record["algorithm"] = f"scatter ops of {piece} bytes per peer through the {self.scatter_through}"
        elif collective == "all_reduce_swing":
            record["algorithm"] = f"swing through the {self.swing_through}"
            record["traffic"] = "swing"
        elif collective in ("all_reduce_oneshot", "all_reduce_twoshot"):
            record["algorithm"] = collective[len("all_reduce_"):]
            record["traffic"] = record["algorithm"]
        elif collective == "all_reduce":
            record["algorithm"] = self.session.select_algorithm(nbytes, mode=mode)
            record["traffic"] = "oneshot" if record["algorithm"] == "oneshot" else "twoshot"
        elif collective == "all_reduce_large":
            plan = self.session.large_reduce_plan(nbytes, mode=mode)
            chain = any(piece.chain for piece in plan)
            ring = any(getattr(piece, "ring", False) for piece in plan)
            record["schedule"] = self.session.large_schedule
            record["algorithm"] = (f"ring, pieces of {self._link_piece(collective)}" if ring
                                   else f"chain, chunks of {self.session.chain_chunk_bytes}" if chain
                                   else f"pieces of {self.session.large_piece_bytes}")
            record["traffic"] = "ring" if ring else "chain" if chain else "twoshot"
        elif collective == "all_gather_large":
            probe = torch.empty(shape, dtype=torch.bfloat16, device="meta")
            uses_chain = getattr(self.session, "gather_uses_chain", None)
            uses_ring = getattr(self.session, "gather_uses_ring", None)
            if callable(uses_ring) and uses_ring(probe, dim, mode=mode):
                record["algorithm"] = f"ring, pieces of {self._link_piece(collective)}"
                record["traffic"] = "ring"
            elif callable(uses_chain) and uses_chain(probe, dim, mode=mode):
                record["algorithm"] = f"chain, pieces of {self._link_piece(collective)}"
                record["traffic"] = "chain"
            else:
                record["algorithm"] = f"pieces of {self.session.gather_piece_bytes}"
                record["traffic"] = "scatter"
            record["schedule"] = getattr(self.session, "gather_schedule", None)
        elif scatter:
            from ..oneshot import _scatter_ops

            chunk = nbytes // self.world
            uses_chain = getattr(self.session, "scatter_uses_chain", None)
            uses_ring = getattr(self.session, "scatter_uses_ring", None)
            probe = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
            ring_scatter = bool(callable(uses_ring) and self.scatter_through == "session"
                                and uses_ring(probe, mode=mode))
            chained_scatter = (not ring_scatter and bool(callable(uses_chain) and self.scatter_through == "session"
                                                         and uses_chain(probe, mode=mode)))
            if ring_scatter:
                chained_scatter = "ring"
                record["algorithm"] = f"ring, pieces of {self._link_piece(collective)}"
                record["traffic"] = "ring"
            elif chained_scatter:
                record["algorithm"] = f"chain, pieces of {self._link_piece(collective)}"
                record["traffic"] = "chain"
            else:
                record["algorithm"] = (f"scatter ops of {_scatter_ops.piece_bytes(self.session, chunk)} bytes per "
                                       f"peer through the {self.scatter_through}")
                record["traffic"] = "scatter"
            record["schedule"] = getattr(self.session, "scatter_schedule", None)
            record["allreduce_mismatched_elements"] = 0
            plan = getattr(self.session, "large_reduce_plan", None)
            if callable(plan):
                record["all_reduce_large_schedule"] = (
                    "chain" if any(getattr(piece, "chain", False) for piece in plan(nbytes, aligned=True, mode=mode))
                    else "pieces")
        if exchange:
            record["traffic"] = "scatter"
        if record.get("traffic") in ("chain", "ring"):
            record["chain_order"] = list(self.session.chain_order)
        x = torch.empty(shape, dtype=torch.bfloat16, device=self.device)
        y = torch.empty(out_shape, dtype=torch.bfloat16, device=self.device)
        graph = None
        if mode == "graph":
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            torch.cuda.synchronize()
            with session.capture():
                with torch.cuda.graph(graph, stream=stream):
                    self._call(collective, x, y, dim)
            torch.cuda.synchronize()

        def once() -> None:
            if graph is not None:
                graph.replay()
            else:
                self._call(collective, x, y, dim)

        checks = 1 if large else self.options["correctness_iterations"]
        make_inputs = self._world_inputs if tp else self._inputs
        for iteration in range(checks):
            inputs = make_inputs(label, shape, case, iteration)
            x.copy_(inputs[rank])
            once()
            torch.cuda.synchronize()
            session.check_health()
            wrong = self._differs(y, self._expected(collective, inputs, dim, chained_scatter, out_shape, mode))
            if large:
                # The same inputs again: a collective's bits do not depend on timing.
                first = y.detach().clone()
                once()
                torch.cuda.synchronize()
                session.check_health()
                repeat = int((first.view(torch.int16) != y.view(torch.int16)).sum().item())
                record["repeat_differing_elements"] = record.get("repeat_differing_elements", 0) + repeat
                wrong += repeat
            if scatter:
                share = out_shape[0]
                rows = self._allreduce_rows(x, share)
                differ = int((rows.view(torch.int16) != y.view(torch.int16)).sum().item())
                if chained_scatter:
                    # A chain reduce-scatter rounds at every hop: its rows differ from the
                    # one-shot rows in the last place, which is counted, not a mismatch.
                    record["oneshot_rows_differing_elements"] = (
                        record.get("oneshot_rows_differing_elements", 0) + differ)
                else:
                    record["allreduce_mismatched_elements"] += differ
                    wrong += differ
                large_rows = self.session.all_reduce_large(x)[self.rank * share:(self.rank + 1) * share]
                torch.cuda.synchronize()
                record["all_reduce_large_differing_elements"] = int(
                    (large_rows.view(torch.int16) != y.view(torch.int16)).sum().item())
            record["checked"] += 1
            if wrong:
                record["mismatched_calls"] += 1
                record["mismatched_elements"] += wrong
        timed_inputs = make_inputs(label, shape, case, 1000)
        x.copy_(timed_inputs[rank])
        warmup = 2 if large else self.options["warmup_iterations"]
        for _ in range(warmup):
            once()
        torch.cuda.synchronize()
        # The world session's cases start together on every rank of the world.
        self.dist.barrier(group=None if tp else self.process_group)
        for _ in range(self.options.get("post_barrier_warmup", 0)):
            once()
        torch.cuda.synchronize()
        progress_before = self._progress(session)
        cpu_before = cpus.current_cpu()
        if large:
            count = self.options.get("large_iterations", 20)
        else:
            count = self.options["eager_iterations"] if mode == "eager" else self.options["graph_iterations"]
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(count)]
        profiled = mode == "eager" and collective in PROFILED
        profile = getattr(session, "call_profile", None) if profiled and self.options.get("eager_profile") else None
        if profile is not None:
            profile(reset=True)
        self._adapter_times = []
        for index in range(count):
            starts[index].record()
            once()
            ends[index].record()
        cpu_after = cpus.current_cpu()
        torch.cuda.synchronize()
        if profile is not None:
            summary = profile(reset=True)
            record["call_profile"] = summary["rows"] if summary else []
        if profiled and self._adapter_times:
            plans = [entry[0] for entry in self._adapter_times]
            executes = [entry[1] for entry in self._adapter_times]
            record["adapter_profile"] = {"plan_us": round(sorted(plans)[len(plans) // 2] / 1e3, 2),
                                         "execute_us": round(sorted(executes)[len(executes) // 2] / 1e3, 2),
                                         "methods": sorted({entry[2] for entry in self._adapter_times})}
        session.check_health()
        progress_after = self._progress(session)
        record["placement"] = {
            "main_cpu_start": cpu_before, "main_cpu_end": cpu_after,
            "proxy_cpu": progress_after["proxy_cpu"],
            "proxy_migrations": progress_after["migrations"] - progress_before["migrations"],
        }
        if progress_before.get("post_lanes") is not None and progress_after.get("post_lanes") is not None:
            lanes = progress_after["post_lanes"] - progress_before["post_lanes"]
            spent = progress_after["post_ns"] - progress_before["post_ns"]
            if lanes > 0:
                record["posting"] = {"lanes": lanes, "ns": spent, "us_per_lane": round(spent / lanes / 1e3, 4)}
        if progress_before.get("forward_waits") is not None and progress_after.get("forward_waits") is not None:
            # Waits of relayed lanes for room in their forward windows during the timed calls, per call.
            record["forward"] = {
                "waits_per_call": round((progress_after["forward_waits"] - progress_before["forward_waits"]) / count, 3),
                "wait_us_per_call": round((progress_after["forward_wait_ns"] - progress_before["forward_wait_ns"])
                                          / count / 1e3, 3),
                "proven_kib_per_call": round((progress_after["forward_proven_bytes"]
                                              - progress_before["forward_proven_bytes"]) / count / 1024, 3)}
        times = [round(start.elapsed_time(end) * 1000.0, 2) for start, end in zip(starts, ends)]
        wrong = self._differs(y, self._expected(collective, timed_inputs, dim, chained_scatter, out_shape, mode))
        record["checked"] += 1
        if wrong:
            record["mismatched_calls"] += 1
            record["mismatched_elements"] += wrong
        record["correct"] = record["mismatched_calls"] == 0
        if large and getattr(session, "event_trace", 0) and record.get("traffic") in ("chain", "ring"):
            record["event_trace"] = self._traced_call(session, once)
        record["times_us"] = times
        record.update(summarize_times(times))
        record.update(bandwidths(collective, nbytes, world, record.get("p50_us")))
        chain_row = str(record.get("algorithm", "")).startswith("chain")
        for size, target, stretch in self.options.get("chain_targets" if chain_row else "targets", ()):
            if reduce and size == nbytes:
                record["target_ms"], record["stretch_ms"] = target, stretch
        if str(record.get("algorithm", "")).startswith("ring"):
            family = "all_reduce" if reduce else "reduce_scatter" if scatter else "all_gather"
            for name, size, target, stretch in self.options.get("ring_targets", ()):
                if name == family and size == nbytes:
                    record["target_ms"], record["stretch_ms"] = target, stretch
        if collective == "all_gather_large" and chain_row:
            for size, target, stretch in self.options.get("gather_targets", ()):
                if size == nbytes:
                    record["target_ms"], record["stretch_ms"] = target, stretch
        for size, target, stretch in self.options.get("reduce_scatter_targets", ()):
            if scatter and size == nbytes:
                record["target_ms"], record["stretch_ms"] = target, stretch
        self.mismatches += record["mismatched_calls"]
        return record

    def _traced_call(self, session, once) -> dict:
        """The event trace of one more call of the case (every rank of the group runs it)."""
        torch = self.torch
        session.event_trace_records()
        once()
        torch.cuda.synchronize()
        session.check_health()
        trace = session.event_trace_records()
        return {"offset_ns": trace.get("offset_ns"), "offset_error_ns": trace.get("offset_error_ns"),
                "lost": trace["lost"], "records": [list(record) for record in trace["records"]]}

    def cases(self, collective: str) -> list[tuple[tuple[int, ...], int]]:
        if collective == "all_reduce":
            return [((size // 2,), 0) for size in self.options["allreduce_sizes"]]
        flat = [((size // 2,), 0) for size in self.options["allgather_sizes"]]
        rows = [((rows, cols), 1) for rows, cols in self.options["allgather_shapes"]]
        return flat + rows

    def _sweep(self, kind: str) -> tuple:
        """The link pieces swept for one link collective: its own list, else the common one, else the
        session's piece (None)."""
        return tuple(self.options.get(f"{kind}_link_chunks", ()) or self.options.get("link_chunks", ()) or (None,))

    def large_cases(self) -> list[tuple[str, tuple[int, ...], int, dict | None]]:
        """(collective, BF16 shape, dim, schedule variant) of the large-message cases."""
        if not self.options.get("large"):
            return []
        found = []
        for size in self.options.get("large_allreduce_sizes", ()):
            for schedule in self.options.get("large_schedules", ("auto",)):
                if schedule == "chain":
                    for chunk in self.options.get("chain_chunks", ()) or (None,):
                        found.append(("all_reduce_large", (size // 2,), 0, {"schedule": "chain", "chunk": chunk}))
                    continue
                if schedule == "ring":
                    for chunk in self._sweep("reduce"):
                        found.append(("all_reduce_large", (size // 2,), 0, {"schedule": "ring", "link_chunk": chunk}))
                    continue
                collective = "all_reduce" if size <= self.session.max_size else "all_reduce_large"
                found.append((collective, (size // 2,), 0, {"schedule": schedule}))
        for size in self.options.get("large_allgather_sizes", ()):
            found.append(("all_gather_large", (size // 2,), 0, None))
            if size % 4096 == 0:
                found.append(("all_gather_large", (size // 4096, 2048), 1, None))
        for size in self.options.get("chain_gather_sizes", ()):
            for schedule in self.options.get("gather_schedules", ("auto",)):
                if schedule in ("chain", "ring"):
                    for chunk in self._sweep("gather"):
                        found.append(("all_gather_large", (size // 8192, 4096), 0,
                                      {"gather_schedule": schedule, "link_chunk": chunk}))
                    continue
                found.append(("all_gather_large", (size // 8192, 4096), 0, {"gather_schedule": schedule}))
        for size in self.options.get("large_reduce_scatter_sizes", ()):
            for schedule in self.options.get("scatter_schedules", ("auto",)):
                if schedule in ("chain", "ring"):
                    for chunk in self._sweep("scatter"):
                        found.append(("reduce_scatter", (size // 8192, 4096), 0,
                                      {"scatter_schedule": schedule, "link_chunk": chunk}))
                    continue
                found.append(("reduce_scatter", (size // 8192, 4096), 0, {"scatter_schedule": schedule}))
        return found

    def dcp_cases(self) -> list[tuple[str, tuple[int, ...], int, int, bool]]:
        """(collective, BF16 shape, dim, rows, decode) of the DCP cases; decode cases also run in graph replay.

        Per row count: the a2a combine's all-to-all, ``[W * rows, heads * 514]``
        BF16 (the bytes of vLLM's ``[W, rows, heads, 514]`` buffer); with
        ``dcp_gathers`` the query all-gather, ``[rows, heads * 576]`` along
        dimension 1, and the indexer all-gather, rows of 16 KiB along
        dimension 1; at decode row counts, with the world session, its
        all-reduce of ``[rows, tp_hidden]``.
        """
        from .plan import DCP_A2A_VALUES, DCP_HEADS, DCP_INDEXER_ROW_BYTES, DCP_QUERY_VALUES, TP_HIDDEN

        options = self.options
        heads = options.get("dcp_heads") or DCP_HEADS
        hidden = options.get("tp_hidden") or TP_HIDDEN
        found = []
        rows_list = ([(rows, True) for rows in options.get("dcp_decode_rows") or ()]
                     + [(rows, False) for rows in options.get("dcp_prefill_rows") or ()])
        for rows, decode in rows_list:
            found.append(("all_to_all", (self.world * rows, heads * DCP_A2A_VALUES), 0, rows, decode))
            if options.get("dcp_gathers"):
                found.append(("all_gather_large", (rows, heads * DCP_QUERY_VALUES), 1, rows, decode))
                found.append(("all_gather_large", (rows, DCP_INDEXER_ROW_BYTES // 2), 1, rows, decode))
            if decode and self.tp_session is not None:
                found.append(("tp_all_reduce", (rows, hidden), 0, rows, decode))
        return found

    def swing_cases(self) -> list[tuple[str, tuple[int, ...], int]]:
        """(collective, BF16 shape, dim) of the Swing cases: per size, Swing and the session's default
        all-reduce of the same message."""
        found = []
        for size in self.options.get("swing_sizes") or ():
            found.append(("all_reduce_swing", (size // 2,), 0))
            found.append(("all_reduce", (size // 2,), 0))
        return found

    def latency_cases(self) -> list[tuple[str, tuple[int, ...], int, str]]:
        """(collective, BF16 shape, dim, posting order) of the latency cases: per size, the one-shot and the
        two-shot all-reduce, each with every runnable posting order back to back."""
        found = []
        for size in self.options.get("latency_sizes") or ():
            for collective in ("all_reduce_oneshot", "all_reduce_twoshot"):
                for order in self.latency_orders:
                    found.append((collective, (size // 2,), 0, order))
        return found

    def _set_large_blocks(self, cap: int | None) -> None:
        """Every session's grid cap of two-shot and large-message launches for the cases that follow."""
        if cap is None:
            return
        for session in (self.session, self.tp_session):
            if session is None:
                continue
            setter = getattr(session, "set_large_blocks", None)
            if not callable(setter):
                raise RuntimeError("this session class cannot change its grid cap between ops; run once per cap "
                                   "with --session-env SIRCL_LARGE_BLOCKS=<blocks> instead of --large-blocks")
            setter(int(cap))

    def _record(self, record: dict, cap: int | None) -> None:
        if cap is not None:
            record["large_blocks"] = int(cap)
        self.result["runs"].append(record)

    def run(self) -> None:
        totals_before = counters.snapshot(self.devices)
        self.dist.barrier()
        # Every rank compiled and prepared: from here a lagging peer means a failure.
        self.session.enter_serving()
        if self.tp_session is not None:
            self.tp_session.enter_serving()
        # With --large-blocks every case that launches the two-shot or large-message grid runs once per grid
        # cap; the small cases and the one-shot latency rows, whose grids the cap does not set, run once.
        caps = tuple(self.options.get("large_blocks") or ()) or (None,)
        if self.options.get("tune"):
            self._run_tune()
            caps = ()
        for index, cap in enumerate(caps):
            self._set_large_blocks(cap)
            sweep = (self.session.untuned() if cap is not None and callable(getattr(self.session, "untuned", None))
                     else contextlib.nullcontext())
            with sweep:
                self._run_cases(cap, first=index == 0)
        totals_after = counters.snapshot(self.devices)
        self.result["counters_total"] = counters.key_deltas(counters.delta(totals_before, totals_after))
        self.result["counter_sources"] = {device: totals_after[device]["source"] for device in totals_after}
        self.result["proxy"] = {key: value for key, value in self.session.stats().items()
                                if key in ("ops_posted", "writes_completed", "last_seq", "proxy_cpu",
                                           "proxy_cpu_migrations", "bytes_posted_per_hca",
                                           "later_phases_posted", "forward_chunks_posted",
                                           "forward_max_unacked_bytes", "chain_ops", "chain_chunks_posted",
                                           "chain_credits_sent", "chain_bytes_posted", "chain_order",
                                           "link_ops", "link_items_posted", "link_credits_sent",
                                           "link_bytes_posted", "link_window_chunks_posted",
                                           "ring_window_bytes")}
        self.result["tuning"] = self.session.stats().get("tuning")
        if self.tp_session is not None:
            self.result["world_tuning"] = self.tp_session.stats().get("tuning")
            self.result["world_proxy"] = {key: value for key, value in self.tp_session.stats().items()
                                          if key in ("ops_posted", "writes_completed", "last_seq", "proxy_cpu",
                                                     "later_phases_posted", "forward_chunks_posted",
                                                     "forward_max_unacked_bytes")}
        self.dist.barrier()
        if getattr(self, "nccl", None) is not None:
            self.nccl.close()
        self.session.close()
        if self.tp_session is not None:
            self.tp_session.close()

    def tune_candidates(self, collective: str, size: int) -> list[tuple[dict, str, tuple[int, ...], dict | None]]:
        """(choice, harness collective, BF16 shape, variant) of every SIRCL candidate of ``collective`` at
        ``size`` bytes per rank that this session can run; NCCL's candidate is added by the caller."""
        session, options, world = self.session, self.options, self.world
        grids = tuple(options.get("tune_grids") or ()) or (None,)
        pieces = tuple(options.get("tune_pieces") or ())
        staggers = tuple(options.get("tune_staggers") or (0,))
        large = size >= int(options.get("tune_large_from", 262144))
        shape = (size // 2,)
        found: list[tuple[dict, str, tuple[int, ...], dict | None]] = []

        def grid_choice(base: dict, grid) -> dict:
            return {**base, "grid": grid} if grid else dict(base)

        ring = bool(getattr(session, "ring_available", False))
        if collective == "all_reduce":
            if size <= session.max_size:
                for algorithm in ("oneshot", "twoshot"):
                    if session._available.get(algorithm):
                        for grid in grids:
                            found.append((grid_choice({"algorithm": algorithm}, grid), f"all_reduce_{algorithm}",
                                          shape, {"grid": grid} if grid else None))
                if self.swing_through == "session":
                    found.append(({"algorithm": "swing"}, "all_reduce_swing", shape, None))
            if large:
                if size > session.max_size:
                    for grid in grids:
                        found.append((grid_choice({"schedule": "pieces"}, grid), "all_reduce_large", shape,
                                      {"schedule": "pieces", **({"grid": grid} if grid else {})}))
                if session.chain_available:
                    for piece in pieces:
                        if piece <= session.chain_slot_bytes:
                            found.append(({"schedule": "chain", "piece": piece}, "all_reduce_large", shape,
                                          {"schedule": "chain", "chunk": piece}))
                if ring and size % (16 * world) == 0:
                    for piece in pieces:
                        for stagger in staggers:
                            for gather_stagger in staggers:
                                found.append(({"schedule": "ring", "piece": piece, "stagger": stagger,
                                               "gather_stagger": gather_stagger}, "all_reduce_large", shape,
                                              {"schedule": "ring", "link_chunk": piece, "stagger": stagger,
                                               "gather_stagger": gather_stagger}))
        elif collective == "all_gather":
            if session.max_gather_bytes > 0:
                for grid in grids:
                    found.append((grid_choice({"schedule": "pieces"}, grid), "all_gather_large", shape,
                                  {"gather_schedule": "pieces", **({"grid": grid} if grid else {})}))
                if large and session.link_available:
                    for piece in pieces:
                        found.append(({"schedule": "chain", "piece": piece}, "all_gather_large", shape,
                                      {"gather_schedule": "chain", "link_chunk": piece}))
                        if ring:
                            for gather_stagger in staggers:
                                found.append(({"schedule": "ring", "piece": piece, "gather_stagger": gather_stagger},
                                              "all_gather_large", shape,
                                              {"gather_schedule": "ring", "link_chunk": piece,
                                               "gather_stagger": gather_stagger}))
        elif collective in ("reduce_scatter", "all_to_all"):
            if size % (16 * world) or self.scatter_through is None:
                return []
            if collective == "all_to_all":
                return [(grid_choice({}, grid), "all_to_all", shape, {"grid": grid} if grid else None) for grid in grids]
            for grid in grids:
                found.append((grid_choice({"schedule": "pieces"}, grid), "reduce_scatter", shape,
                              {"scatter_schedule": "pieces", **({"grid": grid} if grid else {})}))
            if large and session.link_available and self.scatter_through == "session":
                for piece in pieces:
                    found.append(({"schedule": "chain", "piece": piece}, "reduce_scatter", shape,
                                  {"scatter_schedule": "chain", "link_chunk": piece}))
                    if ring:
                        for stagger in staggers:
                            found.append(({"schedule": "ring", "piece": piece, "stagger": stagger}, "reduce_scatter",
                                          shape, {"scatter_schedule": "ring", "link_chunk": piece,
                                                  "stagger": stagger}))
        return found

    def _group_p50(self, record: dict) -> float:
        """The largest median of the group's ranks for one case (the same value on every rank)."""
        values: list = [None] * self.world
        self.dist.all_gather_object(values, record.get("p50_us"), group=self.process_group)
        known = [float(value) for value in values if value is not None]
        return max(known) if known else float("inf")

    def _run_tune(self) -> None:
        """Every candidate of every planned collective, size and mode. From ``tune_prune_from`` bytes on, a
        candidate slower than the fastest by ``tune_prune`` at two sizes in a row, its ratio not falling
        between them, runs no larger size: a family whose fixed costs amortize keeps running. Each rank
        decides from the group's largest medians, so every rank runs the same cases. The families this
        session can run at the largest size are recorded (``tune_families``)."""
        options = self.options
        prune = float(options.get("tune_prune", 1.5))
        prune_from = int(options.get("tune_prune_from", 4 << 20))
        sizes = tuple(options.get("tune_sizes") or ())
        families = {}
        for collective in options.get("tune_collectives") or ():
            found = {tuning_family(choice) for choice, *_ in self.tune_candidates(collective, sizes[-1])} if sizes else set()
            if self.nccl is not None and collective in ("all_reduce", "all_gather"):
                found.add("nccl")
            families[collective] = sorted(found)
        self.result["tune_families"] = families
        case = 2000
        for collective in options.get("tune_collectives") or ():
            for mode in options.get("tune_modes") or ():
                ratios: dict[str, list[float]] = {}
                dropped: set[str] = set()
                for size in options.get("tune_sizes") or ():
                    timings: dict[str, float] = {}
                    candidates = [(choice, name, shape, variant, False)
                                  for choice, name, shape, variant in self.tune_candidates(collective, size)]
                    if self.nccl is not None and collective in ("all_reduce", "all_gather"):
                        candidates.append(({"backend": "nccl"}, None, (size // 2,), None, True))
                    for choice, name, shape, variant, is_nccl in candidates:
                        label = json.dumps(choice, sort_keys=True)
                        if label in dropped:
                            continue
                        before = counters.snapshot(self.devices)
                        large = size > self.session.max_size
                        if is_nccl:
                            record = self._run_nccl_case(collective, mode, shape, case, large)
                        else:
                            record = self.run_case(name, mode, shape, 0, case, large=large, variant=variant)
                        after = counters.snapshot(self.devices)
                        record["counters"] = counters.key_deltas(counters.delta(before, after))
                        record["tune"] = {"collective": collective, "choice": choice}
                        self._record(record, None)
                        timings[label] = self._group_p50(record)
                        case += 1
                    if not timings:
                        continue
                    if size < prune_from:
                        continue
                    best = min(timings.values())
                    for label, value in timings.items():
                        history = ratios.setdefault(label, [])
                        history.append(value / best if best > 0 else 1.0)
                        if (len(history) >= 2 and history[-1] > prune and history[-2] > prune
                                and history[-1] >= history[-2]):
                            dropped.add(label)

    def _run_cases(self, cap: int | None, first: bool) -> None:
        """Every case family once at the current grid cap (``cap`` None: the session's own)."""
        small = () if self.options.get("large_only") or not first else ("all_reduce", "all_gather")
        for collective in small:
            for mode in ("eager", "graph"):
                self.dist.barrier()
                for case, (shape, dim) in enumerate(self.cases(collective)):
                    before = counters.snapshot(self.devices)
                    record = self.run_case(collective, mode, shape, dim, case)
                    after = counters.snapshot(self.devices)
                    record["counters"] = counters.key_deltas(counters.delta(before, after))
                    self._record(record, None)
                    self._nccl_after(collective, mode, shape, dim, case, False)
        for mode in ("eager", "graph"):
            self.dist.barrier()
            for case, (collective, shape, dim, variant) in enumerate(self.large_cases()):
                before = counters.snapshot(self.devices)
                record = self.run_case(collective, mode, shape, dim, 500 + case, large=True, variant=variant)
                after = counters.snapshot(self.devices)
                record["counters"] = counters.key_deltas(counters.delta(before, after))
                self._record(record, cap)
                self._nccl_after(collective, mode, shape, dim, 500 + case, True)
        dcp = self.dcp_cases()
        for mode in ("eager", "graph") if dcp else ():
            self.dist.barrier()
            for case, (collective, shape, dim, rows, decode) in enumerate(dcp):
                if mode == "graph" and not decode:
                    continue
                before = counters.snapshot(self.devices)
                record = self.run_case(collective, mode, shape, dim, 700 + case, large=not decode)
                after = counters.snapshot(self.devices)
                record["counters"] = counters.key_deltas(counters.delta(before, after))
                record["dcp_rows"] = rows
                self._record(record, cap)
        swing = self.swing_cases()
        for mode in ("eager", "graph") if swing else ():
            self.dist.barrier()
            for case, (collective, shape, dim) in enumerate(swing):
                before = counters.snapshot(self.devices)
                record = self.run_case(collective, mode, shape, dim, 900 + case)
                after = counters.snapshot(self.devices)
                record["counters"] = counters.key_deltas(counters.delta(before, after))
                self._record(record, cap)
        latency = [entry for entry in self.latency_cases() if first or entry[0] != "all_reduce_oneshot"]
        for mode in ("eager", "graph") if latency else ():
            self.dist.barrier()
            for case, (collective, shape, dim, order) in enumerate(latency):
                before = counters.snapshot(self.devices)
                record = self.run_case(collective, mode, shape, dim, 1100 + case, variant={"post_order": order})
                after = counters.snapshot(self.devices)
                record["counters"] = counters.key_deltas(counters.delta(before, after))
                record["post_order"] = order
                peers = getattr(self.session, "post_order_peers", None) or (
                    self._built_order_peers if order == self.latency_orders[0] else ())
                record["post_order_peers"] = [int(peer) for peer in peers]
                self._record(record, None if collective == "all_reduce_oneshot" else cap)


# The collectives whose eager rows the call profile and the adapter path cover.
PROFILED = ("all_reduce", "all_gather", "all_reduce_large", "all_gather_large")


def forced(variant: dict | None) -> bool:
    """Whether a case variant fixes a setting a tuning table would otherwise choose: a schedule other than
    ``auto``, a piece, chunk, grid or stagger. A posting order is the session's own setting."""
    if not variant:
        return False
    return any(value is not None and key != "post_order" and not (key.endswith("schedule") and value == "auto")
               for key, value in variant.items())


def tuning_family(choice: dict) -> str:
    """The family of a tune candidate: ``nccl``, its schedule (``pieces``, ``chain``, ``ring``) or its
    algorithm (``oneshot``, ``twoshot``, ``swing``); ``grid`` for a candidate of a grid alone (all-to-all)."""
    if choice.get("backend") == "nccl":
        return "nccl"
    return choice.get("schedule") or choice.get("algorithm") or "grid"


class TransportHarness:
    """Transport-only run: the native layer on real NICs, the kernels' part played on the CPU.

    No CUDA kernel is compiled or launched. The arena is ordinary host memory
    that the devices register; Python stages each payload, writes the op word
    and the doorbell, waits for every peer's lane flags and checks every
    received byte. Per-op times run from the doorbell to the last flag seen
    and include the polling loop's own cost. This checks route maps, relays,
    lane checks and posting on the ring before the GPU path runs.
    """

    def __init__(self, plan: dict, global_rank: int, result: dict,
                 placement: cpus.Placement | None = None) -> None:
        import ctypes

        import torch.distributed as dist

        from .. import protocol, roce_gid, routes
        from ..oneshot import _proxy

        self.ctypes, self.dist, self.protocol = ctypes, dist, protocol
        self.plan, self.options, self.result = plan, plan["options"], result
        me = plan["ranks"][global_rank]
        result["environment"] = {"hostname": socket.gethostname(), "mode": "transport-only"}
        dist.init_process_group("gloo", init_method=plan["rendezvous"], rank=global_rank,
                                world_size=plan["world"], timeout=datetime.timedelta(seconds=600))
        handles = [dist.new_group(ranks=list(group["global_ranks"]), backend="gloo") for group in plan["groups"]]
        self.group = plan["groups"][me["group"]]
        self.process_group = handles[me["group"]]
        self.rank = me["group_rank"]
        self.world = len(self.group["global_ranks"])
        route_map = routes.parse_peer_routes(me["peer_routes"])
        layout = routes.Layout.parse(self.group["layout"])
        self.lanes = routes.validate_route_map(self.rank, self.world, route_map, layout=layout)
        names = list(dict.fromkeys(d for _, devices in sorted(route_map.items()) for d in devices))
        lane_devices = [() if peer == self.rank else tuple(names.index(d) for d in route_map[peer])
                        for peer in range(self.world)]
        gid = plan.get("gid_index")
        gids = [int(gid) if gid is not None else roce_gid.resolve_device_gid_index(name) for name in names]
        self.slot_bytes = protocol.slot_bytes_for(plan["capacity"], plan["gather_capacity"])
        self.arena = protocol.ArenaLayout(self.world, self.slot_bytes)
        self._buffer = ctypes.create_string_buffer(self.arena.total_bytes + 4096)
        self.base = ctypes.addressof(self._buffer) + (-ctypes.addressof(self._buffer)) % 4096
        self.native = _proxy.load()
        self.proxy = _proxy.Proxy(world_size=self.world, rank=self.rank, hca_names=names,
                                  peer_lane_devices=lane_devices, lane_count=self.lanes, gid_indices=gids,
                                  region_ptr=self.base, region_bytes=self.arena.total_bytes,
                                  slot_bytes=self.slot_bytes)
        blobs: list = [None] * self.world
        dist.all_gather_object(blobs, self.proxy.local_blob(), group=self.process_group)
        self.proxy.connect(blobs)
        dist.barrier(group=self.process_group)
        self.proxy.lane_check(self.options["lane_check_ms"])
        dist.barrier(group=self.process_group)
        window = self.options.get("forward_window")
        window = routes.DEFAULT_FORWARD_WINDOW if window is None else int(window)
        maps = [routes.parse_peer_routes(text) for text in self.group["route_texts"]]
        table = routes.forward_windows(layout, maps, self.rank, max_window=window)
        if any(any(row) for row in table):
            self.proxy.set_forward(table, routes.DEFAULT_FORWARD_CHUNK)
        progress = placement.progress_cpu_list if placement is not None else None
        if progress is not None:
            os.environ["SIRCL_PROGRESS_CPU"] = progress
        self.proxy.start()
        result["session"] = {"devices": names, "gid_indices": gids, "lane_count": self.lanes,
                             "peer_routes": me["peer_routes"], "slot_bytes": self.slot_bytes,
                             "forward_windows": {str(peer): row for peer, row in enumerate(table) if any(row)}}
        self.devices = [d for d in CANONICAL_DEVICES if (counters.SYSFS / d).exists()]
        self.seq = 0
        self.mismatches = 0

    def _word(self, offset: int):
        return self.ctypes.c_uint32.from_address(self.base + offset)

    @staticmethod
    def _payload(rank: int, seq: int, nbytes: int) -> bytes:
        block = (rank * 2654435761 ^ seq * 40503).to_bytes(8, "little") * 2
        return (block * (nbytes // 16 + 1))[:nbytes]

    def op(self, nbytes: int, check: bool) -> tuple[float, bool]:
        protocol, ctypes = self.protocol, self.ctypes
        self.seq += 1
        seq, slot = self.seq, self.seq & 1
        payload = self._payload(self.rank, seq, nbytes)
        ctypes.memmove(self.base + self.arena.send_off + slot * self.slot_bytes, payload, nbytes)
        ctrl = self.arena.ctrl_off
        self._word(ctrl + 4 * (protocol.Ctrl.OP_WORD + slot)).value = nbytes
        self._word(ctrl + 4 * protocol.Ctrl.NBYTES).value = nbytes
        started = time.perf_counter_ns()
        self.native.roce_store_release_u32(self.base + ctrl + 4 * protocol.Ctrl.DOORBELL, seq)
        deadline = time.monotonic() + 20
        load = self.native.roce_load_acquire_u32
        for source in range(self.world):
            if source == self.rank:
                continue
            for lane in range(self.lanes):
                line = protocol.flag_index(0, source, slot, lane, self.world, self.lanes)
                address = self.base + self.arena.flag_off + line * protocol.FLAG_STRIDE
                while load(address) != seq:
                    if self.proxy.failed():
                        raise RuntimeError(f"progress thread failed: {self.proxy.error()}")
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"rank {self.rank} waited for rank {source} lane {lane} at sequence {seq}")
        elapsed = (time.perf_counter_ns() - started) / 1000.0
        good = True
        if check:
            for source in range(self.world):
                if source == self.rank:
                    continue
                start = self.base + self.arena.recv_off + (source * protocol.SLOTS + slot) * self.slot_bytes
                good &= ctypes.string_at(start, nbytes) == self._payload(source, seq, nbytes)
        return elapsed, good

    def sizes(self) -> list[int]:
        sizes = set() if self.options.get("large_only") else (
            set(self.options["allreduce_sizes"]) | set(self.options["allgather_sizes"]))
        if self.options.get("large"):
            sizes |= {size for size in self.options.get("large_allreduce_sizes", ()) if size <= self.slot_bytes}
        return sorted(sizes)

    def run(self) -> None:
        totals_before = counters.snapshot(self.devices)
        self.dist.barrier()
        for nbytes in self.sizes():
            before = counters.snapshot(self.devices)
            record = {"collective": "transport", "mode": "cpu", "dtype": "bytes", "shape": [nbytes], "dim": 0,
                      "bytes": nbytes, "checked": 0, "mismatched_calls": 0}
            for _ in range(self.options["correctness_iterations"]):
                _, good = self.op(nbytes, check=True)
                record["checked"] += 1
                record["mismatched_calls"] += 0 if good else 1
            self.dist.barrier(group=self.process_group)
            for _ in range(self.options.get("post_barrier_warmup", 0)):
                self.op(nbytes, check=False)
            migrations = self.proxy.stats()["proxy_cpu_migrations"]
            cpu_before = cpus.current_cpu()
            times = [round(self.op(nbytes, check=False)[0], 2) for _ in range(self.options["eager_iterations"])]
            stats = self.proxy.stats()
            record["placement"] = {"main_cpu_start": cpu_before, "main_cpu_end": cpus.current_cpu(),
                                   "proxy_cpu": stats["proxy_cpu"],
                                   "proxy_migrations": stats["proxy_cpu_migrations"] - migrations}
            record["correct"] = record["mismatched_calls"] == 0
            record["times_us"] = times
            record.update(summarize_times(times))
            record.update(bandwidths("transport", nbytes, self.world, record.get("p50_us")))
            after = counters.snapshot(self.devices)
            record["counters"] = counters.key_deltas(counters.delta(before, after))
            self.result["runs"].append(record)
            self.mismatches += record["mismatched_calls"]
        totals_after = counters.snapshot(self.devices)
        self.result["counters_total"] = counters.key_deltas(counters.delta(totals_before, totals_after))
        self.result["counter_sources"] = {device: totals_after[device]["source"] for device in totals_after}
        self.result["proxy"] = self.proxy.stats()
        self.dist.barrier()
        self.proxy.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One rank of the SIRCL ring harness.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--global-rank", required=True, type=int)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    me = plan["ranks"][args.global_rank]
    out = args.plan.parent / f"rank-{args.global_rank}.json"
    result = {
        "schema": "sircl-ring-rank-result/v1", "configuration": plan["configuration"], "run_id": plan["run_id"],
        "global_rank": args.global_rank, "group": me["group"], "group_rank": me["group_rank"],
        "host": me["host"], "position": me["position"], "started": _now(), "runs": [], "error": None,
    }

    def expire() -> None:
        result["error"] = f"watchdog: the rank did not finish within {plan['options']['worker_timeout_s']} s"
        result["finished"] = _now()
        _write(out, result)
        os._exit(3)

    watchdog = threading.Timer(plan["options"]["worker_timeout_s"], expire)
    watchdog.daemon = True
    watchdog.start()
    code = 0
    try:
        # Pinned before torch loads, so every thread the process creates inherits it.
        placement = cpus.plan(plan["options"].get("cpu_policy", "performance"))
        cpus.apply(placement)
        result["placement"] = placement.to_json()
        kind = TransportHarness if plan["options"].get("transport_only") else Harness
        harness = kind(plan, args.global_rank, result, placement)
        harness.run()
        if harness.mismatches:
            code = 2
    except BaseException as error:  # noqa: BLE001 - every failure is recorded, then the rank ends
        result["error"] = f"{type(error).__name__}: {error}"
        result["traceback"] = traceback.format_exc()
        code = 1
    watchdog.cancel()
    result["finished"] = _now()
    result["exit_code"] = code
    _write(out, result)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":  # pragma: no cover
    main()
