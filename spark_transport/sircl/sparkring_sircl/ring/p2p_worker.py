"""One rank of the point-to-point ring cases (:mod:`sparkring_sircl.ring.p2p`), inside the serving image.

``python -m sparkring_sircl.ring.p2p_worker --plan <p2p-plan.json> --global-rank N``

The worker joins the control exchange (gloo over the wired LAN), builds the
group's point-to-point channels on GPU 0 (:class:`sparkring_sircl.p2p.PointToPoint`
with the plan's route map and layout; windows as every rank derives them),
prepares the kernels and runs every case of the plan, each size after a
barrier of every rank (see the module docstring of :mod:`.p2p`). Message bytes
come from the plan's seed, the case, the direction, the size and the
iteration, so every receiver regenerates what it must have received and
compares bit for bit. It writes ``p2p-rank-<N>.json`` next to the plan: setup,
the channels' counters, the RDMA error counters before and after, and one
record per case and size.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import socket
import sys
import threading
import time
import traceback
import zlib
from pathlib import Path

from .. import cpus
from . import counters
from .p2p import RESULT_SCHEMA, burst_messages, latency_rounds
from .worker import CANONICAL_DEVICES, _now, _write, summarize_times

ACK_BYTES = 16


class P2PHarness:
    """The rank's run: channel construction, cases and records."""

    def __init__(self, plan: dict, global_rank: int, result: dict, placement: cpus.Placement | None) -> None:
        import torch
        import torch.distributed as dist

        from .. import routes

        self.torch, self.dist = torch, dist
        self.plan = plan
        self.p2p = plan["p2p"]
        self.options = self.p2p["options"]
        self.rank = global_rank
        self.world = plan["world"]
        self.me = plan["ranks"][global_rank]
        self.result = result
        self.mismatches = 0
        torch.cuda.set_device(0)
        self.device = torch.device("cuda", 0)
        result["environment"] = {"hostname": socket.gethostname(), "torch": torch.__version__,
                                 "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        os.environ["SIRCL_STARTUP_WAIT_S"] = str(self.options["startup_wait_s"])
        os.environ["SIRCL_SERVING_WAIT_S"] = str(self.options["serving_wait_s"])
        os.environ["SIRCL_P2P_WINDOW_BYTES"] = str(self.options["window_bytes"])
        os.environ["SIRCL_P2P_CHUNK_BYTES"] = str(self.options["chunk_bytes"])
        for name, value in self.options.get("session_env", ()):
            os.environ[str(name)] = str(value)
        if placement is not None and placement.progress_cpu_list:
            os.environ["SIRCL_P2P_PROGRESS_CPU"] = placement.progress_cpu_list
        dist.init_process_group("gloo", init_method=plan["rendezvous"], rank=global_rank, world_size=self.world,
                                timeout=datetime.timedelta(seconds=600))
        from ..p2p import PointToPoint

        started = time.perf_counter()
        (group,) = plan["groups"]
        self.channels = PointToPoint(exchange_group=dist.group.WORLD, device=self.device,
                                     peer_routes=routes.parse_peer_routes(self.me["peer_routes"]),
                                     layout=group["layout"], gid_index=plan.get("gid_index"),
                                     lane_check_ms=self.options["lane_check_ms"])
        self.channels.prepare()
        result["setup_seconds"] = round(time.perf_counter() - started, 3)
        stats = self.channels.stats()
        result["channels"] = {key: stats.get(key) for key in (
            "rank", "world_size", "lane_count", "slots", "slot_bytes", "chunk_bytes", "blocks", "windows",
            "arena_bytes", "lane_check", "layout")}
        result["channels"]["unavailable"] = {str(peer): self.channels.channel_problem(peer)
                                             for peer in range(self.world)
                                             if peer != self.rank and not self.channels.has_channel(peer)}

    # -- data ----------------------------------------------------------------------------------

    def _bytes(self, case: str, source: int, destination: int, size: int, iteration: int):
        torch = self.torch
        generator = torch.Generator(device=self.device)
        # CRC-32, not hash(): string hashes differ between processes, and every rank must derive the same bytes.
        key = zlib.crc32(f"{case}|{source}|{destination}|{size}|{iteration}".encode())
        generator.manual_seed((int(self.options["seed"]) * 1_000_003 + key) & 0x7FFFFFFFFFFFFFFF)
        return torch.randint(0, 256, (size,), dtype=torch.uint8, device=self.device, generator=generator)

    def _barrier(self) -> None:
        self.dist.barrier()

    # -- cases ---------------------------------------------------------------------------------

    def run(self) -> None:
        records = self.result.setdefault("records", [])
        before = counters.snapshot(CANONICAL_DEVICES)
        self.channels.enter_serving()
        for case in self.p2p["cases"]:
            for size in self.options["sizes"]:
                self._barrier()
                if case["kind"] == "pair":
                    record = self._pair(case["ranks"], int(size))
                else:
                    record = self._shift(int(case["distance"]), int(size))
                if record is not None:
                    records.append(record)
                self.channels.check_health()
        self._barrier()
        after = counters.snapshot(CANONICAL_DEVICES)
        self.result["counters"] = counters.key_deltas(counters.delta(before, after))
        self.result["channels"]["stats"] = {key: value for key, value in self.channels.stats().items()
                                            if key not in ("layout",)}
        self.channels.close()

    def _pair(self, ranks: list[int], size: int) -> dict | None:
        torch = self.torch
        a, b = int(ranks[0]), int(ranks[1])
        name = f"pair {a}-{b}"
        if self.rank not in (a, b):
            return None
        peer = b if self.rank == a else a
        record = {"case": name, "bytes": size, "hops": self.p2p["hops"][a][b], "checked": 0, "mismatches": 0}
        problem = self.channels.channel_problem(peer)
        if problem:
            record["skipped"] = problem
            return record
        out = torch.empty(size, dtype=torch.uint8, device=self.device)
        for iteration in range(int(self.options["checked_iterations"])):
            for source, destination in ((a, b), (b, a)):
                if self.rank == source:
                    self.channels.send(self._bytes(name, source, destination, size, iteration), destination)
                else:
                    self.channels.recv(out, source)
                    torch.cuda.current_stream().synchronize()
                    record["checked"] += 1
                    if not torch.equal(out, self._bytes(name, source, destination, size, iteration)):
                        record["mismatches"] += 1
        self.mismatches += record["mismatches"]
        message = self._bytes(name, self.rank, peer, size, 0)
        rounds = latency_rounds(size, int(self.options["latency_iterations"]))
        warmup = int(self.options["warmup_iterations"])
        events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(rounds)]
        for index in range(warmup + rounds):
            timed = index >= warmup
            if self.rank == a:
                if timed:
                    events[index - warmup][0].record()
                self.channels.send(message, peer)
                self.channels.recv(out, peer)
                if timed:
                    events[index - warmup][1].record()
            else:
                self.channels.recv(out, peer)
                self.channels.send(message, peer)
        torch.cuda.current_stream().synchronize()
        if self.rank == a:
            record["latency"] = summarize_times([start.elapsed_time(end) * 1000.0 / 2.0 for start, end in events])
        count = burst_messages(size, int(self.options["bandwidth_bytes"]))
        ack = torch.zeros(ACK_BYTES, dtype=torch.uint8, device=self.device)
        times = []
        for repeat in range(int(self.options["bandwidth_repeats"]) + 1):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            if self.rank == a:
                start.record()
                works = [self.channels.isend(message, peer) for _ in range(count)]
                self.channels.recv(ack, peer)
                end.record()
                for work in works:
                    work.wait()
            else:
                works = [self.channels.irecv(out, peer) for _ in range(count)]
                for work in works:
                    work.wait()
                self.channels.send(ack, peer)
            torch.cuda.current_stream().synchronize()
            if self.rank == a and repeat:
                times.append(start.elapsed_time(end) * 1000.0)
        if self.rank == a and times:
            best = summarize_times(times)["p50_us"]
            record["bandwidth_gbps"] = round(count * size / (best * 1e-6) / 1e9, 3)
            record["burst_messages"] = count
        return record

    def _shift(self, distance: int, size: int) -> dict:
        torch = self.torch
        name = f"shift {distance}"
        destination = (self.rank + distance) % self.world
        source = (self.rank - distance) % self.world
        record = {"case": name, "bytes": size, "hops": self.p2p["hops"][self.rank][destination], "checked": 0,
                  "mismatches": 0}
        problems = [p for p in (self.channels.channel_problem(destination), self.channels.channel_problem(source)) if p]
        flags = self._all_flags(bool(problems))
        if any(flags):
            record["skipped"] = problems[0] if problems else "another rank of the shift has no channel"
            return record
        out = torch.empty(size, dtype=torch.uint8, device=self.device)
        for iteration in range(int(self.options["checked_iterations"])):
            works = self.channels.batch_isend_irecv([
                ("recv", out, source), ("send", self._bytes(name, self.rank, destination, size, iteration),
                                        destination)])
            for work in works:
                work.wait()
            torch.cuda.current_stream().synchronize()
            record["checked"] += 1
            if not torch.equal(out, self._bytes(name, source, self.rank, size, iteration)):
                record["mismatches"] += 1
        self.mismatches += record["mismatches"]
        message = self._bytes(name, self.rank, destination, size, 0)
        count = burst_messages(size, int(self.options["bandwidth_bytes"]))
        times = []
        for repeat in range(int(self.options["bandwidth_repeats"]) + 1):
            self._barrier()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            ops = []
            for _ in range(count):
                ops += [("recv", out, source), ("send", message, destination)]
            for work in self.channels.batch_isend_irecv(ops):
                work.wait()
            end.record()
            torch.cuda.current_stream().synchronize()
            if repeat:
                times.append(start.elapsed_time(end) * 1000.0)
        best = summarize_times(times)["p50_us"]
        record["bandwidth_gbps"] = round(count * size / (best * 1e-6) / 1e9, 3)
        record["burst_messages"] = count
        return record

    def _all_flags(self, flag: bool) -> list[bool]:
        gathered: list = [None] * self.world
        self.dist.all_gather_object(gathered, flag)
        return [bool(v) for v in gathered]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="One rank of the SIRCL point-to-point ring cases.")
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--global-rank", required=True, type=int)
    args = parser.parse_args(argv)
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    me = plan["ranks"][args.global_rank]
    out = args.plan.parent / f"p2p-rank-{args.global_rank}.json"
    result = {"schema": RESULT_SCHEMA, "run_id": plan["run_id"], "global_rank": args.global_rank,
              "host": me["host"], "position": me["position"], "started": _now(), "records": [], "error": None}
    timeout = plan["p2p"]["options"]["worker_timeout_s"]

    def expire() -> None:
        result["error"] = f"watchdog: the rank did not finish within {timeout} s"
        result["finished"] = _now()
        _write(out, result)
        os._exit(3)

    watchdog = threading.Timer(timeout, expire)
    watchdog.daemon = True
    watchdog.start()
    code = 0
    try:
        placement = cpus.plan(plan["p2p"]["options"].get("cpu_policy", "performance"))
        cpus.apply(placement)
        result["placement"] = placement.to_json()
        harness = P2PHarness(plan, args.global_rank, result, placement)
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
