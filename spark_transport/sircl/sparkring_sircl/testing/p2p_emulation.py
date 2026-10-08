"""GPU emulation of SIRCL's point-to-point channels (test support).

Every rank of a group runs as a thread of one process on one CUDA device, as
in :mod:`.gpu_emulation`: the native layers are the test build that holds both
the collective library and the point-to-point library over one in-memory
verbs stand-in (:func:`.p2p_build.build_shared_library`), so a rank's
collective session and its point-to-point channels share one emulated fabric.
The point-to-point sessions (:class:`sparkring_sircl.p2p.PointToPoint`), their
kernels and streams run unchanged.

Checks, each against the bytes the sender sent:

- every ordered pair at once, in one batch per rank, for messages of 16 B to
  4 MiB, and 64 MiB on chosen pairs (neighbors and the farthest ranks, whose
  lanes cross relays);
- dtypes and shapes that take the scratch path (odd byte counts,
  non-contiguous and unaligned tensors, bool, int32, int64, BF16);
- first-in first-out order of several messages on one channel;
- both ranks of every pair sending first, then receiving, with messages
  larger than a channel's slots;
- point-to-point transfers between collective ops of the same group, on the
  ranks' streams;
- a receive whose size differs from the message, refused on every rank;
- a receive that no rank answers, timed out under the serving wait limit;
- item counters that start 16 items before the 32-bit wrap.

Order. Every rank shares one GPU here, so the streams of all ranks share the
GPU's hardware queues (``CUDA_DEVICE_MAX_CONNECTIONS``, 32), and a receive
waiting for its peer holds back the commands queued behind it on its queue,
whichever rank issued them. Every check therefore issues every rank's sends,
waits at a barrier of the rank threads, and only then issues the receives;
and the emulation's channels have 32 slots of 1 MiB, so that no message of a
check (at most 64 MiB, two rounds of slots) makes its send kernel wait for the
receiver. On a ring every rank has its own GPU, and only its own order
matters (``p2p/session.py``, "issue order").

``python -m sparkring_sircl.testing.p2p_emulation --layout ring:8 --lanes 2``
prints one line per check. Requirements: CUDA, torch with CUDA, CUDA Python,
the CuTe DSL and a GCC-compatible compiler for the test library.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from .. import routes as routes_mod
from . import p2p_build
from .gpu_emulation import EmulatedGroup, ThreadDist, ThreadGroup, _path_latency, device_name

SERVING_LIMIT_S = 0.5
EMULATION_SETTINGS = {"SIRCL_P2P_BLOCKS": "2", "SIRCL_P2P_THREADS": "256", "SIRCL_P2P_SLOTS": "32",
                      "SIRCL_P2P_SLOT_BYTES": str(1 << 20)}
BARRIER_S = 600.0


class P2PGroup:
    """A group's collective sessions (:class:`.gpu_emulation.EmulatedGroup`) and its point-to-point channels."""

    def __init__(self, layout_text: str, lanes: int = 2, *, library: str | os.PathLike,
                 path_latency: tuple[int, int, int, int] | None = None, collectives: bool = True,
                 environment: dict[str, str] | None = None) -> None:
        import torch

        from ..p2p import session as p2p_session

        self.torch = torch
        self.p2p_session = p2p_session
        os.environ["SIRCL_P2P_NATIVE_LIBRARY"] = str(library)
        settings = dict(EMULATION_SETTINGS)
        settings.update(environment or {})
        for name, value in settings.items():
            os.environ[name] = value
        self.library = str(library)
        self.group = EmulatedGroup(layout_text, lanes, max_size=256 << 10, max_gather_bytes=64 << 10,
                                   library=library, path_latency=path_latency,
                                   environment={"SIRCL_SERVING_WAIT_S": str(SERVING_LIMIT_S)})
        self.layout = routes_mod.Layout.parse(layout_text)
        self.world = self.group.world
        self.lanes = lanes
        self.collectives = collectives
        self.barrier = threading.Barrier(self.world)
        self.channels: list[Any] = []
        self.build_channels()

    def build_channels(self, start_item: int = 0) -> None:
        derived = routes_mod.derive_routes(self.layout, self.lanes)
        exchange = ThreadGroup(self.world)
        sessions: list[Any] = [None] * self.world
        layout_text = self.group.layout_text

        def construct(rank: int, _session) -> None:
            exchange.bind(rank)
            routes = {peer: tuple(device_name(rank, device) for device in devices)
                      for peer, devices in derived.route_map(rank).items()}
            sessions[rank] = self.p2p_session.PointToPoint(
                exchange_group=exchange, device=self.torch.device("cuda", 0), peer_routes=routes, layout=layout_text,
                gid_index=3, lane_check_ms=10000, library=self.library, start_item=start_item)

        saved = self.p2p_session.dist
        self.p2p_session.dist = ThreadDist
        try:
            self.group.each(construct)
        finally:
            self.p2p_session.dist = saved
        self.channels = sessions

    def prepare(self, dtypes: Sequence[Any]) -> None:
        """Compile and load every kernel, one rank after another (compilation is not a collective)."""
        torch = self.torch
        for rank in range(self.world):
            with torch.cuda.stream(self.group.streams[rank]):
                self.channels[rank].prepare()
                if self.collectives:
                    self.group.sessions[rank].prepare(tuple(dtypes))
        if self.collectives:
            self.group.load_modules(dtypes)

    def each(self, operation: Callable[[int, Any, Any], Any]) -> list[Any]:
        """``operation(rank, p2p channels, collective session)`` on every rank at once, on the rank's stream.

        A rank that raises breaks the ranks' barrier, so ranks waiting there raise at once; the barrier is
        ready again for the next operation."""
        def body(rank: int, session: Any) -> Any:
            try:
                return operation(rank, self.channels[rank], session)
            except BaseException:
                self.barrier.abort()
                raise

        try:
            return self.group.each(body)
        finally:
            if self.barrier.broken:
                self.barrier.reset()

    def close(self) -> None:
        for channels in self.channels:
            try:
                if channels is not None:
                    channels.close()
            except Exception:  # noqa: BLE001 - teardown
                pass
        self.group.close()


# -- payloads ---------------------------------------------------------------------------------


def _payload(torch, src: int, dst: int, nbytes: int, salt: int):
    generator = torch.Generator().manual_seed((src * 1009 + dst * 9176 + salt * 31 + nbytes) % (1 << 62))
    return torch.randint(0, 256, (nbytes,), generator=generator, dtype=torch.uint8)


def _same(torch, got, want) -> bool:
    return got.shape == want.shape and bool(torch.equal(got.cpu(), want.cpu()))


def _issue(group: P2PGroup, channels, sends: Sequence[tuple[Any, int]], recvs: Sequence[tuple[Any, int]], *,
           batch: bool = False) -> list[Any]:
    """Issue this rank's sends, wait until every rank issued its sends, then issue the receives (module
    docstring, "Order"); with ``batch``, each half as one ``batch_isend_irecv``."""
    if batch:
        works = list(channels.batch_isend_irecv([("send", t, peer) for t, peer in sends])) if sends else []
    else:
        works = [channels.isend(t, peer) for t, peer in sends]
    group.barrier.wait(BARRIER_S)
    if batch:
        works += list(channels.batch_isend_irecv([("recv", t, peer) for t, peer in recvs])) if recvs else []
    else:
        works += [channels.irecv(t, peer) for t, peer in recvs]
    return works


def _run(group: P2PGroup, name: str, operation: Callable[[int, Any, Any], Any],
         verify: Callable[[list[Any]], str]) -> tuple[str, bool, str]:
    started = time.perf_counter()
    try:
        results = group.each(operation)
        problem = verify(results)
    except Exception as error:  # noqa: BLE001 - reported as a failed check
        traceback.print_exc()
        return name, False, f"{type(error).__name__}: {error}"
    detail = problem or f"{time.perf_counter() - started:.2f} s"
    return name, not problem, detail


def _pairs_check(group: P2PGroup, nbytes: int, pairs: Sequence[tuple[int, int]] | None = None,
                 salt: int = 1) -> tuple[str, bool, str]:
    """Every ordered pair (or ``pairs``, both directions) exchanges one message of ``nbytes`` in one batch per rank."""
    torch = group.torch
    world = group.world
    chosen = ({(a, b) for a in range(world) for b in range(world) if a != b} if pairs is None else
              {p for a, b in pairs for p in ((a, b), (b, a))})
    label = "every pair" if pairs is None else "pairs " + ",".join(f"{a}-{b}" for a, b in pairs)

    def operation(rank: int, channels, _session):
        outputs = {peer: torch.empty(nbytes, dtype=torch.uint8, device=channels.device)
                   for peer in range(world) if (peer, rank) in chosen}
        sends = [(_payload(torch, rank, peer, nbytes, salt).to(channels.device), peer)
                 for peer in range(world) if (rank, peer) in chosen]
        for work in _issue(group, channels, sends, [(out, peer) for peer, out in outputs.items()], batch=True):
            work.wait()
        return {peer: out.cpu() for peer, out in outputs.items()}

    def verify(results) -> str:
        wrong = [(src, dst) for dst, received in enumerate(results) for src, got in received.items()
                 if not _same(torch, got, _payload(torch, src, dst, nbytes, salt))]
        return f"pairs {wrong} differ" if wrong else ""

    return _run(group, f"pairs {nbytes} B ({label})", operation, verify)


def _shapes_check(group: P2PGroup) -> tuple[str, bool, str]:
    """Odd sizes, non-contiguous and unaligned tensors and several dtypes, between ring neighbors."""
    torch = group.torch
    world = group.world

    def tensors(src: int, dst: int):
        base = _payload(torch, src, dst, 1 << 16, 7)
        return [
            base[:13].clone(),                                             # 13 bytes
            base[:20].view(torch.int32).view(5, 1).clone(),               # 20 bytes
            base[:2 * 7 * 333].view(torch.bfloat16).view(7, 333).t(),     # non-contiguous
            base[4:4 + 4 * 97].view(torch.float32),                        # 4-byte offset: unaligned
            (base[:9] & 1).bool(),                                         # bool
            base[:48].view(torch.int64).view(3, 2),                        # whole packs
            base[:0],                                                      # empty
        ]

    def operation(rank: int, channels, _session):
        nxt, prev = (rank + 1) % world, (rank - 1) % world
        sends = [t.to(channels.device) for t in tensors(rank, nxt)]
        if sends[2].is_contiguous():
            raise AssertionError("the transposed tensor is contiguous")
        received = []
        for want in tensors(prev, rank):
            out = torch.empty(want.shape, dtype=want.dtype, device=channels.device)
            if want.dim() == 2 and want.dtype == torch.bfloat16:
                out = torch.empty(want.shape[::-1], dtype=want.dtype, device=channels.device).t()
            received.append(out)
        for work in _issue(group, channels, [(t, nxt) for t in sends], [(out, prev) for out in received]):
            work.wait()
        return [out.cpu() for out in received]

    def verify(results) -> str:
        wrong = []
        for rank, got in enumerate(results):
            prev = (rank - 1) % world
            for index, (out, want) in enumerate(zip(got, tensors(prev, rank))):
                same = (out.shape == want.shape and out.dtype == want.dtype and
                        torch.equal(out.contiguous().view(torch.uint8), want.contiguous().view(torch.uint8)))
                if not same:
                    wrong.append((rank, index))
        return f"(rank, tensor) {wrong} differ" if wrong else ""

    return _run(group, "dtypes and shapes (odd sizes, non-contiguous, unaligned, bool, empty)", operation, verify)


def _fifo_check(group: P2PGroup) -> tuple[str, bool, str]:
    """Five messages of different sizes on every channel toward the next rank, received in order."""
    torch = group.torch
    world = group.world
    sizes = (16, 700000, 32, 3 << 20, 4096)

    def operation(rank: int, channels, _session):
        nxt, prev = (rank + 1) % world, (rank - 1) % world
        outs = [torch.empty(n, dtype=torch.uint8, device=channels.device) for n in sizes]
        sends = [(_payload(torch, rank, nxt, n, 11 + i).to(channels.device), nxt) for i, n in enumerate(sizes)]
        for work in _issue(group, channels, sends, [(out, prev) for out in outs]):
            work.wait()
        return [out.cpu() for out in outs]

    def verify(results) -> str:
        wrong = [(rank, i) for rank, outs in enumerate(results) for i, (out, n) in enumerate(zip(outs, sizes))
                 if not _same(torch, out, _payload(torch, (rank - 1) % world, rank, n, 11 + i))]
        return f"(rank, message) {wrong} differ" if wrong else ""

    return _run(group, f"first-in first-out, messages of {list(sizes)} B", operation, verify)


def _send_first_check(group: P2PGroup, nbytes: int) -> tuple[str, bool, str]:
    """Both ranks of every neighbor pair send first and receive second, with blocking calls."""
    torch = group.torch
    world = group.world

    def operation(rank: int, channels, _session):
        # Pairs (0,1), (2,3), ...: each rank's partner; an odd last rank pairs with rank 0's partner chain.
        partner = rank ^ 1 if (rank ^ 1) < world else (rank - 1)
        out = torch.empty(nbytes, dtype=torch.uint8, device=channels.device)
        channels.send(_payload(torch, rank, partner, nbytes, 21).to(channels.device), partner)
        group.barrier.wait(BARRIER_S)
        channels.recv(out, partner)
        return partner, out.cpu()

    def verify(results) -> str:
        wrong = [rank for rank, (partner, got) in enumerate(results)
                 if not _same(torch, got, _payload(torch, partner, rank, nbytes, 21))]
        return f"ranks {wrong} differ" if wrong else ""

    if world % 2:
        return "send first, then receive", True, "skipped: an odd group has no pairing"
    return _run(group, f"send first, then receive, {nbytes} B both ways", operation, verify)


def _mixed_check(group: P2PGroup) -> tuple[str, bool, str]:
    """All-reduce, a transfer to the next rank, all-reduce again: collective ops and channels interleaved."""
    torch = group.torch
    world = group.world
    count = 4096

    def inputs(rank: int, step: int):
        generator = torch.Generator().manual_seed(rank * 101 + step)
        return torch.randn(count, generator=generator).to(torch.bfloat16)

    def total(step: int):
        acc = inputs(0, step).float()
        for rank in range(1, world):
            acc += inputs(rank, step).float()
        return acc.to(torch.bfloat16)

    def operation(rank: int, channels, session):
        nxt, prev = (rank + 1) % world, (rank - 1) % world
        first = session.all_reduce(inputs(rank, 1).to(channels.device))
        out = torch.empty(count * 2, dtype=torch.uint8, device=channels.device)
        for work in _issue(group, channels, [(first.view(torch.uint8), nxt)], [(out, prev)]):
            work.wait()
        second = session.all_reduce(inputs(rank, 2).to(channels.device) + out.view(torch.bfloat16) * 0)
        return first.cpu(), out.view(torch.bfloat16).cpu(), second.cpu()

    def verify(results) -> str:
        reference1, reference2 = total(1), total(2)
        view = torch.int16
        wrong = [rank for rank, (a, b, c) in enumerate(results)
                 if not (torch.equal(a.view(view), reference1.view(view)) and torch.equal(b.view(view), reference1.view(view))
                         and torch.equal(c.view(view), reference2.view(view)))]
        return f"ranks {wrong} differ" if wrong else ""

    if not group.collectives:
        return "collectives and channels interleaved", True, "skipped: no collective sessions"
    return _run(group, "collectives and channels interleaved on the ranks' streams", operation, verify)


def _mismatch_check(group: P2PGroup) -> tuple[str, bool, str]:
    """Rank 0 sends 64 KiB then 4 KiB to the farthest rank, which receives 4 KiB first: every rank refuses."""
    torch = group.torch
    world = group.world
    far = world // 2

    def operation(rank: int, channels, _session):
        sends = ([(torch.zeros(65536, dtype=torch.uint8, device=channels.device), far),
                  (torch.zeros(4096, dtype=torch.uint8, device=channels.device), far)] if rank == 0 else [])
        recvs = [(torch.empty(4096, dtype=torch.uint8, device=channels.device), 0)] if rank == far else []
        for work in _issue(group, channels, sends, recvs):
            work.wait()
        deadline = time.monotonic() + 30
        while not channels.poisoned and time.monotonic() < deadline:
            time.sleep(0.01)
        try:
            channels.check_health()
        except RuntimeError as error:
            return str(error)
        return None

    def verify(results) -> str:
        missing = [rank for rank, text in enumerate(results) if not text]
        if missing:
            return f"ranks {missing} kept running"
        if "different sizes" not in results[far]:
            return f"rank {far} did not name the size mismatch: {results[far]}"
        if f"on rank {far}" not in results[0]:
            return f"rank 0 did not name rank {far}: {results[0]}"
        return ""

    name = f"out-of-order sizes on 0->{far} refused on every rank"
    check = _run(group, name, operation, verify)
    if check[1]:
        check = (name, True, f"rank {far}: " + group.each(lambda r, c, s: str(_health(c)))[far][:220])
    return check


def _health(channels) -> str:
    try:
        channels.check_health()
    except RuntimeError as error:
        return str(error)
    return "healthy"


def _timeout_check(group: P2PGroup) -> tuple[str, bool, str]:
    """Rank 1 receives from rank 0, which sends nothing: the serving limit ends the wait on every rank."""
    torch = group.torch

    def operation(rank: int, channels, _session):
        channels.enter_serving()
        started = time.monotonic()
        if rank == 1:
            channels.irecv(torch.empty(4096, dtype=torch.uint8, device=channels.device), 0)
        deadline = time.monotonic() + 30
        while not channels.poisoned and time.monotonic() < deadline:
            time.sleep(0.005)
        return time.monotonic() - started, _health(channels)

    def verify(results) -> str:
        if any(text == "healthy" for _, text in results):
            return f"ranks {[rank for rank, (_, text) in enumerate(results) if text == 'healthy']} kept running"
        if "timed out waiting for item 0 from rank 0" not in results[1][1]:
            return f"rank 1: {results[1][1]}"
        waited = results[1][0]
        if not SERVING_LIMIT_S * 0.8 <= waited <= SERVING_LIMIT_S + 10:
            return f"rank 1 waited {waited:.2f} s against a limit of {SERVING_LIMIT_S} s"
        return ""

    return _run(group, f"unanswered receive ends after the {SERVING_LIMIT_S} s serving limit on every rank", operation,
                verify)


class _Checks(list):
    def __init__(self, report) -> None:
        super().__init__()
        self._report = report

    def append(self, check) -> None:
        super().append(check)
        if self._report is not None:
            self._report(check)


def run_checks(layout_text: str = "path:0-3", lanes: int = 2, *, library: str | os.PathLike | None = None,
               path_latency: tuple[int, int, int, int] | None = None, large: int = 64 << 20,
               report: Callable[[tuple[str, bool, str]], None] | None = None) -> list[tuple[str, bool, str]]:
    import torch

    if library is None:
        build = Path(os.environ.get("SIRCL_TEST_BUILD_DIR", Path.cwd() / ".build" / "sim"))
        library = p2p_build.build_shared_library(build)
    checks = _Checks(report)
    started = time.perf_counter()
    group = P2PGroup(layout_text, lanes, library=library, path_latency=path_latency)
    world = group.world
    try:
        group.prepare((torch.bfloat16,))
        stats = group.channels[0].stats()
        checks.append(("prepare", True, f"{time.perf_counter() - started:.1f} s; rank 0 channels {stats['channels']}, "
                                        f"{stats['slots']} slots of {stats['slot_bytes']} B, windows "
                                        f"{stats['windows']}"))
        for nbytes in (16, 4096, 4096 + 16, 65536, 524288, 524288 + 16, 4 << 20):
            checks.append(_pairs_check(group, nbytes))
        far = world // 2
        large_pairs = [(0, 1), (0, far)] + ([(1, far + 1)] if world >= 4 else [])
        checks.append(_pairs_check(group, large, sorted(set(large_pairs))))
        checks.append(_pairs_check(group, large // 2 + 48, [(0, far)], salt=2))
        checks.append(_shapes_check(group))
        checks.append(_fifo_check(group))
        checks.append(_send_first_check(group, 48 << 20))     # more than the channel's 32 slots of 1 MiB
        checks.append(_mixed_check(group))
        native = group.channels[0].stats()
        checks.append(("native counters", True,
                       f"rank 0: {native['items_posted']} items posted, {native['items_released']} released, "
                       f"{native['credits_sent']} credits, window waits {native['window_waits']}, largest unacknowledged "
                       f"{native['window_max_unacked_bytes']} B, {native['proven_bytes']} B proven by credits"))
        healthy = [not channels.poisoned for channels in group.channels]
        checks.append(("health", all(healthy), "" if all(healthy) else f"poisoned ranks {healthy}"))
        checks.append(_mismatch_check(group))
        for channels in group.channels:
            channels.close()
        group.build_channels(start_item=0)
        for rank in range(world):
            with torch.cuda.stream(group.group.streams[rank]):
                group.channels[rank].prepare()
        checks.append(_timeout_check(group))
        for channels in group.channels:
            channels.close()
        group.build_channels(start_item=0xFFFFFFF0)
        checks.append(_wrap_check(group))
    finally:
        group.close()
    return checks


def _wrap_check(group: P2PGroup) -> tuple[str, bool, str]:
    """Channels whose item counters start 16 items before the 32-bit wrap carry messages across it."""
    torch = group.torch
    world = group.world
    sizes = [8 << 20, (3 << 20) + 16, 65536 + 16, 6 << 20]     # 19 items of 1 MiB: across the wrap

    def operation(rank: int, channels, _session):
        nxt, prev = (rank + 1) % world, (rank - 1) % world
        outs = [torch.empty(n, dtype=torch.uint8, device=channels.device) for n in sizes]
        sends = [(_payload(torch, rank, nxt, n, 31 + i).to(channels.device), nxt) for i, n in enumerate(sizes)]
        for work in _issue(group, channels, sends, [(out, prev) for out in outs]):
            work.wait()
        return [out.cpu() for out in outs]

    def verify(results) -> str:
        wrong = [(rank, i) for rank, outs in enumerate(results) for i, (out, n) in enumerate(zip(outs, sizes))
                 if not _same(torch, out, _payload(torch, (rank - 1) % world, rank, n, 31 + i))]
        sent = group.channels[0].stats().get("per_peer", {}).get(str(1 % world), {}).get("sent")
        if wrong:
            return f"(rank, message) {wrong} differ"
        if sent is None or sent >= 0xFFFFFFF0:
            return f"rank 0's channel toward rank 1 did not pass the wrap (sent word {sent})"
        return ""

    return _run(group, "item counters across the 32-bit wrap", operation, verify)


def main(argv: list[str] | None = None) -> int:
    os.environ.setdefault("CUDA_MODULE_LOADING", "EAGER")
    # Every rank runs here on one GPU, so the channels' streams outnumber the GPU's hardware queues; a receive
    # kernel waiting for its peer holds back what follows it on its queue. The most queues CUDA offers (read when
    # the CUDA context is created) keep the fewest streams on one queue.
    os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", default="path:0-3")
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--large", type=int, default=64 << 20, help="bytes of the largest messages (default 64 MiB)")
    parser.add_argument("--path-latency", default="",
                        help="BASE_NS,RELAY_NS[,BYTES_PER_US[,ACK_DELAY_NS]] as in gpu_emulation")
    args = parser.parse_args(argv)

    def show(check: tuple[str, bool, str]) -> None:
        name, ok, detail = check
        print(f"{'PASS' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)

    checks = run_checks(args.layout, args.lanes, path_latency=_path_latency(args.path_latency), large=args.large,
                        report=show)
    failed = sum(1 for _, ok, _ in checks if not ok)
    print(f"{len(checks)} checks, {failed} failed")
    sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
