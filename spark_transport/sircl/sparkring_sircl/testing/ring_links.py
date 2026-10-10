"""Ring collectives on the in-memory verbs stand-in: the native link layer of every rank in this process,
with the ring kernel's part played by threads (torch-free).

:class:`RingLinks` sets up the links of a :class:`.fabric.LocalSession` (the chain order of the layout, the
ring that closes it and, on a path, the window of the ring lanes through relays, as a session does) and
:func:`run_ring_ops` runs ring all-gathers, reduce-scatters and all-reduces back to back through them.
Each rank's part follows the ring kernel (``oneshot/_links_cute.py``, ``LinkRing``) role by role, each role
in its own thread: ``start`` stages the partial of piece ``t`` for the owner at ring index ``i - 1``
(link 2, type 0 of round ``t``); ``relay`` waits for inbound partial ``r`` of round ``t`` (piece
``t - r D``), adds this rank's values of its owner and stages the sum as outbound type ``r + 1``
``D`` rounds later, or completes the own chunk at the last type (the all-reduce stages the result as
link 3's own item); ``own`` stages the all-gather's own piece (link 3, type 0); ``copy`` copies inbound
finished piece ``r`` of round ``t`` (piece ``t - r D3``) to its owner's place. Items outside a type's
pieces are empty and only consumed. Payloads are summed as little-endian 32-bit words modulo 2^32, so
every output has one exact value whatever the order.
"""

from __future__ import annotations

import ctypes
import threading
import time
from collections.abc import Sequence

from .. import protocol as proto
from .. import routes as routes_mod
from .fabric import LocalSession, _add_words, payload

RING_OPS = {"gather": proto.LinkOp.RING_GATHER, "scatter": proto.LinkOp.RING_SCATTER,
            "reduce": proto.LinkOp.RING_REDUCE}


class RingLinks(LocalSession):
    """Every rank of one session with its link area, ring links set as a session sets them."""

    def __init__(self, library_path: str, layout: routes_mod.Layout, *, lanes: int = 2, slots: int = 8,
                 slot_bytes: int = 4096, timeout: float = 20.0) -> None:
        arena = proto.ArenaLayout(layout.world, 16384)
        self.link_offset = proto.chain_offset(arena.total_bytes)
        self.links = proto.LinkLayout(lanes, slots, slot_bytes)
        super().__init__(library_path, layout, lanes=lanes, slot_bytes=16384,
                         region_bytes=self.link_offset + self.links.total_bytes)
        maps = [self.routes.route_map(rank) for rank in range(self.world)]
        order = routes_mod.chain_order(layout, maps)
        if order is None:
            raise ValueError(f"{layout.identity()} is not a chain of cable neighbors")
        window, problems = routes_mod.ring_window(layout, maps, order, chunk=proto.LINK_WINDOW_CHUNK)
        if problems:
            raise ValueError("the ring cannot run: " + "; ".join(problems))
        relayed = {rank for members in routes_mod.ring_queues(layout, maps, order).values()
                   for rank, _, _ in members}
        self.order = tuple(order)
        self.slots = slots
        self.slot_size = slot_bytes
        self.timeout = timeout
        self._ring_window = window
        self._relayed = relayed
        # The kernel's device counters per rank: link 2 own and inbound, link 3 own and inbound, sequence.
        self.counters = [{"own2": 0, "in2": 0, "own3": 0, "in3": 0, "seq": 0} for _ in range(self.world)]
        self.failure: str | None = None

    def connect(self, lane_check_ms: int = 2000) -> None:
        """Connect every rank, set its links (before its progress thread starts) and start it."""
        self.fabric.start()
        blobs = [proxy.local_blob() for proxy in self.proxies]
        for proxy in self.proxies:
            proxy.connect(blobs)
        for proxy in self.proxies:
            proxy.lane_check(lane_check_ms)
        for rank, proxy in enumerate(self.proxies):
            index = self.order.index(rank)
            prev_rank = self.order[index - 1] if index > 0 else -1
            next_rank = self.order[index + 1] if index < self.world - 1 else -1
            proxy.set_links(prev_rank, next_rank, index, self.slots, self.slot_size, self.link_offset,
                            ring_prev=self.order[(index - 1) % self.world],
                            ring_next=self.order[(index + 1) % self.world],
                            ring_window=self._ring_window if rank in self._relayed else 0)
        for proxy in self.proxies:
            proxy.start()

    def delay_link_ops(self, rank: int, delay_us: int) -> None:
        """Make rank ``rank``'s progress thread take each link op ``delay_us`` after its doorbell."""
        proxy = self.proxies[rank]
        function = proxy._lib.roce_test_delay_link_ops
        function.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        function.restype = None
        function(proxy._handle(), int(delay_us))

    # -- the link area -----------------------------------------------------------------------------

    def _at(self, rank: int, offset: int) -> int:
        return self.addresses[rank] + self.link_offset + offset

    def _load(self, rank: int, offset: int) -> int:
        return self.library.roce_load_acquire_u32(self._at(rank, offset))

    def _store(self, rank: int, offset: int, value: int) -> None:
        self.library.roce_store_release_u32(self._at(rank, offset), value & 0xFFFFFFFF)

    def _wait(self, what: str, condition) -> None:
        deadline = time.monotonic() + self.timeout
        while not condition():
            if self.failure is not None:
                raise RuntimeError(f"stopped: {self.failure}")
            if time.monotonic() > deadline:
                raise TimeoutError(what)
            time.sleep(0.00005)

    def _wait_own_slot(self, rank: int, link: int, item: int) -> None:
        sent = self.links.sent_word(link)
        self._wait(f"rank {rank} link {link} own slot of item {item}",
                   lambda: ((self._load(rank, sent) - (item + 1 - self.slots)) & 0xFFFFFFFF) < 1 << 31)

    def _wait_inbound(self, rank: int, link: int, g: int) -> None:
        m = g % self.slots
        tag = (g + 1) & 0xFFFFFFFF
        for lane in range(self.lanes):
            line = self.links.flag_line(link, m, lane)
            self._wait(f"rank {rank} link {link} inbound item {g} lane {lane}", lambda: self._load(rank, line) == tag)

    def _stage(self, rank: int, link: int, item: int, data: bytes) -> None:
        ctypes.memmove(self._at(rank, self.links.own_slot(link, item % self.slots)), data, len(data))
        self._store(rank, self.links.ready_word(link, item % self.slots), item + 1)

    def _received(self, rank: int, link: int, g: int, nbytes: int) -> bytes:
        return ctypes.string_at(self._at(rank, self.links.recv_slot(link, g % self.slots)), nbytes)

    def _consume(self, rank: int, link: int, g: int) -> None:
        self._store(rank, self.links.consumed_word(link, g % self.slots), g + 1)

    def _ring(self, rank: int, op: int, nbytes: int, piece: int, stagger: int, gather_stagger: int) -> None:
        """Ring the link doorbell of one op, as block 0 of the ring kernel does."""
        seq = self.counters[rank]["seq"] + 1
        params = self.links.param_word(seq & 1)
        self._store(rank, params, proto.ring_op_word(op, stagger, gather_stagger))
        self._store(rank, params + 4, nbytes)
        self._store(rank, params + 8, piece)
        self._store(rank, self.links.ctrl_off, seq)

    # -- one rank's part of one op --------------------------------------------------------------------

    def _rank_op(self, rank: int, kind: str, block: Sequence[bytes], piece: int, stagger: int, gather_stagger: int,
                 output: bytearray) -> None:
        """Rank ``rank``'s roles of one ring op: ``block`` holds its W chunks (reduce-scatter, all-reduce;
        chunk ``k`` belongs to place ``k``) or its one shard (all-gather); ``output`` receives its result."""
        world, order = self.world, self.order
        index = order.index(rank)
        per_round = world - 1
        nbytes = len(block[0])
        pieces = proto.link_pieces(nbytes, piece)
        rounds2 = proto.ring_rounds(pieces, world, stagger)
        rounds3 = proto.ring_rounds(pieces, world, gather_stagger)
        start = dict(self.counters[rank])
        self._ring(rank, int(RING_OPS[kind]), nbytes, piece, stagger if kind != "gather" else 0,
                   gather_stagger if kind != "scatter" else 0)

        def span(p: int) -> slice:
            return slice(p * piece, p * piece + proto.link_piece_bytes(nbytes, piece, p))

        def place(offset: int) -> int:
            return order[(index + offset) % world]

        def role_start() -> None:
            for p in range(pieces):
                item = start["own2"] + p * per_round
                self._wait_own_slot(rank, 2, item)
                self._stage(rank, 2, item, block[place(-1)][span(p)])

        def role_relay() -> None:
            for i in range(rounds2 * per_round):
                g = start["in2"] + i
                p, r = proto.ring_partial(i, world, pieces, stagger)
                out_item = start["own2"] + i + stagger * per_round + 1
                self._wait_inbound(rank, 2, g)
                if p is not None:
                    total = bytearray(self._received(rank, 2, g, span(p).stop - span(p).start))
                    if r == per_round - 1:
                        _add_words(total, block[place(0)][span(p)])
                        if kind == "reduce":
                            output[place(0) * nbytes + span(p).start:place(0) * nbytes + span(p).stop] = total
                            self._wait_own_slot(rank, 3, start["own3"] + p)
                            self._stage(rank, 3, start["own3"] + p, bytes(total))
                        else:
                            output[span(p)] = total
                    else:
                        _add_words(total, block[place(-2 - r)][span(p)])
                        self._wait_own_slot(rank, 2, out_item)
                        self._stage(rank, 2, out_item, bytes(total))
                self._consume(rank, 2, g)

        def role_own() -> None:
            for p in range(pieces):
                item = start["own3"] + p
                self._wait_own_slot(rank, 3, item)
                data = block[0][span(p)]
                output[place(0) * nbytes + span(p).start:place(0) * nbytes + span(p).stop] = data
                self._stage(rank, 3, item, data)

        def role_copy() -> None:
            for i in range(rounds3 * per_round):
                g = start["in3"] + i
                p, r = proto.ring_partial(i, world, pieces, gather_stagger)
                self._wait_inbound(rank, 3, g)
                if p is not None:
                    owner = place(-1 - r)
                    data = self._received(rank, 3, g, span(p).stop - span(p).start)
                    output[owner * nbytes + span(p).start:owner * nbytes + span(p).stop] = data
                self._consume(rank, 3, g)

        roles = {"gather": (role_own, role_copy), "scatter": (role_start, role_relay),
                 "reduce": (role_start, role_relay, role_copy)}[kind]
        errors: list[str] = []

        def run(role) -> None:
            try:
                role()
            except Exception as error:  # noqa: BLE001 - every role's failure stops the op
                errors.append(f"rank {rank} {role.__name__}: {type(error).__name__}: {error}")
                self.failure = self.failure or errors[-1]

        threads = [threading.Thread(target=run, args=(role,), daemon=True) for role in roles]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        if errors:
            raise RuntimeError("; ".join(errors))
        counters = self.counters[rank]
        if kind != "gather":
            counters["own2"] += rounds2 * per_round
            counters["in2"] += rounds2 * per_round
        if kind != "scatter":
            counters["own3"] += rounds3
            counters["in3"] += rounds3 * per_round
        counters["seq"] += 1


def expected_items(links: RingLinks, steps: Sequence[tuple[str, int, int, int, int]]) -> list[int]:
    """Outbound link items every rank posts for ``steps`` of (kind, bytes per rank, piece, stagger,
    gather stagger): each staggered link's rounds (``protocol.ring_rounds``), empty items included."""
    counts = [0] * links.world
    for kind, nbytes, piece, stagger, gather_stagger in steps:
        pieces = proto.link_pieces(nbytes, piece)
        for rank in range(links.world):
            index = links.order.index(rank)
            for link, d in ((2, stagger), (3, gather_stagger)):
                rounds = proto.link_rounds(RING_OPS[kind], links.world, index, link)
                counts[rank] += proto.ring_rounds(pieces, links.world, d) * rounds.out
    return counts


def run_ring_ops(links: RingLinks, steps: Sequence[tuple[str, int, int, int, int]], seed: int = 1) -> None:
    """Ring ops back to back on every rank, each rank's ops in its own thread: ``steps`` of (kind ``gather``,
    ``scatter`` or ``reduce``, bytes per rank, piece bytes, stagger D, all-gather stagger D3). Raises on a
    wrong output, a failed or stuck rank, or link items other than :func:`expected_items` gives."""
    world = links.world
    inputs, wanted = [], []
    for number, (kind, nbytes, piece, stagger, gather_stagger) in enumerate(steps):
        if kind == "gather":
            shards = [payload(rank, seed + number, nbytes) for rank in range(world)]
            inputs.append([[shard] for shard in shards])
            wanted.append([b"".join(shards)] * world)
            continue
        blocks = [[payload(rank, (seed + number) * 64 + k, nbytes) for k in range(world)] for rank in range(world)]
        inputs.append(blocks)
        sums = []
        for k in range(world):
            total = bytearray(nbytes)
            for rank in range(world):
                _add_words(total, blocks[rank][k])
            sums.append(bytes(total))
        wanted.append(sums if kind == "scatter" else [b"".join(sums)] * world)
    before = [proxy.stats()["link_items_posted"] for proxy in links.proxies]
    outputs: list[list[bytes | None]] = [[None] * len(steps) for _ in range(world)]
    errors: list[str] = []

    def rank_ops(rank: int) -> None:
        try:
            for number, (kind, nbytes, piece, stagger, gather_stagger) in enumerate(steps):
                size = nbytes if kind == "scatter" else nbytes * world
                output = bytearray(size)
                links._rank_op(rank, kind, inputs[number][rank], piece, stagger, gather_stagger, output)
                outputs[rank][number] = bytes(output)
        except Exception as error:  # noqa: BLE001 - reported with every rank's failure
            errors.append(f"rank {rank}: {error}")
            links.failure = links.failure or errors[-1]

    threads = [threading.Thread(target=rank_ops, args=(rank,), daemon=True) for rank in range(world)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise AssertionError("; ".join(errors))
    wrong = [f"{steps[number][0]} {number} rank {rank}" for number in range(len(steps)) for rank in range(world)
             if outputs[rank][number] != wanted[number][rank]]
    if wrong:
        raise AssertionError("wrong outputs: " + ", ".join(wrong[:6]))
    # The progress threads post an op's last items after the kernels' part finished; wait for them.
    counts = expected_items(links, steps)
    deadline = time.monotonic() + links.timeout
    while True:
        items = [proxy.stats()["link_items_posted"] - count for proxy, count in zip(links.proxies, before)]
        if items == counts or time.monotonic() > deadline:
            break
        time.sleep(0.001)
    if items != counts:
        raise AssertionError(f"link items {items}, the staggered rounds give {counts}")


__all__ = ["RING_OPS", "RingLinks", "expected_items", "run_ring_ops"]
