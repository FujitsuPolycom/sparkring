"""Latency model of small all-reduces on a ring session (torch-free).

At a few KiB an all-reduce on the ring of eight is bound by latency, not by
bandwidth: the bytes of a one-shot all-reduce of 8 KiB cross a Spark's NIC host
interface in under 3 us, while every op pays the progress thread's posting of
its lanes and the NIC relays of its longest paths. Toward 128 KiB the one-shot
all-reduce becomes bound by the bytes it sends. This model estimates the
network phases of one op for given posting orders, from the moment every
rank's progress thread starts posting a phase to the moment the last lane of
the phase has landed, and, with a calibration, whole ops and the one-shot
limit.

Model of one network phase:

1. Every rank posts its lanes one after another in its posting order, all
   lanes of one peer together, lane 0 first (:mod:`sparkring_sircl.posting`);
   the k-th lane a rank posts is handed to its NIC ``(k + 1) * post_us`` after
   the phase starts. A relayed lane with a forward window whose stripe exceeds
   one forward chunk or the window is posted after every other lane of the
   phase, as the native layer posts it (its chunking is not modelled).
2. A rank's NIC moves the lanes' bytes through its host interface in hand-off
   order at ``host_gbps``: a lane leaves once it is handed over and the bytes of
   the lanes before it have left.
3. A lane's last byte lands ``write_us + relays * relay_us`` after it leaves,
   where ``relays`` counts the NIC relays on its route.
4. Every rank's incoming bytes cross its host interface at ``host_gbps`` too,
   and every cable direction carries the bytes of the lanes routed over it at
   ``cable_gbps``: a phase lasts at least the busiest receiver's bytes and the
   busiest cable direction's bytes at those rates (one-shot on a path of four
   puts four times the message on its middle cable, three times through each
   host interface).
5. The phase ends when the last lane has landed.

A one-shot all-reduce is one phase in which every rank writes the whole
message to every peer. A two-shot all-reduce is two: chunk ``p`` to rank
``p``, then each rank's reduced chunk to every peer. Phases add up.

Outside the model, and so inside the difference between a measured time and
the model: the kernel launch or graph replay, staging the input into the send
slot, the progress thread noticing the doorbell, flag polling, the reduction
and the copy out. They are paid per op and, for the two-shot all-reduce, per
phase.

Parameters (:class:`Parameters`; each can be set):

- ``relay_us`` 0.75: the added one-way latency of one NIC relay (ib_write_lat
  across relays on the ring measured 0.6 to 0.9 us per relay);
- ``post_us`` 0.30: the progress thread's time to post one lane (the stripe's
  RDMA write and its flag write, one doorbell), assumed: the native layer of
  this package does not count posting time. :func:`allreduce` takes a value
  per rank, and the ring harness passes a session's measured value where the
  session reports ``post_ns_total`` and ``post_lanes_total``;
- ``write_us`` 0: the one-way latency of a direct RDMA write, excluded unless
  given, so that the model holds only posting, bytes and relays;
- ``host_gbps`` 24: a Spark's NIC host interface, each way;
- ``cable_gbps`` 24: one cable direction over both lanes;
- forward windows: the session defaults (``SIRCL_FORWARD_WINDOW_BYTES``), 0 for
  none.

Whole ops and the one-shot limit (:func:`predicted_us`,
:func:`oneshot_limit`): a :class:`Calibration` adds the costs outside the
network phases, per algorithm a fixed part and a part per KiB of message whose
slope changes at ``knee_kib`` (64 KiB), fitted (:func:`fit`) to graph-replay
medians of the harness's latency cases from 4 to 128 KiB. On both measured
layouts the cost per KiB grows above 64 KiB, where the one-shot stripes of two
lanes exceed the 32 KiB forward chunk and the kernels' grids double; one slope
fitted below 64 KiB would misplace the crossover of the path of four, which
lies above 64 KiB. A group size with a calibration of its own uses it; another
takes the nearest calibrated size, the one-shot slopes scaled by ``(W + 2)``
(the one-shot kernel stages the message, reads it from every rank and writes
the result: about ``W + 2`` times the message through host memory; the
two-shot kernel moves about three times the message whatever the group).
:func:`oneshot_limit` is the largest multiple of 4 KiB at and below which every
multiple of 4 KiB is predicted no slower with the one-shot all-reduce than with
the two-shot one; a session with a layout (``oneshot/runtime.py``) takes it as
its one-shot limit when ``SIRCL_ONESHOT_MAX_BYTES`` is unset.

Built-in calibrations, each fitted by :func:`fit` with the default parameters
to graph replay medians of two runs (one per posting order) from 4 to 128 KiB,
rows whose 90th percentile exceeds 1.3 times the median left out, then rows
more than three median absolute deviations off the fitted curve:

- :data:`RING8_GRAPH`: ``ring8-latency`` on the ring of eight, results
  ``20261007-045043`` (rank order) and ``20261007-045155`` (``ring-farthest``).
  The two-shot all-reduce was faster from 32 KiB farthest first and from 48 KiB
  in rank order; the model gives limits of 28,672 and 36,864 bytes;
- :data:`PATH4_GRAPH`: ``path4-latency`` on Sparks 0-3, results
  ``20261007-064421`` (rank order) and ``20261007-064524`` (``farthest``). The
  one-shot all-reduce was faster at 64 KiB and the two-shot at 96 KiB in both
  orders; the model gives limits of 73,728 bytes farthest first and 81,920 in
  rank order.

The limits of other group sizes and layouts are predictions: no latency run
has measured them.

Status: implemented; CPU tests in ``tests/test_latency_harness.py``. It is a
model with stated parameters, not a measured bound.
"""

from __future__ import annotations

import dataclasses
import functools
from collections.abc import Callable, Mapping, Sequence

from . import protocol as proto
from . import routes as routes_mod

ALGORITHMS = ("oneshot", "twoshot")


@dataclasses.dataclass(frozen=True)
class Parameters:
    relay_us: float = 0.75
    post_us: float = 0.30
    write_us: float = 0.0
    host_gbps: float = 24.0
    forward_window: int = routes_mod.DEFAULT_FORWARD_WINDOW
    forward_chunk: int = routes_mod.DEFAULT_FORWARD_CHUNK
    cable_gbps: float = 24.0

    def __post_init__(self) -> None:
        if min(self.relay_us, self.post_us, self.write_us) < 0 or min(self.host_gbps, self.cable_gbps) <= 0:
            raise ValueError("latency model parameters are non-negative and the rates positive")


@dataclasses.dataclass(frozen=True)
class Phase:
    """One network phase: its time and the lane that landed last."""

    us: float
    sender: int
    receiver: int
    lane: int
    handed_us: float      # when the progress thread handed that lane to the NIC
    left_us: float        # when its last byte left the sender's host interface
    relays: int
    ingress_us: float     # the busiest receiver's incoming bytes at the host rate
    cable_us: float = 0.0  # the busiest cable direction's bytes at the cable rate

    def to_json(self) -> dict[str, float | int]:
        return {"us": round(self.us, 3), "sender": self.sender, "receiver": self.receiver, "lane": self.lane,
                "handed_us": round(self.handed_us, 3), "left_us": round(self.left_us, 3), "relays": self.relays,
                "ingress_us": round(self.ingress_us, 3), "cable_us": round(self.cable_us, 3)}


@dataclasses.dataclass(frozen=True)
class Estimate:
    algorithm: str
    nbytes: int
    us: float
    phases: tuple[Phase, ...]

    def to_json(self) -> dict[str, object]:
        return {"algorithm": self.algorithm, "bytes": self.nbytes, "us": round(self.us, 3),
                "phases": [phase.to_json() for phase in self.phases]}


def _per_rank(value: float | Sequence[float], world: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * world
    values = [float(item) for item in value]
    if len(values) != world:
        raise ValueError(f"a per-rank value needs {world} entries, got {len(values)}")
    return values


@functools.lru_cache(maxsize=64)
def _routes(layout: routes_mod.Layout, lanes: int, window: int,
            chunk: int) -> tuple[routes_mod.GroupRoutes, tuple[tuple[tuple[int, ...], ...], ...]]:
    """The lanes of every rank and every rank's forward windows (``[rank][peer][lane]``; zeros without windows)."""
    group = routes_mod.derive_routes(layout, lanes)
    maps = [group.route_map(rank) for rank in range(layout.world)]
    windows = tuple(
        tuple(tuple(row) for row in (routes_mod.forward_windows(layout, maps, rank, max_window=window, chunk=chunk)
                                     if window > 0 else [[0] * lanes for _ in range(layout.world)]))
        for rank in range(layout.world))
    return group, windows


def phase(layout: routes_mod.Layout, lanes: int, orders: Sequence[Sequence[int]],
          lane_bytes: Callable[[int, int, int], int], parameters: Parameters = Parameters(),
          post_us: float | Sequence[float] | None = None) -> Phase:
    """One network phase in which rank ``s`` writes ``lane_bytes(s, peer, lane)`` bytes on every lane.

    ``orders[s]`` lists the peers of rank ``s`` in its posting order;
    ``post_us`` (one value, or one per rank) replaces ``parameters.post_us``.
    """
    world = layout.world
    group, windows_by_rank = _routes(layout, int(lanes), parameters.forward_window, parameters.forward_chunk)
    posting = _per_rank(parameters.post_us if post_us is None else post_us, world)
    rate = parameters.host_gbps * 1e3          # bytes per microsecond
    incoming = [0] * world
    cables: dict[tuple[int, int], int] = {}
    worst: Phase | None = None
    for sender in range(world):
        order = [int(peer) for peer in orders[sender]]
        if sorted(order) != [peer for peer in range(world) if peer != sender]:
            raise ValueError(f"the posting order of rank {sender} must name every peer once, got {order}")
        windows = windows_by_rank[sender]
        now_lanes, later_lanes = [], []
        for peer in order:
            for route in group.lanes_to(sender, peer):
                size = int(lane_bytes(sender, peer, route.lane))
                window = windows[peer][route.lane]
                deferred = bool(window) and (size > parameters.forward_chunk or size + 4 > window)
                (later_lanes if deferred else now_lanes).append((peer, route.lane, size, len(route.relays)))
                for step in route.steps:
                    cables[(step.src, step.dst)] = cables.get((step.src, step.dst), 0) + size
        left = 0.0
        for index, (peer, lane, size, relays) in enumerate(now_lanes + later_lanes):
            handed = (index + 1) * posting[sender]
            left = max(handed, left) + size / rate
            landed = left + parameters.write_us + relays * parameters.relay_us
            incoming[peer] += size
            if worst is None or landed > worst.us:
                worst = Phase(landed, sender, peer, lane, handed, left, relays, 0.0)
    ingress = max(incoming) / rate
    cable = max(cables.values(), default=0) / (parameters.cable_gbps * 1e3)
    assert worst is not None
    return dataclasses.replace(worst, us=max(worst.us, ingress, cable), ingress_us=ingress, cable_us=cable)


def allreduce(layout: routes_mod.Layout, lanes: int, nbytes: int, algorithm: str,
              orders: Sequence[Sequence[int]], parameters: Parameters = Parameters(),
              post_us: float | Sequence[float] | None = None) -> Estimate:
    """The network phases of one ``algorithm`` all-reduce of ``nbytes`` on ``layout``."""
    if algorithm not in ALGORITHMS:
        raise ValueError(f"the latency model covers {', '.join(ALGORITHMS)}, not {algorithm!r}")
    if nbytes <= 0 or nbytes % proto.PACK_BYTES:
        raise ValueError("all-reduce messages are positive multiples of 16 bytes")
    world = layout.world
    packs = nbytes // proto.PACK_BYTES

    def striped(count: int, lane: int) -> int:
        return proto.stripe(count, lanes, lane)[1] * proto.PACK_BYTES

    if algorithm == "oneshot":
        phases = (phase(layout, lanes, orders, lambda s, p, lane: striped(packs, lane), parameters, post_us),)
    else:
        scatter = phase(layout, lanes, orders,
                        lambda s, p, lane: striped(proto.chunk(packs, world, p)[1], lane), parameters, post_us)
        gather = phase(layout, lanes, orders,
                       lambda s, p, lane: striped(proto.chunk(packs, world, s)[1], lane), parameters, post_us)
        phases = (scatter, gather)
    return Estimate(algorithm, int(nbytes), sum(item.us for item in phases), phases)


@dataclasses.dataclass(frozen=True)
class Calibration:
    """Costs outside the network phases of one op, fitted at group size ``world``: per algorithm a fixed
    part (us) and a part per KiB of message whose slope changes at ``knee_kib`` (us per KiB below and
    above it). For another group size the one-shot slopes scale by ``(W + 2) / (world + 2)``."""

    world: int
    oneshot_us: float
    oneshot_us_per_kib: float
    oneshot_us_per_kib_above: float
    twoshot_us: float
    twoshot_us_per_kib: float
    twoshot_us_per_kib_above: float
    knee_kib: float = 64.0

    def extra_us(self, algorithm: str, world: int, nbytes: int) -> float:
        kib = nbytes / 1024
        below, above = min(kib, self.knee_kib), max(0.0, kib - self.knee_kib)
        if algorithm == "oneshot":
            scale = (world + 2) / (self.world + 2)
            return (self.oneshot_us + scale * (self.oneshot_us_per_kib * below
                                               + self.oneshot_us_per_kib_above * above))
        return self.twoshot_us + self.twoshot_us_per_kib * below + self.twoshot_us_per_kib_above * above


# Fitted by :func:`fit` with the default parameters to the results named in the module docstring.
RING8_GRAPH = Calibration(world=8, oneshot_us=15.93, oneshot_us_per_kib=0.2609, oneshot_us_per_kib_above=0.1364,
                          twoshot_us=21.92, twoshot_us_per_kib=0.1302, twoshot_us_per_kib_above=0.1524)
PATH4_GRAPH = Calibration(world=4, oneshot_us=14.04, oneshot_us_per_kib=0.0807, oneshot_us_per_kib_above=0.1906,
                          twoshot_us=18.25, twoshot_us_per_kib=0.1281, twoshot_us_per_kib_above=0.0959)
CALIBRATIONS = {4: PATH4_GRAPH, 8: RING8_GRAPH}
LIMIT_STEP = 4096


def calibration_for(world: int) -> Calibration:
    """The calibration of ``world`` ranks, else that of the nearest calibrated group size (larger on a tie)."""
    nearest = min(CALIBRATIONS, key=lambda size: (abs(size - world), -size))
    return CALIBRATIONS[nearest]


def predicted_us(layout: routes_mod.Layout, lanes: int, nbytes: int, algorithm: str,
                 orders: Sequence[Sequence[int]], parameters: Parameters = Parameters(),
                 calibration: Calibration | None = None, post_us: float | Sequence[float] | None = None) -> float:
    """One whole op: its network phases plus the calibrated costs outside them."""
    calibration = calibration or calibration_for(layout.world)
    network = allreduce(layout, lanes, nbytes, algorithm, orders, parameters, post_us).us
    return network + calibration.extra_us(algorithm, layout.world, nbytes)


def oneshot_limit(layout: routes_mod.Layout, lanes: int, order: str = "farthest",
                  parameters: Parameters = Parameters(), calibration: Calibration | None = None, *,
                  step: int = LIMIT_STEP, cap: int = 131072) -> int:
    """The largest multiple of ``step`` up to ``cap`` at and below which every multiple of ``step`` is
    predicted no slower one-shot than two-shot; at least ``step`` (or ``cap`` when smaller)."""
    if step <= 0 or step % proto.PACK_BYTES:
        raise ValueError("the limit step is a positive multiple of 16 bytes")
    orders = orders_for(layout, lanes, order)
    limit = min(step, cap)
    size = step
    while size <= cap:
        oneshot = predicted_us(layout, lanes, size, "oneshot", orders, parameters, calibration)
        twoshot = predicted_us(layout, lanes, size, "twoshot", orders, parameters, calibration)
        if oneshot > twoshot:
            break
        limit = size
        size += step
    return limit


def _least_squares(rows: Sequence[Sequence[float]], values: Sequence[float]) -> list[float]:
    """Coefficients minimising ``sum (row . c - value)^2`` (normal equations, Gaussian elimination)."""
    size = len(rows[0])
    matrix = [[sum(row[i] * row[j] for row in rows) for j in range(size)]
              + [sum(row[i] * value for row, value in zip(rows, values))] for i in range(size)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(matrix[index][column]))
        if abs(matrix[pivot][column]) < 1e-12:
            raise ValueError("the calibration rows do not determine every coefficient")
        matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
        for index in range(size):
            if index != column:
                factor = matrix[index][column] / matrix[column][column]
                matrix[index] = [a - factor * b for a, b in zip(matrix[index], matrix[column])]
    return [matrix[index][size] / matrix[index][index] for index in range(size)]


def fit(cases: Sequence[Mapping[str, object]], layout: routes_mod.Layout, lanes: int,
        parameters: Parameters = Parameters(), *, max_bytes: int | None = None, max_spread: float = 1.3,
        knee_kib: float = 64.0) -> Calibration:
    """A calibration from merged latency rows (``summary.merge`` cases that name a posting order).

    Graph rows (of at most ``max_bytes`` when given) whose 90th percentile is at most ``max_spread``
    times the median; per algorithm, the measured median minus the network phases as a fixed part plus
    slopes per KiB below and above ``knee_kib``, fitted again without the rows more than three median
    absolute deviations off the first fit. Rows on one side of the knee only give one slope for both.
    """
    found: dict[str, tuple[float, float, float]] = {}
    for algorithm in ALGORITHMS:
        points = []
        for case in cases:
            if (case.get("collective") != f"all_reduce_{algorithm}" or case.get("mode") != "graph"
                    or case.get("post_order") is None or (max_bytes is not None and int(case["bytes"]) > max_bytes)):
                continue
            median, tail = case.get("slowest_p50_us"), case.get("slowest_p90_us")
            if not median or (tail and tail > max_spread * median):
                continue
            orders = orders_for(layout, lanes, str(case["post_order"]))
            network = allreduce(layout, lanes, int(case["bytes"]), algorithm, orders, parameters).us
            kib = int(case["bytes"]) / 1024
            points.append((min(kib, knee_kib), max(0.0, kib - knee_kib), float(median) - network))
        if len(points) < 2 or len({point[0] + point[1] for point in points}) < 2:
            raise ValueError(f"a {algorithm} calibration needs graph rows of at least two sizes, found {len(points)}")

        def solve(chosen: Sequence[tuple[float, float, float]]) -> tuple[float, float, float]:
            split = any(point[1] > 0 for point in chosen) and len({point[0] for point in chosen}) > 1
            if split:
                fixed, below, above = _least_squares([(1.0, point[0], point[1]) for point in chosen],
                                                     [point[2] for point in chosen])
                return fixed, below, above
            fixed, slope = _least_squares([(1.0, point[0] + point[1]) for point in chosen],
                                          [point[2] for point in chosen])
            return fixed, slope, slope

        first = solve(points)

        def residual(point: tuple[float, float, float], coefficients: tuple[float, float, float]) -> float:
            return abs(point[2] - coefficients[0] - coefficients[1] * point[0] - coefficients[2] * point[1])

        deviations = sorted(residual(point, first) for point in points)
        deviation = deviations[len(deviations) // 2]
        kept = [point for point in points if residual(point, first) <= 3 * deviation]
        found[algorithm] = solve(kept) if len(kept) >= 2 and len({p[0] + p[1] for p in kept}) >= 2 else first
    one, two = found["oneshot"], found["twoshot"]
    return Calibration(world=layout.world, oneshot_us=round(one[0], 2), oneshot_us_per_kib=round(one[1], 4),
                       oneshot_us_per_kib_above=round(one[2], 4), twoshot_us=round(two[0], 2),
                       twoshot_us_per_kib=round(two[1], 4), twoshot_us_per_kib_above=round(two[2], 4),
                       knee_kib=knee_kib)


def orders_for(layout: routes_mod.Layout, lanes: int, order: str) -> tuple[tuple[int, ...], ...]:
    """Every rank's peers in the named posting order on ``layout``."""
    from . import posting

    group = routes_mod.derive_routes(layout, lanes)
    return tuple(posting.resolve(order, rank, layout.world, layout, group.route_map(rank))
                 for rank in range(layout.world))


def parameters_from(options: Mapping[str, object]) -> Parameters:
    """:class:`Parameters` from harness options (``latency_relay_us``, ``latency_post_us``, ``latency_write_us``,
    ``host_cap_gbps``, ``cable_gbps``, ``forward_window``)."""
    window = options.get("forward_window")
    return Parameters(
        relay_us=float(options.get("latency_relay_us", Parameters.relay_us)),
        post_us=float(options.get("latency_post_us", Parameters.post_us)),
        write_us=float(options.get("latency_write_us", Parameters.write_us)),
        host_gbps=float(options.get("host_cap_gbps", Parameters.host_gbps)),
        cable_gbps=float(options.get("cable_gbps", Parameters.cable_gbps)),
        forward_window=routes_mod.DEFAULT_FORWARD_WINDOW if window is None else int(window),
    )


__all__ = ["ALGORITHMS", "CALIBRATIONS", "Calibration", "Estimate", "LIMIT_STEP", "PATH4_GRAPH", "Parameters",
           "Phase", "RING8_GRAPH", "allreduce", "calibration_for", "fit", "main", "oneshot_limit", "orders_for",
           "parameters_from", "phase", "predicted_us"]


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m sparkring_sircl.latency_model [result.json ...]``: the calibration fitted to the merged
    results of latency runs on one layout (else the built-in one) and the one-shot limits it gives."""
    import argparse
    import json
    import pathlib

    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.latency_model", description=main.__doc__)
    parser.add_argument("results", nargs="*", type=pathlib.Path,
                        help="result.json of ring-latency or path4-latency runs on one layout")
    parser.add_argument("--relay-us", type=float, default=Parameters.relay_us)
    parser.add_argument("--post-us", type=float, default=Parameters.post_us)
    parser.add_argument("--lanes", type=int, default=2)
    args = parser.parse_args(argv)
    parameters = Parameters(relay_us=args.relay_us, post_us=args.post_us)
    calibration = None
    if args.results:
        cases: list[Mapping[str, object]] = []
        layouts = set()
        for path in args.results:
            result = json.loads(path.read_text(encoding="utf-8"))
            layouts.update(group["layout"] for group in result["groups"])
            cases.extend(result["cases"])
        if len(layouts) != 1:
            parser.error(f"the results must come from one layout, got {sorted(layouts)}")
        calibration = fit(cases, routes_mod.Layout.parse(layouts.pop()), args.lanes, parameters)
        print(json.dumps(dataclasses.asdict(calibration)))
    for text, order in (("ring:8", "farthest"), ("ring:8", "rank"), ("path:0-3", "farthest"),
                        ("path:0-3", "rank"), ("ring:4", "farthest"), ("ring:2", "farthest")):
        layout = routes_mod.Layout.parse(text)
        chosen = calibration or calibration_for(layout.world)
        limit = oneshot_limit(layout, args.lanes, order, parameters, chosen)
        print(f"{text}, {args.lanes} lanes, {order}: SIRCL_ONESHOT_MAX_BYTES={limit} (calibration of "
              f"{chosen.world} ranks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
