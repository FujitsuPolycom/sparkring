#!/usr/bin/env python3
"""Per-rank libsircl settings of a SIRCL layout, from SIRCL's own route planner.

For a layout such as ``path:0-1``, ``path:0-3`` or ``ring:8`` and a lane count, SIRCL's routes module
(``sparkring_sircl.routes``: ``Layout.parse``, ``derive_routes``, ``chain_order``, ``forward_windows``,
``ring_window`` and ``ring_queues``) gives every rank's route map, the chain order of cable neighbors,
the forward window of every lane that crosses relays and the plan of the ring that closes the chain, and
SIRCL's point-to-point budget (``sparkring_sircl.p2p.budget``: ``LaneSet``, ``session_reservation``,
``allocate``) the window of every relayed lane of a point-to-point channel. This tool prints them as
libsircl's environment, one line per rank (rank r of the layout runs at libsircl position r):

  LIBSIRCL_POSITION=r SIRCL_PEER_ROUTES=... LIBSIRCL_CHAIN_ORDER=... LIBSIRCL_FORWARD_WINDOWS=...
  LIBSIRCL_RING_WINDOW=... LIBSIRCL_P2P_WINDOWS=...

LIBSIRCL_RING_WINDOW appears on every rank when the ring can run (no relay hairpin queue carries two of
its lanes): the bytes each of the rank's lanes toward the next rank of the ring keeps unacknowledged
through relays, 0 when those lanes are cables. It is absent on every rank when the ring cannot run, and
the ring schedules are then unavailable.

LIBSIRCL_P2P_WINDOWS gives the point-to-point channels of a communicator of the layout's ranks (libsircl's
LIBSIRCL_P2P_CHANNELS) a window on every relayed lane that SIRCL's budget can give one: every relay hairpin
queue's share (75 % of SIRCL_HAIRPIN_QUEUE_BYTES) left after the reservation of ``--p2p-reserve``, divided
evenly among the channels' lanes through it, capped at ``--p2p-max-window`` and rounded down to whole
``--p2p-chunk`` chunks (SIRCL's ``p2p.budget.allocate``). ``session`` (the default, SIRCL's rule for the
channels of a group with a collective session) reserves first the collective session's forward windows,
and, with ``--ring-schedules`` (the site runs SIRCL_LARGE_SCHEDULE, SIRCL_GATHER_SCHEDULE or
SIRCL_SCATTER_SCHEDULE=ring), its ring windows; libsircl's ring lanes carry traffic only under the ring
schedules, so without them the ring windows are not reserved. ``none`` sizes the channels as the only
relayed traffic. A relayed lane left with less than one chunk gets no window, and libsircl then gives its
pair no channel; the JSON output names those pairs and the queue that left them without one.

``--session-share F`` (above 0, at most 1; default 1, the whole share) sizes the collective session
against F of every relay queue's share: its forward windows (and the ring window) are computed for a queue
of F x SIRCL_HAIRPIN_QUEUE_BYTES, so they are smaller where the share binds them, still at least one chunk
(SIRCL's rule), and the channels get what the session's windows leave of the whole share. ``--max-window``
caps the forward windows directly. On ring:8 with two lanes, the session's forward windows at the whole
share (64 to 128 KiB) leave no relayed pair a channel. Windows are whole chunks, so any share below 1 takes at
least one 32 KiB chunk from every relayed session lane (at 0.95: 32 to 96 KiB) and leaves every one of the 40
relayed ordered pairs a 32 KiB window; ``--max-window 32768`` (every session lane one chunk) does too.

Limit: the windows are one plan per process, as libsircl reads them; communicators that run at once
through the same relay queues are not budgeted against each other.

Usage (SIRCL's package on PYTHONPATH, or installed):
  python3 tools/site_routes.py --layout ring:8 --lanes 2 [--p2p-reserve session|none] [--ring-schedules]
      [--session-share F] [--json]
"""
from __future__ import annotations

import argparse
import json
import sys


def share_value(text: str) -> float:
    value = float(text)
    if not 0 < value <= 1:
        raise argparse.ArgumentTypeError(f"--session-share {text} must be above 0 and at most 1")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", required=True)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--max-window", type=int, default=None, help="SIRCL_FORWARD_WINDOW_BYTES (SIRCL's default)")
    parser.add_argument("--chunk", type=int, default=None, help="SIRCL_FORWARD_CHUNK_BYTES (SIRCL's default)")
    parser.add_argument("--p2p-reserve", choices=("session", "none"), default="session",
                        help="what the point-to-point windows leave room for in every relay queue (see above)")
    parser.add_argument("--p2p-max-window", type=int, default=None,
                        help="SIRCL_P2P_WINDOW_BYTES, the largest point-to-point window (SIRCL's default)")
    parser.add_argument("--p2p-chunk", type=int, default=None, help="SIRCL_P2P_CHUNK_BYTES (SIRCL's default)")
    parser.add_argument("--ring-schedules", action="store_true",
                        help="the site runs the ring schedules: the point-to-point windows leave room for the ring windows")
    parser.add_argument("--session-share", type=share_value, default=1.0,
                        help="the fraction of every relay queue's share the collective session is sized against")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    from sparkring_sircl import routes
    from sparkring_sircl.p2p import budget
    from sparkring_sircl.p2p import protocol as p2p_protocol

    layout = routes.Layout.parse(args.layout)
    group = routes.derive_routes(layout, args.lanes)
    world = layout.world
    maps = [group.route_map(rank) for rank in range(world)]
    order = routes.chain_order(layout, maps)
    max_window = routes.DEFAULT_FORWARD_WINDOW if args.max_window is None else args.max_window
    chunk = routes.DEFAULT_FORWARD_CHUNK if args.chunk is None else args.chunk
    # The queue the collective session is sized against (--session-share); the channels share the whole one.
    session_queue = int(args.session_share * routes.DEFAULT_HAIRPIN_QUEUE)
    # The ring plan, as SIRCL's session makes it (_plan_ring): ring lanes post in chunks of the native
    # link window chunk; a rank whose ring lanes cross a relay queue gets the window, the others 0.
    ring_window, problems, relayed = None, ["no chain order"], set()
    if order is not None:
        from sparkring_sircl import protocol
        ring_window, problems = routes.ring_window(layout, maps, order, chunk=protocol.LINK_WINDOW_CHUNK,
                                                   queue_bytes=session_queue)
        relayed = {rank for members in routes.ring_queues(layout, maps, order).values() for rank, _, _ in members}
    # Point-to-point windows of every ordered pair's relayed lanes (SIRCL's p2p/budget.py).
    p2p_max = p2p_protocol.DEFAULT_WINDOW_BYTES if args.p2p_max_window is None else args.p2p_max_window
    p2p_chunk = p2p_protocol.DEFAULT_CHUNK_BYTES if args.p2p_chunk is None else args.p2p_chunk
    lane_set = budget.LaneSet.of("layout", layout, args.lanes)
    reserved = {}
    if args.p2p_reserve == "session":
        settings = budget.SessionSettings(forward_window=max_window, forward_chunk=chunk,
                                          ring_schedule=args.ring_schedules and not problems)
        reserved = budget.session_reservation(lane_set, settings, queue_bytes=session_queue)
    allocation = budget.allocate([[lane_set]], reserved=reserved, max_window=p2p_max, chunk=p2p_chunk,
                                 queue_bytes=routes.DEFAULT_HAIRPIN_QUEUE)
    p2p_table = allocation.table(lane_set, args.lanes)
    p2p_unavailable = allocation.unavailable_pairs("layout")
    relayed_pairs = sum(1 for a in range(world) for b in range(world)
                        if a != b and any(lane.relays for lane in group.lanes_to(a, b)))
    rows = []
    for rank in range(world):
        table = routes.forward_windows(layout, maps, rank, max_window=max_window, chunk=chunk,
                                       queue_bytes=session_queue)
        env = {
            "LIBSIRCL_POSITION": str(rank),
            "SIRCL_PEER_ROUTES": ",".join(f"{peer}={'/'.join(devices)}" for peer, devices in sorted(maps[rank].items())),
        }
        if order is not None:
            env["LIBSIRCL_CHAIN_ORDER"] = ",".join(str(r) for r in order)
        windows = [f"{peer}={'/'.join(str(w) for w in table[peer])}" for peer in range(world)
                   if peer != rank and any(table[peer])]
        if windows:
            env["LIBSIRCL_FORWARD_WINDOWS"] = ",".join(windows)
            env["SIRCL_FORWARD_CHUNK_BYTES"] = str(chunk)
        if not problems:
            env["LIBSIRCL_RING_WINDOW"] = str(ring_window if rank in relayed else 0)
        p2p_windows = [f"{peer}={'/'.join(str(w) for w in p2p_table[rank][peer])}" for peer in range(world)
                       if peer != rank and any(p2p_table[rank][peer])]
        if p2p_windows:
            env["LIBSIRCL_P2P_WINDOWS"] = ",".join(p2p_windows)
            env["SIRCL_P2P_CHUNK_BYTES"] = str(p2p_chunk)
        rows.append({"rank": rank, "env": env})
    if args.json:
        print(json.dumps({"layout": args.layout, "lanes": args.lanes, "chain_order": order,
                          "ring_problems": problems, "p2p_reserve": args.p2p_reserve,
                          "ring_schedules": args.ring_schedules, "session_share": args.session_share,
                          "p2p_relayed_pairs": relayed_pairs,
                          "p2p_unavailable": [[a, b, reason] for (a, b), reason in sorted(p2p_unavailable.items())],
                          "ranks": rows}, indent=1))
    else:
        for row in rows:
            print(" ".join(f"{name}={value}" for name, value in row["env"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
