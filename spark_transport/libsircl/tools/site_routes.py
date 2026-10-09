#!/usr/bin/env python3
"""Per-rank libsircl settings of a SIRCL layout, from SIRCL's own route planner.

For a layout such as ``path:0-1``, ``path:0-3`` or ``ring:8`` and a lane count, SIRCL's routes module
(``sparkring_sircl.routes``: ``Layout.parse``, ``derive_routes``, ``chain_order``, ``forward_windows``,
``ring_window`` and ``ring_queues``) gives every rank's route map, the chain order of cable neighbors,
the forward window of every lane that crosses relays and the plan of the ring that closes the chain.
This tool prints them as libsircl's environment, one line per rank (rank r of the layout runs at
libsircl position r):

  LIBSIRCL_POSITION=r SIRCL_PEER_ROUTES=... LIBSIRCL_CHAIN_ORDER=... LIBSIRCL_FORWARD_WINDOWS=...
  LIBSIRCL_RING_WINDOW=...

LIBSIRCL_RING_WINDOW appears on every rank when the ring can run (no relay hairpin queue carries two of
its lanes): the bytes each of the rank's lanes toward the next rank of the ring keeps unacknowledged
through relays, 0 when those lanes are cables. It is absent on every rank when the ring cannot run, and
the ring schedules are then unavailable.

Usage (SIRCL's package on PYTHONPATH, or installed):
  python3 tools/site_routes.py --layout ring:8 --lanes 2 [--json]
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--layout", required=True)
    parser.add_argument("--lanes", type=int, default=2)
    parser.add_argument("--max-window", type=int, default=None, help="SIRCL_FORWARD_WINDOW_BYTES (SIRCL's default)")
    parser.add_argument("--chunk", type=int, default=None, help="SIRCL_FORWARD_CHUNK_BYTES (SIRCL's default)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    from sparkring_sircl import routes

    layout = routes.Layout.parse(args.layout)
    group = routes.derive_routes(layout, args.lanes)
    world = layout.world
    maps = [group.route_map(rank) for rank in range(world)]
    order = routes.chain_order(layout, maps)
    max_window = routes.DEFAULT_FORWARD_WINDOW if args.max_window is None else args.max_window
    chunk = routes.DEFAULT_FORWARD_CHUNK if args.chunk is None else args.chunk
    # The ring plan, as SIRCL's session makes it (_plan_ring): ring lanes post in chunks of the native
    # link window chunk; a rank whose ring lanes cross a relay queue gets the window, the others 0.
    ring_window, problems, relayed = None, ["no chain order"], set()
    if order is not None:
        from sparkring_sircl import protocol
        ring_window, problems = routes.ring_window(layout, maps, order, chunk=protocol.LINK_WINDOW_CHUNK,
                                                   queue_bytes=routes.DEFAULT_HAIRPIN_QUEUE)
        relayed = {rank for members in routes.ring_queues(layout, maps, order).values() for rank, _, _ in members}
    rows = []
    for rank in range(world):
        table = routes.forward_windows(layout, maps, rank, max_window=max_window, chunk=chunk)
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
        rows.append({"rank": rank, "env": env})
    if args.json:
        print(json.dumps({"layout": args.layout, "lanes": args.lanes, "chain_order": order,
                          "ring_problems": problems, "ranks": rows}, indent=1))
    else:
        for row in rows:
            print(" ".join(f"{name}={value}" for name, value in row["env"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
