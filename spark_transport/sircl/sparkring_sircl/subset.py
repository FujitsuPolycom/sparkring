"""Admission and conservative queue budgets for independent subset sessions.

Only a cable-connected arc leaving at most two adjacent non-members is admitted.
Those non-members have no relayed traffic in the universal shortest-path table,
including when TP and DCP sessions run concurrently. No shared-queue reservation
protocol is implemented; larger complements are refused even when idle.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from . import routes

VARIABLE = "SIRCL_SUBSET_RING"
FORWARD_CHUNK = 4096
# RC's smallest path MTU is 256. Allow 128 bytes per packet for framing,
# transport headers, CRC and queue accounting. A signaled credit may cover
# two WRs; each QP has at most 64 such credits (_roce_proxy.c).
MIN_MTU = 256
FRAME_ALLOWANCE = 128
CONTROL_BYTES_PER_LANE = 64 * 2 * FRAME_ALLOWANCE


def enabled(environ: Mapping[str, str] | None = None) -> bool:
    raw = (os.environ if environ is None else environ).get(VARIABLE, "0").strip() or "0"
    if raw not in ("0", "1"):
        raise routes.RouteError(f"{VARIABLE} must be 0 or 1, got {raw!r}")
    return raw == "1"


def order(layout: routes.Layout) -> tuple[int, ...]:
    """Admit a physical arc, including wrapping arcs and permuted rank order."""
    fabric = layout.fabric
    members = set(layout.positions)
    outside = set(fabric.positions) - members
    if (fabric.kind != "cycle" or not 4 <= len(fabric.positions) <= 8
            or len(members) < 3 or not 1 <= len(outside) <= 2):
        raise routes.RouteError("subset ring needs 3 or more ranks on a cycle of 4 to 8, leaving one or two "
                                "adjacent non-members; other subsets need coordinated relay reservations")
    starts = [p for p in members if fabric.cross(p, 1).dst not in members]
    if len(starts) != 1:
        raise routes.RouteError("subset ring ranks must form one cable-connected arc")
    positions = [starts[0]]
    while len(positions) < len(members):
        p = fabric.cross(positions[-1], 0).dst
        if p not in members or p in positions:
            raise routes.RouteError("subset ring ranks must form one cable-connected arc")
        positions.append(p)
    return tuple(layout.positions.index(p) for p in positions)


def validate(layout, maps, *, queue_bytes=routes.DEFAULT_HAIRPIN_QUEUE):
    """Reject custom lanes and queue capacities outside the proved model."""
    result = order(layout)
    if not 4096 <= queue_bytes <= routes.DEFAULT_HAIRPIN_QUEUE:
        raise routes.RouteError("subset ring hairpin capacity must be between 4096 and 524288 bytes")
    lanes = len(next(iter(maps[0].values())))
    derived = routes.derive_routes(layout, lanes)
    def roles(route_map):
        return {peer: tuple(routes.role_of(device) for device in devices)
                for peer, devices in route_map.items()}

    if [roles(m) for m in maps] != [roles(derived.route_map(rank)) for rank in range(layout.world)]:
        raise routes.RouteError("subset ring requires the universal shortest-path lane maps")
    return result


def payload_budget(layout, maps, *, queue_bytes=routes.DEFAULT_HAIRPIN_QUEUE) -> int:
    """One common payload allowance for each direction of every relay queue.

    Let n count all session lanes through q and its reverse. Reserve n*16384
    bytes for short writes, short final packets and their ACKs. The remaining
    bytes hold arena and ring payload simultaneously in both directions:
    2*B payload + 2*B/2 framing + 2*B/2 reverse ACKs = 4*B.
    This is deliberately below the legacy 75-percent payload ceiling.
    """
    validate(layout, maps, queue_bytes=queue_bytes)
    queues = routes.relay_queues(layout, maps)
    n = max((len(v) + len(queues.get((p, c, 1 - e), ()))
             for (p, c, e), v in queues.items()), default=0)
    return min(int(routes.RELAY_QUEUE_SHARE * queue_bytes),
               max(0, (queue_bytes - n * CONTROL_BYTES_PER_LANE) // 4))
