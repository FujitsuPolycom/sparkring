"""Posting orders of the native progress thread (torch-free).

In every network phase of an op, the progress thread of a rank posts one RDMA
write per lane (the lane's stripe, then the lane's flag write on the same queue
pair), peer after peer in the rank's posting order, all lanes of one peer
together, lane 0 first. The order decides when each write starts, never what it
carries, so every rank may use its own order and ranks need not agree on it.

Orders (``SIRCL_POST_ORDER``, the session's ``post_order`` argument):

- ``rank``: peers in ascending rank, the session default;
- ``ring-farthest``: folded rank distance from ``W // 2`` down to 1, the
  clockwise peer before the counter-clockwise one
  (``protocol.ring_farthest``). This is farthest first only on a ring whose
  ranks sit in ring order;
- ``farthest``: the session's own lanes on its layout, most NIC relays first
  (:func:`farthest_first`). On a ring in ring order it equals
  ``ring-farthest``; on paths and on subgroups it still posts the longest relay
  paths first;
- an explicit list that names every peer once.

Farthest first starts the writes that cross the most relays (each relay adds
latency) before the direct ones, so a phase whose duration is set by posting
rather than by the wire ends sooner by up to the relay latency of the longest
path.

Status: implemented (CPU tests in ``tests/test_posting.py``); the native layer
takes any order as an explicit peer list.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from . import protocol as proto
from . import routes as routes_mod

POST_ORDERS = ("rank", "ring-farthest", "farthest")


def lane_relays(layout: routes_mod.Layout, rank: int, route_map: Mapping[int, Sequence[str]], *,
                roles: Mapping[str, routes_mod.Role] | None = None) -> dict[int, tuple[int, ...]]:
    """Relays crossed by every lane of ``rank``: ``{peer: (relays of lane 0, relays of lane 1, ...)}``.

    A lane leaves through the port of its device's role and follows the cables
    of ``layout`` to the peer; its relays are the Sparks between the ends.
    """
    found: dict[int, tuple[int, ...]] = {}
    here = layout.positions[rank]
    for peer, devices in sorted(route_map.items()):
        counts = []
        for device in devices:
            role = routes_mod.role_of(device, roles)
            if role is None:
                raise routes_mod.RouteError(f"device {device} of the lane from rank {rank} to rank {peer} has no "
                                            "known role")
            steps = layout.fabric.walk(here, role.port, layout.positions[peer], limit=len(layout.fabric.positions))
            if steps is None:
                raise routes_mod.RouteError(f"the lane from rank {rank} to rank {peer} on {device} does not reach "
                                            "the peer on this layout")
            counts.append(len(steps) - 1)
        found[int(peer)] = tuple(counts)
    return found


def farthest_first(layout: routes_mod.Layout, rank: int, route_map: Mapping[int, Sequence[str]], *,
                   roles: Mapping[str, routes_mod.Role] | None = None) -> tuple[int, ...]:
    """The peers of ``rank`` with the most relays first.

    A peer counts the relays of its longest lane. Peers with equal counts
    alternate between the egress ports of their lane 0, port 0 first, and keep
    ascending rank within one port; on a ring in ring order port 0 is the
    clockwise direction, so the result equals ``ring-farthest``.
    """
    relays = lane_relays(layout, rank, route_map, roles=roles)
    ports = {}
    for peer, devices in route_map.items():
        ports[int(peer)] = routes_mod.role_of(devices[0], roles).port
    order: list[int] = []
    for count in sorted(set(max(lanes) for lanes in relays.values()), reverse=True):
        queues = [sorted(peer for peer, lanes in relays.items() if max(lanes) == count and ports[peer] == port)
                  for port in (0, 1)]
        while queues[0] or queues[1]:
            for queue in queues:
                if queue:
                    order.append(queue.pop(0))
    return tuple(order)


def resolve(order: str | None, rank: int, world: int, layout: routes_mod.Layout | None = None,
            route_map: Mapping[int, Sequence[str]] | None = None, *,
            roles: Mapping[str, routes_mod.Role] | None = None) -> tuple[int, ...]:
    """The peers of ``rank`` in posting order ``order`` (``rank`` when empty).

    ``farthest`` needs the session's layout and route map; the other forms are
    those of ``protocol.post_order``. Raises ``ValueError`` for an order that
    does not name every peer of ``rank`` once.
    """
    text = (order or "rank").strip() or "rank"
    if text == "farthest":
        if layout is None or route_map is None:
            raise ValueError("the farthest posting order needs the session's layout and route map")
        peers = farthest_first(layout, rank, route_map, roles=roles)
        if sorted(peers) != [peer for peer in range(world) if peer != rank]:
            raise ValueError(f"the route map of rank {rank} does not name every peer of a group of {world}")
        return peers
    return proto.post_order(rank, world, text)


__all__ = ["POST_ORDERS", "farthest_first", "lane_relays", "resolve"]
