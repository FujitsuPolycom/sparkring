"""Setup agreement of a SIRCL ring session (torch-free).

Every rank of a session contributes ``(error, connection record, setup
record)`` through one all-gather over the session's CPU process group.
:func:`agreement_failures` turns the gathered tuples into the list of reasons
construction must fail on every rank; an empty list means every rank may
connect. Rank-local fields (devices, GID indices, lane counts, route map) may
differ between ranks; every other field must equal rank 0's.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Optional

from . import routes as routes_mod

RANK_LOCAL_FIELDS = ("devices", "gid_indices", "lane_counts", "route_map")


def agreement_failures(statuses: Sequence[tuple[Optional[str], bytes, Mapping[str, Any]]],
                       layout: routes_mod.Layout | None) -> list[str]:
    """Every reason construction must fail, from all ranks' (error, record, settings).

    Shared settings must equal rank 0's (the message names the rank and every
    differing field with both values); every rank's lane count toward every
    other rank must be the agreed ``L`` and 0 toward itself; with a layout,
    every lane must pair with its peer's lane: the peer lists, for the same
    lane, the function of the same class on the port where the lane arrives.
    """
    failures = [f"rank {index}: {status[0]}" for index, status in enumerate(statuses) if status[0] is not None]
    if failures:
        return failures
    reference = statuses[0][2]
    for index, status in enumerate(statuses):
        differing = {key: (reference[key], status[2].get(key)) for key in reference
                     if key not in RANK_LOCAL_FIELDS and status[2].get(key) != reference[key]}
        if differing:
            failures.append(f"rank {index} configuration differs from rank 0: {differing}")
    lanes = reference.get("lane_count")
    for index, status in enumerate(statuses):
        counts = list(status[2].get("lane_counts", ()))
        expected = [0 if peer == index else lanes for peer in range(len(statuses))]
        if counts != expected:
            failures.append(f"rank {index} has lane counts {counts}; every peer needs {lanes} and itself 0")
    if not failures:
        maps = [{int(peer): tuple(devices) for peer, devices in status[2]["route_map"].items()}
                for status in statuses]
        if layout is not None:
            failures.extend(routes_mod.check_complementary(layout, maps))
        else:
            for rank, routes in enumerate(maps):
                for peer, devices in routes.items():
                    for lane, device in enumerate(devices):
                        mine = routes_mod.role_of(device)
                        theirs = routes_mod.role_of(maps[peer][rank][lane])
                        if mine is not None and theirs is not None and mine.secondary != theirs.secondary:
                            failures.append(f"lane {lane} of rank {rank} toward rank {peer} uses a "
                                            f"{'secondary' if mine.secondary else 'primary'} function and "
                                            f"rank {peer}'s lane {lane} does not")
    return failures
