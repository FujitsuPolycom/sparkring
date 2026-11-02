"""Offline, conditional packet bounds and cost estimates; never imported by dispatch.

Run ``python -m sparkring_sircl.schedule_selection --fabric ring8 --placement 0-5
--co-resident 6-7``. Output is research advice, not a runtime configuration.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re

from . import routes, subset, tuning

MODEL = "sircl-placement-model/v1"
MEASUREMENTS = "sircl-placement-measurements/v1"
COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter")
QUEUE = routes.DEFAULT_HAIRPIN_QUEUE
CHUNK = 4096
LINK_CHUNK = 32768


def positions(text):
    """Ordered comma-separated positions and inclusive ascending ranges."""
    result = []
    for item in text.split(","):
        if not re.fullmatch(r"\d+(?:-\d+)?", item):
            raise ValueError(f"invalid placement {text!r}")
        ends = [int(v) for v in item.split("-")]
        if len(ends) == 2 and ends[1] < ends[0]:
            raise ValueError("placement ranges must ascend")
        result.extend(range(ends[0], ends[-1] + 1))
    if len(set(result)) != len(result):
        raise ValueError("placement positions must be distinct")
    return tuple(result)


def fabric_named(name):
    if name == "pair":
        return routes.Fabric.ring(2)
    match = re.fullmatch(r"(ring|cycle|path)-?([2-8])", name)
    if not match:
        raise ValueError("fabric must be pair, ring2..ring8, cycle2..cycle8, or path2..path8")
    kind, count = match.groups()
    return routes.Fabric.path(range(int(count))) if kind == "path" else routes.Fabric.ring(int(count))


def document_fabric(path):
    """Read setup's validated document; no host access or environment mutation.

    Setup records the installed relay plan's digest, not its forwarding rules.
    Predictions assume that plan implements SIRCL's universal shortest paths.
    """
    from runtime.common import fabric_document

    document = fabric_document.load(path)
    routes.roles_from_fabric_document(document)
    cables = []
    for cable in document["cables"]:
        a, b = cable["ends"]
        cables.append(routes.Cable(a["position"], a["port"], b["position"], b["port"]))
    return routes.Fabric(tuple(cables)), {
        "id": document["id"], "document_sha256": fabric_document.digest(document),
        "relay_plan": document["relays"], "transports": document["transports"],
    }


def rtt_us(window, relays, *, congested=False):
    """Effective SIRCL RTT inferred from bus bandwidth, not a measured latency.

    Two-relay anchors: 2*65536/9.4 GB/s and 2*393216/21.11 GB/s.
    Other hop counts use an unmeasured 4 us per relay. The 57/46.4 factor
    exposes the measured net-sircl queueing sensitivity, not a third flight-size
    anchor: those observations also changed posting policy and channel load.
    """
    low = 2 * 65536 / (9.4 * 1000)
    high = 2 * 393216 / (21.11 * 1000)
    value = max(4.0, low + (window - 65536) * (high - low) / (393216 - 65536)
                + 4.0 * (relays - 2))
    return value * (57.0 / 46.4 if congested else 1.0)


def _reverse(key):
    return (*key[:2], 1 - key[2])


def _queue_keys(lane):
    return tuple((s.dst, lane.local.secondary, 1 - s.dst_port) for s in lane.steps[:-1])


def _all_lanes(fabric, members, lanes, group):
    return {(group, r, p, lane.lane): lane
            for r, a in enumerate(members) for p, b in enumerate(members) if r != p
            for lane in routes.derive_lanes(fabric, a, b, lanes)}


def _order(fabric, members):
    """A direct physical chain, independent of rank numbering."""
    for start in members:
        for port in (0, 1):
            order, here, out = [start], start, port
            while len(order) < len(members):
                step = fabric.cross(here, out)
                if step is None or step.dst not in members or step.dst in order:
                    break
                order.append(step.dst)
                here, out = step.dst, 1 - step.dst_port
            if len(order) == len(members):
                return tuple(members.index(p) for p in order)
    return None


def _traffic(schedule, collective, size, world, order):
    traffic = {}
    if schedule in ("oneshot", "twoshot", "pieces"):
        volume = size if collective == "all_gather" or schedule == "oneshot" else size / world
        if collective == "all_reduce" and schedule != "oneshot":
            volume *= 2
        traffic = {(a, b): volume for a in range(world) for b in range(world) if a != b}
    elif schedule == "ring":
        volume = size * (world - 1) * (1 if collective == "all_gather" else 1 / world)
        if collective == "all_reduce":
            volume *= 2
        traffic = {(a, order[(i + 1) % world]): volume for i, a in enumerate(order)}
    else:
        for i, (a, b) in enumerate(zip(order, order[1:])):
            traffic[a, b] = size if collective == "all_reduce" else size * (i + 1)
            traffic[b, a] = size if collective == "all_reduce" else size * (world - i - 1)
            if collective == "reduce_scatter":
                traffic[a, b] /= world
                traffic[b, a] /= world
    return traffic


def packet_windows(lanes, *, queue_bytes=QUEUE, chunk=CHUNK):
    """Equal-share, completion-only windows with data and reverse ACK bounds.

    All supplied lanes can be live together. Each queue reserves control
    bytes for every QP through it or its reverse. A lane is constrained by
    both its data queues and the reverse queues carrying its ACKs.
    """
    if type(queue_bytes) is not int or not 4096 <= queue_bytes <= QUEUE:
        raise ValueError("queue capacity must be between 4096 and 524288 bytes")
    if type(chunk) is not int or chunk <= 0 or chunk % 16:
        raise ValueError("chunk must be a positive multiple of 16 bytes")
    queues = defaultdict(list)
    for identity, lane in lanes.items():
        for key in _queue_keys(lane):
            queues[key].append(identity)
    keys = set(queues) | {_reverse(q) for q in queues}
    limits = {}
    for key in keys:
        forward, reverse = len(queues[key]), len(queues[_reverse(key)])
        control = (forward + reverse) * subset.CONTROL_BYTES_PER_LANE
        coefficient = 3 * forward + reverse
        limits[key] = max(0, (2 * (queue_bytes - control)) // coefficient // chunk * chunk)
    windows = {}
    for identity, lane in lanes.items():
        keys_for_lane = _queue_keys(lane)
        windows[identity] = min((limits[k] for q in keys_for_lane for k in (q, _reverse(q))),
                                default=0)
        if keys_for_lane and windows[identity] < chunk:
            raise ValueError("packet envelope cannot hold one chunk on every relayed lane")
        # The native credit ring must hold all chunks plus control entries.
        windows[identity] = min(windows[identity], 60 * chunk)
    proof = []
    for key in sorted(keys):
        reverse = _reverse(key)
        forward_bytes = sum(windows[k] for k in queues[key])
        reverse_bytes = sum(windows[k] for k in queues[reverse])
        control = (len(queues[key]) + len(queues[reverse])) * subset.CONTROL_BYTES_PER_LANE
        occupied = (3 * forward_bytes + reverse_bytes + 1) // 2 + control
        if occupied > queue_bytes:
            raise ValueError("packet envelope exceeds queue capacity")
        proof.append({"queue": list(key), "data_bytes": forward_bytes,
                      "reverse_data_bytes": reverse_bytes, "control_bytes": control,
                      "bound_bytes": occupied, "capacity_bytes": queue_bytes})
    return windows, proof


def _resources(lane):
    # Functions share a cable and a NIC; do not give each function 24 GB/s.
    return ([('cable', s.src, s.src_port, s.dst, s.dst_port) for s in lane.steps]
            + [('send', lane.steps[0].src), ('receive', lane.steps[-1].dst)])


def _estimate(traffic, active, windows, co_lanes, schedule, collective, size, world, congested=False):
    load = defaultdict(float)
    users = defaultdict(set)
    for identity, lane in co_lanes.items():
        for resource in _resources(lane):
            users[resource].add(identity[0])
    lane_count = max(k[3] for k in active) + 1
    window_time = 0.0
    for identity, lane in active.items():
        volume = traffic[identity[1:3]] / lane_count
        for resource in _resources(lane):
            load[resource] += volume
        if lane.relays:
            window_time = max(window_time, volume / windows[identity] *
                              rtt_us(windows[identity], len(lane.relays), congested=congested))
    resource_time = max((volume * (1 + len(users[resource])) /
                         (26800 if resource[0] == 'receive' else 24000)
                         for resource, volume in load.items()), default=0)
    stages = {"oneshot": 1, "twoshot": 2, "pieces": 2,
              "chain": world - 1, "ring": world - 1}[schedule]
    if collective == "all_reduce" and schedule in ("chain", "ring"):
        stages *= 2
    # A deliberately simple GPU/launch term. No claim of calibrated small-message latency.
    startup = 5.0 + stages * 1.5
    if schedule == "pieces":
        startup += max(0, math.ceil(size / (2 << 20)) - 1) * 1.5
    return round(startup + max(resource_time, window_time), 3)


def plan(fabric, placement, *, co_resident=(), sizes=(4096, 65536, 4 << 20, 32 << 20),
         collective="all_reduce", mode="graph", flat=True, fabric_record=None, measurements=None,
         conditions=None):
    """Return deterministic per-size choices, rejected candidates and queue certificates.

    The modeled sessions use only the candidate's data paths, with drained
    transitions between schedules. Co-resident groups reserve all-pairs lanes
    at two lanes per peer. These requirements are not installed by this tool.
    """
    placement = tuple(placement)
    co_resident = tuple(sorted(tuple(p) for p in co_resident))
    if collective not in COLLECTIVES or mode not in tuning.MODES:
        raise ValueError("unsupported collective or execution mode")
    if not 2 <= len(fabric.positions) <= 8 or not 2 <= len(placement) <= 8:
        raise ValueError("fabric and group need 2 to 8 Sparks")
    # Derivation also rejects disconnected cable descriptions.
    routes.derive_routes(routes.Layout(fabric, fabric.positions))
    groups = (placement, *co_resident)
    claimed = set()
    for group in groups:
        if not group:
            raise ValueError("co-resident placement must not be empty")
        routes.Layout(fabric, group)
        if claimed.intersection(group):
            raise ValueError("deployments must occupy disjoint positions")
        claimed.update(group)
    sizes = tuple(sorted(set(sizes)))
    if not sizes or any(type(size) is not int or size <= 0 or size % 16 for size in sizes):
        raise ValueError("sizes must be positive multiples of 16 bytes")
    key = {"model": MODEL, "fabric": [c.text for c in sorted(fabric.cables, key=lambda c: c.text)],
           "fabric_record": fabric_record, "placement": list(placement),
           "co_resident": [list(p) for p in co_resident], "collective": collective,
           "mode": mode, "flat": flat, "conditions": conditions, "native": tuning.native_hash(),
           "kernels": tuning.kernels_hash(), "sircl": tuning.sircl_version()}
    co_lanes = {}
    for index, group in enumerate(co_resident, 1):
        co_lanes.update(_all_lanes(fabric, group, 2, index))
    order = _order(fabric, placement)
    result = {"schema": MODEL, "advisory_only": True, "runtime_configuration": False, "key": key,
              "requirements": ["universal shortest-path relay table",
                               "all listed sessions use the returned joint reservations",
                               "completion-only credit release; MTU >= 256; overhead <= 128 B/packet",
                               "no retained retransmission copies; no unlisted relayed clients",
                               "drain data and ACKs before changing schedules or windows"], "sizes": []}
    measured_rows = {}
    if measurements is not None:
        required = ("image", "drivers", "dtype", "shape", "rotate_buffers", "co_workload")
        if not isinstance(conditions, dict) or any(not conditions.get(k) for k in required):
            raise ValueError("measured selection requires image, drivers, dtype, shape, rotate_buffers and co_workload conditions")
        if type(conditions["rotate_buffers"]) is not int or conditions["rotate_buffers"] < 1:
            raise ValueError("rotate_buffers must be a positive integer")
        if not isinstance(measurements, dict) or measurements.get("schema") != MEASUREMENTS or measurements.get("key") != key:
            raise ValueError("measurement schema or placement/build/conditions key does not match")
        rows = measurements.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError("measurement rows must be a list")
        for row in rows:
            if (not isinstance(row, dict) or type(row.get("bytes")) is not int
                    or row["bytes"] <= 0 or row["bytes"] % 16 or not isinstance(row.get("candidate"), str)):
                raise ValueError("measurement case needs aligned positive bytes and a candidate ID")
            value = row.get("period_us")
            if (row.get("exact") is not True or row.get("drops") != 0 or row.get("counters_valid") is not True
                    or row.get("isolation_passed") is not True or not row.get("evidence")
                    or type(value) not in (int, float) or not math.isfinite(value) or value <= 0):
                raise ValueError("measurement needs positive time, evidence, exactness, clean valid counters and isolation")
            identity = (row.get("bytes"), row.get("candidate"))
            if identity in measured_rows:
                raise ValueError("duplicate measurement case")
            measured_rows[identity] = value
    for size in sizes:
        candidates, rejected = [], []
        for schedule in ("oneshot", "twoshot", "pieces", "chain", "ring"):
            for count in (1, 2):
                name = f"{schedule}/lanes={count}"
                try:
                    if schedule in ("oneshot", "twoshot") and collective != "all_reduce":
                        raise ValueError("one-shot and two-shot candidates model all-reduce only")
                    if schedule == "oneshot" and size > 131072 or schedule == "twoshot" and size > 2 << 20:
                        raise ValueError("exceeds modeled single-op capacity")
                    if schedule in ("chain", "ring") and order is None:
                        raise ValueError("no cable-connected chain through the placement")
                    if schedule == "chain" and collective == "all_gather" and not flat:
                        raise ValueError("chain gather requires flat input")
                    traffic = _traffic(schedule, collective, size, len(placement), order)
                    all_lanes = _all_lanes(fabric, placement, count, 0)
                    active = {k: v for k, v in all_lanes.items() if k[1:3] in traffic}
                    if any(len(v.relays) > routes.DEFAULT_MAX_RELAYS for v in (*active.values(), *co_lanes.values())):
                        raise ValueError("route exceeds the supported relay-hop limit")
                    if fabric_record is not None and any(v.relays for v in (*active.values(), *co_lanes.values())):
                        if not fabric_record.get("relay_plan") or "sircl" not in fabric_record.get("transports", ()):
                            raise ValueError("fabric document has no SIRCL relay plan")
                    if schedule == "ring":
                        queue_count = defaultdict(int)
                        for lane in active.values():
                            for q in _queue_keys(lane):
                                queue_count[q] += 1
                        if max(queue_count.values(), default=0) > 1:
                            raise ValueError("more than one ring lane uses a relay queue")
                    windows, proof = packet_windows({**active, **co_lanes},
                                                   chunk=LINK_CHUNK if schedule == "ring" else CHUNK)
                    predicted = _estimate(traffic, active, windows, co_lanes, schedule,
                                          collective, size, len(placement))
                    slow = _estimate(traffic, active, windows, co_lanes, schedule,
                                     collective, size, len(placement), congested=True)
                    paths = [{"group": k[0], "rank": k[1], "peer": k[2], "lane": k[3],
                              "path": [v.steps[0].src, *(s.dst for s in v.steps)],
                              "function": "secondary" if v.local.secondary else "primary",
                              "window_bytes": windows[k]} for k, v in sorted({**active, **co_lanes}.items())]
                    candidate_id = name + "/" + hashlib.sha256(tuning.canonical_bytes(paths)).hexdigest()[:16]
                    candidates.append({"id": candidate_id, "schedule": schedule, "lanes": count,
                                       "predicted_us": predicted, "congested_us": slow,
                                       "paths": paths, "queues": proof,
                                       "measured_us": measured_rows.get((size, candidate_id))})
                except ValueError as error:
                    rejected.append({"candidate": name, "reason": str(error)})
        measured = [c for c in candidates if c["measured_us"] is not None]
        chosen = min(measured or candidates, key=lambda c: (c["measured_us"] if measured else c["predicted_us"],
                                                           c["lanes"], c["id"]), default=None)
        result["sizes"].append({"bytes": size, "chosen": chosen, "source": "measured" if measured else "model",
                                "candidates": candidates, "rejected": rejected})
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fabric")
    source.add_argument("--fabric-document", type=Path)
    parser.add_argument("--placement", required=True)
    parser.add_argument("--co-resident", action="append", default=[])
    parser.add_argument("--sizes", default="4096,65536,4194304,33554432")
    parser.add_argument("--collective", choices=COLLECTIVES, default="all_reduce")
    parser.add_argument("--mode", choices=tuning.MODES, default="graph")
    parser.add_argument("--non-flat", action="store_true")
    parser.add_argument("--measurements", type=Path)
    parser.add_argument("--conditions", type=Path, help="JSON measurement conditions; required with --measurements")
    args = parser.parse_args(argv)
    try:
        fabric, record = document_fabric(args.fabric_document) if args.fabric_document else (fabric_named(args.fabric), None)
        measured = json.loads(args.measurements.read_text(encoding="utf-8")) if args.measurements else None
        conditions = json.loads(args.conditions.read_text(encoding="utf-8")) if args.conditions else None
        result = plan(fabric, positions(args.placement), co_resident=[positions(p) for p in args.co_resident],
                      sizes=[int(v) for v in args.sizes.split(",")], collective=args.collective, mode=args.mode,
                      flat=not args.non_flat, fabric_record=record, measurements=measured, conditions=conditions)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0 if all(row["chosen"] is not None for row in result["sizes"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
