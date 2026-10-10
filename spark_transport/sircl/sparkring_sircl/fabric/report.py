"""Text reports of the installer: installed state per Spark, and comparisons with a plan.

``show`` labels every relay object with its owner: the group named in the
Spark's record for objects that carry the installer's marks (protocol 82,
reserved filter preferences, marker processes), ``no record`` when they carry
the marks but the Spark has no record, and ``unowned`` for relay routes and
neighbours without protocol 82. Addresses and MACs are resolved to the Spark
and network device that hold them.
"""

from __future__ import annotations

from collections.abc import Mapping

from .commands import MarkerConfig
from .diff import SparkDiff
from .ops import Outcome
from .plan import FABRIC_NETDEVS, relays_left, role_of_netdev
from .state import HostState


def _lookups(states: Mapping[int, HostState]) -> tuple[dict[str, str], dict[str, str]]:
    addresses, macs = {}, {}
    for state in states.values():
        for name, link in state.links.items():
            if name not in FABRIC_NETDEVS:
                continue
            if link.mac:
                macs[link.mac] = f"{state.name} {name}"
            for address, _ in link.addresses:
                addresses[address] = f"{state.name} {name}"
    return addresses, macs


def render_show(states: Mapping[int, HostState], marker: MarkerConfig) -> str:
    addresses, macs = _lookups(states)
    lines = []
    for position, state in sorted(states.items()):
        if not state.reachable:
            lines.append(f"{state.name} (position {position}, {state.ssh}): unreachable: {state.error}")
            continue
        record = state.record
        if record:
            owner = f"group {record.get('group')}"
            heading = (f"group {record.get('group')} (layout {record.get('layout')}, relay egress "
                       f"{record.get('relay_egress')}, plan {record.get('plan')})")
        else:
            owner = "no record"
            heading = "no group record"
        lines.append(f"{state.name} (position {position}, {state.ssh}): {heading}")
        if state.hostname != state.name:
            lines.append(f"  host name: {state.hostname or 'unknown'} (the site lists {state.name})")
        for problem in ([state.record_problem] if state.record_problem else []) + state.parse_problems:
            lines.append(f"  problem: {problem}")
        ports = []
        for netdev in FABRIC_NETDEVS:
            link = state.links.get(netdev)
            if link is None:
                ports.append(f"{netdev} missing")
            else:
                text = ",".join(f"{a}/{p}" for a, p in link.addresses) or "no IPv4"
                ports.append(f"{netdev} {text} {link.mac} {link.operstate}")
        lines.append("  fabric: " + "; ".join(ports))
        binary = f"sha256 {state.marker_binary[:16]}" if state.marker_binary else "missing"
        lines.append(f"  marker executable {marker.path}: {binary}")
        tags = {}
        for process in state.markers:
            for destination, value in process.rules:
                tags[(process.device, destination)] = value
        relay_routes = state.relay_routes()
        neighbours = {(n.destination, n.dev): n for n in state.relay_neighbours()}
        lines.append(f"  routes: {len(relay_routes)} relay routes "
                     f"({sum(1 for r in relay_routes if not r.owned)} unowned)")
        for route in sorted(relay_routes, key=lambda r: (FABRIC_NETDEVS.index(r.dev) if r.dev in FABRIC_NETDEVS
                                                         else 9, r.destination)):
            neighbour = neighbours.pop((route.destination, route.dev), None)
            device = role_of_netdev(route.dev).device if route.dev in FABRIC_NETDEVS else None
            value = tags.get((device, route.destination))
            tag_text = f"tag 0x{value:04x}" if value is not None else "no tag"
            if neighbour is None:
                next_text = "no permanent neighbour"
            else:
                next_text = (f"neighbour {neighbour.mac} ({macs.get(neighbour.mac, 'unknown')})"
                             + ("" if neighbour.owned == route.owned else
                                f" [{owner if neighbour.owned else 'unowned'}]"))
            label = owner if route.owned else "unowned"
            lines.append(f"    {route.destination}/32 dev {route.dev} src {route.source} proto {route.protocol_text}"
                         f" -> {addresses.get(route.destination, 'unknown')}; {next_text}; {tag_text} [{label}]")
        for (destination, dev), neighbour in sorted(neighbours.items()):
            label = owner if neighbour.owned else "unowned"
            lines.append(f"    neighbour without route: {neighbour.text()} [{label}]")
        lines.append(f"  markers: {len(state.markers)}")
        for process in sorted(state.markers, key=lambda p: (p.device or "", p.pid)):
            log = state.marker_logs.get(process.device or "") or {}
            installed = (f"installed {log.get('installed')}/{len(log.get('rules', []))}" if "installed" in log
                         else "no log")
            lines.append(f"    {process.text()}; {installed} [{owner}]")
        filters = [entry for netdev in FABRIC_NETDEVS for entry in state.reserved_filters(netdev)]
        lines.append(f"  relay filters: {len(filters)}")
        for entry in filters:
            k = relays_left(entry.protocol) if entry.protocol is not None else None
            k_text = f"k={k}" if k else "not a relay tag"
            target = macs.get(entry.dst_mac or "", "unknown") if entry.dst_mac else "unknown"
            lines.append(f"    {entry.text()} ({k_text}; next {target}) [{owner}]")
        qdiscs = [f"{netdev} {kind}" for netdev, kind in state.ingress_qdisc.items() if kind]
        lines.append(f"  ingress qdiscs: {', '.join(qdiscs) or 'none'}"
                     + (f" (added by the installer: {', '.join(record.get('qdiscs', []))})"
                        if record and record.get("qdiscs") else ""))
        for entry in state.foreign_tag_filters():
            lines.append(f"  note: filter outside the reserved preferences matches a relay tag: {entry.text()}")
    return "\n".join(lines)


def _spark_lines(diff: SparkDiff, *, verbose: bool, scripts: bool) -> list[str]:
    counts = {action: diff.count(action) for action in ("add", "remove", "replace", "mark", "keep", "write")}
    pending = [f"{counts[action]} {action}" for action in ("add", "remove", "replace", "mark") if counts[action]]
    if counts["write"]:
        pending.append("record write")
    lines = [f"  {diff.name} (position {diff.position}): {', '.join(pending) or 'no changes'}; "
             f"{counts['keep']} unchanged"]
    for change in diff.changes:
        # Unchanged objects and in-place ownership marks are counted above; --verbose lists them.
        if change.action not in ("keep", "mark") or verbose:
            lines.append(f"    {change.text()}")
    for blocker in diff.blockers:
        lines.append(f"    BLOCKER: {blocker}")
    if diff.needs_adopt:
        lines.append(f"    NEEDS --adopt: {diff.needs_adopt}")
    for note in diff.notes:
        lines.append(f"    note: {note}")
    if scripts and diff.script:
        lines.append("    script:")
        lines.extend(f"      {line}" for line in diff.commands)
    return lines


def render_outcome(outcome: Outcome, *, verbose: bool = False, scripts: bool = False) -> str:
    layout = outcome.layout
    selected = ", ".join(group.label for group in outcome.groups)
    lines = [f"layout {layout.name} ({len(layout.groups)} group(s), relay egress {layout.relay_egress}); "
             f"{outcome.mode} for {selected}"]
    for problem in outcome.problems:
        lines.append(f"PROBLEM: {problem}")
    for group in outcome.groups:
        group_plan = outcome.plans.get(group.label)
        if group_plan is not None:
            bound = (f"per-peer op sizes at or below {group_plan.per_peer_bytes} bytes (busiest relay queue "
                     f"{group_plan.busiest_queue} SIRCL lanes, load factor {group_plan.load_factor:g})"
                     if group_plan.per_peer_bytes else "no relayed lanes")
            lines.append(f"group {group.label}: plan {group_plan.digest}; {bound}")
        else:
            lines.append(f"group {group.label}")
        for diff in outcome.diffs:
            if diff.group == group.label:
                lines.extend(_spark_lines(diff, verbose=verbose, scripts=scripts))
    return "\n".join(lines)


def summary_line(outcome: Outcome) -> str:
    diffs = outcome.diffs
    host = [change for d in diffs for change in d.host_changes]
    marks = sum(d.count("mark") for d in diffs)
    records = sum(1 for d in diffs for change in d.changes if change.kind == "record")
    sparks = len({d.position for d in diffs if d.host_changes})
    blockers = sum(len(d.blockers) for d in diffs) + len(outcome.problems)
    adopt = sum(1 for d in diffs if d.needs_adopt)
    if host:
        counts = {action: sum(1 for change in host if change.action == action)
                  for action in ("add", "remove", "replace")}
        text = (f"{len(host)} host change(s) on {sparks} Spark(s): {counts['add']} to add, {counts['remove']} to "
                f"remove, {counts['replace']} to replace")
    else:
        text = f"no host changes on {len(diffs)} Spark(s)"
    if marks:
        text += f"; {marks} unowned object(s) equal to the plan to mark in place (--adopt)"
    if records:
        text += f"; {records} record change(s)"
    if adopt:
        text += f"; --apply needs --adopt on {adopt} Spark(s)"
    if blockers:
        text += f"; {blockers} blocker(s)"
    return f"{outcome.layout.name} {outcome.mode}: {text}"


def render_marker_status(states: Mapping[int, HostState], marker: MarkerConfig) -> str:
    lines = []
    for position, state in sorted(states.items()):
        if not state.reachable:
            lines.append(f"{state.name}: unreachable: {state.error}")
            continue
        status = f"sha256 {state.marker_binary}" if state.marker_binary else "missing"
        running = ", ".join(f"{p.device} pid {p.pid}" for p in state.markers) or "no marker running"
        lines.append(f"{state.name}: {marker.path} {status}; {running}")
    return "\n".join(lines)
