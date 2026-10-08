"""Record, show and verify a SparkRing cluster's fabric: ``sparkring fabric show|verify`` and setup's last step.

Setup builds the fabric document (``runtime/common/fabric_document.py``,
``sparkring-fabric/v1``) from its plan (``prepare``), renders the relay plan
(``runtime/host/relays.py``) from it and gives each Spark its part in its
fabric record. After the network is persisted and the cables are measured,
``finish`` checks every Spark (``check_local`` through ``sparkring node
fabric-check``), records the result in the document, writes the document and
the relay plan on Node A and the same document bytes on every Spark
(``install_document`` through ``sparkring node fabric-document``), and
confirms each copy.

``sudo sparkring fabric verify`` repeats that check at any time, for example
after a reboot, and writes ``fabric-verify-<time>.json``
(``sparkring-fabric-verify/v1``) in Node A's controller directory. Per Spark
it checks the fabric links, addresses, MTU and RoCE GID index 3, the approved
routes, forwarding settings and rules, the ConnectX hairpin setting where the
Spark relays, the relay table's routes, neighbors, filters and markers, that
the boot units are enabled, that the Spark's document copy and relay plan are
the recorded ones, and that every other Spark's fabric address answers ICMP
(the reachability matrix; relayed addresses then cross the kernel's
forwarding). ``--traffic light`` adds one bidirectional RDMA write test per
relayed lane, which crosses the hardware relays.
"""
import argparse
import concurrent.futures
import datetime
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from runtime.common import fabric_document, fabric_layout
from runtime.host import relays, topology
from scripts import hairpin_setting

STATE = Path("/var/lib/sparkring/controller")
VERIFY_SCHEMA = "sparkring-fabric-verify/v1"
CHECK_SCHEMA = "sparkring-fabric-check/v1"
REPORTS = "fabric-reports"
SPARKRING = "/usr/bin/sparkring"
FABRIC_UNIT = "sparkring-fabric.service"
HAIRPIN_UNIT = "sparkring-hairpin.service"
PING_SECONDS = 2
FAILURES = (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError)


# The document.

def transports(layout, relays_installed):
    """What the fabric can carry: SIRCL wherever relays are not needed or are installed; ``prepared`` on a pair or a four-Spark cycle."""
    found = []
    if not fabric_layout.relayed(layout) or relays_installed:
        found.append("sircl")
    if layout in (fabric_layout.layout(fabric_layout.PAIR, 2), fabric_layout.layout(fabric_layout.CYCLE, 4)):
        found.append("prepared")
    return found


def _discovered(record):
    """``{role: {"netdev", "rdma_device", "mac"}}`` of a Spark's inspection, or the DGX OS names without MACs."""
    try:
        return topology.endpoints(record)
    except (KeyError, TypeError, ValueError):
        return {role: {"netdev": fabric_layout.NETDEVS[role], "rdma_device": device, "mac": None}
                for role, device in fabric_layout.DEVICES.items()}


def _functions(host, port, discovered):
    """Both functions of ``port``: the plan's data interface where it is cabled, else the discovered names."""
    rows = {}
    for function in fabric_layout.FUNCTIONS:
        role = fabric_layout.port_role(port, function)
        data = next((p for p in host["data_interfaces"] if p["role"] == role), None)
        if data is None:
            found = discovered[role]
            rows[function] = {"role": role, "netdev": found["netdev"], "rdma": found["rdma_device"],
                              "mac": str(found["mac"]).lower() if found.get("mac") else None, "address": None}
        else:
            rows[function] = {"role": role, "netdev": data["netdev"], "rdma": data["rdma_device"],
                              "mac": str(data["mac"]).lower(), "address": data["address"]}
    return rows


def planned_addresses(plan, layout):
    """Whether every fabric address of the plan is the one the layout's rule gives in its supernet."""
    try:
        return all(port["address"] == fabric_layout.address(plan["fabric_cidr"], layout, rank, port["role"])
                   for rank, host in enumerate(plan["spec"]["hosts"]) for port in host["data_interfaces"])
    except ValueError:
        return False


def prepare(plan, *, cluster, api_address=None, marker=None):
    """``(document, relay plan or None)`` of a setup plan, before setup applies it.

    The relay table is planned when the layout relays and the package
    carries the relay marker (``marker``, ``relays.marker_artifact``); a
    fabric without relays (a pair, a three-Spark cycle) gets an empty table,
    which needs no marker. ``health`` is ``unmeasured`` and ``verified`` is
    None until ``finish``.
    """
    layout = topology.layout_of(plan)
    hosts, nodes = plan["spec"]["hosts"], plan.get("nodes") or []
    positions = []
    for rank, host in enumerate(hosts):
        record = nodes[rank] if rank < len(nodes) and isinstance(nodes[rank], dict) else {}
        discovered = _discovered(record)
        cabled = {fabric_layout.role_port(role) for role in fabric_layout.roles(layout, rank)}
        ports = {}
        for port in (0, 1):
            far = fabric_layout.peer(layout, rank, port) if port in cabled else None
            ports[str(port)] = {"cable": fabric_layout.cable_of(layout, rank, port) if far else None,
                                "peer": {"position": far[0], "port": far[1]} if far else None,
                                "functions": _functions(host, port, discovered)}
        row = {"position": rank, "node_id": host["node_id"], "hostname": record.get("hostname") or host["host"],
               "management": {"address": host["management_address"], "interface": host["management_netdev"]},
               "ports": ports}
        if rank == 0 and api_address:
            row["lan_address"] = api_address
        positions.append(row)
    cables = []
    for number, first, second in fabric_layout.cables(layout):
        cables.append({"cable": number, "ends": [{"position": first[0], "port": first[1]},
                                                 {"position": second[0], "port": second[1]}],
                       "subnets": {function: str(ipaddress.IPv4Interface(
                                       positions[first[0]]["ports"][str(first[1])]["functions"][function]["address"]
                                   ).network) for function in fabric_layout.FUNCTIONS},
                       "seen_from": [positions[first[0]]["hostname"], positions[second[0]]["hostname"]],
                       "health": {"state": "unmeasured"}})
    document = {"schema": fabric_document.SCHEMA, "cluster": cluster, "shape": layout["shape"],
                "size": layout["size"], "head": 0, "fabric_cidr": plan["fabric_cidr"],
                "addressing": "planned" if planned_addresses(plan, layout) else "preserved",
                "positions": positions, "cables": cables,
                "hairpin": {"required": fabric_layout.relayed(layout),
                            "queue_size": hairpin_setting.HAIRPIN_QUEUE_SIZE,
                            "num_queues": hairpin_setting.HAIRPIN_NUM_QUEUES,
                            "positions": [rank for rank in range(layout["size"])
                                          if fabric_layout.relayed(layout) and fabric_layout.forwards(layout, rank)]},
                "relays": None, "verified": None}
    document["id"] = fabric_document.identity(document)
    plan_value = None
    if marker is not None or not fabric_layout.relayed(layout):
        plan_value = relays.render(document, marker=marker)
        document["relays"] = {"schema": relays.SCHEMA, "table": relays.TABLE, "egress": relays.EGRESS,
                              "max_relays": plan_value["max_relays"], "plan_sha256": plan_value["sha256"],
                              "persistent": True}
    document["transports"] = transports(layout, plan_value is not None)
    fabric_document.validate(document)
    return document, plan_value


def relay_lines(document, plan_value):
    """The plan's ``Relays`` section."""
    layout = fabric_document.layout(document)
    if not fabric_layout.relayed(layout):
        return ["Relays: none; every Spark is a cable neighbor of every other."]
    if plan_value is None:
        lines = ["Relays: not installed: this SparkRing package was built without the relay marker "
                 "(sparkring-relay-marker)."]
        if "prepared" in document["transports"]:
            lines.append("  Four-Spark models then use a per-deployment mesh service for their two-hop paths.")
        return lines
    routes = sum(len(row["routes"]) for row in plan_value["positions"])
    filters = sum(len(row["filters"]) for row in plan_value["positions"])
    markers = sum(len(row["markers"]) for row in plan_value["positions"])
    lines = [f"Relays: one table for every Spark, at most {plan_value['max_relays']} "
             f"relay{'s' if plan_value['max_relays'] != 1 else ''} on a route; {routes} routes and neighbor entries, "
             f"{filters} ConnectX ingress rules and {markers} marker processes on {layout['size']} Sparks; "
             "restored at every boot by sparkring-fabric.service and sparkring-relay-marker.service."]
    if plan_value["max_relays"] > plan_value["qualified_max_relays"]:
        lines.append(f"  Routes with more than {plan_value['qualified_max_relays']} relays are research-only: ring "
                     "measurements qualify up to that many.")
    if plan_value["marker"]["modes"]["source_port"]:
        lines.append("  The table also tags the four-Spark prepared transport's two-hop traffic (UDP source port "
                     f"{plan_value['marker']['modes']['source_port']}), so its models need no mesh service.")
    return lines


def port_lines(document):
    """The port-to-Spark map of a fabric document."""
    names = [row["hostname"] for row in document["positions"]]
    lines = []
    for row in document["positions"]:
        parts = []
        for port in ("0", "1"):
            entry = row["ports"][port]
            if entry["peer"] is None:
                parts.append(f"port {port} free")
            else:
                far = entry["peer"]
                parts.append(f"port {port} → position {far['position']} {names[far['position']]} port {far['port']} "
                             f"(cable {entry['cable']})")
        lines.append(f"  position {row['position']} {row['hostname']}: " + "; ".join(parts))
    return lines


def plan_lines(document, plan_value):
    """What ``sparkring setup`` prints about the fabric before it asks."""
    layout = fabric_document.layout(document)
    lines = [f"Layout: {fabric_layout.name(layout)}; fabric addresses from {document['fabric_cidr']} "
             f"({fabric_layout.cable_count(layout)} cable{'s' if fabric_layout.cable_count(layout) != 1 else ''}, "
             "two /24 subnets each)", "Ports:"] + port_lines(document)
    lines += relay_lines(document, plan_value)
    lines.append("Transports this fabric can carry: " + (", ".join(document["transports"]) or "none"))
    return lines


# Each Spark's checks (sparkring node fabric-check).

def _row(kind, what, state, detail=None):
    row = {"kind": kind, "what": what, "state": state}
    if detail:
        row["detail"] = detail
    return row


def _ping(address, *, run):
    done = run(["ping", "-n", "-c", "1", "-W", str(PING_SECONDS), "-M", "do", "-s", "8972", address],
               capture_output=True, text=True, timeout=PING_SECONDS + 10)
    return done.returncode == 0


def check_local(document, *, root="/", run=subprocess.run, collect=None):
    """This Spark's part of a fabric verification: ``sparkring-fabric-check/v1`` rows.

    ``document`` is Node A's fabric document. Each row is ``{"kind",
    "what", "state", "detail"}`` with ``state`` ``ok`` or a problem
    (``missing``, ``different``, ``failed``, ``no-link``, ``disabled``,
    ``unreachable``, ``absent``). Reads only, apart from ICMP echo requests.
    """
    from runtime.host import node
    fabric_document.validate(document)
    rows = []
    config = node.read(root, "/etc/sparkring/fabric.json")
    node.validate(config)
    identity = node.read(root, "/etc/sparkring/node.json")["node_id"]
    position = fabric_document.position_of(document, identity)
    if config["rank"] != position or config["node_id"] != identity:
        rows.append(_row("record", "fabric record", "different",
                         f"this Spark's record is rank {config['rank']}, the document says position {position}"))

    def call(argv, accepted=(0,)):
        return node.call(argv, run=run, accepted=accepted)

    # Links, addresses, MTU and RoCE GID index 3.
    try:
        node.require_links(config, root=root)
        facts = node.observe(config, **({"collect": collect} if collect else {}))
        rows.append(_row("interfaces", "fabric functions", "ok"))
    except node.LinkDown as error:
        facts = None
        rows.append(_row("interfaces", "fabric links", "no-link", str(error)))
    except FAILURES as error:
        facts = None
        rows.append(_row("interfaces", "fabric functions", "failed", str(error)))
    # The ConnectX hairpin setting where this Spark relays.
    if node.relays_hairpin(config) and facts is not None:
        try:
            node.require_hairpin(node.hairpin_rows(config, facts))
            rows.append(_row("hairpin", "ConnectX hairpin setting", "ok"))
        except FAILURES as error:
            rows.append(_row("hairpin", "ConnectX hairpin setting", "failed", str(error)))
    # Approved routes and per-interface forwarding settings.
    routes = json.loads(call(["ip", "-j", "-4", "route", "show", "table", "all"]).stdout or "[]")
    addresses = node.ipv4_addresses(json.loads(call(["ip", "-j", "-4", "address", "show"]).stdout or "[]"))
    for route, state in node.route_states(config, routes, addresses, root=root):
        rows.append(_row("route", node.route_text(route), "ok" if state == "present" else state))
    for _, key, value in node.approved_settings(config):
        found = call(["sysctl", "-n", key], accepted=(0, 1, 255)).stdout.strip()
        rows.append(_row("setting", key, "ok" if found == value else "different",
                         None if found == value else f"{found or 'absent'}, approved {value}"))
    # The relay table and its markers.
    table = config.get("relays")
    expected = (document.get("relays") or {}).get("plan_sha256")
    if expected and (not table or table.get("plan_sha256") != expected):
        rows.append(_row("relays", "relay table record", "different",
                         "this Spark's fabric record holds another relay plan; run sudo sparkring setup"))
    if table:
        for item in relays.observe(table, call=call, root=root) + relays.check_markers(table, root=root):
            what = item.get("dst") or item.get("addr") or item.get("rdma") or f"{item['dev']} preference {item['pref']}"
            rows.append(_row("relay-" + item["kind"], what, "ok" if item["state"] == "present" else item["state"]))
    # Boot units.
    units = [FABRIC_UNIT] + ([relays.MARKER_UNIT] if table and table.get("markers") else [])
    units += [HAIRPIN_UNIT] if node.relays_hairpin(config) else []
    for unit in units:
        state = call(["systemctl", "is-enabled", unit], accepted=(0, 1, 4)).stdout.strip() or "absent"
        rows.append(_row("unit", unit, "ok" if state == "enabled" else "disabled", None if state == "enabled" else state))
    # The document copy.
    copy_path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    if not copy_path.exists():
        rows.append(_row("document", fabric_document.HOST_PATH, "absent"))
    else:
        digest = hashlib.sha256(copy_path.read_bytes()).hexdigest()
        rows.append(_row("document", fabric_document.HOST_PATH,
                         "ok" if digest == fabric_document.digest(document) else "different"))
    # Reachability of every other Spark's fabric addresses.
    targets = []
    for row in document["positions"]:
        if row["position"] == position:
            continue
        for entry in row["ports"].values():
            for function in entry["functions"].values():
                if function["address"] is not None:
                    targets.append((row["position"], str(ipaddress.IPv4Interface(function["address"]).ip)))
    layout = fabric_document.layout(document)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        answers = list(pool.map(lambda target: _ping(target[1], run=run), targets))
    for (peer, address), answered in zip(targets, answers, strict=True):
        relayed = fabric_layout.hops(layout, position, peer) - 1
        rows.append(_row("reach", f"position {peer} {address}", "ok" if answered else "unreachable",
                         f"{relayed} relay{'s' if relayed != 1 else ''}" if relayed else None))
    return {"schema": CHECK_SCHEMA, "position": position, "node_id": identity, "rows": rows}


def install_document(text, *, root="/"):
    """Write the fabric document bytes ``text`` as this Spark's copy, after checking that it names this Spark."""
    from runtime.host import node
    document = fabric_document.validate(json.loads(text))
    if text != fabric_document.encoded(document):
        raise ValueError("The fabric document is not in its canonical form")
    identity = node.read(root, "/etc/sparkring/node.json")["node_id"]
    position = fabric_document.position_of(document, identity)
    path = node.location(root, fabric_document.HOST_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.chmod(0o644)
    temporary.replace(path)
    return {"written": str(path), "position": position, "sha256": fabric_document.digest(document)}


# Node A.

class Access:
    """``sparkring node`` commands on the recorded Sparks as root; Node A runs its own."""

    def __init__(self, plan, *, invoke=None, run=subprocess.run):
        from runtime.host import discovery
        self.hosts = plan["spec"]["hosts"]
        self.invoke = invoke or discovery.ssh
        self.run = run

    def node(self, rank, argv, *, data=None):
        command = ["sudo", "-n", SPARKRING, "node", *argv]
        if rank == 0:
            root = hasattr(os, "geteuid") and os.geteuid() == 0
            done = self.run(command[2:] if root else command, input=data, capture_output=True, text=True, timeout=600)
            if done.returncode:
                raise RuntimeError((done.stderr or done.stdout).strip() or "sparkring node failed")
            return done.stdout
        return self.invoke(self.hosts[rank]["host"], command, data=data)


def read_document(state=STATE):
    path = Path(state) / "fabric.json"
    return fabric_document.validate(json.loads(path.read_text(encoding="utf-8")))


def _write(path, text, mode=0o644):
    path = Path(path)
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.chmod(mode)
    temporary.replace(path)


def health_from_bandwidth(document, bandwidth):
    """The document with each cable's ``health`` from a ``sparkring-fabric-bandwidth/v1`` document, when it has one."""
    if not isinstance(bandwidth, dict) or not bandwidth.get("cables"):
        return document
    measured = {}
    for cable in bandwidth["cables"]:
        ends = frozenset((end["rank"], end["port"]) for end in cable["ends"])
        measured[ends] = cable.get("verdict")
    stamp = bandwidth.get("measured_at")
    for cable in document["cables"]:
        ends = frozenset((end["position"], end["port"]) for end in cable["ends"])
        verdict = measured.get(ends)
        if verdict in fabric_document.HEALTH_STATES:
            cable["health"] = {"state": verdict, "measured": _iso(stamp) if stamp else None,
                               "report": "fabric-bandwidth.json"}
    return document


def _iso(seconds):
    return datetime.datetime.fromtimestamp(float(seconds), datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _stamp(seconds):
    return datetime.datetime.fromtimestamp(float(seconds), datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def gather(document, plan, *, access, say=print, traffic="none", lanes=None):
    """``sparkring-fabric-verify/v1`` of a fabric: every Spark's ``check_local`` rows, then any traffic tests."""
    text = fabric_document.encoded(document)
    sparks = []

    def one(rank):
        try:
            return json.loads(access.node(rank, ["fabric-check"], data=text))
        except FAILURES as error:
            return {"schema": CHECK_SCHEMA, "position": rank, "rows": [_row("access", "sparkring node fabric-check",
                                                                             "failed", str(error).strip()[-300:])]}

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(plan["spec"]["hosts"])) as pool:
        sparks = list(pool.map(one, range(len(plan["spec"]["hosts"]))))
    for spark in sparks:
        spark["hostname"] = document["positions"][spark["position"]]["hostname"]
    report = {"schema": VERIFY_SCHEMA, "fabric": document["id"], "layout": fabric_document.layout(document),
              "traffic": traffic, "sparks": sparks, "lanes": lanes or []}
    problems = [(spark, row) for spark in sparks for row in spark["rows"] if row["state"] != "ok"]
    problems += [(None, lane) for lane in report["lanes"] if lane["state"] != "ok"]
    report["result"] = "healthy" if not problems else "problems"
    report["problems"] = len(problems)
    return report


def relayed_lanes(document, plan_value):
    """One bidirectional lane per relayed route pair: ``[{"client", "server", "function", "relays"}]``.

    A lane runs from a route's source device to its destination's address;
    the reverse direction of the same test crosses the destination's route
    back, so each pair of routes is tested once.
    """
    lanes, seen = [], set()
    names = [row["hostname"] for row in document["positions"]]
    for row in plan_value["positions"]:
        position = row["position"]
        for route in row["routes"]:
            destination = route["dst"].split("/")[0]
            source = route["src"]
            key = frozenset((source, destination))
            if key in seen:
                continue
            seen.add(key)
            local = next(entry["functions"][function] for entry in document["positions"][position]["ports"].values()
                         for function in entry["functions"]
                         if entry["functions"][function]["netdev"] == route["dev"])
            remote = next((target, entry["functions"][function])
                          for target in range(document["size"])
                          for entry in document["positions"][target]["ports"].values()
                          for function in entry["functions"]
                          if str(ipaddress.IPv4Interface(entry["functions"][function]["address"]).ip) == destination)
            lanes.append({"client": {"rank": position, "hostname": names[position], "rdma_device": local["rdma"],
                                     "address": source},
                          "server": {"rank": remote[0], "hostname": names[remote[0]], "rdma_device": remote[1]["rdma"],
                                     "address": destination},
                          "function": fabric_layout.role_function(local["role"]), "relays": route["relays"]})
    return lanes


def traffic_lanes(document, plan_value, plan, *, say=print, access=None):
    """Run ``fabric_bandwidth.measure`` once per relayed lane; rows with ``state`` ``ok`` when data crossed."""
    from runtime.host import fabric_bandwidth
    access = access or fabric_bandwidth.Access(plan)
    rows = []
    for lane in relayed_lanes(document, plan_value):
        say(f"RDMA write {lane['client']['hostname']} {lane['client']['address']} ↔ {lane['server']['hostname']} "
            f"{lane['server']['address']} ({lane['relays']} relay{'s' if lane['relays'] != 1 else ''})")
        result = fabric_bandwidth.measure(access, lane)
        rows.append({"kind": "lane", "what": f"{lane['client']['address']} ↔ {lane['server']['address']}",
                     "relays": lane["relays"], "gbps": result["gbps"],
                     "state": "ok" if result["gbps"] else "failed", "detail": result["reason"]})
    return rows


def report_lines(report):
    """Terminal lines of a verification: one summary line, then each problem."""
    layout = report["layout"]
    sparks = len(report["sparks"])
    lines = []
    if report["result"] == "healthy":
        counts = {}
        for spark in report["sparks"]:
            for row in spark["rows"]:
                counts[row["kind"]] = counts.get(row["kind"], 0) + 1
        relay_rules = counts.get("relay-filter", 0)
        cables = fabric_layout.cable_count(layout)
        text = f"Fabric verified: {cables} cable{'s' if cables != 1 else ''} on {sparks} Sparks ({fabric_layout.name(layout)})"
        if relay_rules or counts.get("relay-route"):
            text += f", relays {relay_rules} rules and {counts.get('relay-route', 0)} routes"
        text += ", reboot-persistent"
        if report["lanes"]:
            text += f", {len(report['lanes'])} relayed lanes carried RDMA writes"
        lines.append(text + ".")
        return lines
    lines.append(f"Fabric verification found {report['problems']} problem{'s' if report['problems'] != 1 else ''}:")
    for spark in report["sparks"]:
        for row in spark["rows"]:
            if row["state"] != "ok":
                lines.append(f"  {spark.get('hostname')} (position {spark['position']}) {row['kind']} {row['what']}: "
                             f"{row['state']}" + (f"; {row['detail']}" if row.get("detail") else ""))
    for lane in report["lanes"]:
        if lane["state"] != "ok":
            lines.append(f"  lane {lane['what']}: {lane['state']}" + (f"; {lane['detail']}" if lane.get("detail") else ""))
    lines.append("sudo sparkring setup repairs what setup owns; sudo sparkring hairpin applies the ConnectX hairpin "
                 "setting.")
    return lines


def save_report(state, report, now=time.time):
    """Write ``fabric-verify-<UTC time>.json`` (``-2``, ``-3`` ... for more in one second); returns its name."""
    directory = Path(state) / REPORTS
    directory.mkdir(parents=True, exist_ok=True, mode=0o755)
    moment = now()
    report["at"] = _iso(moment)
    name, number = f"fabric-verify-{_stamp(moment)}.json", 1
    while (directory / name).exists():
        number += 1
        name = f"fabric-verify-{_stamp(moment)}-{number}.json"
    _write(directory / name, json.dumps(report, indent=2) + "\n")
    return name


def latest_report(state=STATE):
    directory = Path(state) / REPORTS
    names = sorted(directory.glob("fabric-verify-*.json")) if directory.is_dir() else []
    if not names:
        return None
    try:
        return json.loads(names[-1].read_text(encoding="utf-8")) | {"file": names[-1].name}
    except (OSError, ValueError):
        return None


def finish(state, cluster, document, plan_value, *, access=None, bandwidth=None, say=print, now=time.time):
    """Setup's last step: verify the fabric, then record the document on Node A and on every Spark.

    The verification runs before the document is distributed, so its
    ``document`` rows say ``absent`` or ``different`` for the copies and are
    not counted; the copies are confirmed after they are written. Returns
    the recorded document.
    """
    plan = cluster["plan"]
    access = access or Access(plan)
    document = health_from_bandwidth(document, bandwidth)
    report = gather(document, plan, access=access, say=say)
    for spark in report["sparks"]:
        spark["rows"] = [row for row in spark["rows"] if row["kind"] != "document"]
    problems = [row for spark in report["sparks"] for row in spark["rows"] if row["state"] != "ok"]
    report["result"] = "healthy" if not problems else "problems"
    report["problems"] = len(problems)
    name = save_report(state, report, now=now)
    document["verified"] = {"at": report["at"], "result": report["result"], "report": name, "after_boot": False}
    fabric_document.validate(document)
    text = fabric_document.encoded(document)
    _write(Path(state) / "fabric.json", text)
    if plan_value is not None:
        _write(relays.plan_path(state), relays.encoded(plan_value))
    for rank in range(len(plan["spec"]["hosts"])):
        written = json.loads(access.node(rank, ["fabric-document"], data=text))
        if written.get("sha256") != fabric_document.digest(document) or written.get("position") != rank:
            raise ValueError(f"Position {rank} did not confirm its copy of the fabric document")
    for line in report_lines(report):
        say(line)
    say(f"Fabric document {document['id']} recorded on {len(plan['spec']['hosts'])} Sparks.")
    return document


PREPARED_DOCUMENT = "fabric-prepared.json"
PREPARED_PLAN = "relay-plan-prepared.json"


def save_prepared(directory, document, plan_value):
    """Keep the document and relay plan that setup's persistent records were made from in its receipt directory."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _write(directory / PREPARED_DOCUMENT, fabric_document.encoded(document), 0o600)
    if plan_value is not None:
        _write(directory / PREPARED_PLAN, relays.encoded(plan_value), 0o600)


def finish_setup(state, cluster, directory, *, bandwidth=None, access=None, say=print):
    """``finish`` with the document and relay plan that ``controller.apply`` saved in ``directory``.

    A failure here leaves the network configured and the cluster recorded;
    the error says so, and running setup again records the document.
    """
    directory = Path(directory)
    document = fabric_document.validate(json.loads((directory / PREPARED_DOCUMENT).read_text(encoding="utf-8")))
    path = directory / PREPARED_PLAN
    plan_value = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    try:
        return finish(state, cluster, document, plan_value, access=access, bandwidth=bandwidth, say=say)
    except FAILURES as error:
        raise ValueError(f"Setup configured the fabric but could not record its fabric document ({error}); "
                         "sudo sparkring setup records it again") from error


def _cluster(state):
    from runtime.common import installer
    path = Path(state) / "cluster.json"
    if not path.exists():
        raise ValueError("No fabric is recorded on this Spark; sudo sparkring setup on Node A records one")
    return installer.read(path)


def summary(state, cluster):
    """What ``sparkring status`` says about the fabric: its layout, identity, relay table and last verification."""
    layout = topology.layout_of(cluster["plan"])
    value = {"layout": fabric_layout.name(layout), "document": None, "relays": None, "verified": None}
    path = Path(state) / "fabric.json"
    if not path.exists():
        value["relays"] = "mesh-service" if layout == fabric_layout.layout(fabric_layout.CYCLE, 4) else (
            "none" if not fabric_layout.relayed(layout) else "absent")
        return value
    try:
        document = read_document(state)
    except (OSError, ValueError) as error:
        return dict(value, error=str(error))
    relays_ = document.get("relays") or {}
    value.update(document=document["id"], transports=document["transports"])
    if not fabric_layout.relayed(layout):
        value["relays"] = "none"
    elif relays_.get("persistent"):
        value["relays"] = "table"
    else:
        value["relays"] = "mesh-service" if "prepared" in document["transports"] else "absent"
    latest = latest_report(state)
    value["verified"] = ({"at": latest.get("at"), "result": latest.get("result"), "report": latest["file"]}
                         if latest else document.get("verified"))
    return value


def status_line(value):
    """One ``sparkring status`` line from ``summary``."""
    if value.get("error"):
        return f"Fabric: the recorded fabric document cannot be read ({value['error']}); sudo sparkring setup records it"
    relays_ = {"table": "relay table restored at every boot", "none": "no relays needed",
               "mesh-service": "relays: mesh service (run sudo sparkring setup to move to the fabric relay table)",
               "absent": "no relay table (run sudo sparkring setup with a package that carries the relay marker)"}
    text = f"Fabric: {value['layout']}, {relays_[value['relays']]}"
    if value.get("document") is None:
        return text + "; no fabric document (sudo sparkring setup records it)"
    verified = value.get("verified") or {}
    if verified:
        text += f"; last verified {verified.get('result')} at {verified.get('at')}"
    return text + "; sudo sparkring fabric verify checks it"


def show(state=STATE, *, json_output=False):
    """``sparkring fabric show``: the recorded document, its port map, relays and last verification."""
    path = Path(state) / "fabric.json"
    if not path.exists():
        cluster = _cluster(state)
        print("This cluster has no fabric document; it was set up by a SparkRing package that did not record one. "
              f"sudo sparkring setup records it ({len(cluster['plan']['spec']['hosts'])} Sparks).")
        print("Relays: the per-deployment mesh service carries four-Spark two-hop paths until then.")
        return 1
    document = read_document(state)
    latest = latest_report(state)
    if json_output:
        print(json.dumps({"document": document, "latest_verification": latest}, indent=2))
        return 0
    layout = fabric_document.layout(document)
    plan_path = relays.plan_path(state)
    plan_value = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
    print(f"Fabric {document['id']} of cluster {document['cluster']}: {fabric_layout.name(layout)}, "
          f"fabric addresses {document['fabric_cidr']}")
    print("Ports:")
    for line in port_lines(document):
        print(line)
    print("Cables:")
    for cable in document["cables"]:
        a, b = cable["ends"]
        health = (cable.get("health") or {}).get("state", "unmeasured")
        print(f"  cable {cable['cable']}: position {a['position']} port {a['port']} ↔ position {b['position']} "
              f"port {b['port']}; {cable['subnets']['primary']}, {cable['subnets']['secondary']}; {health}")
    hairpin = document["hairpin"]
    print("ConnectX hairpin setting: " + (f"required on positions {', '.join(map(str, hairpin['positions']))}"
                                          if hairpin["required"] else "not needed"))
    for line in relay_lines(document, plan_value if document["relays"] else None):
        print(line)
    print("Transports: " + (", ".join(document["transports"]) or "none"))
    verified = document.get("verified") or {}
    print(f"Verified by setup: {verified.get('result', 'never')} at {verified.get('at', '-')}")
    if latest is not None:
        print(f"Last verification: {latest.get('result')} at {latest.get('at')} ({latest['file']})")
    return 0


def verify(state=STATE, *, traffic="none", allow_serving=False, json_output=False, access=None, now=time.time):
    """``sudo sparkring fabric verify``: check every Spark and save ``fabric-verify-<time>.json``; the exit status."""
    from runtime.host import fabric_bandwidth
    cluster = _cluster(state)
    plan = cluster["plan"]
    document = read_document(state)
    if [row["node_id"] for row in document["positions"]] != [host["node_id"] for host in plan["spec"]["hosts"]]:
        raise ValueError("The fabric document does not describe the recorded cluster; run sudo sparkring setup")
    say = (lambda line: print(line, file=sys.stderr)) if json_output else print
    lanes = []
    if traffic == "light":
        busy = fabric_bandwidth.serving(state, len(plan["spec"]["hosts"]))
        if busy and not allow_serving:
            raise ValueError("A model is serving on this fabric; --traffic light would slow it. Stop it first, or "
                             "add --while-serving")
        plan_path = relays.plan_path(state)
        if plan_path.exists():
            lanes = traffic_lanes(document, json.loads(plan_path.read_text(encoding="utf-8")), plan, say=say)
    report = gather(document, plan, access=access or Access(plan), say=say, traffic=traffic, lanes=lanes)
    report["boot_id"] = _boot_id()
    name = save_report(state, report, now=now)
    if json_output:
        print(json.dumps(report | {"file": name}, indent=2))
    else:
        for line in report_lines(report):
            print(line)
        print(f"Report: {Path(state) / REPORTS / name}")
    return 0 if report["result"] == "healthy" else 1


def _boot_id():
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    except OSError:
        return None


def main(argv=None):
    """``sparkring fabric show|verify|spread-check``."""
    parser = argparse.ArgumentParser(prog="sparkring fabric", description=(
        "Show or verify this cluster's fabric: the Sparks' positions, ports and cables, the relay table and the "
        "boot units that restore it, or time a spread of test files along its cables. Run on Node A."))
    commands = parser.add_subparsers(dest="command", required=True)
    shown = commands.add_parser("show", help="print the recorded fabric document and its port map; changes nothing")
    shown.add_argument("--json", action="store_true", help="print the document and the last verification as JSON")
    checked = commands.add_parser("verify", help="check every Spark's links, addresses, routes, relays, markers and "
                                                 "boot units, and the reachability matrix; changes nothing")
    checked.add_argument("--traffic", choices=("none", "light"), default="none",
                         help="light: also run one short bidirectional RDMA write per relayed lane "
                              "(about 10 seconds each)")
    checked.add_argument("--while-serving", action="store_true", help="with --traffic light, test while a model serves")
    checked.add_argument("--json", action="store_true", help="print the sparkring-fabric-verify/v1 report")
    spreading = commands.add_parser("spread-check", help=(
        "spread test files from Node A to every Spark along the cables, as an install spreads the image and the "
        "checkpoint; prints when each Spark finished and removes the files once all hold them"))
    spreading.add_argument("--files", type=int, default=16, help="number of test files (default 16)")
    spreading.add_argument("--size", type=int, default=1024, help="size of each test file in MiB (default 1024)")
    spreading.add_argument("--while-serving", action="store_true", help="run while a model serves")
    args = parser.parse_args(argv)
    try:
        if args.command == "show":
            return show(json_output=args.json)
        if not hasattr(os, "geteuid") or os.geteuid() != 0:
            raise ValueError(f"Run sudo sparkring fabric {args.command}: it reads root-only state on every Spark")
        if args.command == "spread-check":
            from runtime.host import spread_check
            if args.files < 1 or args.size < 1:
                raise ValueError("--files and --size must be at least 1")
            return spread_check.main(args, STATE)
        return verify(traffic=args.traffic, allow_serving=args.while_serving, json_output=args.json)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
