"""The universal relay table of a SparkRing fabric: plan, install, check and restore it.

Sparks that are not cable neighbors reach each other through the Sparks
between them, whose ConnectX cards forward the packets in hardware. One relay
table on every Spark serves every pair of positions at once, whichever
deployment sends the traffic (``table: universal``, ``egress: same``).

The objects come from SIRCL's relay plan derivation
(``spark_transport/sircl/sparkring_sircl/fabric/plan.py``,
``build_group_plan``) for one group of every Spark of the fabric, so the
relays installed here and the lanes SIRCL derives cannot disagree. A cycle is
SIRCL's whole-ring group; a path is the group of every position on a ring one
Spark longer, which owns the path's cables and no closing cable. ``render``
converts that plan into ``sparkring-relay-plan/v1``, adds the source-port
marker mode for the four-Spark ``prepared`` transport and SIRCL's route maps,
and the rest of this module installs, checks and restores the result at boot.
Per Spark:

- **Routes and neighbors.** For each Spark ``q`` two or more cables away,
  each function of position ``p`` has a /32 ``scope link`` route to ``q``'s
  near-side address (the address of the port the packet arrives on) out of
  the port that faces ``q`` on the shorter way, with ``p``'s own address as
  source and protocol 82, and a permanent neighbor entry that maps that
  address to the cable neighbor's MAC. On a cycle with an even number of
  Sparks the opposite Spark is reached both ways, at both of its addresses.
- **Sender tags.** A marker program (``sparkring-relay-marker``,
  ``spark_transport/fabric/relay_marker.c``) per RDMA device installs one
  RDMA transmit rule per relayed destination that rewrites the packet's
  EtherType from IPv4 (``0x0800``) to ``0x88b4 + k``, where ``k`` is the
  number of relays on the way. Only RDMA traffic is tagged; IP traffic to the
  same addresses takes the kernel's forwarding.
- **Relays.** Each Spark that sits between two cables has, per function and
  per ``k`` that a route through it needs, an ingress TC flower rule (``skip_sw``, preference ``10 + k``,
  handle ``k``, under a ``clsact`` queue) on each port that rewrites the
  destination MAC to the next Spark's, sets the EtherType to the tag for
  ``k - 1`` (IPv4 when ``k`` is 1) and redirects the packet out of the other
  port of the same function.

The four-Spark ``prepared`` transport sends its opposite-peer traffic, which
crosses one relay, with UDP source port 65535 on two devices per Spark. On a
four-Spark cycle the marker of those devices also rewrites every RDMA
transmit packet with that source port to ``0x88b5`` (``source_port``), which
is the tag for one relay left; the relay rule then restores IPv4 exactly as
the per-deployment mesh service's rule does. With the table installed, a
``prepared`` deployment needs no mesh service (``persistent_reference``).

Persistence: ``sparkring-fabric.service`` (``sparkring node restore``)
installs routes, neighbors, ingress queues and filters at boot after the
hairpin service; ``sparkring-relay-marker.service`` runs the markers;
``sparkring-agent`` adds back what a link change removed. The plan rendered
here is stored on Node A beside the fabric document, and each Spark's part
(``section``) is the ``relays`` field of its ``/etc/sparkring/fabric.json``.
"""
import copy
import hashlib
import ipaddress
import json
from pathlib import Path
import re
import signal
import subprocess
import time

from runtime.common import fabric_document, fabric_layout

SCHEMA = "sparkring-relay-plan/v1"
SECTION_SCHEMA = "sparkring-relay-section/v1"
TABLE = "universal"
EGRESS = "same"
ETHERTYPE_BASE = 0x88b4
IPV4 = 0x0800
ROUTE_PROTOCOL = 82
FILTER_PREFERENCE_BASE = 10
# The installed package's payload, and the prebuilt relay marker it ships
# (built from spark_transport/fabric/relay_marker.c; scripts/build_deb.py).
PACKAGE_ROOT = "/usr/lib/sparkring"
MARKER_BINARY = PACKAGE_ROOT + "/bin/sparkring-relay-marker"
MARKER_UNIT = "sparkring-relay-marker.service"
# The UDP source port of the prepared transport's opposite-peer queue pairs.
PREPARED_SOURCE_PORT = 65535
# The most relays on a route that SIRCL's ring measurements qualify; longer routes are research-only.
QUALIFIED_MAX_RELAYS = 3
PLAN_FILE = "relay-plan.json"
_MAC = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")


def tag(relays_left):
    """The EtherType that tells a relay how many relays, itself included, the packet still crosses."""
    return IPV4 if relays_left == 0 else ETHERTYPE_BASE + relays_left


def hex16(value):
    return f"0x{value:04x}"


def encoded(value):
    return json.dumps(value, sort_keys=True, indent=2) + "\n"


def sha256(value):
    """The plan digest: SHA-256 of the canonical serialization without the ``sha256`` field."""
    return hashlib.sha256(encoded({k: v for k, v in value.items() if k != "sha256"}).encode()).hexdigest()


def _ip(value):
    return str(ipaddress.IPv4Interface(value).ip)


def prepared_devices(value, position):
    """The RDMA roles that carry the four-Spark prepared transport's opposite-peer traffic from ``position``.

    The transport reaches its opposite Spark both ways around the cycle, on
    the primary function through the Spark after the lower-numbered of the
    two and on the secondary function through the Spark before it, as
    ``spark_transport/fabric/cx7_hairpin_diagonal/fabric.build_rocenante_plan``
    selects. So positions 0 and 1 send on ``cw_primary`` and
    ``ccw_secondary``, and positions 2 and 3 on ``cw_secondary`` and
    ``ccw_primary``. Other layouts carry no prepared traffic.
    """
    if value != fabric_layout.layout(fabric_layout.CYCLE, 4):
        return []
    return ["cw_primary", "ccw_secondary"] if position in (0, 1) else ["ccw_primary", "cw_secondary"]


def _sircl():
    """SIRCL's route module and relay plan modules, from the repository layout of this package."""
    from spark_transport.sircl.sparkring_sircl import routes as sircl_routes
    from spark_transport.sircl.sparkring_sircl.fabric import layouts as sircl_layouts, plan as sircl_plan
    return sircl_routes, sircl_layouts, sircl_plan


def sircl_group(value):
    """SIRCL's group of every position of the fabric: the whole ring of a cycle, or a path on a ring one longer."""
    _, sircl_layouts, _ = _sircl()
    members = tuple(range(value["size"]))
    ring = value["size"] if value["shape"] == fabric_layout.CYCLE else value["size"] + 1
    return sircl_layouts.Group(members, ring)


def sircl_facts(document):
    """SIRCL's facts (addresses and MACs of the cabled functions) from a fabric document."""
    _, _, sircl_plan = _sircl()
    sparks = []
    for row in document["positions"]:
        ports = []
        for entry in row["ports"].values():
            if entry["cable"] is None:
                continue
            for function in entry["functions"].values():
                interface = ipaddress.IPv4Interface(function["address"])
                ports.append(sircl_plan.Port(function["netdev"], function["mac"], str(interface.ip),
                                             interface.network.prefixlen))
        sparks.append(sircl_plan.SparkFacts(row["position"], row["hostname"], tuple(ports)))
    return sircl_plan.Facts(sparks)


def sircl_plan_of(document):
    """SIRCL's ``GroupPlan`` of the whole fabric with same-function relay egress, or None without relays.

    SIRCL names the functions by the DGX OS names (or those of the document
    its ``SIRCL_FABRIC_DOCUMENT`` names), the same on every Spark, so a
    fabric whose Sparks use other names has no relay plan here.
    """
    value = fabric_document.layout(document)
    if not fabric_layout.relayed(value):
        return None
    if not fabric_document.default_names(document):
        raise ValueError("The relay plan needs the DGX OS names of the fabric functions on every Spark")
    _, _, sircl_plan = _sircl()
    return sircl_plan.build_group_plan(sircl_group(value), sircl_facts(document), relay_egress=EGRESS,
                                       max_relays=fabric_layout.max_relays(value))


def _position_objects(spark):
    """Routes, neighbours, filters and markers of one ``SparkPlan`` in this module's form."""
    routes = [{"dst": route.destination + "/32", "dev": route.netdev, "src": route.source, "scope": "link",
               "peer": route.peer, "hops": route.hops, "relays": route.hops - 1,
               "relay_positions": list(route.relays), "sircl_lanes": list(route.lanes)} for route in spark.routes]
    neighbours = [{"addr": route.destination, "lladdr": route.next_mac, "dev": route.netdev}
                  for route in spark.routes]
    filters = [{"dev": rule.in_netdev, "protocol": hex16(rule.protocol), "pref": rule.pref, "handle": rule.handle,
                "flower": {"skip_sw": True},
                "actions": [{"pedit": {"eth_dst": rule.next_mac, "eth_type": hex16(rule.new_type)}},
                            {"mirred": {"redirect": rule.out_netdev}}]} for rule in spark.filters]
    markers = [{"rdma": rules.device, "rules": [{"dst": destination, "ethertype": hex16(value)}
                                                for destination, value in rules.rules]} for rules in spark.markers]
    return routes, neighbours, filters, markers


def render(document, *, marker=None):
    """The ``sparkring-relay-plan/v1`` plan of a fabric document.

    ``marker`` is ``{"binary", "sha256"}`` of the relay marker that every
    Spark runs; a layout without relays needs none, and its plan has no
    marker. The plan lists, per position, its routes, neighbors, ingress
    filters and marker rules; ``sha256`` identifies it.
    """
    value = fabric_document.layout(document)
    if fabric_layout.relayed(value) and marker is None:
        raise ValueError("A fabric that relays needs the relay marker binary")
    group = sircl_plan_of(document)
    planned = {spark.position: spark for spark in group.sparks} if group is not None else {}
    positions = []
    for position in range(value["size"]):
        spark = planned.get(position)
        routes, neighbours, filters, markers = _position_objects(spark) if spark is not None else ([], [], [], [])
        for role in prepared_devices(value, position):
            rdma = document["positions"][position]["ports"][str(fabric_layout.role_port(role))]["functions"][
                fabric_layout.role_function(role)]["rdma"]
            row = next((row for row in markers if row["rdma"] == rdma), None)
            if row is None:
                row = {"rdma": rdma, "rules": []}
                markers.append(row)
            row["source_port"] = {"port": PREPARED_SOURCE_PORT, "ethertype": hex16(tag(1))}
        positions.append({"position": position, "node_id": document["positions"][position]["node_id"],
                          "routes": routes, "neighbours": neighbours, "filters": filters, "markers": markers})
    plan = {"schema": SCHEMA, "fabric": document["id"], "table": TABLE, "egress": EGRESS,
            "max_relays": fabric_layout.max_relays(value), "qualified_max_relays": QUALIFIED_MAX_RELAYS,
            "ethertype_base": hex16(ETHERTYPE_BASE), "route_protocol": ROUTE_PROTOCOL,
            "filter_preference_base": FILTER_PREFERENCE_BASE,
            "marker": None if not fabric_layout.relayed(value) else {
                "binary": marker["binary"], "sha256": marker["sha256"],
                "modes": {"by_destination": True,
                          "source_port": PREPARED_SOURCE_PORT if prepared_devices(value, 0) else None}},
            "sircl": None if group is None else {
                "group": group.group.label, "digest": group.digest, "relay_egress": group.relay_egress,
                "route_maps": [group.sircl.route_text(rank) for rank in range(group.sircl.world)],
                "busiest_queue_lanes": group.busiest_queue, "load_factor": group.load_factor,
                "per_peer_bytes": group.per_peer_bytes},
            "positions": positions}
    plan["sha256"] = sha256(plan)
    return plan


def section(plan, position):
    """Position ``position``'s part of ``plan``: the ``relays`` field of its ``/etc/sparkring/fabric.json``."""
    row = plan["positions"][position]
    return {"schema": SECTION_SCHEMA, "plan_sha256": plan["sha256"], "fabric": plan["fabric"],
            "position": position, "route_protocol": plan["route_protocol"],
            "marker": None if plan["marker"] is None else {"binary": plan["marker"]["binary"],
                                                           "sha256": plan["marker"]["sha256"]},
            "routes": copy.deepcopy(row["routes"]), "neighbours": copy.deepcopy(row["neighbours"]),
            "filters": copy.deepcopy(row["filters"]), "markers": copy.deepcopy(row["markers"])}


def validate_section(value, config):
    """``value`` after checking a fabric record's ``relays`` field; ValueError otherwise.

    Every device and network interface must be one of the record's fabric
    functions, every route a /32 out of one of them with its own address as
    source, and every filter preference inside SparkRing's range.
    """
    if not isinstance(value, dict) or value.get("schema") != SECTION_SCHEMA:
        raise ValueError("Invalid relay table record")
    if value.get("position") != config["rank"] or value.get("route_protocol") != ROUTE_PROTOCOL:
        raise ValueError("Relay table record belongs to another position")
    if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("plan_sha256"))):
        raise ValueError("Relay table record lacks its plan digest")
    marker = value.get("marker") or {}
    if value.get("markers") and (not isinstance(marker.get("binary"), str) or not marker["binary"].startswith("/")
                                 or not re.fullmatch(r"[0-9a-f]{64}", str(marker.get("sha256")))):
        raise ValueError("Relay table record lacks its marker binary and digest")
    interfaces = {port["netdev"]: port for port in config["interfaces"]}
    devices = {port["rdma_device"] for port in config["interfaces"]}
    for route in value.get("routes") or []:
        network = ipaddress.IPv4Network(route["dst"])
        port = interfaces.get(route["dev"])
        if (network.prefixlen != 32 or port is None or route.get("scope") != "link"
                or route["src"] != _ip(port["address"])):
            raise ValueError("Relay route must be a /32 out of a fabric function with its own source address")
    for neighbour in value.get("neighbours") or []:
        if neighbour["dev"] not in interfaces or not _MAC.fullmatch(str(neighbour["lladdr"])):
            raise ValueError("Relay neighbor must name a fabric function and a MAC")
        ipaddress.IPv4Address(neighbour["addr"])
    for rule in value.get("filters") or []:
        actions = rule.get("actions") or []
        if (rule["dev"] not in interfaces or len(actions) != 2
                or actions[1].get("mirred", {}).get("redirect") not in interfaces
                or not FILTER_PREFERENCE_BASE < rule["pref"] <= FILTER_PREFERENCE_BASE + fabric_layout.MAX_SPARKS
                or rule["handle"] != rule["pref"] - FILTER_PREFERENCE_BASE
                or not _MAC.fullmatch(str(actions[0].get("pedit", {}).get("eth_dst")))):
            raise ValueError("Relay filter must stay between fabric functions within SparkRing's preferences")
    for row in value.get("markers") or []:
        if row["rdma"] not in devices:
            raise ValueError("Relay marker must run on a fabric RDMA device")
    return value


# Commands that install and observe the table on this Spark.

def route_command(route, *, protocol=ROUTE_PROTOCOL):
    return ["ip", "route", "replace", route["dst"], "dev", route["dev"], "src", route["src"], "scope", "link",
            "proto", str(protocol)]


def neighbour_command(neighbour):
    return ["ip", "neigh", "replace", neighbour["addr"], "lladdr", neighbour["lladdr"], "dev", neighbour["dev"],
            "nud", "permanent", "proto", str(ROUTE_PROTOCOL)]


def qdisc_command(netdev):
    """A ``clsact`` queue gives the ingress hook; the per-deployment mesh service accepts it and refuses ``ingress``."""
    return ["tc", "qdisc", "add", "dev", netdev, "clsact"]


def filter_command(rule, *, verb="replace"):
    pedit, mirred = rule["actions"][0]["pedit"], rule["actions"][1]["mirred"]
    return ["tc", "filter", verb, "dev", rule["dev"], "ingress", "protocol", rule["protocol"],
            "pref", str(rule["pref"]), "handle", str(rule["handle"]), "flower", "skip_sw",
            "action", "pedit", "ex", "munge", "eth", "dst", "set", pedit["eth_dst"], "pipe",
            "action", "pedit", "ex", "munge", "eth", "type", "set", pedit["eth_type"], "pipe",
            "action", "mirred", "egress", "redirect", "dev", mirred["redirect"]]


def marker_argv(value, row):
    """The marker command line of one marker row of a ``section``."""
    argv = [value["marker"]["binary"], "--device", row["rdma"]]
    for rule in row["rules"]:
        argv += ["--rule", f"{rule['dst']}={rule['ethertype']}"]
    if "source_port" in row:
        argv += ["--source-port", f"{row['source_port']['port']}={row['source_port']['ethertype']}"]
    return argv + ["--managed"]


def _json(call, argv):
    output = call(argv).stdout
    return json.loads(output or "[]")


def _route_rows(call):
    return _json(call, ["ip", "-j", "-4", "route", "show", "table", "main"])


def _route_present(rows, route):
    destination = route["dst"].split("/")[0]
    return any(row.get("dst") in (destination, route["dst"]) and row.get("dev") == route["dev"]
               and row.get("prefsrc") == route["src"] and not row.get("gateway") for row in rows)


def _route_conflict(rows, route):
    """Another route to the destination that SparkRing does not own (protocol 82)."""
    destination = route["dst"].split("/")[0]
    return [row for row in rows if row.get("dst") in (destination, route["dst"]) and not _route_present([row], route)
            and str(row.get("protocol")) not in (str(ROUTE_PROTOCOL), "sparkring")]


def _neighbour_present(rows, neighbour):
    return any(row.get("dst") == neighbour["addr"] and row.get("dev", neighbour["dev"]) == neighbour["dev"]
               and str(row.get("lladdr", "")).lower() == neighbour["lladdr"]
               and "PERMANENT" in (row.get("state") or []) for row in rows)


def _filter_prefs(call, netdev):
    return {row.get("pref") for row in _json(call, ["tc", "-j", "filter", "show", "dev", netdev, "ingress"])}


def _has_ingress(call, netdev):
    """Whether ``netdev`` has an ingress hook: a ``clsact`` queue, or an ``ingress`` queue that was there before."""
    return any(row.get("kind") in ("clsact", "ingress")
               for row in _json(call, ["tc", "-j", "qdisc", "show", "dev", netdev]))


def _link_up(netdev, root):
    try:
        return (Path(root) / "sys/class/net" / netdev / "carrier").read_text().strip() == "1"
    except OSError:
        return False


def restore(value, *, call, root="/", log=None):
    """Install the parts of a ``section`` that are missing on this Spark; return one row per object.

    A missing ingress queue is added, a filter preference that is absent is
    installed with ``tc filter replace`` (which leaves an identical filter as
    it is), and a missing route or neighbor is installed. A route to a relayed
    destination that another owner installed is reported as ``conflict`` and
    left alone. Objects on a function without carrier are ``no-link``: the
    kernel drops a function's routes with its address, and the agent adds
    them again when the link returns. Nothing is ever removed.
    """
    rows = []
    netdevs = sorted({rule["dev"] for rule in value["filters"]})
    for netdev in netdevs:
        if not (Path(root) / "sys/class/net" / netdev).exists():
            rows.append({"kind": "qdisc", "dev": netdev, "state": "absent"})
            continue
        state = "present"
        if not _has_ingress(call, netdev):
            call(qdisc_command(netdev))
            state = "restored"
        rows.append({"kind": "qdisc", "dev": netdev, "state": state})
        present = _filter_prefs(call, netdev)
        for rule in (r for r in value["filters"] if r["dev"] == netdev):
            if rule["pref"] in present:
                rows.append({"kind": "filter", "dev": netdev, "pref": rule["pref"], "state": "present"})
                continue
            call(filter_command(rule))
            rows.append({"kind": "filter", "dev": netdev, "pref": rule["pref"], "state": "restored"})
    routes = _route_rows(call)
    for route in value["routes"]:
        if _route_present(routes, route):
            state = "present"
        elif _route_conflict(routes, route):
            state = "conflict"
        elif not _link_up(route["dev"], root):
            state = "no-link"
        else:
            try:
                call(route_command(route))
                state = "restored"
            except (OSError, RuntimeError, subprocess.SubprocessError) as error:
                state = "failed"
                if log:
                    log(f"could not restore relay route {route['dst']} dev {route['dev']}: {error}")
        rows.append({"kind": "route", "dst": route["dst"], "dev": route["dev"], "state": state})
    neighbours = _json(call, ["ip", "-j", "-4", "neigh", "show", "nud", "permanent"])
    for neighbour in value["neighbours"]:
        if _neighbour_present(neighbours, neighbour):
            state = "present"
        elif not _link_up(neighbour["dev"], root):
            state = "no-link"
        else:
            call(neighbour_command(neighbour))
            state = "restored"
        rows.append({"kind": "neighbour", "addr": neighbour["addr"], "dev": neighbour["dev"], "state": state})
    return rows


def missing(rows):
    """The rows of ``restore`` or ``observe`` that are not in place."""
    return [row for row in rows if row["state"] not in ("present", "restored")]


def _filter_matches(row, rule):
    options = row.get("options") or {}
    keys = options.get("keys") or {}
    eth_type = str(keys.get("eth_type", "")).lower().removeprefix("0x")
    redirect = [action.get("to_dev") for action in options.get("actions") or [] if action.get("kind") == "mirred"]
    return (options.get("handle") == rule["handle"] and eth_type == rule["protocol"].removeprefix("0x")
            and options.get("skip_sw") is True and redirect == [rule["actions"][1]["mirred"]["redirect"]])


def observe(value, *, call, root="/"):
    """Compare a ``section`` with this Spark without changing anything; one row per object.

    A filter is ``present`` when its preference has a flower rule with the
    handle, EtherType, ``skip_sw`` and redirect of the plan, ``not-in-hardware``
    when the card did not accept it, and ``different`` or ``missing``
    otherwise.
    """
    rows = []
    for netdev in sorted({rule["dev"] for rule in value["filters"]}):
        listed = _json(call, ["tc", "-j", "filter", "show", "dev", netdev, "ingress"])
        for rule in (r for r in value["filters"] if r["dev"] == netdev):
            found = [row for row in listed if row.get("pref") == rule["pref"] and row.get("options")]
            if not found:
                state = "missing"
            elif not any(_filter_matches(row, rule) for row in found):
                state = "different"
            elif not any(row["options"].get("in_hw") for row in found if _filter_matches(row, rule)):
                state = "not-in-hardware"
            else:
                state = "present"
            rows.append({"kind": "filter", "dev": netdev, "pref": rule["pref"], "state": state})
    routes = _route_rows(call)
    for route in value["routes"]:
        state = "present" if _route_present(routes, route) else "conflict" if _route_conflict(routes, route) else (
            "missing" if _link_up(route["dev"], root) else "no-link")
        rows.append({"kind": "route", "dst": route["dst"], "dev": route["dev"], "state": state})
    neighbours = _json(call, ["ip", "-j", "-4", "neigh", "show", "nud", "permanent"])
    for neighbour in value["neighbours"]:
        state = "present" if _neighbour_present(neighbours, neighbour) else (
            "missing" if _link_up(neighbour["dev"], root) else "no-link")
        rows.append({"kind": "neighbour", "addr": neighbour["addr"], "dev": neighbour["dev"], "state": state})
    return rows


def marker_processes(*, root="/"):
    """``{rdma: argv}`` of the running relay marker processes, from ``/proc``."""
    found = {}
    for entry in (Path(root) / "proc").glob("[0-9]*"):
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        argv = [part.decode(errors="replace") for part in argv if part]
        if argv and Path(argv[0]).name == Path(MARKER_BINARY).name and "--device" in argv:
            index = argv.index("--device")
            if index + 1 < len(argv):
                found[argv[index + 1]] = argv
    return found


def check_markers(value, *, root="/"):
    """Rows for the planned markers: ``present`` when a process runs with the planned command line."""
    running = marker_processes(root=root)
    rows = []
    for row in value["markers"]:
        argv = running.get(row["rdma"])
        state = "missing" if argv is None else "present" if argv == marker_argv(value, row) else "different"
        rows.append({"kind": "marker", "rdma": row["rdma"], "state": state})
    return rows


def binary_digest(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def supervise(value, *, popen=subprocess.Popen, digest=binary_digest, poll=1.0, clock=time.monotonic):
    """Run one marker per planned RDMA device until a signal stops this process or a marker exits.

    ``sparkring-relay-marker.service`` runs this (``sparkring node
    relay-markers``); its ``Restart=on-failure`` starts every marker again
    when one exits, for example after a ConnectX driver restart removed its
    device. The binary must match the recorded digest. Returns 0 after a stop
    signal and 1 when a marker exited.
    """
    if not value["markers"]:
        return 0
    binary = value["marker"]["binary"]
    if digest(binary) != value["marker"]["sha256"]:
        raise ValueError(f"{binary} differs from the relay marker that setup recorded; reinstall the package")
    stopping = []

    def stop(signum, frame):
        stopping.append(signum)

    previous = {number: signal.signal(number, stop) for number in (signal.SIGTERM, signal.SIGINT)}
    children = []
    try:
        for row in value["markers"]:
            children.append(popen(marker_argv(value, row), stdin=subprocess.DEVNULL))
        while not stopping:
            if any(child.poll() is not None for child in children):
                return 1
            time.sleep(poll)
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        for number, handler in previous.items():
            signal.signal(number, handler)


# The prepared transport over the table.

def persistent_reference(state_dir, cluster):
    """The ``fabric`` reference a four-Spark ``prepared`` deployment uses instead of a mesh service, or None.

    It exists when Node A's fabric document (``state_dir/fabric.json``)
    describes this cluster's four-Spark cycle, lists the ``prepared``
    transport and records a relay table that boot units restore. The
    reference has the form of a mesh site reference: the path of each
    Spark's copy of the fabric document, that copy's SHA-256 and the relay
    plan's SHA-256.
    """
    path = Path(state_dir) / "fabric.json"
    try:
        text = path.read_text(encoding="utf-8")
        document = fabric_document.validate(json.loads(text))
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be used: {error}") from None
    relays = document.get("relays") or {}
    nodes = [host.get("node_id") for host in cluster["plan"]["spec"]["hosts"]]
    if (fabric_document.layout(document) != fabric_layout.layout(fabric_layout.CYCLE, 4)
            or "prepared" not in document["transports"] or not relays.get("persistent")
            or [row["node_id"] for row in document["positions"]] != nodes):
        return None
    return {"site_path": fabric_document.HOST_PATH, "site_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "plan_sha256": relays["plan_sha256"]}


def group_reference(state_dir, cluster):
    """The ``fabric`` reference of a SIRCL group whose ranks reach each other through relays.

    It has the form of ``persistent_reference`` (the path of each Spark's
    copy of the fabric document, that copy's SHA-256 and the relay plan's
    SHA-256) for any layout whose fabric document, Node A's
    ``state_dir/fabric.json``, describes this cluster, lists the ``sircl``
    transport and records a relay table that boot units restore. Each rank's
    relay check (``check_position``) compares its own copies with it. Raises
    ValueError when the document is missing, unreadable, describes other
    Sparks or records no such table.
    """
    path = Path(state_dir) / "fabric.json"
    try:
        text = path.read_text(encoding="utf-8")
        document = fabric_document.validate(json.loads(text))
    except FileNotFoundError:
        raise ValueError("This cluster has no fabric document; sudo sparkring setup records one") from None
    except (OSError, ValueError) as error:
        raise ValueError(f"{path} cannot be used: {error}") from None
    relays = document.get("relays") or {}
    nodes = [host.get("node_id") for host in cluster["plan"]["spec"]["hosts"]]
    if [row["node_id"] for row in document["positions"]] != nodes:
        raise ValueError("The recorded fabric document describes other Sparks than the cluster; run sudo sparkring "
                         "setup")
    if "sircl" not in document["transports"] or not relays.get("persistent") or not relays.get("plan_sha256"):
        raise ValueError("The fabric's relay table is not installed and restored at boot; sudo sparkring setup "
                         "installs it")
    return {"site_path": fabric_document.HOST_PATH, "site_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "plan_sha256": relays["plan_sha256"]}


def check_position(reference, position, gid, host_ip, *, root="/", call=None):
    """The read-only relay check of one rank of a SIRCL group at fabric ``position``.

    This Spark's copy of the fabric document must have the referenced digest,
    its fabric record the referenced relay plan and the position; RoCE GID
    index 3; ``host_ip`` this position's management address, over which the
    group's ranks bootstrap. Then every route, neighbor and filter of the
    record's relay table and every marker must be in place. Raises ValueError
    naming the first difference; returns the rows.
    """
    from runtime.host import node
    call = call or node.call
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    text = path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != reference["site_sha256"]:
        raise ValueError("This Spark's fabric document differs from the deployment's; run sudo sparkring setup")
    document = fabric_document.validate(json.loads(text))
    record = _fabric_record(root)
    value = record.get("relays") or {}
    if value.get("plan_sha256") != reference["plan_sha256"] or record.get("rank") != position:
        raise ValueError("This Spark's relay table differs from the deployment's; run sudo sparkring setup")
    if gid != 3:
        raise ValueError("The group's GID index differs from the fabric's index 3")
    if host_ip != document["positions"][position]["management"].get("address"):
        raise ValueError("Rank bootstrap address differs from the fabric document's management address")
    rows = observe(value, call=call, root=root) + check_markers(value, root=root)
    absent = missing(rows)
    if absent:
        first = absent[0]
        what = first.get("dst") or first.get("addr") or first.get("rdma") or f"{first.get('dev')} pref {first.get('pref')}"
        raise ValueError(f"Relay table incomplete on this Spark: {first['kind']} {what} is {first['state']}"
                         + ("" if len(absent) == 1 else f" ({len(absent)} objects)")
                         + "; sudo sparkring fabric verify names each")
    return rows


def position_devices(position, *, root="/"):
    """The RDMA devices of this Spark's cabled functions, in role order, as its copy of the fabric document names them."""
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    document = fabric_document.validate(json.loads(path.read_text(encoding="utf-8")))
    return [rdma for rdma, row in fabric_document.devices(document, position).items() if row["address"] is not None]


def is_reference(reference):
    """Whether a deployment's ``fabric`` reference names the fabric document rather than a mesh site."""
    return isinstance(reference, dict) and reference.get("site_path") == fabric_document.HOST_PATH


def _fabric_record(root):
    return json.loads((Path(root) / "etc/sparkring/fabric.json").read_text(encoding="utf-8"))


def check_reference(reference, rank, hcas, gid, host_ip, *, root="/", call=None):
    """The read-only ring check of a ``prepared`` four-Spark deployment over the relay table.

    The fabric document copy must have the referenced digest and this
    Spark's fabric record the referenced relay plan; ``hcas`` must list
    port 0's then port 1's primary, then the secondaries, as the mesh check
    expects; RoCE GID index 3; ``host_ip`` this position's management
    address. Then every route, neighbor and filter of the record's relay
    table and every marker must be in place. Raises ValueError naming the
    first difference; returns the rows.
    """
    from runtime.host import node
    call = call or node.call
    path = Path(root) / fabric_document.HOST_PATH.lstrip("/")
    text = path.read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != reference["site_sha256"]:
        raise ValueError("This Spark's fabric document differs from the deployment's; run sudo sparkring setup")
    document = fabric_document.validate(json.loads(text))
    record = _fabric_record(root)
    value = record.get("relays") or {}
    if value.get("plan_sha256") != reference["plan_sha256"] or record.get("rank") != rank:
        raise ValueError("This Spark's relay table differs from the deployment's; run sudo sparkring setup")
    devices = fabric_document.devices(document, rank)
    by_role = {row["role"]: rdma for rdma, row in devices.items()}
    expected = [by_role[role] for role in ("cw_primary", "ccw_primary", "cw_secondary", "ccw_secondary")]
    if hcas != expected:
        raise ValueError("TP4 HCA order differs from the fabric document's devices")
    if gid != 3:
        raise ValueError("TP4 GID index differs from the fabric's index 3")
    if host_ip != document["positions"][rank]["management"].get("address"):
        raise ValueError("Rank bootstrap address differs from the fabric document's management address")
    rows = observe(value, call=call, root=root) + check_markers(value, root=root)
    absent = missing(rows)
    if absent:
        first = absent[0]
        what = first.get("dst") or first.get("addr") or first.get("rdma") or f"{first.get('dev')} pref {first.get('pref')}"
        raise ValueError(f"Relay table incomplete on this Spark: {first['kind']} {what} is {first['state']}"
                         + ("" if len(absent) == 1 else f" ({len(absent)} objects)")
                         + "; sudo sparkring fabric verify names each")
    return rows


def plan_path(state_dir):
    return Path(state_dir) / PLAN_FILE


def marker_artifact(root):
    """``{"binary", "sha256"}`` of the relay marker the installed package carries, or None.

    The package build records the binary's digest in ``distribution.json``
    (``relay_marker``); a package built without the marker has none.
    """
    try:
        record = json.loads((Path(root) / "distribution.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = record.get("relay_marker") if isinstance(record, dict) else None
    if not isinstance(value, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256"))):
        return None
    return {"binary": MARKER_BINARY, "sha256": value["sha256"]}



def check_installed(root=PACKAGE_ROOT, *, digest=binary_digest):
    """The installed package's relay marker against the digest its ``distribution.json`` records.

    The package's ``postinst`` runs this (``sparkring node relay-marker-check``)
    before anything else, so an installation whose marker is missing or
    differs fails. Returns ``{"relay_marker": "verified", "sha256"}``, or
    ``{"relay_marker": None}`` for a package built without the marker
    (``build_deb.py --relay-marker skip``); raises ValueError otherwise.
    """
    record = json.loads((Path(root) / "distribution.json").read_text(encoding="utf-8"))
    value = record.get("relay_marker")
    if value is None:
        return {"relay_marker": None}
    if (not isinstance(value, dict) or value.get("path") != MARKER_BINARY[len(PACKAGE_ROOT) + 1:]
            or not re.fullmatch(r"[0-9a-f]{64}", str(value.get("sha256")))):
        raise ValueError("The package's distribution.json records the relay marker in an unknown form")
    path = Path(root) / value["path"]
    if not path.is_file():
        raise ValueError(f"The package's relay marker {MARKER_BINARY} is missing")
    actual = digest(path)
    if actual != value["sha256"]:
        raise ValueError(f"The package's relay marker {MARKER_BINARY} has sha256 {actual}; the package records "
                         f"{value['sha256']}. Reinstall the package.")
    return {"relay_marker": "verified", "sha256": actual}
