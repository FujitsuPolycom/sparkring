"""Order 2 to 8 authenticated Sparks from LLDP and NIC facts and plan their fabric addresses and routes.

The layout (``runtime/common/fabric_layout.py``) is a pair, a path (line) or a
cycle (ring) of up to eight Sparks; ``cabling`` diagnoses it from LLDP. Node A
is position (rank) 0 and position r+1 is the Spark on position r's port 0.

Pairs and four-Spark cycles keep the addresses, routes and plan identity of
records that name only their Spark count: their setup specification carries
no ``layout`` key, and ``layout_of`` derives the layout from the count.
"""
import copy
import hashlib
import ipaddress
import json
import re

from runtime.common import fabric_layout
from runtime.host import cabling
from runtime.host.cabling import lldp_rows  # noqa: F401  (part of this module's interface)
from scripts import deploy_network

DEVICES = dict(fabric_layout.DEVICES)


def endpoints(node):
    interfaces = {row["name"]: row for row in node["facts"]["interfaces"]}
    rdma = {row["device"]: row for row in node["facts"]["rdma"]}
    result = {}
    for role, device in DEVICES.items():
        function = rdma.get(device)
        if not function or function.get("driver") != "mlx5_core":
            raise ValueError(f"{node['hostname']}: missing verified RDMA function {device}")
        netdev = function["netdev"]
        interface = interfaces[netdev]
        if not isinstance(interface.get("mac"), str) or not re.fullmatch(r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", interface["mac"]):
            raise ValueError(f"{node['hostname']}: MAC address unavailable for {netdev}; refresh interface inventory")
        if netdev == node["facts"]["management"]["interface"] or interface.get("master"):
            raise ValueError("Fabric function is a management or bridge/bond interface")
        result[role] = {"role": role, "rdma_device": device, "netdev": netdev, "mac": interface["mac"].lower(),
                        "ipv4": interface["ipv4"], "port": 0 if role.startswith("cw_") else 1}
    return result


def layout_of(plan):
    """The fabric layout ``{"shape", "size"}`` of a setup plan or its specification.

    A plan records ``layout``; a pair's or four-Spark cycle's specification
    does not, so a record that names only two or four Sparks is a pair or a
    four-Spark cycle.
    """
    spec = plan.get("spec", plan)
    value = plan.get("layout") or spec.get("layout")
    if value is not None:
        return fabric_layout.checked(value)
    return fabric_layout.legacy(len(spec["hosts"]))


def arrange(nodes, head_id):
    """``(nodes in position order, layout)`` when the authenticated Sparks' cables form a supported layout.

    The cables come from each Spark's LLDP observations (``cabling``): a pair
    needs a cable between both ports 0; a cycle needs one loop, and a path one
    line that starts at Node A, in which every cable joins port 0 of a Spark to
    port 1 of the next; position r+1 is the Spark on position r's port 0.
    Every cable must be confirmed from both ends. Any other cabling raises
    ``cabling.CablingError``, whose message names the physical change;
    nothing is renumbered or remapped to accept it.
    """
    count = len(nodes)
    if not fabric_layout.MIN_SPARKS <= count <= fabric_layout.MAX_SPARKS or len({node["node_id"] for node in nodes}) != count:
        raise ValueError(f"Select two to {fabric_layout.WORDS[fabric_layout.MAX_SPARKS]} distinct authenticated Sparks")
    by_id = {node["node_id"]: node for node in nodes}
    if head_id not in by_id:
        raise ValueError("The controller/head must be one of the selected Sparks")
    sparks = [cabling.from_inspection(node, endpoints(node)) for node in nodes]
    result = cabling.diagnose(sparks, head_id, strict=True, whole=True)
    if not result["ready"] or result["layout"] not in cabling.SHAPE_OF or result["layout_size"] != count:
        raise cabling.CablingError(result)
    value = fabric_layout.layout(result["shape"], result["layout_size"])
    return [by_id[ident] for ident in result["order"]], value


def ordered_nodes(nodes, head_id):
    """The authenticated Sparks in position order, Node A first (``arrange`` without the layout)."""
    return arrange(nodes, head_id)[0]


def fabric_supernet(value, fabric_cidr=None):
    """The fabric supernet for ``value``: ``fabric_cidr``, or the /21 up to four cables and the /20 above.

    An explicit supernet must be an isolated /16 through /21 with room for
    two /24 subnets per cable; a /21 holds four cables.
    """
    cidr = fabric_cidr or fabric_layout.default_cidr(value)
    network = ipaddress.IPv4Network(cidr, strict=True)
    if network.prefixlen > 21 or network.prefixlen < 16:
        raise ValueError("Choose an isolated /16 through /21 fabric supernet")
    cables = fabric_layout.cable_count(value)
    if fabric_layout.capacity(cidr) < cables:
        raise ValueError(f"The fabric supernet {cidr} holds {fabric_layout.capacity(cidr)} cables, but this "
                         f"{fabric_layout.name(value)} has {cables}; use --fabric-cidr {fabric_layout.WIDE_CIDR} or "
                         "leave --fabric-cidr out")
    return str(network)


def build_spec(nodes, head_id, *, name="sparkring", fabric_cidr=None, reset=False, preserve_control=False):
    """The ``sparkring-appliance-plan/v1`` setup plan of the authenticated Sparks.

    ``fabric_cidr`` None takes ``fabric_supernet``'s default for the
    layout. The plan records ``layout`` and the supernet it used.
    """
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Choose a lowercase cluster name")
    nodes, value = arrange(nodes, head_id)
    fabric_cidr = fabric_supernet(value, fabric_cidr)
    maps = [endpoints(node) for node in nodes]
    selected = [maps[rank][role] for rank in range(len(nodes)) for role in fabric_layout.roles(value, rank)]
    existing = [port["ipv4"] for port in selected]
    if not reset and any(existing) and not all(len(addresses) == 1 for addresses in existing):
        raise ValueError("Partial fabric addressing requires inspection; no automatic renumbering")
    preserve = any(existing) and not reset
    hosts, inventory = [], {}
    controller = nodes[0]["facts"]["management"]["address"]
    for rank, node in enumerate(nodes):
        facts = copy.deepcopy(node["facts"])
        facts["rank"] = rank
        host = {"rank": rank, "host": facts["ssh_target"], "node_id": node["node_id"],
                "management_address": facts["management"]["address"], "management_netdev": facts["management"]["interface"],
                "controller_probe_address": facts["management"]["controller_address"],
                "backup_dir": f"/var/lib/sparkring/backups/{name}/rank{rank}", "data_interfaces": []}
        for role in fabric_layout.roles(value, rank):
            endpoint = maps[rank][role]
            address = endpoint["ipv4"][0] if preserve else fabric_layout.address(fabric_cidr, value, rank, role)
            if ipaddress.IPv4Interface(address).network.prefixlen != 24:
                raise ValueError("Supported persistent fabric addresses use /24")
            port = {key: endpoint[key] for key in ("role", "rdma_device", "netdev", "mac")}
            port["address"] = address
            observed = next(i for i in facts["interfaces"] if i["name"] == port["netdev"])
            prior = observed.get("network_manager", {}).get("connection_uuid")
            if prior:
                # The exact existing data connection is visible in the reviewed
                # plan; management interfaces can never enter this list.
                port["replace_connection_uuid"] = prior
            if preserve_control:
                if not prior:
                    raise ValueError("Fabric control requires an existing NetworkManager link-local connection")
                port["update_connection_uuid"] = prior
            host["data_interfaces"].append(port)
        hosts.append(host)
        inventory[host["host"]] = facts
    spec = {"owner": name, "controller_address": controller, "network": {"backend": "NetworkManager"}, "hosts": hosts,
            "require_idle_gpu": True, "preserve_control_ipv6": preserve_control}
    if value != fabric_layout.legacy_or_none(len(nodes)):
        # Layouts other than a pair or a four-Spark cycle name themselves; the
        # specifications of those two stay as records of their count expect.
        spec["layout"] = value
    plan = deploy_network.plan_network(spec, inventory)
    ident = hashlib.sha256(json.dumps({"nodes": [n["node_id"] for n in nodes], "spec": spec}, sort_keys=True).encode()).hexdigest()
    return {"schema": "sparkring-appliance-plan/v1", "id": ident, "nodes": nodes, "spec": spec,
            "inventory": {"hosts": inventory}, "network": plan, "preserve_existing_addresses": preserve,
            "reset_requested": reset, "fabric_cidr": fabric_cidr, "layout": value}


def _port(host, role):
    return next(p for p in host["data_interfaces"] if p["role"] == role)


def routes(plan, rank):
    """The approved routes of ``rank`` to the subnets of cables it is not attached to.

    Each goes to the cable's /24 through the cable neighbor in the direction
    of the shorter way, clockwise (port 0) on a tie; on a path the only way.
    The order is by cable, primary function first.
    """
    value = layout_of(plan)
    hosts = plan["spec"]["hosts"]
    size = value["size"]
    ports = {p["role"]: p for p in hosts[rank]["data_interfaces"]}
    attached = {str(ipaddress.ip_interface(p["address"]).network) for p in ports.values()}
    result = []
    if value["shape"] == fabric_layout.PAIR:
        return result
    for cable, first, second in fabric_layout.cables(value):
        for function in fabric_layout.FUNCTIONS:
            target = _port(hosts[first[0]], fabric_layout.port_role(first[1], function))
            subnet = str(ipaddress.ip_interface(target["address"]).network)
            if subnet in attached:
                continue
            if value["shape"] == fabric_layout.CYCLE:
                cw = min((first[0] - rank) % size, (second[0] - rank) % size)
                ccw = min((rank - first[0]) % size, (rank - second[0]) % size)
                direction = "cw" if cw <= ccw else "ccw"
            else:
                direction = "cw" if first[0] > rank else "ccw"
            peer_rank = (rank + (1 if direction == "cw" else -1)) % size
            peer = _port(hosts[peer_rank], ("ccw" if direction == "cw" else "cw") + "_" + function)
            result.append({"destination": subnet, "via": str(ipaddress.ip_interface(peer["address"]).ip),
                           "dev": ports[direction + "_" + function]["netdev"]})
    return result


def forwarding(plan, rank):
    """``[[in, out], ...]``: the netdev pairs between which ``rank`` forwards, both ways, per function."""
    value = layout_of(plan)
    if not fabric_layout.forwards(value, rank):
        return []
    ports = {p["role"]: p for p in plan["spec"]["hosts"][rank]["data_interfaces"]}
    result = []
    for function in fabric_layout.FUNCTIONS:
        a, b = (ports[side + "_" + function]["netdev"] for side in ("cw", "ccw"))
        result += [[a, b], [b, a]]
    return result


def persistent_config(plan, rank, *, relays=None):
    """The ``sparkring-fabric-state/v1`` boot record of ``rank``.

    ``layout`` names the fabric layout; ``relays``, when given, is the
    position's part of the relay plan (``runtime/host/relays.section``),
    which ``sparkring node restore`` installs at boot.
    """
    hosts = plan["spec"]["hosts"]
    host = hosts[rank]
    value = layout_of(plan)
    config = {"schema": "sparkring-fabric-state/v1", "cluster_id": plan["id"], "rank": rank,
              "node_id": host["node_id"], "management": {"address": host["management_address"], "interface": host["management_netdev"],
                                                          "witness": host["controller_probe_address"]},
              "interfaces": host["data_interfaces"], "routes": routes(plan, rank), "forwarding": forwarding(plan, rank),
              "ssh_target": host["host"], "size": len(hosts), "layout": value}
    if relays is not None:
        config["relays"] = relays
    return config


def cable_rows(plan):
    """The cables of a setup plan: ``[{"cable", "ends": [{"rank", "port", "roles": {function: role}}, ...]}]``.

    The first end is the cable's port-0 end (position ``cable`` on a path or
    cycle, position 0 on a pair).
    """
    value = layout_of(plan)
    rows = []
    for cable, *ends in fabric_layout.cables(value):
        rows.append({"cable": cable, "ends": [
            {"rank": position, "port": port,
             "roles": {function: fabric_layout.port_role(port, function) for function in fabric_layout.FUNCTIONS}}
            for position, port in ends]})
    return rows
