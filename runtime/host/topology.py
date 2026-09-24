"""Identify the supported p0 pair or directed four-node ring from LLDP and NIC facts."""
import copy
import hashlib
import ipaddress
import json
import re

from scripts import deploy_network

DEVICES = {"cw_primary": "rocep1s0f0", "cw_secondary": "roceP2p1s0f0",
           "ccw_primary": "rocep1s0f1", "ccw_secondary": "roceP2p1s0f1"}


def lldp_rows(document):
    interfaces = document.get("lldp", {}).get("interface", [])
    if isinstance(interfaces, dict):
        interfaces = [{name: value} for name, value in interfaces.items()]
    rows = []
    for entry in interfaces:
        for netdev, neighbors in entry.items():
            for neighbor in neighbors if isinstance(neighbors, list) else [neighbors]:
                chassis = neighbor.get("chassis", {})
                entries = [("", chassis)] if "id" in chassis else chassis.items()
                for name, details in entries:
                    if not isinstance(details, dict):
                        continue
                    port = neighbor.get("port", {}).get("id", {})
                    rows.append({"netdev": netdev, "hostname": details.get("name", name),
                                 "chassis": details.get("id", {}).get("value", "").lower(),
                                 "port": str(port.get("value", "")), "port_type": port.get("type")})
    return rows


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


def ordered_nodes(nodes, head_id):
    if len(nodes) not in (2, 4) or len({node["node_id"] for node in nodes}) != len(nodes):
        raise ValueError("Select exactly two or four distinct authenticated Sparks")
    by_id = {node["node_id"]: node for node in nodes}
    if head_id not in by_id:
        raise ValueError("The controller/head must be one of the selected Sparks")
    ports = {node["node_id"]: endpoints(node) for node in nodes}
    edges = {}
    for node in nodes:
        ident = node["node_id"]
        local_ports = {p["netdev"]: p for p in ports[ident].values()}
        for observation in lldp_rows(node["lldp"]):
            local = local_ports.get(observation["netdev"])
            if local is None or len(nodes) == 2 and local["port"] != 0:
                continue
            matches = []
            for peer in nodes:
                if peer["node_id"] == ident:
                    continue
                chassis_macs = {item["mac"].lower() for item in peer["facts"]["interfaces"]}
                named = str(observation["hostname"]).rstrip(".") in {peer["hostname"], peer["hostname"] + ".local"}
                for endpoint in ports[peer["node_id"]].values():
                    match_mac = observation["port"].lower() == endpoint["mac"]
                    match_name = observation["port"] == endpoint["netdev"] and (named or observation["chassis"] in chassis_macs)
                    if match_mac or match_name:
                        matches.append((peer["node_id"], endpoint["port"]))
            matches = set(matches)
            if len(matches) != 1:
                raise ValueError("LLDP peer cannot be matched uniquely to authenticated NIC inventory")
            key, value = (ident, local["port"]), matches.pop()
            if key in edges and edges[key] != value:
                raise ValueError("Socket Direct functions disagree about the physical cable peer")
            edges[key] = value
    expected_ports = (0,) if len(nodes) == 2 else (0, 1)
    for ident in by_id:
        for port in expected_ports:
            peer = edges.get((ident, port))
            if peer is None or edges.get(peer) != (ident, port):
                raise ValueError("Missing reciprocal LLDP cable evidence; wait for LLDP or inspect cabling")
            if peer[1] != (0 if len(nodes) == 2 else 1 - port):
                raise ValueError("Supported cabling is p0-p0 for a pair or p0-to-next-p1 for a ring")
    order = [head_id]
    while len(order) < len(nodes):
        next_id = edges[order[-1], 0][0]
        if next_id in order:
            raise ValueError("Selected Sparks do not form one complete cable cycle")
        order.append(next_id)
    if edges[order[-1], 0][0] != head_id:
        raise ValueError("The ring does not close back to rank 0")
    return [by_id[ident] for ident in order]


def build_spec(nodes, head_id, *, name="sparkring", fabric_cidr="198.18.0.0/21", reset=False, preserve_control=False):
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Choose a lowercase cluster name")
    nodes = ordered_nodes(nodes, head_id)
    width = len(nodes)
    roles = list(DEVICES) if width == 4 else list(DEVICES)[:2]
    maps = [endpoints(node) for node in nodes]
    selected = [ports[role] for ports in maps for role in roles]
    existing = [port["ipv4"] for port in selected]
    if not reset and any(existing) and not all(len(addresses) == 1 for addresses in existing):
        raise ValueError("Partial fabric addressing requires inspection; no automatic renumbering")
    preserve = any(existing) and not reset
    network = ipaddress.IPv4Network(fabric_cidr, strict=True)
    if network.prefixlen > 21 or network.prefixlen < 16:
        raise ValueError("Choose an isolated /16 through /21 fabric supernet")
    networks = list(network.subnets(new_prefix=24))
    hosts, inventory = [], {}
    controller = nodes[0]["facts"]["management"]["address"]
    for rank, node in enumerate(nodes):
        facts = copy.deepcopy(node["facts"])
        facts["rank"] = rank
        host = {"rank": rank, "host": facts["ssh_target"], "node_id": node["node_id"],
                "management_address": facts["management"]["address"], "management_netdev": facts["management"]["interface"],
                "controller_probe_address": facts["management"]["controller_address"],
                "backup_dir": f"/var/lib/sparkring/backups/{name}/rank{rank}", "data_interfaces": []}
        for role in roles:
            endpoint = maps[rank][role]
            secondary = role.endswith("secondary")
            clockwise = role.startswith("cw_")
            subnet = networks[int(secondary)] if width == 2 else networks[2 * (rank if clockwise else (rank - 1) % width) + int(secondary)]
            address = endpoint["ipv4"][0] if preserve else str(subnet.network_address + (rank + 1 if width == 2 else 1 if clockwise else 2)) + "/24"
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
    plan = deploy_network.plan_network(spec, inventory)
    ident = hashlib.sha256(json.dumps({"nodes": [n["node_id"] for n in nodes], "spec": spec}, sort_keys=True).encode()).hexdigest()
    return {"schema": "sparkring-appliance-plan/v1", "id": ident, "nodes": nodes, "spec": spec,
            "inventory": {"hosts": inventory}, "network": plan, "preserve_existing_addresses": preserve,
            "reset_requested": reset, "fabric_cidr": fabric_cidr}


def persistent_config(plan, rank):
    hosts = plan["spec"]["hosts"]
    host = hosts[rank]
    ports = {p["role"]: p for p in host["data_interfaces"]}
    routes, forwarding = [], []
    if len(hosts) == 4:
        attached = {str(ipaddress.ip_interface(p["address"]).network) for p in ports.values()}
        for edge in range(4):
            for function in ("primary", "secondary"):
                target = next(p for p in hosts[edge]["data_interfaces"] if p["role"] == "cw_" + function)
                subnet = str(ipaddress.ip_interface(target["address"]).network)
                if subnet in attached:
                    continue
                cw = min((edge-rank) % 4, (edge+1-rank) % 4)
                ccw = min((rank-edge) % 4, (rank-edge-1) % 4)
                direction = "cw" if cw <= ccw else "ccw"
                peer_rank = (rank + (1 if direction == "cw" else -1)) % 4
                peer_role = ("ccw" if direction == "cw" else "cw") + "_" + function
                peer = next(p for p in hosts[peer_rank]["data_interfaces"] if p["role"] == peer_role)
                routes.append({"destination": subnet, "via": str(ipaddress.ip_interface(peer["address"]).ip),
                               "dev": ports[direction + "_" + function]["netdev"]})
        for function in ("primary", "secondary"):
            a, b = (ports[p + "_" + function]["netdev"] for p in ("cw", "ccw"))
            forwarding += [[a, b], [b, a]]
    return {"schema": "sparkring-fabric-state/v1", "cluster_id": plan["id"], "rank": rank,
            "node_id": host["node_id"], "management": {"address": host["management_address"], "interface": host["management_netdev"],
                                                        "witness": host["controller_probe_address"]},
            "interfaces": host["data_interfaces"], "routes": routes, "forwarding": forwarding,
            "ssh_target": host["host"], "size": len(hosts)}
