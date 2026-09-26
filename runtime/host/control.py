"""Private management over the fabric: an authenticated WireGuard tree.

IPv6 link-local endpoints survive data IPv4 renumbering. The tree gives each
control address one unambiguous authenticated path and can share Node A's uplink.
The high-bandwidth inference fabric remains outside this interface.
"""
import base64
import ipaddress
import re

INTERFACE = "sr-control"
PORT = 51871
SSH_PORT = 2222


def key(value):
    try:
        if len(base64.b64decode(value, validate=True)) != 32:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError("WireGuard requires a 32-byte base64 key") from None
    return value


def netdev(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,15}", value):
        raise ValueError("Invalid fabric interface")
    return value


def plan(nodes, edges, head, *, subnet="10.253.255.0/29", share_uplink=True):
    """nodes: authenticated inventories; edges: reciprocal physical cable ends."""
    ids = {n["id"] for n in nodes}
    if len(nodes) not in (2, 4) or len(ids) != len(nodes) or head not in ids:
        raise ValueError("Control setup requires two or four distinct authenticated Sparks")
    network = ipaddress.IPv4Network(subnet)
    if network.prefixlen != 29 or not network.is_private:
        raise ValueError("Choose a private /29 for SparkRing management")
    by_id = {n["id"]: n for n in nodes}
    adjacency = {ident: {} for ident in ids}
    for edge in edges:
        a, b = edge
        if a["id"] not in ids or b["id"] not in ids or a["id"] == b["id"]:
            raise ValueError("Cable has unknown or identical endpoints")
        for local, peer in ((a, b), (b, a)):
            netdev(local["netdev"])
            if not ipaddress.IPv6Address(peer["address"]).is_link_local:
                raise ValueError("Bootstrap control endpoints must be IPv6 link-local")
            if peer["id"] in adjacency[local["id"]]:
                raise ValueError("Multiple physical cables between the same nodes are unsupported")
            adjacency[local["id"]][peer["id"]] = (local, peer)
    if any(len(peers) != (1 if len(nodes) == 2 else 2) for peers in adjacency.values()):
        raise ValueError("Cables must form a pair or one complete four-node ring")
    # BFS picks a deterministic tree; the non-tree ring edge remains data-only.
    order, parent = [head], {head: None}
    for ident in order:
        for peer in sorted(adjacency[ident]):
            if peer not in parent:
                order.append(peer)
                parent[peer] = ident
    if len(order) != len(nodes):
        raise ValueError("The selected Sparks do not form one connected ring")
    addresses = {ident: str(network.network_address + rank + 1) for rank, ident in enumerate(order)}
    for n in nodes:
        key(n["public_key"])
        for route in n["routes"]:
            if route.get("dst", "default") in ("default", "0.0.0.0/0"):
                continue
            if ipaddress.ip_network(route["dst"], strict=False).overlaps(network):
                raise ValueError("Control subnet overlaps an existing route on " + n["hostname"])

    def descendants(ident):
        result = [ident]
        for member in result:
            result.extend(child for child in order if parent[child] == member)
        return result

    result = []
    for ident in order:
        n = by_id[ident]
        peers = []
        for peer in adjacency[ident]:
            if parent[peer] != ident and parent[ident] != peer:
                continue
            local, remote = adjacency[ident][peer]
            downstream = parent[peer] == ident
            routed = descendants(peer) if downstream else [i for i in order if i not in descendants(ident)]
            allowed = [addresses[i] + "/32" for i in routed]
            if not downstream and share_uplink:
                allowed = ["0.0.0.0/0"]
            peers.append({"id": peer, "key": by_id[peer]["public_key"], "allowed_ips": allowed,
                          "endpoint": f"[{remote['address']}%{local['netdev']}]:{PORT}", "netdev": local["netdev"]})
        value = {"schema": "sparkring-control/v1", "id": ident, "address": addresses[ident],
                 "head": ident == head, "head_address": addresses[head], "subnet": str(network),
                 "share_uplink": share_uplink, "peers": peers,
                 "links": [{"netdev": pair[0]["netdev"], "mac": pair[0]["mac"], "address": pair[0]["address"]}
                           for peer, pair in adjacency[ident].items() if parent[peer] == ident or parent[ident] == peer],
                 "uplink": n.get("uplink") if ident == head and share_uplink else None}
        if value["head"] and share_uplink:
            netdev(value["uplink"])
            if value["uplink"] in {f["netdev"] for f in n.get("functions", [])}:
                raise ValueError("Node A's Internet/SSH uplink must use its separate management Ethernet port")
        result.append(value)
    return result


def render(config, private_key, *, without_endpoint=()):
    """The wg-quick configuration of one Spark.

    Peers reached over a netdev in ``without_endpoint`` get no ``Endpoint``
    line: wg-quick cannot resolve a link-local endpoint whose interface is
    missing, and WireGuard learns such a peer's endpoint from its first
    authenticated packet.
    """
    address = ipaddress.IPv4Address(config["address"])
    rows = ["[Interface]", "Address = " + str(address) + "/32", "PrivateKey = " + key(private_key),
            f"ListenPort = {PORT}", "MTU = 1420"]
    for peer in config["peers"]:
        interface = netdev(peer["netdev"])
        endpoint = peer["endpoint"]
        match = re.fullmatch(r"\[([0-9A-Fa-f:]+)%([A-Za-z0-9_-]+)\]:51871", endpoint)
        if not match or match[2] != interface or not ipaddress.IPv6Address(match[1]).is_link_local:
            raise ValueError("Invalid control peer endpoint")
        allowed = [str(ipaddress.IPv4Network(ip)) for ip in peer["allowed_ips"]]
        rows += ["", "[Peer]", "PublicKey = " + key(peer["key"]), "AllowedIPs = " + ", ".join(allowed)]
        if interface not in without_endpoint:
            rows.append("Endpoint = " + endpoint)
        rows.append("PersistentKeepalive = 15")
    return "\n".join(rows) + "\n"


def firewall(config):
    """Only the explicitly approved control network receives management/NAT rules."""
    network = str(ipaddress.IPv4Network(config["subnet"]))
    rules = []
    for interface in sorted({p["netdev"] for p in config["peers"]}):
        netdev(interface)
        rules.append(("ip6tables", ["INPUT", "-i", interface, "-s", "fe80::/10", "-p", "udp", "--dport", str(PORT), "-j", "ACCEPT"]))
    rules.append(("iptables", ["INPUT", "-i", INTERFACE, "-s", network, "-p", "tcp", "--dport", str(SSH_PORT), "-j", "ACCEPT"]))
    rules.append(("iptables", ["FORWARD", "-i", INTERFACE, "-o", INTERFACE, "-j", "ACCEPT"]))
    if config["head"] and config["share_uplink"]:
        uplink = netdev(config["uplink"])
        rules += [("iptables", ["FORWARD", "-i", INTERFACE, "-o", uplink, "-s", network, "-j", "ACCEPT"]),
                  ("iptables", ["FORWARD", "-i", uplink, "-o", INTERFACE, "-d", network, "-m", "conntrack", "--ctstate", "ESTABLISHED,RELATED", "-j", "ACCEPT"]),
                  ("iptables", ["-t", "nat", "POSTROUTING", "-s", network, "-o", uplink, "-j", "MASQUERADE"])]
        for protocol in ("udp", "tcp"):
            rules.append(("iptables", ["INPUT", "-i", INTERFACE, "-s", network, "-p", protocol, "--dport", "53", "-j", "ACCEPT"]))
    return rules
