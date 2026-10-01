"""Private management over the fabric: an authenticated WireGuard tree.

IPv6 link-local endpoints survive data IPv4 renumbering. The tree gives each
control address one unambiguous authenticated path and can share Node A's uplink.
The high-bandwidth inference fabric remains outside this interface.

Each tree link has a primary path, the fabric cable that discovery recorded
for it (the peer's ``endpoint`` and ``netdev``), and may list fallback paths
(``alternates``): the other fabric function pairs between the same two
Sparks, then their LAN addresses. Fallbacks change
only the UDP endpoint of a WireGuard peer, never its keys, addresses or
routes, so the tree and its routing stay the same on every path. ``choose``
decides which path a peer uses; ``control_node.up`` applies it.
"""
import base64
import copy
import ipaddress
import re

INTERFACE = "sr-control"
PORT = 51871
SSH_PORT = 2222
# Seconds after which a peer's latest handshake shows that its path carries
# nothing. WireGuard rejects a session 180 s after its handshake
# (REJECT_AFTER_TIME). Every peer has PersistentKeepalive = 15, so one side
# sends at least every 15 s; the session's initiator renews it on sending
# after 120 s, and on receiving after 165 s, which a keepalive brings by 180 s
# when only the other side sends. A reachable peer's handshake is therefore at
# most about 180 s old; 200 s adds room for lost handshake attempts, which
# WireGuard repeats every 5 s.
HANDSHAKE_STALE = 200
# Seconds after a refresh moves a peer to another path before the next
# refresh judges whether the peer answered there. The refresh timer runs every
# 20 s, so this is the next run.
VERIFY_AFTER = 15
# Seconds the primary path's link must pass its check before the tunnel
# returns to it: two refreshes, by which NetworkManager has restored the
# function's link-local address after a cable returns.
PRIMARY_SETTLE = 15
# Seconds the tunnel stays on a fallback after a return to the primary path
# failed, doubling with each consecutive failure up to PRIMARY_RETRY_MAX. A
# primary path whose link goes down and up again is tried at once.
PRIMARY_RETRY = 600
PRIMARY_RETRY_MAX = 3600
# Peer fields that fallback paths add to a configuration; configure() accepts
# an installed configuration that differs only in these.
EXTENSION = ("alternates", "address")
CABLE_FIELDS = {"via", "netdev", "mac", "address", "peer"}
LAN_FIELDS = {"via", "netdev", "peer"}


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


def port(function):
    """The ConnectX port of an inventory function from its RDMA device name (``rocep1s0f1``: 1), or None."""
    match = re.search(r"f(\d+)$", str(function.get("device") or ""))
    return int(match[1]) if match else None


def lan(inventory, subnet):
    """``{"netdev", "address"}`` of a Spark's LAN connection from its inventory, or None.

    The LAN connection carries the main routing table's default route
    (``uplink``) and a private IPv4 address (``api_address``) outside the
    administration subnet. A fabric function or the administration interface
    never counts.
    """
    interface, address = inventory.get("uplink"), inventory.get("api_address")
    if not interface or not address or interface == INTERFACE:
        return None
    if interface in {f.get("netdev") for f in inventory.get("functions") or []}:
        return None
    try:
        netdev(interface)
        value = ipaddress.IPv4Address(address)
    except ValueError:
        return None
    if not value.is_private or value.is_loopback or value.is_link_local or value in ipaddress.IPv4Network(subnet):
        return None
    return {"netdev": interface, "address": str(value)}


def cable_pairs(a, b):
    """The fabric function pairs that connect two Sparks, from their inventories.

    Returns ``{(a netdev, b netdev): (a end, b end)}``; each end is
    ``{"netdev", "mac", "address", "port"}`` with the function's link-local
    address. A pair counts when either Spark's neighbor cache holds, on one of
    its functions, the other function's MAC and link-local address, and that
    entry answered the inventory's echo.
    """
    def functions(inventory):
        return {f["netdev"]: f for f in inventory.get("functions") or [] if f.get("netdev") and f.get("addresses")}

    def end(function, address):
        return {"netdev": function["netdev"], "mac": str(function["mac"]).lower(), "address": address,
                "port": port(function)}

    found = {}
    for local, remote, flip in ((a, b, False), (b, a, True)):
        own, others = functions(local), functions(remote)
        by_mac = {str(f.get("mac", "")).lower(): f for f in others.values()}
        for neighbor in local.get("neighbors") or []:
            function = own.get(neighbor.get("dev"))
            other = by_mac.get(str(neighbor.get("lladdr", "")).lower())
            try:
                address = ipaddress.IPv6Address(str(neighbor.get("dst", "")).split("%")[0])
            except ValueError:
                continue
            if (not function or not other or neighbor.get("answered") is not True or not address.is_link_local
                    or str(address) not in other["addresses"]):
                continue
            pair = (end(function, function["addresses"][0]), end(other, str(address)))
            if flip:
                pair = pair[::-1]
            found.setdefault((pair[0]["netdev"], pair[1]["netdev"]), pair)
    return found


def extend(configs, inventories):
    """Every Spark's configuration with its peers' control addresses and fallback paths.

    ``configs`` are all configurations of one administration network
    (``plan``); ``inventories`` maps each Spark's ID to its inventory
    (``bootstrap.probe``). For each tree link, ``alternates`` lists in
    preference order the other fabric function pairs between the two Sparks,
    pairs on another ConnectX port (another cable) first, then the two Sparks'
    LAN addresses when both have one. Both ends of a link list the same paths
    in the same order. A pair never uses a function that the primary path or
    an earlier pair uses. Each peer's ``address`` is its control address,
    which the refresh contacts to confirm a path it moves a peer to.
    """
    result = copy.deepcopy(configs)
    by_id = {config["id"]: config for config in result}
    for config in result:
        for peer in config["peers"]:
            peer["address"] = by_id[peer["id"]]["address"]
            peer["alternates"] = []
    for low in result:
        for peer in low["peers"]:
            if low["id"] > peer["id"]:
                continue
            high = by_id[peer["id"]]
            back = next((p for p in high["peers"] if p["id"] == low["id"]), None)
            if back is None:
                raise ValueError("Control configurations disagree about a tree link")
            low_inventory, high_inventory = inventories.get(low["id"]) or {}, inventories.get(high["id"]) or {}
            used = ({p["netdev"] for p in low["peers"]}, {p["netdev"] for p in high["peers"]})
            recorded = next((port(f) for f in low_inventory.get("functions") or [] if f.get("netdev") == peer["netdev"]),
                            None)

            def preference(pair, recorded=recorded):
                # Another cable first; then same-named functions, the parallel pairs of a port.
                other_cable = pair[0]["port"] is not None and recorded is not None and pair[0]["port"] != recorded
                return (not other_cable, pair[0]["netdev"] != pair[1]["netdev"], pair[0]["netdev"], pair[1]["netdev"])

            for near, far in sorted(cable_pairs(low_inventory, high_inventory).values(), key=preference):
                if near["netdev"] in used[0] or far["netdev"] in used[1]:
                    continue
                used[0].add(near["netdev"])
                used[1].add(far["netdev"])
                for path, mine, theirs in ((peer, near, far), (back, far, near)):
                    path["alternates"].append({"via": "cable", "netdev": mine["netdev"], "mac": mine["mac"],
                                               "address": mine["address"], "peer": theirs["address"]})
            near, far = lan(low_inventory, low["subnet"]), lan(high_inventory, high["subnet"])
            if near and far and near["address"] != far["address"]:
                peer["alternates"].append({"via": "lan", "netdev": near["netdev"], "peer": far["address"]})
                back["alternates"].append({"via": "lan", "netdev": far["netdev"], "peer": near["address"]})
    return result


def base(config):
    """The configuration without the peer fields that fallback paths add (EXTENSION)."""
    value = copy.deepcopy(config)
    for peer in value.get("peers") or []:
        for field in EXTENSION:
            peer.pop(field, None)
    return value


def endpoint(path):
    """The WireGuard endpoint of a path: ``[fe80::2%enp1s0f0np0]:51871`` or ``192.0.2.12:51871``."""
    if path["via"] == "lan":
        return f"{path['peer']}:{PORT}"
    return f"[{path['peer']}%{path['netdev']}]:{PORT}"


def parse_endpoint(text):
    """``(address, scope, port)`` of an endpoint as WireGuard prints it, or None.

    The scope is the interface name or index of an IPv6 link-local address,
    or None. WireGuard prints an index when the interface it recorded no
    longer exists, as after a driver restart.
    """
    if not text or text == "(none)":
        return None
    try:
        match = re.fullmatch(r"\[([0-9A-Fa-f:.]+)(?:%([^\]]+))?\]:(\d+)", text)
        if match:
            return ipaddress.IPv6Address(match[1]), match[2], int(match[3])
        match = re.fullmatch(r"([0-9.]+):(\d+)", text)
        if match:
            return ipaddress.IPv4Address(match[1]), None, int(match[2])
    except ValueError:
        return None
    return None


def paths(config, peer):
    """The paths of one tunnel peer in preference order: the primary path, then its fallbacks.

    Each path is ``{"via": "cable" | "lan", "netdev", "peer", ...}``: the local
    interface and the peer's address on that path. A cable path also names its
    local function's ``mac`` and link-local ``address`` when they are recorded.
    """
    recorded = parse_endpoint(peer["endpoint"])
    primary = {"via": "cable", "netdev": peer["netdev"], "peer": str(recorded[0]) if recorded else None}
    link = next((row for row in config.get("links") or [] if row.get("netdev") == peer["netdev"]), None)
    if link:
        primary.update(mac=link["mac"], address=link["address"])
    return [primary, *(dict(path) for path in peer.get("alternates") or [])]


def match(options, text):
    """The index of the path among ``options`` whose address and port the endpoint ``text`` names, or None.

    The scope is ignored, so an endpoint whose interface index is stale still
    names its path.
    """
    observed = parse_endpoint(text)
    if observed is None:
        return None
    for index, path in enumerate(options):
        expected = parse_endpoint(endpoint(path)) if path.get("peer") else None
        if expected and expected[0] == observed[0] and expected[2] == observed[2]:
            return index
    return None


def same_endpoint(text, path):
    """Whether WireGuard's endpoint ``text`` is exactly the path's, interface name included."""
    observed, expected = parse_endpoint(text), parse_endpoint(endpoint(path))
    return observed is not None and observed == expected


def path_text(path):
    """``cable enp1s0f0np0`` or ``LAN 192.0.2.12``: a path as status lines name it."""
    return f"LAN {path['peer']}" if path["via"] == "lan" else f"cable {path['netdev']}"


def validate(config):
    """Raise ValueError unless every peer's control address and fallback paths are well formed."""
    network = ipaddress.IPv4Network(config["subnet"])
    for peer in config["peers"]:
        if "address" in peer:
            address = ipaddress.IPv4Address(peer["address"])
            if address not in network or str(address) == config["address"]:
                raise ValueError("Control peer address must be another address of the administration subnet")
        seen = {peer["endpoint"]}
        for path in peer.get("alternates") or []:
            if not isinstance(path, dict) or path.get("via") not in ("cable", "lan"):
                raise ValueError("Fallback path must be a cable or the LAN")
            if set(path) != (CABLE_FIELDS if path["via"] == "cable" else LAN_FIELDS):
                raise ValueError("Fallback path has unexpected fields")
            netdev(path["netdev"])
            if path["via"] == "cable":
                if not re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", str(path["mac"])):
                    raise ValueError("Fallback cable needs its function's lowercase MAC")
                for value in (path["address"], path["peer"]):
                    if not ipaddress.IPv6Address(value).is_link_local:
                        raise ValueError("Fallback cable endpoints must be IPv6 link-local")
            else:
                value = ipaddress.IPv4Address(path["peer"])
                if not value.is_private or value.is_loopback or value.is_link_local or value in network:
                    raise ValueError("Fallback LAN address must be a private IPv4 address outside the administration subnet")
            if endpoint(path) in seen:
                raise ValueError("Fallback path repeats another path of the same peer")
            seen.add(endpoint(path))


def choose(state, usable, current, handshake, received, now, *, has_endpoint=False):
    """The path one administration tunnel peer should use, from one refresh's observations.

    ``usable`` says per path (primary path first, ``paths``) whether its local
    link passes its check; ``current`` is the index of the path whose address
    WireGuard's endpoint names, or None; ``handshake`` is the Unix time of the
    latest handshake (0: none); ``received`` counts the bytes WireGuard
    received from the peer; ``has_endpoint`` says whether WireGuard has an
    endpoint at all. ``state`` is what the previous call returned, or None.

    Returns ``(target, check, reason, state)``: ``target`` is the path whose
    endpoint the peer should have (None: leave WireGuard's endpoint), ``check``
    is True after a move that the caller confirms by contacting the peer,
    ``reason`` names why the peer moves (``link``, ``answer``, ``handshake``,
    ``endpoint`` or ``primary``) or is None.

    Rules, in order:

    - A move is confirmed when, by the next refresh (VERIFY_AFTER), a handshake
      followed it or WireGuard received bytes over the path moved to. Otherwise, or
      when the peer's packets moved the endpoint elsewhere first, that path
      failed. A failed path is skipped until a later handshake succeeds or its
      link fails its check; a failed primary path is also held back
      (PRIMARY_RETRY, doubling).
    - Without a known path in use (and no working endpoint), the first
      usable path is chosen.
    - When the link of the path in use fails, the first usable path that has not
      failed is chosen, the primary path first unless it is held back.
    - When the path in use has had no handshake for HANDSHAKE_STALE seconds,
      it failed and the next usable path is chosen.
    - After this refresh moved the peer off its primary path, it returns once
      that path's link has passed for PRIMARY_SETTLE seconds. A peer whose
      endpoint the other Spark's packets moved stays where they arrive:
      WireGuard follows the source of authenticated packets, so the Spark that
      left the primary path brings both ends back to it.
    - When every usable path failed, the most preferred usable one is kept
      without further moves, so a powered-off peer causes no cycling.
    """
    s = copy.deepcopy(state) if state else {}
    s.setdefault("since", now)
    failed = {int(k): v for k, v in (s.get("failed") or {}).items()}
    count = len(usable)
    fresh = bool(handshake) and now - handshake <= HANDSHAKE_STALE
    for index in list(failed):
        if index >= count or not usable[index] or handshake > failed[index]:
            del failed[index]
    if usable[0]:
        if s.get("primary_up_since") is None:
            s["primary_up_since"] = now
    else:
        s.update(primary_up_since=None, held=None, returns=0)
    verify, reason = s.get("verify"), None

    def fail(index):
        failed[index] = now
        if index == 0:
            s["returns"] = s.get("returns", 0) + 1
            s["held"] = now + min(PRIMARY_RETRY * 2 ** (s["returns"] - 1), PRIMARY_RETRY_MAX)

    if current != s.get("path"):
        if verify:
            fail(verify["path"])
            verify = None
            s["left"] = current != 0
        else:
            s["left"] = False
        s.update(path=current, since=now)
    retry = False
    if verify:
        if handshake > verify["at"] or received > verify["received"]:
            verify = None
            failed.clear()
            if current == 0:
                s["returns"] = 0
        elif now - verify["at"] >= VERIFY_AFTER:
            fail(current)
            verify, retry, reason = None, True, "answer"
    held = (s.get("held") or 0) > now

    def best(exclude):
        for index in range(count):
            if usable[index] and index != exclude and index not in failed and not (index == 0 and held):
                return index, True
        for index in range(count):
            if usable[index]:
                return index, False
        return None, False

    target, check = current, False
    if verify:
        pass
    elif current is None:
        if not (has_endpoint and fresh):
            (target, check), reason = best(None), "endpoint"
    elif not usable[current]:
        (target, check), reason = best(current), "link"
    elif retry:
        target, check = best(current)
    elif not fresh and now - s["since"] >= HANDSHAKE_STALE and current not in failed:
        fail(current)
        (target, check), reason = best(current), "handshake"
    elif (current != 0 and s.get("left") and usable[0] and not held and 0 not in failed
          and now - s["primary_up_since"] >= PRIMARY_SETTLE):
        target, check, reason = 0, True, "primary"
    if target is not None and target != current:
        s.update(path=target, since=now, left=target != 0)
        verify = {"path": target, "at": now, "received": received} if check else None
    else:
        check = False
    if s.get("path") == 0:
        s["left"] = False
    s["verify"] = verify
    s["failed"] = {str(k): v for k, v in sorted(failed.items())}
    return target, check, reason, s


def plan(nodes, edges, head, *, subnet="10.253.255.0/29", share_uplink=True):
    """nodes: authenticated inventories; edges: reciprocal physical cable ends.

    The primary path of each tree link is its edge; the inventories' other
    cables and LAN addresses become its fallback paths (``extend``).
    """
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
    return extend(result, by_id)


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
    """Only the explicitly approved control network receives management/NAT rules.

    The tunnel's UDP port is open on each recorded link to link-local
    sources, and on each fallback path's interface only to that peer's
    address on the path.
    """
    network = str(ipaddress.IPv4Network(config["subnet"]))
    rules = []
    for interface in sorted({p["netdev"] for p in config["peers"]}):
        netdev(interface)
        rules.append(("ip6tables", ["INPUT", "-i", interface, "-s", "fe80::/10", "-p", "udp", "--dport", str(PORT), "-j", "ACCEPT"]))
    for peer in config["peers"]:
        for path in peer.get("alternates") or []:
            interface = netdev(path["netdev"])
            if path["via"] == "cable":
                rule = ("ip6tables", ["INPUT", "-i", interface, "-s", str(ipaddress.IPv6Address(path["peer"])) + "/128",
                                      "-p", "udp", "--dport", str(PORT), "-j", "ACCEPT"])
            else:
                rule = ("iptables", ["INPUT", "-i", interface, "-s", str(ipaddress.IPv4Address(path["peer"])) + "/32",
                                     "-p", "udp", "--dport", str(PORT), "-j", "ACCEPT"])
            if rule not in rules:
                rules.append(rule)
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
