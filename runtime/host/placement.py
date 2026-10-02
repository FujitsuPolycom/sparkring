"""Where a deployment runs on a four-Spark ring: the whole ring, or one half of it.

A four-Spark ring serves one four-rank model on all four Sparks, or two
independent two-rank models, one on each half: Sparks 0 and 1, and Sparks 2
and 3. The Sparks of a half share one cable, port 0 of the lower rank to port
1 of the higher rank (``topology`` cabling), so a two-rank model on a half uses
that cable exactly as a pair uses its own. Sparks 1 and 2, or 3 and 0, also
share a cable; they are not offered, so the two halves never overlap and
both can serve at the same time.

A placement is the tuple of ring ranks a deployment uses: ``(0, 1)`` or
``(2, 3)`` for a half. A deployment on every Spark of its cluster (a pair, or
a four-rank model on a ring) has no placement (``None``).

Each rank of a half reaches its partner through the port functions that face
it: the lower rank through its port 0 functions (roles ``cw_primary`` and
``cw_secondary`` of the setup plan), the higher rank through its port 1
functions (``ccw_primary`` and ``ccw_secondary``). ``fabric_rows`` takes each
rank's host address, socket interface and RDMA devices from those functions.
"""
import ipaddress

from runtime.host import control

HALVES = ((0, 1), (2, 3))
# What a whole ring is called in messages, beside the halves' "Sparks 0 and 1".
WHOLE = "all four Sparks"


def parse(text):
    """The placement that ``--on`` names: ``0,1`` or ``2,3``.

    Raises ValueError naming the accepted values for anything else.
    """
    try:
        ranks = tuple(int(part) for part in str(text).replace(" ", "").split(","))
    except ValueError:
        ranks = ()
    if ranks not in HALVES:
        raise ValueError(f"--on {text} is not a half of the ring; use --on 0,1 or --on 2,3")
    return ranks


def text(placement):
    """``Sparks 2 and 3`` for a half, ``all four Sparks`` for ``None``."""
    return WHOLE if placement is None else f"Sparks {placement[0]} and {placement[1]}"


def flag(placement):
    """``--on 2,3`` for a half, the empty string for ``None``."""
    return "" if placement is None else f"--on {placement[0]},{placement[1]}"


def check(placement, *, cluster_size, profile_nodes, profile):
    """Refuse a placement the cluster or the profile cannot take; returns the placement.

    A half needs a four-Spark ring and a two-rank profile. Without a
    placement, the profile must use every Spark of the cluster; a two-rank
    profile on a four-Spark ring is resolved by ``choose`` instead.
    """
    if placement is None:
        return None
    if cluster_size != 4:
        raise ValueError("--on places a two-Spark model on half of a four-Spark ring; this cluster is a pair, "
                         "and its models already use both Sparks")
    if profile_nodes != 2:
        raise ValueError(f"{profile} uses all four Sparks; --on applies to two-Spark profiles")
    return placement


def network(port):
    return ipaddress.IPv4Interface(port["address"]).network


def fabric_rows(cluster, placement):
    """The site rows of a half: each rank's host, management address and the fabric facing its partner.

    Each row holds ``host``, ``management_ip``, ``fabric_ip`` (the primary
    function's address, the model's host IP and the API rank's master
    address), ``interface`` (that function's netdev, for NCCL and Gloo
    sockets) and ``hcas`` (the primary and secondary RDMA devices, in the
    order a pair lists ``rocep1s0f0, roceP2p1s0f0``). Raises ValueError when
    the recorded addresses show that the two Sparks do not share that cable.
    """
    hosts = cluster["plan"]["spec"]["hosts"]
    if len(hosts) != 4 or placement not in HALVES:
        raise ValueError("A half of the ring needs the four-Spark ring's setup plan")
    low, high = placement
    rows = []
    for rank, side, partner, other in ((low, "cw", high, "ccw"), (high, "ccw", low, "cw")):
        ports = {port["role"]: port for port in hosts[rank]["data_interfaces"]}
        facing = {port["role"]: port for port in hosts[partner]["data_interfaces"]}
        for function in ("primary", "secondary"):
            mine, theirs = ports[side + "_" + function], facing[other + "_" + function]
            if network(mine) != network(theirs):
                raise ValueError(f"Spark {rank} {mine['netdev']} ({mine['address']}) and Spark {partner} "
                                 f"{theirs['netdev']} ({theirs['address']}) are not on one subnet, so "
                                 f"{text(placement)} do not share the cable a two-Spark model needs. Run "
                                 "sudo sparkring setup to review the ring's fabric.")
        primary, secondary = ports[side + "_primary"], ports[side + "_secondary"]
        rows.append({"host": hosts[rank]["host"], "management_ip": hosts[rank]["management_address"],
                     "fabric_ip": str(ipaddress.IPv4Interface(primary["address"]).ip),
                     "interface": primary["netdev"], "hcas": [primary["rdma_device"], secondary["rdma_device"]]})
    return rows


def lan_address(plan, rank):
    """The IPv4 address of a Spark's own LAN connection, from its inventory in the setup plan; None without one.

    The LAN connection carries the main routing table's default route, as
    ``control.lan`` defines it for setup's discovery. A default route through
    the administration network (``control.INTERFACE``), as on a worker that
    reaches the Internet through Node A, or through a fabric function does
    not count. The route's preferred source address wins; otherwise the
    interface's first address that is not link-local.
    """
    host = plan["spec"]["hosts"][rank]
    facts = ((plan.get("inventory") or {}).get("hosts") or {}).get(host["host"]) or {}
    fabric = {port["netdev"] for port in host["data_interfaces"]}
    fabric |= {row.get("netdev") for row in facts.get("rdma") or [] if isinstance(row, dict)}
    routes = [route for route in facts.get("routes") or [] if isinstance(route, dict)
              and route.get("dst") == "default" and route.get("table", "main") == "main" and route.get("dev")]
    for route in sorted(routes, key=lambda route: route.get("metric") or 0):
        device = route["dev"]
        if device == control.INTERFACE or device in fabric:
            continue
        interface = next((row for row in facts.get("interfaces") or [] if row.get("name") == device), None)
        addresses = []
        for cidr in (interface or {}).get("ipv4") or []:
            try:
                value = ipaddress.IPv4Interface(cidr).ip
            except ValueError:
                continue
            if not (value.is_loopback or value.is_link_local or value.is_unspecified or value.is_multicast):
                addresses.append(str(value))
        if route.get("prefsrc") in addresses:
            return route["prefsrc"]
        if addresses:
            return addresses[0]
    return None


def api_address(cluster, placement):
    """``(address, note)``: where the model of ``placement`` serves its API, and a line about it or None.

    The API rank of the whole ring and of half (0, 1) is Node A, which serves
    at the cluster's recorded LAN address (``api_address``, set by setup
    through Node A's uplink) or else at its management address; this function
    then returns ``(None, None)`` and the site keeps that rule. The API rank of
    half (2, 3) is Spark 2. On a cluster whose management addresses are the
    LAN addresses (no ``api_address`` recorded), Spark 2 serves at its
    management address, also ``(None, None)``. Otherwise Spark 2 serves at its
    own LAN address (``lan_address``); a Spark 2 without one is reached at its
    administration address, which only Node A routes, and the note says so.
    """
    if placement is None or placement[0] == 0 or not cluster.get("api_address"):
        return None, None
    rank = placement[0]
    address = lan_address(cluster["plan"], rank)
    if address is not None:
        return address, None
    management = cluster["plan"]["spec"]["hosts"][rank]["management_address"]
    return None, (f"Spark {rank} has no LAN connection of its own, so its model's API is reached at its "
                  f"administration address {management}, from Node A only.")


def instance_label(placement):
    """The lowercase instance name ``sparkring up PROFILE --on`` gives a new half deployment: ``on-2-3``."""
    return f"on-{placement[0]}-{placement[1]}"


def from_lock(lock):
    """The placement a deployment lock records, or None for a deployment on every Spark."""
    value = (lock.get("site") or {}).get("placement")
    return tuple(value) if value else None

