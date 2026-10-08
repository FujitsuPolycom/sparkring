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

The installer's profiles run on a pair or a four-Spark ring (``require_layout``);
setup also forms lines and rings of other sizes, which they do not serve.
"""
import ipaddress
from pathlib import Path

from runtime.common import fabric_layout, installer
from runtime.host import control, node

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


# The layouts the installer's profiles serve.
SERVED_LAYOUTS = (fabric_layout.layout(fabric_layout.PAIR, 2), fabric_layout.layout(fabric_layout.CYCLE, 4))


def require_layout(cluster):
    """Refuse a cluster whose fabric layout no installer profile serves; returns the layout."""
    from runtime.host import topology
    value = topology.layout_of(cluster["plan"])
    if value not in SERVED_LAYOUTS:
        raise ValueError(f"This fabric is a {fabric_layout.name(value)}; the installer's profiles run on a pair or a "
                         "four-Spark ring (cycle-4). sudo sparkring fabric show describes the fabric")
    return value


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



# Slots ------------------------------------------------------------------------
#
# A cluster has one slot per placement: a pair has only the whole cluster, a
# four-Spark ring the whole ring and its two halves. Each slot records its own
# active deployment and its own model switch. The whole cluster keeps
# ``active.json`` and ``transaction.json`` in Node A's controller directory;
# a half keeps them in ``slots/0-1`` or ``slots/2-3`` there. The whole ring
# and a half never serve at the same time: a switch into one stops the models
# of the other (``displaced``).

def slot_directory(state_root, placement):
    """The directory holding the slot's ``active.json`` and ``transaction.json``."""
    root = Path(state_root)
    return root if placement is None else root / "slots" / f"{placement[0]}-{placement[1]}"


def slots(cluster_size):
    """Every slot of a cluster: ``[None]`` for a pair, the whole ring and both halves for a ring."""
    return [None] if cluster_size != 4 else [None, *HALVES]


def conflicting(placement):
    """The slots that cannot serve while ``placement`` serves: the halves for the whole ring, else the whole ring."""
    return list(HALVES) if placement is None else [None]


def recorded(state_root, placement):
    """The deployment directory the slot's ``active.json`` names, or None without a record."""
    path = slot_directory(state_root, placement) / "active.json"
    if not path.exists():
        return None
    return Path(installer.read(path)["path"])


def record(state_root, placement, directory):
    """Make ``directory`` the slot's active deployment."""
    node.save(slot_directory(state_root, placement), "active.json", {"path": str(directory)}, mode=0o600)


def forget(state_root, placement, directory):
    """Remove the slot's active record when it still names ``directory``."""
    current = recorded(state_root, placement)
    if current is not None and Path(current).resolve() == Path(directory).resolve():
        (slot_directory(state_root, placement) / "active.json").unlink()


def clear_conflicting(state_root, placement):
    """Remove the active records of the slots that conflict with ``placement``; their models are stopped."""
    for slot in conflicting(placement):
        (slot_directory(state_root, slot) / "active.json").unlink(missing_ok=True)


def journal(state_root, placement):
    """The slot's model switch record (``transaction.json``), or None."""
    path = slot_directory(state_root, placement) / "transaction.json"
    return installer.read(path) if path.exists() else None


def of_directory(directory):
    """The placement of the deployment in ``directory``; None for one on every Spark or without a lock."""
    try:
        return from_lock(installer.read(Path(directory) / "deployment.lock.json"))
    except (OSError, ValueError):
        return None


def exists(directory):
    return directory is not None and (Path(directory) / "deployment.lock.json").exists()


def stopped(directory):
    """Whether the deployment runs no model: its last operation is a completed down, or none ran.

    A deployment that automatic release removed (``retention.release_record``)
    has a completed down as its last operation too.
    """
    path = Path(directory) / "state.json"
    if not path.exists():
        return True
    state = installer.read(path)
    return state.get("operation") == "down" and bool(state.get("complete"))


def actives(state_root, cluster_size):
    """``{slot: deployment directory}`` of every slot whose record names an existing deployment."""
    result = {}
    for slot in slots(cluster_size):
        directory = recorded(state_root, slot)
        if exists(directory):
            result[slot] = directory
    return result


def displaced(state_root, placement, cluster_size):
    """The deployments of the conflicting slots that a switch into ``placement`` stops, as directories.

    Only deployments that may run count; a stopped one is only forgotten.
    """
    if cluster_size != 4:
        return []
    found = []
    for slot in conflicting(placement):
        directory = recorded(state_root, slot)
        if exists(directory) and not stopped(directory):
            found.append(Path(directory))
    return found


PENDING = ("stopping-previous", "starting", "verifying", "recovering-previous", "needs-attention")


def unfinished_switches(state_root, placement, cluster_size):
    """``[(slot, record)]`` of conflicting slots whose model switch stopped before it finished."""
    if cluster_size != 4:
        return []
    result = []
    for slot in conflicting(placement):
        value = journal(state_root, slot)
        if value and value.get("state") in PENDING:
            result.append((slot, value))
    return result


def choose(state_root, profile):
    """The half for a two-rank profile installed on a four-Spark ring without ``--on``.

    The one half that serves no model is chosen; otherwise ValueError names
    both choices and what each half holds.
    """
    holding = {}
    for half in HALVES:
        directory = recorded(state_root, half)
        holding[half] = directory if exists(directory) and not stopped(directory) else None
    free = [half for half in HALVES if holding[half] is None]
    if len(free) == 1:
        return free[0]
    lines = [f"--on {half[0]},{half[1]}: {text(half)}, " + ("free" if holding[half] is None
                                                             else "serving " + profile_of(holding[half]))
             for half in HALVES]
    error = ValueError(f"{profile} uses two Sparks and this ring has four. Choose a half with --on 0,1 or --on 2,3. "
                       "Nothing has been changed.")
    error.details = {"lines": lines}
    raise error


def profile_of(directory):
    """The profile of the deployment in ``directory``, or its directory name when its lock cannot be read."""
    try:
        return installer.read(Path(directory) / "deployment.lock.json")["selection"]["profile"]
    except (OSError, ValueError, KeyError, TypeError):
        return Path(directory).name
