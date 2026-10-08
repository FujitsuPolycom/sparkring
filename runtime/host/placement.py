"""Where a deployment runs on a fabric of two to eight Sparks: every Spark, or an arc of them.

A fabric (``runtime/common/fabric_layout.py``) numbers its Sparks by
position, Node A at 0. An *arc* is a run of consecutive positions in cable
order: each next position is the Spark on the previous one's port 0, so on a
cycle an arc may cross the cable from the last position to Node A (positions
6, 7, 0 and 1 of an eight-Spark cycle). A deployment's rank ``r`` runs on its
arc's ``r``-th position; rank 0, the arc's first position, serves the API.

A *placement* is the tuple of an arc's positions, or None for a deployment
on every Spark of the fabric in position order (Node A is rank 0). ``--on``
names an arc by its positions (``0,1``, ``6,7,0,1``) or by its first and
last positions (``0-3``, ``4-7``, ``6-1``); an arc of every Spark from Node A
on is the placement None, and an arc of every Spark from another position is
refused. A placement on a layout has a *group*: its shape (``pair`` for two
positions, ``cycle`` for every Spark of a cycle, ``path`` otherwise), size
and positions (``group``).

What each transport serves (``prepared_serves``, ``unsupported``):

- The prepared transport of the published installer images runs a pair, a
  four-Spark cycle, and a two-Spark model on a half of a four-Spark cycle:
  positions 0 and 1, or 2 and 3 (``HALVES``).
- SIRCL ring sessions (``runtime/common/transport.py``) run every group whose
  lanes cross at most three relays (``MAX_RELAYS``, SIRCL's qualified limit):
  pairs, paths of three to five Sparks and whole cycles of three to eight. A
  path of six or more Sparks is unsupported.

A placement is also a *slot*. Each slot records its own active deployment
and its own model switch: the whole fabric keeps ``active.json`` and
``transaction.json`` in Node A's controller directory, an arc in
``slots/<its positions joined by "-">`` there (``slots/0-1``,
``slots/4-5-6-7``). Two slots *conflict* when they share a Spark. A switch
into one slot stops the models of every conflicting slot (``displaced``), so
models on disjoint arcs serve at once: two four-Spark groups on an
eight-Spark cycle, four pairs, or a group of four beside two pairs.

A group of two (``fabric_rows``) serves on the cable it shares: its first
Spark on the port 0 functions facing the second (roles ``cw_primary`` and
``cw_secondary`` of the setup plan), the second on its port 1 functions
(``ccw_primary`` and ``ccw_secondary``). A larger group's ranks reach each
other through relays, which carry only tagged RDMA traffic, so its ranks
bootstrap over their management addresses (``controller.model_site``).
"""
import ipaddress
from pathlib import Path

from runtime.common import fabric_layout, installer
from runtime.host import control, node

HALVES = ((0, 1), (2, 3))
# The most relays a SIRCL lane may cross (sparkring_sircl.vllm.fabric.DEFAULT_MAX_RELAYS).
MAX_RELAYS = 3
WORDS = {**fabric_layout.WORDS, 1: "one"}


def layout_of(cluster):
    """The fabric layout ``{"shape", "size"}`` of a cluster record."""
    from runtime.host import topology
    return topology.layout_of(cluster["plan"])


# Arcs -------------------------------------------------------------------------

def arc(layout, start, count):
    """The ``count`` positions from ``start`` in cable order; ValueError where the layout has none."""
    size = layout["size"]
    if not 0 <= start < size or not 1 <= count <= size:
        raise ValueError(f"A {fabric_layout.name(layout)} has no {count} consecutive Sparks from position {start}")
    if layout["shape"] != fabric_layout.CYCLE and start + count > size:
        raise ValueError(f"A {fabric_layout.name(layout)} ends at position {size - 1}; it has no "
                         f"{WORDS.get(count, count)} consecutive Sparks from position {start}")
    return tuple((start + step) % size for step in range(count))


def is_arc(layout, positions):
    """Whether ``positions`` are consecutive Sparks of ``layout`` in cable order."""
    positions = tuple(positions)
    try:
        return bool(positions) and arc(layout, positions[0], len(positions)) == positions
    except ValueError:
        return False


def wraps(placement):
    """Whether an arc crosses the cable from a cycle's last position to Node A."""
    return placement is not None and any(b < a for a, b in zip(placement, placement[1:]))


def _numbers(text):
    return tuple(int(part) for part in text.split(","))


def examples(layout):
    """The ``--on`` forms a layout accepts, for messages."""
    size = layout["size"]
    forms = ["--on 0,1", f"--on 0-{min(3, size - 1)}" if size > 2 else None]
    if layout["shape"] == fabric_layout.CYCLE and size >= 4:
        forms.append(f"--on {size - 2}-1")
    forms = [form for form in forms if form]
    return ", ".join(forms[:-1]) + " or " + forms[-1] if len(forms) > 1 else forms[0]


def parse(text, layout):
    """The placement ``--on TEXT`` names on ``layout``: an arc's positions in rank order, or None for every Spark.

    ``TEXT`` lists positions (``0,1``, ``6,7,0,1``) or names an arc's first
    and last positions (``0-3``; ``6-1`` crosses a cycle's last cable). An
    arc of every Spark from Node A on is None. Raises ValueError naming the
    accepted forms for anything else, and on a pair, whose models use both
    Sparks.
    """
    value = str(text).replace(" ", "")
    size = layout["size"]
    name = fabric_layout.name(layout)
    if layout["shape"] == fabric_layout.PAIR:
        raise ValueError("--on places a model on some of the fabric's Sparks; this cluster is a pair, and its models "
                         "already use both Sparks")
    refusal = (f"--on {text} names no consecutive Sparks of this {name}; name them in cable order, such as "
               f"{examples(layout)}")
    try:
        if "-" in value and "," not in value:
            first, last = (int(part) for part in value.split("-"))
            if not (0 <= first < size and 0 <= last < size) or first == last:
                raise ValueError
            if last < first and layout["shape"] != fabric_layout.CYCLE:
                raise ValueError
            positions = arc(layout, first, (last - first) % size + 1)
        else:
            positions = _numbers(value)
    except ValueError:
        raise ValueError(refusal) from None
    if len(positions) < 2 or not is_arc(layout, positions):
        raise ValueError(refusal)
    if len(positions) == size:
        if positions[0] != 0:
            raise ValueError(f"--on {text} names every Spark from position {positions[0]} on; a model on every Spark "
                             "runs with Node A as rank 0, so leave out --on")
        return None
    return positions


def positions(placement, size):
    """The positions a placement uses: every position for None."""
    return tuple(range(size)) if placement is None else tuple(placement)


def group(layout, placement):
    """``{"shape", "size", "name", "positions"}`` of a placement's group on ``layout``.

    Two positions are a ``pair``; every Spark of a cycle of three or more is a
    ``cycle``; anything else is a ``path``.
    """
    members = positions(placement, layout["size"])
    count = len(members)
    if count == 2:
        shape = fabric_layout.PAIR
    elif placement is None and layout["shape"] == fabric_layout.CYCLE:
        shape = fabric_layout.CYCLE
    else:
        shape = fabric_layout.PATH
    return {"shape": shape, "size": count, "name": shape if shape == fabric_layout.PAIR else f"{shape}-{count}",
            "positions": list(members)}


def relays(value):
    """The most relays a lane of a ``group`` crosses: none on a pair, ``n - 2`` on a path, ``n // 2 - 1`` on a cycle."""
    if value["shape"] == fabric_layout.PAIR:
        return 0
    if value["shape"] == fabric_layout.PATH:
        return value["size"] - 2
    return value["size"] // 2 - 1


def relayed(layout, placement):
    """Whether some rank of the placement reaches another through relays."""
    return relays(group(layout, placement)) > 0


def forwarding_positions(layout, placement):
    """The positions whose ConnectX relays the group's lanes: a path's inner Sparks, every Spark of a larger cycle."""
    value = group(layout, placement)
    if not relays(value):
        return []
    if value["shape"] == fabric_layout.CYCLE:
        return list(value["positions"])
    return list(value["positions"][1:-1])


def unsupported(layout, placement):
    """Why no transport runs the placement's group, or None."""
    value = group(layout, placement)
    count = relays(value)
    if count > MAX_RELAYS:
        return (f"{text(placement, layout['size'])} form a line of {WORDS[value['size']]} Sparks whose ends are "
                f"{WORDS[count]} relays apart; SIRCL ring sessions cross at most {WORDS[MAX_RELAYS]} relays, so a line "
                "of six or more Sparks is unsupported. Use a line of at most five Sparks, or every Spark of a cycle")
    return None


def prepared_serves(layout, placement):
    """Whether the prepared transport runs the placement: a whole pair or four-Spark cycle, or a half of the cycle."""
    if layout == fabric_layout.layout(fabric_layout.PAIR, 2):
        return placement is None
    if layout == fabric_layout.layout(fabric_layout.CYCLE, 4):
        return placement is None or tuple(placement) in HALVES
    return False


def prepared_refusal(layout, placement, text_given, reason):
    """The refusal for a placement only SIRCL runs, when the deployment would use the prepared transport.

    ``text_given`` is the ``--on`` text, or None; ``reason`` says why SIRCL
    cannot run, or None when the operator chose ``--transport prepared``.
    """
    because = (f"SIRCL ring sessions, which run it, cannot: {reason}" if reason else
               "--transport prepared was chosen")
    if (layout == fabric_layout.layout(fabric_layout.CYCLE, 4) and placement is not None and len(placement) == 2
            and text_given is not None):
        return (f"--on {text_given} is not a half of the ring; use --on 0,1 or --on 2,3. The prepared transport runs "
                f"two-Spark models only on the ring's halves, and {because}")
    where = text(placement, layout["size"])
    return (f"The prepared transport runs a pair, a four-Spark ring and the ring's halves; {where} of this "
            f"{fabric_layout.name(layout)} need SIRCL ring sessions, and {because}")


def text(placement, size=None):
    """``Sparks 2 and 3``, ``Sparks 4-7``, ``Sparks 6, 7, 0 and 1``; for None ``all four Sparks`` (``every Spark``)."""
    if placement is None:
        if size == 2:
            return "both Sparks"
        return f"all {WORDS[size]} Sparks" if size in WORDS else "every Spark"
    if len(placement) == 2:
        return f"Sparks {placement[0]} and {placement[1]}"
    if wraps(placement):
        return "Sparks " + ", ".join(map(str, placement[:-1])) + f" and {placement[-1]}"
    return f"Sparks {placement[0]}-{placement[-1]}"


def flag(placement):
    """``--on 2,3`` for two Sparks, ``--on 4-7`` or ``--on 6-1`` for more; the empty string for None."""
    if placement is None:
        return ""
    if len(placement) == 2:
        return f"--on {placement[0]},{placement[1]}"
    return f"--on {placement[0]}-{placement[-1]}"


def check(placement, *, layout, profile_nodes, profile):
    """Refuse a placement the fabric or the profile cannot take; returns the placement.

    An arc must have the profile's node count. Without a placement the
    profile must use every Spark of the fabric; a smaller profile is placed
    by ``choose`` instead. A group no transport runs is refused.
    """
    size = layout["size"]
    if placement is None:
        if profile_nodes != size:
            raise ValueError(f"{profile} serves {WORDS.get(profile_nodes, profile_nodes)} Sparks and this "
                             f"{fabric_layout.name(layout)} has {WORDS[size]}")
        problem = unsupported(layout, None)
        if problem:
            raise ValueError(problem)
        return None
    if profile_nodes != len(placement):
        if profile_nodes == size:
            smaller = sorted({count for count in installer_sizes() if count < size})
            noun = " and ".join(f"{WORDS[count]}-Spark" for count in smaller) or "smaller"
            raise ValueError(f"{profile} uses all {WORDS[size]} Sparks; --on applies to {noun} profiles")
        raise ValueError(f"{profile} serves {WORDS.get(profile_nodes, profile_nodes)} Sparks; {text(placement)} are "
                         f"{WORDS[len(placement)]}")
    problem = unsupported(layout, placement)
    if problem:
        raise ValueError(problem)
    return placement


def installer_sizes():
    """The node counts of the profiles ``sparkring install`` installs."""
    from runtime.host import models
    return {row["nodes"] for row in models.catalog() if row["automated"]}


def fitting_profiles(layout):
    """Installer profiles whose node count fits ``layout`` and whose group a transport runs, sorted."""
    from runtime.host import models
    size = layout["size"]
    found = []
    for row in models.catalog():
        nodes = row["nodes"]
        if not row["automated"] or not isinstance(nodes, int) or nodes > size or nodes < 2:
            continue
        if nodes == size or any(unsupported(layout, candidate) is None for candidate in aligned(layout, nodes)):
            found.append(row["profile"])
    return sorted(found)


def network(port):
    return ipaddress.IPv4Interface(port["address"]).network


def fabric_rows(cluster, placement):
    """The site rows of a two-Spark arc: each rank's host, management address and the fabric facing its partner.

    Each row holds ``host``, ``management_ip``, ``fabric_ip`` (the primary
    function's address, the model's host IP and the API rank's master
    address), ``interface`` (that function's netdev, for NCCL and Gloo
    sockets) and ``hcas`` (the primary and secondary RDMA devices, in the
    order a pair lists ``rocep1s0f0, roceP2p1s0f0``). The first Spark faces
    the second through its port 0, the second faces the first through its
    port 1. Raises ValueError when the recorded addresses show that the two
    Sparks do not share that cable.
    """
    hosts = cluster["plan"]["spec"]["hosts"]
    layout = layout_of(cluster)
    if placement is None or len(placement) != 2 or not is_arc(layout, placement) or len(hosts) != layout["size"]:
        raise ValueError("A two-Spark arc needs two cable neighbors of the cluster's setup plan")
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
                                 "sudo sparkring setup to review the fabric.")
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

    A group's API rank is its first position. The API rank of every Spark and
    of an arc from Node A is Node A, which serves at the cluster's recorded
    LAN address (``api_address``, set by setup through Node A's uplink) or
    else at its management address; this function then returns ``(None,
    None)`` and the site keeps that rule. On a cluster whose management
    addresses are the LAN addresses (no ``api_address`` recorded), another
    first Spark serves at its management address, also ``(None, None)``.
    Otherwise it serves at its own LAN address (``lan_address``); a first
    Spark without one is reached at its administration address, which only
    Node A routes, and the note says so.
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
    """The lowercase instance name ``sparkring up PROFILE --on`` gives a new arc's deployment: ``on-2-3``, ``on-4-5-6-7``."""
    return "on-" + "-".join(str(position) for position in placement)


def from_lock(lock):
    """The placement a deployment lock records, or None for a deployment on every Spark."""
    value = (lock.get("site") or {}).get("placement")
    return tuple(value) if value else None



# Slots ------------------------------------------------------------------------
#
# A cluster has one slot per placement that has recorded a deployment: the
# whole fabric, and each arc under ``slots/``. Each slot records its own
# active deployment and its own model switch. Slots that share a Spark never
# serve at the same time: a switch into one stops the models of the others
# (``displaced``).

def slot_name(placement):
    """``0-1``, ``4-5-6-7``: the directory of an arc's slot under ``slots/``."""
    return "-".join(str(position) for position in placement)


def slot_directory(state_root, placement):
    """The directory holding the slot's ``active.json`` and ``transaction.json``."""
    root = Path(state_root)
    return root if placement is None else root / "slots" / slot_name(placement)


def _slot_of(name):
    try:
        value = tuple(int(part) for part in name.split("-"))
    except ValueError:
        return None
    if len(value) < 2 or len(set(value)) != len(value) or any(not 0 <= part < fabric_layout.MAX_SPARKS
                                                               for part in value) or slot_name(value) != name:
        return None
    return value


def slots(state_root):
    """Every slot that holds records: the whole fabric (None) first, then each arc by its first position."""
    root = Path(state_root) / "slots"
    found = [_slot_of(path.name) for path in root.iterdir() if path.is_dir()] if root.is_dir() else []
    return [None, *sorted((slot for slot in found if slot), key=lambda slot: (slot[0], len(slot), slot))]


def overlap(a, b):
    """Whether two placements share a Spark; None holds every Spark."""
    return a is None or b is None or bool(set(a) & set(b))


def conflicting(state_root, placement):
    """The slots that cannot serve while ``placement`` serves: every other slot that shares a Spark with it."""
    return [slot for slot in slots(state_root) if slot != placement and overlap(slot, placement)]


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
    for slot in conflicting(state_root, placement):
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


def actives(state_root):
    """``{slot: deployment directory}`` of every slot whose record names an existing deployment."""
    result = {}
    for slot in slots(state_root):
        directory = recorded(state_root, slot)
        if exists(directory):
            result[slot] = directory
    return result


def running(state_root):
    """``{slot: deployment directory}`` of every slot whose active deployment may run a model."""
    return {slot: directory for slot, directory in actives(state_root).items() if not stopped(directory)}


def displaced(state_root, placement):
    """The deployments of the conflicting slots that a switch into ``placement`` stops, as directories.

    Only deployments that may run count; a stopped one is only forgotten.
    """
    found = []
    for slot in conflicting(state_root, placement):
        directory = recorded(state_root, slot)
        if exists(directory) and not stopped(directory):
            found.append(Path(directory))
    return found


PENDING = ("stopping-previous", "starting", "verifying", "recovering-previous", "needs-attention")


def unfinished_switches(state_root, placement):
    """``[(slot, record)]`` of conflicting slots whose model switch stopped before it finished."""
    result = []
    for slot in conflicting(state_root, placement):
        value = journal(state_root, slot)
        if value and value.get("state") in PENDING:
            result.append((slot, value))
    return result


def aligned(layout, nodes):
    """The arcs of ``nodes`` Sparks that tile the fabric from Node A: ``(0, 1), (2, 3)``, or ``(0-3), (4-7)``."""
    size = layout["size"]
    return [arc(layout, start, nodes) for start in range(0, size - nodes + 1, nodes)]


def choose(state_root, profile, layout, nodes):
    """The arc for a profile of fewer Sparks than the fabric, installed without ``--on``.

    The candidates tile the fabric from Node A (``aligned``): the halves of a
    four-Spark ring for two Sparks, positions 0-3 and 4-7 of an
    eight-Spark ring for four. A candidate is free when no model may run on
    any of its Sparks. The one free candidate is chosen; otherwise
    ValueError names every candidate and what holds it.
    """
    candidates = [candidate for candidate in aligned(layout, nodes) if unsupported(layout, candidate) is None]
    serving = running(state_root)
    holding = {}
    for candidate in candidates:
        holders = [directory for slot, directory in serving.items() if overlap(slot, candidate)]
        holding[candidate] = holders
    free = [candidate for candidate in candidates if not holding[candidate]]
    if len(free) == 1:
        return free[0]
    lines = [f"{flag(candidate)}: {text(candidate)}, " + ("free" if not holding[candidate] else
                                                         "serving " + ", ".join(map(profile_of, holding[candidate])))
             for candidate in candidates]
    part = "a half" if len(candidates) == 2 and 2 * nodes == layout["size"] else "its Sparks"
    noun = "ring" if layout["shape"] == fabric_layout.CYCLE else "line"
    choices = " or ".join(flag(candidate) for candidate in candidates)
    error = ValueError(f"{profile} uses {WORDS[nodes]} Sparks and this {noun} has {WORDS[layout['size']]}. Choose "
                       f"{part} with {choices}. Nothing has been changed.")
    error.details = {"lines": lines}
    raise error


def part_name(layout, nodes):
    """``half`` when two arcs of ``nodes`` Sparks tile the fabric, else ``group of Sparks``."""
    return "half" if 2 * nodes == layout["size"] else "group of Sparks"


def profile_of(directory):
    """The profile of the deployment in ``directory``, or its directory name when its lock cannot be read."""
    try:
        return installer.read(Path(directory) / "deployment.lock.json")["selection"]["profile"]
    except (OSError, ValueError, KeyError, TypeError):
        return Path(directory).name
