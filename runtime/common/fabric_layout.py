"""Geometry of a SparkRing fabric: positions, cables, port roles and relay distances for 2 to 8 DGX Sparks.

A fabric is one of three layouts of ``size`` Sparks, each Spark at a
*position* from 0 (Node A) to ``size - 1``:

- ``pair``: two Sparks with one cable between their ports 0.
- ``path``: 3 to 8 Sparks in a line. Cable ``e`` runs from position ``e``'s
  port 0 to position ``e + 1``'s port 1, so Node A's port 1 and the last
  Spark's port 0 are free.
- ``cycle``: 3 to 8 Sparks in a loop. Cable ``e`` runs from position ``e``'s
  port 0 to position ``(e + 1) % size``'s port 1.

Each ConnectX port carries two network functions (Socket Direct): the
*primary* (``enp1s0f<port>np<port>``, ``rocep1s0f<port>``) and the
*secondary* (``enP2p1s0f<port>np<port>``, ``roceP2p1s0f<port>``). A function's
*role* names its port's direction and its function: ``cw_*`` is port 0,
which leads to the next position, and ``ccw_*`` is port 1, which leads to the
previous one. A pair uses only the ``cw_*`` roles on both Sparks.

Addresses: cable ``e`` uses the ``2e``-th /24 of the fabric supernet for its
primary function and the ``(2e + 1)``-th for its secondary; the end at
position ``e`` is ``.1`` and the other end ``.2``. A /21 supernet holds four
cables and a /20 eight.

Relays: a Spark reaches a Spark that is not its cable neighbor through the
Sparks between them, which forward in ConnectX hardware. ``hops`` counts the
cables on the shortest way and ``hops - 1`` is the number of relays.

This module uses only the Python standard library, so programs outside the
installer (the SIRCL session layer, the serving containers) can import or
vendor it with ``fabric_document``.
"""
import ipaddress

PAIR, PATH, CYCLE = "pair", "path", "cycle"
SHAPES = (PAIR, PATH, CYCLE)
MIN_SPARKS = 2
MAX_SPARKS = 8
# The port roles in their canonical order, with each role's port and function.
ROLES = ("cw_primary", "cw_secondary", "ccw_primary", "ccw_secondary")
FUNCTIONS = ("primary", "secondary")
# DGX OS names of a Spark's ConnectX fabric functions by role.
DEVICES = {"cw_primary": "rocep1s0f0", "cw_secondary": "roceP2p1s0f0",
           "ccw_primary": "rocep1s0f1", "ccw_secondary": "roceP2p1s0f1"}
NETDEVS = {"cw_primary": "enp1s0f0np0", "cw_secondary": "enP2p1s0f0np0",
           "ccw_primary": "enp1s0f1np1", "ccw_secondary": "enP2p1s0f1np1"}
# The default fabric supernets: a /21 holds four cables, a /20 eight.
NARROW_CIDR = "198.18.0.0/21"
WIDE_CIDR = "198.18.0.0/20"
NARROW_CABLES = 4
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"}


def layout(shape, size):
    """``{"shape", "size"}`` after checking that SparkRing supports it; ValueError otherwise."""
    if shape not in SHAPES or type(size) is not int:
        raise ValueError(f"Unknown fabric layout {shape!r} of {size!r} Sparks")
    if shape == PAIR and size != 2 or shape != PAIR and not 3 <= size <= MAX_SPARKS:
        raise ValueError(f"A {shape} of {size} Sparks is not a supported fabric layout; "
                         f"SparkRing supports a pair, or a path or cycle of 3 to {MAX_SPARKS} Sparks")
    return {"shape": shape, "size": size}


def checked(value):
    """A layout document ``{"shape", "size"}`` validated by ``layout``."""
    if not isinstance(value, dict) or set(value) != {"shape", "size"}:
        raise ValueError("A fabric layout is {\"shape\", \"size\"}")
    return layout(value["shape"], value["size"])


def legacy(size):
    """The layout of a record that names only its Spark count: a pair or a four-Spark cycle."""
    if size == 2:
        return layout(PAIR, 2)
    if size == 4:
        return layout(CYCLE, 4)
    raise ValueError(f"A record of {size} Sparks must name its fabric layout")


def legacy_or_none(size):
    """``legacy(size)`` for two or four Sparks, else None."""
    return legacy(size) if size in (2, 4) else None


def name(value):
    """``pair``, ``path-5`` or ``cycle-8``."""
    return PAIR if value["shape"] == PAIR else f"{value['shape']}-{value['size']}"


def cable_count(value):
    return {PAIR: 1, PATH: value["size"] - 1, CYCLE: value["size"]}[value["shape"]]


def cables(value):
    """``[(cable, (position, port), (position, port))]``: each cable's first and second end."""
    if value["shape"] == PAIR:
        return [(0, (0, 0), (1, 0))]
    size = value["size"]
    return [(e, (e, 0), ((e + 1) % size, 1)) for e in range(cable_count(value))]


def port_role(port, function):
    return ("cw_" if port == 0 else "ccw_") + function


def role_port(role):
    return 0 if role.startswith("cw_") else 1


def role_function(role):
    return role.split("_", 1)[1]


def roles(value, position):
    """The roles of ``position``'s cabled functions, in ROLES order."""
    size = value["size"]
    if not 0 <= position < size:
        raise ValueError(f"Position {position} is outside a fabric of {size} Sparks")
    if value["shape"] == PAIR:
        return list(ROLES[:2])
    if value["shape"] == PATH and position == 0:
        return list(ROLES[:2])
    if value["shape"] == PATH and position == size - 1:
        return list(ROLES[2:])
    return list(ROLES)


def cable_of(value, position, port):
    """The cable at ``position``'s ``port``, or None for a free port."""
    for cable, *ends in cables(value):
        if (position, port) in ends:
            return cable
    return None


def peer(value, position, port):
    """``(position, port)`` at the other end of ``position``'s ``port``, or None for a free port."""
    for _, a, b in cables(value):
        if a == (position, port):
            return b
        if b == (position, port):
            return a
    return None


def neighbors(value, position):
    """The positions one cable away, ``port 0``'s first."""
    found = []
    for port in (0, 1):
        far = peer(value, position, port)
        if far is not None and far[0] not in found:
            found.append(far[0])
    return found


def hops(value, a, b):
    """Cables on the shortest way from position ``a`` to ``b``."""
    if value["shape"] == CYCLE:
        distance = abs(a - b) % value["size"]
        return min(distance, value["size"] - distance)
    return abs(a - b)


def max_relays(value):
    """The most relays any route crosses: 0 on a pair or a cycle of three."""
    if value["shape"] == PAIR:
        return 0
    if value["shape"] == PATH:
        return value["size"] - 2
    return value["size"] // 2 - 1


def relayed(value):
    """Whether some Spark reaches another through relays, which need the ConnectX hairpin setting."""
    return max_relays(value) > 0


def forwards(value, position):
    """Whether ``position`` relays traffic between its two cables."""
    return value["shape"] != PAIR and len(roles(value, position)) == 4


def capacity(cidr):
    """The number of cables a fabric supernet holds: two /24 subnets each."""
    network = ipaddress.IPv4Network(cidr, strict=True)
    if network.prefixlen > 24:
        return 0
    return 2 ** (24 - network.prefixlen) // 2


def default_cidr(value):
    """The /21 for at most four cables, else the /20; both start at the same /24, so cables 0-3 keep their subnets."""
    return NARROW_CIDR if cable_count(value) <= NARROW_CABLES else WIDE_CIDR


def subnet(cidr, cable, function):
    """The /24 of ``cable``'s ``function``."""
    networks = ipaddress.IPv4Network(cidr, strict=True).subnets(new_prefix=24)
    index = 2 * cable + FUNCTIONS.index(function)
    for number, network in enumerate(networks):
        if number == index:
            return network
    raise ValueError(f"The fabric supernet {cidr} has no subnet for cable {cable}")


def address(cidr, value, position, role):
    """``position``'s planned IPv4 interface address of ``role``, such as ``198.18.4.1/24``."""
    port = role_port(role)
    cable = cable_of(value, position, port)
    if cable is None:
        raise ValueError(f"Position {position} has no cable on port {port}")
    first = cables(value)[cable][1]
    network = subnet(cidr, cable, role_function(role))
    return f"{network.network_address + (1 if first == (position, port) else 2)}/24"
