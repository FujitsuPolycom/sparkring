"""Load and validate ``sparkring-fabric/v1``, the cluster-wide description of a SparkRing fabric.

``sudo sparkring setup`` writes one document per cluster to
``/var/lib/sparkring/controller/fabric.json`` on Node A and the same bytes to
``/etc/sparkring/fabric/topology.json`` on every Spark. It records what setup
discovered and configured, so every program that depends on the cabling
derives it from here instead of a fixed table:

- ``shape`` and ``size``: the layout (``runtime/common/fabric_layout.py``),
  with ``head`` (Node A's position, always 0), ``cluster``, ``fabric_cidr``
  and ``addressing``: ``planned`` when every address follows the layout's
  rule in ``fabric_cidr``, ``preserved`` when setup kept addresses the Sparks
  already had (each cable function still has its own /24, shared by its two
  ends).
- ``positions``: per Spark, in position order, its SparkRing node identity,
  hostname, management address and both ports, ``"0"`` and ``"1"``. Each
  port lists its cable and far end (both null for a free port) and its two
  network functions, ``primary`` and ``secondary``, with their discovered
  network interface (``netdev``), RDMA device (``rdma``), MAC, port role and
  address (null on a free port).
- ``cables``: per cable, its two ends, its two /24 subnets, the Sparks that
  observed it and its last measured health.
- ``hairpin``: whether relays need the ConnectX hairpin setting, and its values.
- ``relays``: the relay plan's identity (``sparkring-relay-plan/v1``) and
  whether boot units restore it, or null when no relay table is installed.
- ``transports``: the collective transports the fabric can carry.
- ``verified``: setup's verification of the fabric before it wrote the
  document.

``id`` identifies the cabling: ``sha256:`` and the SHA-256 of ``positions``
and ``cables`` with every ``address``, ``lan_address``, ``mac``,
``hostname``, ``health`` and ``seen_from`` field removed, serialized with
sorted keys, two-space indent and a final LF. Addresses, names and health can
change without changing the identity; a changed Spark, port or cable changes it.

Programs outside the installer use ``load`` (or ``validate`` on a parsed
document) and ``devices``, which returns each RDMA device's role and the
neighbor it serves. SIRCL reads a document that ``SIRCL_FABRIC_DOCUMENT``
names (``spark_transport/sircl/sparkring_sircl/routes.py``,
``roles_from_fabric_document``): it takes ``rdma`` and ``netdev`` of
``positions[].ports.<port>.functions.<function>`` and needs every Spark to
name its functions alike (``uniform_names``). This module uses only the
Python standard library and ``fabric_layout``.
"""
import copy
import hashlib
import ipaddress
import json
from pathlib import Path
import re

from runtime.common import fabric_layout

SCHEMA = "sparkring-fabric/v1"
RELAY_SCHEMA = "sparkring-relay-plan/v1"
# Node A's copy and every Spark's copy; both hold the same bytes.
CONTROLLER_PATH = "/var/lib/sparkring/controller/fabric.json"
HOST_PATH = "/etc/sparkring/fabric/topology.json"
TRANSPORTS = ("sircl", "prepared")
IDENTITY_EXCLUDED = ("address", "lan_address", "mac", "hostname", "health", "seen_from")
HEALTH_STATES = ("healthy", "degraded", "failed", "skipped", "unmeasured")
_MAC = re.compile(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}")
_NAME = re.compile(r"[A-Za-z0-9_.:-]{1,64}")
_DIGEST = re.compile(r"[0-9a-f]{64}")


class FabricDocumentError(ValueError):
    """The document is not a valid ``sparkring-fabric/v1`` document."""


def encoded(document):
    """The canonical serialization: sorted keys, two-space indent, LF; also the bytes setup writes."""
    return json.dumps(document, sort_keys=True, indent=2) + "\n"


def _stripped(value):
    if isinstance(value, dict):
        return {key: _stripped(item) for key, item in value.items() if key not in IDENTITY_EXCLUDED}
    if isinstance(value, list):
        return [_stripped(item) for item in value]
    return value


def identity(document):
    """``sha256:<hex>`` of the document's positions and cables without addresses, names and health."""
    core = _stripped({"positions": document["positions"], "cables": document["cables"]})
    return "sha256:" + hashlib.sha256(encoded(core).encode()).hexdigest()


def digest(document):
    """The SHA-256 of the document's serialization, as each Spark's copy is compared."""
    return hashlib.sha256(encoded(document).encode()).hexdigest()


def _require(condition, text):
    if not condition:
        raise FabricDocumentError(text)


def _interface(value, label):
    try:
        address = ipaddress.IPv4Interface(value)
    except (TypeError, ValueError):
        raise FabricDocumentError(f"{label}: {value!r} is not an IPv4 interface address") from None
    _require(address.network.prefixlen == 24, f"{label}: fabric addresses use /24 subnets")
    return address


def _check_position(document, layout, index, row, devices, netdevs):
    label = f"position {index}"
    _require(isinstance(row, dict) and row.get("position") == index, f"{label}: positions are listed in order")
    _require(isinstance(row.get("node_id"), str) and row["node_id"], f"{label}: node_id is required")
    _require(isinstance(row.get("management"), dict) and isinstance(row["management"].get("interface"), str),
             f"{label}: management interface is required")
    ports = row.get("ports")
    _require(isinstance(ports, dict) and sorted(ports) == ["0", "1"], f"{label}: ports \"0\" and \"1\" are required")
    cabled = {fabric_layout.role_port(role) for role in fabric_layout.roles(layout, index)}
    for port_text, port in ports.items():
        number = int(port_text)
        where = f"{label} port {number}"
        far = fabric_layout.peer(layout, index, number) if number in cabled else None
        _require(isinstance(port, dict) and port.get("cable") == (
            fabric_layout.cable_of(layout, index, number) if far else None), f"{where}: cable differs from the layout")
        _require(port.get("peer") == ({"position": far[0], "port": far[1]} if far else None),
                 f"{where}: far end differs from the layout")
        functions = port.get("functions")
        _require(isinstance(functions, dict) and sorted(functions) == sorted(fabric_layout.FUNCTIONS),
                 f"{where}: primary and secondary functions are required")
        for function, value in functions.items():
            role = fabric_layout.port_role(number, function)
            _require(isinstance(value, dict) and value.get("role") == role, f"{where} {function}: role must be {role}")
            for key in ("netdev", "rdma"):
                _require(isinstance(value.get(key), str) and _NAME.fullmatch(value[key]),
                         f"{where} {function}: {key} is required")
            _require(value.get("mac") is None and not far or isinstance(value.get("mac"), str)
                     and _MAC.fullmatch(value["mac"]), f"{where} {function}: a lowercase MAC is required")
            if far is None:
                _require(value.get("address") is None, f"{where} {function}: a free port has no fabric address")
            else:
                address = _interface(value.get("address"), f"{where} {function}")
                if document.get("addressing") == "planned":
                    subnet = fabric_layout.subnet(document["fabric_cidr"], port["cable"], function)
                    _require(address.network == subnet,
                             f"{where} {function}: {address} is not in its cable's subnet {subnet}")
            _require(value["rdma"] not in devices, f"{label}: RDMA device {value['rdma']} appears more than once")
            _require(value["netdev"] not in netdevs, f"{label}: network interface {value['netdev']} appears more than once")
            devices.add(value["rdma"])
            netdevs.add(value["netdev"])


def validate(document):
    """``document`` after checking it against the schema and its own layout; FabricDocumentError otherwise.

    Every position lists exactly the ports its layout cables, each with its
    cable, far end and two functions; every RDMA device and network interface
    of a Spark appears once, on both ports, a free one with no cable, far
    end or address; the two ends of each cable function share its
    own /24 (the layout's subnet when ``addressing`` is ``planned``); ``id``
    matches the content.
    """
    _require(isinstance(document, dict) and document.get("schema") == SCHEMA, f"expected {SCHEMA}")
    try:
        layout = fabric_layout.layout(document.get("shape"), document.get("size"))
    except ValueError as error:
        raise FabricDocumentError(str(error)) from None
    _require(document.get("head") == 0, "head must be position 0 (Node A)")
    _require(isinstance(document.get("cluster"), str) and document["cluster"], "cluster name is required")
    try:
        cidr = str(ipaddress.IPv4Network(document.get("fabric_cidr"), strict=True))
    except (TypeError, ValueError):
        raise FabricDocumentError("fabric_cidr must be an IPv4 network") from None
    _require(document.get("addressing") in ("planned", "preserved"), "addressing must be planned or preserved")
    _require(document["addressing"] == "preserved" or fabric_layout.capacity(cidr) >= fabric_layout.cable_count(layout),
             f"fabric_cidr {cidr} holds fewer than {fabric_layout.cable_count(layout)} cables")
    positions = document.get("positions")
    _require(isinstance(positions, list) and len(positions) == layout["size"],
             f"positions must list {layout['size']} Sparks")
    nodes = set()
    for index, row in enumerate(positions):
        _check_position(document, layout, index, row, set(), set())
        _require(row["node_id"] not in nodes, f"node {row['node_id']} appears at more than one position")
        nodes.add(row["node_id"])
    cables = document.get("cables")
    seen_subnets = set()
    _require(isinstance(cables, list) and len(cables) == fabric_layout.cable_count(layout),
             f"cables must list {fabric_layout.cable_count(layout)} cables")
    for (number, first, second), cable in zip(fabric_layout.cables(layout), cables, strict=True):
        where = f"cable {number}"
        ends = [{"position": first[0], "port": first[1]}, {"position": second[0], "port": second[1]}]
        _require(isinstance(cable, dict) and cable.get("cable") == number and cable.get("ends") == ends,
                 f"{where}: ends differ from the layout")
        subnets = {}
        for function in fabric_layout.FUNCTIONS:
            a = ipaddress.IPv4Interface(positions[first[0]]["ports"][str(first[1])]["functions"][function]["address"])
            b = ipaddress.IPv4Interface(positions[second[0]]["ports"][str(second[1])]["functions"][function]["address"])
            _require(a.network == b.network and a.ip != b.ip,
                     f"{where} {function}: its two ends need two addresses in one /24")
            _require(str(a.network) not in seen_subnets, f"{where} {function}: {a.network} is used by another cable")
            seen_subnets.add(str(a.network))
            subnets[function] = str(a.network)
        _require(cable.get("subnets") == subnets, f"{where}: subnets must be {subnets}")
        health = cable.get("health")
        _require(health is None or isinstance(health, dict) and health.get("state") in HEALTH_STATES,
                 f"{where}: health state must be one of {', '.join(HEALTH_STATES)}")
    hairpin = document.get("hairpin")
    _require(isinstance(hairpin, dict) and hairpin.get("required") is fabric_layout.relayed(layout),
             "hairpin.required must say whether the layout relays")
    relays = document.get("relays")
    if relays is not None:
        _require(isinstance(relays, dict) and relays.get("schema") == RELAY_SCHEMA
                 and relays.get("max_relays") == fabric_layout.max_relays(layout)
                 and isinstance(relays.get("persistent"), bool)
                 and isinstance(relays.get("plan_sha256"), str) and _DIGEST.fullmatch(relays["plan_sha256"]),
                 f"relays must name a {RELAY_SCHEMA} plan with max_relays {fabric_layout.max_relays(layout)}")
    transports = document.get("transports")
    _require(isinstance(transports, list) and set(transports) <= set(TRANSPORTS)
             and len(set(transports)) == len(transports), f"transports may list {', '.join(TRANSPORTS)}")
    _require(document.get("id") == identity(document), "id does not match the positions and cables")
    return document


def load(path=HOST_PATH):
    """The validated document at ``path`` (default: this Spark's copy)."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FabricDocumentError(f"{path} does not exist; sudo sparkring setup writes it") from None
    except (OSError, ValueError) as error:
        raise FabricDocumentError(f"{path} cannot be read: {error}") from None
    return validate(document)


def layout(document):
    return fabric_layout.layout(document["shape"], document["size"])


def position_of(document, node_id):
    """The position of the Spark with SparkRing node identity ``node_id``."""
    for row in document["positions"]:
        if row["node_id"] == node_id:
            return row["position"]
    raise FabricDocumentError(f"node {node_id} is not part of fabric {document['id']}")


def devices(document, position):
    """Each RDMA device of ``position`` with its role and the neighbor it serves.

    Returns ``{rdma: {"netdev", "role", "port", "function", "cable",
    "address", "neighbor"}}`` in role order (``fabric_layout.ROLES``).
    ``role`` is ``cw_*`` for port 0, which leads to the next position, and
    ``ccw_*`` for port 1, which leads to the previous one; ``address`` is the
    IPv4 address without its prefix. ``neighbor`` is ``{"position", "port",
    "rdma", "netdev", "address"}`` of the far function on the same cable, and
    ``cable``, ``address`` and ``neighbor`` are None on a free port. The names
    are the ones discovered on the Spark; on DGX OS they are
    ``fabric_layout.DEVICES`` and ``fabric_layout.NETDEVS``.
    """
    row = document["positions"][position]
    result = {}
    for role in fabric_layout.ROLES:
        port, function = fabric_layout.role_port(role), fabric_layout.role_function(role)
        entry = row["ports"][str(port)]
        local = entry["functions"][function]
        far = entry["peer"]
        neighbor = None
        if far is not None:
            remote = document["positions"][far["position"]]["ports"][str(far["port"])]["functions"][function]
            neighbor = {"position": far["position"], "port": far["port"], "rdma": remote["rdma"],
                        "netdev": remote["netdev"], "address": str(ipaddress.IPv4Interface(remote["address"]).ip)}
        result[local["rdma"]] = {
            "netdev": local["netdev"], "role": role, "port": port, "function": function, "cable": entry["cable"],
            "address": None if local["address"] is None else str(ipaddress.IPv4Interface(local["address"]).ip),
            "neighbor": neighbor}
    return result


def default_names(document):
    """Whether every function uses the DGX OS device and interface names of its role."""
    for row in document["positions"]:
        for entry in row["ports"].values():
            for value in entry["functions"].values():
                if (fabric_layout.DEVICES[value["role"]] != value["rdma"]
                        or fabric_layout.NETDEVS[value["role"]] != value["netdev"]):
                    return False
    return True


def uniform_names(document):
    """Whether every Spark names each role's device and interface alike, as SIRCL's loader requires."""
    seen = {}
    for row in document["positions"]:
        for entry in row["ports"].values():
            for value in entry["functions"].values():
                if seen.setdefault(value["role"], (value["rdma"], value["netdev"])) != (value["rdma"], value["netdev"]):
                    return False
    return True


def without_identity(document):
    """A copy without ``id``; ``identity`` of the result gives the value to record."""
    value = copy.deepcopy(document)
    value.pop("id", None)
    return value
