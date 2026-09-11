"""Plan persistent ConnectX networking from a reviewed four-host inventory.

Status: implemented for NetworkManager. This module does not run commands.
Plans preserve previous connection UUIDs and isolate disruptive driver reloads.
"""

from __future__ import annotations

import ipaddress
import re
import uuid
from collections.abc import Mapping
from pathlib import PurePosixPath

from spark_transport.fabric.cx7_hairpin_diagonal.fabric import RANK_COUNT


PLAN_SCHEMA = "sparkring-deploy-network-plan/v1"
INVENTORY_SCHEMA = "sparkring-deploy-host-inventory/v1"
ROLES = ("cw_primary", "cw_secondary", "ccw_primary", "ccw_secondary")
_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,63}\Z")
_NETDEV = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.:-]{0,14}\Z")
_BDF = re.compile(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]\Z")


class NetworkPlanError(ValueError):
    """Inventory cannot support the requested data-network changes."""


def _object(value, label):
    if not isinstance(value, Mapping):
        raise NetworkPlanError(f"{label} must be an object")
    return value


def _name(value, label, pattern=_NAME):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise NetworkPlanError(f"{label} is invalid")
    return value


def _uuid(value, label):
    try:
        return str(uuid.UUID(value))
    except (AttributeError, TypeError, ValueError):
        raise NetworkPlanError(f"{label} must be a connection UUID") from None


def _path(value, label):
    if not isinstance(value, str) or not value.startswith("/"):
        raise NetworkPlanError(f"{label} must be an absolute host path")
    path = PurePosixPath(value)
    if str(path) != value or ".." in path.parts or len(path.parts) < 4:
        raise NetworkPlanError(f"{label} must identify a dedicated directory")
    if any(ch.isspace() or ch in "\0" for ch in value):
        raise NetworkPlanError(f"{label} has unsupported characters")
    if path.parts[1:3] not in (("var", "tmp"), ("var", "lib"), ("srv", "sparkring")):
        raise NetworkPlanError(
            f"{label} must be under /var/tmp, /var/lib, or /srv/sparkring"
        )
    return value


def _address(value, label):
    try:
        address = ipaddress.ip_interface(value)
    except (TypeError, ValueError):
        raise NetworkPlanError(
            f"{label} must be an IPv4 host address with /24"
        ) from None
    if address.version != 4 or address.network.prefixlen != 24:
        raise NetworkPlanError(
            f"{label} must use /24; /32 belongs in the fabric endpoint locator"
        )
    if (
        address.ip
        in (address.network.network_address, address.network.broadcast_address)
        or address.ip.is_loopback
        or address.ip.is_multicast
        or address.ip.is_unspecified
    ):
        raise NetworkPlanError(f"{label} must identify a usable unicast endpoint")
    return address


def _command(host, identifier, argv, *, risk="read-only", **fields):
    return {
        "id": identifier,
        "host": host,
        "argv": argv,
        "risk": risk,
        "requires_stopped_models": risk != "read-only",
        "requires_no_rdma_users": risk != "read-only",
        **fields,
    }


def _read(host, identifier, argv, expected=None):
    return _command(
        host, identifier, argv, **({"expected": expected} if expected else {})
    )


def _prepare_spec(spec):
    root = _object(spec, "spec")
    owner = _name(root.get("owner"), "owner")
    settings = _object(root.get("network"), "network")
    if settings.get("backend") != "NetworkManager":
        raise NetworkPlanError(
            "Only an explicitly selected NetworkManager backend is supported"
        )
    for key, value in {
        "mtu": 9000,
        "gid_index": 3,
        "hairpin_num_queues": 4,
        "hairpin_queue_size": 1024,
    }.items():
        if settings.get(key, value) != value:
            raise NetworkPlanError(f"The managed mesh requires {key}={value}")
    hosts = root.get("hosts")
    if not isinstance(hosts, list) or len(hosts) != RANK_COUNT:
        raise NetworkPlanError("hosts must contain exactly four ranks")
    result = []
    for item in hosts:
        host = _object(item, "host")
        rank = host.get("rank")
        if type(rank) is not int or rank not in range(RANK_COUNT):
            raise NetworkPlanError("host rank must be 0, 1, 2, or 3")
        ssh = host.get("host")
        if (
            not isinstance(ssh, str)
            or not ssh
            or ssh.startswith("-")
            or any(c.isspace() for c in ssh)
        ):
            raise NetworkPlanError(
                f"rank {rank} host must be an SSH alias or user@host"
            )
        management = _name(host.get("management_netdev"), "management_netdev", _NETDEV)
        ports = host.get("data_interfaces")
        if not isinstance(ports, list) or len(ports) != 4:
            raise NetworkPlanError(f"{ssh}: four data_interfaces are required")
        parsed = []
        for port in ports:
            port = _object(port, "data interface")
            if port.get("role") not in ROLES:
                raise NetworkPlanError(
                    f"{ssh}: each data interface needs a clockwise/counter-clockwise primary/secondary role"
                )
            netdev = _name(port.get("netdev"), "data netdev", _NETDEV)
            if netdev == management:
                raise NetworkPlanError(
                    f"{ssh}: data netdev {netdev} is the management interface"
                )
            parsed.append(
                {
                    **port,
                    "netdev": netdev,
                    "rdma_device": _name(port.get("rdma_device"), "RDMA device"),
                    "address": str(
                        _address(port.get("address"), f"{ssh}/{netdev} address")
                    ),
                }
            )
        for field in ("role", "netdev", "rdma_device"):
            if len({port[field] for port in parsed}) != 4:
                raise NetworkPlanError(
                    f"{ssh}: data interface {field} values must be distinct"
                )
        parsed.sort(key=lambda port: ROLES.index(port["role"]))
        result.append(
            {
                **host,
                "host": ssh,
                "data_interfaces": parsed,
                "backup_dir": _path(host.get("backup_dir"), f"{ssh} backup_dir"),
            }
        )
    result.sort(key=lambda host: host["rank"])
    if [host["rank"] for host in result] != list(range(RANK_COUNT)):
        raise NetworkPlanError("hosts must contain ranks 0, 1, 2, and 3 once each")
    if len({host["host"] for host in result}) != RANK_COUNT:
        raise NetworkPlanError("SSH hosts must be distinct")
    _cables(result)
    owned = root.get("owned_connection_uuids", [])
    if not isinstance(owned, list):
        raise NetworkPlanError(
            "owned_connection_uuids must be a list from the execution receipt"
        )
    return owner, result, {_uuid(value, "owned connection") for value in owned}


def _cables(hosts):
    """Two functions share a physical cable but keep separate IPv4 subnets."""
    endpoints = [p for host in hosts for p in host["data_interfaces"]]
    if len({str(ipaddress.ip_interface(p["address"]).ip) for p in endpoints}) != 16:
        raise NetworkPlanError("Every data function needs a distinct IPv4 address")
    subnets = set()
    for rank, host in enumerate(hosts):
        ports = {p["role"]: p for p in host["data_interfaces"]}
        peer = {p["role"]: p for p in hosts[(rank + 1) % RANK_COUNT]["data_interfaces"]}
        for function in ("primary", "secondary"):
            network = ipaddress.ip_interface(ports[f"cw_{function}"]["address"]).network
            other = ipaddress.ip_interface(peer[f"ccw_{function}"]["address"]).network
            if network != other:
                raise NetworkPlanError(
                    f"Cable {rank}-{(rank + 1) % RANK_COUNT} {function} endpoints must share a /24"
                )
            if network in subnets:
                raise NetworkPlanError("Each cable function needs its own /24 subnet")
            subnets.add(network)


def _management(host, inventory):
    ssh, netdev = host["host"], host["management_netdev"]
    management = _object(inventory.get("management"), f"{ssh} management inventory")
    if management.get("error") or management.get("interface") != netdev:
        raise NetworkPlanError(f"{ssh}: management route does not match {netdev}")
    route = management.get("route_to_controller")
    if not isinstance(route, Mapping) or route.get("dev") != netdev:
        raise NetworkPlanError(
            f"{ssh}: an independent management return path must be recorded"
        )
    try:
        destination = ipaddress.ip_interface(route["dst"])
    except (KeyError, TypeError, ValueError):
        raise NetworkPlanError(
            f"{ssh}: management return destination is missing"
        ) from None
    if destination.version != 4:
        raise NetworkPlanError(f"{ssh}: the management return path must use IPv4")
    routes = inventory.get("routes")
    if not isinstance(routes, list):
        raise NetworkPlanError(f"{ssh}: all IPv4 routes must be inventoried")
    data = {p["netdev"] for p in host["data_interfaces"]}
    for row in routes:
        if not isinstance(row, Mapping):
            raise NetworkPlanError(f"{ssh}: invalid route inventory")
        destination = row.get("dst", "default")
        if destination in ("default", "0.0.0.0/0"):
            if row.get("dev") in data:
                raise NetworkPlanError(
                    f"{ssh}: a data interface carries a default route"
                )
            continue
        try:
            network = ipaddress.ip_network(destination, strict=False)
        except ValueError:
            raise NetworkPlanError(
                f"{ssh}: invalid IPv4 route {destination!r}"
            ) from None
        if row.get("dev") not in data:
            for port in host["data_interfaces"]:
                if network.overlaps(ipaddress.ip_interface(port["address"]).network):
                    raise NetworkPlanError(
                        f"{ssh}: data subnet overlaps route {destination} on {row.get('dev')}"
                    )
    try:
        address = ipaddress.ip_interface(management["address"]).ip
    except (KeyError, TypeError, ValueError):
        raise NetworkPlanError(f"{ssh}: management IPv4 address is missing") from None
    for port in host["data_interfaces"]:
        if address in ipaddress.ip_interface(port["address"]).network:
            raise NetworkPlanError(
                f"{ssh}: data subnet overlaps the management address"
            )


def _interfaces(host, inventory):
    values = inventory.get("interfaces")
    rdma = inventory.get("rdma")
    if not isinstance(values, list) or not isinstance(rdma, list):
        raise NetworkPlanError(
            f"{host['host']}: interface and RDMA inventory are required"
        )
    found = []
    for port in host["data_interfaces"]:
        interfaces = [v for v in values if v.get("name") == port["netdev"]]
        functions = [v for v in rdma if v.get("device") == port["rdma_device"]]
        if len(interfaces) != 1 or len(functions) != 1:
            raise NetworkPlanError(
                f"{host['host']}: ambiguous or missing mapping for {port['netdev']}"
            )
        interface, function = interfaces[0], functions[0]
        if (
            function.get("netdev") != port["netdev"]
            or function.get("driver") != "mlx5_core"
        ):
            raise NetworkPlanError(
                f"{host['host']}: {port['netdev']} is not the verified mlx5 data function"
            )
        if interface.get("master"):
            raise NetworkPlanError(
                f"{host['host']}: {port['netdev']} is owned by bridge/bond {interface['master']}"
            )
        nm = _object(
            interface.get("network_manager"), "NetworkManager interface inventory"
        )
        if (
            nm.get("error")
            or nm.get("available") is not True
            or nm.get("managed") is not True
        ):
            raise NetworkPlanError(
                f"{host['host']}: NetworkManager does not manage {port['netdev']}"
            )
        _name(function.get("pci_address"), "ConnectX PCI address", _BDF)
        found.append((port, interface, function))
    if len({function["pci_address"] for _, _, function in found}) != 4:
        raise NetworkPlanError(
            f"{host['host']}: four distinct data PCI functions are required"
        )
    return found


def _nm_matches(interface, address):
    nm = interface["network_manager"]
    try:
        addresses = [
            str(ipaddress.ip_interface(value)) for value in interface.get("ipv4", [])
        ]
    except (ValueError, TypeError):
        return False
    return (
        addresses == [address]
        and interface.get("mtu") == 9000
        and nm.get("ipv4_addresses") == [address]
        and type(nm.get("ethernet_mtu")) is int and nm["ethernet_mtu"] == 9000
        and nm.get("connection_uuid")
        and nm.get("ipv4_method") == "manual"
        and nm.get("ipv4_never_default") is True
        and nm.get("ipv4_ignore_auto_dns") is True
        and nm.get("ipv6_method") in ("link-local", "auto")
        and nm.get("ipv6_never_default") is True
        and nm.get("autoconnect") is True
    )


def _connection(owner, host, port, interface, connections, owned):
    ssh = host["host"]
    nm = interface["network_manager"]
    previous = nm.get("connection_uuid")
    if previous:
        previous = _uuid(previous, f"{ssh} previous connection")
        if (not isinstance(nm.get('ipv4_addresses'), list)
                or type(nm.get('ethernet_mtu')) is not int):
            raise NetworkPlanError(
                f"{ssh}: saved addresses or MTU are unavailable on {port['netdev']}; rediscover before planning")
    record = {
        "netdev": port["netdev"],
        "address": port["address"],
        "endpoint_locator": str(ipaddress.ip_interface(port["address"]).ip) + "/32",
        "previous_connection_uuid": previous,
        "previous_connection_name": nm.get("connection_name"),
        "action": "none",
        "created_connection_uuid": None,
    }
    if _nm_matches(interface, port["address"]):
        return record, [], []
    authorized_previous = port.get("replace_connection_uuid")
    if authorized_previous:
        authorized_previous = _uuid(authorized_previous, "replace_connection_uuid")
        if previous != authorized_previous:
            raise NetworkPlanError(
                f"{ssh}: replacement UUID does not match active profile on {port['netdev']}"
            )
    if previous and previous not in owned and previous != authorized_previous:
        raise NetworkPlanError(
            f"{ssh}: {port['netdev']} uses unowned connection {previous}; record that exact replacement UUID before changing it"
        )
    if not previous and interface.get("ipv4"):
        raise NetworkPlanError(
            f"{ssh}: {port['netdev']} has addresses without an identified active connection"
        )
    identifier = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"sparkring/network/{owner}/{host['rank']}/{port['role']}/{port['netdev']}/{port['address']}",
        )
    )
    name = f"sparkring-{owner}-r{host['rank']}-{port['role'].replace('_', '-')}"
    collisions = [
        c for c in connections if c.get("uuid") == identifier or c.get("name") == name
    ]
    if collisions and (
        len(collisions) != 1
        or collisions[0].get("uuid") != identifier
        or identifier not in owned
    ):
        raise NetworkPlanError(
            f"{ssh}: planned NetworkManager name or UUID already belongs to another profile"
        )
    apply, rollback = [], []
    if collisions and previous != identifier:
        raise NetworkPlanError(
            f"{ssh}: an inactive owned profile must be inspected before resuming activation"
        )
    if previous == identifier:
        raise NetworkPlanError(
            f"{ssh}: owned profile {identifier} differs from the plan; inspect its saved settings"
        )
    create = [
        "sudo",
        "-n",
        "nmcli",
        "connection",
        "add",
        "type",
        "ethernet",
        "ifname",
        port["netdev"],
        "con-name",
        name,
        "connection.uuid",
        identifier,
        "connection.autoconnect",
        "yes",
        "connection.autoconnect-priority",
        "100",
        "ipv4.method",
        "manual",
        "ipv4.addresses",
        port["address"],
        "ipv4.never-default",
        "yes",
        "ipv4.ignore-auto-dns",
        "yes",
        "ipv6.method",
        "link-local",
        "ipv6.never-default",
        "yes",
        "802-3-ethernet.mtu",
        "9000",
    ]
    apply.append(
        _command(
            ssh,
            f"{port['role']}-create",
            create,
            risk="mutates-host",
            receipt_before={"connection_uuid": identifier, "state": "creating"},
            receipt_after={"connection_uuid": identifier, "state": "created"},
        )
    )
    apply.append(
        _command(
            ssh,
            f"{port['role']}-activate",
            ["sudo", "-n", "nmcli", "connection", "up", "uuid", identifier],
            risk="mutates-host",
        )
    )
    rollback.append(
        _command(
            ssh,
            f"{port['role']}-remove-created",
            ["sudo", "-n", "nmcli", "connection", "delete", "uuid", identifier],
            risk="mutates-host",
            only_if_owned_uuid=identifier,
        )
    )
    if previous:
        rollback.append(
            _command(
                ssh,
                f"{port['role']}-restore-prior",
                ["sudo", "-n", "nmcli", "connection", "up", "uuid", previous],
                risk="mutates-host",
            )
        )
    record.update(
        action="create",
        created_connection_uuid=identifier,
        created_connection_name=name,
    )
    return record, apply, rollback


def _driver(host, port, interface, function):
    ssh, netdev = host["host"], port["netdev"]
    link = _object(function.get("devlink"), f"{ssh}/{netdev} devlink")
    device = "pci/" + function["pci_address"].lower()
    if (
        link.get("available") is not True
        or link.get("error")
        or link.get("device") != device
    ):
        raise NetworkPlanError(
            f"{ssh}: devlink identity or capability is missing for {netdev}"
        )
    if (
        link.get("eswitch_mode"),
        link.get("eswitch_inline_mode"),
        link.get("eswitch_encap_mode"),
    ) != ("legacy", "none", "basic"):
        raise NetworkPlanError(
            f"{ssh}: {netdev} requires eSwitch legacy/none/basic; no automatic mode change is supported"
        )
    params = _object(link.get("parameters"), "devlink parameters")
    steering = _object(params.get("flow_steering_mode"), "flow_steering_mode")
    if steering.get("value") != "hmfs" or steering.get("cmode") != "runtime":
        raise NetworkPlanError(
            f"{ssh}: {netdev} must already expose runtime hmfs steering; driver/firmware replacement is unsupported"
        )
    apply, rollback = [], []
    for key, wanted in (("hairpin_num_queues", 4), ("hairpin_queue_size", 1024)):
        parameter = _object(params.get(key), key)
        prior = parameter.get("value")
        if isinstance(prior, str) and prior.isdigit():
            prior = int(prior)
        if (
            type(prior) is not int
            or prior < 0
            or parameter.get("cmode") != "driverinit"
        ):
            raise NetworkPlanError(
                f"{ssh}: {netdev} lacks an inventoried driverinit {key}"
            )
        permitted = parameter.get("allowed_values")
        if permitted and str(wanted) not in {str(value) for value in permitted}:
            raise NetworkPlanError(f"{ssh}: driver does not advertise {key}={wanted}")
        if prior != wanted:
            argv = [
                "sudo",
                "-n",
                "devlink",
                "dev",
                "param",
                "set",
                device,
                "name",
                key,
                "value",
                str(wanted),
                "cmode",
                "driverinit",
            ]
            apply.append(
                _command(ssh, f"{port['role']}-{key}", argv, risk="mutates-host")
            )
            argv = argv.copy()
            argv[argv.index("value") + 1] = str(prior)
            rollback.append(
                _command(
                    ssh, f"{port['role']}-restore-{key}", argv, risk="mutates-host"
                )
            )
    if apply:
        reload_argv = [
            "sudo",
            "-n",
            "devlink",
            "dev",
            "reload",
            device,
            "action",
            "driver_reinit",
        ]
        apply.append(
            _command(
                ssh,
                f"{port['role']}-driver-reload",
                reload_argv,
                risk="driver-reload",
                requires_independent_management=True,
                stop_after=True,
                resume="Rediscover interface, IP, GID, and driver state before changing another function.",
            )
        )
        rollback.append(
            _command(
                ssh,
                f"{port['role']}-rollback-driver-reload",
                reload_argv,
                risk="driver-reload",
                requires_independent_management=True,
                stop_after=True,
            )
        )
    offload = interface.get("hw_tc_offload")
    if offload is None:
        raise NetworkPlanError(
            f"{ssh}: hardware TC offload capability is unknown on {netdev}"
        )
    if offload is False:
        if interface.get("hw_tc_offload_fixed") is not False:
            raise NetworkPlanError(
                f"{ssh}: hardware TC offload cannot be enabled from the recorded capability on {netdev}"
            )
        apply.append(
            _command(
                ssh,
                f"{port['role']}-tc-offload",
                ["sudo", "-n", "ethtool", "-K", netdev, "hw-tc-offload", "on"],
                risk="mutates-host",
            )
        )
        rollback.append(
            _command(
                ssh,
                f"{port['role']}-restore-tc-offload",
                ["sudo", "-n", "ethtool", "-K", netdev, "hw-tc-offload", "off"],
                risk="mutates-host",
            )
        )
    return apply, rollback


def _verification(host, port, function):
    ssh, netdev = host["host"], port["netdev"]
    root = f"/sys/class/infiniband/{port['rdma_device']}/ports/1"
    address = str(ipaddress.ip_interface(port["address"]).ip)
    device = "pci/" + function["pci_address"].lower()
    return [
        _read(
            ssh,
            f"{port['role']}-ip",
            ["ip", "-j", "address", "show", "dev", netdev],
            {"interface": netdev, "ipv4": [port["address"]], "mtu": 9000},
        ),
        _read(
            ssh,
            f"{port['role']}-nm",
            ["nmcli", "--terse", "--fields", "GENERAL", "device", "show", netdev],
        ),
        _read(
            ssh,
            f"{port['role']}-gid",
            ["cat", root + "/gids/3"],
            {"ipv4_mapped_gid": address},
        ),
        _read(
            ssh,
            f"{port['role']}-gid-netdev",
            ["cat", root + "/gid_attrs/ndevs/3"],
            {"text": netdev},
        ),
        _read(
            ssh,
            f"{port['role']}-gid-type",
            ["cat", root + "/gid_attrs/types/3"],
            {"text": "RoCE v2"},
        ),
        _read(
            ssh,
            f"{port['role']}-verbs",
            ["ibv_devinfo", "-d", port["rdma_device"], "-i", "1"],
            {"active_mtu": 4096},
        ),
        _read(
            ssh,
            f"{port['role']}-devlink",
            ["sudo", "-n", "devlink", "-j", "dev", "param", "show", device],
            {
                "hairpin_num_queues": 4,
                "hairpin_queue_size": 1024,
                "flow_steering_mode": "hmfs",
            },
        ),
        _read(
            ssh,
            f"{port['role']}-offload",
            ["ethtool", "-k", netdev],
            {"hw-tc-offload": True},
        ),
    ]


def _backups(host, inventory, functions):
    ssh, root = host["host"], host["backup_dir"]
    paths = _object(inventory.get("paths"), f"{ssh} paths")
    directories = []
    for directory in ("/etc/NetworkManager/system-connections", "/etc/netplan"):
        state = _object(paths.get(directory), f"{ssh} backup source {directory}")
        if state.get("error") or type(state.get("exists")) is not bool:
            raise NetworkPlanError(
                f"{ssh}: backup source {directory} was not inventoried"
            )
        if state["exists"]:
            if state.get("type") != "directory":
                raise NetworkPlanError(
                    f"{ssh}: backup source {directory} is not a directory"
                )
            directories.append(directory.lstrip("/"))
    if "etc/NetworkManager/system-connections" not in directories:
        raise NetworkPlanError(f"{ssh}: NetworkManager connection directory is missing")
    result = [
        _read(ssh, "unused-backup-path", ["test", "!", "-e", root]),
        _read(ssh, "backup-not-symlink", ["test", "!", "-L", root]),
        _command(
            ssh,
            "create-backup-directory",
            ["sudo", "-n", "install", "-d", "-m", "0700", root],
            risk="mutates-host",
        ),
        _command(
            ssh,
            "archive-network-config",
            [
                "sudo",
                "-n",
                "tar",
                "-C",
                "/",
                "-czf",
                root + "/network-config.tar.gz",
                *directories,
            ],
            risk="mutates-host",
        ),
    ]
    captures = [
        ("addresses", ["ip", "-j", "address", "show"]),
        ("routes", ["ip", "-j", "route", "show", "table", "all"]),
        ("neighbors", ["ip", "-j", "neigh", "show"]),
        (
            "connections",
            [
                "nmcli",
                "--terse",
                "--fields",
                "NAME,UUID,TYPE,DEVICE",
                "connection",
                "show",
            ],
        ),
    ]
    for name, argv in captures:
        result.append(
            _read(ssh, "save-" + name, argv) | {"stdout_path": f"{root}/{name}.json"}
        )
    for port, _, function in functions:
        for name, argv in (
            (
                "tc-filters",
                [
                    "sudo",
                    "-n",
                    "tc",
                    "-j",
                    "-s",
                    "filter",
                    "show",
                    "dev",
                    port["netdev"],
                    "ingress",
                ],
            ),
            (
                "tc-qdisc",
                ["sudo", "-n", "tc", "-j", "qdisc", "show", "dev", port["netdev"]],
            ),
            (
                "driver",
                [
                    "sudo",
                    "-n",
                    "devlink",
                    "-j",
                    "dev",
                    "param",
                    "show",
                    "pci/" + function["pci_address"].lower(),
                ],
            ),
        ):
            result.append(
                _read(ssh, f"save-{port['role']}-{name}", argv)
                | {"stdout_path": f"{root}/{port['role']}-{name}.json"}
            )
    return result


def plan_network(spec, inventory):
    """Return exact host commands; callers must recheck inventory before apply.

    ``spec`` supplies owner, NetworkManager settings, and four hosts with data
    interface roles, /24 addresses, and backup directories. ``inventory`` maps
    each SSH host to ``sparkring-deploy-host-inventory/v1``. The execution receipt
    supplies ``owned_connection_uuids`` when resuming a prepared host.
    """
    owner, hosts, owned = _prepare_spec(spec)
    inventory = _object(inventory, "inventory")
    plans = []
    for host in hosts:
        ssh = host["host"]
        found = _object(inventory.get(ssh), f"{ssh} inventory")
        if (
            found.get("schema") != INVENTORY_SCHEMA
            or found.get("rank") != host["rank"]
            or found.get("ssh_target") != ssh
        ):
            raise NetworkPlanError(f"{ssh}: inventory identity does not match the host")
        network = _object(found.get("network"), f"{ssh} network inventory")
        if (
            network.get("backend") != "NetworkManager"
            or network.get("network_manager_active") is not True
        ):
            raise NetworkPlanError(
                f"{ssh}: selected interfaces require an active NetworkManager backend"
            )
        _management(host, found)
        functions = _interfaces(host, found)
        connections = network.get("connections")
        if not isinstance(connections, list):
            raise NetworkPlanError(f"{ssh}: connection UUID inventory is required")
        records, apply, rollback, driver, verify = [], [], [], [], []
        for port, interface, function in functions:
            record, actions, reverse = _connection(
                owner, host, port, interface, connections, owned
            )
            records.append(record)
            apply.extend(actions)
            rollback[0:0] = reverse
            driver_actions, driver_reverse = _driver(host, port, interface, function)
            driver.extend(driver_actions)
            rollback[0:0] = driver_reverse
            verify.extend(_verification(host, port, function))
        changed = bool(apply or driver)
        blockers = []
        if changed:
            containers = _object(found.get("docker"), "docker inventory").get(
                "containers"
            )
            if containers is None:
                blockers.append(
                    "Container state is unknown; verify model processes are stopped."
                )
            elif any(c.get("state") == "running" for c in containers):
                blockers.append(
                    "Running containers must be classified and all model/RDMA users stopped."
                )
            resources = network.get("rdma_resources")
            if resources is None:
                blockers.append("RDMA resource state is unknown.")
            elif resources:
                blockers.append("RDMA resources remain in use.")
        plans.append(
            {
                "host": ssh,
                "rank": host["rank"],
                "action": "configure" if changed else "none",
                "management_netdev": host["management_netdev"],
                "interfaces": records,
                "backup": _backups(host, found, functions) if changed else [],
                "check": [
                    _read(
                        ssh,
                        "management-route",
                        [
                            "ip",
                            "-j",
                            "route",
                            "get",
                            found["management"]["route_to_controller"]["dst"],
                        ],
                        {"interface": host["management_netdev"]},
                    )
                ],
                "apply": apply,
                "driver_steps": driver,
                "verify": verify,
                "rollback": rollback,
                "apply_permitted": not blockers,
                "blocked_by": blockers,
            }
        )
    return {
        "schema": PLAN_SCHEMA,
        "status": "implemented",
        "owner": owner,
        "hosts": plans,
        "requires_hardware_validation": True,
        "execution_order": ["check", "backup", "apply", "driver_steps", "verify"],
        "driver_step_policy": "Stop after each driver reload, rediscover, and regenerate the plan.",
        "limitations": [
            "Only NetworkManager data interfaces are supported.",
            "Mesh routes, TC rules, and services belong to the managed-mesh installer.",
            "Driver, firmware, eSwitch, and steering-mode replacements are unsupported.",
        ],
    }


def verify_network(spec, inventory):
    """Check a fresh inventory after preparation; command exit status is insufficient."""
    plan = plan_network(spec, inventory)
    _, hosts, _ = _prepare_spec(spec)
    for host, prepared in zip(hosts, plan["hosts"]):
        if prepared["action"] != "none":
            raise NetworkPlanError(
                f"{host['host']}: persistent addresses or driver settings do not match the plan"
            )
        for port, interface, function in _interfaces(host, inventory[host["host"]]):
            expected = ipaddress.ip_interface(port["address"]).ip
            try:
                mapped = ipaddress.IPv6Address(function["gid"]).ipv4_mapped
            except (KeyError, TypeError, ValueError):
                mapped = None
            if function.get("gid_index") != 3 or mapped != expected:
                raise NetworkPlanError(
                    f"{host['host']}: GID index 3 does not match {port['netdev']} IPv4 address"
                )
            if function.get("gid_type") != "RoCE v2":
                raise NetworkPlanError(f"{host['host']}: GID index 3 must use RoCE v2")
            if function.get('gid_netdev') != port['netdev']:
                raise NetworkPlanError(f"{host['host']}: GID index 3 interface must be {port['netdev']}")
            if function.get("active_mtu") != 4096:
                raise NetworkPlanError(
                    f"{host['host']}: RDMA active MTU must be 4096 on {port['rdma_device']}"
                )
            if (
                function.get("state") not in ("ACTIVE", "PORT_ACTIVE")
                or interface.get("operstate") != "UP"
            ):
                raise NetworkPlanError(
                    f"{host['host']}: data link {port['netdev']} is not active"
                )
    return {
        "schema": "sparkring-deploy-network-verification/v1",
        "status": "implemented",
        "ready": True,
        "hosts": [host["host"] for host in hosts],
        "data_functions": 16,
        "limitations": [
            "This checks host configuration, not RDMA traffic or hardware forwarding correctness."
        ],
    }
