"""Read-only host facts for deployment planning; no host changes or model requests."""

from __future__ import annotations

import copy
import inspect
import ipaddress
from pathlib import PurePosixPath
import re
from typing import Any, Sequence


SCHEMA = "sparkring-deploy-host-inventory/v1"
MANAGED_PATHS = (
    "/opt/sparkring/managed-mesh",
    "/etc/sparkring/managed-mesh",
    "/run/sparkring-mesh",
    "/etc/systemd/system/sparkring-mesh.service",
    "/etc/systemd/system/sparkring-mesh-model.service",
)
NETWORK_PATHS = ("/etc/NetworkManager/system-connections", "/etc/netplan")
TOOLS = (
    "python3",
    "ip",
    "docker",
    "nvidia-smi",
    "nvidia-ctk",
    "sudo",
    "nmcli",
    "networkctl",
    "systemctl",
    "ibdev2netdev",
    "ibv_devinfo",
    "rdma",
    "devlink",
    "tc",
    "ethtool",
    "lspci",
    "curl",
    "sha256sum",
    "tar",
    "scp",
    "ssh",
)


def _request(
    rank: int,
    ssh_target: str,
    management_address: str,
    paths: Sequence[str],
    controller_address: str | None,
) -> dict[str, Any]:
    if type(rank) is not int or rank not in range(4):
        raise ValueError("rank must be an integer from 0 to 3")
    if not isinstance(ssh_target, str) or not re.fullmatch(
        r"[A-Za-z0-9_][A-Za-z0-9_.@:-]{0,254}", ssh_target
    ):
        raise ValueError(
            "ssh_target must be one SSH alias or user@host, without options"
        )
    for name, value in (
        ("management_address", management_address),
        ("controller_address", controller_address),
    ):
        if value is None and name == "controller_address":
            continue
        if not isinstance(value, str):
            raise ValueError(f"{name} must be an IPv4 address")
        try:
            address = ipaddress.IPv4Address(value)
        except (ValueError, TypeError, ipaddress.AddressValueError) as error:
            raise ValueError(f"{name} must be an IPv4 address") from error
        if address.is_unspecified or address.is_multicast or address.is_loopback:
            raise ValueError(f"{name} must identify a network host")
    if isinstance(paths, (str, bytes)) or len(paths) > 32:
        raise ValueError("paths must contain at most 32 absolute host paths")
    selected = []
    for value in ("/", *MANAGED_PATHS, *NETWORK_PATHS, *paths):
        if (
            not isinstance(value, str)
            or not value.startswith("/")
            or ".." in PurePosixPath(value).parts
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError(
                "inventory paths must be absolute without parent traversal"
            )
        if str(PurePosixPath(value)) != value:
            raise ValueError("inventory paths must use normalized POSIX spelling")
        if value not in selected:
            selected.append(value)
    return {
        "schema": SCHEMA,
        "rank": rank,
        "ssh_target": ssh_target,
        "management_address": management_address,
        "controller_address": controller_address,
        "paths": selected,
        "managed_paths": list(MANAGED_PATHS),
        "tools": list(TOOLS),
    }


def collect_command(
    *,
    rank: int,
    ssh_target: str,
    management_address: str,
    paths: Sequence[str] = (),
    controller_address: str | None = None,
) -> list[str]:
    """Return a Linux Python probe argv. The caller chooses whether and where to run it."""
    request = _request(rank, ssh_target, management_address, paths, controller_address)
    source = inspect.getsource(_collect_local)
    source += (
        "\nimport json\nprint(json.dumps(_collect_local("
        + repr(request)
        + "), sort_keys=True))\n"
    )
    return ["python3", "-c", source]


def _collect_local(
    options, *, run=None, root="/", tool_path=None, platform_info=None, euid=None
):
    """Collect local facts. Injectable command and filesystem readers support offline fixtures."""
    import csv
    import datetime
    import ipaddress
    import json
    import os
    from pathlib import Path
    import platform
    import re
    import shutil
    import stat
    import subprocess

    execute = run or subprocess.run
    locate = tool_path or shutil.which
    base = Path(root)
    tools = {name: locate(name) for name in options["tools"]}
    errors = []

    def file(path):
        return base / str(path).lstrip("/")

    def read(path):
        try:
            return file(path).read_text().strip()
        except OSError:
            return None

    def command(argv, *, accepted=(0,)):
        if not locate(argv[0]):
            return None, f"{argv[0]} is not installed or not on PATH"
        try:
            result = execute(
                argv,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
                env={**os.environ, "LC_ALL": "C"},
            )
        except (OSError, subprocess.SubprocessError) as error:
            return None, f"{argv[0]} query unavailable: {type(error).__name__}"
        if result.returncode not in accepted:
            return None, f"{argv[0]} query exited {result.returncode}"
        return result.stdout.strip(), None

    def json_command(argv):
        output, error = command(argv)
        if error:
            return None, error
        try:
            return json.loads(output), None
        except (ValueError, TypeError):
            return None, f"{argv[0]} returned invalid JSON"

    def service(name):
        output, error = command(
            [
                "systemctl",
                "show",
                name,
                "--no-pager",
                "--property=LoadState,ActiveState,SubState",
            ]
        )
        if error:
            return {
                "load_state": None,
                "active_state": None,
                "sub_state": None,
                "error": error,
            }
        values = dict(row.split("=", 1) for row in output.splitlines() if "=" in row)
        return {
            "load_state": values.get("LoadState"),
            "active_state": values.get("ActiveState"),
            "sub_state": values.get("SubState"),
            "error": None,
        }

    def yes_no(value):
        return {"yes": True, "no": False, "true": True, "false": False}.get(value)

    uid = os.geteuid() if euid is None else euid
    sudo, sudo_error = command(["sudo", "-n", "id", "-u"])
    privilege = {
        "root": uid == 0,
        "sudo_noninteractive": sudo == "0" if sudo is not None else None,
        "error": sudo_error
        or (
            "sudo did not report root identity"
            if sudo is not None and sudo != "0"
            else None
        ),
    }
    platform_values = platform_info or (platform.system(), platform.machine())
    gpu_text, gpu_error = command(
        ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"]
    )
    devices = None
    if gpu_text is not None:
        rows = list(csv.reader(gpu_text.splitlines()))
        if all(len(row) == 3 for row in rows):
            devices = [
                {
                    "name": row[0].strip(),
                    "driver_version": row[1].strip(),
                    "uuid": row[2].strip(),
                }
                for row in rows
            ]
        else:
            gpu_error = "nvidia-smi returned an unexpected GPU table"
    gpu = {
        "available": bool(devices) if devices is not None else None,
        "devices": devices,
        "error": gpu_error,
    }
    docker_info, docker_error = json_command(
        ["docker", "info", "--format", "{{json .}}"]
    )
    if docker_info is not None and not isinstance(docker_info, dict):
        docker_info, docker_error = None, "docker info returned an unexpected object"
    containers = None
    container_text, container_error = command(
        ["docker", "ps", "--all", "--no-trunc", "--format", "{{json .}}"]
    )
    if container_text is not None:
        try:
            rows = [json.loads(line) for line in container_text.splitlines() if line]
            containers = [
                {
                    "id": row.get("ID"),
                    "name": row.get("Names"),
                    "image": row.get("Image"),
                    "state": row.get("State"),
                    "status": row.get("Status"),
                }
                for row in rows
            ]
        except (ValueError, AttributeError):
            container_error = "docker ps returned an unexpected container table"
    docker = {
        "available": True if docker_info is not None else None,
        "server_version": (docker_info or {}).get("ServerVersion"),
        "runtimes": sorted((docker_info or {}).get("Runtimes", {}))
        if docker_info
        else None,
        "containers": containers,
        "error": docker_error or container_error,
    }
    toolkit_text, toolkit_error = command(["nvidia-ctk", "--version"])
    toolkit = {
        "available": True if toolkit_text is not None else None,
        "version": toolkit_text,
        "error": toolkit_error,
    }
    link_rows, link_error = json_command(["ip", "-j", "-4", "address", "show"])
    if link_rows is not None and not isinstance(link_rows, list):
        link_rows, link_error = None, "ip returned an unexpected interface table"
    route_rows, route_error = json_command(
        ["ip", "-j", "-4", "route", "show", "table", "all"]
    )
    if route_rows is not None and not isinstance(route_rows, list):
        route_rows, route_error = None, "ip returned an unexpected route table"
    nm_service, networkd_service = (
        service("NetworkManager.service"),
        service("systemd-networkd.service"),
    )

    def active(value):
        if value["active_state"] is None:
            return None
        return value["active_state"] == "active"

    nm_active, networkd_active = active(nm_service), active(networkd_service)
    if nm_active is True and networkd_active is True:
        backend = "ambiguous"
    elif nm_active is True:
        backend = "NetworkManager"
    elif networkd_active is True:
        backend = "systemd-networkd"
    else:
        backend = "unknown"
    nm_fields = {
        "connection_name": "connection.id",
        "interface": "connection.interface-name",
        "owner": "connection.permissions",
        "ipv4_method": "ipv4.method",
        "ipv4_never_default": "ipv4.never-default",
        "ipv4_ignore_auto_dns": "ipv4.ignore-auto-dns",
        "ipv6_method": "ipv6.method",
        "ipv6_never_default": "ipv6.never-default",
        "autoconnect": "connection.autoconnect",
    }

    def connection(uuid):
        result = {"uuid": uuid}
        faults = []
        for key, field in nm_fields.items():
            output, error = command(
                ["nmcli", "--escape", "no", "-g", field, "connection", "show", uuid]
            )
            if error:
                faults.append(error)
            if key in {
                "ipv4_never_default",
                "ipv4_ignore_auto_dns",
                "ipv6_never_default",
                "autoconnect",
            }:
                output = yes_no(output)
            elif key == "owner" and output == "":
                output = "system"
            result[key] = output
        result["name"] = result["connection_name"]
        result["error"] = "; ".join(dict.fromkeys(faults)) or None
        return result

    nm_connections, nm_error = None, None
    uuids, nm_error = command(["nmcli", "-g", "UUID", "connection", "show"])
    if uuids is not None:
        nm_connections = []
        for uuid in uuids.splitlines():
            if re.fullmatch(r"[0-9a-fA-F-]{36}", uuid):
                nm_connections.append(connection(uuid))
            else:
                nm_error = "nmcli returned an invalid connection UUID"
    interfaces = []
    for row in link_rows or []:
        name = row.get("ifname")
        if not isinstance(name, str):
            continue
        nm = {
            "available": None,
            "managed": None,
            "connection_uuid": None,
            "connection_name": None,
            "owner": None,
            "error": nm_error,
        }
        state, state_error = command(
            [
                "nmcli",
                "--escape",
                "no",
                "-g",
                "GENERAL.NM-MANAGED,GENERAL.CON-UUID",
                "device",
                "show",
                name,
            ]
        )
        if state is not None:
            fields = state.splitlines()
            if len(fields) == 2:
                uuid = fields[1] if fields[1] != "--" else None
                saved = next(
                    (item for item in nm_connections or [] if item["uuid"] == uuid), {}
                )
                nm.update(saved)
                nm.update(
                    available=True, managed=yes_no(fields[0]), connection_uuid=uuid
                )
            else:
                state_error = "nmcli returned an unexpected device status"
        nm["error"] = state_error or nm.get("error")
        offload, offload_error = command(["ethtool", "-k", name])
        match = re.search(
            r"^hw-tc-offload:\s+(on|off)(\s+\[fixed\])?\s*$",
            offload or "",
            re.MULTILINE,
        )
        interfaces.append(
            {
                "name": name,
                "mac": row.get("address"),
                "mtu": row.get("mtu"),
                "operstate": row.get("operstate"),
                "master": row.get("master"),
                "ipv4": [
                    str(item["local"]) + "/" + str(item["prefixlen"])
                    for item in row.get("addr_info", [])
                    if item.get("family") == "inet"
                    and "local" in item
                    and "prefixlen" in item
                ],
                "network_manager": nm,
                "hw_tc_offload": match.group(1) == "on" if match else None,
                "hw_tc_offload_fixed": bool(match.group(2)) if match else None,
                "hw_tc_offload_error": offload_error if not match else None,
            }
        )
    management_matches = [
        row["name"]
        for row in interfaces
        if any(
            cidr.split("/", 1)[0] == options["management_address"]
            for cidr in row["ipv4"]
        )
    ]
    management_interface = (
        management_matches[0] if len(management_matches) == 1 else None
    )
    controller = options.get("controller_address")
    if controller is None:
        values = os.environ.get("SSH_CONNECTION", "").split()
        if values:
            try:
                controller = str(ipaddress.IPv4Address(values[0]))
            except ValueError:
                pass
    route_to_controller, controller_error = None, None
    if controller:
        rows, controller_error = json_command(
            ["ip", "-j", "-4", "route", "get", controller]
        )
        if isinstance(rows, list) and len(rows) == 1:
            route_to_controller = rows[0]
        elif not controller_error:
            controller_error = "management return route is missing or ambiguous"
    else:
        controller_error = (
            "controller IPv4 address was not supplied or observed through SSH"
        )
    management = {
        "address": options["management_address"],
        "interface": management_interface,
        "controller_address": controller,
        "route_to_controller": route_to_controller,
        "error": link_error
        or (
            "management address must occur on exactly one interface"
            if len(management_matches) != 1
            else controller_error
        ),
    }

    def devlink(pci):
        result = {
            "available": None,
            "device": "pci/" + pci if pci else None,
            "parameters": {},
            "eswitch_mode": None,
            "eswitch_inline_mode": None,
            "eswitch_encap_mode": None,
            "error": None,
        }
        if pci is None:
            result["error"] = "RDMA PCI address is unavailable"
            return result
        faults = []
        for name in ("hairpin_num_queues", "hairpin_queue_size", "flow_steering_mode"):
            rows, error = json_command(
                ["devlink", "-j", "dev", "param", "show", "pci/" + pci, "name", name]
            )
            values = (
                (rows or {}).get("param", {}).get("pci/" + pci, [])
                if isinstance(rows, dict)
                else []
            )
            entry = next((item for item in values if item.get("name") == name), None)
            if entry:
                expected_mode = (
                    "runtime" if name == "flow_steering_mode" else "driverinit"
                )
                setting = next(
                    (
                        item
                        for item in entry.get("values", [])
                        if item.get("cmode") == expected_mode
                    ),
                    {},
                )
                result["parameters"][name] = {
                    "value": setting.get("value"),
                    "cmode": setting.get("cmode"),
                    "allowed_values": entry.get("allowed_values"),
                }
            else:
                faults.append(error or "devlink parameter unavailable: " + name)
                result["parameters"][name] = {
                    "value": None,
                    "cmode": None,
                    "allowed_values": None,
                }
        rows, error = json_command(
            ["devlink", "-j", "dev", "eswitch", "show", "pci/" + pci]
        )
        settings = (
            (rows or {}).get("dev", {}).get("pci/" + pci, {})
            if isinstance(rows, dict)
            else {}
        )
        if settings:
            result.update(
                eswitch_mode=settings.get("mode"),
                eswitch_inline_mode=settings.get("inline-mode"),
                eswitch_encap_mode=settings.get("encap-mode"),
            )
        else:
            faults.append(error or "devlink eswitch state unavailable")
        result["available"] = (
            True
            if any(item["value"] is not None for item in result["parameters"].values())
            else None
        )
        result["error"] = "; ".join(dict.fromkeys(faults)) or None
        return result

    rdma_rows = []
    mappings, mapping_error = command(["ibdev2netdev"])
    for row in (mappings or "").splitlines():
        match = re.fullmatch(r"(\S+) port (\d+) ==> (\S+) \(([^)]+)\)", row.strip())
        if not match:
            errors.append("ibdev2netdev returned an unrecognized mapping")
            continue
        device, port_text, netdev, link_state = match.groups()
        port = int(port_text)
        sys_device = file("/sys/class/infiniband/" + device + "/device")
        pci = sys_device.resolve().name if sys_device.exists() else None
        uevent = dict(
            row.split("=", 1)
            for row in (
                read("/sys/class/infiniband/" + device + "/device/uevent") or ""
            ).splitlines()
            if "=" in row
        )
        if not re.fullmatch(
            r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci or ""
        ):
            pci = uevent.get("PCI_SLOT_NAME")
        if not re.fullmatch(
            r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]", pci or ""
        ):
            pci = None
        driver_path = sys_device / "driver"
        driver = (
            driver_path.resolve().name if driver_path.exists() else uevent.get("DRIVER")
        )
        prefix = "/sys/class/infiniband/" + device + "/ports/" + str(port)
        info, info_error = command(["ibv_devinfo", "-d", device, "-i", str(port)])
        mtu = re.search(
            r"^\s*active_mtu:\s+(\d+)\s+\(\d+\)\s*$", info or "", re.MULTILINE
        )
        layer = re.search(r"^\s*link_layer:\s+(\S+)\s*$", info or "", re.MULTILINE)
        state = re.search(r"^\s*state:\s+(\S+)\s+\(\d+\)\s*$", info or "", re.MULTILINE)
        gid = read(prefix + "/gids/3")
        rdma_rows.append(
            {
                "device": device,
                "port": port,
                "netdev": netdev,
                "pci_address": pci,
                "driver": driver,
                "gid_index": 3,
                "gid": gid,
                "gid_netdev": read(prefix + "/gid_attrs/ndevs/3"),
                "gid_type": read(prefix + "/gid_attrs/types/3"),
                "active_mtu": int(mtu.group(1)) if mtu else None,
                "state": state.group(1) if state else None,
                "link_state": link_state,
                "link_layer": layer.group(1) if layer else None,
                "devlink": devlink(pci),
                "error": info_error,
            }
        )
    rdma_resources, resource_error = json_command(
        ["rdma", "-j", "resource", "show", "qp"]
    )
    if rdma_resources is not None and not isinstance(rdma_resources, list):
        rdma_resources, resource_error = (
            None,
            "rdma returned an unexpected resource table",
        )
    paths = {}
    for name in options["paths"]:
        target = file(name)
        details = {
            "exists": None,
            "type": None,
            "nonempty": None,
            "mode": None,
            "owner_uid": None,
            "free_bytes": None,
            "error": None,
        }
        try:
            observed = target.lstat()
            details.update(
                exists=True,
                owner_uid=observed.st_uid,
                mode=format(stat.S_IMODE(observed.st_mode), "04o"),
            )
            if stat.S_ISLNK(observed.st_mode):
                details["type"] = "symlink"
            elif stat.S_ISDIR(observed.st_mode):
                details.update(
                    type="directory", nonempty=next(target.iterdir(), None) is not None
                )
            elif stat.S_ISREG(observed.st_mode):
                details.update(type="file", nonempty=observed.st_size > 0)
            else:
                details["type"] = "other"
        except FileNotFoundError:
            details.update(exists=False, nonempty=False)
        except OSError as error:
            details["error"] = type(error).__name__
        try:
            ancestor = target
            while not ancestor.exists() and ancestor != ancestor.parent:
                ancestor = ancestor.parent
            details["free_bytes"] = shutil.disk_usage(ancestor).free
        except OSError as error:
            details["error"] = details["error"] or type(error).__name__
        paths[name] = details
    units = {
        name: service(name)
        for name in ("sparkring-mesh.service", "sparkring-mesh-model.service")
    }
    installed = [paths[name]["exists"] for name in options["managed_paths"]]
    for error in (link_error, route_error, mapping_error, resource_error, nm_error):
        if error:
            errors.append(error)
    return {
        "schema": options["schema"],
        "rank": options["rank"],
        "ssh_target": options["ssh_target"],
        "collected_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "platform": {"system": platform_values[0], "architecture": platform_values[1]},
        "gpu": gpu,
        "docker": docker,
        "toolkit": toolkit,
        "privilege": privilege,
        "tools": tools,
        "management": management,
        "interfaces": interfaces,
        "routes": route_rows,
        "rdma": rdma_rows,
        "network": {
            "backend": backend,
            "network_manager_active": nm_active,
            "networkd_active": networkd_active,
            "connections": nm_connections,
            "rdma_resources": rdma_resources,
            "errors": list(dict.fromkeys(errors)),
        },
        "paths": paths,
        "managed": {
            "units": units,
            "installation_present": True
            if True in installed
            else None
            if None in installed
            else False,
        },
    }


def validate_inventory(document: Any, *, require_ready: bool = False) -> dict[str, Any]:
    """Validate recorded facts; require_ready also checks basic host identity and authority."""
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError("inventory schema must be " + SCHEMA)
    management = document.get("management")
    if not isinstance(management, dict):
        raise ValueError("inventory management must be an object")
    _request(
        document.get("rank"),
        document.get("ssh_target"),
        management.get("address"),
        (),
        management.get("controller_address"),
    )
    for name in (
        "platform",
        "gpu",
        "docker",
        "toolkit",
        "privilege",
        "tools",
        "network",
        "paths",
        "managed",
    ):
        if not isinstance(document.get(name), dict):
            raise ValueError(f"inventory {name} must be an object")
    for role in ("gpu", "docker", "toolkit"):
        if (
            document[role].get("available") is not None
            and type(document[role]["available"]) is not bool
        ):
            raise ValueError(f"{role}.available must be true, false, or null")
    for name in ("root", "sudo_noninteractive"):
        value = document["privilege"].get(name)
        if value is not None and type(value) is not bool:
            raise ValueError(f"privilege.{name} must be true, false, or null")
    for name in ("interfaces", "rdma"):
        if not isinstance(document.get(name), list) or any(
            not isinstance(row, dict) for row in document[name]
        ):
            raise ValueError(f"inventory {name} must be a list of objects")
    if document.get("routes") is not None and (
        not isinstance(document["routes"], list)
        or any(not isinstance(row, dict) for row in document["routes"])
    ):
        raise ValueError("inventory routes must be a list of objects or null")
    for path, facts in document["paths"].items():
        _request(
            document["rank"],
            document["ssh_target"],
            management["address"],
            (path,),
            None,
        )
        if not isinstance(facts, dict):
            raise ValueError("path observations must be objects")
        for name in ("exists", "nonempty"):
            if facts.get(name) is not None and type(facts[name]) is not bool:
                raise ValueError(f"path {name} must be true, false, or null")
    interfaces = {}
    for row in document["interfaces"]:
        name = row.get("name")
        if (
            not isinstance(name, str)
            or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,15}", name)
            or name in interfaces
        ):
            raise ValueError("interface names must be valid and distinct")
        interfaces[name] = row
        if not isinstance(row.get("ipv4"), list):
            raise ValueError("interface IPv4 addresses must be a list")
        for address in row["ipv4"]:
            try:
                ipaddress.IPv4Interface(address)
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid IPv4 address on interface {name}") from error
        if row.get("mac") is not None and not re.fullmatch(
            r"(?:[0-9A-Fa-f]{2}:){5}[0-9A-Fa-f]{2}", row["mac"]
        ):
            raise ValueError(f"invalid MAC address on interface {name}")
        if row.get("mtu") is not None and (
            type(row["mtu"]) is not int or row["mtu"] <= 0
        ):
            raise ValueError(f"invalid MTU on interface {name}")
    functions = set()
    for row in document["rdma"]:
        device, port, netdev = row.get("device"), row.get("port"), row.get("netdev")
        if not isinstance(device, str) or not re.fullmatch(
            r"[A-Za-z0-9_.:-]{1,64}", device
        ):
            raise ValueError("RDMA device name is invalid")
        if type(port) is not int or port < 1 or (device, port) in functions:
            raise ValueError("RDMA device/port mappings must be distinct")
        functions.add((device, port))
        if netdev not in interfaces:
            raise ValueError(
                f"RDMA device {device} maps to an unobserved network interface"
            )
        if row.get("gid") is not None:
            try:
                ipaddress.IPv6Address(row["gid"])
            except (ValueError, TypeError) as error:
                raise ValueError(f"invalid GID on RDMA device {device}") from error
        if row.get("active_mtu") is not None and (
            type(row["active_mtu"]) is not int
            or row["active_mtu"] not in (256, 512, 1024, 2048, 4096)
        ):
            raise ValueError(f"invalid active RoCE MTU on RDMA device {device}")
    if require_ready:
        system = document["platform"]
        if system.get("system") != "Linux" or system.get("architecture") not in (
            "aarch64",
            "arm64",
        ):
            raise ValueError("deployment requires Linux ARM64 hosts")
        privilege = document["privilege"]
        if (
            privilege.get("root") is not True
            and privilege.get("sudo_noninteractive") is not True
        ):
            raise ValueError(
                "root or verified noninteractive sudo is required before host preparation"
            )
        netdev = management.get("interface")
        if netdev not in interfaces or not any(
            str(ipaddress.IPv4Interface(value).ip) == management["address"]
            for value in interfaces[netdev]["ipv4"]
        ):
            raise ValueError(
                "management address must identify one observed network interface"
            )
        if (
            len(
                [
                    name
                    for name, row in interfaces.items()
                    if any(
                        str(ipaddress.IPv4Interface(value).ip) == management["address"]
                        for value in row["ipv4"]
                    )
                ]
            )
            != 1
        ):
            raise ValueError(
                "management address appears on more than one network interface"
            )
        if (
            len(functions) != 4
            or len({row["device"] for row in document["rdma"]}) != 4
            or len({row["netdev"] for row in document["rdma"]}) != 4
            or any(row["port"] != 1 for row in document["rdma"])
        ):
            raise ValueError(
                "four distinct RDMA functions are required for this mesh profile"
            )
        if netdev in {row["netdev"] for row in document["rdma"]}:
            raise ValueError("management traffic must not share a mesh data interface")
        route = management.get("route_to_controller")
        if not isinstance(route, dict) or route.get("dev") != netdev:
            raise ValueError(
                "management return route is unavailable or uses a different interface"
            )
    return copy.deepcopy(document)


def summarise_inventory(document: Any) -> str:
    """Summarise observed facts without turning unavailable evidence into success."""
    data = validate_inventory(document)
    missing = sorted(name for name, path in data["tools"].items() if path is None)
    system = data["platform"]
    gpu = data["gpu"]
    gpu_text = ", ".join(
        item.get("name", "unknown") for item in gpu.get("devices") or []
    )
    lines = [
        f"Rank {data['rank']} ({data['ssh_target']}): {system.get('system', 'unknown')} {system.get('architecture', 'unknown')}",
        "GPU: "
        + (
            gpu_text
            or ("none reported" if gpu.get("available") is False else "unavailable")
        ),
        "Management: "
        + str(data["management"].get("interface") or "unavailable")
        + " / "
        + data["management"]["address"],
        f"RDMA: {len(data['rdma'])} observed function(s); network manager: {data['network'].get('backend', 'unknown')}",
    ]
    if missing:
        lines.append("Tools absent from PATH: " + ", ".join(missing))
    running = [
        row
        for row in data["docker"].get("containers") or []
        if row.get("state") == "running"
    ]
    lines.append(
        "Running containers: "
        + (
            str(len(running))
            if data["docker"].get("containers") is not None
            else "unavailable"
        )
    )
    installed = data["managed"].get("installation_present")
    lines.append(
        "Managed mesh files: "
        + (
            "present"
            if installed is True
            else "absent"
            if installed is False
            else "unavailable"
        )
    )
    occupied = [
        path
        for path, fact in data["paths"].items()
        if path != "/" and fact.get("nonempty") is True
    ]
    if occupied:
        lines.append("Nonempty paths: " + ", ".join(occupied))
    try:
        validate_inventory(data, require_ready=True)
    except ValueError as error:
        lines.append("Preparation blocker: " + str(error))
    lines.append(
        "Hardware forwarding and model startup are not tested by this inventory."
    )
    return "\n".join(lines)
