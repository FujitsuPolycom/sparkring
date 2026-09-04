#!/usr/bin/env python3
"""Configure and verify one persistent NetworkManager profile for a SIRCL rail.

Planning is the default and performs no host inspection or mutation. ``--verify``
checks the saved profile, live IPv4/MTU state, RoCEv2 GID, and a don't-fragment
peer ping. ``--execute`` creates or updates the named profile, activates it, and
runs the same verification. Execution requires root and an exact confirmation.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


PLAN_SCHEMA = "sparkring-sircl-rail-plan/v1"
VERIFY_SCHEMA = "sparkring-sircl-rail-verification/v1"
APPLY_SCHEMA = "sparkring-sircl-rail-application/v1"
CONFIRMATION = "CONFIGURE_SIRCL_RAIL"

_INTERFACE = re.compile(r"[A-Za-z0-9_.:-]{1,15}\Z")
_DEVICE = re.compile(r"[A-Za-z0-9_.:-]{1,64}\Z")
_CONNECTION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,63}\Z")


class RailConfigError(ValueError):
    """The requested rail profile is unsafe or internally inconsistent."""


@dataclass(frozen=True)
class RailConfig:
    interface: str
    management_interface: str
    address: ipaddress.IPv4Interface
    peer_address: ipaddress.IPv4Address
    rdma_device: str
    rdma_port: int
    gid_index: int
    mtu: int
    connection_name: str
    ping_timeout_seconds: int

    @property
    def expected_gid(self) -> str:
        value = int(self.address.ip)
        high = (value >> 16) & 0xFFFF
        low = value & 0xFFFF
        return f"0000:0000:0000:0000:0000:ffff:{high:04x}:{low:04x}"

    @property
    def jumbo_payload_bytes(self) -> int:
        return self.mtu - 28


Runner = Callable[..., subprocess.CompletedProcess[str]]


def rail_config(
    *,
    interface: str,
    management_interface: str,
    address_cidr: str,
    peer_address: str,
    rdma_device: str,
    rdma_port: int = 1,
    gid_index: int = 3,
    mtu: int = 9000,
    connection_name: str | None = None,
    ping_timeout_seconds: int = 3,
) -> RailConfig:
    """Validate one dedicated IPv4 RoCE rail configuration."""
    if _INTERFACE.fullmatch(interface) is None:
        raise RailConfigError(
            "interface must be a Linux interface name of at most 15 safe characters"
        )
    if _INTERFACE.fullmatch(management_interface) is None:
        raise RailConfigError(
            "management interface must be a Linux interface name of at most 15 safe characters"
        )
    if interface == management_interface:
        raise RailConfigError("SIRCL rail interface must differ from the management interface")
    if _DEVICE.fullmatch(rdma_device) is None:
        raise RailConfigError("RDMA device must contain only safe name characters")
    if not 1 <= rdma_port <= 8:
        raise RailConfigError("RDMA port must be between 1 and 8")
    if not 0 <= gid_index <= 255:
        raise RailConfigError("GID index must be between 0 and 255")
    if not 1500 <= mtu <= 9216:
        raise RailConfigError("MTU must be between 1500 and 9216 bytes")
    if not 1 <= ping_timeout_seconds <= 60:
        raise RailConfigError("ping timeout must be between 1 and 60 seconds")

    try:
        address = ipaddress.ip_interface(address_cidr)
    except ValueError as exc:
        raise RailConfigError(f"address CIDR is invalid: {exc}") from None
    if not isinstance(address, ipaddress.IPv4Interface):
        raise RailConfigError("address CIDR must be IPv4")
    if not 8 <= address.network.prefixlen <= 30:
        raise RailConfigError("address CIDR prefix must be between /8 and /30")
    if address.ip == address.network.network_address:
        raise RailConfigError("local address must not be the subnet network address")
    if address.ip == address.network.broadcast_address:
        raise RailConfigError("local address must not be the subnet broadcast address")

    try:
        peer = ipaddress.ip_address(peer_address)
    except ValueError as exc:
        raise RailConfigError(f"peer address is invalid: {exc}") from None
    if not isinstance(peer, ipaddress.IPv4Address):
        raise RailConfigError("peer address must be IPv4")
    if peer == address.ip:
        raise RailConfigError("peer address must differ from the local address")
    if peer not in address.network:
        raise RailConfigError("peer address must be in the same IPv4 subnet")
    if peer in {address.network.network_address, address.network.broadcast_address}:
        raise RailConfigError("peer address must be a usable host address")

    profile = connection_name or f"sparkring-sircl-{interface}"
    if _CONNECTION.fullmatch(profile) is None:
        raise RailConfigError(
            "connection name must be 1-64 safe characters and start alphanumeric"
        )
    return RailConfig(
        interface=interface,
        management_interface=management_interface,
        address=address,
        peer_address=peer,
        rdma_device=rdma_device,
        rdma_port=rdma_port,
        gid_index=gid_index,
        mtu=mtu,
        connection_name=profile,
        ping_timeout_seconds=ping_timeout_seconds,
    )


def _profile_settings(config: RailConfig) -> tuple[str, ...]:
    return (
        "connection.interface-name",
        config.interface,
        "connection.autoconnect",
        "yes",
        "connection.autoconnect-priority",
        "100",
        "connection.autoconnect-retries",
        "0",
        "802-3-ethernet.mtu",
        str(config.mtu),
        "ipv4.method",
        "manual",
        "ipv4.addresses",
        str(config.address),
        "ipv4.gateway",
        "",
        "ipv4.dns",
        "",
        "ipv4.never-default",
        "yes",
        "ipv6.method",
        "disabled",
    )


def add_profile_command(config: RailConfig) -> tuple[str, ...]:
    return (
        "nmcli",
        "connection",
        "add",
        "type",
        "ethernet",
        "con-name",
        config.connection_name,
        "ifname",
        config.interface,
        *_profile_settings(config),
    )


def modify_profile_command(config: RailConfig) -> tuple[str, ...]:
    return (
        "nmcli",
        "connection",
        "modify",
        config.connection_name,
        *_profile_settings(config),
    )


def activate_profile_command(config: RailConfig) -> tuple[str, ...]:
    return (
        "nmcli",
        "connection",
        "up",
        config.connection_name,
        "ifname",
        config.interface,
    )


def plan_document(config: RailConfig) -> dict[str, Any]:
    """Describe the exact persistent state and commands without inspecting a host."""
    return {
        "schema": PLAN_SCHEMA,
        "mode": "plan",
        "mutates_host": False,
        "connection_name": config.connection_name,
        "interface": config.interface,
        "management_interface": config.management_interface,
        "address_cidr": str(config.address),
        "peer_address": str(config.peer_address),
        "mtu": config.mtu,
        "profile_contract": {
            "autoconnect": True,
            "autoconnect_priority": 100,
            "autoconnect_retries": "forever",
            "ipv4_method": "manual",
            "ipv4_never_default": True,
            "ipv6_method": "disabled",
        },
        "roce": {
            "device": config.rdma_device,
            "port": config.rdma_port,
            "gid_index": config.gid_index,
            "expected_gid": config.expected_gid,
            "expected_type": "RoCE v2",
        },
        "jumbo_ping": {
            "destination": str(config.peer_address),
            "payload_bytes": config.jumbo_payload_bytes,
            "timeout_seconds": config.ping_timeout_seconds,
        },
        "application": {
            "profile_missing": list(add_profile_command(config)),
            "profile_present": list(modify_profile_command(config)),
            "activate": list(activate_profile_command(config)),
            "confirmation": CONFIRMATION,
        },
        "warning": (
            "Activation replaces the live connection on this dedicated interface. "
            "Confirm the interface is not the management path before execution."
        ),
    }


def _run(
    arguments: Sequence[str],
    *,
    runner: Runner = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    try:
        return runner(
            list(arguments),
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return subprocess.CompletedProcess(arguments, 127, "", str(exc))


def _nmcli_value(
    field: str,
    object_type: str,
    name: str,
    *,
    runner: Runner,
) -> tuple[bool, str]:
    result = _run(("nmcli", "-g", field, object_type, "show", name), runner=runner)
    return result.returncode == 0, result.stdout.strip()


def _check(check_id: str, expected: Any, observed: Any, ok: bool) -> dict[str, Any]:
    return {
        "id": check_id,
        "ok": ok,
        "expected": expected,
        "observed": observed,
    }


def _default_route_devices(
    *,
    runner: Runner,
) -> tuple[bool, list[str], str]:
    result = _run(("ip", "-j", "-4", "route", "show", "default"), runner=runner)
    if result.returncode != 0:
        return False, [], (result.stderr or result.stdout).strip()
    try:
        routes = json.loads(result.stdout)
        devices = sorted({
            str(route["dev"])
            for route in routes
            if isinstance(route, dict) and route.get("dev")
        })
    except (json.JSONDecodeError, TypeError, KeyError):
        return False, [], "default-route output was not valid JSON"
    if not devices:
        return False, [], "no IPv4 default-route interface was reported"
    return True, devices, ""


def verify_rail(
    config: RailConfig,
    *,
    runner: Runner = subprocess.run,
    sys_net_root: Path = Path("/sys/class/net"),
    sys_rdma_root: Path = Path("/sys/class/infiniband"),
) -> dict[str, Any]:
    """Inspect the persistent profile and active data path without changing them."""
    checks: list[dict[str, Any]] = []

    expected_fields = (
        ("connection.interface-name", config.interface),
        ("connection.autoconnect", "yes"),
        ("connection.autoconnect-priority", "100"),
        ("connection.autoconnect-retries", "0"),
        ("ipv4.method", "manual"),
        ("ipv4.addresses", str(config.address)),
        ("ipv4.never-default", "yes"),
        ("ipv6.method", "disabled"),
        ("802-3-ethernet.mtu", str(config.mtu)),
    )
    for field, expected in expected_fields:
        command_ok, observed = _nmcli_value(
            field,
            "connection",
            config.connection_name,
            runner=runner,
        )
        checks.append(
            _check(f"profile.{field}", expected, observed, command_ok and observed == expected)
        )

    active_ok, active_name = _nmcli_value(
        "GENERAL.CONNECTION",
        "device",
        config.interface,
        runner=runner,
    )
    checks.append(
        _check(
            "live.active_connection",
            config.connection_name,
            active_name,
            active_ok and active_name == config.connection_name,
        )
    )

    route_ok, route_devices, route_detail = _default_route_devices(runner=runner)
    observed_routes: Any = route_devices if route_ok else route_detail
    checks.append(
        _check(
            "safety.rail_not_default_route",
            "absent",
            observed_routes,
            route_ok and config.interface not in route_devices,
        )
    )
    checks.append(
        _check(
            "safety.management_default_route",
            config.management_interface,
            observed_routes,
            route_ok and config.management_interface in route_devices,
        )
    )

    address_result = _run(
        ("ip", "-j", "-4", "address", "show", "dev", config.interface),
        runner=runner,
    )
    live_cidrs: list[str] = []
    try:
        documents = json.loads(address_result.stdout) if address_result.returncode == 0 else []
        for document in documents:
            for value in document.get("addr_info", []):
                if value.get("family") == "inet":
                    live_cidrs.append(f"{value.get('local')}/{value.get('prefixlen')}")
    except (json.JSONDecodeError, TypeError, AttributeError):
        live_cidrs = []
    checks.append(
        _check(
            "live.address",
            str(config.address),
            live_cidrs,
            str(config.address) in live_cidrs,
        )
    )

    link_result = _run(
        ("ip", "-j", "link", "show", "dev", config.interface),
        runner=runner,
    )
    live_mtu: int | None = None
    live_state: str | None = None
    try:
        link_document = json.loads(link_result.stdout)[0]
        live_mtu = int(link_document.get("mtu"))
        live_state = str(link_document.get("operstate", "")).upper()
    except (json.JSONDecodeError, IndexError, KeyError, TypeError, ValueError):
        pass
    checks.append(_check("live.mtu", config.mtu, live_mtu, live_mtu == config.mtu))
    checks.append(_check("live.link", "UP", live_state, live_state == "UP"))

    gid_root = (
        sys_rdma_root
        / config.rdma_device
        / "ports"
        / str(config.rdma_port)
    )
    gid_path = gid_root / "gids" / str(config.gid_index)
    gid_type_path = gid_root / "gid_attrs" / "types" / str(config.gid_index)
    gid_netdev_path = gid_root / "gid_attrs" / "ndevs" / str(config.gid_index)

    def read(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return ""

    observed_gid = read(gid_path).lower()
    observed_type = " ".join(read(gid_type_path).split())
    observed_netdev = read(gid_netdev_path)
    checks.append(
        _check("roce.gid", config.expected_gid, observed_gid, observed_gid == config.expected_gid)
    )
    checks.append(
        _check("roce.type", "RoCE v2", observed_type, observed_type == "RoCE v2")
    )
    checks.append(
        _check("roce.netdev", config.interface, observed_netdev, observed_netdev == config.interface)
    )

    netdev_exists = (sys_net_root / config.interface).exists()
    management_exists = (sys_net_root / config.management_interface).exists()
    checks.append(_check("live.interface", True, netdev_exists, netdev_exists))
    checks.append(
        _check(
            "safety.management_interface",
            True,
            management_exists,
            management_exists,
        )
    )

    observed_port_state = read(gid_root / "state")
    normalized_port_state = observed_port_state.split(":", 1)[-1].strip().upper()
    observed_link_layer = read(gid_root / "link_layer")
    checks.append(
        _check(
            "roce.port_state",
            "ACTIVE",
            observed_port_state,
            normalized_port_state == "ACTIVE",
        )
    )
    checks.append(
        _check(
            "roce.link_layer",
            "Ethernet",
            observed_link_layer,
            observed_link_layer == "Ethernet",
        )
    )

    ping_arguments = (
        "ping",
        "-4",
        "-I",
        config.interface,
        "-M",
        "do",
        "-s",
        str(config.jumbo_payload_bytes),
        "-c",
        "1",
        "-W",
        str(config.ping_timeout_seconds),
        str(config.peer_address),
    )
    ping_result = _run(ping_arguments, runner=runner)
    ping_detail = (ping_result.stdout or ping_result.stderr).strip()
    checks.append(
        _check("peer.jumbo_ping", "success", ping_detail, ping_result.returncode == 0)
    )

    return {
        "schema": VERIFY_SCHEMA,
        "mode": "verify",
        "mutates_host": False,
        "connection_name": config.connection_name,
        "interface": config.interface,
        "passed": all(item["ok"] for item in checks),
        "checks": checks,
    }


def apply_rail(
    config: RailConfig,
    *,
    runner: Runner = subprocess.run,
    sys_net_root: Path = Path("/sys/class/net"),
    sys_rdma_root: Path = Path("/sys/class/infiniband"),
) -> dict[str, Any]:
    """Create or update one persistent profile, activate it, and verify it."""
    missing = [name for name in ("nmcli", "ip", "ping") if shutil.which(name) is None]
    if missing:
        raise RailConfigError("required command(s) not found: " + ", ".join(missing))
    missing_interfaces = [
        name
        for name in (config.interface, config.management_interface)
        if not (sys_net_root / name).exists()
    ]
    if missing_interfaces:
        raise RailConfigError(
            "interface(s) not present; no NetworkManager changes were made: "
            + ", ".join(missing_interfaces)
        )
    route_ok, route_devices, route_detail = _default_route_devices(runner=runner)
    if not route_ok:
        raise RailConfigError(
            "IPv4 default route could not be verified; no NetworkManager changes "
            f"were made: {route_detail}"
        )
    if config.interface in route_devices:
        raise RailConfigError(
            f"rail interface {config.interface!r} owns an IPv4 default route; "
            "no NetworkManager changes were made"
        )
    if config.management_interface not in route_devices:
        raise RailConfigError(
            f"declared management interface {config.management_interface!r} does "
            "not own an IPv4 default route; no NetworkManager changes were made"
        )

    profile_probe = _run(
        (
            "nmcli",
            "-g",
            "connection.interface-name",
            "connection",
            "show",
            config.connection_name,
        ),
        runner=runner,
    )
    if profile_probe.returncode == 0:
        existing_interface = profile_probe.stdout.strip()
        if existing_interface != config.interface:
            raise RailConfigError(
                f"connection {config.connection_name!r} belongs to interface "
                f"{existing_interface!r}, not {config.interface!r}"
            )
        action = "modified"
        command = modify_profile_command(config)
    else:
        action = "created"
        command = add_profile_command(config)

    changed = _run(command, runner=runner)
    if changed.returncode != 0:
        detail = (changed.stderr or changed.stdout).strip()
        raise RailConfigError(f"NetworkManager profile was not {action}: {detail}")
    activated = _run(activate_profile_command(config), runner=runner)
    if activated.returncode != 0:
        detail = (activated.stderr or activated.stdout).strip()
        raise RailConfigError(f"NetworkManager profile was not activated: {detail}")

    verification = verify_rail(
        config,
        runner=runner,
        sys_net_root=sys_net_root,
        sys_rdma_root=sys_rdma_root,
    )
    return {
        "schema": APPLY_SCHEMA,
        "mode": "execute",
        "mutates_host": True,
        "profile_action": action,
        "connection_name": config.connection_name,
        "interface": config.interface,
        "passed": verification["passed"],
        "verification": verification,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interface", required=True)
    parser.add_argument("--management-interface", required=True)
    parser.add_argument("--address-cidr", required=True)
    parser.add_argument("--peer-address", required=True)
    parser.add_argument("--rdma-device", required=True)
    parser.add_argument("--rdma-port", type=int, default=1)
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument("--mtu", type=int, default=9000)
    parser.add_argument("--connection-name")
    parser.add_argument("--ping-timeout-seconds", type=int, default=3)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--confirmation")
    parser.add_argument("--output", type=Path)
    return parser


def _emit(document: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.confirmation and not args.execute:
        parser.error("--confirmation is only valid with --execute")
    if args.execute and args.confirmation != CONFIRMATION:
        parser.error(f"--execute requires --confirmation {CONFIRMATION}")
    try:
        config = rail_config(
            interface=args.interface,
            management_interface=args.management_interface,
            address_cidr=args.address_cidr,
            peer_address=args.peer_address,
            rdma_device=args.rdma_device,
            rdma_port=args.rdma_port,
            gid_index=args.gid_index,
            mtu=args.mtu,
            connection_name=args.connection_name,
            ping_timeout_seconds=args.ping_timeout_seconds,
        )
        if args.execute:
            get_euid = getattr(os, "geteuid", None)
            if get_euid is None or get_euid() != 0:
                raise RailConfigError("--execute must run as root, normally through sudo")
            document = apply_rail(config)
        elif args.verify:
            document = verify_rail(config)
        else:
            document = plan_document(config)
    except RailConfigError as exc:
        parser.error(str(exc))
    _emit(document, args.output)
    return 0 if document.get("passed", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
