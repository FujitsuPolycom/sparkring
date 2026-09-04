from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import configure_sircl_rail as rail  # noqa: E402


SCRIPT = Path(__file__).with_name("configure_sircl_rail.py")


def _base_args() -> list[str]:
    return [
        "--interface",
        "enp2s0f1",
        "--management-interface",
        "enp1s0",
        "--address-cidr",
        "198.51.100.1/30",
        "--peer-address",
        "198.51.100.2",
        "--rdma-device",
        "mlx5_1",
        "--rdma-port",
        "1",
        "--gid-index",
        "3",
        "--mtu",
        "9000",
    ]


def _run(*extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *_base_args(), *extra],
        text=True,
        capture_output=True,
        check=False,
    )


def test_default_mode_prints_a_non_mutating_persistent_profile_plan() -> None:
    result = _run()

    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["schema"] == "sparkring-sircl-rail-plan/v1"
    assert plan["mode"] == "plan"
    assert plan["mutates_host"] is False
    assert plan["connection_name"] == "sparkring-sircl-enp2s0f1"
    assert plan["interface"] == "enp2s0f1"
    assert plan["management_interface"] == "enp1s0"
    assert plan["address_cidr"] == "198.51.100.1/30"
    assert plan["peer_address"] == "198.51.100.2"
    assert plan["mtu"] == 9000
    assert plan["profile_contract"] == {
        "autoconnect": True,
        "autoconnect_priority": 100,
        "autoconnect_retries": "forever",
        "ipv4_method": "manual",
        "ipv4_never_default": True,
        "ipv6_method": "disabled",
    }
    assert plan["roce"] == {
        "device": "mlx5_1",
        "port": 1,
        "gid_index": 3,
        "expected_gid": "0000:0000:0000:0000:0000:ffff:c633:6401",
        "expected_type": "RoCE v2",
    }
    assert plan["jumbo_ping"]["payload_bytes"] == 8972


@pytest.mark.parametrize(
    ("flag", "value", "message"),
    [
        ("--peer-address", "203.0.113.2", "same IPv4 subnet"),
        ("--peer-address", "198.51.100.1", "must differ"),
        ("--address-cidr", "198.51.100.0/30", "network address"),
        ("--mtu", "1200", "MTU must be between"),
        ("--management-interface", "enp2s0f1", "must differ"),
    ],
)
def test_invalid_rail_contract_is_rejected(
    flag: str,
    value: str,
    message: str,
) -> None:
    args = _base_args()
    args[args.index(flag) + 1] = value
    result = subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert message in result.stderr


def test_execute_requires_the_exact_confirmation() -> None:
    missing = _run("--execute")
    wrong = _run("--execute", "--confirmation", "YES")

    assert missing.returncode == 2
    assert "--confirmation CONFIGURE_SIRCL_RAIL" in missing.stderr
    assert wrong.returncode == 2
    assert "--confirmation CONFIGURE_SIRCL_RAIL" in wrong.stderr


def test_confirmation_is_rejected_outside_execute_mode() -> None:
    result = _run("--confirmation", "CONFIGURE_SIRCL_RAIL")

    assert result.returncode == 2
    assert "only valid with --execute" in result.stderr


def test_verify_and_execute_are_mutually_exclusive() -> None:
    result = _run(
        "--verify",
        "--execute",
        "--confirmation",
        "CONFIGURE_SIRCL_RAIL",
    )

    assert result.returncode == 2
    assert "not allowed with argument" in result.stderr


def _verification_runner(
    config: rail.RailConfig,
) -> tuple[Callable[..., subprocess.CompletedProcess[str]], list[list[str]]]:
    fields = {
        "connection.interface-name": config.interface,
        "connection.autoconnect": "yes",
        "connection.autoconnect-priority": "100",
        "connection.autoconnect-retries": "0",
        "ipv4.method": "manual",
        "ipv4.addresses": str(config.address),
        "ipv4.never-default": "yes",
        "ipv6.method": "disabled",
        "802-3-ethernet.mtu": str(config.mtu),
    }
    observed: list[list[str]] = []

    def run(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        observed.append(arguments)
        if arguments[:2] == ["nmcli", "-g"]:
            field = arguments[2]
            value = (
                config.connection_name
                if field == "GENERAL.CONNECTION"
                else fields[field]
            )
            return subprocess.CompletedProcess(arguments, 0, value + "\n", "")
        if arguments[:5] == ["ip", "-j", "-4", "address", "show"]:
            payload = [{
                "ifname": config.interface,
                "addr_info": [{
                    "family": "inet",
                    "local": str(config.address.ip),
                    "prefixlen": config.address.network.prefixlen,
                }],
            }]
            return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), "")
        if arguments[:6] == ["ip", "-j", "-4", "route", "show", "default"]:
            payload = [{"dst": "default", "dev": config.management_interface}]
            return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), "")
        if arguments[:4] == ["ip", "-j", "link", "show"]:
            payload = [{
                "ifname": config.interface,
                "mtu": config.mtu,
                "operstate": "UP",
            }]
            return subprocess.CompletedProcess(arguments, 0, json.dumps(payload), "")
        if arguments[0] == "ping":
            return subprocess.CompletedProcess(arguments, 0, "1 packets received", "")
        raise AssertionError(arguments)

    return run, observed


def _write_sysfs(root: Path, config: rail.RailConfig) -> tuple[Path, Path]:
    net = root / "net"
    (net / config.interface).mkdir(parents=True)
    (net / config.management_interface).mkdir(parents=True)
    gid_root = root / "infiniband" / config.rdma_device / "ports" / "1"
    (gid_root / "gids").mkdir(parents=True)
    (gid_root / "gid_attrs" / "types").mkdir(parents=True)
    (gid_root / "gid_attrs" / "ndevs").mkdir(parents=True)
    (gid_root / "gids" / "3").write_text(config.expected_gid + "\n")
    (gid_root / "gid_attrs" / "types" / "3").write_text("RoCE v2\n")
    (gid_root / "gid_attrs" / "ndevs" / "3").write_text(
        config.interface + "\n"
    )
    (gid_root / "state").write_text("4: ACTIVE\n")
    (gid_root / "link_layer").write_text("Ethernet\n")
    return net, root / "infiniband"


def test_verification_covers_profile_live_link_gid_and_jumbo_peer(
    tmp_path: Path,
) -> None:
    config = rail.rail_config(
        interface="enp2s0f1",
        management_interface="enp1s0",
        address_cidr="198.51.100.1/30",
        peer_address="198.51.100.2",
        rdma_device="mlx5_1",
    )
    runner, commands = _verification_runner(config)
    net_root, rdma_root = _write_sysfs(tmp_path, config)

    result = rail.verify_rail(
        config,
        runner=runner,
        sys_net_root=net_root,
        sys_rdma_root=rdma_root,
    )

    assert result["passed"] is True
    assert {check["id"] for check in result["checks"]} == {
        "profile.connection.interface-name",
        "profile.connection.autoconnect",
        "profile.connection.autoconnect-priority",
        "profile.connection.autoconnect-retries",
        "profile.ipv4.method",
        "profile.ipv4.addresses",
        "profile.ipv4.never-default",
        "profile.ipv6.method",
        "profile.802-3-ethernet.mtu",
        "live.active_connection",
        "safety.rail_not_default_route",
        "safety.management_default_route",
        "live.address",
        "live.mtu",
        "live.link",
        "roce.gid",
        "roce.type",
        "roce.netdev",
        "roce.port_state",
        "roce.link_layer",
        "live.interface",
        "safety.management_interface",
        "peer.jumbo_ping",
    }
    ping = next(command for command in commands if command[0] == "ping")
    assert ping == [
        "ping",
        "-4",
        "-I",
        "enp2s0f1",
        "-M",
        "do",
        "-s",
        "8972",
        "-c",
        "1",
        "-W",
        "3",
        "198.51.100.2",
    ]


def test_verification_reports_a_missing_gid_without_mutation(tmp_path: Path) -> None:
    config = rail.rail_config(
        interface="enp2s0f1",
        management_interface="enp1s0",
        address_cidr="198.51.100.1/30",
        peer_address="198.51.100.2",
        rdma_device="mlx5_1",
    )
    runner, _commands = _verification_runner(config)
    net_root, rdma_root = _write_sysfs(tmp_path, config)
    (rdma_root / "mlx5_1" / "ports" / "1" / "gids" / "3").unlink()

    result = rail.verify_rail(
        config,
        runner=runner,
        sys_net_root=net_root,
        sys_rdma_root=rdma_root,
    )

    assert result["passed"] is False
    gid_check = next(check for check in result["checks"] if check["id"] == "roce.gid")
    assert gid_check["ok"] is False
    assert gid_check["observed"] == ""


@pytest.mark.parametrize("missing", ["enp2s0f1", "enp1s0"])
def test_execute_checks_both_rail_and_management_interfaces_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    missing: str,
) -> None:
    config = rail.rail_config(
        interface="enp2s0f1",
        management_interface="enp1s0",
        address_cidr="198.51.100.1/30",
        peer_address="198.51.100.2",
        rdma_device="mlx5_1",
    )
    net_root = tmp_path / "net"
    for name in {config.interface, config.management_interface} - {missing}:
        (net_root / name).mkdir(parents=True)
    commands: list[list[str]] = []

    def runner(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(rail.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(rail.RailConfigError, match="no NetworkManager changes"):
        rail.apply_rail(
            config,
            runner=runner,
            sys_net_root=net_root,
            sys_rdma_root=tmp_path / "infiniband",
        )

    assert commands == []


@pytest.mark.parametrize(
    ("declared_management", "default_device", "message"),
    [
        ("enp3s0", "enp2s0f1", "owns an IPv4 default route"),
        ("enp3s0", "enp1s0", "does not own an IPv4 default route"),
    ],
)
def test_execute_proves_the_live_management_route_before_nmcli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    declared_management: str,
    default_device: str,
    message: str,
) -> None:
    config = rail.rail_config(
        interface="enp2s0f1",
        management_interface=declared_management,
        address_cidr="198.51.100.1/30",
        peer_address="198.51.100.2",
        rdma_device="mlx5_1",
    )
    net_root = tmp_path / "net"
    for name in {config.interface, config.management_interface, default_device}:
        (net_root / name).mkdir(parents=True, exist_ok=True)
    commands: list[list[str]] = []

    def runner(arguments: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        commands.append(arguments)
        if arguments[:6] == ["ip", "-j", "-4", "route", "show", "default"]:
            return subprocess.CompletedProcess(
                arguments,
                0,
                json.dumps([{"dst": "default", "dev": default_device}]),
                "",
            )
        raise AssertionError(arguments)

    monkeypatch.setattr(rail.shutil, "which", lambda name: f"/usr/bin/{name}")
    with pytest.raises(rail.RailConfigError, match=message):
        rail.apply_rail(
            config,
            runner=runner,
            sys_net_root=net_root,
            sys_rdma_root=tmp_path / "infiniband",
        )

    assert commands == [["ip", "-j", "-4", "route", "show", "default"]]


@pytest.mark.parametrize(
    ("filename", "contents", "check_id"),
    [
        ("state", "2: INIT\n", "roce.port_state"),
        ("link_layer", "InfiniBand\n", "roce.link_layer"),
    ],
)
def test_verification_rejects_inactive_or_non_ethernet_rdma_port(
    tmp_path: Path,
    filename: str,
    contents: str,
    check_id: str,
) -> None:
    config = rail.rail_config(
        interface="enp2s0f1",
        management_interface="enp1s0",
        address_cidr="198.51.100.1/30",
        peer_address="198.51.100.2",
        rdma_device="mlx5_1",
    )
    runner, _commands = _verification_runner(config)
    net_root, rdma_root = _write_sysfs(tmp_path, config)
    (rdma_root / "mlx5_1" / "ports" / "1" / filename).write_text(contents)

    result = rail.verify_rail(
        config,
        runner=runner,
        sys_net_root=net_root,
        sys_rdma_root=rdma_root,
    )

    assert result["passed"] is False
    check = next(item for item in result["checks"] if item["id"] == check_id)
    assert check["ok"] is False
