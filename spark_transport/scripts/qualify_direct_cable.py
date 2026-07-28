#!/usr/bin/env python3
"""Fail-closed qualification for one direct-attached Spark cable.

The controller uses SSH for read-only inspection and to run an existing
SparkRing data-plane probe. It never changes an address, route, MTU, qdisc,
offload, IRQ, driver binding, or model process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SCHEMA = "sparkring.direct-cable-qualification.v1"
EXIT_QUALIFIED = 0
EXIT_FAILED = 1
EXIT_ERROR = 2
EXIT_INCOMPLETE = 3
SSH_TARGET_RE = re.compile(r"^[A-Za-z0-9_.@:-]+$")
INTERFACE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
RDMA_DEVICE_RE = re.compile(r"^[A-Za-z0-9_.:-]+$")
ABSOLUTE_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9_./+-]+$")

RAW_ERROR_FIELDS = (
    "header_crc_errors",
    "fragment_crc_errors",
    "payload_crc_errors",
    "missing_fragments",
    "lost_messages",
    "timeouts",
    "kernel_drops",
)

PHY_COUNTER_MARKERS = (
    "crc",
    "align",
    "symbol",
    "carrier",
    "fec_uncorrectable",
    "uncorrectable",
    "mac_error",
    "frame_error",
    "length_error",
    "rx_errors",
    "tx_errors",
)

PRESSURE_COUNTER_MARKERS = (
    "drop",
    "discard",
    "miss",
    "fifo",
    "overrun",
    "no_buffer",
    "timeout",
    "freeze",
)


REMOTE_SNAPSHOT = r"""
import json
import os
import re
import subprocess
import sys

iface, local_ip, peer_ip, tier, rdma_device, gid_index = sys.argv[1:]

def read_text(path):
    try:
        with open(path, "r", encoding="utf-8") as stream:
            return stream.read().strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None

def run(command):
    try:
        completed = subprocess.run(
            command, text=True, capture_output=True, timeout=15, check=False)
        return {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"returncode": 127, "stdout": "", "stderr": str(error)}

base = "/sys/class/net/" + iface
result = {
    "hostname": os.uname().nodename,
    "interface": iface,
    "interface_exists": os.path.isdir(base),
    "expected_local_ip": local_ip,
    "peer_ip": peer_ip,
    "tier": tier,
}

if not result["interface_exists"]:
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0)

for name in ("address", "mtu", "operstate", "carrier", "speed", "duplex"):
    result[name] = read_text(base + "/" + name)

device_path = os.path.realpath(base + "/device")
driver_path = os.path.realpath(base + "/device/driver")
result["pci_device"] = os.path.basename(device_path)
result["driver"] = os.path.basename(driver_path)

stats = {}
stats_dir = base + "/statistics"
try:
    for name in sorted(os.listdir(stats_dir)):
        value = read_text(stats_dir + "/" + name)
        if value is not None and re.fullmatch(r"-?[0-9]+", value):
            stats["sysfs." + name] = int(value)
except OSError:
    pass

ethtool = run(["ethtool", iface])
result["ethtool"] = ethtool
if result.get("speed") in (None, "-1"):
    match = re.search(r"^\s*Speed:\s*([0-9]+)Mb/s\s*$",
                      ethtool["stdout"], re.MULTILINE)
    if match:
        result["speed"] = match.group(1)

ethtool_stats = run(["ethtool", "-S", iface])
result["ethtool_stats_status"] = {
    "returncode": ethtool_stats["returncode"],
    "stderr": ethtool_stats["stderr"],
}
for line in ethtool_stats["stdout"].splitlines():
    match = re.match(r"^\s*([^:]+):\s*(-?[0-9]+)\s*$", line)
    if match:
        stats["ethtool." + match.group(1).strip()] = int(match.group(2))
result["counters"] = stats

address = run(["ip", "-j", "address", "show", "dev", iface])
route = run(["ip", "-j", "route", "get", peer_ip])
result["address_query"] = address
result["route_query"] = route
try:
    address_json = json.loads(address["stdout"])
except json.JSONDecodeError:
    address_json = []
try:
    route_json = json.loads(route["stdout"])
except json.JSONDecodeError:
    route_json = []

result["ip_addresses"] = sorted(
    entry.get("local")
    for link in address_json
    for entry in link.get("addr_info", [])
    if entry.get("family") == "inet" and entry.get("local")
)
result["route"] = route_json[0] if route_json else None

ping = run(["ping", "-n", "-I", local_ip, "-c", "5", "-W", "1", peer_ip])
result["ping"] = ping
received = re.search(r"([0-9]+) received", ping["stdout"])
result["ping_received"] = int(received.group(1)) if received else 0
neighbour = run(["ip", "-j", "neigh", "show", "to", peer_ip, "dev", iface])
result["neighbour_query"] = neighbour
try:
    neighbour_json = json.loads(neighbour["stdout"])
except json.JSONDecodeError:
    neighbour_json = []
result["neighbour"] = neighbour_json[0] if neighbour_json else None

if tier == "roce200":
    ib_base = "/sys/class/infiniband/" + rdma_device
    result["rdma"] = {
        "device": rdma_device,
        "exists": os.path.isdir(ib_base),
        "gid_index": int(gid_index),
        "link_layer": read_text(ib_base + "/ports/1/link_layer"),
        "state": read_text(ib_base + "/ports/1/state"),
        "physical_state": read_text(ib_base + "/ports/1/phys_state"),
        "gid": read_text(ib_base + "/ports/1/gids/" + gid_index),
        "gid_type": read_text(
            ib_base + "/ports/1/gid_attrs/types/" + gid_index),
        "gid_netdev": read_text(
            ib_base + "/ports/1/gid_attrs/ndevs/" + gid_index),
    }

print(json.dumps(result, sort_keys=True))
"""


@dataclass(frozen=True)
class Endpoint:
    label: str
    ssh: str
    interface: str
    ip: str
    rdma_device: str | None
    cpu: int | None


class QualificationError(RuntimeError):
    """A controlled qualification setup or remote-execution failure."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def validate_identifier(value: str, pattern: re.Pattern[str], name: str) -> str:
    if not pattern.fullmatch(value):
        raise QualificationError(f"unsafe or invalid {name}: {value!r}")
    return value


def validate_ip(value: str, name: str) -> str:
    try:
        parsed = ipaddress.ip_address(value)
    except ValueError as error:
        raise QualificationError(f"invalid {name}: {value!r}") from error
    if parsed.version != 4:
        raise QualificationError(f"{name} must be IPv4")
    return str(parsed)


def validate_remote_path(value: str) -> str:
    if not ABSOLUTE_REMOTE_PATH_RE.fullmatch(value) or ".." in Path(value).parts:
        raise QualificationError("--probe-binary must be a simple absolute remote path")
    return value


def ssh_argv(target: str, remote_argv: Sequence[str]) -> list[str]:
    validate_identifier(target, SSH_TARGET_RE, "SSH target")
    command = " ".join(shlex.quote(part) for part in remote_argv)
    return [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        target,
        command,
    ]


def run_remote(
    endpoint: Endpoint,
    remote_argv: Sequence[str],
    *,
    input_text: str | None = None,
    timeout: int = 30,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ssh_argv(endpoint.ssh, remote_argv),
        input=input_text,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def remote_snapshot(
    endpoint: Endpoint,
    peer: Endpoint,
    tier: str,
    gid_index: int,
) -> dict[str, Any]:
    encoded = REMOTE_SNAPSHOT
    command = [
        "python3",
        "-c",
        encoded,
        endpoint.interface,
        endpoint.ip,
        peer.ip,
        tier,
        endpoint.rdma_device or "-",
        str(gid_index),
    ]
    completed = run_remote(endpoint, command, timeout=45)
    if completed.returncode != 0:
        raise QualificationError(
            f"{endpoint.label} snapshot failed: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise QualificationError(
            f"{endpoint.label} snapshot returned invalid JSON"
        ) from error


def gate(
    name: str,
    passed: bool,
    evidence: Any,
    *,
    domain: str,
    hard: bool = True,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "hard": hard,
        "domain": domain,
        "evidence": evidence,
    }


def integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def neighbour_state_usable(value: Any) -> bool:
    if isinstance(value, list):
        states = {str(item).upper() for item in value}
    else:
        states = {str(value or "").upper()}
    return not states.intersection({"FAILED", "INCOMPLETE"})


def evaluate_snapshot(
    snapshot: dict[str, Any],
    endpoint: Endpoint,
    *,
    tier: str,
    expected_mtu: int,
    expected_speed_mbps: int,
    gid_index: int,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    checks.append(
        gate(
            f"{endpoint.label}.interface_exists",
            snapshot.get("interface_exists") is True,
            snapshot.get("interface"),
            domain="configuration",
        )
    )
    checks.append(
        gate(
            f"{endpoint.label}.carrier",
            str(snapshot.get("carrier")) == "1" and snapshot.get("operstate") == "up",
            {
                "carrier": snapshot.get("carrier"),
                "operstate": snapshot.get("operstate"),
            },
            domain="link_or_cable",
        )
    )
    checks.append(
        gate(
            f"{endpoint.label}.speed",
            integer_or_none(snapshot.get("speed")) == expected_speed_mbps,
            {
                "actual_mbps": integer_or_none(snapshot.get("speed")),
                "expected_mbps": expected_speed_mbps,
            },
            domain="link_or_cable",
        )
    )
    checks.append(
        gate(
            f"{endpoint.label}.mtu",
            integer_or_none(snapshot.get("mtu")) == expected_mtu,
            {
                "actual": integer_or_none(snapshot.get("mtu")),
                "expected": expected_mtu,
            },
            domain="configuration",
        )
    )
    addresses = snapshot.get("ip_addresses") or []
    checks.append(
        gate(
            f"{endpoint.label}.exact_ip",
            endpoint.ip in addresses,
            {"expected": endpoint.ip, "actual": addresses},
            domain="configuration",
        )
    )
    route = snapshot.get("route") or {}
    checks.append(
        gate(
            f"{endpoint.label}.direct_route",
            route.get("dev") == endpoint.interface
            and route.get("prefsrc") == endpoint.ip,
            {
                "expected_interface": endpoint.interface,
                "expected_source": endpoint.ip,
                "actual": route,
            },
            domain="configuration",
        )
    )
    checks.append(
        gate(
            f"{endpoint.label}.peer_reachable",
            snapshot.get("ping", {}).get("returncode") == 0
            and integer_or_none(snapshot.get("ping_received")) == 5,
            {
                "received": snapshot.get("ping_received"),
                "stderr": snapshot.get("ping", {}).get("stderr"),
            },
            domain="link_or_cable",
        )
    )

    mac = str(snapshot.get("address") or "")
    mac_ok = bool(
        re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", mac.lower())
        and mac != "00:00:00:00:00:00"
    )
    checks.append(
        gate(
            f"{endpoint.label}.mac",
            mac_ok,
            mac,
            domain="configuration",
        )
    )

    if tier == "roce200":
        rdma = snapshot.get("rdma") or {}
        checks.extend(
            [
                gate(
                    f"{endpoint.label}.rdma_device",
                    rdma.get("exists") is True
                    and rdma.get("device") == endpoint.rdma_device,
                    rdma,
                    domain="configuration",
                ),
                gate(
                    f"{endpoint.label}.rdma_gid_binding",
                    rdma.get("gid_netdev") == endpoint.interface
                    and rdma.get("gid_type") == "RoCE v2"
                    and rdma.get("link_layer") == "Ethernet",
                    {
                        "gid_index": gid_index,
                        "gid_netdev": rdma.get("gid_netdev"),
                        "gid_type": rdma.get("gid_type"),
                        "link_layer": rdma.get("link_layer"),
                    },
                    domain="configuration",
                ),
                gate(
                    f"{endpoint.label}.rdma_port_active",
                    str(rdma.get("state") or "").startswith("4:")
                    and str(rdma.get("physical_state") or "").startswith("5:"),
                    {
                        "state": rdma.get("state"),
                        "physical_state": rdma.get("physical_state"),
                    },
                    domain="link_or_cable",
                ),
            ]
        )
    return checks


def counter_deltas(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, dict[str, int]]:
    result = {"phy": {}, "pressure": {}, "other": {}, "reset": {}}
    before_counters = before.get("counters") or {}
    after_counters = after.get("counters") or {}
    for name in sorted(set(before_counters) | set(after_counters)):
        old = integer_or_none(before_counters.get(name))
        new = integer_or_none(after_counters.get(name))
        if old is None or new is None or old == new:
            continue
        delta = new - old
        if delta < 0:
            result["reset"][name] = delta
            continue
        lowered = name.lower()
        if any(marker in lowered for marker in PHY_COUNTER_MARKERS):
            result["phy"][name] = delta
        elif any(marker in lowered for marker in PRESSURE_COUNTER_MARKERS):
            result["pressure"][name] = delta
        else:
            result["other"][name] = delta
    return result


def parse_json_object(output: str, role: str | None = None) -> dict[str, Any]:
    for line in reversed(output.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and (role is None or value.get("role") == role):
            return value
    raise QualificationError("probe did not emit the expected JSON object")


def parse_result_line(output: str) -> dict[str, Any]:
    for line in output.splitlines():
        if not line.startswith("RESULT "):
            continue
        parsed: dict[str, Any] = {}
        for item in line.split()[1:]:
            if "=" not in item:
                continue
            key, value = item.split("=", 1)
            try:
                parsed[key] = float(value) if "." in value else int(value)
            except ValueError:
                parsed[key] = value
        return parsed
    raise QualificationError("RDMA probe did not emit a RESULT line")


def parse_verify_line(output: str) -> dict[str, Any]:
    for line in output.splitlines():
        if not line.startswith("VERIFY "):
            continue
        parsed: dict[str, Any] = {}
        for item in line.split()[1:]:
            if "=" in item:
                key, value = item.split("=", 1)
                parsed[key] = value
        return parsed
    raise QualificationError("RDMA server did not emit a VERIFY line")


def remote_binary_hash(endpoint: Endpoint, binary: str) -> str | None:
    completed = run_remote(
        endpoint,
        [
            "sh",
            "-c",
            f"test -x {shlex.quote(binary)} && sha256sum {shlex.quote(binary)}",
        ],
        timeout=20,
    )
    if completed.returncode != 0:
        return None
    match = re.match(r"^([0-9a-fA-F]{64})\s+", completed.stdout)
    return match.group(1).lower() if match else None


def probe_command(binary: str, arguments: Sequence[str], use_sudo: bool) -> list[str]:
    command = [binary, *arguments]
    return ["sudo", "-n", *command] if use_sudo else command


def start_remote(
    endpoint: Endpoint,
    remote_argv: Sequence[str],
    *,
    remote_timeout_seconds: int = 90,
) -> subprocess.Popen[str]:
    bounded_argv = [
        "timeout",
        "--signal=TERM",
        "--kill-after=5s",
        f"{remote_timeout_seconds}s",
        *remote_argv,
    ]
    return subprocess.Popen(
        ssh_argv(endpoint.ssh, bounded_argv),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def collect_process(
    process: subprocess.Popen[str], timeout: int
) -> tuple[int, str, str]:
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
        raise QualificationError("remote receiver exceeded timeout") from error
    return process.returncode, stdout, stderr


def run_raw_direction(
    sender: Endpoint,
    receiver: Endpoint,
    *,
    sender_snapshot: dict[str, Any],
    receiver_snapshot: dict[str, Any],
    binary: str,
    payload_bytes: int,
    warmup: int,
    iterations: int,
    use_sudo: bool,
    startup_delay: float,
) -> dict[str, Any]:
    total = warmup + iterations
    common_receiver = [
        "receiver",
        "--interface",
        receiver.interface,
        "--local-mac",
        str(receiver_snapshot["address"]),
        "--peer-mac",
        str(sender_snapshot["address"]),
        "--ack-dedicated-interface",
        "--payload-bytes",
        str(payload_bytes),
        "--messages",
        str(total),
        "--spin-us",
        "1000",
        "--rx-ring",
        "v2",
        "--idle-timeout-ms",
        "5000",
    ]
    if receiver.cpu is not None:
        common_receiver.extend(["--cpu", str(receiver.cpu)])
    receiver_process = start_remote(
        receiver, probe_command(binary, common_receiver, use_sudo)
    )
    time.sleep(startup_delay)
    if receiver_process.poll() is not None:
        code, stdout, stderr = collect_process(receiver_process, 5)
        raise QualificationError(
            f"raw receiver exited before sender (rc={code}): "
            f"{stderr.strip() or stdout.strip()}"
        )

    sender_arguments = [
        "sender",
        "--interface",
        sender.interface,
        "--local-mac",
        str(sender_snapshot["address"]),
        "--peer-mac",
        str(receiver_snapshot["address"]),
        "--ack-dedicated-interface",
        "--payload-bytes",
        str(payload_bytes),
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
        "--spin-us",
        "1000",
        "--rx-ring",
        "v2",
        "--timeout-ms",
        "1000",
    ]
    if sender.cpu is not None:
        sender_arguments.extend(["--cpu", str(sender.cpu)])
    timeout = max(60, int(total * 1.1) + 30)
    sender_completed = run_remote(
        sender,
        probe_command(binary, sender_arguments, use_sudo),
        timeout=timeout,
    )
    receiver_code, receiver_stdout, receiver_stderr = collect_process(
        receiver_process, timeout=15
    )
    sender_result = parse_json_object(sender_completed.stdout, "sender")
    receiver_result = parse_json_object(receiver_stdout, "receiver")
    return {
        "direction": f"{sender.label}->{receiver.label}",
        "payload_bytes": payload_bytes,
        "sender_returncode": sender_completed.returncode,
        "receiver_returncode": receiver_code,
        "sender_stderr": sender_completed.stderr.strip(),
        "receiver_stderr": receiver_stderr.strip(),
        "sender": sender_result,
        "receiver": receiver_result,
    }


def run_rdma_direction(
    client: Endpoint,
    server: Endpoint,
    *,
    binary: str,
    payload_bytes: int,
    warmup: int,
    iterations: int,
    gid_index: int,
    port: int,
    startup_delay: float,
) -> dict[str, Any]:
    if client.rdma_device is None or server.rdma_device is None:
        raise QualificationError("RDMA device names are required")
    server_arguments = [
        "--server",
        "--device",
        server.rdma_device,
        "--gid",
        str(gid_index),
        "--control-port",
        str(port),
        "--bytes",
        str(payload_bytes),
        "--memory",
        "host",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    server_process = start_remote(server, [binary, *server_arguments])
    time.sleep(startup_delay)
    if server_process.poll() is not None:
        code, stdout, stderr = collect_process(server_process, 5)
        raise QualificationError(
            f"RDMA server exited before client (rc={code}): "
            f"{stderr.strip() or stdout.strip()}"
        )
    client_arguments = [
        "--client",
        server.ip,
        "--device",
        client.rdma_device,
        "--gid",
        str(gid_index),
        "--control-port",
        str(port),
        "--bytes",
        str(payload_bytes),
        "--memory",
        "host",
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    timeout = max(60, int((warmup + iterations) / 100) + 30)
    client_completed = run_remote(client, [binary, *client_arguments], timeout=timeout)
    server_code, server_stdout, server_stderr = collect_process(
        server_process, timeout=15
    )
    return {
        "direction": f"{client.label}->{server.label}",
        "payload_bytes": payload_bytes,
        "iterations": iterations,
        "client_returncode": client_completed.returncode,
        "server_returncode": server_code,
        "client_stderr": client_completed.stderr.strip(),
        "server_stderr": server_stderr.strip(),
        "result": parse_result_line(client_completed.stdout),
        "verify": parse_verify_line(server_stdout),
    }


def raw_probe_gates(
    runs: Sequence[dict[str, Any]], max_p99_us: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for run in runs:
        identity = f"{run['direction']}.{run['payload_bytes']}"
        sender = run["sender"]
        receiver = run["receiver"]
        errors = {
            f"sender.{name}": integer_or_none(sender.get(name)) or 0
            for name in RAW_ERROR_FIELDS
        }
        errors.update(
            {
                f"receiver.{name}": integer_or_none(receiver.get(name)) or 0
                for name in RAW_ERROR_FIELDS
            }
        )
        integrity = (
            run["sender_returncode"] == 0
            and run["receiver_returncode"] == 0
            and sender.get("valid") is True
            and receiver.get("valid") is True
            and all(value == 0 for value in errors.values())
        )
        checks.append(
            gate(
                f"raw.{identity}.integrity",
                integrity,
                {
                    "sender_returncode": run["sender_returncode"],
                    "receiver_returncode": run["receiver_returncode"],
                    "errors": errors,
                },
                domain="cable_or_phy",
            )
        )
        p99 = float(sender.get("p99_us", float("inf")))
        checks.append(
            gate(
                f"raw.{identity}.latency",
                p99 <= max_p99_us,
                {
                    "p99_us": p99,
                    "target_us": max_p99_us,
                    "note": (
                        "AF_PACKET latency includes kernel/NAPI/software overhead; "
                        "a miss without integrity errors does not condemn the cable"
                    ),
                },
                domain="software_latency",
                hard=False,
            )
        )
    return checks


def rdma_probe_gates(
    runs: Sequence[dict[str, Any]], max_p99_us: float
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for run in runs:
        identity = f"{run['direction']}.{run['payload_bytes']}"
        result = run["result"]
        verify = run["verify"]
        integrity = (
            run["client_returncode"] == 0
            and run["server_returncode"] == 0
            and verify.get("correct") == "true"
            and integer_or_none(result.get("samples")) == run["iterations"]
        )
        checks.append(
            gate(
                f"rdma.{identity}.integrity",
                integrity,
                {
                    "client_returncode": run["client_returncode"],
                    "server_returncode": run["server_returncode"],
                    "verify": verify,
                    "samples": result.get("samples"),
                    "expected_samples": run["iterations"],
                },
                domain="cable_or_phy",
            )
        )
        p99 = float(result.get("p99_us", float("inf")))
        checks.append(
            gate(
                f"rdma.{identity}.latency",
                p99 <= max_p99_us,
                {
                    "p99_us": p99,
                    "target_us": max_p99_us,
                    "note": (
                        "RC completion latency includes verbs/CQ software; "
                        "a miss with clean integrity/counters is a path-performance "
                        "failure, not proof of a bad cable"
                    ),
                },
                domain="software_latency",
                hard=False,
            )
        )
    return checks


def parse_payloads(value: str) -> tuple[int, ...]:
    try:
        payloads = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "payloads must be comma-separated integers"
        ) from error
    if not payloads or any(item not in (12288, 16384) for item in payloads):
        raise argparse.ArgumentTypeError(
            "cable qualification payloads must be 12288 and/or 16384"
        )
    return payloads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Qualify one direct-attached 200G RoCE or 10GbE cable without "
            "changing live network/model state. JSON is written to stdout."
        )
    )
    parser.add_argument("--tier", choices=("roce200", "diagonal10"), required=True)
    parser.add_argument("--left", required=True, help="SSH target, e.g. user@host")
    parser.add_argument("--right", required=True, help="SSH target")
    parser.add_argument("--left-interface", required=True)
    parser.add_argument("--right-interface", required=True)
    parser.add_argument("--left-ip", required=True)
    parser.add_argument("--right-ip", required=True)
    parser.add_argument("--expected-mtu", type=int, required=True)
    parser.add_argument("--left-rdma-device")
    parser.add_argument("--right-rdma-device")
    parser.add_argument("--gid-index", type=int, default=3)
    parser.add_argument(
        "--probe-binary",
        help=(
            "absolute path present on both endpoints; defaults to "
            "/tmp/spark_transport_probe or /tmp/ten_gbe_raw_bench"
        ),
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="skip payload test; produces INCOMPLETE exit 3, never qualified",
    )
    parser.add_argument(
        "--payloads",
        type=parse_payloads,
        default=(12288, 16384),
        help="12288,16384 by default",
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--left-cpu", type=int)
    parser.add_argument("--right-cpu", type=int)
    parser.add_argument(
        "--use-sudo",
        action="store_true",
        help="run only the 10GbE raw probe via sudo -n (never sudo SSH itself)",
    )
    parser.add_argument("--startup-delay", type=float, default=2.0)
    parser.add_argument("--base-port", type=int, default=19410)
    parser.add_argument(
        "--max-p99-us",
        type=float,
        help="default 20 us for RoCE, 30 us for raw 10GbE",
    )
    parser.add_argument(
        "--strict-latency",
        action="store_true",
        help="return FAILED when integrity passes but the software latency target misses",
    )
    parser.add_argument("--output", help="also atomically write the JSON result here")
    return parser


def make_endpoint(
    label: str,
    ssh: str,
    interface: str,
    ip: str,
    rdma_device: str | None,
    cpu: int | None,
) -> Endpoint:
    validate_identifier(ssh, SSH_TARGET_RE, f"{label} SSH target")
    validate_identifier(interface, INTERFACE_RE, f"{label} interface")
    if rdma_device is not None:
        validate_identifier(rdma_device, RDMA_DEVICE_RE, f"{label} RDMA device")
    if cpu is not None and cpu < 0:
        raise QualificationError(f"{label} CPU must be nonnegative")
    return Endpoint(
        label, ssh, interface, validate_ip(ip, f"{label} IP"), rdma_device, cpu
    )


def write_result(result: dict[str, Any], output: str | None) -> None:
    text = json.dumps(result, indent=2, sort_keys=True)
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
        temporary.write_text(text + "\n", encoding="utf-8")
        os.replace(temporary, path)
    print(text)


def finalize(
    result: dict[str, Any],
    gates: Sequence[dict[str, Any]],
    *,
    probe_completed: bool,
    strict_latency: bool,
) -> int:
    result["gates"] = list(gates)
    hard_failures = [item for item in gates if item["hard"] and not item["passed"]]
    latency_failures = [
        item
        for item in gates
        if item["domain"] == "software_latency" and not item["passed"]
    ]
    result["cable_qualified"] = probe_completed and not hard_failures
    result["latency_target_met"] = probe_completed and not latency_failures
    result["model_path_ready"] = result["cable_qualified"] and (
        result["latency_target_met"] or not strict_latency
    )

    if not probe_completed and not hard_failures:
        result.update(
            status="incomplete",
            failure_domain="probe_missing",
            exit_code=EXIT_INCOMPLETE,
        )
        return EXIT_INCOMPLETE
    if hard_failures:
        domains = sorted({item["domain"] for item in hard_failures})
        result.update(
            status="failed",
            failure_domain=",".join(domains),
            exit_code=EXIT_FAILED,
        )
        return EXIT_FAILED
    if latency_failures and strict_latency:
        result.update(
            status="failed",
            failure_domain="software_latency",
            exit_code=EXIT_FAILED,
        )
        return EXIT_FAILED
    if latency_failures:
        result.update(
            status="cable_qualified_with_latency_warning",
            failure_domain="software_latency",
            exit_code=EXIT_QUALIFIED,
        )
        return EXIT_QUALIFIED
    result.update(
        status="qualified",
        failure_domain=None,
        exit_code=EXIT_QUALIFIED,
    )
    return EXIT_QUALIFIED


def run(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    if args.expected_mtu <= 0:
        raise QualificationError("--expected-mtu must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise QualificationError("warmup must be nonnegative and iterations positive")
    if not 1 <= args.base_port <= 65531:
        raise QualificationError("--base-port must leave room for all probe runs")
    if args.startup_delay < 0:
        raise QualificationError("--startup-delay must be nonnegative")
    if args.tier == "roce200" and (
        not args.left_rdma_device or not args.right_rdma_device
    ):
        raise QualificationError(
            "roce200 requires --left-rdma-device and --right-rdma-device"
        )
    if args.tier == "diagonal10" and args.use_sudo not in (True, False):
        raise QualificationError("invalid sudo option")

    left = make_endpoint(
        "left",
        args.left,
        args.left_interface,
        args.left_ip,
        args.left_rdma_device,
        args.left_cpu,
    )
    right = make_endpoint(
        "right",
        args.right,
        args.right_interface,
        args.right_ip,
        args.right_rdma_device,
        args.right_cpu,
    )
    if left.ip == right.ip:
        raise QualificationError("left and right IPs must differ")
    expected_speed = 200000 if args.tier == "roce200" else 10000
    max_p99 = args.max_p99_us
    if max_p99 is None:
        max_p99 = 20.0 if args.tier == "roce200" else 30.0
    if max_p99 <= 0:
        raise QualificationError("--max-p99-us must be positive")

    binary = args.probe_binary or (
        "/tmp/spark_transport_probe"
        if args.tier == "roce200"
        else "/tmp/ten_gbe_raw_bench"
    )
    validate_remote_path(binary)

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "started_utc": utc_now(),
        "tier": args.tier,
        "configuration": {
            "left": {
                "ssh": left.ssh,
                "interface": left.interface,
                "ip": left.ip,
                "rdma_device": left.rdma_device,
            },
            "right": {
                "ssh": right.ssh,
                "interface": right.interface,
                "ip": right.ip,
                "rdma_device": right.rdma_device,
            },
            "expected_speed_mbps": expected_speed,
            "expected_mtu": args.expected_mtu,
            "gid_index": args.gid_index if args.tier == "roce200" else None,
            "payloads": list(args.payloads),
            "warmup": args.warmup,
            "iterations": args.iterations,
            "max_p99_us": max_p99,
            "strict_latency": args.strict_latency,
            "probe_binary": binary,
        },
        "safety": {
            "network_configuration_mutations": 0,
            "driver_or_qdisc_mutations": 0,
            "model_process_actions": 0,
            "note": "SSH inspection plus an already-installed userspace probe only",
        },
    }
    gates: list[dict[str, Any]] = []

    progress("Capturing pre-test link state on both endpoints")
    before = {
        "left": remote_snapshot(left, right, args.tier, args.gid_index),
        "right": remote_snapshot(right, left, args.tier, args.gid_index),
    }
    result["preflight"] = before
    gates.extend(
        evaluate_snapshot(
            before["left"],
            left,
            tier=args.tier,
            expected_mtu=args.expected_mtu,
            expected_speed_mbps=expected_speed,
            gid_index=args.gid_index,
        )
    )
    gates.extend(
        evaluate_snapshot(
            before["right"],
            right,
            tier=args.tier,
            expected_mtu=args.expected_mtu,
            expected_speed_mbps=expected_speed,
            gid_index=args.gid_index,
        )
    )
    gates.append(
        gate(
            "peer_mac_distinct",
            before["left"].get("address") != before["right"].get("address"),
            {
                "left": before["left"].get("address"),
                "right": before["right"].get("address"),
            },
            domain="configuration",
        )
    )
    left_neighbour = before["left"].get("neighbour") or {}
    right_neighbour = before["right"].get("neighbour") or {}
    gates.append(
        gate(
            "exact_l2_peers",
            str(left_neighbour.get("lladdr") or "").lower()
            == str(before["right"].get("address") or "").lower()
            and str(right_neighbour.get("lladdr") or "").lower()
            == str(before["left"].get("address") or "").lower()
            and neighbour_state_usable(left_neighbour.get("state"))
            and neighbour_state_usable(right_neighbour.get("state")),
            {
                "left_expected_peer_mac": before["right"].get("address"),
                "left_neighbour": left_neighbour,
                "right_expected_peer_mac": before["left"].get("address"),
                "right_neighbour": right_neighbour,
            },
            domain="configuration",
        )
    )

    preflight_failed = any(item["hard"] and not item["passed"] for item in gates)
    probe_completed = False
    if preflight_failed:
        progress("Preflight failed; payload probe was not started")
        result["probe"] = {"status": "not_run_preflight_failed"}
    elif args.preflight_only:
        progress(
            "Preflight-only mode; no cable can be qualified without payload traffic"
        )
        result["probe"] = {"status": "skipped_by_operator"}
    else:
        progress("Verifying the exact probe binary on both endpoints")
        hashes = {
            "left": remote_binary_hash(left, binary),
            "right": remote_binary_hash(right, binary),
        }
        result["probe_binary_sha256"] = hashes
        if not hashes["left"] or not hashes["right"]:
            result["probe"] = {
                "status": "not_available",
                "note": "Build/install the documented probe at the same path on both endpoints",
            }
        elif hashes["left"] != hashes["right"]:
            gates.append(
                gate(
                    "probe_binary_identical",
                    False,
                    hashes,
                    domain="configuration",
                )
            )
            result["probe"] = {"status": "not_run_hash_mismatch"}
        else:
            gates.append(
                gate(
                    "probe_binary_identical",
                    True,
                    hashes,
                    domain="configuration",
                )
            )
            runs: list[dict[str, Any]] = []
            run_number = 0
            try:
                for payload in args.payloads:
                    for sender, receiver in ((left, right), (right, left)):
                        progress(
                            f"Testing {sender.label}->{receiver.label}, {payload} bytes"
                        )
                        if args.tier == "diagonal10":
                            runs.append(
                                run_raw_direction(
                                    sender,
                                    receiver,
                                    sender_snapshot=before[sender.label],
                                    receiver_snapshot=before[receiver.label],
                                    binary=binary,
                                    payload_bytes=payload,
                                    warmup=args.warmup,
                                    iterations=args.iterations,
                                    use_sudo=args.use_sudo,
                                    startup_delay=args.startup_delay,
                                )
                            )
                        else:
                            runs.append(
                                run_rdma_direction(
                                    sender,
                                    receiver,
                                    binary=binary,
                                    payload_bytes=payload,
                                    warmup=args.warmup,
                                    iterations=args.iterations,
                                    gid_index=args.gid_index,
                                    port=args.base_port + run_number,
                                    startup_delay=args.startup_delay,
                                )
                            )
                        run_number += 1
                probe_completed = True
                result["probe"] = {"status": "completed", "runs": runs}
                if args.tier == "diagonal10":
                    gates.extend(raw_probe_gates(runs, max_p99))
                else:
                    gates.extend(rdma_probe_gates(runs, max_p99))
            except (
                QualificationError,
                subprocess.TimeoutExpired,
                OSError,
            ) as error:
                result["probe"] = {
                    "status": "failed",
                    "error": str(error),
                    "completed_runs": runs,
                }
                gates.append(
                    gate(
                        "probe_execution",
                        False,
                        str(error),
                        domain="probe_execution",
                    )
                )

    progress("Capturing post-test link state and counter deltas")
    after = {
        "left": remote_snapshot(left, right, args.tier, args.gid_index),
        "right": remote_snapshot(right, left, args.tier, args.gid_index),
    }
    result["postflight"] = after
    deltas = {
        "left": counter_deltas(before["left"], after["left"]),
        "right": counter_deltas(before["right"], after["right"]),
    }
    result["counter_deltas"] = deltas
    for endpoint in (left, right):
        endpoint_deltas = deltas[endpoint.label]
        gates.append(
            gate(
                f"{endpoint.label}.no_phy_error_delta",
                not endpoint_deltas["phy"],
                endpoint_deltas["phy"],
                domain="cable_or_phy",
            )
        )
        gates.append(
            gate(
                f"{endpoint.label}.no_counter_reset",
                not endpoint_deltas["reset"],
                endpoint_deltas["reset"],
                domain="instrumentation",
            )
        )
        gates.append(
            gate(
                f"{endpoint.label}.no_pressure_delta",
                not endpoint_deltas["pressure"],
                {
                    "deltas": endpoint_deltas["pressure"],
                    "note": (
                        "drop/miss/overrun counters usually indicate host or ring "
                        "pressure, not a bad cable"
                    ),
                },
                domain="software_pressure",
                hard=False,
            )
        )

    exit_code = finalize(
        result,
        gates,
        probe_completed=probe_completed,
        strict_latency=args.strict_latency,
    )
    result["finished_utc"] = utc_now()
    return result, exit_code


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: dict[str, Any]
    try:
        result, exit_code = run(args)
    except (QualificationError, subprocess.TimeoutExpired, OSError) as error:
        result = {
            "schema": SCHEMA,
            "started_utc": utc_now(),
            "finished_utc": utc_now(),
            "tier": getattr(args, "tier", None),
            "status": "error",
            "failure_domain": "orchestration",
            "error": str(error),
            "cable_qualified": False,
            "latency_target_met": False,
            "model_path_ready": False,
            "exit_code": EXIT_ERROR,
        }
        exit_code = EXIT_ERROR
    write_result(result, args.output)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
