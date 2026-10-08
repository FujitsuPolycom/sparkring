"""Shell commands the installer sends to a Spark.

Every remote action is one ``bash -s`` script over SSH in batch mode, one
command per line, every interpolated value shell-quoted. The read script
(:func:`read_script`) uses read-only commands and needs no root: it is the
READ-ONLY REMOTE part of ``show``, ``diff`` and every dry run. Apply scripts
change host state through ``sudo -n`` (MUTATES HOST), stop at the first
failing command (``set -euo pipefail``) and first check the host's name
(:func:`guard`), so a wrong SSH target changes nothing.

Ownership marks, by which ``down`` removes exactly the installer's objects:

- routes and permanent neighbours carry routing protocol number 82
  (:data:`ROUTE_PROTOCOL`, ``proto 82``);
- relay filters occupy the reserved preferences 11 to 17 of chain 0 on the
  ingress of the four fabric network devices (preference ``10 + k``, handle
  ``k``, protocol ``TAG(k)``);
- marker processes run the marker executable, whose first 15 characters are
  the process name ``pgrep -x`` matches;
- the record ``/run/sparkring-fabric/relay-state.json`` names the Spark's
  group, layout, relay egress, plan digest and the ingress qdiscs the
  installer added. ``/run`` is cleared at boot, like every object of the plan.

Nothing else is changed: not the NetworkManager connections, addresses, links,
other routes, other neighbours, other filters, other qdiscs or other processes.
"""

from __future__ import annotations

import dataclasses
import ipaddress
import json
import posixpath
import re
import shlex
from collections.abc import Iterable, Mapping

from .layouts import FabricError
from .plan import FABRIC_NETDEVS, RDMA_DEVICES, RelayFilter

ROUTE_PROTOCOL = 82
STATE_DIR = "/run/sparkring-fabric"
RECORD_PATH = f"{STATE_DIR}/relay-state.json"
RECORD_SCHEMA = "sparkring-fabric-relay-record/v1"
DEFAULT_MARKER = "/var/tmp/ring8-mesh-marker"
SECTION = "@@"
HEADER = "set -euo pipefail"
SETTLE = "sleep 1.5"
_PATH = re.compile(r"^/[A-Za-z0-9_./+-]+$")
_MAC = re.compile(r"^[0-9a-f]{2}(?::[0-9a-f]{2}){5}$")
_HOSTNAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,62}$")


def q(value: object) -> str:
    return shlex.quote(str(value))


@dataclasses.dataclass(frozen=True)
class MarkerConfig:
    """Where the marker executable lives on every Spark and where each marker logs."""

    path: str = DEFAULT_MARKER
    log_dir: str = "/tmp"

    def __post_init__(self) -> None:
        for label, value in (("marker path", self.path), ("marker log directory", self.log_dir)):
            if not _PATH.match(value) or "/../" in f"{value}/":
                raise FabricError(f"{label} {value!r} must be an absolute path of letters, digits and _./+-")

    @property
    def name(self) -> str:
        return posixpath.basename(self.path)

    @property
    def comm(self) -> str:
        """The process name the kernel keeps for the executable (its first 15 characters)."""
        return self.name[:15]

    def log(self, device: str) -> str:
        return f"{self.log_dir}/{self.name}-{device}.log"


def _ipv4(value: str) -> str:
    try:
        return str(ipaddress.IPv4Address(value))
    except ValueError:
        raise FabricError(f"{value!r} is not an IPv4 address; apply needs the Sparks' facts, not placeholders") from None


def _mac(value: str) -> str:
    if not _MAC.match(value.lower()):
        raise FabricError(f"{value!r} is not a MAC address; apply needs the Sparks' facts, not placeholders")
    return value.lower()


def _netdev(value: str) -> str:
    if value not in FABRIC_NETDEVS:
        raise FabricError(f"{value!r} is not a fabric network device")
    return value


def _device(value: str) -> str:
    if value not in RDMA_DEVICES:
        raise FabricError(f"{value!r} is not a fabric RDMA device")
    return value


# -- read script ---------------------------------------------------------------------------


def read_script(marker: MarkerConfig) -> str:
    """Read-only facts and relay objects of one Spark, as ``@@<section>`` blocks."""
    lines = ["set -u"]

    def section(name: str, *commands: str) -> None:
        lines.append(f"echo {SECTION}{name}")
        lines.extend(commands)

    section("hostname", "hostname")
    section("versions", "ip -V 2>&1 || true", "tc -V 2>&1 || true")
    section("links", "ip -j addr show")
    for device in RDMA_DEVICES:
        section(f"rdma:{device}", f"ls /sys/class/infiniband/{device}/device/net 2>/dev/null || true")
    section("routes", "ip -j -d route show table main")
    section("neighbours", "ip -j neigh show nud permanent")
    for netdev in FABRIC_NETDEVS:
        section(f"qdisc:{netdev}", f"tc -j qdisc show dev {netdev} 2>/dev/null || true")
        section(f"filters:{netdev}", f"tc -j filter show dev {netdev} ingress 2>/dev/null || true")
    section("markers", f"pgrep -a -x {q(marker.comm)} || true")
    section("marker-binary", f"sha256sum {q(marker.path)} 2>/dev/null || true")
    for device in RDMA_DEVICES:
        section(f"marker-log:{device}", f"head -c 4096 {q(marker.log(device))} 2>/dev/null || true")
    section("record", f"cat {RECORD_PATH} 2>/dev/null || true")
    lines.append(f"echo {SECTION}end")
    return "\n".join(lines) + "\n"


def route_get_script(destinations: Iterable[str]) -> str:
    """``ip -o route get`` of every destination (read-only), as ``@@route:<address>`` blocks.

    The ring harness and the vLLM launcher check origin routes the same way.
    """
    lines = ["set -u"]
    for destination in destinations:
        address = _ipv4(destination)
        lines += [f"echo {SECTION}route:{address}", f"ip -o route get {address} 2>&1 || true"]
    lines.append(f"echo {SECTION}end")
    return "\n".join(lines) + "\n"


# -- apply commands --------------------------------------------------------------------------


def guard(hostname: str) -> str:
    """Stop the script unless it runs on ``hostname``."""
    if not _HOSTNAME.match(hostname):
        raise FabricError(f"host name {hostname!r} cannot be checked safely")
    return (f'test "$(hostname)" = {q(hostname)} || '
            f'{{ echo {q("refusing: this host is not " + hostname)} >&2; exit 3; }}')


def qdisc_add(netdev: str) -> str:
    return f"sudo -n tc qdisc add dev {_netdev(netdev)} ingress"


def qdisc_del(netdev: str) -> str:
    return f"sudo -n tc qdisc del dev {_netdev(netdev)} ingress"


def filter_add(relay_filter: RelayFilter) -> str:
    return (f"sudo -n tc filter add dev {_netdev(relay_filter.in_netdev)} ingress protocol "
            f"0x{relay_filter.protocol:04x} pref {relay_filter.pref} handle {relay_filter.handle} flower skip_sw "
            f"action pedit ex munge eth dst set {_mac(relay_filter.next_mac)} pipe "
            f"action pedit ex munge eth type set 0x{relay_filter.new_type:04x} pipe "
            f"action mirred egress redirect dev {_netdev(relay_filter.out_netdev)}")


def filter_del(netdev: str, pref: int, *, protocol: int | None = None, handle: int | None = None) -> str:
    """Delete one filter (protocol and handle known) or every filter of a reserved preference."""
    if protocol is None or handle is None:
        return f"sudo -n tc filter del dev {_netdev(netdev)} ingress pref {int(pref)}"
    return (f"sudo -n tc filter del dev {_netdev(netdev)} ingress protocol 0x{int(protocol):04x} "
            f"pref {int(pref)} handle {int(handle)} flower")


def neigh_replace(destination: str, mac: str, netdev: str) -> str:
    return (f"sudo -n ip neigh replace {_ipv4(destination)} lladdr {_mac(mac)} dev {_netdev(netdev)} "
            f"nud permanent proto {ROUTE_PROTOCOL}")


def neigh_del(destination: str, netdev: str) -> str:
    return f"sudo -n ip neigh del {_ipv4(destination)} dev {_netdev(netdev)}"


def route_replace(destination: str, netdev: str, source: str) -> str:
    return (f"sudo -n ip route replace {_ipv4(destination)}/32 dev {_netdev(netdev)} src {_ipv4(source)} "
            f"scope link proto {ROUTE_PROTOCOL}")


def route_del(destination: str, netdev: str, *, protocol: int | None, metric: int = 0) -> str:
    """Delete one /32 route; naming its protocol makes the kernel refuse any other route."""
    command = f"sudo -n ip route del {_ipv4(destination)}/32 dev {_netdev(netdev)}"
    if protocol is not None:
        command += f" proto {int(protocol)}"
    if metric:
        command += f" metric {int(metric)}"
    return command


def marker_start(marker: MarkerConfig, device: str, rules: Iterable[tuple[str, int]]) -> str:
    """Start one managed marker in its own session; it keeps its rules until SIGTERM."""
    rule_words = " ".join(f"--rule {_ipv4(destination)}=0x{int(value):04x}" for destination, value in rules)
    if not rule_words:
        raise FabricError(f"a marker for {device} needs at least one rule")
    return (f"(sudo -n setsid {q(marker.path)} --device {_device(device)} {rule_words} --managed "
            f"< /dev/null > {q(marker.log(device))} 2>&1 &)")


def marker_stop(marker: MarkerConfig, pid: int) -> tuple[str, str]:
    """Signal one marker process (only while it still is one) and wait up to 5 s for it to exit.

    The marker removes its RDMA-TX rules when it exits.
    """
    pid = int(pid)
    return (f'[ "$(cat /proc/{pid}/comm 2>/dev/null)" != {q(marker.comm)} ] || sudo -n kill {pid}',
            f"for i in $(seq 1 50); do [ -e /proc/{pid} ] || break; sleep 0.1; done; [ ! -e /proc/{pid} ]")


def record_text(record: Mapping[str, object]) -> str:
    return json.dumps(record, sort_keys=True, separators=(",", ":"))


def record_write(record: Mapping[str, object]) -> tuple[str, str]:
    return (f"sudo -n mkdir -p {STATE_DIR}",
            f"printf '%s\\n' {q(record_text(record))} | sudo -n tee {RECORD_PATH} >/dev/null")


def record_remove() -> str:
    return f"sudo -n rm -f {RECORD_PATH}"


# -- marker executable -----------------------------------------------------------------------

MARKER_SOURCE_END = "SPARKRING_MARKER_SOURCE_END"


def marker_build_script(hostname: str, marker: MarkerConfig, source: str) -> str:
    """Compile the marker source on a Spark (rdma-core headers needed) and install it at the marker path."""
    if any(line.strip() == MARKER_SOURCE_END for line in source.splitlines()):
        raise FabricError("the marker source contains the here-document delimiter")
    return "\n".join([
        HEADER,
        guard(hostname),
        "work=$(mktemp -d)",
        "trap 'rm -rf \"$work\"' EXIT",
        f"cat > \"$work/mesh_marker.c\" <<'{MARKER_SOURCE_END}'",
        source.rstrip("\n"),
        MARKER_SOURCE_END,
        "cc -O2 -Wall -Wextra \"$work/mesh_marker.c\" -o \"$work/marker\" -libverbs -lmlx5",
        f"install -m 0755 \"$work/marker\" {q(marker.path)}",
        f"sha256sum {q(marker.path)}",
    ]) + "\n"
