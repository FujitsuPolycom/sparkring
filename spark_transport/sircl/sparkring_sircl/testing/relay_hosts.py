"""Host-command simulator: simulated Sparks that answer the relay plan installer's scripts.

A :class:`SimSpark` keeps the state the installer reads and changes: the four
fabric network devices (MAC, IPv4 address and subnet), the main routing table,
the neighbour table, ingress qdiscs and flower filters per device, processes
and files. :meth:`SimSpark.run` executes a script line by line as ``bash -s``
would (``set -e`` stops at the first failing command, ``exit`` ends the
script), accepting exactly the command forms the installer generates and the
forms of a plain shell installer that the tests use as a reference. Reads are
answered with JSON shaped like iproute2's ``ip -j`` and ``tc -j`` output, in
two spellings (``style``): ``iproute2-6`` prints filter handles as ``0x1``
strings, ``iproute2-5`` as numbers. A line the simulator does not know raises
:class:`SimulatorError`, so a generator change the simulator has not learned
fails the tests instead of passing silently.

Kernel rules modelled: ``ip route replace`` keys on destination, prefix and
metric and needs a local source address; ``ip route del`` with ``proto``
deletes only a route of that protocol; ``tc filter add`` needs an ingress
qdisc and refuses an existing preference with another protocol or an
existing handle; a deleted ingress qdisc takes its filters with it; a marker
starts only from an existing executable with valid rules, and exiting removes
its RDMA-TX rules.

:class:`SimRing` cables the Sparks as a ring (position ``i`` port 0 to
position ``i + 1`` port 1), serves as the installer's executor (SSH target ->
Spark) and follows frames through the installed state (:meth:`SimRing.deliver`):
origin route, neighbour, marker tag, cable, relay filters, delivery. That walk
is independent of the plan module, so it checks what was installed, not what
was planned.
"""

from __future__ import annotations

import dataclasses
import hashlib
import ipaddress
import json
import re
import shlex
from collections.abc import Sequence

from ..ring import remote
from ..ring.site import Host, Site

FABRIC = (("enp1s0f0np0", "rocep1s0f0", 0, False), ("enP2p1s0f0np0", "roceP2p1s0f0", 0, True),
          ("enp1s0f1np1", "rocep1s0f1", 1, False), ("enP2p1s0f1np1", "roceP2p1s0f1", 1, True))
PROTOCOL_NAMES = {2: "kernel", 3: "boot", 4: "static", 16: "dhcp"}
STYLES = ("iproute2-6", "iproute2-5")


class SimulatorError(AssertionError):
    """A script line the simulator does not recognise."""


class _Exit(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(code)
        self.code = code


@dataclasses.dataclass
class Netdev:
    name: str
    device: str
    port: int
    secondary: bool
    mac: str
    address: str
    prefixlen: int

    @property
    def network(self) -> ipaddress.IPv4Network:
        return ipaddress.IPv4Interface(f"{self.address}/{self.prefixlen}").network


@dataclasses.dataclass
class Filter:
    netdev: str
    pref: int
    handle: int
    protocol: int
    dst_mac: str
    new_type: int
    out_dev: str
    in_hw: bool = True


class SimSpark:
    """One simulated Spark."""

    def __init__(self, name: str, position: int, netdevs: Sequence[Netdev], *, style: str = "iproute2-6",
                 lan_address: str = "192.0.2.1", marker_path: str | None = "/var/tmp/ring8-mesh-marker") -> None:
        if style not in STYLES:
            raise ValueError(f"style must be one of {STYLES}")
        self.name = name
        self.hostname = name
        self.position = position
        self.style = style
        self.netdevs = {netdev.name: netdev for netdev in netdevs}
        self.lan_address = lan_address
        self.routes: list[dict] = [{"dst": str(n.network.network_address), "plen": n.prefixlen, "dev": n.name,
                                    "src": n.address, "scope": "link", "proto": 2, "metric": 0, "gateway": None}
                                   for n in netdevs]
        self.routes.append({"dst": "0.0.0.0", "plen": 0, "dev": "enP7s7", "src": None, "scope": "global",
                            "proto": 16, "metric": 100, "gateway": "192.0.2.254"})
        self.neighbours: list[dict] = []
        self.qdiscs: dict[str, str | None] = {n.name: None for n in netdevs}
        self.filters: dict[str, list[Filter]] = {n.name: [] for n in netdevs}
        self.processes: dict[int, dict] = {}
        self.files: dict[str, str] = {}
        if marker_path:
            self.files[marker_path] = "\x7fELF ring8-mesh-marker"
        self.next_pid = 4100
        self.mutations: list[str] = []
        self.reachable = True
        self.fail_on: set[str] = set()
        self.policy_routes: dict[str, str] = {}     # address -> device that `ip route get` reports instead

    # -- helpers --

    def netdev_by_mac(self, mac: str) -> Netdev | None:
        for netdev in self.netdevs.values():
            if netdev.mac == mac.lower():
                return netdev
        return None

    def local_address(self, address: str) -> bool:
        return any(netdev.address == address for netdev in self.netdevs.values()) or address == self.lan_address

    def marker_processes(self) -> list[dict]:
        return [process for process in self.processes.values() if process["comm"] == "ring8-mesh-mark"]

    def lookup_route(self, address: str) -> dict | None:
        value = ipaddress.IPv4Address(address)
        matches = [route for route in self.routes
                   if value in ipaddress.IPv4Network(f"{route['dst']}/{route['plen']}", strict=False)]
        if not matches:
            return None
        return sorted(matches, key=lambda r: (-r["plen"], r["metric"]))[0]

    def neighbour(self, address: str, dev: str) -> dict | None:
        for entry in self.neighbours:
            if entry["dst"] == address and entry["dev"] == dev:
                return entry
        return None

    # -- reads --

    def _addr_json(self) -> str:
        links = [{"ifindex": 1, "ifname": "lo", "flags": ["LOOPBACK", "UP"], "mtu": 65536, "operstate": "UNKNOWN",
                  "address": "00:00:00:00:00:00",
                  "addr_info": [{"family": "inet", "local": "127.0.0.1", "prefixlen": 8, "scope": "host"}]}]
        for index, netdev in enumerate(self.netdevs.values(), start=2):
            links.append({"ifindex": index, "ifname": netdev.name, "flags": ["BROADCAST", "MULTICAST", "UP"],
                          "mtu": 9000, "operstate": "UP", "address": netdev.mac,
                          "addr_info": [{"family": "inet", "local": netdev.address, "prefixlen": netdev.prefixlen,
                                         "scope": "global"},
                                        {"family": "inet6", "local": "fe80::1", "prefixlen": 64}]})
        links.append({"ifindex": 9, "ifname": "enP7s7", "flags": ["UP"], "mtu": 1500, "operstate": "UP",
                      "address": "02:00:00:00:99:01",
                      "addr_info": [{"family": "inet", "local": self.lan_address, "prefixlen": 24}]})
        return json.dumps(links)

    def _routes_json(self) -> str:
        rows = []
        for route in self.routes:
            destination = "default" if route["plen"] == 0 else (
                route["dst"] if route["plen"] == 32 else f"{route['dst']}/{route['plen']}")
            row = {"type": "unicast", "dst": destination}
            if route["gateway"]:
                row["gateway"] = route["gateway"]
            row.update({"dev": route["dev"], "protocol": PROTOCOL_NAMES.get(route["proto"], str(route["proto"])),
                        "scope": route["scope"] if route["scope"] == "link" else "global"})
            if route["src"]:
                row["prefsrc"] = route["src"]
            if route["metric"]:
                row["metric"] = route["metric"]
            row["flags"] = []
            rows.append(row)
        return json.dumps(rows)

    def _neigh_json(self) -> str:
        rows = []
        for entry in self.neighbours:
            if entry["state"] != "PERMANENT":
                continue
            row = {"dst": entry["dst"], "dev": entry["dev"], "lladdr": entry["lladdr"], "state": ["PERMANENT"]}
            if entry.get("proto"):
                row["protocol"] = PROTOCOL_NAMES.get(entry["proto"], str(entry["proto"]))
            rows.append(row)
        return json.dumps(rows)

    def _qdisc_json(self, dev: str) -> str:
        rows = [{"kind": "mq", "handle": "0:", "root": True, "options": {}}]
        if self.qdiscs.get(dev):
            rows.append({"kind": self.qdiscs[dev], "handle": "ffff:", "parent": "ffff:fff1", "options": {}})
        return json.dumps(rows)

    @staticmethod
    def _pedit_keys(mac: str | None, new_type: int | None) -> list[dict]:
        keys = []
        if mac is not None:
            octets = [int(part, 16) for part in mac.split(":")]
            first = int.from_bytes(bytes(octets[:4]), "big")
            second = int.from_bytes(bytes(octets[4:6] + [0, 0]), "big")
            keys.append({"htype": "eth", "offset": 0, "cmd": "set", "val": f"{first:x}", "mask": "0"})
            keys.append({"htype": "eth", "offset": 4, "cmd": "set", "val": f"{second:x}", "mask": "ffff"})
        if new_type is not None:
            keys.append({"htype": "eth", "offset": 12, "cmd": "set", "val": f"{new_type << 16:x}", "mask": "ffff"})
        return keys

    def _filters_json(self, dev: str) -> str:
        rows = []
        by_pref: dict[int, list[Filter]] = {}
        for entry in self.filters.get(dev, []):
            by_pref.setdefault(entry.pref, []).append(entry)
        index = 1
        for pref in sorted(by_pref):
            protocol = f"[{by_pref[pref][0].protocol}]"
            rows.append({"protocol": protocol, "pref": pref, "kind": "flower", "chain": 0})
            for entry in sorted(by_pref[pref], key=lambda f: f.handle):
                handle: object = f"0x{entry.handle:x}" if self.style == "iproute2-6" else entry.handle
                actions = [
                    {"order": 1, "kind": "pedit", "control_action": {"type": "pipe"}, "nkeys": 2, "index": index,
                     "ref": 1, "bind": 1, "keys": self._pedit_keys(entry.dst_mac, None)},
                    {"order": 2, "kind": "pedit", "control_action": {"type": "pipe"}, "nkeys": 1, "index": index + 1,
                     "ref": 1, "bind": 1, "keys": self._pedit_keys(None, entry.new_type)},
                    {"order": 3, "kind": "mirred", "mirred_action": "redirect", "direction": "egress",
                     "to_dev": entry.out_dev, "control_action": {"type": "stolen"}, "index": index, "ref": 1,
                     "bind": 1},
                ]
                index += 2
                options = {"handle": handle, "keys": {"eth_type": f"{entry.protocol:04x}"}, "skip_sw": True}
                if entry.in_hw:
                    options.update({"in_hw": True, "in_hw_count": 1})
                options["actions"] = actions
                rows.append({"protocol": protocol, "pref": pref, "kind": "flower", "chain": 0, "options": options})
        return json.dumps(rows)

    # -- script execution --

    def run(self, script: str) -> remote.Result:
        if not self.reachable:
            return remote.Result(255, "", f"ssh: connect to host {self.name} port 22: Connection timed out")
        out: list[str] = []
        err: list[str] = []
        errexit = False
        lines = script.splitlines()
        index = 0
        code = 0
        try:
            while index < len(lines):
                line = lines[index].strip()
                index += 1
                if not line:
                    continue
                heredoc = re.match(r"^cat > \"\$work/mesh_marker\.c\" <<'(\w+)'$", line)
                if heredoc:
                    body = []
                    while index < len(lines) and lines[index] != heredoc.group(1):
                        body.append(lines[index])
                        index += 1
                    index += 1
                    self.files["$work/mesh_marker.c"] = "\n".join(body)
                    continue
                if line in ("set -euo pipefail", "set -e"):
                    errexit = True
                    continue
                if line == "set -u":
                    continue
                code = self._execute(line, out, err)
                if code and errexit:
                    return remote.Result(code, "\n".join(out) + "\n" if out else "", "\n".join(err))
        except _Exit as stop:
            return remote.Result(stop.code, "\n".join(out) + "\n" if out else "", "\n".join(err))
        return remote.Result(code, "\n".join(out) + "\n" if out else "", "\n".join(err))

    def _mutate(self, line: str) -> None:
        self.mutations.append(line)
        for text in self.fail_on:
            if text in line:
                raise _Failure(f"simulated failure of: {line}")

    def _execute(self, line: str, out: list[str], err: list[str]) -> int:
        try:
            return self._dispatch(line, out, err)
        except _Failure as failure:
            err.append(str(failure))
            return 2

    def _dispatch(self, line: str, out: list[str], err: list[str]) -> int:
        guard = re.match(r'^test "\$\(hostname\)" = (\S+) \|\| \{ echo (.+) >&2; exit 3; \}$', line)
        if guard:
            if shlex.split(guard.group(1))[0] != self.hostname:
                err.append(shlex.split(guard.group(2))[0])
                raise _Exit(3)
            return 0
        plain_guard = re.match(r'^\[ "\$\(hostname\)" = "(\S+)" \]$', line)
        if plain_guard:
            return 0 if plain_guard.group(1) == self.hostname else 1
        start = re.match(r"^\(sudo -n setsid (\S+) (.+) < /dev/null > (\S+) 2>&1 &\)$", line)
        if start:
            self._mutate(line)
            self._start_marker(start.group(1), shlex.split(start.group(2)), start.group(3))
            return 0
        stop = re.match(r'^\[ "\$\(cat /proc/(\d+)/comm 2>/dev/null\)" != (\S+) \] \|\| sudo -n kill (\d+)$', line)
        if stop:
            pid = int(stop.group(1))
            process = self.processes.get(pid)
            if process is not None and process["comm"] == shlex.split(stop.group(2))[0]:
                self._mutate(line)
                del self.processes[pid]
            return 0
        wait = re.match(r"^for i in \$\(seq 1 50\); do \[ -e /proc/(\d+) \] \|\| break; sleep 0\.1; done; "
                        r"\[ ! -e /proc/(\d+) \]$", line)
        if wait:
            return 1 if int(wait.group(1)) in self.processes else 0
        record = re.match(r"^printf '%s\\n' (.+) \| sudo -n tee (\S+) >/dev/null$", line)
        if record:
            self._mutate(line)
            self.files[record.group(2)] = shlex.split(record.group(1))[0] + "\n"
            return 0
        if line in ("work=$(mktemp -d)", "trap 'rm -rf \"$work\"' EXIT"):
            return 0
        compile_line = re.match(r'^cc -O2 -Wall -Wextra "\$work/mesh_marker\.c" -o "\$work/marker" -libverbs -lmlx5$',
                                line)
        if compile_line:
            source = self.files.get("$work/mesh_marker.c", "")
            if "mlx5dv_create_flow" not in source:
                err.append("compile error")
                return 1
            self.files["$work/marker"] = "\x7fELF built " + hashlib.sha256(source.encode()).hexdigest()
            return 0
        install = re.match(r'^install -m 0755 "\$work/marker" (\S+)$', line)
        if install:
            self._mutate(line)
            self.files[install.group(1)] = self.files["$work/marker"]
            return 0
        words = shlex.split(line)
        tolerant = False
        while words and words[-1] in ("true", "||", "2>/dev/null", ">/dev/null", "2>&1"):
            if words[-1] == "true" and len(words) > 1 and words[-2] == "||":
                tolerant = True
            words.pop()
        code = self._command(words, line, out, err)
        return 0 if tolerant else code

    def _command(self, words: list[str], line: str, out: list[str], err: list[str]) -> int:
        if not words:
            raise SimulatorError(line)
        if words[0] == "echo":
            out.append(" ".join(words[1:]))
            return 0
        if words == ["hostname"]:
            out.append(self.hostname)
            return 0
        if words in (["ip", "-V"], ["tc", "-V"]):
            out.append(f"{words[0]} utility, iproute2-{'6.1.0' if self.style == 'iproute2-6' else '5.15.0'}")
            return 0
        if words == ["ip", "-j", "addr", "show"]:
            out.append(self._addr_json())
            return 0
        if words == ["ip", "-j", "-d", "route", "show", "table", "main"]:
            out.append(self._routes_json())
            return 0
        if words == ["ip", "-j", "neigh", "show", "nud", "permanent"]:
            out.append(self._neigh_json())
            return 0
        if words[:4] == ["ip", "-o", "route", "get"] and len(words) == 5:
            if words[4] in self.policy_routes:          # a policy rule or another table decides first
                out.append(f"{words[4]} dev {self.policy_routes[words[4]]} table 100 uid 1000 \\    cache ")
                return 0
            route = self.lookup_route(words[4])          # the script merges stderr into stdout (2>&1)
            if route is None:
                out.append("RTNETLINK answers: Network is unreachable")
                return 2
            netdev = self.netdevs.get(route["dev"])
            source = route["src"] or (netdev.address if netdev else self.lan_address)
            via = f" via {route['gateway']}" if route["gateway"] else ""
            out.append(f"{words[4]}{via} dev {route['dev']} src {source} uid 1000 \\    cache ")
            return 0
        if len(words) == 2 and words[0] == "ls" and words[1].startswith("/sys/class/infiniband/"):
            device = words[1].split("/")[4]
            for netdev in self.netdevs.values():
                if netdev.device == device:
                    out.append(netdev.name)
                    return 0
            return 2
        if words[:4] == ["tc", "-j", "qdisc", "show"] and len(words) == 6:
            out.append(self._qdisc_json(words[5]))
            return 0
        if words[:4] == ["tc", "-j", "filter", "show"] and len(words) == 7 and words[6] == "ingress":
            if not self.qdiscs.get(words[5]):
                err.append("Error: Cannot find specified qdisc on specified device.")
                return 2
            out.append(self._filters_json(words[5]))
            return 0
        if words[:3] == ["pgrep", "-a", "-x"] and len(words) == 4:
            found = [f"{pid} {' '.join(p['argv'])}" for pid, p in sorted(self.processes.items())
                     if p["comm"] == words[3]]
            out.extend(found)
            return 0 if found else 1
        if words[0] == "sha256sum" and len(words) == 2:
            if words[1] not in self.files:
                return 1
            out.append(f"{hashlib.sha256(self.files[words[1]].encode()).hexdigest()}  {words[1]}")
            return 0
        if words[:2] == ["head", "-c"] and len(words) == 4:
            if words[3] not in self.files:
                return 1
            out.append(self.files[words[3]][:int(words[2])].rstrip("\n"))
            return 0
        if words[0] == "cat" and len(words) == 2:
            if words[1] not in self.files:
                return 1
            out.append(self.files[words[1]].rstrip("\n"))
            return 0
        if words[0] == "sleep":
            return 0
        if words[:2] == ["sudo", "-n"]:
            return self._privileged(words[2:], line, err)
        raise SimulatorError(line)

    def _privileged(self, words: list[str], line: str, err: list[str]) -> int:
        if words[:2] == ["mkdir", "-p"] and len(words) == 3:
            return 0
        if words[:2] == ["rm", "-f"] and len(words) == 3:
            self._mutate(line)
            self.files.pop(words[2], None)
            return 0
        if words[:2] == ["pkill", "-x"] and len(words) == 3:
            self._mutate(line)
            victims = [pid for pid, p in self.processes.items() if p["comm"] == words[2]]
            for pid in victims:
                del self.processes[pid]
            return 0 if victims else 1
        if words[:3] == ["tc", "qdisc", "add"] and words[3] == "dev" and words[5:] == ["ingress"]:
            self._mutate(line)
            if self.qdiscs.get(words[4]):
                err.append("RTNETLINK answers: File exists")
                return 2
            self.qdiscs[words[4]] = "ingress"
            return 0
        if words[:3] == ["tc", "qdisc", "del"] and words[3] == "dev" and words[5:] == ["ingress"]:
            self._mutate(line)
            if not self.qdiscs.get(words[4]):
                err.append("Error: Cannot find specified qdisc on specified device.")
                return 2
            self.qdiscs[words[4]] = None
            self.filters[words[4]] = []
            return 0
        if words[:3] == ["tc", "filter", "add"]:
            self._mutate(line)
            return self._filter_add(words, line, err)
        if words[:3] == ["tc", "filter", "del"]:
            self._mutate(line)
            return self._filter_del(words, line, err)
        if words[:3] == ["ip", "route", "replace"]:
            self._mutate(line)
            return self._route_replace(words, line, err)
        if words[:3] == ["ip", "route", "del"]:
            self._mutate(line)
            return self._route_del(words, line, err)
        if words[:3] == ["ip", "neigh", "replace"]:
            self._mutate(line)
            return self._neigh_replace(words, line, err)
        if words[:3] == ["ip", "neigh", "del"] and len(words) == 6 and words[4] == "dev":
            self._mutate(line)
            entry = self.neighbour(words[3], words[5])
            if entry is None:
                err.append("RTNETLINK answers: No such file or directory")
                return 2
            self.neighbours.remove(entry)
            return 0
        if words[:1] == ["kill"] and len(words) == 2:
            self._mutate(line)
            return 0 if self.processes.pop(int(words[1]), None) else 1
        raise SimulatorError(line)

    def _filter_add(self, words: list[str], line: str, err: list[str]) -> int:
        pattern = ["tc", "filter", "add", "dev", None, "ingress", "protocol", None, "pref", None, "handle", None,
                   "flower", "skip_sw", "action", "pedit", "ex", "munge", "eth", "dst", "set", None, "pipe",
                   "action", "pedit", "ex", "munge", "eth", "type", "set", None, "pipe", "action", "mirred",
                   "egress", "redirect", "dev", None]
        if len(words) != len(pattern) or any(p is not None and p != w for p, w in zip(pattern, words)):
            raise SimulatorError(line)
        dev, protocol, pref, handle, mac, new_type, out_dev = (words[4], int(words[7], 0), int(words[9]),
                                                               int(words[11], 0), words[21].lower(),
                                                               int(words[30], 0), words[37])
        if not self.qdiscs.get(dev):
            err.append("Error: Cannot find specified qdisc on specified device.")
            return 2
        if out_dev not in self.netdevs or not re.match(r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$", mac):
            err.append("Error: hardware offload refused (skip_sw)")
            return 2
        for entry in self.filters[dev]:
            if entry.pref == pref and entry.protocol != protocol:
                err.append("RTNETLINK answers: Invalid argument")
                return 2
            if entry.pref == pref and entry.handle == handle:
                err.append("RTNETLINK answers: File exists")
                return 2
        self.filters[dev].append(Filter(dev, pref, handle, protocol, mac, new_type, out_dev))
        return 0

    def _filter_del(self, words: list[str], line: str, err: list[str]) -> int:
        if len(words) == 8 and words[3] == "dev" and words[5:7] == ["ingress", "pref"]:
            dev, pref = words[4], int(words[7])
            kept = [entry for entry in self.filters.get(dev, []) if entry.pref != pref]
            if len(kept) == len(self.filters.get(dev, [])):
                err.append("Error: Filter with specified priority/protocol not found.")
                return 2
            self.filters[dev] = kept
            return 0
        pattern = ["tc", "filter", "del", "dev", None, "ingress", "protocol", None, "pref", None, "handle", None,
                   "flower"]
        if len(words) != len(pattern) or any(p is not None and p != w for p, w in zip(pattern, words)):
            raise SimulatorError(line)
        dev, protocol, pref, handle = words[4], int(words[7], 0), int(words[9]), int(words[11], 0)
        for entry in self.filters.get(dev, []):
            if (entry.pref, entry.handle, entry.protocol) == (pref, handle, protocol):
                self.filters[dev].remove(entry)
                return 0
        err.append("Error: Filter with specified handle not found.")
        return 2

    @staticmethod
    def _options(words: list[str], start: int, line: str) -> dict[str, str]:
        options = {}
        index = start
        while index < len(words):
            if index + 1 >= len(words):
                raise SimulatorError(line)
            options[words[index]] = words[index + 1]
            index += 2
        return options

    def _route_replace(self, words: list[str], line: str, err: list[str]) -> int:
        destination, _, length = words[3].partition("/")
        options = self._options(words, 4, line)
        if set(options) - {"dev", "src", "scope", "proto"} or options.get("scope") != "link" or length != "32":
            raise SimulatorError(line)
        if options["dev"] not in self.netdevs:
            err.append(f'Cannot find device "{options["dev"]}"')
            return 1
        if "src" in options and not self.local_address(options["src"]):
            err.append("Error: Invalid prefsrc address.")
            return 2
        new = {"dst": destination, "plen": 32, "dev": options["dev"], "src": options.get("src"), "scope": "link",
               "proto": int(options.get("proto", 3)), "metric": 0, "gateway": None}
        for index, route in enumerate(self.routes):
            if route["dst"] == destination and route["plen"] == 32 and route["metric"] == 0:
                self.routes[index] = new
                return 0
        self.routes.append(new)
        return 0

    def _route_del(self, words: list[str], line: str, err: list[str]) -> int:
        destination, _, length = words[3].partition("/")
        options = self._options(words, 4, line)
        if set(options) - {"dev", "proto", "metric"} or length != "32":
            raise SimulatorError(line)
        for route in self.routes:
            if (route["dst"] == destination and route["plen"] == 32 and route["dev"] == options["dev"]
                    and ("proto" not in options or route["proto"] == int(options["proto"]))
                    and ("metric" not in options or route["metric"] == int(options["metric"]))):
                self.routes.remove(route)
                return 0
        err.append("RTNETLINK answers: No such process")
        return 2

    def _neigh_replace(self, words: list[str], line: str, err: list[str]) -> int:
        destination = words[3]
        options = self._options(words, 4, line)
        if set(options) - {"lladdr", "dev", "nud", "proto"} or options.get("nud") != "permanent":
            raise SimulatorError(line)
        if options["dev"] not in self.netdevs:
            err.append(f'Cannot find device "{options["dev"]}"')
            return 1
        entry = self.neighbour(destination, options["dev"])
        if entry is None:
            entry = {"dst": destination, "dev": options["dev"], "proto": None}
            self.neighbours.append(entry)
        entry.update({"lladdr": options["lladdr"].lower(), "state": "PERMANENT"})
        if "proto" in options:
            entry["proto"] = int(options["proto"])
        return 0

    def _start_marker(self, path: str, arguments: list[str], log: str) -> None:
        if path not in self.files:
            self.files[log] = f"sudo: {path}: command not found\n"
            return
        device, rules, managed = None, [], False
        index = 0
        while index < len(arguments):
            if arguments[index] == "--device":
                device = arguments[index + 1]
                index += 2
            elif arguments[index] == "--rule":
                rules.append(arguments[index + 1])
                index += 2
            elif arguments[index] == "--managed":
                managed = True
                index += 1
            else:
                self.files[log] = "usage: ...\n"
                return
        addresses = [rule.partition("=")[0] for rule in rules]
        valid = (device in {n.device for n in self.netdevs.values()} and managed and 0 < len(rules) <= 64
                 and len(set(addresses)) == len(addresses)
                 and all(int(rule.partition("=")[2], 0) not in (0, 0x0800) for rule in rules))
        if not valid:
            self.files[log] = "--rule must be IPV4=ETHERTYPE (not 0x0800), at most 64\n"
            return
        pid = self.next_pid
        self.next_pid += 1
        self.processes[pid] = {"comm": path.rsplit("/", 1)[-1][:15], "argv": [path, *arguments], "device": device,
                               "rules": {a: int(v, 0) for a, _, v in (r.partition("=") for r in rules)}}
        listed = ",".join(f'{{"dst":"{a}","ethertype":"0x{int(v, 0):04x}"}}'
                          for a, _, v in (r.partition("=") for r in rules))
        self.files[log] = (f'{{"device":"{device}","rules":[{listed}],"installed":{len(rules)},'
                           '"managed":true,"run_seconds":0}\n')


class _Failure(Exception):
    pass


def ring_netdevs(position: int, size: int, *, prefixlen: int = 24) -> list[Netdev]:
    """The four fabric functions of ring position ``position``: one subnet per link of each cable.

    Cable ``i`` (position ``i`` port 0 to position ``i + 1`` port 1) carries
    subnet ``198.18.(2i + class).0``, ``.1`` at port 0 and ``.2`` at port 1.
    """
    netdevs = []
    for name, device, port, secondary in FABRIC:
        cable = position if port == 0 else (position - 1) % size
        host = 1 if port == 0 else 2
        mac = f"02:5a:{position:02x}:{port:02x}:{int(secondary):02x}:01"
        netdevs.append(Netdev(name, device, port, secondary, mac, f"198.18.{2 * cable + int(secondary)}.{host}",
                              prefixlen))
    return netdevs


class SimRing:
    """Simulated Sparks cabled as a ring, and the installer's executor for them."""

    def __init__(self, size: int = 8, *, names: Sequence[str] | None = None, style: str = "iproute2-6",
                 prefixlen: int = 24) -> None:
        self.size = size
        names = list(names) if names else [f"spark{position}" for position in range(size)]
        self.sparks = [SimSpark(names[position], position, ring_netdevs(position, size, prefixlen=prefixlen),
                                style=style, lan_address=f"192.0.2.{10 + position}") for position in range(size)]
        self.calls: list[tuple[str, str]] = []

    def site(self) -> Site:
        hosts = tuple(Host(spark.name, f"op@{spark.lan_address}", spark.lan_address) for spark in self.sparks)
        return Site("sha256:simulated", "enP7s7", 29650, "/tmp/sircl-ring", hosts)

    def spark(self, target: str) -> SimSpark:
        for spark in self.sparks:
            if target == f"op@{spark.lan_address}":
                return spark
        raise KeyError(target)

    def executor(self, target: str, script: str, timeout: float) -> remote.Result:
        self.calls.append((target, script))
        return self.spark(target).run(script)

    def run(self, position: int, script: str) -> remote.Result:
        return self.sparks[position].run(script)

    def mutations(self) -> dict[int, list[str]]:
        return {spark.position: list(spark.mutations) for spark in self.sparks}

    def clear_mutations(self) -> None:
        for spark in self.sparks:
            spark.mutations.clear()

    def snapshot(self, position: int) -> str:
        """Everything a Spark's forwarding depends on, as comparable text (marker logs under /tmp excluded)."""
        spark = self.sparks[position]
        return json.dumps({"routes": spark.routes, "neighbours": spark.neighbours, "qdiscs": spark.qdiscs,
                           "filters": {k: [dataclasses.asdict(f) for f in v] for k, v in spark.filters.items()},
                           "processes": {str(k): v for k, v in spark.processes.items()},
                           "files": {k: v for k, v in spark.files.items()
                                     if not k.startswith("$work") and not k.startswith("/tmp/")}},
                          sort_keys=True)

    # -- frames --

    def _cable_end(self, position: int, port: int) -> tuple[int, int]:
        return ((position + 1) % self.size, 1) if port == 0 else ((position - 1) % self.size, 0)

    def deliver(self, origin: int, destination: str, *, limit: int = 16) -> tuple[str, list[tuple[int, str]]]:
        """Send a RoCE frame from ``origin`` to ``destination`` through the installed state.

        Returns the outcome (``delivered``, ``unrouted``, ``unresolved``,
        ``dropped`` or ``misdelivered``) and the (position, receiving device)
        of every cable crossed.
        """
        spark = self.sparks[origin]
        route = spark.lookup_route(destination)
        if route is None or route["dev"] not in spark.netdevs:
            return "unrouted", []
        out_dev = route["dev"]
        entry = spark.neighbour(destination, out_dev)
        if entry is not None:
            mac = entry["lladdr"]
        else:
            there, there_port = self._cable_end(origin, spark.netdevs[out_dev].port)
            far = [n for n in self.sparks[there].netdevs.values() if n.port == there_port and n.address == destination]
            if not far:
                return "unresolved", []
            mac = far[0].mac
        ethertype = 0x0800
        device = spark.netdevs[out_dev].device
        for process in spark.marker_processes():
            if process["device"] == device and destination in process["rules"]:
                ethertype = process["rules"][destination]
        here = origin
        hops: list[tuple[int, str]] = []
        for _ in range(limit):
            there, there_port = self._cable_end(here, self.sparks[here].netdevs[out_dev].port)
            receiver = self.sparks[there].netdev_by_mac(mac)
            if receiver is None or receiver.port != there_port:
                return "dropped", hops
            hops.append((there, receiver.name))
            if ethertype == 0x0800:
                return ("delivered" if receiver.address == destination else "misdelivered"), hops
            matching = [f for f in self.sparks[there].filters.get(receiver.name, []) if f.protocol == ethertype]
            if not matching:
                return "dropped", hops
            relay = matching[0]
            mac, ethertype, here, out_dev = relay.dst_mac, relay.new_type, there, relay.out_dev
        return "looped", hops

    def inject(self, position: int, netdev: str, ethertype: int) -> tuple[str, list[tuple[int, str]]]:
        """A tagged frame arriving on ``netdev`` of ``position``: how far relays carry it."""
        spark = self.sparks[position]
        matching = [f for f in spark.filters.get(netdev, []) if f.protocol == ethertype]
        if not matching:
            return "dropped", [(position, netdev)]
        relay = matching[0]
        here, out_dev, mac, value = position, relay.out_dev, relay.dst_mac, relay.new_type
        hops = [(position, netdev)]
        for _ in range(16):
            there, there_port = self._cable_end(here, self.sparks[here].netdevs[out_dev].port)
            receiver = self.sparks[there].netdev_by_mac(mac)
            if receiver is None or receiver.port != there_port:
                return "dropped", hops
            hops.append((there, receiver.name))
            if value == 0x0800:
                return "delivered", hops
            matching = [f for f in self.sparks[there].filters.get(receiver.name, []) if f.protocol == value]
            if not matching:
                return "dropped", hops
            relay = matching[0]
            here, out_dev, mac, value = there, relay.out_dev, relay.dst_mac, relay.new_type
        return "looped", hops
