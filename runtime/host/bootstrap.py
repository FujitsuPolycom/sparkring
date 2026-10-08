"""Explore authenticated IPv6 neighbors and provision packages through SSH hops."""
import hashlib
import inspect
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from runtime.common import fabric_layout

# The most cable hops between Node A and a Spark it reaches: the far end of an
# eight-Spark line. A fabric of ``size`` Sparks needs at most ``size - 1``
# (``hop_limit``).
MAX_HOPS = fabric_layout.MAX_SPARKS - 1


def hop_limit(size=None):
    """The longest SSH route through the cables to any of ``size`` Sparks: ``size - 1``, or ``MAX_HOPS``."""
    if size is None:
        return MAX_HOPS
    if not fabric_layout.MIN_SPARKS <= size <= fabric_layout.MAX_SPARKS:
        raise ValueError(f"SparkRing sets up {fabric_layout.MIN_SPARKS} to {fabric_layout.MAX_SPARKS} Sparks")
    return size - 1


def fabric_identity(guids, machine_id):
    """A Spark's discovery identity: a hash of its RDMA node GUIDs.

    The GUIDs come from the ConnectX hardware and differ on every Spark.
    /etc/machine-id does not identify a Spark: every unit flashed from one
    factory image carries the same value. A host without RDMA devices keeps
    its machine ID.
    """
    import hashlib
    guids = sorted(g.strip().lower() for g in guids if g.strip())
    if not guids:
        return machine_id
    return hashlib.sha256("\n".join(guids).encode()).hexdigest()[:32]


def probe():
    """Self-contained read-only probe suitable for a Spark without this package.

    Runs with fabric_identity() defined beside it (see inventory()).
    """
    import json
    from pathlib import Path
    import platform
    import re
    import socket
    import subprocess

    def command(argv):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout

    links = json.loads(command(["ip", "-j", "address", "show"]))
    functions, guids = [], []
    for directory in sorted(Path("/sys/class/infiniband").iterdir()):
        if (directory / "node_guid").is_file():
            guids.append((directory / "node_guid").read_text())
        for nic in (directory / "device/net").iterdir():
            link = next(row for row in links if row["ifname"] == nic.name)
            try:
                carrier = (Path("/sys/class/net") / nic.name / "carrier").read_text().strip() == "1"
            except OSError:
                carrier = False
            functions.append({"device": directory.name, "netdev": nic.name, "mac": link.get("address"), "carrier": carrier,
                              "addresses": [a["local"] for a in link.get("addr_info", []) if a["family"] == "inet6" and a["scope"] == "link"]})
    # Every host on a fabric link answers the all-nodes multicast echo; the
    # replies name the addresses in use now.
    pings = {interface: subprocess.Popen(["ping", "-6", "-n", "-w", "2", "-I", interface, "ff02::1"],
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
             for interface in sorted({f["netdev"] for f in functions if f["addresses"]})}
    replies = {}
    for interface, process in pings.items():
        try:
            output = process.communicate(timeout=5)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = process.communicate()[0]
        replies[interface] = set(re.findall(r"from (fe80:[0-9a-f:]+)", output))
    neighbors = json.loads(command(["ip", "-j", "-6", "neigh", "show"]))
    # A cached neighbor keeps its link-layer address for a while after the host
    # stops using the IPv6 address, so an entry that missed the multicast echo
    # gets one unicast echo before discovery signs in to it. A host that ignores
    # multicast echo still answers here.
    checks = {}
    for index, neighbor in enumerate(neighbors):
        interface, address = neighbor.get("dev"), neighbor.get("dst", "").split("%")[0]
        if interface in replies and "lladdr" in neighbor and address not in replies[interface]:
            checks[index] = subprocess.Popen(["ping", "-6", "-n", "-c", "1", "-W", "1", "-I", interface, address],
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for index, neighbor in enumerate(neighbors):
        if index in checks:
            try:
                neighbor["answered"] = checks[index].wait(timeout=5) == 0
            except subprocess.TimeoutExpired:
                checks[index].kill()
                neighbor["answered"] = False
        elif neighbor.get("dev") in replies and "lladdr" in neighbor:
            neighbor["answered"] = True
    routes = json.loads(command(["ip", "-j", "-4", "route", "show", "table", "main"]))
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    uplink = next((r.get("dev") for r in routes if r.get("dst") == "default"), None)
    api_address = next((a["local"] for row in links if row["ifname"] == uplink for a in row.get("addr_info", []) if a["family"] == "inet"), None)
    machine_id = Path("/etc/machine-id").read_text().strip()
    return {"id": fabric_identity(guids, machine_id), "machine_id": machine_id, "hostname": socket.gethostname(),
            "architecture": platform.machine(), "os": release, "functions": functions, "neighbors": neighbors,
            "routes": routes, "uplink": uplink, "api_address": api_address}


UNANSWERED = ("timed out", "No route to host", "Network is unreachable")


class Unanswered(ValueError):
    """An SSH sign-in that got no answer from the neighbor address.

    The two PCIe functions of a ConnectX port share one cable, so a neighbor
    address can appear in the cache of both while only one reaches it.
    Discovery records such an address and tries the next one.
    """


def login_failure(hop, errors):
    """One line saying why an interactive SSH sign-in to hop failed, from OpenSSH's error output.

    The installer shows only the last line of an error, so the cause comes
    first and OpenSSH's own message follows in parentheses.
    """
    lines = [line.strip() for line in errors.splitlines()
             if line.strip() and not line.startswith("Warning: Permanently added") and not set(line.strip()) <= {"@"}]
    where = f"{hop['address']} on {hop['interface']}" if hop["interface"] else f"{hop['address']} on the LAN"
    target = hop["address"] + (f"%{hop['interface']}" if hop["interface"] else "")
    offending = next((line for line in lines if line.startswith("Offending")), None)
    if any(text in errors for text in ("Permission denied", "Too many authentication failures")):
        message = (f"{where} did not accept the password or account {hop['user']}; test it with "
                   f"ssh -p {hop['port']} {hop['user']}@{target} true, and add "
                   "--ssh-user NAME if that Spark's account has another name")
    elif "Connection refused" in errors:
        message = (f"{where} refused SSH on port {hop['port']}; start SSH on that Spark "
                   "(sudo systemctl enable --now ssh) or prepare it offline with sudo sparkring setup --worker-bundle")
    elif any(text in errors for text in UNANSWERED):
        message = f"{where} did not answer SSH; check the fabric cable to that Spark"
    elif "REMOTE HOST IDENTIFICATION HAS CHANGED" in errors or offending:
        message = (f"{where} presented an SSH host key that differs from the one recorded for it; if that Spark "
                   "was reinstalled, remove its known_hosts line")
    elif "Host key verification failed" in errors:
        # OpenSSH prints the same last line when strict checking meets a key it has no record of.
        message = (f"{where} has no recorded SSH host key yet; run the command in a terminal to compare and "
                   "accept its fingerprint")
    elif any(text in errors for text in ("Connection closed", "Connection reset")):
        message = (f"{where} closed the SSH connection before sign-in finished; SSH closes a password prompt "
                   "left unanswered for about 2 minutes, so run the command again and answer it")
    else:
        message = f"SSH sign-in to {hop['user']}@{where} failed"
    detail = offending or (lines[-1] if lines else "")
    return message + (f" (ssh: {detail.removeprefix('ssh: ')})" if detail else "")


def validate_hop(hop):
    """A fabric hop is a link-local IPv6 address on a named interface; a LAN hop
    (interface None) is a private IPv4 address that setup found on Node A's LAN."""
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", hop["user"]):
        raise ValueError("Invalid SSH login name")
    if hop["interface"] is None:
        if not ipaddress.IPv4Address(hop["address"]).is_private or hop["port"] != 22:
            raise ValueError("A LAN bootstrap hop requires a private IPv4 address on port 22")
        return hop
    if not ipaddress.IPv6Address(hop["address"]).is_link_local:
        raise ValueError("Bootstrap requires link-local IPv6 peers")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,15}", hop["interface"]) or hop["port"] not in (22, 2222):
        raise ValueError("Invalid bootstrap SSH interface/port")
    return hop


def ssh_argv(route, directory, *, interactive=False, identity=None, trust_new=False):
    """Every hop authenticates from Node A; no private key is sent to a worker.

    An interactive login asks about an unknown host key, or with ``trust_new``
    records it on first contact. New keys go to a SparkRing-owned file in
    ``directory``, unhashed, so setup can print what it trusted; keys already
    in the user's known_hosts are still honored.
    """
    if not route or len(route) > MAX_HOPS:
        raise ValueError(f"Bootstrap SSH route requires one through {MAX_HOPS} hops")
    hop = validate_hop(route[-1])
    socket_id = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()[:20]
    check = ("accept-new" if trust_new else "ask") if interactive else "yes"
    command = ["ssh", "-o", "StrictHostKeyChecking=" + check,
               "-o", "UserKnownHostsFile=" + str(Path(directory) / "known_hosts") + " ~/.ssh/known_hosts",
               "-o", "HashKnownHosts=no",
               "-o", "BatchMode=no" if interactive else "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "ControlMaster=auto", "-o", "ControlPersist=600", "-o", "ControlPath=" + str(Path(directory) / socket_id),
               "-p", str(hop["port"])]
    if len(route) > 1:
        jump = ssh_argv(route[:-1], directory, identity=identity)
        # The final zone belongs to the jump host, which resolves -W's target.
        jump[-1:-1] = ["-W", f"[{hop['address']}%{hop['interface']}]:{hop['port']}"]
        # ssh expands % tokens in ProxyCommand before the shell sees it. Each
        # nesting level must preserve the link-local zone for its own hop.
        command += ["-o", "ProxyCommand=" + shlex.join(jump).replace("%", "%%")]
    if identity is not None:
        command += ["-i", str(identity)]
    command.append(hop["user"] + "@" + hop["address"] + ("" if hop["interface"] is None else "%" + hop["interface"]))
    return command


class SSH:
    def __init__(self, directory, *, identity=None, run=subprocess.run, trust_new=False):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run = run
        self.identity = identity
        self.trust_new = trust_new

    def argv(self, route, interactive=False):
        return ssh_argv(route, self.directory, interactive=interactive, identity=self.identity,
                        trust_new=self.trust_new)

    def trusted(self):
        """Fingerprint lines for host keys this setup recorded on first contact."""
        known = self.directory / "known_hosts"
        if not known.is_file():
            return []
        result = self.run(["ssh-keygen", "-l", "-f", str(known)], capture_output=True, text=True)
        return result.stdout.splitlines() if result.returncode == 0 else []

    def login(self, route):
        # OpenSSH owns password and host-key prompts, which use the terminal
        # directly; secrets never enter Python or JSON. Its error output goes
        # to a file, so a failure can say why.
        with tempfile.TemporaryFile("w+") as errors:
            result = self.run([*self.argv(route, interactive=True), "true"], stderr=errors)
            errors.seek(0)
            text = errors.read()
        if result.returncode:
            unanswered = any(marker in text for marker in UNANSWERED)
            raise (Unanswered if unanswered else ValueError)(login_failure(route[-1], text))

    def command(self, route, argv, *, data=None, tty=False):
        command = argv if not route else [*self.argv(route), *( ["-tt"] if tty else []), shlex.join(argv)]
        result = self.run(command, input=data, capture_output=not tty, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError("Bootstrap command failed: " + (result.stderr or "inspect terminal output"))
        return result.stdout

    def inventory(self, route):
        code = (inspect.getsource(fabric_identity) + "\n" + inspect.getsource(probe)
                + "\nimport json\nprint(json.dumps(probe()))\n")
        return json.loads(self.command(route, ["python3", "-I", "-c", code]))

    def close(self):
        """End the SSH connections this transport keeps open (ControlPersist) through its directory's sockets."""
        for path in sorted(self.directory.glob("[0-9a-f]" * 20)):
            if path.is_socket():
                self.run(["ssh", "-o", "ControlPath=" + str(path), "-O", "exit", "sparkring-control-socket"],
                         capture_output=True, text=True, timeout=30)


def discover(transport, *, user="root", port=22, select=lambda peer: True, hops=MAX_HOPS):
    """Authenticate the Sparks reachable over fabric link-local addresses.

    A neighbor is followed only on routes of at most ``hops`` cables
    (``hop_limit`` of the expected number of Sparks).

    Returns the head's identity, every authenticated Spark's inventory, the
    cables between them, the SSH route to each, and warnings: Sparks that
    share /etc/machine-id are named, because discovery tells them apart but
    other software on them may not.
    """
    head = transport.inventory([])
    nodes = {head["id"]: head}
    routes = {head["id"]: []}
    queue, observed, edges, skipped = [head["id"]], set(), {}, []

    def link(ident, interface, peer, other, address):
        cable = tuple(sorted((ident, peer)))
        if cable not in edges:
            if not interface["addresses"]:
                raise ValueError("Local fabric link has no IPv6 link-local address")
            edges[cable] = [{"id": ident, "netdev": interface["netdev"], "address": interface["addresses"][0], "mac": interface["mac"]},
                            {"id": peer, "netdev": other["netdev"], "address": address, "mac": other["mac"]}]

    for ident in queue:
        current = nodes[ident]
        local = {f["netdev"]: f for f in current["functions"]}
        # The two PCIe functions of one physical port share its cable, so a
        # host sees its own sibling function as a link-local neighbor.
        own = {str(f["mac"]).lower() for f in current["functions"]}
        # Links on which some other host answered the probe's echo. There, an
        # entry that did not answer is a stale cache entry, not a host.
        answering = {n.get("dev") for n in current["neighbors"]
                     if n.get("answered") and str(n.get("lladdr", "")).lower() not in own}
        # Answering entries come first, so a Spark is authenticated through
        # the address it uses before any stale entry for its MAC is examined.
        for neighbor in sorted(current["neighbors"], key=lambda n: not n.get("answered")):
            interface = local.get(neighbor.get("dev"))
            if not interface or "lladdr" not in neighbor or neighbor["lladdr"].lower() in own:
                continue
            address = ipaddress.IPv6Address(neighbor["dst"].split("%")[0])
            if not address.is_link_local or str(address) in interface["addresses"]:
                continue
            observation = (ident, interface["netdev"], str(address))
            if observation in observed:
                continue
            observed.add(observation)
            mac = neighbor["lladdr"].lower()
            others = [(nid, f) for nid, n in nodes.items() if nid != ident for f in n["functions"] if str(f["mac"]).lower() == mac]
            # The inventory of an authenticated Spark lists every function's MAC
            # and link-local address, which identifies its other functions and
            # the return paths to it without signing in again.
            known = [(nid, f) for nid, f in others if str(address) in f["addresses"]]
            if len(known) == 1:
                link(ident, interface, known[0][0], known[0][1], str(address))
                continue
            if others:
                # The MAC belongs to an authenticated Spark that does not hold this address: a stale cache entry.
                continue
            if neighbor.get("answered") is False and neighbor.get("dev") in answering:
                skipped.append(f"{address} on {current['hostname']}'s {interface['netdev']} (no echo reply)")
                continue
            # An unrecognized function can still lead to an already enrolled
            # machine; authenticate before assigning either identity or rank.
            route = [*routes[ident], {"user": user, "address": str(address), "interface": interface["netdev"], "port": port}]
            if len(route) > min(hops, MAX_HOPS) or not select({"via": current["hostname"], "interface": interface["netdev"], "address": str(address)}):
                continue
            try:
                transport.login(route)
            except Unanswered:
                skipped.append(f"{address} on {current['hostname']}'s {interface['netdev']} (no SSH answer)")
                continue
            peer = transport.inventory(route)
            if peer["id"] == ident:
                raise ValueError(f"{current['hostname']} reached itself at {address} through {interface['netdev']}; "
                                 "each fabric cable must connect two different Sparks")
            matches = [f for f in peer["functions"] if str(address) in f["addresses"] and f["mac"].lower() == mac]
            if len(matches) != 1:
                raise ValueError(f"{peer['hostname']} answered at {address} on {current['hostname']}'s {interface['netdev']}, "
                                 f"but none of its fabric functions has that address with MAC {mac}; check the fabric cabling")
            if peer["architecture"] not in ("aarch64", "arm64"):
                raise ValueError("Fabric neighbor is not Linux ARM64")
            link(ident, interface, peer["id"], matches[0], str(address))
            if peer["id"] not in nodes:
                if len(nodes) >= fabric_layout.MAX_SPARKS:
                    raise ValueError(f"More than {fabric_layout.WORDS[fabric_layout.MAX_SPARKS]} Sparks found; "
                                     "SparkRing sets up two to eight cabled Sparks")
                nodes[peer["id"]], routes[peer["id"]] = peer, route
                queue.append(peer["id"])
    if not fabric_layout.MIN_SPARKS <= len(nodes) <= fabric_layout.MAX_SPARKS:
        found = ", ".join(n["hostname"] for n in nodes.values())
        detail = f" Skipped neighbor addresses that did not answer: {', '.join(skipped)}." if skipped else ""
        raise ValueError(f"Found {len(nodes)} Spark{'s' if len(nodes) != 1 else ''} ({found}); setup needs two to "
                         "eight."
                         + detail + " Check the fabric cables and that each Spark accepts SSH.")
    by_machine = {}
    for n in nodes.values():
        if n.get("machine_id"):
            by_machine.setdefault(n["machine_id"], []).append(n["hostname"])
    warnings = [" and ".join(names) + " share /etc/machine-id, as Sparks flashed from one factory image do. "
                "SparkRing tells them apart by their ConnectX hardware; other software, such as DHCP, may not. "
                "To give a Spark its own ID: sudo rm -f /etc/machine-id && sudo systemd-machine-id-setup && sudo reboot"
                for names in by_machine.values() if len(names) > 1]
    return {"head": head["id"], "nodes": list(nodes.values()), "edges": list(edges.values()), "routes": routes,
            "warnings": warnings}
