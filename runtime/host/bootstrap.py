"""Explore authenticated IPv6 neighbors and provision packages through SSH hops."""
import hashlib
import inspect
import ipaddress
import json
from pathlib import Path
import re
import shlex
import subprocess


def probe():
    """Self-contained read-only probe suitable for a Spark without this package."""
    import json
    from pathlib import Path
    import platform
    import socket
    import subprocess

    def command(argv):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=20)
        if result.returncode:
            raise RuntimeError(result.stderr.strip())
        return result.stdout

    links = json.loads(command(["ip", "-j", "address", "show"]))
    functions = []
    for directory in sorted(Path("/sys/class/infiniband").iterdir()):
        for nic in (directory / "device/net").iterdir():
            link = next(row for row in links if row["ifname"] == nic.name)
            functions.append({"device": directory.name, "netdev": nic.name, "mac": link.get("address"),
                              "addresses": [a["local"] for a in link.get("addr_info", []) if a["family"] == "inet6" and a["scope"] == "link"]})
    # Multicast is discovery only; missing replies never imply missing hosts.
    for interface in {f["netdev"] for f in functions if f["addresses"]}:
        subprocess.run(["ping", "-6", "-n", "-c", "1", "-w", "2", "-I", interface, "ff02::1"],
                       capture_output=True, timeout=5)
    neighbors = json.loads(command(["ip", "-j", "-6", "neigh", "show"]))
    routes = json.loads(command(["ip", "-j", "-4", "route", "show", "table", "main"]))
    release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    uplink = next((r.get("dev") for r in routes if r.get("dst") == "default"), None)
    api_address = next((a["local"] for row in links if row["ifname"] == uplink for a in row.get("addr_info", []) if a["family"] == "inet"), None)
    return {"id": Path("/etc/machine-id").read_text().strip(), "hostname": socket.gethostname(),
            "architecture": platform.machine(), "os": release, "functions": functions, "neighbors": neighbors,
            "routes": routes, "uplink": uplink, "api_address": api_address}


def validate_hop(hop):
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", hop["user"]):
        raise ValueError("Invalid SSH login name")
    if not ipaddress.IPv6Address(hop["address"]).is_link_local:
        raise ValueError("Bootstrap requires link-local IPv6 peers")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,15}", hop["interface"]) or hop["port"] not in (22, 2222):
        raise ValueError("Invalid bootstrap SSH interface/port")
    return hop


def ssh_argv(route, directory, *, interactive=False, identity=None):
    """Every hop authenticates from Node A; no private key is sent to a worker."""
    if not route or len(route) > 3:
        raise ValueError("Bootstrap SSH route requires one through three hops")
    hop = validate_hop(route[-1])
    identity = hashlib.sha256(json.dumps(route, sort_keys=True).encode()).hexdigest()[:20]
    command = ["ssh", "-o", "StrictHostKeyChecking=ask" if interactive else "StrictHostKeyChecking=yes",
               "-o", "BatchMode=no" if interactive else "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "ControlMaster=auto", "-o", "ControlPersist=600", "-o", "ControlPath=" + str(Path(directory) / identity),
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
    command.append(hop["user"] + "@" + hop["address"] + "%" + hop["interface"])
    return command


class SSH:
    def __init__(self, directory, *, identity=None, run=subprocess.run):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.run = run
        self.identity = identity

    def argv(self, route, interactive=False):
        return ssh_argv(route, self.directory, interactive=interactive, identity=self.identity)

    def login(self, route):
        # OpenSSH owns password/key prompts. Secrets never enter Python/JSON.
        result = self.run([*self.argv(route, interactive=True), "true"])
        if result.returncode:
            raise ValueError("SSH enrollment failed. Enable SSH locally or run the worker preparation bundle.")

    def command(self, route, argv, *, data=None, tty=False):
        command = argv if not route else [*self.argv(route), *( ["-tt"] if tty else []), shlex.join(argv)]
        result = self.run(command, input=data, capture_output=not tty, text=True, timeout=1800)
        if result.returncode:
            raise RuntimeError("Bootstrap command failed: " + (result.stderr or "inspect terminal output"))
        return result.stdout

    def inventory(self, route):
        code = inspect.getsource(probe) + "\nimport json\nprint(json.dumps(probe()))\n"
        return json.loads(self.command(route, ["python3", "-I", "-c", code]))


def discover(transport, *, user="root", port=22, select=lambda peer: True):
    head = transport.inventory([])
    nodes = {head["id"]: head}
    routes = {head["id"]: []}
    queue, observed, edges = [head["id"]], set(), {}
    for ident in queue:
        current = nodes[ident]
        local = {f["netdev"]: f for f in current["functions"]}
        for neighbor in current["neighbors"]:
            interface = local.get(neighbor.get("dev"))
            if not interface or "lladdr" not in neighbor:
                continue
            address = ipaddress.IPv6Address(neighbor["dst"].split("%")[0])
            if not address.is_link_local or str(address) in interface["addresses"]:
                continue
            observation = (ident, interface["netdev"], str(address))
            if observation in observed:
                continue
            observed.add(observation)
            # A second Socket Direct function can lead to an already enrolled
            # machine; authenticate before assigning either identity or rank.
            route = [*routes[ident], {"user": user, "address": str(address), "interface": interface["netdev"], "port": port}]
            if len(route) > 3 or not select({"via": current["hostname"], "interface": interface["netdev"], "address": str(address)}):
                continue
            transport.login(route)
            peer = transport.inventory(route)
            matches = [f for f in peer["functions"] if str(address) in f["addresses"] and f["mac"].lower() == neighbor["lladdr"].lower()]
            if len(matches) != 1 or peer["id"] == ident:
                raise ValueError("Neighbor cannot be matched to its authenticated fabric interface")
            if peer["architecture"] not in ("aarch64", "arm64"):
                raise ValueError("Fabric neighbor is not Linux ARM64")
            other = matches[0]
            cable = tuple(sorted((ident, peer["id"])))
            if cable not in edges:
                if not interface["addresses"]:
                    raise ValueError("Local fabric link has no IPv6 link-local address")
                edges[cable] = [{"id": ident, "netdev": interface["netdev"], "address": interface["addresses"][0], "mac": interface["mac"]},
                                {"id": peer["id"], "netdev": other["netdev"], "address": str(address), "mac": other["mac"]}]
            if peer["id"] not in nodes:
                if len(nodes) >= 4:
                    raise ValueError("More than four Sparks found; select a supported pair/ring")
                nodes[peer["id"]], routes[peer["id"]] = peer, route
                queue.append(peer["id"])
    if len(nodes) not in (2, 4):
        raise ValueError("Could not authenticate a pair/ring. Check links, SSH, or run the worker preparation bundle.")
    return {"head": head["id"], "nodes": list(nodes.values()), "edges": list(edges.values()), "routes": routes}
