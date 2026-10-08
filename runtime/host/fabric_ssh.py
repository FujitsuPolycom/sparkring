"""Authenticated bulk paths derived from the approved fabric, on Node A only."""
from collections import deque
import ipaddress
import json
from pathlib import Path
import shlex
import subprocess

from runtime.common import fabric_layout
from runtime.host import control as control_network, control_node, node, topology


def routes(cluster):
    """The bulk-transfer tree from Node A: each Spark's parent and its primary fabric address on their cable.

    Breadth first along the cables, port 0's neighbor first, so a ring
    splits into two directions and a line has one.
    """
    hosts = cluster["plan"]["spec"]["hosts"]
    size = len(hosts)
    layout = topology.layout_of(cluster["plan"])
    if [h["rank"] for h in hosts] != list(range(size)):
        raise ValueError("Bulk transfer requires the discovered Sparks in position order")
    result = {0: {"rank": 0, "parent": None, "address": None}}
    queue = deque([0])
    while queue:
        parent = queue.popleft()
        for port in (0, 1):
            far = fabric_layout.peer(layout, parent, port)
            if far is None or far[0] in result:
                continue
            rank = far[0]
            source_role = fabric_layout.port_role(port, "primary")
            target_role = fabric_layout.port_role(far[1], "primary")
            source = next(p for p in hosts[parent]["data_interfaces"] if p["role"] == source_role)
            target = next(p for p in hosts[rank]["data_interfaces"] if p["role"] == target_role)
            a, b = (ipaddress.IPv4Interface(p["address"]) for p in (source, target))
            if a.network != b.network or a.ip == b.ip:
                raise ValueError("Discovered cable endpoints do not share their recorded fabric subnet")
            result[rank] = {"rank": rank, "parent": parent, "address": str(b.ip), "source_interface": source["netdev"]}
            queue.append(rank)
    return [result[n] for n in range(size)]


class Transport:
    """Keep control credentials on the head; bulk bytes never traverse the caller's PC.

    ``document`` is the cluster's fabric document (``sparkring-fabric/v1``)
    when one is recorded for these Sparks; transfers then spread along its
    cables (``runtime/host/spread.py``). ``positions`` gives each rank's
    fabric position, which equals the rank for the whole cluster.
    """
    def __init__(self, cluster, directory, *, root="/", run=subprocess.run, document=None):
        self.cluster, self.run = cluster, run
        self.hosts = cluster["plan"]["spec"]["hosts"]
        self.document = document
        self.positions = list(range(len(self.hosts)))
        self.routes = routes(cluster)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.config = self.directory / "ssh_config"
        self.mode = "fiber-ssh"
        control = node.location(root, "/etc/sparkring/control.json")
        config = node.read(root, "/etc/sparkring/control.json") if control.exists() else None
        uses_control = (config is not None and config.get("schema") == "sparkring-control/v1" and config.get("head") is True
                        and self.hosts[0]["management_address"] == config.get("address")
                        and all(ipaddress.ip_address(h["management_address"]) in ipaddress.ip_network(config["subnet"])
                                for h in self.hosts))
        if uses_control:
            # Bulk paths need every administration link; one failing link stops here.
            control_node.require_underlay(root=root, run=run)
            for host in self.hosts[1:]:
                observed = run(["ip", "-j", "route", "get", host["management_address"]], capture_output=True, text=True, check=True)
                rows = json.loads(observed.stdout)
                if not rows or any(row.get("dev") != control_network.INTERFACE for row in rows):
                    raise ValueError("Administration address is not routed through the verified fabric control network")
            self.mode = "fabric-control"
        else:
            self._write_config()

    def _write_config(self):
        lines = []
        for route in self.routes[1:]:
            rank = route["rank"]
            resolved = self.run(["ssh", "-G", self.hosts[rank]["host"]], capture_output=True, text=True, check=True).stdout
            settings = {}
            for line in resolved.splitlines():
                key, _, value = line.partition(" ")
                settings.setdefault(key, []).append(value.strip())

            def first(key, fallback):
                return settings.get(key, [fallback])[0]

            values = {
                "HostName": route["address"], "User": first("user", "root"), "Port": first("port", "22"),
                "HostKeyAlias": first("hostkeyalias", first("hostname", self.hosts[rank]["host"].split("@")[-1])),
                "StrictHostKeyChecking": "yes", "BatchMode": "yes", "ConnectTimeout": "10",
                "Ciphers": "aes128-gcm@openssh.com,aes256-gcm@openssh.com,chacha20-poly1305@openssh.com",
            }
            if route["parent"]:
                values["ProxyJump"] = "sparkring-r" + str(route["parent"])
            lines.append("Host sparkring-r" + str(rank))
            for key, value in values.items():
                if any(c in value for c in "\r\n\0"):
                    raise ValueError("Invalid resolved SSH setting")
                lines.append("  " + key + " " + json.dumps(value))
            for path in settings.get("identityfile", []):
                lines.append("  IdentityFile " + json.dumps(path))
            for key in ("userknownhostsfile", "globalknownhostsfile"):
                if key in settings:
                    paths = shlex.split(settings[key][0])
                    lines.append("  " + key + " " + " ".join(json.dumps(p) for p in paths))
        if self.config.is_symlink():
            raise ValueError("Bulk SSH configuration cannot be a symlink")
        self.config.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.config.chmod(0o600)

    def local(self, rank):
        """Whether rank ``rank`` is Node A, which runs this transport's commands itself."""
        return rank == 0

    def view(self, ranks):
        """This transport limited to ``ranks``, renumbered from 0 in that order (``View``)."""
        return View(self, ranks)

    def argv(self, rank):
        if rank == 0:
            return []
        if self.mode == "fabric-control":
            return ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes", self.hosts[rank]["host"]]
        return ["ssh", "-F", str(self.config), "sparkring-r" + str(rank)]

    def command(self, rank, argv):
        if rank == 0:
            return list(argv)
        return [*self.argv(rank), shlex.join(argv)]

    def forwarded(self, rank, remote_port, local_port, argv):
        """Run argv on a worker whose loopback remote_port reaches Node A's loopback local_port."""
        if rank == 0:
            raise ValueError("Node A reaches its own loopback services directly")
        ssh = self.argv(rank)
        return [ssh[0], "-o", "ExitOnForwardFailure=yes", "-R", f"127.0.0.1:{remote_port}:127.0.0.1:{local_port}",
                *ssh[1:], shlex.join(argv)]

    def verify(self):
        """Authenticate the selected physical path against the enrolled node ID."""
        probe = "import json; print(json.load(open('/etc/sparkring/node.json'))['node_id'])"
        # Prove each SSH endpoint's kernel route uses the discovered data link.
        # A static address alone must not silently route traffic over Ethernet.
        if self.mode == "fiber-ssh":
            for route in self.routes[1:]:
                result = self.run(self.command(route["parent"], ["ip", "-j", "route", "get", route["address"]]),
                                  capture_output=True, text=True, timeout=30, check=True)
                observed = json.loads(result.stdout)
                if not observed or any(row.get("dev") != route["source_interface"] for row in observed):
                    raise ValueError("Bulk transfer route does not use the discovered fabric interface")
        for host in self.hosts:
            rank = host["rank"]
            result = self.run(self.command(rank, ["python3", "-I", "-c", probe]),
                              capture_output=True, text=True, timeout=30, check=True)
            if result.stdout.strip() != host["node_id"]:
                raise ValueError(f"Node {rank}: fabric path reached a different persistent node identity")
        return {"transport": self.mode, "ranks": len(self.hosts), "caller_relay": False}


class View:
    """A transport limited to some of the cluster's ranks, renumbered from 0 in their order.

    A deployment on half of a four-Spark ring numbers its two Sparks 0 and 1;
    checkpoint transfers address them by those numbers, and this view sends
    each command to the cluster rank it stands for. Node A runs a rank's
    commands itself only when that rank is Node A (``local``).
    """

    def __init__(self, transport, ranks):
        self.transport, self.ranks = transport, list(ranks)
        self.hosts = [transport.hosts[rank] for rank in self.ranks]
        self.mode = transport.mode
        self.document = getattr(transport, "document", None)
        positions = getattr(transport, "positions", range(len(transport.hosts)))
        self.positions = [positions[rank] for rank in self.ranks]

    def local(self, rank):
        return self.transport.local(self.ranks[rank])

    def argv(self, rank):
        return self.transport.argv(self.ranks[rank])

    def command(self, rank, argv):
        return self.transport.command(self.ranks[rank], argv)

    def forwarded(self, rank, remote_port, local_port, argv):
        return self.transport.forwarded(self.ranks[rank], remote_port, local_port, argv)

    def verify(self):
        return self.transport.verify()
