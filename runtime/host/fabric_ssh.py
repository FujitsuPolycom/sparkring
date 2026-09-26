"""Authenticated bulk paths derived from the approved fabric, on Node A only."""
from collections import deque
import ipaddress
import json
from pathlib import Path
import shlex
import subprocess

from runtime.host import control as control_network, control_node, node


def routes(cluster):
    hosts = cluster["plan"]["spec"]["hosts"]
    size = len(hosts)
    if size not in (2, 4) or [h["rank"] for h in hosts] != list(range(size)):
        raise ValueError("Bulk transfer requires the discovered pair or four-node ring")
    result = {0: {"rank": 0, "parent": None, "address": None}}
    queue = deque([0])
    while queue:
        parent = queue.popleft()
        neighbors = [1 - parent] if size == 2 else [(parent + 1) % size, (parent - 1) % size]
        for rank in neighbors:
            if rank in result:
                continue
            clockwise = size == 2 or rank == (parent + 1) % size
            source_role = "cw_primary" if clockwise else "ccw_primary"
            target_role = "cw_primary" if size == 2 else "ccw_primary" if clockwise else "cw_primary"
            source = next(p for p in hosts[parent]["data_interfaces"] if p["role"] == source_role)
            target = next(p for p in hosts[rank]["data_interfaces"] if p["role"] == target_role)
            a, b = (ipaddress.IPv4Interface(p["address"]) for p in (source, target))
            if a.network != b.network or a.ip == b.ip:
                raise ValueError("Discovered cable endpoints do not share their recorded fabric subnet")
            result[rank] = {"rank": rank, "parent": parent, "address": str(b.ip), "source_interface": source["netdev"]}
            queue.append(rank)
    return [result[n] for n in range(size)]


class Transport:
    """Keep control credentials on the head; bulk bytes never traverse the caller's PC."""
    def __init__(self, cluster, directory, *, root="/", run=subprocess.run):
        self.cluster, self.run = cluster, run
        self.hosts = cluster["plan"]["spec"]["hosts"]
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
