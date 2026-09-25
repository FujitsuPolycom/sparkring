"""One-time local worker preparation when no usable SSH service exists."""
import json
from pathlib import Path
import re
import subprocess
import time

from runtime.host import control, control_node, node


def gpu_containers(run=subprocess.run):
    """Running containers that requested GPUs, as (id, name) pairs."""
    ids = node.call(["docker", "ps", "-q"], run=run).stdout.split()
    if not ids:
        return []
    rows = json.loads(node.call(["docker", "inspect", *ids], run=run).stdout)
    return [(row["Id"], row["Name"].lstrip("/")) for row in rows if row["HostConfig"].get("DeviceRequests")]


def prepare(public_key, *, run=subprocess.run, interfaces=None, stop=None, link_local=None):
    """Prepare fabric ports for discovery.

    Bringing up an unconfigured RDMA port requires that no GPU job or RDMA user is
    running; containers without GPUs (tools, registries) do not matter. When GPU
    containers are running, ``stop(names)`` is asked for approval; approved
    containers are stopped, never removed.

    Discovery uses IPv6 link-local addresses. An existing fabric connection
    without one (for example manual IPv4 with IPv6 disabled) gets IPv6
    link-local added to that same connection after ``link_local(name)``
    approves; its IPv4 addresses and MTU are kept.
    """
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]*)?", public_key.strip()):
        raise ValueError("Use Node A's Ed25519 public key")
    def idle():
        busy = gpu_containers(run)
        if busy and stop is not None:
            stop([name for _, name in busy])
            for ident, _ in busy:
                node.call(["docker", "stop", "--time", "60", ident], run=run)
            busy = gpu_containers(run)
        if busy:
            names = ", ".join(name for _, name in busy)
            raise ValueError(f"GPU containers are running ({names}). Stop them (docker stop NAME) or rerun with "
                             "--stop-workloads, then repeat setup")
        gpu = node.call(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], run=run).stdout.strip()
        if gpu:
            raise ValueError("A GPU job outside containers is running; stop it before preparing fabric ports")
        # Each port always has kernel-owned management queue pairs (GSI/SMI,
        # no pid); only queue pairs owned by a process are RDMA users.
        users = sorted({row.get("comm", "?") for row in json.loads(node.call(["rdma", "-j", "resource", "show", "qp"], run=run).stdout)
                        if row.get("pid") is not None})
        if users:
            raise ValueError("Stop RDMA users before preparing fabric ports: " + ", ".join(users))

    if interfaces is None:
        interfaces = sorted({p.name for d in Path("/sys/class/infiniband").iterdir() for p in (d / "device/net").iterdir()})
    if len(interfaces) != 4:
        raise ValueError("Worker preparation expects four ConnectX RDMA functions")
    for interface in interfaces:
        control.netdev(interface)
        current = node.call(["nmcli", "-g", "GENERAL.CON-UUID", "device", "show", interface], run=run).stdout.strip()
        addresses = json.loads(node.call(["ip", "-j", "-6", "addr", "show", "dev", interface], run=run).stdout)
        if current and current != "--" and any(a["scope"] == "link" for row in addresses for a in row.get("addr_info", [])):
            # Discovering an already configured link does not require detaching
            # the existing native mesh's RDMA marker processes.
            continue
        idle()
        node.call(["ip", "link", "set", "dev", interface, "up"], run=run)
        if (Path("/sys/class/net") / interface / "carrier").read_text().strip() != "1":
            continue
        current = node.call(["nmcli", "-g", "GENERAL.CON-UUID", "device", "show", interface], run=run).stdout.strip()
        if current and current != "--":
            # Keep existing IPv4 and IPv6 configuration if a link-local address
            # already works. Replacing a nonempty profile belongs to setup review.
            addresses = json.loads(node.call(["ip", "-j", "-6", "addr", "show", "dev", interface], run=run).stdout)
            if not any(a["scope"] == "link" for row in addresses for a in row.get("addr_info", [])):
                name = node.call(["nmcli", "-g", "connection.id", "connection", "show", current], run=run).stdout.strip()
                if link_local is None:
                    raise ValueError(f"Fabric connection '{name}' on {interface} has no IPv6 link-local address; "
                                     "repeat setup and approve adding it (IPv4 settings are kept)")
                link_local(f"{name} ({interface})")
                node.call(["nmcli", "connection", "modify", current, "ipv6.method", "link-local"], run=run)
                node.call(["nmcli", "device", "reapply", interface], run=run)
                for _ in range(20):
                    addresses = json.loads(node.call(["ip", "-j", "-6", "addr", "show", "dev", interface], run=run).stdout)
                    if any(a["scope"] == "link" and not a.get("tentative") for row in addresses for a in row.get("addr_info", [])):
                        break
                    time.sleep(0.5)
                else:
                    raise ValueError(f"{interface}: IPv6 link-local address did not become ready")
        else:
            observed = json.loads(node.call(["ip", "-j", "-4", "addr", "show", "dev", interface], run=run).stdout)
            if any(row.get("addr_info") for row in observed):
                raise ValueError("Unmanaged existing IPv4 configuration on " + interface + "; inspect before preparing")
            node.call(["nmcli", "connection", "add", "type", "ethernet", "ifname", interface, "con-name", "sparkring-bootstrap-" + interface,
                       "ipv4.method", "disabled", "ipv6.method", "link-local", "connection.autoconnect", "yes"], run=run)
            node.call(["nmcli", "connection", "up", "sparkring-bootstrap-" + interface], run=run)
    # Preparation and the permanent administration SSH both use TCP 2222. Only
    # SparkRing's own active preparation service may already hold it.
    listeners = node.call(["ss", "-H", "-ltnp", "sport", "=", ":2222"], run=run).stdout
    if listeners.strip():
        holders = sorted(set(re.findall(r'\(\("([^"]+)"', listeners)))
        ours = holders == ["sshd"] and run(["systemctl", "is-active", "--quiet", "sparkring-seed.service"],
                                           capture_output=True, text=True).returncode == 0
        if not ours:
            raise ValueError("TCP port 2222 is in use by " + (", ".join(holders) or "another service")
                             + ". SparkRing's preparation and administration SSH need it; stop that service "
                             "(for example: sudo systemctl disable --now dropbear) and repeat")
    node.call(["systemctl", "enable", "--now", "ssh.service"], run=run)
    control_node.write("/etc/sparkring/seed_keys", public_key.strip() + "\n")
    # This separate SSH service accepts only Node A's public key. Existing SSH
    # authentication policy and authorized_keys files are preserved.
    control_node.write("/etc/sparkring/seed_sshd_config", """Port 2222
AddressFamily inet6
ListenAddress ::
HostKey /etc/ssh/ssh_host_ed25519_key
PidFile /run/sparkring-seed.pid
AuthorizedKeysFile /etc/sparkring/seed_keys
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthenticationMethods publickey
AllowUsers root
UsePAM yes
AllowAgentForwarding no
AllowTcpForwarding yes
PermitUserRC no
""")
    node.call(["/usr/sbin/sshd", "-t", "-f", "/etc/sparkring/seed_sshd_config"], run=run)
    node.call(["systemctl", "enable", "--now", "sparkring-seed.service"], run=run)
    return {"prepared": True, "next_action": "On Node A: sudo sparkring setup --ssh-port 2222"}
