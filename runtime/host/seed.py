"""One-time local worker preparation when no usable SSH service exists."""
import json
from pathlib import Path
import re
import subprocess

from runtime.host import control, control_node, node


def prepare(public_key, *, run=subprocess.run, interfaces=None):
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]*)?", public_key.strip()):
        raise ValueError("Use Node A's Ed25519 public key")
    def idle():
        containers = node.call(["docker", "ps", "-q"], run=run).stdout.strip()
        gpu = node.call(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], run=run).stdout.strip()
        if containers or gpu:
            raise ValueError("Stop running containers and GPU jobs before worker preparation")
        if json.loads(node.call(["rdma", "-j", "resource", "show", "qp"], run=run).stdout):
            raise ValueError("Stop RDMA users before worker preparation")

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
                raise ValueError("Enable IPv6 link-local on existing fabric connection " + current + " after reviewing its settings")
        else:
            observed = json.loads(node.call(["ip", "-j", "-4", "addr", "show", "dev", interface], run=run).stdout)
            if any(row.get("addr_info") for row in observed):
                raise ValueError("Unmanaged existing IPv4 configuration on " + interface + "; inspect before preparing")
            node.call(["nmcli", "connection", "add", "type", "ethernet", "ifname", interface, "con-name", "sparkring-bootstrap-" + interface,
                       "ipv4.method", "disabled", "ipv6.method", "link-local", "connection.autoconnect", "yes"], run=run)
            node.call(["nmcli", "connection", "up", "sparkring-bootstrap-" + interface], run=run)
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
