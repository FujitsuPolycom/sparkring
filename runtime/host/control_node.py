"""Privileged local implementation of the reviewed fabric administration tree."""
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess

from runtime.host import control, node


def write(path, text, *, root="/", mode=0o600):
    destination = node.location(root, path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = node.location(root, str(destination.relative_to(root)) + ".writing")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.chmod(mode)
    temporary.replace(destination)


def public_key(*, root="/", run=subprocess.run):
    private = node.location(root, "/etc/sparkring/control.key")
    if not private.exists():
        generated = node.call(["wg", "genkey"], run=run).stdout.strip()
        control.key(generated)
        write("/etc/sparkring/control.key", generated + "\n", root=root)
    result = run(["wg", "pubkey"], input=private.read_text(), capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise ValueError("Cannot derive control public key")
    return {"public_key": control.key(result.stdout.strip()),
            "host_key": node.location(root, "/etc/ssh/ssh_host_ed25519_key.pub").read_text().strip()}


def underlay(*, root="/", run=subprocess.run):
    path = node.location(root, "/etc/sparkring/control.json")
    if not path.exists():
        return
    config = node.read(root, "/etc/sparkring/control.json")
    for link in config["links"]:
        interface = control.netdev(link["netdev"])
        # sysfs itself uses symlinks: inspect the approved NIC by kernel name.
        actual = Path(root) / "sys/class/net" / interface / "address"
        if actual.read_text().strip().lower() != link["mac"].lower() or not ipaddress.IPv6Address(link["address"]).is_link_local:
            raise ValueError("Control underlay NIC identity changed")
        observed = json.loads(node.call(["ip", "-j", "-6", "address", "show", "dev", interface], run=run).stdout)
        addresses = [a["local"] for row in observed for a in row.get("addr_info", []) if a["scope"] == "link"]
        if addresses != [link["address"]]:
            raise ValueError("Control IPv6 link changed; retain its original NetworkManager connection identity")


def configure(document, *, root="/", run=subprocess.run):
    config = document["control"]
    if config.get("schema") != "sparkring-control/v1" or config["id"] != node.location(root, "/etc/machine-id").read_text().strip():
        raise ValueError("Control configuration belongs to another machine")
    pubkey = document["ssh_key"]
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]*)?", pubkey):
        raise ValueError("Use the controller's Ed25519 public key")
    existing = node.location(root, "/etc/sparkring/control.json")
    if existing.exists() and node.read(root, "/etc/sparkring/control.json") != config:
        raise ValueError("A different control network is installed; inspect before replacing it")
    private = node.location(root, "/etc/sparkring/control.key").read_text().strip()
    rendered = control.render(config, private)
    control.firewall(config)
    node.save(root, "/etc/sparkring/control.json", config, mode=0o600)
    write("/etc/wireguard/sr-control.conf", rendered, root=root)
    write("/etc/sparkring/controller_keys", pubkey + "\n", root=root)
    address = str(ipaddress.IPv4Address(config["address"]))
    write("/etc/sparkring/sshd_config", f"""Port {control.SSH_PORT}
ListenAddress {address}
HostKey /etc/ssh/ssh_host_ed25519_key
PidFile /run/sparkring-access.pid
AuthorizedKeysFile /etc/sparkring/controller_keys
PermitRootLogin prohibit-password
PasswordAuthentication no
KbdInteractiveAuthentication no
AuthenticationMethods publickey
AllowUsers root
UsePAM yes
AllowAgentForwarding no
AllowTcpForwarding yes
X11Forwarding no
PermitUserRC no
Subsystem sftp internal-sftp
""", root=root)
    node.call(["/usr/sbin/sshd", "-t", "-f", "/etc/sparkring/sshd_config"], run=run)
    if config["head"] and config["share_uplink"]:
        write("/etc/sparkring/dnsmasq.conf", f"""port=53
listen-address={address}
bind-interfaces
no-hosts
resolv-file=/etc/resolv.conf
cache-size=150
""", root=root)
    node.call(["systemctl", "enable", "sparkring-control.service", "sparkring-access.service"], run=run)
    node.call(["systemctl", "start", "sparkring-control.service", "sparkring-access.service"], run=run)
    node.call(["systemctl", "enable", "--now", "sparkring-control-refresh.timer"], run=run)
    return {"configured": True, "address": address}


def up(*, root="/", run=subprocess.run):
    config = node.read(root, "/etc/sparkring/control.json")
    underlay(root=root, run=run)
    found = node.call(["ip", "-j", "link", "show", "dev", control.INTERFACE], run=run, accepted=(0, 1)).returncode == 0
    if found:
        expected = public_key(root=root, run=run)["public_key"]
        actual = node.call(["wg", "show", control.INTERFACE, "public-key"], run=run).stdout.strip()
        if expected != actual:
            raise ValueError("Existing control interface belongs to another installation")
    else:
        node.call(["wg-quick", "up", control.INTERFACE], run=run)
    for peer in config["peers"]:
        # Resolve the interface scope again after a driver reload changes ifindex.
        node.call(["wg", "set", control.INTERFACE, "peer", peer["key"], "endpoint", peer["endpoint"]], run=run)
    devices = [control.INTERFACE]
    if config["head"] and config["share_uplink"]:
        devices.append(control.netdev(config["uplink"]))
    for device in devices:
        node.call(["sysctl", "-w", f"net.ipv4.conf.{device}.forwarding=1"], run=run)
    for binary, rule in control.firewall(config):
        table = rule[:2] if rule[0] == "-t" else []
        body = rule[2:] if table else rule
        # The comment makes ownership visible without flushing another tool's rules.
        body = [*body[:-2], "-m", "comment", "--comment", "sparkring-control", *body[-2:]]
        found = node.call([binary, "-w", *table, "-C", *body], run=run, accepted=(0, 1)).returncode == 0
        if not found:
            node.call([binary, "-w", *table, "-I", *body], run=run)
    if config["share_uplink"]:
        if config["head"]:
            node.call(["systemctl", "enable", "sparkring-dns.service"], run=run)
            node.call(["systemctl", "start", "--no-block", "sparkring-dns.service"], run=run)
        else:
            node.call(["resolvectl", "dns", control.INTERFACE, config["head_address"]], run=run)
            node.call(["resolvectl", "domain", control.INTERFACE, "~."], run=run)
            node.call(["resolvectl", "default-route", control.INTERFACE, "yes"], run=run)
    return {"control_up": True}


def ssh_config(configs, keys, identity_file, *, home=None):
    """Exact control addresses use their authenticated host keys and root key."""
    directory = (Path(home) if home is not None else Path.home()) / ".ssh"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    known = directory / "sparkring_known_hosts"
    include = directory / "sparkring_config"
    rows, hosts = [], []
    for config in configs:
        address = str(ipaddress.IPv4Address(config["address"]))
        host_key = keys[config["id"]]["host_key"].split()
        if len(host_key) < 2 or host_key[0] != "ssh-ed25519" or not re.fullmatch(r"[A-Za-z0-9+/=]+", host_key[1]):
            raise ValueError("Authenticated peer did not provide an Ed25519 host key")
        hosts.append(f"[{address}]:{control.SSH_PORT} " + " ".join(host_key[:2]))
        rows += ["Host " + address, "  User root", f"  Port {control.SSH_PORT}",
                 "  IdentityFile " + json.dumps(str(identity_file)), "  IdentitiesOnly yes",
                 "  StrictHostKeyChecking yes", "  UserKnownHostsFile " + json.dumps(str(known))]
    for path, text in ((include, "\n".join(rows) + "\nHost *\n"), (known, "\n".join(hosts) + "\n")):
        if path.is_symlink() or path.exists() and path.read_text() != text:
            raise ValueError("Another controller SSH configuration exists: " + str(path))
        path.write_text(text)
        path.chmod(0o600)
    settings = directory / "config"
    if settings.is_symlink():
        raise ValueError("SSH config is a symlink; add the SparkRing include explicitly")
    text = settings.read_text() if settings.exists() else ""
    line = "Include " + json.dumps(str(include)) + "\n"
    if not text.startswith(line):
        if settings.exists():
            backup = directory / ("config.before-sparkring-" + str(os.getpid()))
            with backup.open("x") as stream:
                stream.write(text)
            backup.chmod(0o600)
        settings.write_text(line + text)
        settings.chmod(0o600)
