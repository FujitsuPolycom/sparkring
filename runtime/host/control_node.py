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


def _link_problem(link, *, root, run):
    """Return why one recorded administration link cannot carry its peer, or None."""
    interface = link["netdev"]
    if not ipaddress.IPv6Address(link["address"]).is_link_local:
        return f"recorded endpoint {link['address']} is not IPv6 link-local"
    # sysfs itself uses symlinks: inspect the approved NIC by kernel name.
    try:
        mac = (Path(root) / "sys/class/net" / interface / "address").read_text().strip().lower()
    except FileNotFoundError:
        return "interface is not present"
    except OSError as error:
        return "cannot read its MAC: " + str(error)
    if mac != link["mac"].lower():
        return f"MAC {mac} differs from the recorded {link['mac'].lower()}; the underlay NIC identity changed"
    try:
        observed = json.loads(node.call(["ip", "-j", "-6", "address", "show", "dev", interface], run=run).stdout)
        addresses = [a["local"] for row in observed for a in row.get("addr_info", []) if a["scope"] == "link"]
    except (ValueError, KeyError, TypeError, AttributeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        return "cannot read its IPv6 addresses: " + str(error)
    if addresses != [link["address"]]:
        return (f"link-local addresses {addresses} differ from the recorded {link['address']}; "
                "retain its original NetworkManager connection identity")
    return None


def underlay(netdev=None, *, root="/", run=subprocess.run):
    """Check the recorded administration links and return one problem per failing link.

    Each problem is {"netdev", "error"}. An empty list means every checked link
    has its recorded MAC and exactly its recorded IPv6 link-local address, so
    WireGuard can resolve that peer's endpoint scope on it. With netdev, only
    that link is checked, and a netdev without a recorded link is a problem.
    Without /etc/sparkring/control.json there are no links to check.
    Callers that need every link raise when the list is not empty
    (require_underlay).
    """
    if netdev is not None:
        control.netdev(netdev)
    path = node.location(root, "/etc/sparkring/control.json")
    links = node.read(root, "/etc/sparkring/control.json")["links"] if path.exists() else []
    for link in links:
        control.netdev(link["netdev"])
    selected = [link for link in links if netdev is None or link["netdev"] == netdev]
    if netdev is not None and not selected:
        return [{"netdev": netdev, "error": "no administration network link is recorded on it"}]
    problems = []
    for link in selected:
        error = _link_problem(link, root=root, run=run)
        if error:
            problems.append({"netdev": link["netdev"], "error": error})
    return problems


def _failure(problems):
    return ValueError("Administration network link check failed: "
                      + "; ".join(p["netdev"] + ": " + p["error"] for p in problems))


def require_underlay(netdev=None, *, root="/", run=subprocess.run):
    """Raise ValueError naming every failing administration link."""
    problems = underlay(netdev, root=root, run=run)
    if problems:
        raise _failure(problems)


def _set_endpoint(peer, *, run):
    # wg resolves the endpoint's interface name again, which updates the scope
    # (ifindex) that a driver restart of that function made stale.
    node.call(["wg", "set", control.INTERFACE, "peer", peer["key"], "endpoint", peer["endpoint"]], run=run)


def refresh_endpoint(netdev, *, root="/", run=subprocess.run):
    """Re-set the endpoint of the one peer reached over netdev, after checking that link.

    Raises ValueError, and changes nothing, when the link's MAC or link-local
    address differs from the record or no single peer uses netdev. Returns the
    peer record from /etc/sparkring/control.json.
    """
    config = node.read(root, "/etc/sparkring/control.json")
    require_underlay(netdev, root=root, run=run)
    peers = [peer for peer in config["peers"] if peer["netdev"] == netdev]
    if len(peers) != 1:
        raise ValueError(f"Expected one administration peer on {netdev}, found {len(peers)}")
    _set_endpoint(peers[0], run=run)
    return dict(peers[0])


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


# wg-quick names the interface after its configuration file.
PARTIAL_CONFIG = "/run/sparkring-control/" + control.INTERFACE + ".conf"


def _write_private(name, text, *, root):
    """Write a root-only file (0600) below a root-only directory (0700); return its path.

    The file is created with its final mode, so the private key it carries is
    never readable by another user.
    """
    destination = node.location(root, name)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    temporary = node.location(root, name + ".writing")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    os.chmod(temporary, 0o600)
    os.replace(temporary, destination)
    return destination


def up(*, root="/", run=subprocess.run):
    """Bring sr-control up over every healthy administration link; raise naming the failed links.

    A link fails its check when its interface is missing, its MAC differs from
    the record, or it does not carry exactly its recorded link-local address.

    - When sr-control does not exist and every link passes, wg-quick creates it
      from /etc/wireguard/sr-control.conf.
    - When sr-control does not exist and some links fail, wg-quick creates it
      from a root-only copy in /run/sparkring-control/ without the Endpoint of
      each failed link's peer. wg-quick cannot resolve a link-local endpoint
      whose interface is missing, and WireGuard learns a peer's endpoint from
      that peer's first authenticated packet. A Spark whose child-facing
      function failed to restart at boot therefore stays reachable over its
      parent-facing link.
    - Every peer whose link passes gets its endpoint set again, which updates
      the interface index that a driver restart of that function made stale.
      A link whose MAC differs from the record is never refreshed.

    The forwarding sysctls, firewall rules and DNS settings are applied in each
    case. The call then raises when a link failed; the refresh timer runs it
    again every 20 seconds and sets each missing endpoint once its link passes.
    """
    config = node.read(root, "/etc/sparkring/control.json")
    problems = underlay(root=root, run=run)
    failed = {problem["netdev"] for problem in problems}
    found = node.call(["ip", "-j", "link", "show", "dev", control.INTERFACE], run=run, accepted=(0, 1)).returncode == 0
    if found:
        expected = public_key(root=root, run=run)["public_key"]
        actual = node.call(["wg", "show", control.INTERFACE, "public-key"], run=run).stdout.strip()
        if expected != actual:
            raise ValueError("Existing control interface belongs to another installation")
    elif problems:
        private = node.location(root, "/etc/sparkring/control.key").read_text().strip()
        path = _write_private(PARTIAL_CONFIG, control.render(config, private, without_endpoint=failed), root=root)
        node.call(["wg-quick", "up", str(path)], run=run)
    else:
        node.call(["wg-quick", "up", control.INTERFACE], run=run)
    for peer in config["peers"]:
        if peer["netdev"] not in failed:
            _set_endpoint(peer, run=run)
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
    if problems:
        raise _failure(problems)
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
