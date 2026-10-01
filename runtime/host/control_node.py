"""Privileged local implementation of the reviewed fabric administration tree."""
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

from runtime.host import bootstrap, control, node

# Where up() keeps each fallback-capable peer's path state between refreshes
# (control.choose). Under /run, so a boot starts on the primary paths.
PATH_STATE = "/run/sparkring-control/paths.json"
# Seconds a confirmation contact waits for the peer's SSH port to answer.
PROBE_TIMEOUT = 3


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


def _set_endpoint(peer, *, run, endpoint=None):
    # wg resolves the endpoint's interface name again, which updates the scope
    # (ifindex) that a driver restart of that function made stale.
    node.call(["wg", "set", control.INTERFACE, "peer", peer["key"], "endpoint", endpoint or peer["endpoint"]], run=run)


def probe(address, *, timeout=PROBE_TIMEOUT):
    """Open and close a TCP connection to a peer's administration SSH port over the tunnel.

    Any answer, an accepted connection or a refusal, crosses the tunnel back,
    so WireGuard counts received bytes when the peer's path works. The
    firewall rules of control.firewall accept this port from the
    administration subnet on every Spark. Returns whether the peer answered.
    """
    try:
        with socket.create_connection((str(ipaddress.IPv4Address(address)), control.SSH_PORT), timeout=timeout):
            return True
    except ConnectionRefusedError:
        return True
    except OSError:
        return False


def _observed(run):
    """{peer key: {"endpoint", "handshake", "received"}} from ``wg show sr-control dump``."""
    result = node.call(["wg", "show", control.INTERFACE, "dump"], run=run, accepted=(0, 1))
    rows = {}
    # The first line describes the interface itself, including its private key.
    for line in result.stdout.splitlines()[1:] if result.returncode == 0 else []:
        fields = line.split("\t")
        if len(fields) >= 6:
            rows[fields[0]] = {"endpoint": None if fields[2] == "(none)" else fields[2],
                               "handshake": int(fields[4]) if fields[4].isdigit() else 0,
                               "received": int(fields[5]) if fields[5].isdigit() else 0}
    return rows


def _usable(path, *, primary, root, run, failed):
    """Whether a path's local link can carry the tunnel now.

    Every path needs carrier on its interface. A cable path also needs its
    function to pass the link check of _link_problem (MAC and link-local
    address); for the primary path, the recorded link, underlay() has run
    that check and ``failed`` holds the result.
    """
    carrier, _ = node.link_state(path["netdev"], root=root)
    if carrier is not True:
        return False
    if primary:
        return path["netdev"] not in failed
    return path["via"] == "lan" or _link_problem(path, root=root, run=run) is None


def _journal(line):
    """Write one line to stderr, which systemd records in the refresh service's journal."""
    print(line, file=sys.stderr, flush=True)


def _read_states(root):
    try:
        value = json.loads(node.location(root, PATH_STATE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def select(config, failed, *, root="/", run=subprocess.run, now=time.time, contact=probe, say=None):
    """Set each peer's endpoint to the path it should use; return the recorded links that fallbacks carry.

    ``failed`` names the recorded links that fail their check (underlay).
    A peer without fallback paths gets its recorded endpoint set again when
    its link passes. A peer with fallback paths gets the path control.choose
    picks from its links, WireGuard's endpoint, latest handshake and received
    bytes; its endpoint is set only when it differs from that path's,
    interface name included, so WireGuard's own move to the source of the
    peer's packets stands. After a move this contacts the peer's control
    address (``contact``), whose answer the next refresh looks for.

    Returns the set of recorded links whose peer uses a usable fallback path.
    """
    say = say or _journal
    plain = [peer for peer in config["peers"] if not peer.get("alternates")]
    for peer in plain:
        if peer["netdev"] not in failed:
            _set_endpoint(peer, run=run)
    capable = [peer for peer in config["peers"] if peer.get("alternates")]
    if not capable:
        return set()
    observed = _observed(run)
    states = _read_states(root)
    carried = set()
    for peer in capable:
        options = control.paths(config, peer)
        usable = [_usable(path, primary=index == 0, root=root, run=run, failed=failed)
                  for index, path in enumerate(options)]
        seen = observed.get(peer["key"], {})
        current = control.match(options, seen.get("endpoint"))
        target, check, reason, state = control.choose(
            states.get(peer["key"]), usable, current, seen.get("handshake", 0), seen.get("received", 0), now(),
            has_endpoint=bool(seen.get("endpoint")))
        if target is not None and not control.same_endpoint(seen.get("endpoint"), options[target]):
            try:
                _set_endpoint(peer, run=run, endpoint=control.endpoint(options[target]))
            except RuntimeError as error:
                # An interface can disappear between its check and wg set, as
                # during a driver restart; the peer keeps its earlier state and
                # the next refresh chooses again.
                say(f"{control.INTERFACE}: peer {peer.get('address') or peer['id']}: {error}")
                continue
            if target != current:
                before = control.path_text(options[current]) if current is not None else "no known path"
                why = {"link": f"{before}: link down", "answer": f"no answer over {before}",
                       "handshake": f"no handshake over {before} for {control.HANDSHAKE_STALE} s",
                       "endpoint": "no known path in use",
                       "primary": f"{control.path_text(options[0])} passes its link check again"}.get(reason, before)
                say(f"{control.INTERFACE}: peer {peer.get('address') or peer['id']} now uses "
                    f"{control.path_text(options[target])} ({why})")
            if check and peer.get("address"):
                contact(peer["address"])
        states[peer["key"]] = state
        chosen = current if target is None else target
        if chosen is not None and chosen != 0 and usable[chosen]:
            carried.add(peer["netdev"])
    _write_private(PATH_STATE, json.dumps(states, indent=2, sort_keys=True) + "\n", root=root)
    return carried


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


def identities(root="/"):
    """The identities a control configuration may name for this machine.

    Discovery names a Spark by bootstrap.fabric_identity() of its RDMA node
    GUIDs; configurations that name the machine ID stay valid.
    """
    machine = node.location(root, "/etc/machine-id").read_text().strip()
    devices = node.location(root, "/sys/class/infiniband")
    guids = [path.read_text() for path in sorted(devices.glob("*/node_guid"))] if devices.is_dir() else []
    return {machine, bootstrap.fabric_identity(guids, machine)}


def _rule_command(binary, rule, action):
    """The iptables command that checks (-C), inserts (-I) or deletes (-D) one rule of control.firewall."""
    table = rule[:2] if rule[0] == "-t" else []
    body = rule[2:] if table else rule
    # The comment makes ownership visible without flushing another tool's rules.
    body = [*body[:-2], "-m", "comment", "--comment", "sparkring-control", *body[-2:]]
    return [binary, "-w", *table, action, *body]


def extend(config, *, root="/", run=subprocess.run):
    """Replace the fallback paths of the installed configuration; change nothing else.

    ``config`` must equal the installed /etc/sparkring/control.json except
    for the peer fields of control.EXTENSION. The firewall rules of fallback
    paths that ``config`` does not list are removed; the refresh service
    then adds the new ones and selects each peer's path.
    """
    installed = node.read(root, "/etc/sparkring/control.json")
    if control.base(installed) != control.base(config):
        raise ValueError("A different control network is installed; inspect before replacing it")
    control.validate(config)
    kept = control.firewall(config)
    node.save(root, "/etc/sparkring/control.json", config, mode=0o600)
    for binary, rule in control.firewall(installed):
        if (binary, rule) not in kept:
            node.call(_rule_command(binary, rule, "-D"), run=run, accepted=(0, 1))
    node.call(["systemctl", "start", "--no-block", "sparkring-control-refresh.service"], run=run)
    return {"configured": True, "address": str(ipaddress.IPv4Address(config["address"])),
            "fallback_paths": sum(len(peer.get("alternates") or []) for peer in config["peers"])}


def configure(document, *, root="/", run=subprocess.run):
    """Install this Spark's administration network configuration, or extend the installed one.

    A configuration that differs from the installed one only in its peers'
    fallback paths (control.EXTENSION) replaces those through extend(); any
    other difference is refused.
    """
    config = document["control"]
    if config.get("schema") != "sparkring-control/v1" or config["id"] not in identities(root):
        raise ValueError("Control configuration belongs to another machine")
    pubkey = document["ssh_key"]
    if not re.fullmatch(r"ssh-ed25519 [A-Za-z0-9+/=]+(?: [^\r\n]*)?", pubkey):
        raise ValueError("Use the controller's Ed25519 public key")
    control.validate(config)
    existing = node.location(root, "/etc/sparkring/control.json")
    if existing.exists() and node.read(root, "/etc/sparkring/control.json") != config:
        return extend(config, root=root, run=run)
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


def up(*, root="/", run=subprocess.run, now=time.time, contact=probe, say=None):
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
    - The forwarding sysctls and firewall rules are applied, including the
      rules of every fallback path, before any endpoint moves.
    - Each peer gets its endpoint from select(): a peer without fallback paths
      gets its recorded endpoint set again when its link passes, which updates
      the interface index that a driver restart of that function made stale;
      a link whose MAC differs from the record is never refreshed. A peer with
      fallback paths moves between its paths by control.choose.

    DNS settings follow. The call then raises naming each failed link whose
    peer no usable fallback path carries; a failed link that a fallback
    carries is only logged. The refresh timer runs this every 20 seconds, so
    each missing endpoint is set once its link passes.
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
    devices = [control.INTERFACE]
    if config["head"] and config["share_uplink"]:
        devices.append(control.netdev(config["uplink"]))
    for device in devices:
        node.call(["sysctl", "-w", f"net.ipv4.conf.{device}.forwarding=1"], run=run)
    for binary, rule in control.firewall(config):
        found = node.call(_rule_command(binary, rule, "-C"), run=run, accepted=(0, 1)).returncode == 0
        if not found:
            node.call(_rule_command(binary, rule, "-I"), run=run)
    carried = select(config, failed, root=root, run=run, now=now, contact=contact, say=say)
    if config["share_uplink"]:
        if config["head"]:
            node.call(["systemctl", "enable", "sparkring-dns.service"], run=run)
            node.call(["systemctl", "start", "--no-block", "sparkring-dns.service"], run=run)
        else:
            node.call(["resolvectl", "dns", control.INTERFACE, config["head_address"]], run=run)
            node.call(["resolvectl", "domain", control.INTERFACE, "~."], run=run)
            node.call(["resolvectl", "default-route", control.INTERFACE, "yes"], run=run)
    unreached = [problem for problem in problems if problem["netdev"] not in carried]
    for problem in problems:
        if problem["netdev"] in carried:
            (say or _journal)(f"{control.INTERFACE}: {problem['netdev']}: {problem['error']}; "
                              "a fallback path carries its peer")
    if unreached:
        raise _failure(unreached)
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
