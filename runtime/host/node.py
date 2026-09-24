"""Local node identity, approved boot configuration, and read-only observations.

Only the node CLI runs privileged operations. No listening control socket or
sudoers rule is installed; setup uses the operator's existing SSH/sudo authority.
"""
import ipaddress
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

from runtime.common import distribution
from runtime.host.discovery import target
from scripts.deploy_inventory import _collect_local, _request, validate_inventory

ROOT = Path(__file__).resolve().parents[2]


def location(root, name):
    path = Path(root) / str(name).lstrip("/")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Node state path contains a symlink: " + str(path))
    return path


def read(root, name):
    return json.loads(location(root, name).read_text(encoding="utf-8"))


def save(root, name, value, mode=0o644):
    path = location(root, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    location(root, temporary.relative_to(Path(root)))
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")
    temporary.chmod(mode)
    temporary.replace(path)


def call(argv, *, run=subprocess.run, accepted=(0,)):
    result = run(argv, capture_output=True, text=True, timeout=90)
    if result.returncode not in accepted:
        raise RuntimeError(" ".join(argv[:4]) + ": " + result.stderr.strip())
    return result


def initialize(root="/"):
    path = location(root, "/etc/sparkring/node.json")
    if path.exists():
        document = read(root, "/etc/sparkring/node.json")
        uuid.UUID(document["node_id"])
    else:
        document = {"schema": "sparkring-node/v1", "node_id": str(uuid.uuid4())}
        save(root, "/etc/sparkring/node.json", document)
    # Avahi advertises candidates; SSH host keys establish their identity.
    service = location(root, "/etc/avahi/services/sparkring.service")
    service.parent.mkdir(parents=True, exist_ok=True)
    service.write_text('''<?xml version="1.0" standalone="no"?>
<!DOCTYPE service-group SYSTEM "avahi-service.dtd">
<service-group><name replace-wildcards="yes">SparkRing on %h</name>
<service><type>_sparkring._tcp</type><port>22</port>
<txt-record>protocol=ssh</txt-record></service></service-group>
''', encoding="utf-8")
    return document


def inspect(rank, ssh_target, management, witness, *, root="/", collect=_collect_local, run=subprocess.run):
    target(ssh_target)
    if management != ssh_target.split("@", 1)[1] or management == witness:
        raise ValueError("Use the host management IP and another Spark as its route witness")
    facts = collect(_request(rank, ssh_target, management, (), witness))
    validate_inventory(facts, require_ready=True)
    lldp = json.loads(call(["lldpctl", "-f", "json"], run=run).stdout)
    identity = read(root, "/etc/sparkring/node.json")
    record = distribution.installed(ROOT)
    return {"schema": "sparkring-node-inventory/v1", "node_id": identity["node_id"],
            "hostname": socket.gethostname(), "revision": record["revision"] if record else None,
            "facts": facts, "lldp": lldp}


def validate(config):
    if config.get("schema") != "sparkring-fabric-state/v1" or not re.fullmatch("[0-9a-f]{64}", config.get("cluster_id", "")):
        raise ValueError("Invalid approved fabric state")
    uuid.UUID(config["node_id"])
    if config["size"] not in (2, 4) or type(config["rank"]) is not int or config["rank"] not in range(config["size"]):
        raise ValueError("Invalid rank or size")
    target(config["ssh_target"])
    management = config["management"]
    for key in ("address", "witness"):
        address = ipaddress.IPv4Address(management[key])
        if address.is_multicast or address.is_unspecified or address.is_loopback:
            raise ValueError("Invalid management address")
    if management["address"] == management["witness"]:
        raise ValueError("Management witness must be another host")
    ports = config["interfaces"]
    count = 2 if config["size"] == 2 else 4
    if len(ports) != count or len({p["netdev"] for p in ports}) != count:
        raise ValueError("Invalid fabric interfaces")
    nets = {}
    for p in ports:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,15}", p["netdev"]) or p["netdev"] == management["interface"]:
            raise ValueError("Invalid data interface or management interface selected")
        if not re.fullmatch(r"(?:[0-9a-f]{2}:){5}[0-9a-f]{2}", p["mac"]):
            raise ValueError("Expected data interface MAC")
        endpoint = ipaddress.IPv4Interface(p["address"])
        if endpoint.network.prefixlen != 24 or any(ipaddress.ip_address(management[k]) in endpoint.network for k in ("address", "witness")):
            raise ValueError("Fabric must use /24 subnets separate from management")
        nets[p["netdev"]] = endpoint.network
    if len(set(nets.values())) != count:
        raise ValueError("Fabric functions require separate subnets")
    for route in config["routes"]:
        network = ipaddress.IPv4Network(route["destination"])
        if (network.prefixlen != 24 or route["dev"] not in nets
                or ipaddress.IPv4Address(route["via"]) not in nets[route["dev"]]
                or any(ipaddress.ip_address(management[k]) in network for k in ("address", "witness"))
                or any(network.overlaps(n) for n in nets.values())):
            raise ValueError("Route must reach a remote fabric /24 through a verified data interface")
    if any(len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= nets.keys() for pair in config["forwarding"]):
        raise ValueError("Forwarding must stay between distinct data interfaces")
    if config["size"] == 2 and (config["routes"] or config["forwarding"]):
        raise ValueError("Pair requires no routed fabric")
    return config


def observe(config, *, collect=_collect_local):
    validate(config)
    m = config["management"]
    facts = collect(_request(config["rank"], config["ssh_target"], m["address"], (), m["witness"]))
    management = facts["management"]
    if (management.get("error") or management.get("interface") != m["interface"]
            or (management.get("route_to_controller") or {}).get("dev") != m["interface"]):
        raise ValueError("Independent management path changed")
    interfaces = {i["name"]: i for i in facts["interfaces"]}
    functions = {i["device"]: i for i in facts["rdma"]}
    for port in config["interfaces"]:
        nic, rdma = interfaces.get(port["netdev"], {}), functions.get(port["rdma_device"], {})
        if (nic.get("mac", "").lower() != port["mac"] or nic.get("master")
                or nic.get("ipv4") != [port["address"]] or nic.get("mtu") != 9000
                or nic.get("operstate") != "UP" or rdma.get("netdev") != port["netdev"]
                or rdma.get("driver") != "mlx5_core"):
            raise ValueError("Approved data interface changed or is unavailable: " + port["netdev"])
        mapped = ipaddress.IPv6Address(rdma.get("gid") or "::").ipv4_mapped
        if (mapped != ipaddress.ip_interface(port["address"]).ip or rdma.get("gid_index") != 3
                or rdma.get("gid_type") != "RoCE v2" or rdma.get("gid_netdev") != port["netdev"]
                or rdma.get("active_mtu") != 4096 or rdma.get("state") not in ("ACTIVE", "PORT_ACTIVE")):
            raise ValueError("GID3/RoCE link is unavailable: " + port["netdev"])
    return facts


def restore(config, *, collect=_collect_local, run=subprocess.run):
    """Restore only missing routes/rules; refuse conflicting routes before mutation."""
    if config.get("ownership") == "observed":
        raise ValueError("This fabric belongs to its existing service; no restoration changes are authorized")
    facts = observe(config, collect=collect)
    missing = []
    for desired in config["routes"]:
        entries = [r for r in facts["routes"] if r.get("dst") == desired["destination"]]
        if entries and (len(entries) != 1 or entries[0].get("dev") != desired["dev"] or entries[0].get("gateway") != desired["via"]):
            raise ValueError("Conflicting existing fabric route: " + desired["destination"])
        if not entries:
            missing.append(["ip", "route", "add", desired["destination"], "via", desired["via"], "dev", desired["dev"], "proto", "static"])
    for argv in missing:
        call(argv, run=run)
    for interface in sorted({i for pair in config["forwarding"] for i in pair}):
        # Per-interface forwarding avoids enabling routing on management NICs.
        call(["sysctl", "-w", f"net.ipv4.conf.{interface}.forwarding=1"], run=run)
        call(["sysctl", "-w", f"net.ipv4.conf.{interface}.rp_filter=0"], run=run)
    for incoming, outgoing in config["forwarding"]:
        rule = ["FORWARD", "-i", incoming, "-o", outgoing, "-m", "comment", "--comment", "sparkring:" + config["cluster_id"][:16], "-j", "ACCEPT"]
        exists = call(["iptables", "-w", "-C", *rule], run=run, accepted=(0, 1)).returncode == 0
        if not exists:
            call(["iptables", "-w", "-I", *rule], run=run)
    return {"configured": True, "hardware_qualified": False}


def verify_persistence(config, facts, *, run=subprocess.run):
    for expected in config["routes"]:
        if not any(r.get("dst") == expected["destination"] and r.get("dev") == expected["dev"]
                   and r.get("gateway") == expected["via"] for r in facts["routes"]):
            raise ValueError("Approved fabric route is missing: " + expected["destination"])
    for incoming, outgoing in config["forwarding"]:
        rule = ["FORWARD", "-i", incoming, "-o", outgoing, "-m", "comment", "--comment", "sparkring:" + config["cluster_id"][:16], "-j", "ACCEPT"]
        call(["iptables", "-w", "-C", *rule], run=run)
    for interface in {i for pair in config["forwarding"] for i in pair}:
        if call(["sysctl", "-n", f"net.ipv4.conf.{interface}.forwarding"], run=run).stdout.strip() != "1":
            raise ValueError("Fabric forwarding is disabled: " + interface)


def configure(config, *, root="/", collect=_collect_local, run=subprocess.run):
    validate(config)
    if config["node_id"] != read(root, "/etc/sparkring/node.json")["node_id"]:
        raise ValueError("Configuration belongs to another node")
    path = location(root, "/etc/sparkring/fabric.json")
    if path.exists() and read(root, "/etc/sparkring/fabric.json") != config:
        raise ValueError("Node already has another approved configuration; inspect before replacing it")
    observe(config, collect=collect)
    save(root, "/etc/sparkring/fabric.json", config)
    call(["systemctl", "enable", "sparkring-fabric.service"], run=run)
    # A failed start stays visible; repeating the same configuration is idempotent.
    call(["systemctl", "restart", "sparkring-fabric.service"], run=run)
    return {"persisted": True, "cluster_id": config["cluster_id"]}


def adopt(config, *, root="/", collect=_collect_local):
    validate(config)
    if config["node_id"] != read(root, "/etc/sparkring/node.json")["node_id"]:
        raise ValueError("Observed configuration belongs to another node")
    if config.get("ownership") != "observed" or config["routes"] or config["forwarding"]:
        raise ValueError("Adoption must not request network changes")
    observe(config, collect=collect)
    path = location(root, "/etc/sparkring/fabric.json")
    if path.exists() and read(root, "/etc/sparkring/fabric.json") != config:
        raise ValueError("Node records another setup; inspect before replacing it")
    save(root, "/etc/sparkring/fabric.json", config)
    return {"adopted": True, "network_changed": False}


def workspace(operator, name, *, root="/"):
    import pwd
    account = pwd.getpwnam(operator)
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,39}", name):
        raise ValueError("Choose a lowercase cluster name")
    # Source stagers claim empty per-model workspaces below this parent.
    path = location(root, "/srv/sparkring/" + name)
    controller = location(root, "/var/lib/sparkring/controller")
    for directory in (path, controller):
        if directory.exists():
            if directory.stat().st_uid != account.pw_uid:
                raise ValueError("State/workspace belongs to another operator: " + str(directory))
        else:
            directory.mkdir(parents=True, mode=0o700)
            os.chown(directory, account.pw_uid, account.pw_gid)
    return {"workspace": str(path), "controller": str(controller)}


def snapshot(*, root="/", collect=_collect_local, run=subprocess.run, now=time.time):
    result = {"schema": "sparkring-node-status/v1", "observed_at": now(), "hostname": socket.gethostname(),
              "hardware_qualified": False, "state": "not-configured", "next_action": "sparkring setup"}
    if not location(root, "/etc/sparkring/fabric.json").exists():
        return result
    try:
        config = read(root, "/etc/sparkring/fabric.json")
        result.update(rank=config["rank"], size=config["size"], cluster_id=config["cluster_id"])
        facts = observe(config, collect=collect)
        verify_persistence(config, facts, run=run)
        if config.get("ownership") == "observed" and config.get("native_mesh"):
            from runtime.common import qwen_mesh
            mesh = config["native_mesh"]
            qwen_mesh.check(mesh["reference"], config["rank"], mesh["hcas"], 3, mesh["host_ip"])
        elif config.get("ownership") != "observed" and call(["systemctl", "is-active", "sparkring-fabric.service"], run=run, accepted=(0, 3)).returncode:
            raise ValueError("Fabric service is not active; inspect journalctl -u sparkring-fabric")
        result.update(state="network-configured", next_action="sparkring models",
                      containers=[{"name": c.get("name"), "state": c.get("state")} for c in (facts["docker"].get("containers") or [])],
                      model_ready=None)
        if config.get("ownership") == "observed":
            result["state"] = "existing-network-verified"
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        result.update(state="needs-attention", next_action="sparkring status --refresh", error=str(error))
    return result


def status(*, root="/", now=time.time):
    try:
        result = read(root, "/run/sparkring/status.json")
    except FileNotFoundError:
        return {"state": "agent-unavailable", "next_action": "systemctl status sparkring-agent", "hardware_qualified": False}
    age = now() - result["observed_at"]
    result["age_seconds"] = round(age, 1)
    if age < -5 or age > 90:
        result.update(state="stale", next_action="systemctl status sparkring-agent")
    return result


def agent(*, root="/", once=False):
    while True:
        save(root, "/run/sparkring/status.json", snapshot(root=root))
        if once:
            return
        time.sleep(30)
