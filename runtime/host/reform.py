"""Re-form Sparks that belonged to other SparkRing clusters into the pair or ring now cabled.

A Spark keeps the setup of the cluster it belonged to: Node A's controller
record (``/var/lib/sparkring/controller``), the admin network (``sr-control``:
``/etc/sparkring/control.json`` and its services), the fabric record
(``/etc/sparkring/fabric.json``) and its boot service, automatic recovery,
mesh services and the ConnectX hairpin approval. Setup refuses a second
cluster on top of that state: a node keeps the admin network configuration it
has and refuses another one.

When the Sparks cabled to Node A are not the Sparks its record names, or a
cabled Spark holds such state, setup re-forms them:

1. ``sparkring setup`` surveys the cabled Sparks (``survey``) and reads each
   one's state (``prior_state``), then lists by Spark what it will move aside
   (``plan``, ``plan_lines``). ``--plan`` stops there.
2. While a SparkRing model container runs on one of them, setup stops and
   prints the command that stops it; it never stops a model itself.
3. After the one setup approval, each worker runs ``retire`` as root, then
   installs Node A's SparkRing package and prepares its fabric ports for
   discovery (the worker preparation of ``sparkring setup --worker-bundle``);
   Node A runs ``retire`` last. ``retire`` moves the state into
   ``/var/lib/sparkring/retired/STAMP/`` (kept, never deleted), disables the
   old cluster's services and writes a receipt with restore steps.
4. Setup continues as a fresh setup of the cabled Sparks and renumbers their
   fabric addresses (the ``--reset-links`` behavior). Fabric IPv4 addresses
   that SparkRing did not set are listed in step 1; setup backs up each
   Spark's NetworkManager connections before it changes them.

Node A keeps its SSH identity (``controller_ed25519``), its fabric SSH known
hosts and the installation lock. Nothing under ``/srv/sparkring``
(checkpoints, images, caches) is touched. Every step skips what an earlier,
interrupted run already did, so setup can run again.
"""
import inspect
import json
import re
import subprocess
import time

from runtime.host import cabling, survey

# Where each Spark keeps the state a re-form moved aside, one directory per re-form.
RETIRED = "/var/lib/sparkring/retired"
# Controller entries that Node A keeps: its SSH identity, the known hosts of
# its fabric SSH sessions (``bootstrap.SSH``) and the lock this setup holds.
NODE_A_KEEPS = ("install.lock", "controller_ed25519", "controller_ed25519.pub", "ssh")
STAMP = re.compile(r"[0-9]{8}T[0-9]{6}Z(?:-[a-z][a-z0-9-]{0,34})?")


def prior_state():
    """This Spark's SparkRing cluster state; reads only and runs on a Spark beside ``survey.observe``.

    Root reads everything; another account reads what file modes allow, and
    ``unreadable`` lists what it could not read. Absent items are None.
    """
    import glob
    import json
    import os
    import subprocess
    from pathlib import Path

    unreadable = []

    def load(path):
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except PermissionError:
            unreadable.append(str(path))
            return {}
        except (OSError, ValueError):
            return {}

    def output(argv):
        try:
            done = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    def unit(name):
        return {"enabled": output(["systemctl", "is-enabled", name]) or "disabled",
                "active": output(["systemctl", "is-active", name]) or "inactive"}

    result = {"root": os.geteuid() == 0}
    controller = Path("/var/lib/sparkring/controller")
    result["controller"] = None
    if controller.is_dir():
        if not os.access(controller, os.R_OK | os.X_OK):
            unreadable.append(str(controller))
            result["controller"] = {"unreadable": True}
        else:
            entries = sorted(p.name for p in controller.iterdir())
            cluster = load(controller / "cluster.json") or {}
            plan = cluster.get("plan") or {}
            active = load(controller / "active.json") or {}
            lock = load(Path(active["path"]) / "deployment.lock.json") if active.get("path") else None
            # Node A's SSH key, known hosts, lock and setup receipts alone are no cluster: every setup creates them.
            if set(entries) & {"cluster.json", "enrolled.json", "active.json", "transaction.json", "deployments"}:
                result["controller"] = {
                    "entries": entries, "name": cluster.get("name"), "plan_id": plan.get("id"),
                    "nodes": [n.get("hostname") for n in plan.get("nodes") or [] if isinstance(n, dict)],
                    "enrolled": (controller / "enrolled.json").exists(), "active_deployment": (lock or {}).get("id")}
    control = load("/etc/sparkring/control.json")
    result["control"] = None if control is None else (
        {"unreadable": True} if not control else
        {"address": control.get("address"), "head": control.get("head"), "subnet": control.get("subnet")})
    result["control_interface"] = Path("/sys/class/net/sr-control").exists()
    fabric = load("/etc/sparkring/fabric.json")
    result["fabric"] = None if fabric is None else {
        "cluster_id": fabric.get("cluster_id"), "rank": fabric.get("rank"), "size": fabric.get("size"),
        "interfaces": [{"netdev": p.get("netdev"), "address": p.get("address")} for p in fabric.get("interfaces") or []],
        "routes": len(fabric.get("routes") or []), "forwarding": len(fabric.get("forwarding") or [])}
    result["files"] = {path: os.path.lexists(path) for path in (
        "/etc/sparkring/hairpin.json", "/etc/sparkring/seed_keys", "/etc/wireguard/sr-control.conf",
        "/var/lib/sparkring/recovery.json")}
    home = Path("/root/.ssh")
    result["ssh_config"] = (home / "sparkring_config").exists() if os.access(home, os.X_OK) else None
    result["units"] = {name: unit(name) for name in (
        "sparkring-recover.timer", "sparkring-recover.service", "sparkring-control.service",
        "sparkring-access.service", "sparkring-control-refresh.timer", "sparkring-dns.service",
        "sparkring-fabric.service", "sparkring-hairpin.service", "sparkring-seed.service")}
    meshes = sorted({Path(p).name for pattern in ("sparkring-mesh.service", "sparkring-*-mesh.service")
                     for p in glob.glob("/etc/systemd/system/" + pattern)})
    result["mesh"] = [dict(unit(name), unit=name) for name in meshes]
    listing = output(["docker", "ps", "--no-trunc", "--format",
                      '{{.Names}}\t{{.Label "io.sparkring.deployment"}}\t{{.Label "org.sparkring.profile"}}'])
    result["containers"] = None if listing is None else [
        {"name": fields[0], "deployment": fields[1] or None, "profile": fields[2] or None}
        for fields in (line.split("\t") + ["", ""] for line in listing.splitlines() if line.strip())
        if fields[1] or fields[2]]
    addresses = {}
    for path in sorted(glob.glob("/sys/class/infiniband/*/device/net/*")):
        netdev = Path(path).name
        try:
            rows = json.loads(output(["ip", "-j", "-4", "address", "show", "dev", netdev]) or "[]")
        except ValueError:
            rows = []
        addresses[netdev] = [f"{a['local']}/{a['prefixlen']}" for row in rows for a in row.get("addr_info", [])
                             if a.get("family") == "inet"]
    result["fabric_ipv4"] = addresses
    result["unreadable"] = unreadable
    return result


def retire(order, call=None, root="/"):
    """Move this Spark's SparkRing cluster state aside and disable that cluster's services; runs as root.

    ``order`` is ``{"stamp", "node_a": bool, "keep": [paths relative to the
    controller directory that stay on Node A]}``; on another Spark the whole
    controller directory moves. Moved paths keep their modes below
    ``/var/lib/sparkring/retired/STAMP/`` with their original path, and
    ``receipt.json`` there names every change and how to restore it. Each
    step skips what is already gone, so a repeated run is safe. Raises
    RuntimeError, before any change, while a SparkRing model container or an
    automatic recovery attempt runs. Self-contained, so it runs on a Spark
    through ``python3 -I -c``; ``call`` and ``root`` replace the command
    runner and the file system root in tests.
    """
    import json
    import os
    import posixpath
    import re
    import shlex
    import shutil
    import subprocess
    from pathlib import Path

    if not re.fullmatch(r"[0-9]{8}T[0-9]{6}Z(?:-[a-z][a-z0-9-]{0,34})?", str(order.get("stamp"))):
        raise ValueError("Invalid retirement stamp")

    def run(argv, accepted=(0,)):
        if call is not None:
            return call(argv)
        done = subprocess.run(argv, capture_output=True, text=True, timeout=120)
        if done.returncode not in accepted:
            raise RuntimeError(" ".join(argv[:4]) + ": " + (done.stderr or done.stdout).strip())
        return done.stdout

    def quiet(argv):
        try:
            return run(argv, accepted=range(256))
        except (OSError, subprocess.SubprocessError):
            return ""

    base = Path(root)
    retired = base / "var/lib/sparkring/retired" / order["stamp"]
    receipt_path = retired / "receipt.json"
    receipt = {"schema": "sparkring-retired/v1", "stamp": order["stamp"], "moved": [], "disabled": [],
               "stopped": [], "removed": []}
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    def state(unit):
        return quiet(["systemctl", "is-enabled", unit]).strip(), quiet(["systemctl", "is-active", unit]).strip()

    def save():
        for directory in (retired.parent, retired):
            directory.mkdir(mode=0o755, parents=True, exist_ok=True)
        receipt["restore"] = (
            ["Restore only after stopping the cluster that replaced this one here (sudo systemctl disable --now "
             "sparkring-control.service sparkring-access.service sparkring-fabric.service), then:"]
            + [f"sudo mkdir -p {shlex.quote(posixpath.dirname(row['from']))} && sudo mv -T {shlex.quote(row['to'])} "
               f"{shlex.quote(row['from'])}" for row in receipt["moved"]]
            + [f"sudo systemctl enable {unit}" for unit in receipt["disabled"]]
            + ["Then reboot this Spark, which brings back its admin network, routes and forwarding rules."])
        temporary = receipt_path.with_name("receipt.json.writing")
        temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        os.chmod(temporary, 0o644)
        os.replace(temporary, receipt_path)

    # A recovery attempt could start a model while this runs: stop new attempts, then refuse a running one.
    enabled, _ = state("sparkring-recover.timer")
    if enabled == "enabled":
        run(["systemctl", "disable", "--now", "sparkring-recover.timer"])
        receipt["disabled"].append("sparkring-recover.timer")
        save()
    if state("sparkring-recover.service")[1] in ("active", "activating"):
        raise RuntimeError("An automatic recovery attempt is running on this Spark; repeat setup in a few minutes")
    listing = quiet(["docker", "ps", "--format",
                     '{{.Names}}\t{{.Label "io.sparkring.deployment"}}{{.Label "org.sparkring.profile"}}'])
    models = [name for name, _, labels in (line.partition("\t") for line in listing.splitlines()) if labels.strip()]
    if models:
        raise RuntimeError("SparkRing model containers run on this Spark: " + ", ".join(models))

    meshes = sorted({p.name for pattern in ("sparkring-mesh.service", "sparkring-*-mesh.service")
                     for p in (base / "etc/systemd/system").glob(pattern)})
    # The refresh timer goes first: its service starts the admin network and fabric services again.
    for unit in [*meshes, "sparkring-hairpin.service", "sparkring-control-refresh.timer",
                 "sparkring-control-refresh.service", "sparkring-access.service", "sparkring-dns.service",
                 "sparkring-control.service", "sparkring-seed.service", "sparkring-fabric.service"]:
        enabled, active = state(unit)
        if enabled == "enabled" or active in ("active", "activating"):
            run(["systemctl", "disable", "--now", unit])
            if enabled == "enabled":
                receipt["disabled"].append(unit)
            if active in ("active", "activating"):
                receipt["stopped"].append(unit)
            save()

    if (base / "sys/class/net/sr-control").exists():
        # wg-quick also removes the routing rules it added for a shared uplink;
        # an interface it did not create is deleted directly.
        if (base / "etc/wireguard/sr-control.conf").exists():
            run(["wg-quick", "down", "sr-control"], accepted=range(256))
        if (base / "sys/class/net/sr-control").exists():
            run(["ip", "link", "delete", "sr-control"], accepted=(0, 1))
        receipt["removed"].append("admin network interface sr-control")
    fabric_path = base / "etc/sparkring/fabric.json"
    if fabric_path.exists():
        fabric = json.loads(fabric_path.read_text(encoding="utf-8"))
        # Stale routes to the fabric /24s would conflict with the renumbered ones.
        for route in fabric.get("routes") or []:
            argv = ["ip", "route", "del", route["destination"], "via", route["via"], "dev", route["dev"]]
            run(argv, accepted=(0, 2))
            receipt["removed"].append("route " + " ".join(argv[3:]))
    # Firewall rules carry the owner in their comment: sparkring-control (admin network) or sparkring:ID (fabric).
    for binary, table in (("iptables", "filter"), ("iptables", "nat"), ("ip6tables", "filter")):
        for line in quiet([binary, "-w", "-t", table, "-S"]).splitlines():
            words = shlex.split(line)
            comment = words[words.index("--comment") + 1] if "--comment" in words[:-1] else ""
            if words[:1] == ["-A"] and (comment == "sparkring-control" or comment.startswith("sparkring:")):
                run([binary, "-w", "-t", table, "-D", *words[1:]], accepted=(0, 1))
                receipt["removed"].append(f"{binary} {table} rule: {line}")
    save()

    def move(path):
        source = base / path.lstrip("/")
        if not os.path.lexists(source):
            return
        target = retired / path.lstrip("/")
        target.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        shutil.move(str(source), str(target))
        receipt["moved"].append({"from": path, "to": "/var/lib/sparkring/retired/" + order["stamp"] + path})
        save()

    for path in ("/etc/sparkring/control.json", "/etc/wireguard/sr-control.conf", "/etc/sparkring/controller_keys",
                 "/etc/sparkring/sshd_config", "/etc/sparkring/dnsmasq.conf", "/etc/sparkring/seed_keys",
                 "/etc/sparkring/seed_sshd_config", "/etc/sparkring/fabric.json", "/etc/sparkring/hairpin.json",
                 "/var/lib/sparkring/hairpin-attempts.json", "/var/lib/sparkring/recovery.json",
                 "/root/.ssh/sparkring_config", "/root/.ssh/sparkring_known_hosts"):
        move(path)
    controller = base / "var/lib/sparkring/controller"
    keep = {Path(path) for path in order.get("keep") or ()}

    def sweep(relative):
        # Move every entry except the kept paths; a directory holding a kept path is swept inside.
        for entry in sorted((controller / relative).iterdir()):
            path = relative / entry.name
            if path in keep:
                continue
            if entry.is_dir() and not entry.is_symlink() and any(path in kept.parents for kept in keep):
                sweep(path)
            else:
                move("/var/lib/sparkring/controller/" + path.as_posix())

    if controller.is_dir():
        if order.get("node_a"):
            sweep(Path("."))
        elif any(controller.iterdir()):
            move("/var/lib/sparkring/controller")
    save()
    return receipt


def stamp(name, *, now=time.gmtime):
    """The retirement stamp of one re-form: UTC time and the cluster name, as ``20261002T143000Z-sparkring``."""
    value = time.strftime("%Y%m%dT%H%M%SZ", now()) + "-" + name
    if not STAMP.fullmatch(value):
        raise ValueError("Choose a lowercase cluster name of at most 35 characters")
    return value


def local_lldp(*, run=subprocess.run):
    """This Spark's LLDP rows on its fabric functions; empty when lldpctl cannot be read."""
    try:
        done = run(["lldpctl", "-f", "json"], capture_output=True, text=True, timeout=30)
        document = json.loads(done.stdout) if done.returncode == 0 else {}
    except (OSError, ValueError, subprocess.SubprocessError):
        document = {}
    return [row for row in cabling.lldp_rows(document) if cabling.FABRIC_NETDEV.fullmatch(row["netdev"])]


def record_mismatch(record, rows, hostname):
    """Why the Sparks cabled to Node A are not its recorded Sparks, or None when they may be.

    ``record`` is Node A's ``cluster.json``, or ``enrolled.json`` of a setup
    that stopped after provisioning, which names its Sparks only by count;
    ``rows`` are Node A's fabric LLDP rows. A neighbor the record does not
    name (or more neighbors than an enrolled record counts) means the cabling
    changed; so does a four-Spark record whose Node A now has both ports on
    one Spark. Anything else keeps the record, and setup's cabling check
    names any cable to move.
    """
    plan = record.get("plan") or {}
    recorded = {str(n.get("hostname")) for n in plan.get("nodes") or [] if isinstance(n, dict)}
    size = len(plan.get("nodes") or record.get("targets") or [])
    by_port = {}
    for row in rows:
        name = str(row["hostname"] or "").rstrip(".")
        if name and name != hostname:
            by_port.setdefault(cabling.port_of(row["netdev"]), set()).add(name)
    neighbors = set().union(*by_port.values()) if by_port else set()
    label = f"cluster \"{record['name']}\"" if record.get("name") else "enrolled Sparks"
    extra = sorted(neighbors - recorded) if recorded else []
    if extra:
        return (" and ".join(extra) + (" is" if len(extra) == 1 else " are") + f" cabled to this Spark but not part of "
                f"its {label} ({size} Sparks)")
    if not recorded and len(neighbors) + 1 > size:
        return f"{len(neighbors) + 1} Sparks are cabled here, but setup enrolled {size}"
    if size == 4 and len(neighbors) == 1 and len(by_port) == 2:
        return f"both ports of this Spark lead to {next(iter(neighbors))}, so the four Sparks of its {label} are no longer a ring"
    return None


def _name(spark):
    return spark["data"]["inventory"].get("hostname") or spark["data"]["inventory"]["id"]


def items(spark, *, node_a=False):
    """Plain lines naming one Spark's SparkRing state that re-forming moves aside or turns off."""
    state = spark["data"].get("state") or {}
    lines = []
    controller = state.get("controller")
    if controller and controller.get("unreadable"):
        lines.append("controller records (/var/lib/sparkring/controller), readable only with sudo: moved aside")
    elif controller:
        if controller.get("name"):
            size = len(controller.get("nodes") or [])
            lines.append(f"Node A of cluster \"{controller['name']}\" ({size} Sparks): its records move aside"
                         + (", except Node A's SSH key" if node_a else ""))
        else:
            lines.append("controller records (/var/lib/sparkring/controller): moved aside"
                         + (", except Node A's SSH key" if node_a else ""))
    units = state.get("units") or {}
    if units.get("sparkring-recover.timer", {}).get("enabled") == "enabled":
        lines.append("automatic recovery: turned off")
    control = state.get("control")
    if control:
        where = f" {control['address']}" if control.get("address") else ""
        lines.append(f"admin network{where} of its cluster: stopped, turned off and its configuration moved aside")
    elif state.get("control_interface"):
        lines.append("admin network interface sr-control: removed")
    fabric = state.get("fabric")
    if fabric:
        size = fabric.get("size")
        lines.append(f"fabric record (rank {fabric.get('rank')} of {size} Sparks): moved aside; its boot service is "
                     "turned off" + (" and its routes removed" if fabric.get("routes") else ""))
    for mesh in state.get("mesh") or []:
        if mesh.get("enabled") == "enabled" or mesh.get("active") == "active":
            lines.append(f"mesh service {mesh['unit']}: stopped and turned off")
    if (state.get("files") or {}).get("/etc/sparkring/hairpin.json"):
        lines.append("ConnectX hairpin approval: moved aside (a ring records it again)")
    for netdev, address in foreign_addresses(state):
        lines.append(f"fabric address {address} on {netdev}, not set by SparkRing: replaced after a backup of its "
                     "NetworkManager connection")
    if state.get("unreadable") and not state.get("root"):
        lines.append("more SparkRing state may be present that only root can read; setup moves it aside as well")
    return lines


def foreign_addresses(state):
    """(netdev, address) of fabric IPv4 addresses that this Spark's SparkRing fabric record does not list."""
    recorded = {(p.get("netdev"), p.get("address")) for p in (state.get("fabric") or {}).get("interfaces") or []}
    def order(item):
        return cabling.port_of(item[0]) or 0, "P2p" in item[0], item[0]
    return [(netdev, address) for netdev, addresses in sorted((state.get("fabric_ipv4") or {}).items(), key=order)
            for address in addresses if (netdev, address) not in recorded]


def blockers(found):
    """``[(hostname, container, command)]`` of a survey: running SparkRing model containers and how to stop each.

    ``sudo sparkring down --execute`` on the Node A whose active deployment
    the container belongs to stops it the way SparkRing does, and automatic
    recovery never restarts what it stopped; elsewhere ``docker stop``.
    """
    owners = {}
    for spark in found["sparks"].values():
        controller = (spark["data"].get("state") or {}).get("controller") or {}
        if controller.get("active_deployment"):
            owners[controller["active_deployment"]] = _name(spark)
    rows = []
    for spark in found["sparks"].values():
        for container in (spark["data"].get("state") or {}).get("containers") or []:
            owner = owners.get(container.get("deployment"))
            command = (f"on {owner}: sudo sparkring down --execute" if owner
                       else f"on {_name(spark)}: sudo docker stop {container['name']}")
            rows.append((_name(spark), container["name"], command))
    return rows


def needed(found):
    """Whether any surveyed Spark holds SparkRing cluster state that a fresh setup would refuse."""
    for spark in found["sparks"].values():
        state = spark["data"].get("state") or {}
        units = state.get("units") or {}
        if (state.get("controller") or state.get("control") or state.get("fabric") or state.get("control_interface")
                or units.get("sparkring-recover.timer", {}).get("enabled") == "enabled"
                or any(m.get("enabled") == "enabled" or m.get("active") == "active" for m in state.get("mesh") or [])):
            return True
    return False


def plan(found, diagnosis, *, name, reason=None, now=time.gmtime):
    """The re-form plan: the Sparks in ring order, each with its items, blockers and the stamp."""
    by_name = {_name(spark): (key, spark) for key, spark in found["sparks"].items()}
    order = [name_ for name_ in diagnosis["order_names"] or [] if name_ in by_name]
    sparks = []
    for index, hostname in enumerate(order):
        key, spark = by_name[hostname]
        sparks.append({"key": key, "name": hostname, "node_a": index == 0,
                       "items": items(spark, node_a=index == 0),
                       "foreign_addresses": [{"netdev": n, "address": a}
                                             for n, a in foreign_addresses(spark["data"].get("state") or {})]})
    return {"schema": "sparkring-reform-plan/v1", "stamp": stamp(name, now=now), "cluster": name,
            "layout": diagnosis["layout"], "order": order, "reason": reason, "sparks": sparks,
            "blockers": [{"spark": s, "container": c, "command": cmd} for s, c, cmd in blockers(found)]}


def plan_lines(value):
    """Terminal lines of a re-form plan."""
    lines = []
    if value.get("reason"):
        lines.append("The cabled Sparks differ from this Spark's cluster record: " + value["reason"] + ".")
    lines.append("Re-form: setup moves aside what these Sparks keep from earlier SparkRing clusters:")
    for spark in value["sparks"]:
        lines.append(f"  {spark['name']}" + (" (Node A)" if spark["node_a"] else "") + ":")
        lines += [f"    - {item}" for item in spark["items"]] or ["    - nothing to move"]
    lines.append(f"Moved state stays in /var/lib/sparkring/retired/{value['stamp']}/ on each Spark, with a receipt "
                 "that lists how to restore it.")
    lines.append("Checkpoints, images and caches in /srv/sparkring stay where they are.")
    shape = "pair" if value["layout"] == "pair" else "ring"
    lines.append(f"Then setup sets up the {shape} " + " → ".join(value["order"]) + " as new, with new fabric addresses.")
    if value["blockers"]:
        lines.append("Setup stops until these SparkRing model containers are stopped:")
        lines += [f"  {row['spark']}: {row['container']}; stop it {row['command']}" for row in value["blockers"]]
    return lines


def worker_script(order, install):
    """Root program for one worker: ``retire`` with ``order``, then the worker bundle's installer and preparation."""
    return (inspect.getsource(retire) + "\nimport json\nimport subprocess\n"
            + f"retire(json.loads({json.dumps(order)!r}))\n"
            + f"subprocess.run(['python3', '-I', {install!r}, '--apply', '--prepare', '--yes'], check=True)\n")


def journal(value, receipts, *, complete, root="/"):
    """Write Node A's re-form receipt: the plan and each Spark's retirement receipt."""
    from runtime.host import node
    document = {"schema": "sparkring-reform/v1", "complete": complete, "plan": value, "receipts": receipts,
                "network_backups": ("Setup archives each Spark's NetworkManager connections in "
                                    f"/var/lib/sparkring/backups/{value['cluster']}/rankN/ before it changes "
                                    "fabric addresses."),
                "restore": "Each Spark's receipt.json in the same retired directory lists its restore steps."}
    node.save(root, f"{RETIRED}/{value['stamp']}/reform.json", document, mode=0o644)
    return document


def execute(value, found, transport, *, archive, transfer, root_command, keep=(), here=None, say=print, root="/"):
    """Retire every planned Spark and prepare the workers; Node A goes last.

    Each worker receives the worker bundle (``archive``), then one root
    program (``worker_script``) retires its state and installs and prepares
    it, so one sudo password covers both. Its receipt is read back
    afterwards. Workers reached through others go first, the farthest
    first, so no worker's route crosses a Spark already changed. Node A
    keeps ``NODE_A_KEEPS`` and ``keep`` (this setup's own directory, which
    holds the worker bundle). Returns the receipts by Spark name.
    """
    if value["blockers"]:
        raise ValueError("SparkRing model containers run on the Sparks to re-form. Stop them, then repeat setup: "
                         + "; ".join(f"{row['container']} on {row['spark']}: {row['command']}" for row in value["blockers"]))
    receipts = {}
    journal(value, receipts, complete=False, root=root)
    workers = [s for s in value["sparks"] if not s["node_a"]]
    workers.sort(key=lambda s: -len(found["sparks"][s["key"]]["reach"].route))
    for spark in workers:
        reach = found["sparks"][spark["key"]]["reach"]
        if reach.kind != "route":
            raise ValueError(f"{spark['name']} was reached only over {reach.label}; re-forming it needs a LAN or fabric "
                             "route")
        destination = "/var/tmp/sparkring-enroll-" + str(time.time_ns())
        say(f"Re-form {spark['name']}: move its earlier cluster state aside, then install and prepare SparkRing")
        transfer(transport, reach.route, archive(), destination)
        order = {"stamp": value["stamp"], "node_a": False, "keep": []}
        root_command(transport, reach.route, ["python3", "-I", "-c", worker_script(order, destination + "/install.py")])
        receipts[spark["name"]] = json.loads(transport.command(
            reach.route, ["cat", f"{RETIRED}/{value['stamp']}/receipt.json"]))
        journal(value, receipts, complete=False, root=root)
    node_a = next(s for s in value["sparks"] if s["node_a"])
    say(f"Re-form {node_a['name']} (Node A): move its earlier cluster state aside")
    receipts[node_a["name"]] = (here or retire)({"stamp": value["stamp"], "node_a": True,
                                                 "keep": [*NODE_A_KEEPS, *keep]})
    journal(value, receipts, complete=True, root=root)
    say(f"Re-form receipt: {RETIRED}/{value['stamp']}/reform.json")
    return receipts


def survey_cabled(transport, *, user, port, say=print, **options):
    """Survey the cabled Sparks with their SparkRing state; signs in over the LAN or the cables, never the old admin network."""
    found = survey.survey(transport, recorded=(), user=user, port=port, state=prior_state, say=say, **options)
    diagnosis = cabling.diagnose(survey.records(found), found["head"])
    return found, diagnosis
