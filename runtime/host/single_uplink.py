"""Install workers through fabric SSH, then hand off to the existing setup engine."""
import argparse
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from runtime.common import distribution, installer
from runtime.host import (bootstrap, cabling, control, control_node, controller, discovery, fabric_bandwidth, lan_peers,
                          node, packages, reform, seed, settings, survey, topology)
from scripts import hairpin_setting

# The approval line for the ConnectX hairpin setting on four-Spark rings. The
# restart timing is measured on running hosts (5.1-7.8 s from command to link up).
HAIRPIN_SCOPE = (
    "  - on a four-Spark ring: apply the ConnectX hairpin setting that four-Spark",
    "    forwarding needs (a hairpin queue of " + str(hairpin_setting.HAIRPIN_QUEUE_SIZE) + " packets on each fabric port",
    "    function), now and at every boot. After fabric addressing is configured,",
    "    each function's driver restarts once, one at a time: its link is down for",
    "    about 8 seconds, about 30 seconds per Spark and about 3 minutes for the",
    "    ring. Every later boot takes about 30 seconds longer before networking",
    "    starts. SparkRing restarts nothing while a model, a mesh service or",
    "    another RDMA program runs on the ring. A worker that reaches Node A only",
    "    through the ring cables stays unreachable if one of its restarts fails,",
    "    until it is power-cycled; it then starts without the setting.",
)


def identity_key(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    private = directory / "controller_ed25519"
    if not private.exists():
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "sparkring-controller", "-f", str(private)], check=True)
    private.chmod(0o600)
    return private, private.with_suffix(".pub").read_text().strip()


def root_command(transport, route, argv, *, data=None):
    if not route or route[-1]["user"] == "root":
        return transport.command(route, argv, data=data)
    # After a one-time sudo authentication, install a root-owned transient
    # helper through the existing terminal. Passwords remain in sudo's prompt.
    if data is not None:
        encoded = __import__("base64").b64encode(data.encode()).decode()
        wrapper = ("import base64,subprocess;subprocess.run(" + repr(argv) + ",input=base64.b64decode(" + repr(encoded) + "),check=True)")
        argv = ["python3", "-I", "-c", wrapper]
    return transport.command(route, ["sudo", "--", *argv], tty=True)


class ControlSSH:
    """Root SSH to administration-network addresses.

    Root's SSH configuration (control_node.ssh_config) supplies each
    address's port, key and authenticated host key.
    """

    @staticmethod
    def argv(target):
        return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", target]


def other_revisions(targets, nodes):
    """Targets whose node record names a SparkRing revision other than Node A's."""
    current = distribution.identity(installer.ROOT)
    return [target for target, record in zip(targets, nodes, strict=True) if record.get("revision") != current]


def match_revisions(targets, nodes, directory):
    """Node records after every worker runs Node A's SparkRing revision.

    Planning uses each node's own inspection, while the network change checks
    every host again with the inspection code that Node A ships in the plan.
    A worker on another revision can report unchanged state in another form,
    and the change then stops before it starts. Such workers install Node A's
    package over the administration network; a package upgrade restarts no
    administration-network unit.
    """
    outdated = other_revisions(targets, nodes)
    if not outdated:
        return nodes
    current = distribution.identity(installer.ROOT)
    # Enrolled access is kept; this bundle does not run worker preparation.
    archive = packages.build(Path(directory) / "worker-update", "")
    for target in outdated:
        print(f"Update SparkRing on {target} to Node A's revision {current[:12]}")
        destination = "/var/tmp/sparkring-enroll-update-" + str(time.time_ns())
        try:
            packages.transfer(ControlSSH, target, archive, destination)
            discovery.ssh(target, ["python3", "-I", destination + "/install.py", "--apply"], timeout=1800)
        except RuntimeError as error:
            raise ValueError(f"Updating SparkRing on {target} failed: {error}") from error
    nodes = controller.collect(targets)
    if other_revisions(targets, nodes):
        raise ValueError("After the update, " + ", ".join(other_revisions(targets, nodes))
                         + f" still runs a SparkRing revision other than Node A's {current[:12]}")
    return nodes


def provision(discovered, transport, archive, *, private_key, public_key, control_cidr, share_uplink, directory):
    journal = Path(directory) / "provision.json"
    if journal.exists():
        raise ValueError("Provisioning receipt exists; inspect worker package/service state before recovery")
    record = {"schema": "sparkring-provision/v1", "complete": False, "steps": []}
    node.save(directory, "provision.json", record, mode=0o600)
    keys = {}
    for n in discovered["nodes"]:
        route = discovered["routes"][n["id"]]
        # Each node generates its private WireGuard key locally; only the
        # public key returns.
        key = ["/usr/bin/sparkring", "node", "control-key"]
        if route:
            destination = "/var/tmp/sparkring-enroll-" + str(time.time_ns())
            packages.transfer(transport, route, archive, destination)
            print("Install SparkRing and its packaged dependencies on " + n["hostname"])
            record["steps"].append({"host": n["id"], "operation": "package-install", "state": "running"})
            node.save(directory, "provision.json", record, mode=0o600)
            install = ["python3", "-I", destination + "/install.py", "--apply"]
            if route[-1]["user"] == "root":
                root_command(transport, route, install)
                result = transport.command(route, key)
            else:
                # One sudo authentication installs the package and exports the key.
                output = "/var/tmp/sparkring-public-" + str(time.time_ns()) + ".json"
                script = ("import subprocess,pathlib;subprocess.run(" + repr(install) + ",check=True);"
                          "p=pathlib.Path(" + repr(output) + ");p.write_bytes(subprocess.check_output(" + repr(key) + "));p.chmod(0o644)")
                root_command(transport, route, ["python3", "-I", "-c", script])
                result = transport.command(route, ["cat", output])
            record["steps"][-1]["state"] = "succeeded"
            node.save(directory, "provision.json", record, mode=0o600)
        else:
            result = transport.command(route, key)
        keys[n["id"]] = json.loads(result)
        n["public_key"] = keys[n["id"]]["public_key"]
    configs = control.plan(discovered["nodes"], discovered["edges"], discovered["head"], subnet=control_cidr, share_uplink=share_uplink)
    installer.write(Path(directory) / "control-plan.json", configs)
    control_node.ssh_config(configs, keys, private_key)
    for config in configs:
        route = discovered["routes"][config["id"]]
        record["steps"].append({"host": config["id"], "operation": "control-install", "state": "running"})
        node.save(directory, "provision.json", record, mode=0o600)
        root_command(transport, route, ["/usr/bin/sparkring", "node", "control-configure"],
                     data=json.dumps({"control": config, "ssh_key": public_key}))
        record["steps"][-1]["state"] = "succeeded"
        node.save(directory, "provision.json", record, mode=0o600)
    targets = ["root@" + config["address"] for config in configs]
    for target in targets:
        discovery.ssh(target, ["true"])
    # Close the optional preparation listener only after every permanent path works.
    for target in targets:
        discovery.ssh(target, ["systemctl", "disable", "--now", "sparkring-seed.service"])
    record["complete"] = True
    node.save(directory, "provision.json", record, mode=0o600)
    return targets


def installed_targets(base):
    """The SSH targets of the installed cluster, Node A first, or None before the first setup."""
    base = Path(base)
    if (base / "cluster.json").exists():
        return [h["host"] for h in installer.read(base / "cluster.json")["plan"]["spec"]["hosts"]]
    if (base / "enrolled.json").exists():
        return installer.read(base / "enrolled.json")["targets"]
    return None


def fallback_lines(configs, hostnames):
    """One line per tunnel link end: the Spark, its peer and the peer's fallback paths in preference order."""
    lines = []
    for config in configs:
        for peer in config["peers"]:
            options = control.paths(config, peer)[1:]
            text = ", ".join(control.path_text(path) for path in options) or "none found"
            lines.append(f"  {hostnames[config['id']]} to {peer['address']}: {text}")
    return lines


def admin_fallback(args, base, public, directory, *, invoke=discovery.ssh, collect=None, root="/"):
    """Add fallback paths to the installed administration network of every Spark; change nothing else.

    Each Spark reports its installed control configuration and a fresh
    inventory (bootstrap.probe) over the administration network. The
    configurations gain the fallback paths that control.extend finds; their
    addresses, keys, recorded endpoints and routes stay the same, which each
    Spark's ``control-configure`` checks again before it accepts them. With
    ``--plan`` the paths are printed only. Workers that run another SparkRing
    revision get Node A's first, as setup does, because older revisions refuse
    the extended configuration.
    """
    targets = installed_targets(base)
    if not targets:
        raise ValueError("No installed cluster here; sudo sparkring setup installs the administration network "
                         "with its fallback paths")
    if not node.location(root, "/etc/sparkring/control.json").exists():
        raise ValueError("This cluster has no SparkRing administration network; its Sparks are reached over "
                         "the addresses that setup was given")
    code = (inspect.getsource(bootstrap.fabric_identity) + "\n" + inspect.getsource(bootstrap.probe)
            + "\nimport json\nprint(json.dumps(probe()))\n")
    configs, inventories, hostnames = [], {}, {}
    for target in targets:
        config = json.loads(invoke(target, ["sudo", "-n", "cat", "/etc/sparkring/control.json"]))
        inventory = json.loads(invoke(target, ["sudo", "-n", "python3", "-I", "-c", code]))
        if config["id"] not in (inventory.get("id"), inventory.get("machine_id")):
            raise ValueError(f"{target} reports another Spark than its administration network configuration names")
        configs.append(config)
        inventories[config["id"]] = inventory
        hostnames[config["id"]] = inventory.get("hostname") or target
    extended = control.extend([control.base(config) for config in configs], inventories)
    print("Fallback paths of the administration network, in the order each Spark tries them:")
    for line in fallback_lines(extended, hostnames):
        print(line)
    if extended == configs:
        print("Every Spark already has these fallback paths.")
        return 0
    if args.plan:
        return 0
    controller.confirm("Add these fallback paths? Addresses, keys and the primary cables stay the same.", args.yes)
    match_revisions(targets, (collect or controller.collect)(targets), directory)
    for target, config in zip(targets, extended, strict=True):
        invoke(target, ["sudo", "-n", "/usr/bin/sparkring", "node", "control-configure"],
               data=json.dumps({"control": config, "ssh_key": public}))
    print("Fallback paths added. Each Spark checks its tunnel paths every 20 seconds; "
          "sudo sparkring status names a worker reached over a fallback path.")
    return 0


def scope_lines(args, *, fresh, follow=None, four=None):
    """The automated setup scope that the one approval covers.

    ``four`` adds the ConnectX hairpin line; it defaults to ``fresh``, because
    a fresh setup does not know the ring size before discovery.
    """
    lines = []
    if fresh:
        lines += ["  - find cabled Sparks over IPv6 link-local fabric addresses, adding link-local"
                  " addressing to fabric connections without it and turning off DHCP on fabric connections"
                  " that have no lease (their IPv4 addresses and MTU are kept)",
                  f"  - sign in as {args.ssh_user} on SSH port {args.ssh_port} (SSH asks for passwords; change with"
                  " --ssh-user) and trust each cabled Spark's SSH host key on first contact; fingerprints are printed"]
        if args.ssh_port == 22:
            lines.append("  - sign in to a cabled Spark on this LAN, when it is there, to install SparkRing and turn off"
                         " DHCP on its fabric connections; later sign-ins use Node A's key")
        lines += [
                  "  - install SparkRing and its packaged dependencies, then a private WireGuard administration network"
                  " that can also use the other fabric cables and the LAN",
                  "  - " + ("do not share" if args.no_share_internet else "share") + " Node A's Internet connection with workers",
                  "  - if a cabled Spark keeps setup from another SparkRing cluster: move that setup aside (kept, with a"
                  " receipt), turn off that cluster's admin network, recovery and mesh services, and replace its fabric"
                  " addresses after backing up their connections; setup stops instead while a SparkRing model runs there"]
    else:
        lines.append("  - install Node A's SparkRing revision on workers that run another one")
    lines.append("  - keep compatible fabric IPv4 addresses and replace incompatible ones, saving connection backups")
    if fresh if four is None else four:
        lines += HAIRPIN_SCOPE
    if follow:
        lines.append("  - " + follow)
    lines.append("Running GPU containers that block setup are listed and stopped only after a separate answer.")
    return lines


def approve(args, *, fresh, follow=None, four=None):
    """One default-yes approval for the whole automated setup scope."""
    print("Automated setup of this Spark (Node A) and the Sparks cabled to it:")
    for line in scope_lines(args, fresh=fresh, follow=follow, four=four):
        print(line)
    controller.confirm("Proceed?", default=True)


def announce(args, *, fresh, follow=None, four=None):
    """Print the scope that --yes approved; asks nothing."""
    print("Approved with --yes:")
    for line in scope_lines(args, fresh=fresh, follow=follow, four=four):
        print(line)


def ring_state(base, reason=None):
    """(fresh, four): whether setup starts from nothing, and whether the ring has, or may have, four Sparks.

    ``reason`` (``record_reason``) says the cabled Sparks differ from the
    record, so setup re-forms them and starts as fresh.
    """
    base = Path(base)
    if reason:
        return True, True
    if (base / "cluster.json").exists():
        return False, len(installer.read(base / "cluster.json")["plan"]["nodes"]) == 4
    if (base / "enrolled.json").exists():
        return False, len(installer.read(base / "enrolled.json")["targets"]) == 4
    return True, True


def record_reason(base, *, lldp=None, hostname=None):
    """Why the Sparks cabled to this Spark differ from its cluster record (``reform.record_mismatch``), or None.

    Reads only this Spark's LLDP neighbors, so an installed cluster whose
    cables still match is set up again without signing in anywhere.
    """
    base = Path(base)
    for name in ("cluster.json", "enrolled.json"):
        if (base / name).exists():
            return reform.record_mismatch(installer.read(base / name), (lldp or reform.local_lldp)(),
                                          hostname or __import__("socket").gethostname())
    return None


def reform_step(args, transport, worker_archive, *, reason, keep=(), say=print, run=None):
    """Survey the cabled Sparks; when they keep another cluster's setup, print the re-form plan and carry it out.

    Returns None when nothing needs re-forming (setup continues as before),
    ``"planned"`` after printing the plan of ``--plan``, and ``"done"`` after
    the re-form; setup then continues as a fresh setup over the workers'
    preparation SSH service with renumbered fabric addresses. Without a
    record mismatch (``reason``) a failed survey only prints a note.
    ``keep`` names this setup's directory, relative to the controller
    directory, which Node A's re-form keeps.
    """
    try:
        # Workers prepared offline admit only root, on port 2222 over the cables.
        prepared = args.ssh_port == 2222
        found, diagnosis = reform.survey_cabled(transport, user="root" if prepared else args.ssh_user,
                                                port=args.ssh_port, lan=not prepared, say=say)
    except (RuntimeError, ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        if reason:
            raise ValueError(f"Setup could not read the cabled Sparks ({error}); sudo sparkring cabling shows what it "
                             "can reach") from error
        say(f"Note: reading the cabled Sparks before setup failed ({error}); setup continues")
        return None
    if not reason and not reform.needed(found):
        return None
    say("Sparks read:")
    for line in survey.checked_lines(found):
        say(line)
    for note in found["notes"]:
        say("Note: " + note)
    if diagnosis["layout"] not in ("pair", "ring") or not diagnosis["ready"]:
        raise cabling.CablingError(diagnosis)
    unreached = [row["name"] for row in diagnosis["sparks"] if not row["reached"]]
    if unreached:
        raise ValueError("Setup could not sign in to " + " and ".join(unreached) + ", so it cannot re-form "
                         + ("it" if len(unreached) == 1 else "them") + "; the notes above say why")
    value = reform.plan(found, diagnosis, name=args.name, reason=reason)
    for line in reform.plan_lines(value):
        say(line)
    if args.plan:
        if diagnosis["layout"] == "ring":
            say("Setup of these Sparks also includes this step:")
            for line in HAIRPIN_SCOPE:
                say(line)
        return "planned"
    (run or reform.execute)(value, found, transport, archive=worker_archive, transfer=packages.transfer,
                            root_command=root_command, keep=keep, say=say)
    # After the re-form, every worker runs the preparation SSH service with Node A's key.
    args.ssh_port, args.ssh_user, args.reset_links = 2222, "root", True
    transport.trust_new = True
    return "done"


def _arguments(argv):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--env", type=Path)
    env, _ = pre.parse_known_args(argv)
    values = settings.load(env.env)
    parser = argparse.ArgumentParser(prog="sparkring setup", description="Run on Node A to install a pair/ring through its fabric links.")
    parser.add_argument("--env", type=Path, help="optional literal preferences; never required for guided setup")
    parser.add_argument("--name", default=values["SPARKRING_NAME"])
    parser.add_argument("--ssh-user", default=values["SPARKRING_SSH_USER"])
    parser.add_argument("--ssh-port", type=int, choices=(22, 2222), default=int(values["SPARKRING_SSH_PORT"]))
    parser.add_argument("--control-cidr", default=values["SPARKRING_CONTROL_CIDR"])
    parser.add_argument("--fabric-cidr", default=values["SPARKRING_FABRIC_CIDR"])
    parser.add_argument("--no-share-internet", action="store_true", default=values["SPARKRING_SHARE_INTERNET"] == "no")
    parser.add_argument("--reset-links", action="store_true", default=values["SPARKRING_LINK_POLICY"] == "reset")
    parser.add_argument("--plan", action="store_true", help="discover/review with existing SSH access; no host configuration")
    parser.add_argument("--yes", action="store_true", help="accept configuration scope; SSH host identity still requires verification")
    parser.add_argument("--allow-driver-reload", action="store_true", help=controller.ALLOW_DRIVER_RELOAD)
    parser.add_argument("--stop-workloads", action="store_true",
                        help="stop (never remove) running GPU containers that block fabric preparation")
    parser.add_argument("--worker-bundle", action="store_true", help="build a USB/offline preparation bundle for workers without SSH")
    parser.add_argument("--admin-fallback", action="store_true",
                        help="on an installed cluster, add fallback paths (the other fabric cables, the LAN) to the "
                             "administration network; changes nothing else")
    return parser.parse_args(argv), env.env


def _default_user(args, argv, env, fresh):
    if fresh and "--ssh-user" not in (argv or []) and not env and args.ssh_port == 22:
        # Sparks are usually set up with one account name; sudo records it.
        args.ssh_user = os.environ.get("SUDO_USER") or "root"


def scope(argv=None, *, follow=None):
    """The approval scope lines that ``main(argv)`` would list, for a non-interactive approval request."""
    args, env = _arguments(argv)
    fresh, four = ring_state(controller.STATE, record_reason(controller.STATE))
    _default_user(args, argv, env, fresh)
    return scope_lines(args, fresh=fresh, follow=follow, four=four)


def main(argv=None, *, follow=None):
    args, env = _arguments(argv)
    if args.allow_driver_reload:
        print("--allow-driver-reload: " + controller.ALLOW_DRIVER_RELOAD)
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,34}", args.name):
        raise ValueError("Choose a lowercase cluster name of at most 35 characters")
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        raise ValueError("Run sudo sparkring setup on the Spark that should be Node A")
    if not distribution.installed(installer.ROOT):
        raise ValueError("Install the local ARM64 Debian package on Node A before fabric provisioning")
    base = controller.STATE
    base.mkdir(parents=True, exist_ok=True, mode=0o700)
    private, public = identity_key(base)
    if args.worker_bundle:
        archive = packages.build(base / ("worker-" + str(time.time_ns())), public)
        print("Copy/extract " + str(archive) + " on a worker, then run: sudo python3 install.py --apply --prepare")
        return 0
    directory = base / "setups" / str(time.time_ns())
    if args.admin_fallback:
        return admin_fallback(args, base, public, directory)
    reason = record_reason(base)
    fresh, four = ring_state(base, reason)
    _default_user(args, argv, env, fresh)
    trust_new = False
    if not args.plan and args.yes:
        announce(args, fresh=fresh, follow=follow, four=four)
    elif not args.plan and sys.stdin.isatty():
        approve(args, fresh=fresh, follow=follow, four=four)
        args.yes = trust_new = True
    if (base / "cluster.json").exists() and not reason:
        cluster = installer.read(base / "cluster.json")
        targets = [h["host"] for h in cluster["plan"]["spec"]["hosts"]]
        api_address = cluster.get("api_address")
    elif (base / "enrolled.json").exists() and not reason:
        enrolled = installer.read(base / "enrolled.json")
        targets, api_address = enrolled["targets"], enrolled.get("api_address")
    else:
        for prior in (base / "setups").glob("*/provision.json"):
            if not installer.read(prior).get("complete"):
                # Every provisioning step repeats safely: packages reinstall,
                # node keys persist, and a control configuration that differs
                # from the installed one is refused on the node itself.
                prior.rename(prior.with_name("provision-incomplete.json"))
                print("A previous setup stopped during provisioning; starting it again (its record: "
                      + str(prior.with_name("provision-incomplete.json")) + ")")
        transport = bootstrap.SSH(base / "ssh", identity=private, trust_new=trust_new)
        bundle = {}

        def worker_archive():
            if "path" not in bundle:
                bundle["path"] = packages.build(directory / "worker-bundle", public)
            return bundle["path"]
        reformed = reform_step(args, transport, worker_archive, reason=reason,
                               keep=[str(directory.relative_to(base).as_posix())])
        if reformed == "planned":
            return 0
        if not args.plan:
            controller.confirm("Prepare unused local fabric ports for discovery? Existing configured links will be kept.", args.yes)
            # --yes approves setup scope only; stopping running GPU work needs
            # --stop-workloads or an explicit answer in a terminal.
            seed.prepare(public, stop=lambda names: controller.confirm(
                "Stop these running GPU containers so fabric ports can be prepared? They are stopped, not removed: "
                + ", ".join(names) + ".", args.stop_workloads),
                link_local=lambda name: controller.confirm(
                    "Add IPv6 link-local addressing to fabric connection " + name
                    + " for discovery? Its IPv4 addresses and MTU are kept.", args.yes),
                dhcp=lambda name: controller.confirm(
                    "Turn off DHCP on fabric connection " + name + ", which a direct cable does not answer, "
                    "and keep IPv6 link-local addressing?", args.yes))
            if args.ssh_port == 22 and not reformed:
                head, _ = lan_peers.wait_for_peers(lambda: transport.inventory([]))
                if lan_peers.prepare(transport, root_command, head=head, user=args.ssh_user, archive=worker_archive):
                    # Prepared Sparks run the preparation service, which admits
                    # root with Node A's key; the LAN sign-in verified each
                    # Spark's fabric hardware, so their fabric host keys are
                    # recorded on first contact.
                    args.ssh_port = 2222
                    transport.trust_new = True
        if args.ssh_port == 2222:
            # The worker preparation service admits only root with Node A's key.
            args.ssh_user = "root"
        try:
            found = bootstrap.discover(transport, user=args.ssh_user, port=args.ssh_port,
                                       select=lambda peer: print(f"Neighbor on {peer['via']}/{peer['interface']}: {peer['address']}") is None)
        except RuntimeError as error:
            raise ValueError(str(error)) from error
        print(f"Found {len(found['nodes'])} authenticated Sparks. Node A: " + found["nodes"][0]["hostname"])
        for n in found["nodes"]:
            print("  " + n["hostname"] + "  " + n["id"][:12])
        for warning in found.get("warnings", []):
            print("Note: " + warning)
        for line in transport.trusted():
            print("  SSH host key trusted on first contact: " + line)
        print("Workers will receive SparkRing and a private administration network over the fabric.")
        print("Node A Internet sharing: " + ("disabled" if args.no_share_internet else "enabled (package/image/model downloads)"))
        installer.write(directory / "discovery.json", found)
        if args.plan:
            if len(found["nodes"]) == 4:
                print("Setup of these Sparks also includes this step:")
                for line in HAIRPIN_SCOPE:
                    print(line)
            print("Discovery saved: " + str(directory / "discovery.json"))
            return 0
        controller.confirm("Install on these Sparks and establish the private administration network?", args.yes)
        archive = worker_archive()
        api_address = next(n["api_address"] for n in found["nodes"] if n["id"] == found["head"])
        targets = provision(found, transport, archive, private_key=private, public_key=public,
                            control_cidr=args.control_cidr, share_uplink=not args.no_share_internet, directory=directory)
        node.save(base, "enrolled.json", {"targets": targets, "api_address": api_address}, mode=0o600)
    nodes = controller.collect(targets)
    if args.plan:
        for target in other_revisions(targets, nodes):
            print(f"Note: {target} runs another SparkRing revision; setup updates it to Node A's before changing networking.")
    else:
        nodes = match_revisions(targets, nodes, directory)
    head = node.read("/", "/etc/sparkring/node.json")["node_id"]
    try:
        plan = topology.build_spec(nodes, head, name=args.name, fabric_cidr=args.fabric_cidr,
                                   reset=args.reset_links, preserve_control=True)
    except ValueError as error:
        address_problem = str(error).startswith(("Partial fabric addressing", "Pair endpoints", "Pair functions", "Every data function",
                                                 "Cable ", "Each cable function", "Supported persistent fabric addresses"))
        if args.reset_links or not sys.stdin.isatty() or not address_problem:
            raise
        print(str(error))
        controller.confirm("Existing fabric addressing is incompatible. Replace fabric IPv4 settings while retaining control access?", args.yes)
        plan = topology.build_spec(nodes, head, name=args.name, fabric_cidr=args.fabric_cidr, reset=True, preserve_control=True)
    controller.summarize(plan)
    installer.write(directory / "plan.json", plan)
    if args.plan:
        return 0
    controller.confirm("Apply these fabric IPv4/MTU settings? Existing connection backups will be retained.", args.yes)
    # The setup approval (the question, or --yes) lists the ConnectX hairpin
    # step on four-Spark rings; controller.apply runs it after addressing.
    final = controller.apply(plan, directory, approved=args.yes,
                             review=lambda p: (controller.summarize(p), controller.confirm("Apply this refreshed fabric plan?", args.yes)))
    cluster = {"schema": "sparkring-appliance-cluster/v1", "name": args.name, "plan": final,
               "api_address": api_address, "setup_receipt": str(directory / "setup.json")}
    node.save(base, "cluster.json", cluster, mode=0o600)
    # Fresh setups, re-forms and repeated setups all end here. A degraded
    # cable is a warning with its repair steps; setup never fails here.
    fabric_bandwidth.after_setup(base, cluster)
    print("Setup complete. Choose a model: sparkring models")
    return 0
