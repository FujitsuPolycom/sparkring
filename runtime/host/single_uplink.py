"""Install workers through fabric SSH, then hand off to the existing setup engine."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from runtime.common import distribution, installer
from runtime.host import bootstrap, control, control_node, controller, discovery, node, packages, seed, settings, topology


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


def provision(discovered, transport, archive, *, private_key, public_key, control_cidr, share_uplink, directory):
    journal = Path(directory) / "provision.json"
    if journal.exists():
        raise ValueError("Provisioning receipt exists; inspect worker package/service state before recovery")
    record = {"schema": "sparkring-provision/v1", "complete": False, "steps": []}
    node.save(directory, "provision.json", record, mode=0o600)
    keys = {}
    for n in discovered["nodes"]:
        route = discovered["routes"][n["id"]]
        if route:
            destination = "/var/tmp/sparkring-enroll-" + str(time.time_ns())
            packages.transfer(transport, route, archive, destination)
            print("Install SparkRing and its packaged dependencies on " + n["hostname"])
            record["steps"].append({"host": n["id"], "operation": "package-install", "state": "running"})
            node.save(directory, "provision.json", record, mode=0o600)
            root_command(transport, route, ["python3", "-I", destination + "/install.py", "--apply"])
            record["steps"][-1]["state"] = "succeeded"
            node.save(directory, "provision.json", record, mode=0o600)
        # Generate private WireGuard keys locally; only public keys return.
        if route and route[-1]["user"] != "root":
            output = "/var/tmp/sparkring-public-" + str(time.time_ns()) + ".json"
            script = ("import subprocess,pathlib;p=pathlib.Path(" + repr(output) + ");p.write_bytes(subprocess.check_output(['/usr/bin/sparkring','node','control-key']));p.chmod(0o644)")
            root_command(transport, route, ["python3", "-I", "-c", script])
            result = transport.command(route, ["cat", output])
        else:
            result = transport.command(route, ["/usr/bin/sparkring", "node", "control-key"])
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


def main(argv=None):
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
    parser.add_argument("--allow-driver-reload", action="store_true")
    parser.add_argument("--stop-workloads", action="store_true",
                        help="stop (never remove) running GPU containers that block fabric preparation")
    parser.add_argument("--worker-bundle", action="store_true", help="build a USB/offline preparation bundle for workers without SSH")
    args = parser.parse_args(argv)
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
    if (base / "cluster.json").exists():
        cluster = installer.read(base / "cluster.json")
        targets = [h["host"] for h in cluster["plan"]["spec"]["hosts"]]
        api_address = cluster.get("api_address")
    elif (base / "enrolled.json").exists():
        enrolled = installer.read(base / "enrolled.json")
        targets, api_address = enrolled["targets"], enrolled.get("api_address")
    else:
        for prior in (base / "setups").glob("*/provision.json"):
            if not installer.read(prior).get("complete"):
                raise ValueError("A provisioning attempt is incomplete; inspect " + str(prior) + " and worker state before recovery")
        transport = bootstrap.SSH(base / "ssh", identity=private)
        if not args.plan:
            controller.confirm("Prepare unused local fabric ports for discovery? Existing configured links will be kept.", args.yes)
            # --yes approves setup scope only; stopping running GPU work needs
            # --stop-workloads or an explicit answer in a terminal.
            seed.prepare(public, stop=lambda names: controller.confirm(
                "Stop these running GPU containers so fabric ports can be prepared? They are stopped, not removed: "
                + ", ".join(names) + ".", args.stop_workloads))
        if sys.stdin.isatty() and not env.env and "--ssh-user" not in (argv or []):
            args.ssh_user = input("Worker SSH username [root]: ").strip() or "root"
        try:
            found = bootstrap.discover(transport, user=args.ssh_user, port=args.ssh_port,
                                       select=lambda peer: print(f"Neighbor on {peer['via']}/{peer['interface']}: {peer['address']}") is None)
        except (ValueError, RuntimeError) as error:
            raise ValueError(str(error) + "\nIf SSH is unavailable: sudo sparkring setup --worker-bundle") from error
        print(f"Found {len(found['nodes'])} authenticated Sparks. Node A: " + found["nodes"][0]["hostname"])
        for n in found["nodes"]:
            print("  " + n["hostname"] + "  " + n["id"][:12])
        print("Workers will receive SparkRing and a private administration network over the fabric.")
        print("Node A Internet sharing: " + ("disabled" if args.no_share_internet else "enabled (package/image/model downloads)"))
        installer.write(directory / "discovery.json", found)
        if args.plan:
            print("Discovery saved: " + str(directory / "discovery.json"))
            return 0
        controller.confirm("Install on these Sparks and establish the private administration network?", args.yes)
        archive = packages.build(directory / "worker-bundle", public)
        api_address = next(n["api_address"] for n in found["nodes"] if n["id"] == found["head"])
        targets = provision(found, transport, archive, private_key=private, public_key=public,
                            control_cidr=args.control_cidr, share_uplink=not args.no_share_internet, directory=directory)
        node.save(base, "enrolled.json", {"targets": targets, "api_address": api_address}, mode=0o600)
    nodes = controller.collect(targets)
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
    final = controller.apply(plan, directory, allow_driver_reload=args.allow_driver_reload,
                             review=lambda p: (controller.summarize(p), controller.confirm("Apply this refreshed fabric plan?", args.yes)))
    node.save(base, "cluster.json", {"schema": "sparkring-appliance-cluster/v1", "name": args.name,
                                    "plan": final, "api_address": api_address, "setup_receipt": str(directory / "setup.json")}, mode=0o600)
    print("Setup complete. Choose a model: sparkring models")
    return 0
