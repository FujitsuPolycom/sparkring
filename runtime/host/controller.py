"""Guided Linux setup; existing network and model engines own all execution."""
import argparse
import getpass
import hashlib
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
import time

from runtime.common import distribution, installer
from runtime.host import discovery, node, topology
from scripts import deploy_engine, deploy_network, deploy_network_run, sparkring_bootstrap

STATE = Path("/var/lib/sparkring/controller")


def confirm(prompt, yes=False, *, default=False):
    """Ask in a terminal; `y`/`yes` in any letter case approves, anything else cancels.

    With ``default`` an empty answer (Enter) also approves.
    """
    if yes:
        return
    answer = input(prompt + (" [Y/n]: " if default else " [y/N]: ")).strip().lower() if sys.stdin.isatty() else "n"
    if answer not in ("y", "yes") and not (default and answer == ""):
        raise ValueError("Cancelled; no further changes")


def collect(targets, *, invoke=discovery.inspect_node):
    if len(targets) not in (2, 4) or len(set(targets)) != len(targets):
        raise ValueError("Select exactly two or four distinct Spark management addresses")
    import concurrent.futures
    from runtime.host import progress

    def inspect(rank):
        with progress.step(f"Node {rank}: inspect hardware, links and software"):
            return invoke(targets[rank], rank, targets[1 if rank == 0 else 0].split("@", 1)[1])

    # Inspection only reads each node, so the nodes are observed concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(targets)) as pool:
        return list(pool.map(inspect, range(len(targets))))


def summarize(plan, *, observe_only=False):
    print(f"{len(plan['nodes'])} Sparks: " + ("p0 pair" if len(plan["nodes"]) == 2 else "p0-to-p1 ring"))
    for host, proposed in zip(plan["spec"]["hosts"], plan["network"]["hosts"], strict=True):
        print(f"  rank {host['rank']}: {host['host']}  " + ("verify existing" if observe_only else proposed["action"]))
        for port in host["data_interfaces"]:
            print(f"    {port['netdev']}  {port['address']}  MTU 9000")
        for problem in proposed["blocked_by"]:
            print("    BLOCKED: " + problem)
    print("Existing networking will be verified and recorded." if observe_only else "Setup saves network state and enables its boot service. Model images/weights are selected by 'sparkring up'.")


def apply(plan, directory, *, inspect_nodes=collect, run=None, invoke=discovery.ssh,
          allow_driver_reload=False, review=lambda p: None):
    """Journal each step and re-observe after reload; never retry an unknown mutation."""
    directory = Path(directory)
    journal = directory / "setup.json"
    if journal.exists():
        raise ValueError("Setup receipt exists; inspect it and host state before recovery: " + str(journal))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = {"schema": "sparkring-setup-receipt/v1", "complete": False, "plan_id": plan["id"], "steps": []}
    deploy_engine.save_receipt(journal, record)
    head = plan["nodes"][0]["node_id"]
    targets = [h["host"] for h in plan["spec"]["hosts"]]
    # At most one reload per data function, followed by a configuration pass.
    for step in range(18):
        executable = deploy_network_run.build_network_plan({"spec": plan["spec"]}, plan["inventory"])
        if executable["driver_reload"] and not allow_driver_reload:
            raise ValueError("Driver reload required; review with --allow-driver-reload on an idle cluster")
        record["steps"].append({"network_plan": executable["sha256"], "state": "running"})
        deploy_engine.save_receipt(journal, record)
        deploy_engine.execute_plan(executable, directory / f"network-{step}.json", executable["sha256"],
                                   runner=run, allow_driver_reload=allow_driver_reload)
        record["steps"][-1]["state"] = "succeeded"
        deploy_engine.save_receipt(journal, record)
        refreshed = inspect_nodes(targets)
        if {n["node_id"] for n in refreshed} != {n["node_id"] for n in plan["nodes"]}:
            raise ValueError("Node identities changed during setup")
        plan = topology.build_spec(refreshed, head, name=plan["spec"]["owner"],
                                   fabric_cidr=plan.get("fabric_cidr", "198.18.0.0/21"), reset=plan.get("reset_requested", False),
                                   preserve_control=plan["spec"].get("preserve_control_ipv6", False))
        if not any(h["action"] != "none" for h in plan["network"]["hosts"]):
            break
        review(plan)
    else:
        raise ValueError("Networking did not converge; inspect setup receipts")
    deploy_network.verify_network(plan["spec"], plan["inventory"]["hosts"])
    # All nodes verify before any persistent service is installed.
    for rank, host in enumerate(plan["spec"]["hosts"]):
        config = topology.persistent_config(plan, rank)
        invoke(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "verify"], data=json.dumps(config))
    for rank, host in enumerate(plan["spec"]["hosts"]):
        config = topology.persistent_config(plan, rank)
        record["steps"].append({"host": host["host"], "persist": "running"})
        deploy_engine.save_receipt(journal, record)
        invoke(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "configure"], data=json.dumps(config))
        invoke(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "workspace", "--operator",
                              host["host"].split("@", 1)[0], "--name", plan["spec"]["owner"]])
        record["steps"][-1]["persist"] = "succeeded"
        deploy_engine.save_receipt(journal, record)
    # Exercise every planned data address, including routed ring peers. This is
    # an IP/jumbo-frame check, not a verbs collective or native relay test.
    for host in plan["spec"]["hosts"]:
        own = {p["address"] for p in host["data_interfaces"]}
        for peer in plan["spec"]["hosts"]:
            for port in peer["data_interfaces"]:
                if port["address"] not in own:
                    invoke(host["host"], ["ping", "-n", "-c", "1", "-W", "3", "-M", "do", "-s", "8972",
                                          str(ipaddress.ip_interface(port["address"]).ip)])
    record.update(complete=True, final_plan_id=plan["id"], hardware_qualified=False)
    deploy_engine.save_receipt(journal, record)
    return plan


def setup(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring setup", description="Discover a pair/ring, review its fabric, then save persistent host setup.")
    parser.add_argument("--node", action="append", help="USER@management-IP (include this head); bypass mDNS")
    parser.add_argument("--name", default="sparkring")
    parser.add_argument("--fabric-cidr", default="198.18.0.0/21")
    parser.add_argument("--plan", action="store_true", help="read existing SSH access only; no enrollment or host changes")
    parser.add_argument("--apply", action="store_true", help="apply the reviewed plan")
    parser.add_argument("--adopt", action="store_true", help="verify/save existing networking without changing links, routes or services")
    parser.add_argument("--yes", action="store_true", help="accept the printed configuration scope; never trusts SSH host keys")
    parser.add_argument("--skip-enroll", action="store_true", help="SSH keys and host trust are already configured")
    parser.add_argument("--allow-driver-reload", action="store_true")
    parser.add_argument("--inventory", type=Path, help="offline array of authenticated-node fixture records; planning only")
    parser.add_argument("--head-id", help="head node UUID for offline inventory")
    parser.add_argument("--output", type=Path, help="new private setup receipt directory")
    args = parser.parse_args(argv)
    if args.plan and args.apply or args.inventory and (args.apply or not args.plan):
        raise ValueError("Offline inventory requires --plan; --plan and --apply are exclusive")
    if args.yes and not args.apply and not args.plan:
        raise ValueError("Noninteractive setup changes require --apply --yes")
    if args.inventory:
        nodes = json.loads(args.inventory.read_text(encoding="utf-8"))
        head = args.head_id
        if not head:
            raise ValueError("Offline inventory requires --head-id")
    else:
        identity = node.read("/", "/etc/sparkring/node.json")
        targets = args.node
        if targets is None:
            candidates = discovery.discover()
            if not candidates:
                raise ValueError("No Sparks advertised. Install the package on each Spark, or pass --node USER@IP for each.")
            for i, candidate in enumerate(candidates, 1):
                print(f"  {i}: {candidate['hostname']}  {candidate['address']} (identity unverified)")
            if not sys.stdin.isatty():
                raise ValueError("Select nodes with --node for noninteractive setup")
            indices = [int(s) - 1 for s in input("Select two or four numbers, including this Spark: ").split()]
            if any(i < 0 or i >= len(candidates) for i in indices):
                raise ValueError("Selection is outside the candidate list")
            targets = [getpass.getuser() + "@" + candidates[i]["address"] for i in indices]
        targets = [discovery.target(t) for t in targets]
        if len(targets) not in (2, 4) or len(set(targets)) != len(targets):
            raise ValueError("Select exactly two or four distinct management targets")
        if not args.skip_enroll and not args.plan:
            confirm("Enroll SSH access to " + ", ".join(targets) + "?", args.yes)
            key = sparkring_bootstrap.ensure_local_key()
            sparkring_bootstrap.authorize_local_key(key)
            for value in targets:
                sparkring_bootstrap.enroll_target(value, key)
        nodes = collect(targets)
        head = identity["node_id"]
        expected = distribution.identity(installer.ROOT)
        if any(n.get("revision") != expected for n in nodes):
            raise ValueError("Install the same SparkRing package revision on all selected nodes")
    plan = topology.build_spec(nodes, head, name=args.name, fabric_cidr=args.fabric_cidr)
    summarize(plan, observe_only=args.adopt)
    directory = args.output or Path.home() / ".local/state/sparkring/setups" / str(time.time_ns())
    installer.write(directory / "plan.json", plan)
    print("Full plan: " + str(directory / "plan.json"))
    if args.plan or not args.apply and not sys.stdin.isatty():
        print("Plan saved. Repeat with --apply to configure these hosts.")
        return 0
    confirm("Record this verified existing fabric without network changes?" if args.adopt else "Apply this network configuration and enable fabric/agent services?", args.yes)

    def review(value):
        summarize(value)
        confirm("Apply the refreshed plan after driver/configuration discovery?", args.yes)

    if args.adopt:
        observed = []
        for rank, host in enumerate(plan["spec"]["hosts"]):
            config = topology.persistent_config(plan, rank)
            config.update(ownership="observed", routes=[], forwarding=[])
            if len(nodes) == 4:
                mesh = json.loads(discovery.ssh(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "native-mesh", "--rank", str(rank)]))["mesh"]
                if not mesh:
                    raise ValueError("No verified native mesh found; ordinary setup can prepare one")
                order = ("cw_primary", "ccw_primary", "cw_secondary", "ccw_secondary")
                config["native_mesh"] = {"reference": mesh["reference"], "host_ip": mesh["host_ip"],
                                         "hcas": [next(p["rdma_device"] for p in host["data_interfaces"] if p["role"] == role) for role in order]}
            discovery.ssh(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "adopt"], data=json.dumps(config))
            discovery.ssh(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "workspace", "--operator", host["host"].split("@", 1)[0], "--name", args.name])
            observed.append({"rank": rank, "adopted": True})
        installer.write(directory / "setup.json", {"complete": True, "network_changed": False, "nodes": observed})
        print("Existing fabric verified. No link, route or service changes.")
    else:
        plan = apply(plan, directory, allow_driver_reload=args.allow_driver_reload, review=review)
    # workspace() established this directory for the SSH operator, not root.
    cluster = {"schema": "sparkring-appliance-cluster/v1", "name": args.name, "plan": plan,
               "setup_receipt": str(directory / "setup.json")}
    path = STATE / "cluster.json"
    if path.exists() and installer.read(path)["plan"]["id"] != plan["id"]:
        raise ValueError("Controller already records another cluster; inspect " + str(path))
    node.save(STATE, "cluster.json", cluster, mode=0o600)
    print("Network configured. Choose a model: sparkring models")
    return 0


def model_site(cluster, profile, instance="main"):
    plan = cluster["plan"]
    rows = []
    identities = plan.get("nodes", [])
    for rank, host in enumerate(plan["spec"]["hosts"]):
        port = next(p for p in host["data_interfaces"] if p["role"] == "cw_primary")
        rows.append({"host": host["host"], "management_ip": host["management_address"],
                     "fabric_ip": str(ipaddress.ip_interface(port["address"]).ip), "interface": port["netdev"]})
        if len(identities) == len(plan["spec"]["hosts"]) and isinstance(identities[rank], dict) and identities[rank].get("node_id"):
            rows[-1]["node_id"] = identities[rank]["node_id"]
    import re
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", instance):
        raise ValueError("Instance must be a short lowercase name")
    identity = profile if instance == "main" else profile + "-" + instance
    name = cluster["name"][:12] + "-" + profile[:18] + "-" + hashlib.sha256(identity.encode()).hexdigest()[:6]
    result = {"schema": "sparkring-install-site/v1", "name": name,
              "workspace": "/srv/sparkring/" + cluster["name"] + "/" + identity,
              "hosts": rows, "controller_address": rows[0]["management_ip"]}
    if cluster.get("api_address"):
        result["api_address"] = cluster["api_address"]
    return result


def lifecycle(argv):
    parser = argparse.ArgumentParser(prog="sparkring " + argv[0])
    parser.add_argument("operation", choices=("up", "down", "status"))
    parser.add_argument("profile", nargs="?", help="exact profile shown by sparkring models")
    parser.add_argument("--model-path", help="serve this complete copy read-only on every rank; it is verified, never changed")
    parser.add_argument("--image-lock", type=Path, help="explicit source-recorded toolchain image for a separate rehearsal")
    parser.add_argument("--fresh-mesh", action="store_true", help="review replacement of an existing native mesh")
    parser.add_argument("--instance", default="main", help="separate local deployment name for a rehearsal")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    image_runtime = None
    if args.image_lock and (args.operation != "up" or not args.profile):
        raise ValueError("--image-lock requires up with an exact profile")
    if args.operation == "up" and args.profile:
        from runtime.common import installer_image
        image_runtime = installer_image.for_profile(args.profile, installer.read(args.image_lock) if args.image_lock else None)
    if args.operation == "status":
        result = node.snapshot() if args.refresh else node.status()
        if (STATE / "cluster.json").exists():
            cluster = installer.read(STATE / "cluster.json")
            result["nodes"] = []
            for host in cluster["plan"]["spec"]["hosts"]:
                try:
                    command = ["/usr/bin/sparkring", "node", "status"]
                    if args.refresh:
                        command = ["sudo", "-n", *command, "--refresh"]
                    observation = json.loads(discovery.ssh(host["host"], command))
                except (ValueError, RuntimeError, OSError, subprocess.SubprocessError) as error:
                    observation = {"state": "unreachable", "error": str(error)}
                result["nodes"].append({"host": host["host"], **observation})
        if (STATE / "active.json").exists():
            path = installer.read(STATE / "active.json")["path"]
            from runtime.host import retained_source
            result["deployment"] = retained_source.apply(path, "status" if args.refresh else "saved-status", cache=STATE / "retained-sources")
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print(result["state"] + " | next: " + result.get("next_action", "sparkring status"))
            for row in result.get("nodes", []):
                print("  " + row["host"] + ": " + row["state"] + (" — " + row["error"] if row.get("error") else ""))
            if result.get("deployment"):
                saved = result["deployment"]
                print("Saved model operation: " + saved["profile"] + " | " + saved["state"]["operation"] + (" complete" if saved["state"].get("complete") else " incomplete"))
                print(saved["api_url"])
                if saved.get("observations"):
                    print(json.dumps(saved["observations"], indent=2))
                else:
                    print("Use --refresh for current model container state.")
            print("Network observations do not qualify GPU/RDMA serving.")
        return 0
    if args.plan and args.execute:
        raise ValueError("Choose --plan or --execute")
    if args.operation == "up" and args.profile:
        from runtime.host import models
        cluster = installer.read(STATE / "cluster.json")
        profile = models.select(args.profile, len(cluster["plan"]["nodes"]))
        directory = STATE / "deployments" / profile
        if args.instance != "main":
            directory = directory.with_name(profile + "-" + args.instance)
        if not directory.exists():
            site = model_site(cluster, profile, args.instance)
            # Every rank uses the cluster's SparkRing checkpoint directory for the
            # profile's revision, whose model operation adopts what that
            # directory holds and downloads the rest on that rank. A copy
            # SparkRing did not create is used only when named, and is then
            # served in place: verified, never written. Copies found elsewhere
            # on the Sparks are adopted by sparkring install.
            model = args.model_path or installer.checkpoint_directory(cluster, installer.setup.selection(profile))
            for row in site["hosts"]:
                row.update(model=model, reuse_verified_model=bool(args.model_path))
            if profile in installer.compose.TP4_PROFILES:
                from runtime.host import native_mesh
                site = native_mesh.select(site, cluster, profile, fresh=args.fresh_mesh)
            installer.init(directory, profile, site, image_runtime=image_runtime)
        else:
            existing = installer.load(directory)
            if image_runtime is not None and existing.get("image_runtime") != image_runtime:
                raise ValueError("Deployment uses another image lock; choose a distinct --instance")
            if args.model_path and any(row["model"] != args.model_path or not row["reuse_verified_model"] for row in existing["site"]["ranks"]):
                raise ValueError("Deployment uses another model path; choose a distinct --instance")
            if args.fresh_mesh and "native_mesh" not in existing["site_input"]:
                raise ValueError("Deployment reuses an existing mesh; use --instance fresh --fresh-mesh for a separate rehearsal")
    else:
        directory = Path(installer.read(STATE / "active.json")["path"])
    result = installer.apply(directory, args.operation, runner=None, execute=False)
    print(f"{args.operation}: {result['profile']} on " + ", ".join(result["hosts"]))
    if image_runtime is not None:
        print("Development image: " + image_runtime["name"] + " | " + image_runtime["image_id"])
    print(" -> ".join(result["phases"]))
    if "native_mesh" in result:
        print("Prepare native ASIC fabric and install its supervised service.")
        for old in result["native_mesh"]["replaces"]:
            print(f"  Stop/disable rank {old['rank']} service: {old['unit']}")
    if args.plan or not args.execute and not sys.stdin.isatty():
        print("Review, then repeat with --execute.")
        return 0
    if args.operation == "up" and (STATE / "active.json").exists():
        active = Path(installer.read(STATE / "active.json")["path"])
        if active != directory:
            previous = installer.status(active)["state"]
            if previous.get("operation") != "down" or not previous.get("complete"):
                raise ValueError("Run sparkring down before selecting another model")
    confirm("Apply these model/image actions?", args.execute)
    from runtime.host import retained_source
    result = retained_source.apply(directory, args.operation, cache=STATE / "retained-sources")
    node.save(STATE, "active.json", {"path": str(directory)}, mode=0o600)
    print(json.dumps(result, indent=2) if args.json else "Model operation complete. sparkring status --refresh")
    return 0


def main(argv):
    try:
        if argv[0] == "setup" and not any(flag in argv for flag in ("--node", "--inventory")):
            from runtime.host.single_uplink import main as uplink_main
            return uplink_main(argv[1:])
        return setup(argv[1:]) if argv[0] == "setup" else lifecycle(argv)
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        return 2
