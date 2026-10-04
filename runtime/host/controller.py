"""Guided Linux setup; existing network and model engines own all execution.

On four-Spark rings setup also applies the ConnectX hairpin setting through
``runtime/host/hairpin_ring.py`` once fabric addressing has converged.
"""
import argparse
import contextlib
import getpass
import hashlib
import ipaddress
import json
from pathlib import Path
import subprocess
import sys
import time

from runtime.common import distribution, installer, process_lock, thinking
from runtime.host import discovery, hairpin_ring, node, topology
from scripts import deploy_engine, deploy_network, deploy_network_run, sparkring_bootstrap

STATE = Path("/var/lib/sparkring/controller")
# Help text and notice of --allow-driver-reload, which setup and install accept.
ALLOW_DRIVER_RELOAD = "not needed: the approval question or --yes covers the ConnectX restarts"
# Addressing converges in one pass; the next pass confirms it.
ADDRESSING_PASSES = 4
# Node status states of a Spark that needs no action.
HEALTHY = ("network-configured", "existing-network-verified")


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


def detail_lines(details):
    """Terminal lines for the details of a needs_input result.

    ``lines`` and ``scope`` are printed as they are; other scalar values and
    lists are printed per key. A whole result document (it carries
    ``schema``) was already printed as the plan and is skipped.
    """
    if not isinstance(details, dict) or not details:
        return []
    lines = []
    for key in ("scope", "lines"):
        if isinstance(details.get(key), list):
            lines += [str(line) for line in details[key]]
    if lines or details.get("schema"):
        return lines
    for key, value in details.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    item = ", ".join(f"{k}: {v}" for k, v in item.items() if not isinstance(v, (dict, list)))
                lines.append("  - " + str(item))
        elif not isinstance(value, dict):
            lines.append(f"{key}: {value}")
    return lines


def summarize(plan, *, observe_only=False):
    print(f"{len(plan['nodes'])} Sparks: " + ("p0 pair" if len(plan["nodes"]) == 2 else "p0-to-p1 ring"))
    hairpin = hairpin_ring.requirement(plan)
    for host, proposed in zip(plan["spec"]["hosts"], plan["network"]["hosts"], strict=True):
        line = f"  rank {host['rank']}: {host['host']}  " + ("verify existing" if observe_only else proposed["action"])
        if hairpin and not observe_only:
            line += "  driver: " + proposed.get("driver_action", "none")
        print(line)
        for port in host["data_interfaces"]:
            print(f"    {port['netdev']}  {port['address']}  MTU 9000")
        for problem in proposed["blocked_by"]:
            print("    BLOCKED: " + problem)
        if hairpin:
            # Adoption restarts no function, so its lines never announce a restart.
            print("    ConnectX hairpin: " + hairpin_ring.summary_line(hairpin[host["rank"]], adopt=observe_only))
    print("Existing networking will be verified and recorded." if observe_only else "Setup saves network state and enables its boot service. Model images/weights are selected by 'sparkring up'.")


def _sync_workers(plan, directory):
    """Update the workers' SparkRing package from Node A over the plan's fabric paths."""
    from runtime.host import fabric_ssh, install_assets
    transport = fabric_ssh.Transport({"plan": plan}, STATE / "bulk-ssh")
    # Each bulk path must reach the enrolled node identity before a package crosses it.
    transport.verify()
    return install_assets.Assets(transport, Path(directory) / "assets").sync_packages()


def apply(plan, directory, *, inspect_nodes=collect, run=None, invoke=discovery.ssh,
          approved=False, review=lambda p: None, ensure=None, update_workers=None):
    """Journal each step and re-observe after each pass; never retry an unknown mutation.

    Fabric addressing passes run NetworkManager changes only
    (``defer_driver=True``) until no host needs one. On a four-Spark ring the
    ConnectX hairpin step (``hairpin_ring.ensure``, approved by ``approved``)
    then applies the driver setting, and the network is verified on the plan
    that step returns.
    """
    directory = Path(directory)
    journal = directory / "setup.json"
    if journal.exists():
        raise ValueError("Setup receipt exists; inspect it and host state before recovery: " + str(journal))
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    record = {"schema": "sparkring-setup-receipt/v1", "complete": False, "plan_id": plan["id"], "steps": []}
    deploy_engine.save_receipt(journal, record)
    head = plan["nodes"][0]["node_id"]
    targets = [h["host"] for h in plan["spec"]["hosts"]]
    options = {"name": plan["spec"]["owner"], "fabric_cidr": plan.get("fabric_cidr", "198.18.0.0/21"),
               "reset": plan.get("reset_requested", False),
               "preserve_control": plan["spec"].get("preserve_control_ipv6", False)}

    def rebuild(found):
        return topology.build_spec(found, head, **options)

    for step in range(ADDRESSING_PASSES):
        executable = deploy_network_run.build_network_plan({"spec": plan["spec"]}, plan["inventory"], defer_driver=True)
        record["steps"].append({"network_plan": executable["sha256"], "state": "running"})
        deploy_engine.save_receipt(journal, record)
        deploy_engine.execute_plan(executable, directory / f"network-{step}.json", executable["sha256"], runner=run)
        record["steps"][-1]["state"] = "succeeded"
        deploy_engine.save_receipt(journal, record)
        refreshed = inspect_nodes(targets)
        if {n["node_id"] for n in refreshed} != {n["node_id"] for n in plan["nodes"]}:
            raise ValueError("Node identities changed during setup")
        plan = rebuild(refreshed)
        if not any(h["action"] != "none" for h in plan["network"]["hosts"]):
            break
        review(plan)
    else:
        raise ValueError("Networking did not converge; inspect setup receipts")
    # Each function restarts under a NetworkManager profile that the passes
    # normalized (autoconnect on), so NetworkManager restores its addresses.
    record["steps"].append({"hairpin": "running"})
    deploy_engine.save_receipt(journal, record)
    outcome = {}
    current = plan
    plan = (ensure or hairpin_ring.ensure)(
        plan, approved=approved, inspect=inspect_nodes, rebuild=rebuild,
        update_workers=update_workers or (lambda: _sync_workers(current, directory)),
        directory=directory, record=outcome)
    record["steps"][-1]["hairpin"] = outcome.get("state", "complete")
    deploy_engine.save_receipt(journal, record)
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
    parser.add_argument("--allow-driver-reload", action="store_true", help=ALLOW_DRIVER_RELOAD)
    parser.add_argument("--inventory", type=Path, help="offline array of authenticated-node fixture records; planning only")
    parser.add_argument("--head-id", help="head node UUID for offline inventory")
    parser.add_argument("--output", type=Path, help="new private setup receipt directory")
    args = parser.parse_args(argv)
    if args.plan and args.apply or args.inventory and (args.apply or not args.plan):
        raise ValueError("Offline inventory requires --plan; --plan and --apply are exclusive")
    if args.yes and not args.apply and not args.plan:
        raise ValueError("Noninteractive setup changes require --apply --yes")
    if args.allow_driver_reload:
        print("--allow-driver-reload: " + ALLOW_DRIVER_RELOAD)
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
    if args.adopt:
        question = "Record this verified existing fabric without network changes"
        if len(nodes) == 4:
            question += (", and record the ConnectX hairpin setting that is in effect and apply it at every boot"
                         " (no driver restart)")
        confirm(question + "?", args.yes)
    else:
        # This answer also approves the ConnectX hairpin step that summarize() listed.
        confirm("Apply this network configuration and enable fabric/agent services?", args.yes)

    def review(value):
        summarize(value)
        confirm("Apply the refreshed plan after configuration discovery?", args.yes)

    def rebuild(found):
        return topology.build_spec(found, head, name=args.name, fabric_cidr=args.fabric_cidr)

    if args.adopt:
        observed = []
        for rank, host in enumerate(plan["spec"]["hosts"]):
            config = topology.persistent_config(plan, rank)
            config.update(ownership="observed", routes=[], forwarding=[])
            if len(nodes) == 4:
                mesh = json.loads(discovery.ssh(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "native-mesh", "--rank", str(rank)]))["mesh"]
                if not mesh or not mesh.get("active", True):
                    raise ValueError("No verified native mesh found; ordinary setup can prepare one")
                if mesh.get("problem"):
                    raise ValueError(mesh["problem"])
                order = ("cw_primary", "ccw_primary", "cw_secondary", "ccw_secondary")
                config["native_mesh"] = {"reference": mesh["reference"], "host_ip": mesh["host_ip"],
                                         "hcas": [next(p["rdma_device"] for p in host["data_interfaces"] if p["role"] == role) for role in order]}
            discovery.ssh(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "adopt"], data=json.dumps(config))
            discovery.ssh(host["host"], ["sudo", "-n", "/usr/bin/sparkring", "node", "workspace", "--operator", host["host"].split("@", 1)[0], "--name", args.name])
            observed.append({"rank": rank, "adopted": True})
        receipt = {"complete": True, "network_changed": False, "nodes": observed}
        armed = []
        if len(nodes) == 4:
            # Adoption requires an active mesh, so no function restarts here:
            # Sparks that need one get M19 and adoption still completes.
            outcome = {}
            current = plan
            plan = hairpin_ring.ensure(plan, approved=True, restart=False, inspect=collect, rebuild=rebuild,
                                       update_workers=lambda: _sync_workers(current, directory),
                                       directory=directory, record=outcome)
            receipt["hairpin"] = outcome.get("state", "complete")
            armed = (list(range(4)) if outcome.get("state") == hairpin_ring.KEPT else
                     [entry["rank"] for entry in outcome.get("ranks") or [] if entry.get("after") == hairpin_ring.KEPT])
        installer.write(directory / "setup.json", receipt)
        if armed:
            print("Existing fabric verified. No link or route changes; sparkring-hairpin.service applies the ConnectX "
                  f"hairpin setting at every boot on {hairpin_ring.ranks_text(armed)}.")
        else:
            print("Existing fabric verified. No link, route or service changes.")
    else:
        plan = apply(plan, directory, approved=True, review=review)
    # workspace() established this directory for the SSH operator, not root.
    cluster = {"schema": "sparkring-appliance-cluster/v1", "name": args.name, "plan": plan,
               "setup_receipt": str(directory / "setup.json")}
    path = STATE / "cluster.json"
    if path.exists() and installer.read(path)["plan"]["id"] != plan["id"]:
        raise ValueError("Controller already records another cluster; inspect " + str(path))
    node.save(STATE, "cluster.json", cluster, mode=0o600)
    from runtime.host import fabric_bandwidth
    # A degraded cable is a warning with its repair steps; setup never fails here.
    fabric_bandwidth.after_setup(STATE, cluster)
    print("Network configured. Choose a model: sparkring models")
    return 0


def deployment_directory(profile, instance="main"):
    """The controller's directory for one deployment of a profile; instance ``main`` adds no suffix.

    ``sparkring install`` names its deployments with instances ``i<hash>``.
    """
    import re
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,79}", profile):
        raise ValueError("Unknown profile. Run 'sparkring models' for exact model/version/topology choices.")
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", instance):
        raise ValueError("Instance must be a short lowercase name")
    return STATE / "deployments" / (profile if instance == "main" else profile + "-" + instance)


def active_deployment(*, report=True, placement=None):
    """The deployment that up last started, or that down last stopped when none was active; None if neither.

    ``placement`` selects the slot (``runtime.host.placement``): None for the
    whole cluster, a ring half otherwise. A recorded directory that no longer
    holds a deployment, for example one moved by hand, counts as none and,
    with ``report``, is noted on stderr.
    """
    from runtime.host import placement as placements
    path = placements.recorded(STATE, placement)
    if path is None:
        return None
    if (path / "deployment.lock.json").exists():
        return path
    if report:
        print(f"Note: the recorded active deployment {path} does not exist; no deployment is active.", file=sys.stderr)
    return None


def active_deployments(*, report=False):
    """``[(placement, directory)]`` of every slot's active deployment, the whole cluster first.

    ``report`` notes each recorded directory that no longer holds a deployment, as ``active_deployment`` does.
    """
    from runtime.host import placement as placements
    return [(slot, path) for slot in placements.slots(4)
            if (path := active_deployment(report=report, placement=slot)) is not None]


def lifecycle_slot(on):
    """The slot that ``sparkring up``, ``down`` or ``status`` without a profile acts on.

    ``--on`` names a half. Without it, the one slot that records a deployment;
    the whole cluster when none does. With several recorded, ValueError lists
    each and the command that names it.
    """
    from runtime.host import placement as placements
    if on:
        return placements.parse(on)
    found = active_deployments()
    if len(found) <= 1:
        return found[0][0] if found else None
    lines = [f"{placements.text(slot)}: {placements.profile_of(path)} "
             + ("(stopped)" if placements.stopped(path) else "(started)") + "; name it with "
             + (placements.flag(slot) if slot else _up_arguments(path)) for slot, path in found]
    error = ValueError("This ring records a model on more than one placement. Name one with --on 0,1, --on 2,3 or "
                       "the deployment's profile.")
    error.details = {"lines": lines}
    raise error


def existing_deployment(profile, instance="main"):
    directory = deployment_directory(profile, instance)
    if not (directory / "deployment.lock.json").exists():
        raise ValueError(f"No deployment of {profile}" + ("" if instance == "main" else f" with instance {instance}")
                         + " exists on this controller")
    return directory


def model_site(cluster, profile, instance="main", placement=None):
    """The raw site of a deployment of ``profile`` on the cluster's Sparks.

    Without ``placement`` the site holds every Spark, each at port 0's primary
    fabric function. With a placement (``runtime.host.placement``) it holds
    the two Sparks of that ring half, each at the port functions facing its
    partner, records the placement, and keeps Node A as the controller. The
    API address is Node A's recorded LAN address, or for half (2, 3) the one
    ``placement.api_address`` gives.
    """
    from runtime.host import placement as placements
    plan = cluster["plan"]
    rows = []
    identities = plan.get("nodes", [])
    ranks = list(range(len(plan["spec"]["hosts"]))) if placement is None else list(placement)
    fabric = placements.fabric_rows(cluster, placement) if placement is not None else None
    for index, rank in enumerate(ranks):
        host = plan["spec"]["hosts"][rank]
        if fabric is None:
            port = next(p for p in host["data_interfaces"] if p["role"] == "cw_primary")
            rows.append({"host": host["host"], "management_ip": host["management_address"],
                         "fabric_ip": str(ipaddress.ip_interface(port["address"]).ip), "interface": port["netdev"]})
        else:
            rows.append(dict(fabric[index]))
        if len(identities) == len(plan["spec"]["hosts"]) and isinstance(identities[rank], dict) and identities[rank].get("node_id"):
            rows[-1]["node_id"] = identities[rank]["node_id"]
    import re
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,19}", instance):
        raise ValueError("Instance must be a short lowercase name")
    identity = profile if instance == "main" else profile + "-" + instance
    name = cluster["name"][:12] + "-" + profile[:18] + "-" + hashlib.sha256(identity.encode()).hexdigest()[:6]
    result = {"schema": "sparkring-install-site/v1", "name": name,
              "workspace": "/srv/sparkring/" + cluster["name"] + "/" + identity,
              "hosts": rows, "controller_address": plan["spec"]["hosts"][0]["management_address"]}
    address = cluster.get("api_address")
    if placement is not None and placement[0] != 0:
        address, _ = placements.api_address(cluster, placement)
    if address:
        result["api_address"] = address
    if placement is not None:
        result["placement"] = list(placement)
    return result


def _hairpin_problem():
    """M6 lines when a Spark of the recorded four-Spark ring lacks the ConnectX hairpin setting, else None.

    One line per Spark, then the remedy; later lines are indented for the
    terminal.
    """
    if not (STATE / "cluster.json").exists():
        return None
    plan = installer.read(STATE / "cluster.json").get("plan") or {}
    hosts = (plan.get("spec") or {}).get("hosts") or []
    if len(hosts) != 4:
        return None
    problem = hairpin_ring.not_in_effect(plan, hairpin_ring.read_statuses(plan, invoke=discovery.ssh))
    return problem.replace("\n", "\n  ") if problem else None


def lifecycle(argv):
    parser = argparse.ArgumentParser(prog="sparkring " + argv[0])
    parser.add_argument("operation", choices=("up", "down", "status"))
    parser.add_argument("profile", nargs="?", help="exact profile shown by sparkring models")
    parser.add_argument("--model-path", help="serve this complete copy read-only on every rank; it is verified, never changed")
    images = parser.add_mutually_exclusive_group()
    images.add_argument("--image", metavar="NAME", help="another installer image: a name or release tag that sparkring images lists")
    images.add_argument("--image-lock", type=Path, help="explicit source-recorded toolchain image for a separate rehearsal")
    parser.add_argument("--fresh-mesh", action="store_true", help="review replacement of an existing native mesh")
    parser.add_argument("--instance", default="main", help="separate local deployment name for a rehearsal")
    parser.add_argument("--on", metavar="RANKS",
                        help="the half of a four-Spark ring: 0,1 or 2,3; without a profile, its model; with up "
                             "PROFILE, where a two-Spark profile's deployment runs")
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--allow-loopback-bind", action="store_true",
                        help="accept a loopback --api-bind such as 127.0.0.1 for a new deployment: only programs on "
                             "Node A can then use the model")
    from runtime.common import serving
    serving.add_arguments(parser)
    args = parser.parse_args(argv)
    settings = serving.from_arguments(args)
    if settings and (args.operation != "up" or not args.profile):
        raise ValueError("Serving settings apply to up with an exact profile")
    image_runtime = None
    if (args.image or args.image_lock) and (args.operation != "up" or not args.profile):
        raise ValueError("--image and --image-lock require up with an exact profile")
    if args.image:
        from runtime.common import installer_image
        args.image_lock = installer_image.lock_path(args.image)
    if args.instance != "main" and not args.profile:
        raise ValueError("--instance names one deployment of a profile; give the profile as well")
    from runtime.host import retained_source
    cache = STATE / "retained-sources"
    if args.operation == "status":
        from runtime.host import fabric_bandwidth
        result = node.snapshot() if args.refresh else node.status()
        plan_id = None
        if (STATE / "cluster.json").exists():
            cluster = installer.read(STATE / "cluster.json")
            plan_id = cluster["plan"].get("id")
            # The saved result of the last bandwidth check; status never measures.
            result["fabric_bandwidth"] = fabric_bandwidth.summary(STATE)
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
        from runtime.host import placement as placements
        from runtime.host import recovery
        if args.profile:
            path = existing_deployment(args.profile, args.instance)
            paths = [(placements.of_directory(path), path)]
        elif args.on:
            slot = placements.parse(args.on)
            paths = [(slot, path) for path in [active_deployment(placement=slot)] if path is not None]
        else:
            paths = active_deployments(report=True)
        views = [_status_view(slot, path, args, result, cache) for slot, path in paths]
        if views:
            # The first slot's model keeps the document's single-deployment fields.
            result.update({key: views[0][key] for key in ("deployment", "recovery", "model") if key in views[0]})
        if len(views) > 1 or any(view["placement"] for view in views):
            result["slots"] = [{key: value for key, value in view.items() if key not in ("lock", "record")}
                               for view in views]
        if args.json:
            print(json.dumps(result, indent=2))
        else:
            for view in views:
                model = view.get("model")
                if model and model["state"] != "serving":
                    # What stops a model comes first, with the step that restarts it.
                    where = f"{placements.text(tuple(view['placement']))}: " if view["placement"] else ""
                    print(where + model["summary"] + " | next: " + model["next_action"])
                    for line in model["details"]:
                        print("  " + line)
            # Node A's own state; each Spark's line follows.
            print(result["state"] + " | next: " + result.get("next_action", "sparkring status"))
            if not result.get("nodes"):
                for warning in result.get("warnings", []):
                    print("  warning: " + warning)
            attention = []
            for rank, row in enumerate(result.get("nodes", [])):
                # Named as the attention line and hairpin messages name Sparks: rank and hostname.
                name = f"rank {rank} ({row.get('hostname') or row['host']})"
                label = f"{name} {row['host']}" if row.get("hostname") else name
                print(f"  {label}: " + row["state"] + (" — " + row["error"] if row.get("error") else ""))
                if row["state"] not in HEALTHY:
                    attention.append(name)
                    if row.get("next_action"):
                        print("    next: " + row["next_action"])
                for warning in row.get("warnings", []):
                    print("    warning: " + warning)
                for unit in (row.get("mesh") or {}).get("failed") or []:
                    print("    mesh: " + node.mesh_failure_text(unit))
                    if name not in attention:
                        attention.append(name)
                reason = recovery.tunnel_reason(result.get("control"), row["host"]) if rank else None
                fallback = recovery.tunnel_fallback(result.get("control"), row["host"]) if rank else None
                if reason:
                    print("    admin tunnel: " + reason.removeprefix("the admin tunnel has "))
                elif fallback:
                    print("    admin tunnel: " + fallback)
            if attention:
                print("Sparks that need attention: " + ", ".join(attention))
            if "fabric_bandwidth" in result:
                for line in fabric_bandwidth.status_lines(result["fabric_bandwidth"], plan_id):
                    print(line)
            for view in views:
                saved, lock, record, model = view["deployment"], view["lock"], view["record"], view.get("model")
                if len(views) > 1 or view["placement"]:
                    where = placements.text(tuple(view["placement"]) if view["placement"] else None)
                    print(where[0].upper() + where[1:] + ":")
                print("Saved model operation: " + saved["profile"] + " | " + saved["state"]["operation"] + (" complete" if saved["state"].get("complete") else " incomplete"))
                # A profile with one checkpoint has no checkpoint name; a derived checkpoint names its base too.
                source = f"{saved['model_repository']} @ {saved['model_revision'][:12]}"
                if saved.get("derived"):
                    source = (f"{saved['derived']['repository']} @ {saved['derived']['revision'][:12]}, "
                              f"derived from {source}")
                print("Checkpoint: " + (f"{saved['checkpoint']} ({source})" if saved["checkpoint"] else source)
                      + f" | Image: {saved['image_release']}")
                if saved.get("serving"):
                    print("Serving settings: " + ", ".join(serving.label(name, value)
                                                           for name, value in sorted(saved["serving"].items())))
                if "thinking" in saved:
                    print("Thinking: " + thinking.deployment_text(saved["thinking"]))
                print(saved["api_url"] + (f" (SparkRing's own checks use {saved['check_url']})"
                                          if saved.get("check_url") else ""))
                if saved.get("observations"):
                    print("Model containers:")
                    for row in recovery.ranks_from_observations(saved["observations"]):
                        print(f"  rank {row['rank']} {row['host']}: " + recovery.container_text(row))
                    if model and model["state"] == "serving":
                        print("Model: " + model["summary"])
                else:
                    print("Use --refresh for current model container state.")
                recorded = view.get("recovery")
                if recorded is not None:
                    for line in recovery.status_lines(None if "error" in recorded else record,
                                                      timer=recorded.get("timer_enabled"), state=saved["state"],
                                                      backend=lock.get("backend")):
                        print(line)
            print("Network observations do not qualify GPU/RDMA serving.")
        return 0
    if args.plan and args.execute:
        raise ValueError("Choose --plan or --execute")
    from runtime.host import placement as placements
    requested = placements.parse(args.on) if args.on else None
    if args.profile and requested is not None and args.operation != "up":
        raise ValueError("--on with a profile names where up places a new deployment; down takes the profile alone")
    slot = None if args.profile else lifecycle_slot(args.on)
    active = active_deployment(placement=slot)
    if args.operation == "up" and args.profile:
        instance = args.instance
        if requested is not None and instance == "main":
            # Each half's deployment of a profile needs its own directory.
            instance = placements.instance_label(requested)
        directory = deployment_directory(args.profile, instance)
        # Refused before a new deployment is created; checked again under the installation lock.
        _refuse_conflicts(requested if not directory.exists() else placements.of_directory(directory))
        if not directory.exists():
            from runtime.common import installer_image
            from runtime.host import models
            cluster = installer.read(STATE / "cluster.json")
            size = len(cluster["plan"]["nodes"])
            nodes = 2 if requested is not None else size
            profile = models.select(args.profile, nodes)
            placements.check(requested, cluster_size=size, profile_nodes=nodes, profile=profile)
            image_runtime = installer_image.for_profile(profile, installer.read(args.image_lock) if args.image_lock else None)
            site = model_site(cluster, profile, instance, requested)
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
            if {"api_port", "api_bind"} & set(settings):
                # The API Spark is checked for the listen address and the
                # port before the deployment is created, as sparkring install
                # checks it.
                from runtime.host import install_workflow
                arguments = install_workflow.profile_arguments(installer.setup.selection(profile))
                serving.apply(arguments, settings)
                install_workflow.check_endpoint(args, cluster, requested, STATE, directory, settings,
                                                settings.get("api_port") or serving.profile_value(arguments, "api_port"))
            installer.init(directory, profile, site, image_runtime=image_runtime, settings=settings)
        else:
            # The deployment's own source validates its lock (retained_source);
            # the installed package may carry other profile inputs or images.
            existing = installer.read(directory / "deployment.lock.json")
            if args.image_lock:
                from runtime.common import installer_image
                image_runtime = installer_image.for_profile(args.profile, installer.read(args.image_lock))
                if existing.get("image_runtime") != image_runtime:
                    raise ValueError("Deployment uses another image lock; choose a distinct --instance")
            if args.model_path and any(row["model"] != args.model_path or not row["reuse_verified_model"] for row in existing["site"]["ranks"]):
                raise ValueError("Deployment uses another model path; choose a distinct --instance")
            if args.fresh_mesh and "native_mesh" not in existing["site_input"]:
                raise ValueError("Deployment reuses an existing mesh; use --instance fresh --fresh-mesh for a separate rehearsal")
            if settings and (existing.get("serving") or {}) != settings:
                raise ValueError("Deployment uses other serving settings; choose a distinct --instance")
            if requested is not None and placements.from_lock(existing) != requested:
                raise ValueError(f"{directory.name} runs on {placements.text(placements.from_lock(existing))}; "
                                 "choose a distinct --instance")
    elif args.profile:
        directory = existing_deployment(args.profile, args.instance)
    elif active is not None:
        directory = active
    else:
        raise ValueError(f"No model deployment is active. To {args.operation} one, name its profile"
                         " and, for a deployment other than main, its --instance.")
    # The slot of the deployment acted on: its own placement when a profile names it.
    slot = placements.of_directory(directory) if args.profile else slot
    active = active_deployment(placement=slot, report=False) if args.profile else active
    from runtime.host import retention
    released = retention.release_record(directory) if args.operation == "down" else None
    if released:
        # Its stop would verify the containers through the workspace's source
        # checkout, which automatic release removed with the containers, also
        # where the release did not finish on every Spark.
        where = ("from the Sparks" if released.get("complete") else
                 "from some Sparks, and sudo sparkring storage lists what stays")
        message = (f"{directory.name} is stopped, and automatic release removed its containers and workspaces "
                   f"{where}; there is nothing to stop. sudo sparkring up " + _up_arguments(directory)
                   + " starts it again.")
        print(json.dumps({"operation": "down", "complete": True, "released": True, "message": message}, indent=2)
              if args.json else message)
        return 0
    result = retained_source.review(directory, args.operation, cache=cache)
    print(f"{args.operation}: {result['profile']} on " + ", ".join(result["hosts"]))
    if image_runtime is not None:
        print("Development image: " + image_runtime["name"] + " | " + image_runtime["image_id"])
    print(" -> ".join(result["phases"]))
    recorded = installer.read(directory / "deployment.lock.json").get("serving")
    if recorded:
        print("Serving settings: " + ", ".join(serving.label(name, value) for name, value in sorted(recorded.items())))
        if settings:
            # Settings named on this command line are compared with the installed profile's values.
            base = installer.specifications(dict(installer.read(directory / "deployment.lock.json"), serving={}), only_rank=0)[0].command
            for line in serving.warnings(recorded, base):
                print("Warning: " + line)
    if "native_mesh" in result:
        print("Prepare native ASIC fabric and install its supervised service.")
        for old in result["native_mesh"]["replaces"]:
            print(f"  Stop/disable rank {old['rank']} service: {old['unit']}")
    if args.plan or not args.execute and not sys.stdin.isatty():
        if args.operation == "up":
            problem = _hairpin_problem()
            if problem:
                print("Warning: " + problem)
        print("Review, then repeat with --execute.")
        return 0
    # The installation lock keeps sparkring install, setup and the hairpin
    # procedure from changing deployments or networking meanwhile.
    with process_lock.hold(STATE / "install.lock"):
        # sparkring install records its deployment active while it holds this
        # lock, so the record read before the lock may be stale. Every decision
        # below uses the record read under the lock; a command without a
        # profile, whose target is the earlier record, stops when they differ.
        held_active = active_deployment(report=False, placement=slot)
        if not args.profile and held_active != active:
            raise ValueError(f"The active deployment changed from {active} to {held_active or 'none'} after this "
                             f"command printed its steps. Review with 'sparkring {args.operation} --plan', then repeat.")
        if args.operation == "up" and held_active is not None and held_active != directory:
            previous = retained_source.apply(held_active, "saved-status", cache=cache)["state"]
            if previous.get("operation") != "down" or not previous.get("complete"):
                raise ValueError("Run sparkring down before selecting another model")
        if args.operation == "up":
            _refuse_conflicts(slot)
        if args.operation == "up" and slot is None:
            # A mesh refused by its hairpin start check would otherwise surface
            # only as a failed systemd job, so nothing starts without the setting.
            problem = _hairpin_problem()
            if problem:
                raise ValueError(problem)
        confirm("Apply these model/image actions?", args.execute)
        from runtime.host import recovery
        # Automatic recovery never takes a state this operation leaves for its
        # own, neither in this deployment nor in the active one it replaces.
        for path in {directory, active} - {None}:
            recovery.forget_attempt(path)
        if args.operation == "up" and slot is not None:
            from runtime.host import install_workflow
            install_workflow.park_ring(installer.read(STATE / "cluster.json"))
        result = retained_source.apply(directory, args.operation, cache=cache)
        from runtime.host import api_endpoint
        result = api_endpoint.present(result, api_endpoint.recorded(directory))
        # Stopping another deployment leaves the active one in place.
        if args.operation == "up" or held_active is None:
            placements.record(STATE, slot, directory)
        if args.operation == "up":
            placements.clear_conflicting(STATE, slot)
        if args.operation == "up" and result.get("complete"):
            # Resets automatic recovery's failures and records the generation's boots.
            recovery.started(directory)
            # With --json, stdout carries only the result document.
            result["retention"] = retention.after_operation(
                STATE, discovery.ssh, write=(lambda line: print(line, file=sys.stderr)) if args.json else print)
    print(json.dumps(result, indent=2) if args.json else "Model operation complete. sparkring status --refresh")
    return 0


def _refuse_conflicts(slot):
    """Refuse to start a model in ``slot`` while the conflicting slots' model runs: the whole ring and its halves never serve at once."""
    from runtime.host import placement as placements
    for other in placements.conflicting(slot):
        running = active_deployment(report=False, placement=other)
        if running is not None and not placements.stopped(running):
            raise ValueError(f"{placements.profile_of(running)} runs on {placements.text(other)}. Stop it first: "
                             "sudo sparkring down " + (placements.flag(other) if other else _up_arguments(running))
                             + " --execute")


def _status_view(slot, path, args, result, cache):
    """One deployment's part of ``sparkring status``: its saved state, recovery record and, with --refresh, model check.

    Returns ``placement``, ``deployment``, ``recovery`` (only for a slot's
    active deployment, which automatic recovery acts on), ``model`` and the
    ``lock`` and ``record`` the terminal text reads.
    """
    from runtime.host import recovery, retained_source
    view = {"placement": list(slot) if slot else None, "record": None}
    view["deployment"] = retained_source.apply(path, "status" if args.refresh else "saved-status", cache=cache)
    # Read here rather than by the retained source, whose revision may
    # predate these fields.
    lock = view["lock"] = installer.read(Path(path) / "deployment.lock.json")
    view["deployment"].update(installer.identity(lock), containers=installer.containers(lock),
                              serving=lock.get("serving") or {})
    # What a request that names no thinking argument gets (thinking.deployment_default), or None
    # without a record, also for a deployment whose checkpoint the installed package does not list.
    selection = lock.get("selection") or {}
    try:
        model = thinking.of(selection.get("profile"), selection.get("target_variant"))
    except (OSError, ValueError):
        model = None
    view["deployment"]["thinking"] = thinking.deployment_default(model, lock.get("serving"))
    try:
        record = view["record"] = recovery.record_of(recovery.load(), path)
        # Automatic recovery acts on each slot's active deployment only.
        if not args.profile or recovery.is_active(path):
            view["recovery"] = {**record, "supported": recovery.supported(lock),
                                "timer_enabled": recovery.timer_enabled()}
    except (OSError, ValueError) as error:
        view["recovery"] = {"error": str(error)}
    if args.refresh:
        # Node A's own document carries its administration tunnel report.
        nodes = result.get("nodes")
        if nodes and slot:
            nodes = [row for row in nodes if row["host"] in {item["host"] for item in lock["site"]["ranks"]}]
        model = recovery.status_assessment(view["deployment"], nodes, view["record"], tunnel=result.get("control"))
        if model is not None:
            view["model"] = model
    # The address the installation named for display replaces the API URL's
    # host; the URL that the checks above used stays as check_url.
    from runtime.host import api_endpoint
    view["deployment"] = api_endpoint.present(view["deployment"], api_endpoint.recorded(path))
    return view


def _up_arguments(directory):
    """``PROFILE [--instance NAME]`` of the deployment in ``directory``, as ``sparkring up`` names it."""
    profile = installer.read(Path(directory) / "deployment.lock.json")["selection"]["profile"]
    instance = Path(directory).name.removeprefix(profile).removeprefix("-")
    return profile + (f" --instance {instance}" if instance else "")


def _setup_lock(argv):
    """``install.lock`` for a setup that may change hosts; planning and help take no lock.

    ``sparkring install`` calls ``single_uplink.main`` inside its own hold of
    this lock, so the lock is never taken twice.
    """
    if any(flag in argv for flag in ("--plan", "--inventory", "-h", "--help")):
        return contextlib.nullcontext()
    return process_lock.hold(STATE / "install.lock")


def main(argv):
    try:
        if argv[0] != "setup":
            return lifecycle(argv)
        with _setup_lock(argv):
            if not any(flag in argv for flag in ("--node", "--inventory")):
                from runtime.host.single_uplink import main as uplink_main
                return uplink_main(argv[1:])
            return setup(argv[1:])
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("SparkRing: " + str(error), file=sys.stderr)
        for line in detail_lines(getattr(error, "details", None)):
            print("  " + line, file=sys.stderr)
        return 2
