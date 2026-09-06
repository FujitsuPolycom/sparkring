"""Standalone SparkRing discovery, preparation plans, and managed model operations."""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import ipaddress
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .deploy_engine import CommandRunner, execute_plan, plan_digest
    from .deploy_inventory import (
        collect_command,
        validate_inventory,
        summarise_inventory,
    )
    from .deploy_network import plan_network, verify_network
except ImportError:
    from deploy_engine import CommandRunner, execute_plan, plan_digest
    from deploy_inventory import (
        collect_command,
        validate_inventory,
        summarise_inventory,
    )
    from deploy_network import plan_network, verify_network

PROFILE = ROOT / "runtime/glm53-spark-mtp3-mesh"
ROLES = ("cw_primary", "cw_secondary", "ccw_primary", "ccw_secondary")
RDMA = ("rocep1s0f0", "roceP2p1s0f0", "rocep1s0f1", "roceP2p1s0f1")


def write_new(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    path.chmod(0o600)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def discover(nodes, controller_address, run=None):
    """Read four hosts through an injected runner; never enroll keys or modify hosts."""
    if len(nodes) != 4 or len(set(nodes)) != 4:
        raise ValueError(
            "Supply exactly four distinct HOST=MANAGEMENT_IP nodes in rank order"
        )
    ipaddress.IPv4Address(controller_address)
    requests = []
    for rank, node in enumerate(nodes):
        host, separator, address = node.partition("=")
        if not separator:
            raise ValueError(
                "Node must be HOST=MANAGEMENT_IP; use an enrolled SSH alias or user@host"
            )
        argv = collect_command(
            rank=rank,
            ssh_target=host,
            management_address=address,
            controller_address=controller_address,
        )
        requests.append((host, argv))
    if len({h for h, _ in requests}) != 4:
        raise ValueError("SSH targets must be distinct")
    runner = run or CommandRunner()

    def probe(item):
        host, argv = item
        result = runner(host, argv, 120)
        if result["returncode"] or result.get("uncertain"):
            raise ValueError(f"{host}: discovery failed: {result['stderr']}")
        facts = validate_inventory(json.loads(result["stdout"]))
        if facts["ssh_target"] != host:
            raise ValueError("Probe host identity differs from request")
        return host, facts

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        hosts = dict(pool.map(probe, requests))
    return {
        "schema": "sparkring-deploy-inventory/v1",
        "controller_address": controller_address,
        "hosts": hosts,
    }


def create_spec(inventory, name, workspace, fabric_range="198.18.0.0/21"):
    """Derive network and profile inputs from host facts and the documented cable cycle."""
    if (
        inventory.get("schema") != "sparkring-deploy-inventory/v1"
        or len(inventory.get("hosts", {})) != 4
    ):
        raise ValueError("Expected a four-host deployment inventory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,40}", name):
        raise ValueError(
            "Deployment name must be a simple name of at most 41 characters"
        )
    if not re.fullmatch(r"/srv/sparkring/[A-Za-z0-9_-]+", workspace):
        raise ValueError(
            "Workspace must be one dedicated /srv/sparkring/NAME directory"
        )
    network = ipaddress.ip_network(fabric_range)
    if network.version != 4 or network.prefixlen != 21:
        raise ValueError(
            "Fabric range must be an IPv4 /21 for eight separate /24 cable functions"
        )
    subnets = list(network.subnets(new_prefix=24))
    facts = sorted(inventory["hosts"].values(), key=lambda item: item["rank"])
    if [f["rank"] for f in facts] != list(range(4)):
        raise ValueError("Inventory must contain ranks 0 through 3 exactly once")
    hosts = []
    template = read(PROFILE / "fabric.example.json")
    for fact in facts:
        validate_inventory(fact, require_ready=True)
        rank = fact["rank"]
        mapping = {f["device"]: f for f in fact["rdma"]}
        if not set(RDMA) <= mapping.keys():
            raise ValueError(
                "This mesh profile requires the documented four ConnectX function roles"
            )
        interfaces = {i["name"]: i for i in fact["interfaces"]}
        host = {
            "rank": rank,
            "host": fact["ssh_target"],
            "management_address": fact["management"]["address"],
            "management_netdev": fact["management"]["interface"],
            "backup_dir": f"/var/lib/sparkring/network-backup/{name}/rank-{rank}",
            "data_interfaces": [],
        }
        fabric_rank = template["ranks"][rank]
        fabric_rank["ssh_alias"] = host["host"]
        fabric_rank["management_netdev"] = host["management_netdev"]
        for index, (role, device) in enumerate(zip(ROLES, RDMA, strict=True)):
            clockwise, function = index < 2, index % 2
            edge = rank if clockwise else (rank - 1) % 4
            address = str(
                subnets[edge + function * 4].network_address + (1 if clockwise else 2)
            )
            netdev = mapping[device]["netdev"]
            host["data_interfaces"].append(
                {
                    "role": role,
                    "rdma_device": device,
                    "netdev": netdev,
                    "address": address + "/24",
                }
            )
            port = fabric_rank["ports"][
                "clockwise" if clockwise else "counter_clockwise"
            ][function]
            port.update(
                netdev=netdev,
                rdma_device=device,
                ipv4_cidr=address + "/32",
                mac=interfaces[netdev]["mac"],
            )
        hosts.append(host)
    pins = read(PROFILE / "pins.json")
    model = f"{workspace}/models/{pins['target']['revision']}"
    site = {
        "schema": "sparkring-glm53-mtp3-mesh-site/v1",
        "topology_file": "fabric.json",
        "management_addresses": [h["management_address"] for h in hosts],
        "model_roots": [model] * 4,
        "cache_roots": [workspace + "/cache"] * 4,
        "bundle_root": workspace + "/artifacts/mtp3-mesh-bundle",
        "container_prefix": name,
        "marker_binary": workspace + "/artifacts/mlx5-rdma-tx-marker",
        "marker_binary_sha256": read(PROFILE / "image-receipt.json")["inside_image"][
            "marker_binary_sha256"
        ],
        "state_root": "/run/sparkring-mtp3-mesh",
    }
    return {
        "schema": "sparkring-deploy-spec/v1",
        "owner": name,
        "workspace": workspace,
        "controller_address": inventory["controller_address"],
        "network": {
            "backend": "NetworkManager",
            "mtu": 9000,
            "gid_index": 3,
            "hairpin_num_queues": 4,
            "hairpin_queue_size": 1024,
        },
        "hosts": hosts,
        "site": site,
        "fabric": template,
        "profile": "glm53-spark-mtp3-mesh",
    }


def lifecycle_capabilities(profile=PROFILE):
    """Advertise memory preparation only when the staged source implements it."""
    names = {"memory-idle", "memory-prepare", "memory-check"}
    paths = [profile / "managed_service.py", profile / "managed_cluster.py"]
    advertised = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        strings = {
            n.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        }
        advertised.append(names & strings)
    if any(advertised) or (profile / "managed_memory.py").exists():
        if (
            any(values != names for values in advertised)
            or not (profile / "managed_memory.py").is_file()
        ):
            raise ValueError("Incomplete managed memory-preparation source contract")
        return sorted(names)
    return []


def check_network(preparation, run=None):
    """Refresh all four inventories before recording a usable network configuration."""
    spec = preparation["spec"]
    facts = discover(
        [h["host"] + "=" + h["management_address"] for h in spec["hosts"]],
        spec["controller_address"],
        run,
    )
    result = verify_network(spec, facts["hosts"])
    result["spec_sha256"] = plan_digest(spec)
    return {**preparation, "network_verification": result}


def require_verified_network(preparation):
    verification = preparation.get("network_verification", {})
    if verification.get("ready") is not True or verification.get(
        "spec_sha256"
    ) != plan_digest(preparation["spec"]):
        raise ValueError(
            "Run network-check for this exact preparation before staging or creating containers"
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("discover", help="read enrolled hosts; never changes them")
    d.add_argument(
        "--node", action="append", required=True, metavar="HOST=MANAGEMENT_IP"
    )
    d.add_argument("--controller-address", required=True)
    d.add_argument("--output", type=Path, required=True)
    p = sub.add_parser(
        "plan", help="build host and model preparation inputs from saved inventory"
    )
    p.add_argument("--inventory", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--fabric-range", default="198.18.0.0/21")
    p.add_argument("--output", type=Path, required=True)
    n = sub.add_parser(
        "network-plan", help="plan network changes from saved host facts"
    )
    n.add_argument("--preparation", type=Path, required=True)
    n.add_argument("--inventory", type=Path, required=True)
    n.add_argument("--output", type=Path, required=True)
    n = sub.add_parser(
        "network-check", help="read all hosts and verify data-network settings"
    )
    n.add_argument("--preparation", type=Path, required=True)
    n.add_argument("--output", type=Path, required=True)
    s = sub.add_parser(
        "stage", help="download and distribute runtime files; never starts a model"
    )
    s.add_argument("--preparation", type=Path, required=True)
    s.add_argument("--state", type=Path, required=True)
    s.add_argument("--execute", action="store_true")
    r = sub.add_parser(
        "runtime-plan", help="plan stopped-container, mesh, or model operations"
    )
    r.add_argument(
        "action",
        choices=(
            "create",
            "install",
            "up",
            "start",
            "ready",
            "native-check",
            "stop",
            "recover",
            "down",
            "status",
            "logs",
        ),
    )
    r.add_argument("--preparation", type=Path, required=True)
    r.add_argument("--output", type=Path, required=True)
    a = sub.add_parser("apply-plan", help="execute an exact reviewed action plan")
    a.add_argument("--plan", type=Path, required=True)
    a.add_argument("--receipt", type=Path, required=True)
    a.add_argument("--approve-sha256", required=True)
    a.add_argument("--resume", action="store_true")
    a.add_argument("--allow-driver-reload", action="store_true")
    a.add_argument("--allow-model-actions", action="store_true")
    a.add_argument("--allow-hardware-tests", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            result = discover(args.node, args.controller_address)
            write_new(args.output, result)
            for item in result["hosts"].values():
                print(summarise_inventory(item))
        elif args.command == "plan":
            inventory = read(args.inventory)
            spec = create_spec(inventory, args.name, args.workspace, args.fabric_range)
            network = plan_network(spec, inventory["hosts"])
            result = {
                "schema": "sparkring-deploy-preparation/v1",
                "spec": spec,
                "network_plan": network,
                "model_started": False,
                "lifecycle_capabilities": lifecycle_capabilities(),
            }
            write_new(args.output, result)
            print(f"Preparation plan written to {args.output}; no host changed.")
        elif args.command == "network-plan":
            from scripts.deploy_network_run import build_network_plan

            result = build_network_plan(read(args.preparation), read(args.inventory))
            write_new(args.output, result)
            print(f"Network plan SHA-256: {result['sha256']}. No host changed.")
        elif args.command == "network-check":
            result = check_network(read(args.preparation))
            write_new(args.output, result)
            print(
                "Network settings match on four hosts. RDMA traffic is not tested by this check."
            )
        elif args.command == "stage":
            if not args.execute:
                print(
                    "Plan: download one pinned image/model, verify copies on four hosts, prepare shared secrets and launch files. Add --execute to stage; no model will start."
                )
            else:
                from scripts.deploy_stage import stage

                stage(read(args.preparation), args.state)
                print(
                    f"Staged runtime; preparation: {args.state / 'prepared.json'}. No model started."
                )
        elif args.command == "runtime-plan":
            from scripts.deploy_runtime import build_runtime_plan

            result = build_runtime_plan(read(args.preparation), args.action)
            write_new(args.output, result)
            print(f"Runtime plan SHA-256: {result['sha256']}. No host changed.")
        else:
            result = execute_plan(
                read(args.plan),
                args.receipt,
                args.approve_sha256,
                resume=args.resume,
                allow_driver_reload=args.allow_driver_reload,
                allow_model_actions=args.allow_model_actions,
                allow_hardware_tests=args.allow_hardware_tests,
            )
            if read(args.plan).get("rediscover_required"):
                print(
                    "Rediscover hosts, regenerate the network plan, then run network-check before staging."
                )
            print(
                json.dumps(
                    {"complete": result["complete"], "receipt": str(args.receipt)}
                )
            )
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"Deployment error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
