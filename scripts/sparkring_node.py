"""Fixed local administrative actions for the Linux appliance package."""
import argparse
import json
import os
import subprocess
import sys

from runtime.host import node


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring node")
    commands = parser.add_subparsers(dest="action", required=True)
    for name in ("initialize", "verify", "configure", "adopt", "restore", "control-key", "control-configure", "control-up"):
        commands.add_parser(name)
    seed = commands.add_parser("seed")
    seed.add_argument("--key-file", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--rank", type=int, required=True)
    inspect.add_argument("--target", required=True)
    inspect.add_argument("--management", required=True)
    inspect.add_argument("--witness", required=True)
    native = commands.add_parser("native-mesh")
    native.add_argument("--rank", type=int, required=True, choices=range(4))
    assets = commands.add_parser("assets", help="survey this Spark for the profile's pinned checkpoint and print "
                                 "where copies are (read-only)")
    assets.add_argument("--profile", required=True)
    checkpoints = commands.add_parser("checkpoints", help="list this Spark's SparkRing checkpoint directories, "
                                      "or release one; Node A sends the request on stdin")
    checkpoints.add_argument("--release", metavar="PATH")
    workspace = commands.add_parser("workspace")
    workspace.add_argument("--operator", required=True)
    workspace.add_argument("--name", required=True)
    agent = commands.add_parser("agent")
    agent.add_argument("--once", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    try:
        if not (args.action == "status" and not args.refresh) and (not hasattr(os, "geteuid") or os.geteuid() != 0):
            raise ValueError("This local administrative action requires sudo")
        if args.action == "initialize":
            result = node.initialize()
        elif args.action.startswith("control-"):
            from runtime.host import control_node
            if args.action == "control-key":
                result = control_node.public_key()
            elif args.action == "control-configure":
                result = control_node.configure(json.load(sys.stdin))
            else:
                result = control_node.up()
        elif args.action == "seed":
            from pathlib import Path
            from runtime.host.seed import prepare
            print("Prepare unused fabric interfaces and enable Node A's SSH key on the preparation service.", file=sys.stderr)
            if input("Continue? [y/N]: ").strip().lower() not in ("y", "yes"):
                raise ValueError("Worker preparation cancelled")
            from runtime.host.controller import confirm
            result = prepare(Path(args.key_file).read_text(),
                             stop=lambda names: confirm("Stop these running GPU containers? They are stopped, not removed: "
                                                        + ", ".join(names) + "."),
                             link_local=lambda name: confirm("Add IPv6 link-local addressing to fabric connection " + name
                                                             + "? Its IPv4 addresses and MTU are kept."))
        elif args.action == "inspect":
            result = node.inspect(args.rank, args.target, args.management, args.witness)
        elif args.action == "native-mesh":
            from runtime.host.native_mesh import inspect_local
            result = inspect_local(args.rank)
        elif args.action == "assets":
            from runtime.host.assets import discover
            result = discover(args.profile)
        elif args.action == "checkpoints":
            from runtime.host import checkpoints
            result = checkpoints.node(args.release)
        elif args.action == "workspace":
            result = node.workspace(args.operator, args.name)
        elif args.action == "restore":
            result = node.restore(node.read("/", "/etc/sparkring/fabric.json"))
        elif args.action in ("verify", "configure", "adopt"):
            config = json.load(sys.stdin)
            if args.action == "configure":
                result = node.configure(config)
            elif args.action == "adopt":
                result = node.adopt(config)
            else:
                node.observe(config)
                result = {"verified": True, "hardware_qualified": False}
        elif args.action == "agent":
            node.agent(once=args.once)
            return 0
        else:
            result = node.snapshot() if args.refresh else node.status()
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("SparkRing node: " + str(error), file=sys.stderr)
        return 2
