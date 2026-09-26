"""Fixed local administrative actions for the Linux appliance package."""
import argparse
import contextlib
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
    assets = commands.add_parser("assets")
    assets.add_argument("--profile", required=True)
    workspace = commands.add_parser("workspace")
    workspace.add_argument("--operator", required=True)
    workspace.add_argument("--name", required=True)
    agent = commands.add_parser("agent")
    agent.add_argument("--once", action="store_true")
    status = commands.add_parser("status")
    status.add_argument("--refresh", action="store_true")
    hairpin = commands.add_parser("hairpin", help="ConnectX hairpin setting of a four-Spark ring member")
    hairpin_actions = hairpin.add_subparsers(dest="hairpin_action", required=True)
    hairpin_actions.add_parser("approve", help="record this Spark's approval; enables and restarts nothing")
    hairpin_apply = hairpin_actions.add_parser("apply", help="run by sparkring-hairpin.service only")
    hairpin_apply.add_argument("--dry-run", action="store_true", help="print the plan as JSON and change nothing")
    hairpin_apply.add_argument("--boot", action="store_true", help="with --dry-run: plan the next boot's run")
    hairpin_status = hairpin_actions.add_parser("status", help="print sparkring-hairpin-status/v1")
    hairpin_status.add_argument("--busy", action="store_true", help="add what blocks a driver restart now")
    hairpin_require = hairpin_actions.add_parser("require", help="mesh start check run by the generated drop-in")
    hairpin_require.add_argument("--unit", required=True)
    hairpin_start = hairpin_actions.add_parser(
        "start", help="start sparkring-hairpin.service without waiting unless a newer run exists (ring procedure)")
    hairpin_start.add_argument("--after", required=True,
                               help="the unit's InvocationID before the dispatch; empty when it has not run")
    hairpin_actions.add_parser("resume", help="start the enabled mesh units that the start check refused (ring "
                                              "procedure)")
    hairpin_actions.add_parser("revoke", help="disable the boot service and remove the approval")
    args = parser.parse_args(argv)
    if args.action == "hairpin" and args.hairpin_action == "apply" and args.boot and not args.dry_run:
        parser.error("--boot requires --dry-run")
    unprivileged = ((args.action == "status" and not args.refresh)
                    or (args.action == "hairpin" and args.hairpin_action == "status" and not args.busy))
    try:
        if not unprivileged and (not hasattr(os, "geteuid") or os.geteuid() != 0):
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
        elif args.action == "hairpin":
            from runtime.host import hairpin
            if args.hairpin_action == "apply" and not args.dry_run:
                return hairpin.apply_unit()
            if args.hairpin_action == "require":
                return hairpin.require(args.unit)
            # The ring procedure parses stdout as one JSON document, so log
            # lines of these actions go to stderr.
            with contextlib.redirect_stdout(sys.stderr):
                if args.hairpin_action == "apply":
                    result = hairpin.preview(boot=args.boot)
                elif args.hairpin_action == "approve":
                    result = hairpin.approve()
                elif args.hairpin_action == "status":
                    result = hairpin.status(busy=args.busy)
                elif args.hairpin_action == "start":
                    result = hairpin.start(args.after)
                elif args.hairpin_action == "resume":
                    result = hairpin.resume()
                else:
                    result = hairpin.revoke()
        else:
            result = node.snapshot() if args.refresh else node.status()
        print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        # Hairpin messages carry their own "SparkRing hairpin: " prefix.
        text = str(error)
        print(text if text.startswith("SparkRing ") else "SparkRing node: " + text, file=sys.stderr)
        return 2
