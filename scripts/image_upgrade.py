"""Plan or execute a bounded, policy-approved image upgrade experiment."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.images.upgrades.contracts import Refused, load_policy, require  # noqa: E402
from runtime.images.upgrades.agent import FileAgent  # noqa: E402
from runtime.images.upgrades.demo import trial  # noqa: E402
from runtime.images.upgrades.discovery import discover_arm64  # noqa: E402
from runtime.images.upgrades.recipes import initialize  # noqa: E402
from runtime.images.upgrades.runner import resolve_uncertain, run  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    init = sub.add_parser("init-r37")
    init.add_argument("--output", required=True, type=Path)
    init.add_argument("--agent-endpoint")
    init.add_argument("--agent-model")
    validate = sub.add_parser("validate")
    validate.add_argument("--policy", required=True, type=Path)
    demo = sub.add_parser("trial")
    demo.add_argument("--output", required=True, type=Path)
    sub.add_parser("discover-arm64")
    for name in ("run", "loop"):
        command = sub.add_parser(name)
        command.add_argument("--policy", required=True, type=Path)
        command.add_argument("--state", required=True, type=Path)
        command.add_argument("--execute", action="store_true")
        command.add_argument("--approved-policy")
        command.add_argument("--build", action="store_true")
        command.add_argument("--publish", action="store_true")
        command.add_argument("--builder-lease", type=Path)
        command.add_argument("--hardware-lease", type=Path)
        command.add_argument(
            "--force",
            action="store_true",
            help="Repeat known inputs; never bypass policy or uncertainty checks",
        )
        command.add_argument(
            "--proposal-dir",
            type=Path,
            help="Read request-bound patch JSON instead of calling an agent endpoint",
        )
        if name == "loop":
            command.add_argument("--nights", type=int, default=3)
            command.add_argument("--interval-hours", type=float, default=24)
    status = sub.add_parser("status")
    status.add_argument("--state", required=True, type=Path)
    resolution = sub.add_parser("resolve")
    resolution.add_argument("--state", required=True, type=Path)
    resolution.add_argument("--run-id", required=True)
    resolution.add_argument("--confirm-owned-work-stopped", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "init-r37":
            result = initialize(
                args.output, endpoint=args.agent_endpoint, model=args.agent_model
            )
        elif args.action == "trial":
            result = trial(args.output)
        elif args.action == "discover-arm64":
            result = discover_arm64()
        elif args.action == "validate":
            policy = load_policy(args.policy)
            result = {
                "policy_sha256": policy["_digest"],
                "sources": [s["id"] for s in policy["sources"]],
            }
        elif args.action == "status":
            result = json.loads((args.state / "state.json").read_text())
        elif args.action == "resolve":
            require(
                args.confirm_owned_work_stopped,
                "Inspect every owned external job before acknowledging resolution",
            )
            resolve_uncertain(args.state, args.run_id)
            result = {"resolved_run": args.run_id}
        else:
            require(
                not (args.build or args.publish) or args.execute,
                "Build/publication require --execute",
            )
            require(
                not args.publish or args.build,
                "Publication requires a verified build in the same run",
            )
            policy = load_policy(args.policy)
            require(
                not args.execute or args.approved_policy == policy["_digest"],
                "Execution requires --approved-policy with the exact validated digest",
            )
            count = args.nights if args.action == "loop" else 1
            require(1 <= count <= 31, "Select one to 31 nightly runs")
            if args.action == "loop":
                require(
                    0 < args.interval_hours <= 168,
                    "Interval must be positive and at most one week",
                )
            reports = []
            for number in range(count):
                if number:
                    time.sleep(args.interval_hours * 3600)
                require(
                    not args.execute
                    or load_policy(args.policy)["_digest"] == args.approved_policy,
                    "Approved policy changed between nights",
                )
                report = run(
                    args.policy,
                    args.state,
                    execute=args.execute,
                    build=args.build,
                    publish=args.publish,
                    builder_lease=args.builder_lease,
                    hardware_lease=args.hardware_lease,
                    force=args.force,
                    agent=FileAgent(args.proposal_dir) if args.proposal_dir else None,
                )
                reports.append(report)
                print(
                    json.dumps(
                        {
                            "run_id": report["run_id"],
                            "status": report["status"],
                            "reason": report.get("reason"),
                            "artifacts": report["artifact_directory"],
                        }
                    ),
                    flush=True,
                )
                if report["status"] == "uncertain":
                    break
            return (
                2
                if any(r["status"] in ("blocked", "uncertain") for r in reports)
                else 0
            )
        print(json.dumps(result, indent=2))
        return 0
    except (Refused, OSError, ValueError, KeyError) as error:
        print("Upgrade refused: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
