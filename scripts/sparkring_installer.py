"""Small operator interface over locked profiles and existing lifecycle adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.common import installer  # noqa: E402
from scripts.installer_runner import Runner, discover  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(prog="sparkring")
    commands = parser.add_subparsers(dest="action", required=True)
    initialize = commands.add_parser("init", help="choose a profile and save a locked installation")
    initialize.add_argument("--site", type=Path, help="use a saved site without SSH discovery")
    initialize.add_argument("--host", action="append", help="SSH targets in rank order; read-only discovery")
    initialize.add_argument("--model", choices=("glm53", "mimo26", "qwen38"))
    initialize.add_argument("--profile", choices=sorted(installer.INSTALLABLE))
    initialize.add_argument("--variant", choices=("nvfp4-spark", "nvfp4-qad"))
    initialize.add_argument("--image-lock", type=Path, help="source-recorded external toolchain image selection")
    initialize.add_argument("--name")
    initialize.add_argument("--workspace")
    initialize.add_argument("--output", type=Path)
    for operation in ("up", "down", "status", "export"):
        command = commands.add_parser(operation)
        command.add_argument("--deployment", type=Path, default=Path(".sparkring/deployment"))
        if operation in ("up", "down"):
            command.add_argument("--execute", action="store_true", help="apply this locked deployment's printed scope")
        if operation == "status":
            command.add_argument("--refresh", action="store_true", help="read current host state over SSH")
        if operation == "export":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--share", action="store_true", help="export an example site and Compose templates, excluding private inputs")
            command.add_argument("--format", choices=("zip", "compose"), default="zip")
            command.add_argument("--profile", help="export a standalone profile without initializing a deployment")
            command.add_argument("--variant", choices=("nvfp4-spark", "nvfp4-qad"))
        command.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "init":
            if args.site and args.host:
                raise ValueError("Choose a site file or SSH discovery, not both")
            if args.profile and args.model:
                raise ValueError("Choose a model default or an explicit profile")
            if args.site:
                raw = installer.read(args.site)
                if args.name or args.workspace:
                    raise ValueError("With --site, set name/workspace in that file")
            else:
                hosts = args.host
                if hosts is None:
                    if not sys.stdin.isatty():
                        raise ValueError("Noninteractive init requires --site or ordered --host values")
                    hosts = input("SSH hosts in rank order (space separated): ").split()
                if len(hosts) not in (2, 4):
                    raise ValueError("Choose two or four Sparks")
                name = args.name or (input("Deployment name: ").strip() if sys.stdin.isatty() else "deployment")
                raw = {"schema": "sparkring-install-site/v1", "name": name,
                       "hosts": [discover(installer.host(value)) for value in hosts]}
                if args.workspace:
                    raw["workspace"] = args.workspace
            model = args.model
            if not args.profile and not model:
                if not sys.stdin.isatty():
                    raise ValueError("Choose --model glm53/mimo26/qwen38 or --profile")
                model = input("Model (glm53, mimo26 or qwen38): ").strip()
            profile = args.profile or installer.DEFAULTS[model, len(raw["hosts"])]
            output = args.output or Path(".sparkring/deployment")
            from runtime.common import installer_image
            image_runtime = installer_image.for_profile(profile, installer.read(args.image_lock) if args.image_lock else None)
            installer.init(output, profile, raw, variant=args.variant, image_runtime=image_runtime)
            print(f"Saved {profile} for {len(raw['hosts'])} ranks in {output}")
            print("No hosts changed. Next: sparkring up --deployment " + str(output))
            return 0
        if args.action == "export":
            if args.format == "compose":
                from runtime.common import standalone_compose
                if args.profile:
                    profile, variant = args.profile, args.variant
                else:
                    lock = installer.load(args.deployment)
                    if "image_runtime" in lock:
                        raise ValueError("Use ZIP export for an image-lock deployment; standalone profile export would discard its image selection")
                    card = lock["selection"]
                    profile, variant = card["profile"], card["target_variant"]
                    if args.variant is not None and args.variant != variant:
                        raise ValueError("Use --profile to select a different standalone variant")
                installer.write(args.output, standalone_compose.render(profile, variant))
                result = {"path": str(args.output), "shareable_template": True, "format": "single-compose-file"}
            else:
                if args.profile or args.variant:
                    raise ValueError("--profile/--variant select standalone Compose; add --format compose")
                result = installer.export(args.deployment, args.output, share=args.share)
        elif args.action == "status":
            result = installer.status(args.deployment)
            if args.refresh:
                runner = Runner(args.deployment)
                plan = installer.operation_plan(runner.lock, "status")
                result["observations"] = []
                for action in plan["phases"][0]["actions"]:
                    observed = runner(action["host"], action["argv"], action["timeout"])
                    if observed["returncode"]:
                        raise ValueError(action["host"] + ": " + observed["stderr"])
                    result["observations"].append(json.loads(observed["stdout"]))
                result["live_observed"] = True
        else:
            # Constructing a plan never even constructs an SSH runner.
            result = installer.apply(args.deployment, args.action,
                                     runner=Runner(args.deployment) if args.execute else None, execute=args.execute)
        if args.json:
            print(json.dumps(result, indent=2))
        elif args.action in ("up", "down") and not args.execute:
            print(f"{args.action}: {result['profile']} on " + ", ".join(result["hosts"]))
            print(" -> ".join(result["phases"]))
            print("No remote actions. Review, then repeat with --execute.")
        elif args.action == "status":
            print(result["profile"] + ": " + result["state"]["operation"] +
                  (" complete" if result["state"].get("complete") else " incomplete"))
            print(result["api_url"] + "  model=" + result["model"])
            for key, value in result.get("actions", {}).items():
                print(f"  {key}: {value}")
            for key, value in result.get("problems", {}).items():
                print(f"  {key}: {value}")
            if "failure" in result:
                print("Failure: " + json.dumps(result["failure"]))
            if "observations" in result:
                print(json.dumps(result["observations"], indent=2))
            else:
                print("Saved progress only; use --refresh for live state.")
        else:
            print(json.dumps(result, indent=2))
        return 0
    except (ValueError, KeyError, TypeError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print("Installer: " + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
