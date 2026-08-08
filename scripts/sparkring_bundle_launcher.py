#!/usr/bin/env python3
"""Bundle launcher for multi-service SparkRing runtime compositions.

Offline conformance commands: validate, explain, diff.
Lifecycle commands: plan (offline), start/stop/rollback (mutation,
requires --execute + --confirmation), status/verify-rollback
(read-only, requires --execute, no confirmation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import sparkring_bundle as bundle_mod  # noqa: E402
import sparkring_runtime as runtime  # noqa: E402
from sparkring_site import SiteConfigError, load_site  # noqa: E402


# ---------------------------------------------------------------------------
# Helper: load bundle (bundle_dir is stored in RuntimeBundle, no document mutation)
# ---------------------------------------------------------------------------


def _load_bundle(path: Path) -> bundle_mod.RuntimeBundle:
    """Load a bundle — bundle_dir is set from the file path."""
    return bundle_mod.load_bundle(path)


# ---------------------------------------------------------------------------
# Conformance commands (offline, no SSH)
# ---------------------------------------------------------------------------


def _conformance_validate(args: Any, parser: Any) -> int:
    try:
        bundle = _load_bundle(Path(args.bundle))
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        parser.error(str(exc))
        return 2
    if args.site:
        try:
            site = load_site(args.site)
            bundle_mod.bundle_plan(bundle, site)
        except (OSError, KeyError, json.JSONDecodeError,
                SiteConfigError, bundle_mod.BundleError,
                runtime.ProfileError) as exc:
            parser.error(str(exc))
            return 2
    print(json.dumps({"valid": True, "bundle_id": bundle.bundle_id}, indent=2))
    return 0


def _conformance_explain(args: Any, parser: Any) -> int:
    try:
        bundle = _load_bundle(Path(args.bundle))
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        parser.error(str(exc))
        return 2
    site = None
    if args.site:
        try:
            site = load_site(args.site)
        except (OSError, KeyError, json.JSONDecodeError,
                SiteConfigError) as exc:
            parser.error(str(exc))
            return 2
    print(json.dumps(bundle_mod.bundle_explain(bundle, site), indent=2))
    return 0


def _conformance_diff(args: Any, parser: Any) -> int:
    try:
        if args.bundle_a and args.bundle_b:
            return _diff_bundles(args)
        if args.site_a and args.site_b:
            return _diff_plans(args)
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    parser.error("diff requires --bundle-a/--bundle-b or --site-a/--site-b")
    return 2


def _diff_bundles(args: Any) -> int:
    left = _load_bundle(Path(args.bundle_a))
    right = _load_bundle(Path(args.bundle_b))
    diffs = bundle_mod.recursive_diff(
        bundle_mod.bundle_projection(left),
        bundle_mod.bundle_projection(right),
    )
    if not diffs:
        return 0
    print(json.dumps(diffs, indent=2))
    return 1


def _diff_plans(args: Any) -> int:
    shared = args.site
    if not shared:
        # If no shared site, use individual sites
        left_site = load_site(args.site_a)
        right_site = load_site(args.site_b)
    else:
        left_site = right_site = load_site(shared)
    left = _load_bundle(Path(args.bundle_a))
    right = _load_bundle(Path(args.bundle_b))
    left_plan = bundle_mod.bundle_plan(left, left_site)
    right_plan = bundle_mod.bundle_plan(right, right_site)
    diffs = bundle_mod.recursive_diff(
        bundle_mod.plan_projection(left_plan, left, left_site),
        bundle_mod.plan_projection(right_plan, right, right_site),
    )
    if not diffs:
        return 0
    print(json.dumps(diffs, indent=2))
    return 1


# ---------------------------------------------------------------------------
# Lifecycle commands
# ---------------------------------------------------------------------------


def _lifecycle_plan(args: Any, parser: Any) -> int:
    try:
        bundle = _load_bundle(Path(args.bundle))
        site = load_site(args.site)
        bundle_mod._validate_service_ranks(bundle, site)  # noqa: SLF001
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        parser.error(str(exc))
        return 2
    plan = bundle_mod.bundle_plan(bundle, site)
    print(json.dumps(plan, indent=2))
    return 0


def _lifecycle_status(args: Any, parser: Any) -> int:
    try:
        bundle = _load_bundle(Path(args.bundle))
        site = load_site(args.site)
        bundle_mod._validate_service_ranks(bundle, site)  # noqa: SLF001
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        parser.error(str(exc))
        return 2
    _reject_bridge_execution(parser, bundle)
    # Item 7: status is read-only, no confirmation required
    if not args.execute:
        plan = bundle_mod.bundle_plan(bundle, site)
        plan["command"] = "status"
        print(json.dumps(plan, indent=2))
        print(
            "DRY RUN: status made no remote connection; add --execute",
            file=sys.stderr,
        )
        return 0
    # Item 7: aggregate results, return nonzero on failed/missing ranks
    result = bundle_mod.execute_native_status(bundle, site)
    print(json.dumps(result, indent=2))
    return 1 if result.get("status") != "ok" else 0


def _lifecycle_verify_rollback(args: Any, parser: Any) -> int:
    try:
        bundle = _load_bundle(Path(args.bundle))
        site = load_site(args.site)
        bundle_mod._validate_service_ranks(bundle, site)  # noqa: SLF001
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        parser.error(str(exc))
        return 2
    _reject_bridge_execution(parser, bundle)
    # Item 7: verify-rollback is read-only, no confirmation required
    if not args.execute:
        plan = bundle_mod.bundle_plan(bundle, site)
        plan["command"] = "verify-rollback"
        print(json.dumps(plan, indent=2))
        print(
            "DRY RUN: verify-rollback made no remote connection; add --execute",
            file=sys.stderr,
        )
        return 0
    result = bundle_mod.execute_native_verify_rollback(bundle, site)
    print(json.dumps(result, indent=2))
    status = result.get("status")
    if status == "absent":
        return 0
    elif status == "present":
        return 1
    else:
        return 2


def _lifecycle_mutation(args: Any, parser: Any, command: str) -> int:
    try:
        bundle = _load_bundle(Path(args.bundle))
        site = load_site(args.site)
        # Action builders intentionally intersect with the effective rank
        # scope. Validate membership first so no mutating command can silently
        # omit an unknown configured rank.
        bundle_mod._validate_service_ranks(bundle, site)  # noqa: SLF001
    except (OSError, KeyError, json.JSONDecodeError,
            SiteConfigError, bundle_mod.BundleError,
            runtime.ProfileError) as exc:
        parser.error(str(exc))
        return 2

    if not args.execute:
        plan = bundle_mod.bundle_plan(bundle, site)
        plan["command"] = command
        print(json.dumps(plan, indent=2))
        print(
            f"DRY RUN: {command} made no remote connection; add --execute",
            file=sys.stderr,
        )
        return 0

    _reject_bridge_execution(parser, bundle)
    _check_confirmation(parser, bundle, args)
    if command == "start":
        result = bundle_mod.execute_native_start(
            bundle, site, confirmation=args.confirmation or None,
        )
        print(json.dumps(result, indent=2))
        for phase in result.get("phases", []):
            if phase.get("failed_ranks"):
                return 1
        if result.get("rollback", {}).get("rollback_status") == "failed":
            return 1
        return 0

    if command == "stop":
        return _execute_stop(parser, bundle, site, args)
    if command == "rollback":
        return _execute_rollback_command(parser, bundle, site, args)
    return 2


def _execute_stop(
    parser: Any, bundle: bundle_mod.RuntimeBundle, site: Any, args: Any,
) -> int:
    reversed_ = bundle_mod.reverse_order(bundle.services)
    any_failed = False
    results: dict[str, Any] = {"phases": []}
    for svc in reversed_:
        if svc.profile is None and svc.structured is None:
            continue
        actions = bundle_mod._native_stop_actions(  # noqa: SLF001
            site, svc, bundle,
        )
        res = bundle_mod._safe_execute(actions, 30)  # noqa: SLF001
        failed_ranks = [
            rank for rank, r in res.items()
            if r.get("exit_code", 0) != 0
        ]
        if failed_ranks:
            any_failed = True
        results["phases"].append({
            "service_id": svc.service_id,
            "results": res,
            "failed_ranks": sorted(failed_ranks),
        })
    print(json.dumps(results, indent=2))
    return 1 if any_failed else 0


def _execute_rollback_command(
    parser: Any, bundle: bundle_mod.RuntimeBundle, site: Any, args: Any,
) -> int:
    reversed_ = bundle_mod.reverse_order(bundle.services)
    any_failed = False
    results: dict[str, Any] = {"phases": []}
    for svc in reversed_:
        if svc.profile is None and svc.structured is None:
            continue
        actions = bundle_mod._native_stop_actions(  # noqa: SLF001
            site, svc, bundle,
        )
        res = bundle_mod._safe_execute(actions, 30)  # noqa: SLF001
        failed_ranks = [
            rank for rank, r in res.items()
            if r.get("exit_code", 0) != 0
        ]
        if failed_ranks:
            any_failed = True
        results["phases"].append({
            "service_id": svc.service_id,
            "results": res,
            "failed_ranks": sorted(failed_ranks),
        })
    print(json.dumps(results, indent=2))
    return 1 if any_failed else 0


def _reject_bridge_execution(parser: Any, bundle: bundle_mod.RuntimeBundle) -> None:
    if any(
        svc.source_kind in bundle_mod.PLAN_ONLY_SOURCE_KINDS
        for svc in bundle.services
    ):
        parser.error(
            "Canonical bridge execution is not supported; "
            "use the canonical launcher"
        )


def _check_confirmation(
    parser: Any, bundle: bundle_mod.RuntimeBundle, args: Any,
) -> None:
    if bundle.confirmation and args.confirmation != bundle.confirmation:
        parser.error(
            f"execute requires --confirmation {bundle.confirmation}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        epilog=(
            "Offline conformance commands: validate checks structure and, "
            "with --site, canonical plan construction; explain reports "
            "ordering, safety, and limits; diff exits 0 for identical, "
            "1 for different, and 2 for invalid input."
        ),
    )
    parser.add_argument("--site", required=False)
    parser.add_argument("--bundle", required=False)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--confirmation", default="",
        help="confirmation token required for mutating commands "
        "(start, stop, rollback) when the bundle declares one",
    )
    parser.add_argument("--bundle-a", dest="bundle_a", default=None)
    parser.add_argument("--bundle-b", dest="bundle_b", default=None)
    parser.add_argument("--site-a", dest="site_a", default=None)
    parser.add_argument("--site-b", dest="site_b", default=None)
    parser.add_argument(
        "command",
        choices=(
            "plan", "start", "stop", "status",
            "rollback", "verify-rollback",
            "validate", "explain", "diff",
        ),
    )
    args = parser.parse_args(argv)

    if args.command in ("validate", "explain", "diff"):
        if args.execute:
            parser.error(f"{args.command} is always offline; remove --execute")
        if args.confirmation:
            parser.error(
                f"--confirmation does not apply to offline {args.command}"
            )

    if args.command == "validate":
        _require_arg(parser, args, "bundle", "validate requires --bundle")
        return _conformance_validate(args, parser)
    if args.command == "explain":
        _require_arg(parser, args, "bundle", "explain requires --bundle")
        return _conformance_explain(args, parser)
    if args.command == "diff":
        _require_arg(parser, args, "bundle_a", "diff requires --bundle-a")
        _require_arg(parser, args, "bundle_b", "diff requires --bundle-b")
        return _conformance_diff(args, parser)

    _require_arg(parser, args, "site", f"{args.command} requires --site")
    _require_arg(parser, args, "bundle", f"{args.command} requires --bundle")

    if args.command == "plan":
        if args.execute:
            parser.error("plan is always offline; remove --execute")
        if args.confirmation:
            parser.error("--confirmation does not apply to plan")
        return _lifecycle_plan(args, parser)
    if args.command == "status":
        # Item 7: status is read-only, no confirmation
        if args.confirmation:
            parser.error("--confirmation does not apply to status")
        return _lifecycle_status(args, parser)
    if args.command == "verify-rollback":
        # Item 7: verify-rollback is read-only, no confirmation
        if args.confirmation:
            parser.error("--confirmation does not apply to verify-rollback")
        return _lifecycle_verify_rollback(args, parser)
    if args.command in ("start", "stop", "rollback"):
        return _lifecycle_mutation(args, parser, args.command)

    parser.error(f"unknown command: {args.command}")
    return 2


def _require_arg(parser: Any, args: Any, arg_name: str, msg: str) -> None:
    if not getattr(args, arg_name):
        parser.error(msg)


if __name__ == "__main__":
    raise SystemExit(main())
