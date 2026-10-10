"""``python -m sparkring_sircl.fabric <command>``: the relay plan installer.

Commands and their safety classes (``RUNBOOK.md``, section "Relay plan
installer"):

- ``plan`` (OFFLINE): a layout's relay plan, from a facts file or with
  placeholders for addresses and MACs;
- ``facts`` (READ-ONLY REMOTE): the Sparks' fabric addresses and MACs as a
  facts file for ``plan``;
- ``show`` (READ-ONLY REMOTE): the relay objects every Spark holds: routes,
  neighbours, marker processes and relay filters, labelled by group;
- ``diff`` (READ-ONLY REMOTE): the Sparks compared with a layout's plan;
- ``up`` and ``down``: without ``--apply`` a dry run (READ-ONLY REMOTE) that
  prints every change and the exact script per Spark; with ``--apply``
  (MUTATES HOST) the scripts run on the selected groups' Sparks, which are
  then read again and compared with the plan to verify;
- ``marker``: the marker executable on every Spark (READ-ONLY REMOTE); with
  ``--apply`` it is compiled and installed where it is missing (MUTATES HOST).

Exit codes: 0 done (``diff``: no host changes); 1 ``diff`` found host
changes, a Spark was unreachable, or applying or verifying failed; 2 invalid
input; 3 blockers stopped a dry run or ``--apply``, or a group cannot be
planned.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .. import routes
from ..ring.site import Site, SiteError
from . import commands, layouts, ops, report
from . import plan as plan_mod
from .layouts import FabricError

MARKER_SOURCE = Path(__file__).with_name("mesh_marker.c")


def _positions(text: str | None, size: int) -> list[int]:
    if text is None or text.strip() == "all":
        return list(range(size))
    return sorted({position for group in layouts.parse_groups(text, size) for position in group})


def _layout(args, ring_size: int) -> layouts.FabricLayout:
    return layouts.resolve(ring_size=ring_size, layout=args.layout, groups=args.groups, name=args.name,
                           relay_egress=args.relay_egress)


def _print_outcome(outcome: ops.Outcome, args, *, scripts: bool) -> None:
    print(report.render_outcome(outcome, verbose=args.verbose, scripts=scripts))
    print(report.summary_line(outcome))


def _route_check(outcome: ops.Outcome, site: Site, executor: ops.Executor, *, installed_only: bool) -> int:
    """``ip route get`` of every lane destination on its origin Spark, as the harness and launcher check it."""
    if outcome.mode != "up" or not outcome.plans:
        return 0
    problems = ops.check_lane_routes(site, outcome, executor, installed_only=installed_only)
    checked = sum(len(found) for found in ops.lane_route_checks(outcome, installed_only=installed_only).values())
    for position, found in sorted(problems.items()):
        for problem in found:
            print(f"ROUTE CHECK {site.host(position).name}: {problem}")
    count = sum(len(found) for found in problems.values())
    print(f"route check: {checked - count} of {checked} lane destination(s) resolve over their lane's device"
          + (" (destinations whose origin route is still to be added are not checked)" if installed_only else ""))
    return count


def _apply(outcome: ops.Outcome, site: Site, executor: ops.Executor, args, marker: commands.MarkerConfig) -> int:
    if outcome.blocked:
        print("apply refused: resolve the blockers above (nothing was changed)")
        return 3
    if not outcome.pending:
        print("nothing to apply: the Sparks already hold the plan")
        return 0
    results = ops.apply(outcome, site, executor)
    failed = False
    for position, result in sorted(results.items()):
        text = (result.stdout + result.stderr).strip()
        print(f"{site.host(position).name}: exit {result.returncode}" + (f": {text[-1500:]}" if text else ""))
        failed |= not result.ok
    positions = [position for group in outcome.groups for position in group.order]
    states = ops.gather(site, positions, executor, marker)
    verify = ops.evaluate(site, outcome.layout, outcome.groups, states, mode=outcome.mode, marker=marker,
                          adopt=args.adopt, check_hostname=not args.skip_hostname_check, max_relays=args.max_relays)
    leftovers = [(d, change) for d in verify.diffs for change in d.pending]
    for d, change in leftovers:
        print(f"after apply, {d.name} still differs: {change.text()}")
    for d in verify.diffs:
        for blocker in d.blockers:
            print(f"after apply, {d.name}: {blocker}")
    route_problems = _route_check(verify, site, executor, installed_only=False)
    if failed or leftovers or verify.blocked or route_problems:
        print("verification failed: run `diff` (or `show`) and inspect the Sparks named above")
        return 1
    print(f"verified: {len(verify.diffs)} Spark(s) hold the {outcome.mode} state of "
          + ", ".join(group.label for group in outcome.groups))
    return 0


def _marker_command(site: Site, positions: list[int], executor: ops.Executor, args,
                    marker: commands.MarkerConfig) -> int:
    states = ops.gather(site, positions, executor, marker)
    print(report.render_marker_status(states, marker))
    targets = [p for p in positions if states[p].reachable and (args.rebuild or states[p].marker_binary is None)]
    if not targets:
        print("every reachable Spark has the marker executable")
        return 0 if all(states[p].reachable for p in positions) else 1
    source = MARKER_SOURCE.read_text(encoding="utf-8")
    scripts = {p: commands.marker_build_script(site.host(p).name, marker, source) for p in targets}
    names = ", ".join(site.host(p).name for p in targets)
    if not args.apply:
        print(f"dry run: --apply compiles {MARKER_SOURCE.name} and installs {marker.path} on {names}")
        if args.scripts:
            print(scripts[targets[0]])
        return 0
    failed = False
    for position in targets:
        result = executor(site.host(position).ssh, scripts[position], ops.APPLY_TIMEOUT)
        text = (result.stdout + result.stderr).strip()
        print(f"{site.host(position).name}: exit {result.returncode}: {text[-800:]}")
        failed |= not result.ok
    return 1 if failed else 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.fabric", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands_parser = parser.add_subparsers(dest="command", required=True)

    def layout_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--layout", help="ring8 (universal relay table), 2xTP4, 4xTP2, or ring<N>")
        sub.add_argument("--groups", help="custom groups instead of --layout: '0-3;4-7', '7,0,1,2' or '7-2'")
        sub.add_argument("--name", help="name of a custom layout (recorded on the Sparks)")
        sub.add_argument("--group", help="act on one group of the layout: index, label (path:4-5-6-7) or members")
        sub.add_argument("--relay-egress", choices=layouts.RELAY_EGRESS,
                         help="relay egress function (default: same for the whole ring, sibling otherwise)")
        sub.add_argument("--max-relays", type=int, default=routes.DEFAULT_MAX_RELAYS,
                         help=f"relay limit per lane (default {routes.DEFAULT_MAX_RELAYS}, the qualified limit; "
                              "higher values are research-only)")

    def remote_options(sub: argparse.ArgumentParser) -> None:
        sub.add_argument("--site", required=True, type=Path, help="site file (sircl-ring-site/v1)")
        sub.add_argument("--marker", default=commands.DEFAULT_MARKER,
                         help=f"marker executable on the Sparks (default {commands.DEFAULT_MARKER})")
        sub.add_argument("--skip-hostname-check", action="store_true",
                         help="accept a host whose name differs from its site entry's name")

    plan_parser = commands_parser.add_parser("plan", help="print a layout's relay plan (OFFLINE)")
    layout_options(plan_parser)
    plan_parser.add_argument("--facts", type=Path, help="facts file (sparkring-fabric-facts/v1); else placeholders")
    plan_parser.add_argument("--site", type=Path, help="site file, for Spark names and the ring size")
    plan_parser.add_argument("--ring-size", type=int, help="ring size without a site or facts file (default 8)")
    plan_parser.add_argument("--json", action="store_true")
    plan_parser.add_argument("--brief", action="store_true", help="groups and route maps only")

    facts_parser = commands_parser.add_parser("facts", help="read the Sparks' fabric addresses and MACs")
    remote_options(facts_parser)
    facts_parser.add_argument("--sparks", help="positions to read (default all), e.g. '0-3'")
    facts_parser.add_argument("--output", type=Path, help="write the facts file here instead of printing it")

    show_parser = commands_parser.add_parser("show", help="show every Spark's relay objects (READ-ONLY REMOTE)")
    remote_options(show_parser)
    show_parser.add_argument("--sparks", help="positions to read (default all), e.g. '0-3'")
    show_parser.add_argument("--raw", action="store_true", help="also print the raw read-script output")

    for name, text in (("diff", "compare the Sparks with a layout's plan (READ-ONLY REMOTE)"),
                       ("up", "install a layout's plan: dry run unless --apply (MUTATES HOST)"),
                       ("down", "remove a layout's groups: dry run unless --apply (MUTATES HOST)")):
        sub = commands_parser.add_parser(name, help=text)
        remote_options(sub)
        layout_options(sub)
        sub.add_argument("--adopt", action="store_true",
                         help="take over relay routes and neighbours that lack the ownership mark")
        sub.add_argument("--verbose", action="store_true", help="also list unchanged objects")
        if name == "diff":
            sub.add_argument("--scripts", action="store_true", help="print the scripts `up` would run")
        else:
            sub.add_argument("--apply", action="store_true", help="change the Sparks (MUTATES HOST)")
            sub.add_argument("--no-scripts", action="store_true", help="omit the per-Spark scripts")

    marker_parser = commands_parser.add_parser("marker", help="the marker executable: status, or build with --apply")
    remote_options(marker_parser)
    marker_parser.add_argument("--sparks", help="positions (default all)")
    marker_parser.add_argument("--apply", action="store_true", help="compile and install it (MUTATES HOST)")
    marker_parser.add_argument("--rebuild", action="store_true", help="rebuild where it exists too")
    marker_parser.add_argument("--scripts", action="store_true", help="print the build script")
    return parser


def main(argv: list[str] | None = None, *, executor: ops.Executor | None = None) -> int:
    args = _parser().parse_args(argv)
    run = executor or ops.ssh_executor
    try:
        if args.command == "plan":
            site = Site.load(args.site) if args.site else None
            if args.facts:
                facts = plan_mod.Facts.from_json(json.loads(args.facts.read_text(encoding="utf-8")))
                size = args.ring_size or (site.size if site else max(facts.positions) + 1)
            else:
                size = args.ring_size or (site.size if site else 8)
                names = [host.name for host in site.ring] if site else [f"spark{i}" for i in range(size)]
                facts = plan_mod.Facts.symbolic_for(names)
            layout = _layout(args, size)
            built = plan_mod.build_layout_plan(layout, facts, groups=layout.select(args.group),
                                               max_relays=args.max_relays)
            if args.json:
                print(json.dumps(built.to_json(), indent=1))
            else:
                print(plan_mod.render_text(built, facts, details=not args.brief))
            return 0
        site = Site.load(args.site)
        marker = commands.MarkerConfig(args.marker)
        if args.command in ("show", "facts", "marker"):
            positions = _positions(args.sparks, site.size)
            if args.command == "marker":
                return _marker_command(site, positions, run, args, marker)
            states = ops.gather(site, positions, run, marker)
            if args.command == "facts":
                problems = [p for state in states.values() for p in state.fact_problems(check_hostname=False)]
                text = json.dumps(ops.facts_from(site, states).to_json(), indent=1)
                if args.output:
                    args.output.write_text(text + "\n", encoding="utf-8")
                else:
                    print(text)
                for problem in problems:
                    print(f"problem: {problem}", file=sys.stderr)
                return 3 if problems else 0
            print(report.render_show(states, marker))
            if args.raw:
                for position, state in sorted(states.items()):
                    print(f"--- {state.name} raw ---")
                    for section, text in state.raw.items():
                        print(f"@@{section}\n{text}")
            return 0 if all(state.reachable for state in states.values()) else 1
        layout = _layout(args, site.size)
        groups = layout.select(args.group)
        positions = [position for group in groups for position in group.order]
        states = ops.gather(site, positions, run, marker)
        mode = "down" if args.command == "down" else "up"
        outcome = ops.evaluate(site, layout, groups, states, mode=mode, marker=marker, adopt=args.adopt,
                               check_hostname=not args.skip_hostname_check, max_relays=args.max_relays)
        if args.command == "diff":
            _print_outcome(outcome, args, scripts=args.scripts)
            if outcome.problems or not all(state.reachable for state in states.values()):
                return 3
            if _route_check(outcome, site, run, installed_only=True):
                return 3
            return 1 if outcome.host_changes else 0
        _print_outcome(outcome, args, scripts=not args.no_scripts)
        if not args.apply:
            print("dry run: nothing was changed; --apply runs the scripts above"
                  + (" (blockers must be resolved first)" if outcome.blocked else ""))
            return 3 if outcome.blocked else 0
        return _apply(outcome, site, run, args, marker)
    except (FabricError, SiteError, routes.RouteError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
