"""Operations on the Sparks: read, plan, compare, apply and verify.

Every Spark is reached through an *executor*: a function that runs one
``bash -s`` script on an SSH target and returns its exit code and output
(:func:`ssh_executor`; tests pass the host-command simulator's instead). The
Sparks of an operation are contacted in parallel.

``evaluate`` reads nothing itself: it takes the states :func:`gather` read,
derives the facts of each group's members (addresses and MACs as the Sparks
report them), checks the cabling of each owned cable (both ends of each link
on one subnet), builds each group's plan and compares every member with it.
``apply`` runs the scripts of a comparison; callers read the Sparks again and
evaluate once more to verify.
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
from collections.abc import Callable, Iterable, Sequence

from .. import routes
from ..ring import remote
from ..ring.site import Site
from . import diff as diff_mod
from . import plan as plan_mod
from . import state as state_mod
from .commands import MarkerConfig, read_script, route_get_script
from .layouts import FabricError, FabricLayout, Group

Executor = Callable[[str, str, float], remote.Result]
READ_TIMEOUT = 60.0
APPLY_TIMEOUT = 180.0


def ssh_executor(target: str, script: str, timeout: float) -> remote.Result:
    """Run ``script`` with ``bash -s`` on ``target`` (batch-mode SSH, deadline ``timeout``)."""
    return remote.ssh(target, "bash -s", input_bytes=script.encode(), timeout=timeout)


def _parallel(jobs: Sequence[int], work: Callable[[int], object]) -> dict[int, object]:
    if not jobs:
        return {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
        return dict(zip(jobs, pool.map(work, jobs)))


def gather(site: Site, positions: Iterable[int], executor: Executor, marker: MarkerConfig,
           *, timeout: float = READ_TIMEOUT) -> dict[int, state_mod.HostState]:
    """Read every listed Spark (READ-ONLY REMOTE)."""
    script = read_script(marker)

    def read(position: int) -> state_mod.HostState:
        host = site.host(position)
        result = executor(host.ssh, script, timeout)
        return state_mod.parse(position, host.name, host.ssh, returncode=result.returncode, stdout=result.stdout,
                               stderr=result.stderr)

    return _parallel(sorted(set(positions)), read)


def facts_from(site: Site, states: dict[int, state_mod.HostState]) -> plan_mod.Facts:
    """The addresses and MACs the Sparks report (only usable ports are included)."""
    return plan_mod.Facts(plan_mod.SparkFacts(position, site.host(position).name, state.ports())
                          for position, state in sorted(states.items()) if state.reachable)


def cable_problems(group: Group, facts: plan_mod.Facts) -> list[str]:
    """Each owned cable's two links must join one subnet: otherwise the site's ring order is not the cabling."""
    problems = []
    for cable in group.layout().fabric.cables:
        for secondary in (False, True):
            here = facts.port(cable.a, routes.role_at(cable.a_port, secondary).netdev)
            there = facts.port(cable.b, routes.role_at(cable.b_port, secondary).netdev)
            networks = (here.network(), there.network())
            if None in networks:
                continue
            if networks[0] != networks[1] or here.address == there.address:
                problems.append(f"cable {cable.text} ({plan_mod.class_name(secondary)} link): {facts.name(cable.a)} "
                                f"{here.netdev} {here.address}/{here.prefixlen} and {facts.name(cable.b)} "
                                f"{there.netdev} {there.address}/{there.prefixlen} are not one subnet; the site's "
                                "ring order does not match the cabling")
    return problems


@dataclasses.dataclass
class Outcome:
    """A comparison of the selected groups' members with their plans (``up``) or their removal (``down``)."""

    layout: FabricLayout
    mode: str
    groups: tuple[Group, ...]
    plans: dict[str, plan_mod.GroupPlan] = dataclasses.field(default_factory=dict)
    diffs: list[diff_mod.SparkDiff] = dataclasses.field(default_factory=list)
    problems: list[str] = dataclasses.field(default_factory=list)
    facts: plan_mod.Facts | None = None

    @property
    def blocked(self) -> bool:
        return bool(self.problems) or any(d.blocked for d in self.diffs)

    @property
    def pending(self) -> bool:
        return any(d.pending for d in self.diffs)

    @property
    def host_changes(self) -> int:
        return sum(len(d.host_changes) for d in self.diffs)


def evaluate(site: Site, layout: FabricLayout, groups: Sequence[Group], states: dict[int, state_mod.HostState],
             *, mode: str, marker: MarkerConfig, adopt: bool, check_hostname: bool = True,
             max_relays: int = routes.DEFAULT_MAX_RELAYS) -> Outcome:
    if mode not in ("up", "down"):
        raise FabricError(f"mode must be up or down, got {mode!r}")
    outcome = Outcome(layout, mode, tuple(groups))
    problems = plan_mod.layout_isolation_problems(layout)
    if problems:
        raise FabricError("isolation: " + "; ".join(problems))
    facts = facts_from(site, states)
    outcome.facts = facts
    for group in groups:
        if mode == "down":
            for position in group.order:
                outcome.diffs.append(diff_mod.compare_down(states[position], group=group.label, marker=marker,
                                                           adopt=adopt, check_hostname=check_hostname))
            continue
        problems = [problem for position in group.order
                    for problem in states[position].fact_problems(check_hostname=False)]
        member_facts = plan_mod.Facts([facts.spark(p) for p in group.order if facts.has(p)])
        problems += member_facts.problems()
        group_plan = None
        if not problems:
            try:
                problems += cable_problems(group, facts)
                if not problems:
                    group_plan = plan_mod.build_group_plan(group, facts, relay_egress=layout.relay_egress,
                                                           max_relays=max_relays)
            except FabricError as error:
                problems.append(str(error))
        if group_plan is None:
            outcome.problems.extend(f"group {group.label}: {problem}" for problem in problems)
            for position in group.order:
                blocked = diff_mod.SparkDiff(position, site.host(position).name, group.label, mode)
                blocked.blockers.append(f"group {group.label} cannot be planned (see the problems above)")
                outcome.diffs.append(blocked)
            continue
        outcome.plans[group.label] = group_plan
        for position in group.order:
            outcome.diffs.append(diff_mod.compare_up(states[position], group_plan.spark(position), group_plan,
                                                     layout_name=layout.name, marker=marker, adopt=adopt,
                                                     check_hostname=check_hostname))
    return outcome


def lane_route_checks(outcome: Outcome, *, installed_only: bool) -> dict[int, dict[str, str]]:
    """Destinations to resolve with ``ip route get`` per Spark, and the network device each must use.

    Every SIRCL lane of every planned group (direct lanes use the connected
    route of their cable) and every other carried path. With
    ``installed_only``, relayed destinations whose origin route the
    comparison still has to add or replace are left out.
    """
    checks: dict[int, dict[str, str]] = {}
    for group_plan in outcome.plans.values():
        layout = group_plan.layout
        pending = {(d.position, change.key.split("/")[0]) for d in outcome.diffs
                   for change in d.changes if change.kind == "route" and change.action in ("add", "replace")}
        for spark in group_plan.sparks:
            for route in spark.routes:
                if not (installed_only and (spark.position, route.destination) in pending):
                    checks.setdefault(spark.position, {})[route.destination] = route.netdev
        facts = outcome.facts
        for rank, peers in enumerate(group_plan.sircl.ranks):
            origin = layout.positions[rank]
            for peer, lanes in peers.items():
                for lane in lanes:
                    if lane.hops == 1 and facts is not None and facts.has(layout.positions[peer]):
                        destination = facts.port(layout.positions[peer], lane.remote.netdev).address
                        checks.setdefault(origin, {})[destination] = lane.local.netdev
    return checks


def check_lane_routes(site: Site, outcome: Outcome, executor: Executor, *, installed_only: bool,
                      timeout: float = READ_TIMEOUT) -> dict[int, list[str]]:
    """Resolve every lane destination on its origin Spark (READ-ONLY REMOTE); problems per Spark."""
    checks = lane_route_checks(outcome, installed_only=installed_only)

    def resolve(position: int) -> list[str]:
        wanted = checks[position]
        result = executor(site.host(position).ssh, route_get_script(wanted), timeout)
        answers = state_mod.parse_sections(result.stdout)
        problems = []
        for destination, netdev in sorted(wanted.items()):
            answer = answers.get(f"route:{destination}", "").strip()
            words = answer.split()
            device = words[words.index("dev") + 1] if "dev" in words[:-1] else None
            if device != netdev:
                problems.append(f"ip route get {destination} uses {device or 'no device'}, the lane needs {netdev} "
                                f"({answer.splitlines()[0] if answer else 'no answer'})")
        return problems

    return {position: problems for position, problems in _parallel(sorted(checks), resolve).items() if problems}


def apply(outcome: Outcome, site: Site, executor: Executor, *, timeout: float = APPLY_TIMEOUT
          ) -> dict[int, remote.Result]:
    """Run every Spark's script of a comparison without blockers (MUTATES HOST)."""
    if outcome.blocked:
        raise FabricError("refusing to apply a comparison with blockers")
    scripts = {d.position: d.script for d in outcome.diffs if d.script}

    def run(position: int) -> remote.Result:
        return executor(site.host(position).ssh, scripts[position], timeout)

    return _parallel(sorted(scripts), run)
