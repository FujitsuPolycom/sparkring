"""Operator commands of the serve launcher: ``python -m sparkring_sircl.vllm.serve <command>``.

The launcher serves a SparkRing profile (default ``glm53-flash-nvfp4-spark-tp4``)
on chosen Sparks of a switchless ring with SIRCL's vLLM adapter in front of
every collective. ``--positions`` names one group of consecutive Sparks, for
example ``0-3`` or ``7,0,1,2``; ``--groups`` names several disjoint groups,
each its own instance with its own run id and ports, for example
``0-1;2-3;4-5;6-7`` for four pairs. It runs on the operator's machine and
reaches each Spark over SSH, like the ring harness. Commands and their safety
classes (``RUNBOOK.md`` has the procedure):

- ``plan`` (OFFLINE): print the launch plan (``--json`` for the plan record);
- ``start --print`` (OFFLINE): the plan plus every remote command of ``stage``
  and ``start``; contacts nothing;
- ``preflight`` (READ-ONLY REMOTE): the ring harness's fabric checks (devices,
  addresses, lane routes) with the serving image, then each rank's model
  directory, memory, ports, running containers and staging state, and with
  ``--overlay`` each rank's overlay directory and the tree it holds;
- ``stage`` (MUTATES HOST: files under ``<remote_dir>/serve`` and
  ``<remote_dir>/build-cache``, one short-lived container per Spark): copy the
  package tree and the seccomp policy, build the native library in the serving
  image and probe what serving will import;
- ``start`` (MUTATES HOST: one serving container per rank): refuses while any
  other container runs on a used Spark, while a run has containers, while
  staging is incomplete, or while an overlay is missing or differs between
  Sparks; ``--wait`` continues with ``wait``;
- ``wait``, ``status``, ``logs`` (READ-ONLY REMOTE): container states, API
  readiness, log lines and every rank's SIRCL receipt line;
- ``check`` (READ-ONLY REMOTE plus inference requests to these runs' servers):
  three known-answer prompts, an optional long-prompt timing probe, every
  rank's receipt line and receipt evaluation;
- ``collect`` (READ-ONLY REMOTE, writes local files): logs and receipts of
  every rank;
- ``stop`` (STOPS SERVING: removes these runs' containers, found by label;
  ``--all-runs`` removes every ``sircl-serve`` container): collects first
  unless ``--no-collect``;
- ``bundle`` (OFFLINE; with ``--stage`` MUTATES HOST) and ``bundle-check``
  (READ-ONLY REMOTE): SIRCL for containers another launcher starts
  (:mod:`.bundle`);
- ``shims`` (OFFLINE): the catalog of SIRCL's version-pinned vLLM shims
  (:mod:`sparkring_sircl.vllm.catalog`), and with ``--vllm-tree PATH`` each
  shim's status in a local vLLM tree, or with ``--probe FILE`` in the vLLM a
  stage probe saw (a saved ``SIRCL-SERVE-PROBE`` line). ``stage`` prints the
  statuses of every Spark's image.

Output: :func:`main` sets standard output and standard error to replace any
character their encoding lacks. Container log tails and model replies may
hold any character (progress bars draw block characters), and a console or a
redirected stream on Windows encodes in the ANSI code page (cp1252); a strict
encoder would end ``wait`` while the containers keep starting.
"""

from __future__ import annotations

import argparse
import dataclasses
import io
import json
import sys
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...ring import remote
from ...ring.site import SiteError
from .. import catalog
from .. import receipt as receipt_mod
from . import bundle, checks, commands, probe
from . import plan as plan_mod
from . import profile as profile_mod
from . import staging
from .plan import ServePlan, ServePlanError
from .profile import ProfileError
from .sitefile import ServeSite

Runner = Callable[..., remote.Result]
UP = ("running", "created", "restarting")
REGISTERED = "SIRCL vLLM adapter registered"
ACTIVATED = "Platform plugin sircl is activated"
RECEIPT = "SIRCL receipt"
FAILURES = ("Failed to load plugin sircl", "SirclSetupError", "NcclAcrossRelayError", "SirclDispatchError",
            "ShimRefused", "poisoned", "Platform plugin")
REPORT_SECONDS = 60.0


def default_runner(target: str, command: str, *, timeout: float = 60,
                   input_bytes: bytes | None = None) -> remote.Result:
    return remote.ssh(target, command, timeout=timeout, input_bytes=input_bytes)


@dataclasses.dataclass
class Context:
    """One serving instance: its plan and the means to reach its Sparks."""

    plan: ServePlan
    tree: staging.StagedTree
    run: Runner = default_runner
    out: Callable[[str], None] = print
    sleep: Callable[[float], None] = time.sleep
    clock: Callable[[], float] = time.monotonic

    @property
    def profile(self) -> profile_mod.ServingProfile:
        return self.plan.profile

    def on(self, launch: plan_mod.RankLaunch, command: str, *, timeout: float = 60,
           input_bytes: bytes | None = None) -> remote.Result:
        return self.run(launch.ssh, command, timeout=timeout, input_bytes=input_bytes)


Instances = Context | Sequence[Context]


def _all(instances: Instances) -> list[Context]:
    return [instances] if isinstance(instances, Context) else list(instances)


def facts(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key.strip()] = value.strip()
    return values


@dataclasses.dataclass(frozen=True)
class Running:
    name: str
    image: str
    status: str
    serve_run: str
    ring_run: str
    deployment: str

    def describe(self) -> str:
        if self.serve_run:
            kind = f"SIRCL serve run {self.serve_run}"
        elif self.ring_run:
            kind = f"SIRCL ring harness run {self.ring_run}"
        elif self.deployment:
            kind = f"SparkRing deployment {self.deployment[:12]}"
        else:
            kind = "container"
        return f"{self.name} ({kind}, image {self.image[:40]}, {self.status})"


def running(text: str) -> list[Running]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = (line.split("\t") + [""] * 6)[:6]
        rows.append(Running(*(part.strip() for part in parts)))
    return rows


def _where(launch: plan_mod.RankLaunch, plan: ServePlan) -> str:
    return f"run {plan.run_id} rank {launch.rank} ({launch.host})"


# -- preflight --------------------------------------------------------------------------------------


def harness_preflight(plans: ServePlan | Sequence[ServePlan]) -> tuple[bool, list[str]]:
    """The ring harness's fabric checks for these groups, with the serving image as the site's image."""
    from ...ring import cli as ring_cli
    from ...ring import plan as ring_plan

    plans = [plans] if isinstance(plans, ServePlan) else list(plans)
    first = plans[0]
    site = dataclasses.replace(first.site, image=first.profile.image_id)
    harness = ring_plan.build_plan(site, "serve", first.run_id, groups=tuple(plan.positions for plan in plans),
                                   options=ring_plan.Options(), digest=ring_plan.source_digest())
    return ring_cli.preflight(site, [harness], force=True)


def memory_utilization(plan: ServePlan) -> float | None:
    """``--gpu-memory-utilization`` of the plan's vLLM arguments, or None when they give none."""
    value = plan_mod._recipe_value(plan.recipe_arguments, "--gpu-memory-utilization")
    return float(value) if value is not None else None


def preflight(instances: Instances, *, allow_running: Sequence[str] = (), fabric: bool = True) -> int:
    contexts = _all(instances)
    out = contexts[0].out
    blockers: list[str] = []
    if fabric:
        ok, lines = harness_preflight([ctx.plan for ctx in contexts])
        for line in lines:
            out(f"fabric: {line}")
            if line.startswith("BLOCKER: "):
                blockers.append(line[len("BLOCKER: "):])
    for ctx in contexts:
        blockers += instance_preflight(ctx, allow_running)
    blockers += overlay_check(contexts)
    for blocker in blockers:
        out(f"BLOCKER: {blocker}")
    out("preflight passed" if not blockers else f"preflight found {len(blockers)} blocker(s)")
    return 0 if not blockers else 1


def overlay_check(contexts: Sequence[Context]) -> list[str]:
    """Every rank's source overlay: the files it must hold, and one tree on every Spark. Prints what it found and
    returns blockers; nothing without ``--overlay``."""
    rows = []
    for ctx in contexts:
        for launch in ctx.plan.ranks:
            mount = commands.overlay_mount(launch)
            if mount is not None:
                answer = ctx.on(launch, commands.overlay_facts(mount.source, launch.sudo), timeout=60)
                rows.append((_where(launch, ctx.plan), mount.source, facts(answer.stdout)))
    if not rows:
        return []
    lines, blockers = checks.overlay_findings(rows)
    for line in lines:
        contexts[0].out(line)
    return blockers


CHECKPOINT_JSON = ("config.json", "model.safetensors.index.json")


def _checkpoint_copy(values: Mapping[str, str]) -> tuple[tuple[str, ...], str] | None:
    """For a checkpoint the profile's manifest does not describe: what identifies one Spark's copy (the two JSON
    digests, the file count and bytes) and its text, or None when a JSON file is unreadable."""
    digests = [values.get(f"sha256:{name}", "") for name in CHECKPOINT_JSON]
    if not all(digests):
        return None
    count, _, size = values.get("files", "0 0").partition(" ")
    text = (f"config.json {digests[0]}, model.safetensors.index.json {digests[1]}, {count} files, "
            f"{int(size) if size.isdigit() else 0:,} bytes")
    return (*digests, count, size), text


def instance_preflight(ctx: Context, allow_running: Sequence[str] = ()) -> list[str]:
    """One instance's host checks; prints what it found and returns blockers."""
    plan, profile = ctx.plan, ctx.profile
    blockers: list[str] = []
    expected = {item.name: item.size for item in profile.checkpoint_files}
    pins = {"config.json": profile.config_sha256, "model.safetensors.index.json": profile.index_sha256}
    utilization = memory_utilization(plan)
    copies: dict[str, tuple[tuple[str, ...], str]] = {}
    for launch in plan.ranks:
        where = _where(launch, plan)
        via = f" through \"{launch.sudo}\"" if launch.sudo else ""
        answer = ctx.on(launch, commands.host_facts(plan, launch, profile), timeout=120)
        if not answer.ok and not answer.stdout:
            blockers.append(f"{where}: host checks{via} failed ({answer.stderr.strip() or answer.returncode})")
            continue
        values = facts(answer.stdout)
        model = launch.mount(plan_mod.MODEL_TARGET).source
        if values.get("model") != "present":
            blockers.append(f"{where}: model directory {model} ({launch.model_source}) is missing or unreadable{via}")
        elif plan.foreign_checkpoint:
            copy = _checkpoint_copy(values)
            if copy is None:
                blockers.append(f"{where}: {model} ({launch.model_source}) has no readable "
                                f"{' or '.join(CHECKPOINT_JSON)}{via}")
            else:
                copies[where] = copy
                ctx.out(f"{where}: model directory {model} ({launch.model_source}) for checkpoint {plan.checkpoint}, "
                        f"which the profile's manifest does not describe: {copy[1]}{via}")
        else:
            missing = [name for name in expected if values.get(f"size:{name}") == "missing"]
            wrong = [name for name, size in expected.items()
                     if values.get(f"size:{name}", "missing") not in ("missing", str(size))]
            if missing:
                blockers.append(f"{where}: {len(missing)} of {len(expected)} checkpoint files are missing in "
                                f"{model} (first: {missing[0]})")
            if wrong:
                blockers.append(f"{where}: {len(wrong)} checkpoint files in {model} have sizes other than the "
                                f"manifest's (first: {wrong[0]})")
            for name, digest in pins.items():
                if values.get(f"sha256:{name}") != digest:
                    blockers.append(f"{where}: {model}/{name} has SHA-256 "
                                    f"{values.get(f'sha256:{name}') or 'nothing'}, the profile pins {digest}")
            if not missing and not wrong:
                ctx.out(f"{where}: model directory {model} ({launch.model_source}) holds the {len(expected)} "
                        f"pinned files ({profile.checkpoint_bytes / 2**30:.1f} GiB, sizes and the two JSON "
                        f"digests checked{via})")
        try:
            total, available = int(values["mem:MemTotal"]), int(values["mem:MemAvailable"])
        except (KeyError, ValueError):
            blockers.append(f"{where}: /proc/meminfo could not be read")
        else:
            if utilization is None:
                ctx.out(f"{where}: memory {available / 2**20:.1f} GiB available of {total / 2**20:.1f} GiB; the "
                        "vLLM arguments give no --gpu-memory-utilization to check against")
            else:
                needed = utilization * total
                ctx.out(f"{where}: memory {available / 2**20:.1f} GiB available of {total / 2**20:.1f} GiB; vLLM "
                        f"asks for {utilization:g} of the total ({needed / 2**20:.1f} GiB)")
                if available < needed:
                    blockers.append(f"{where}: {available / 2**20:.1f} GiB available is below the {needed / 2**20:.1f} "
                                    f"GiB vLLM needs at --gpu-memory-utilization {utilization:g}")
        if launch.rank == 0 and values.get("curl", "missing") == "missing":
            blockers.append(f"{where}: curl is missing; the launcher sends its API requests with curl on this Spark")
        space = values.get("space_kib", "")
        cache = launch.mount(plan_mod.CACHE_TARGET).source
        free = f", {int(space) / 2**20:.0f} GiB free there" if space.isdigit() else ""
        ctx.out(f"{where}: cache {cache} {values.get('cache', 'unknown')}{free}")
        listed = ctx.on(launch, commands.running_containers(launch.docker), timeout=30)
        if not listed.ok:
            blockers.append(f"{where}: cannot list containers with \"{launch.docker}\": {listed.stderr.strip()}")
        else:
            for row in running(listed.stdout):
                if row.name not in allow_running:
                    blockers.append(f"{where}: {row.describe()} is running")
        staged = facts(ctx.on(launch, commands.staged_state(plan), timeout=30).stdout)
        ctx.out(f"{where}: staged package {plan.staged_digest} {staged.get('source', 'unknown')}, library "
                f"{plan.library} {staged.get('library', 'unknown')}, seccomp policy "
                f"{'present' if staged.get('seccomp') == profile.seccomp_sha256 else 'missing'}"
                " (stage writes what is missing)")
    if len({key for key, _ in copies.values()}) > 1:
        blockers.append(f"run {plan.run_id}: the copies of checkpoint {plan.checkpoint} differ between Sparks: "
                        + "; ".join(f"{where}: {text}" for where, (_, text) in copies.items()))
    return blockers + port_blockers(ctx)


def port_blockers(ctx: Context) -> list[str]:
    plan = ctx.plan
    launch = plan.api_rank
    answer = ctx.on(launch, commands.listening_ports((plan.api_port, plan.master_port)), timeout=30)
    blockers = []
    for port in (plan.api_port, plan.master_port):
        state = facts(answer.stdout).get(f"port:{port}", "unknown")
        if state != "free":
            blockers.append(f"{_where(launch, plan)}: TCP port {port} is {state}")
    return blockers


# -- stage ----------------------------------------------------------------------------------------


def stage(instances: Instances) -> int:
    contexts = _all(instances)
    out = contexts[0].out
    failures = 0
    for ctx in contexts:
        plan, profile = ctx.plan, ctx.profile
        tar = ctx.tree.tar()
        for launch in plan.ranks:
            where = _where(launch, plan)
            unpacked = ctx.on(launch, commands.stage_tree(plan.source_dir, plan.staged_digest), input_bytes=tar,
                              timeout=300)
            if not unpacked.ok:
                ctx.out(f"{where}: staging the package tree failed: {(unpacked.stderr or unpacked.stdout).strip()}")
                failures += 1
                continue
            policy = ctx.on(launch, commands.write_once(plan.seccomp_path, profile.seccomp_sha256),
                            input_bytes=profile.seccomp_policy, timeout=60)
            if not policy.ok:
                ctx.out(f"{where}: writing the seccomp policy failed: {(policy.stderr or policy.stdout).strip()}")
                failures += 1
                continue
            built = ctx.on(launch, commands.stage_container(plan, launch), timeout=900)
            record = probe.parse(built.stdout)
            if record is None:
                tail = (built.stdout + built.stderr).strip().splitlines()[-5:]
                ctx.out(f"{where}: the stage container failed (exit code {built.returncode}): {tail}")
                failures += 1
                continue
            blockers, notes = probe.evaluate(record, staged_root=plan_mod.SOURCE_TARGET, library=plan.library,
                                             overlay=plan_mod.OVERLAY_TARGET if plan.overlay else None,
                                             required_shims=plan.required_shims,
                                             mhc_sizes=((len(plan.ranks), plan.dcp_size)
                                                        if plan.mhc_prefill_shard and plan.dcp_size > 1 else None))
            modules = record.get("modules") or {}
            ctx.out(f"{where}: package tree {plan.staged_digest} {unpacked.stdout.strip()}, seccomp policy "
                    f"{policy.stdout.strip()}, library {plan.library} built, sparkring_sircl from "
                    f"{record.get('package')}, vllm from {modules.get('vllm')}, b12x from {modules.get('b12x')}")
            for note in notes:
                ctx.out(f"{where}: {note}")
            for line in catalog.stage_lines(record.get("shim_status")):
                ctx.out(f"{where}: {line}")
            for blocker in blockers:
                ctx.out(f"BLOCKER: {where}: {blocker}")
            failures += bool(blockers)
    out("stage complete" if not failures else f"stage failed on {failures} Spark(s)")
    return 0 if not failures else 1


def shims_command(args: argparse.Namespace) -> int:
    """``shims``: the catalog, or each shim's status in a local vLLM tree or in a saved stage probe record."""
    if args.vllm_tree is not None:
        root = catalog.package_root(args.vllm_tree)
        if not (root / "__init__.py").is_file():
            print(f"error: {args.vllm_tree} holds no vllm package (vllm/__init__.py)", file=sys.stderr)
            return 2
        status = catalog.tree_status(root)
    elif args.probe is not None:
        try:
            text = Path(args.probe).read_text(encoding="utf-8", errors="replace")
        except OSError as error:
            print(f"error: {error}", file=sys.stderr)
            return 2
        record = probe.parse(text)
        status = record.get("shim_status") if isinstance(record, Mapping) else None
        if not isinstance(status, Mapping):
            print(f"error: {args.probe} holds no stage probe record ({probe.PREFIX.strip()} line) with shim "
                  "statuses", file=sys.stderr)
            return 2
    else:
        print(json.dumps(catalog.document(), indent=1) if args.json else catalog.render_catalog())
        return 0
    print(json.dumps(status, indent=1) if args.json else catalog.render_status(status))
    return 0


# -- start, wait, status --------------------------------------------------------------------------


def start_blockers(ctx: Context, allow_running: Sequence[str] = ()) -> list[str]:
    plan, profile = ctx.plan, ctx.profile
    blockers = []
    for launch in plan.ranks:
        where = _where(launch, plan)
        listed = ctx.on(launch, commands.running_containers(launch.docker), timeout=30)
        if not listed.ok:
            blockers.append(f"{where}: cannot list containers with \"{launch.docker}\": "
                            f"{(listed.stderr or listed.stdout).strip()}")
            continue
        for row in running(listed.stdout):
            if row.name in allow_running and row.serve_run != plan.run_id:
                continue
            blockers.append(f"{where}: {row.describe()} is running; this launcher starts only on Sparks that "
                            "run no other container")
        own = ctx.on(launch, commands.run_containers(plan.run_id, launch.docker), timeout=30)
        names = [line.split("\t")[0] for line in own.stdout.splitlines() if line.strip()]
        if names:
            blockers.append(f"{where}: run {plan.run_id} already has containers {names}; stop the run first")
        staged = facts(ctx.on(launch, commands.staged_state(plan), timeout=30).stdout)
        if staged.get("source") != "present":
            blockers.append(f"{where}: package tree {plan.staged_digest} is not staged; run stage")
        if staged.get("library") != "present":
            blockers.append(f"{where}: native library {plan.library} is not built; run stage")
        if staged.get("seccomp") != profile.seccomp_sha256:
            blockers.append(f"{where}: seccomp policy {plan.seccomp_path} is missing or differs; run stage")
    return blockers + port_blockers(ctx)


def start(instances: Instances, *, allow_running: Sequence[str] = (), wait_after: bool = False,
          timeout: float = 3600, interval: float = 10, stop_on_failure: bool = False) -> int:
    """Start every instance, or none: a failed check starts nothing, a failed start removes what started."""
    contexts = _all(instances)
    out = contexts[0].out
    blockers = [blocker for ctx in contexts for blocker in start_blockers(ctx, allow_running)]
    blockers += overlay_check(contexts)
    if blockers:
        for blocker in blockers:
            out(f"BLOCKER: {blocker}")
        out("nothing was started")
        return 1
    for ctx in contexts:
        for launch in ctx.plan.ranks:
            made = ctx.on(launch, commands.make_directories(launch.directories, launch.sudo), timeout=60)
            if not made.ok:
                via = f" through \"{launch.sudo}\"" if launch.sudo else ""
                out(f"{_where(launch, ctx.plan)}: creating {list(launch.directories)}{via} failed: "
                    f"{(made.stderr or made.stdout).strip()}; nothing was started")
                return 1
            for table, command in plan_mod.write_tuning_tables(ctx.plan.tuning.tables, ctx.plan.run_dir,
                                                               launch.sudo):
                written = ctx.on(launch, command, timeout=60, input_bytes=table.data)
                if not written.ok:
                    out(f"{_where(launch, ctx.plan)}: writing tuning table {table.hash} failed: "
                        f"{(written.stderr or written.stdout).strip()}; nothing was started")
                    return 1
    for ctx in contexts:
        for launch in ctx.plan.ranks:
            launched = ctx.on(launch, launch.shell(), timeout=180)
            if not launched.ok:
                out(f"{_where(launch, ctx.plan)}: docker run failed: {(launched.stderr or launched.stdout).strip()}")
                out("removing the containers these runs started")
                for started in contexts:
                    remove(started, all_runs=False)
                return 1
            ctx.out(f"{_where(launch, ctx.plan)}: started {launch.container} ({launched.stdout.strip()[:12]})")
        ctx.out(f"run {ctx.plan.run_id}: {len(ctx.plan.ranks)} containers started; the API will listen on "
                f"http://{ctx.plan.api_rank.lan_address}:{ctx.plan.api_port}/v1")
    if wait_after:
        return wait(contexts, timeout=timeout, interval=interval, stop_on_failure=stop_on_failure)
    return 0


def states(ctx: Context) -> dict[int, tuple[str, int, str]]:
    result = {}
    for launch in ctx.plan.ranks:
        words = ctx.on(launch, commands.container_state(launch.container, launch.docker), timeout=30).stdout.split()
        status = words[0] if words else "unknown"
        code = int(words[1]) if len(words) > 1 and words[1].lstrip("-").isdigit() else -1
        health = words[2] if len(words) > 2 else "none"
        result[launch.rank] = (status, code, health)
    return result


def api_ready(ctx: Context) -> tuple[bool, str]:
    plan = ctx.plan
    answer = ctx.on(plan.api_rank, commands.api_get(plan.api_port, "/v1/models"), timeout=30)
    if not answer.ok:
        return False, (answer.stderr.strip() or "no answer")
    try:
        served = [item["id"] for item in json.loads(answer.stdout)["data"]]
    except (ValueError, KeyError, TypeError):
        return False, f"unexpected answer {answer.stdout[:120]!r}"
    if ctx.profile.served_model_name in served:
        return True, ", ".join(served)
    return False, f"serves {served}, not {ctx.profile.served_model_name}"


def receipt_lines(ctx: Context) -> list[str]:
    """Print every rank's SIRCL lines; returns problems (a rank without a tensor-parallel receipt line)."""
    problems = []
    for launch in ctx.plan.ranks:
        found = ctx.on(launch, commands.log_lines(launch.container, launch.docker, (RECEIPT, REGISTERED, ACTIVATED)),
                       timeout=120).stdout.splitlines()
        receipts = [line for line in found if RECEIPT in line]
        ctx.out(f"{_where(launch, ctx.plan)}: {len(receipts)} receipt line(s), "
                f"{sum(REGISTERED in line for line in found)} registration line(s), "
                f"{sum(ACTIVATED in line for line in found)} platform activation line(s)")
        for line in receipts:
            ctx.out(f"  {line[line.index(RECEIPT):]}")
        if not any("group=tp" in line for line in receipts):
            problems.append(f"{_where(launch, ctx.plan)}: no SIRCL receipt line for the tensor-parallel group; "
                            "the sircl plugins did not run in this rank's worker")
    return problems


def failure_lines(ctx: Context) -> dict[int, list[str]]:
    found = {}
    for launch in ctx.plan.ranks:
        lines = ctx.on(launch, commands.log_lines(launch.container, launch.docker, FAILURES),
                       timeout=120).stdout.splitlines()
        lines = [line for line in lines if ACTIVATED not in line]
        if lines:
            found[launch.rank] = lines[-5:]
    return found


def _report_down(ctx: Context, down: Mapping[int, tuple[str, int, str]]) -> None:
    for rank, (status, code, _) in sorted(down.items()):
        launch = ctx.plan.ranks[rank]
        ctx.out(f"{_where(launch, ctx.plan)}: container {launch.container} is {status} (exit code {code}); its "
                "last log lines:")
        tail = ctx.on(launch, commands.container_logs(launch.container, launch.docker, 40), timeout=60)
        for line in tail.stdout.splitlines():
            ctx.out(f"  {line}")


def _progress(ctx: Context, elapsed: float, detail: str) -> bool:
    """Print one progress report of an instance; True when its logs show a failure."""
    current = states(ctx)
    summary = ", ".join(f"r{rank} {status}/{health}" for rank, (status, _, health) in sorted(current.items()))
    ctx.out(f"{elapsed:5.0f} s: run {ctx.plan.run_id}: {summary}; API: {detail}")
    for launch in ctx.plan.ranks:
        last = ctx.on(launch, commands.container_logs(launch.container, launch.docker, 1), timeout=60)
        ctx.out(f"  r{launch.rank}: {last.stdout.strip()[-200:]}")
    failures = failure_lines(ctx)
    for rank, lines in sorted(failures.items()):
        for line in lines:
            ctx.out(f"  r{rank} failure: {line[-300:]}")
    return bool(failures)


def wait(instances: Instances, *, timeout: float = 3600, interval: float = 10,
         stop_on_failure: bool = False) -> int:
    """Wait until every instance's API serves the profile's model; report each rank's receipt lines."""
    contexts = _all(instances)
    first = contexts[0]
    began = first.clock()
    next_report = began
    pending = list(contexts)
    ready: list[Context] = []
    failed: list[Context] = []
    details: dict[str, str] = {}
    while pending:
        for ctx in list(pending):
            current = states(ctx)
            down = {rank: state for rank, state in current.items() if state[0] not in UP}
            if down:
                _report_down(ctx, down)
                failed.append(ctx)
                pending.remove(ctx)
                continue
            is_ready, detail = api_ready(ctx)
            details[ctx.plan.run_id] = detail
            if is_ready:
                ctx.out(f"API ready after {first.clock() - began:.0f} s: run {ctx.plan.run_id} serves {detail}")
                ready.append(ctx)
                pending.remove(ctx)
        if not pending:
            break
        now = first.clock()
        if now >= next_report:
            for ctx in list(pending):
                if _progress(ctx, now - began, details.get(ctx.plan.run_id, "")):
                    failed.append(ctx)
                    pending.remove(ctx)
            next_report = now + REPORT_SECONDS
        if pending and now - began >= timeout:
            for ctx in pending:
                ctx.out(f"run {ctx.plan.run_id}: the API did not become ready within {timeout:.0f} s")
            failed += pending
            pending = []
        if pending:
            first.sleep(interval)
    problems = [problem for ctx in ready for problem in receipt_lines(ctx)]
    for problem in problems:
        first.out(f"PROBLEM: {problem}")
    for ctx in failed:
        _failed(ctx, stop_on_failure)
    return 0 if not failed and not problems else 1


def _failed(ctx: Context, stop_on_failure: bool) -> None:
    if stop_on_failure:
        ctx.out(f"stopping run {ctx.plan.run_id} (--stop-on-failure)")
        remove(ctx, all_runs=False)
    else:
        ctx.out(f"the containers of run {ctx.plan.run_id} are left for inspection; `logs` shows them and "
                "`stop` removes them")


def status(instances: Instances) -> int:
    code = 0
    for ctx in _all(instances):
        current = states(ctx)
        for launch in ctx.plan.ranks:
            state, exit_code, health = current[launch.rank]
            ctx.out(f"{_where(launch, ctx.plan)}: {launch.container} {state} (exit code {exit_code}, health {health})")
        ready, detail = api_ready(ctx)
        ctx.out(f"run {ctx.plan.run_id}: API http://{ctx.plan.api_rank.lan_address}:{ctx.plan.api_port}/v1 "
                f"{'ready, serves ' + detail if ready else 'not ready (' + detail + ')'}")
        if ready:
            receipt_lines(ctx)
        code = max(code, 0 if ready and all(state[0] == "running" for state in current.values()) else 1)
    return code


# -- check ----------------------------------------------------------------------------------------


def _post(ctx: Context, body: Mapping[str, Any], timeout: float) -> remote.Result:
    plan = ctx.plan
    return ctx.on(plan.api_rank, commands.api_post(plan.api_port, "/v1/chat/completions", timeout),
                  input_bytes=json.dumps(body).encode(), timeout=timeout + 30)


def read_receipts(ctx: Context) -> tuple[dict[int, list[dict[str, Any]]], list[str]]:
    records, problems = {}, []
    for launch in ctx.plan.ranks:
        answer = ctx.on(launch, commands.receipts(ctx.plan.receipt_dir, launch.rank), timeout=60)
        parsed, bad = checks.parse_receipts(answer.stdout)
        records[launch.rank] = parsed
        problems += [f"{_where(launch, ctx.plan)}: {item}" for item in bad]
    return records, problems


def _tensor_parallel_stats(records: Mapping[int, Sequence[Mapping[str, Any]]]) -> Mapping[str, Any] | None:
    """Session statistics of rank 0's tensor-parallel receipt."""
    for record in records.get(0, ()):
        if str(record.get("group", "")).startswith("tp") and isinstance(record.get("session_stats"), Mapping):
            return record["session_stats"]
    return None


def _long_prompt_line(plan: ServePlan, tokens: int, seconds: float, stats: Mapping[str, Any] | None) -> str:
    """The measured prefill, with its session ops counted and priced from the session's statistics."""
    line = (f"long prompt: {tokens:,} prompt tokens answered in {seconds:.2f} s "
            f"({tokens / seconds:,.0f} tokens/s)")
    estimate = plan.prefill(tokens)
    if estimate is None:
        rate = plan_mod.PROFILE_PREFILL_RATES.get(plan.profile.id)
        return line + (f"; the profile's own deployment prefills {rate:,.0f} tokens/s" if rate is not None else "")
    session = plan_mod.SessionOps.from_stats(stats)
    if session is None:
        bound = estimate.bound_seconds
        return line + ("; rank 0's receipt states no session op sizes"
                       + (f"; at most {bound[0]:.1f}-{bound[1]:.1f} s expected at the one-shot rate" if bound else ""))
    reduce_ops, scatter_ops, gather_ops = estimate.ops(session)
    line += (f"; session ops ({session.describe()}, from rank 0's receipt): {reduce_ops:,} all-reduce, "
             f"{scatter_ops:,} reduce-scatter and {gather_ops:,} all-gather ops")
    per_op = estimate.implied_op_seconds(seconds, session)
    if per_op is None:
        return line + "; compute unknown for this profile"
    basis = "measured on the path" if estimate.compute_measured else "the profile's bound"
    return line + (f"; after compute of {estimate.compute_seconds:.1f} s ({basis}) the collectives took "
                   f"{seconds - estimate.compute_seconds:.1f} s, {per_op * 1e3:.2f} ms per op")


def check(instances: Instances, *, long_prompt: int = 0, request_timeout: float = 300) -> int:
    code = 0
    for ctx in _all(instances):
        code = max(code, _check_one(ctx, long_prompt=long_prompt, request_timeout=request_timeout))
    return code


def _check_one(ctx: Context, *, long_prompt: int, request_timeout: float) -> int:
    plan, profile = ctx.plan, ctx.profile
    ready, detail = api_ready(ctx)
    if not ready:
        ctx.out(f"run {plan.run_id}: the API is not ready: {detail}")
        return 1
    failures = 0
    for prompt in checks.PROMPTS:
        answer = _post(ctx, checks.chat_body(profile, prompt.content), request_timeout)
        result = checks.read_answer(prompt.name, answer.stdout, prompt.accepts)
        if not answer.ok and not result.error:
            result = dataclasses.replace(result, ok=False, error=answer.stderr.strip())
        ctx.out(result.describe(prompt.expectation))
        failures += not result.ok
    if long_prompt:
        body = checks.long_prompt_body(profile, long_prompt, uuid.uuid4().hex[:12])
        answer = _post(ctx, body, 1800)
        result = checks.read_answer("long prompt", answer.stdout, lambda reply: True)
        tokens = int(result.usage.get("prompt_tokens", 0) or 0)
        # Receipts refresh their counts from the worker's post-step check at most
        # every COUNT_REFRESH_SECONDS; one short request after that period writes
        # counts that include the long prompt.
        ctx.sleep(receipt_mod.COUNT_REFRESH_SECONDS + 0.5)
        _post(ctx, checks.chat_body(profile, checks.PROMPTS[0].content), request_timeout)
        if result.error or not tokens or result.seconds is None:
            ctx.out(f"long prompt: FAILED ({result.error or answer.stderr.strip() or 'no usage'})")
            failures += 1
        else:
            records, _ = read_receipts(ctx)
            ctx.out(_long_prompt_line(plan, tokens, result.seconds, _tensor_parallel_stats(records)))
    problems = receipt_lines(ctx)
    records, bad = read_receipts(ctx)
    found, lines = checks.evaluate_receipts(records, len(plan.ranks), dcp=plan.dcp_size)
    imports, wrong = checks.import_findings(records, plan_mod.OVERLAY_TARGET if plan.overlay else None)
    grids, unequal = checks.large_blocks_findings(records, plan.large_blocks)
    tuned, untuned = checks.tuning_findings(records, plan.tuning.expected())
    for line in lines + imports + grids + tuned:
        ctx.out(line)
    problems += bad + found + wrong + unequal + untuned
    if plan.require_no_nccl:
        free, nccl = checks.nccl_free_findings(records, len(plan.ranks))
        logged = {launch.rank: ctx.on(launch, commands.log_lines(launch.container, launch.docker,
                                                                  plan_mod.NCCL_LOG_PATTERNS),
                                      timeout=120).stdout.splitlines() for launch in plan.ranks}
        scan, created = checks.nccl_log_findings(logged, debug=plan.nccl_debug)
        for line in free + scan:
            ctx.out(line)
        problems += nccl + created
    for problem in problems:
        ctx.out(f"PROBLEM: {problem}")
    passed = not failures and not problems
    ctx.out(f"run {plan.run_id}: check passed" if passed else
            f"run {plan.run_id}: check failed: {failures} answer(s), {len(problems)} receipt problem(s)")
    return 0 if passed else 1


# -- logs, collect, stop --------------------------------------------------------------------------


def logs(instances: Instances, *, rank: int | None = None, tail: int = 100) -> int:
    for ctx in _all(instances):
        for launch in ctx.plan.ranks:
            if rank is not None and launch.rank != rank:
                continue
            answer = ctx.on(launch, commands.container_logs(launch.container, launch.docker, tail), timeout=120)
            ctx.out(f"==> {_where(launch, ctx.plan)} {launch.container}")
            ctx.out(answer.stdout.rstrip())
    return 0


def collect(instances: Instances, output: Path) -> int:
    for ctx in _all(instances):
        plan = ctx.plan
        folder = output / plan.run_id
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "plan.json").write_text(json.dumps(plan.to_json(), indent=1), encoding="utf-8")
        for launch in plan.ranks:
            log = ctx.on(launch, commands.container_logs(launch.container, launch.docker), timeout=300)
            (folder / f"rank-{launch.rank}.log").write_text(log.stdout, encoding="utf-8")
            found = ctx.on(launch, commands.receipts(plan.receipt_dir, launch.rank), timeout=60)
            records, _ = checks.parse_receipts(found.stdout)
            (folder / f"rank-{launch.rank}-receipts.json").write_text(json.dumps(records, indent=1),
                                                                      encoding="utf-8")
        ctx.out(f"logs and receipts of run {plan.run_id} are in {folder}")
    return 0


def remove(instances: Instances, *, all_runs: bool) -> int:
    """Remove the containers of these runs (by their labels), or with ``all_runs`` every ``sircl-serve`` one."""
    contexts = _all(instances)
    if all_runs:
        site = contexts[0].plan.site
        targets = [(contexts[0], host.name, host.ssh, host.docker, None) for host in site.ring]
    else:
        targets = [(ctx, launch.host, launch.ssh, launch.docker, ctx.plan.run_id)
                   for ctx in contexts for launch in ctx.plan.ranks]
    clean = True
    for ctx, name, ssh, docker, run_id in targets:
        answer = ctx.run(ssh, commands.remove_containers(run_id, docker), timeout=120)
        if answer.ok:
            left = answer.stdout.strip()
            ctx.out(f"{name}: {left} {'sircl-serve' if run_id is None else 'run ' + run_id} containers left")
            clean &= left == "0"
        else:
            ctx.out(f"{name}: removal with \"{docker}\" failed: "
                    f"{(answer.stderr or answer.stdout).strip() or f'exit code {answer.returncode}'}")
            clean = False
    return 0 if clean else 1


def stop(instances: Instances, *, all_runs: bool = False, collect_first: bool = True,
         output: Path = Path("sircl-serve-results")) -> int:
    contexts = _all(instances)
    if collect_first and not all_runs:
        try:
            collect(contexts, output)
        except OSError as error:
            contexts[0].out(f"collecting before stop failed ({error}); stopping anyway")
    return remove(contexts, all_runs=all_runs)


# -- entry point ------------------------------------------------------------------------------------


def groups_from_args(args: argparse.Namespace, size: int, tensor_parallel: int) -> tuple[tuple[int, ...], ...]:
    if args.groups and args.positions:
        raise ServePlanError("give --positions (one group) or --groups (several), not both")
    if args.groups:
        return plan_mod.parse_groups(args.groups, size)
    if args.positions:
        return (plan_mod.parse_positions(args.positions, size),)
    return (tuple(range(tensor_parallel)),)


def contexts_from_args(args: argparse.Namespace, *, run: Runner = default_runner) -> list[Context]:
    site = ServeSite.load(args.site)
    profile = profile_mod.load(args.repository, args.profile)
    groups = groups_from_args(args, site.site.size, profile.tensor_parallel)
    used = [position for group in groups for position in group]
    model_path, model_paths = plan_mod.split_paths(args.model_path or [], used, "--model-path")
    cache_path, cache_paths = plan_mod.split_paths(args.cache_path or [], used, "--cache-path")
    overlay, overlays = plan_mod.split_paths(args.overlay or [], used, "--overlay")
    defaults = plan_mod.Options(positions=groups[0])
    options = plan_mod.Options(
        positions=groups[0], model_paths=model_paths, model_path=model_path, cache_paths=cache_paths,
        cache_path=cache_path, api_port=args.api_port, master_port=args.master_port, run_id=args.run_id,
        capacity=args.capacity or defaults.capacity, dispatch=args.dispatch, oneshot_max=args.oneshot_max,
        large_blocks=args.large_blocks, nccl_debug=args.nccl_debug, require_no_nccl=args.require_no_nccl,
        gather=args.gather or defaults.gather,
        spin_limit=args.spin_limit, startup_wait=args.startup_wait, serving_wait=args.serving_wait,
        gid_index=args.gid_index, nccl_mode=args.nccl, large_allreduce=args.large_allreduce, dcp_size=args.dcp_size,
        mhc_prefill_shard=args.mhc_prefill_shard, extra_env=plan_mod.parse_env(args.env or []),
        large_schedule=args.large_schedule, gather_schedule=args.gather_schedule,
        scatter_schedule=args.scatter_schedule, ring_min=args.ring_min, chain_min=args.chain_min,
        link_sizes=plan_mod.link_values(args), link_slots=args.link_slots,
        ring_gather_stagger=args.ring_gather_stagger,
        tuning_tables=tuple(args.tuning_table or ()),
        reasoning_effort=args.reasoning_effort, overlay=overlay, overlays=overlays,
        vllm_edits=plan_mod.edits_from_args(args), checkpoint_id=args.checkpoint_id,
        thinking_behaviour=args.thinking_behaviour, b12x_cache_dir=args.b12x_cache_dir,
    )
    tree = staging.staged_tree()
    plans = plan_mod.build_plans(site, profile, groups, options, staged_digest=tree.digest,
                                 library=staging.library_name())
    return [Context(plan=plan, tree=tree, run=run) for plan in plans]


COMMANDS = ("plan", "preflight", "stage", "start", "wait", "status", "check", "logs", "collect", "stop")


def parser() -> argparse.ArgumentParser:
    top = argparse.ArgumentParser(prog="python -m sparkring_sircl.vllm.serve", description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = top.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        command.add_argument("--site", required=True, type=Path, help="site description (sircl-ring-site/v1)")
        command.add_argument("--repository", required=True, type=Path, help="SparkRing repository checkout")
        command.add_argument("--profile", default=profile_mod.DEFAULT_PROFILE)
        command.add_argument("--positions", help="ring positions of ranks 0..N-1, consecutive along the ring: "
                             "0-3, 7,0,1,2 or 7-2 (default: the first N Sparks)")
        command.add_argument("--groups", help="several instances, groups separated by ';': 0-1;2-3;4-5;6-7")
        command.add_argument("--model-path", action="append",
                             help="checkpoint directory: PATH for every Spark without a site model_path, or N=PATH "
                                  "for Spark N (overrides the site file)")
        command.add_argument("--cache-path", action="append",
                             help="cache directory: PATH or N=PATH (default <remote_dir>/serve/cache/<profile>)")
        command.add_argument("--api-port", type=int, default=plan_mod.DEFAULT_API_PORT,
                             help="API port of the first instance; instance k listens on it + k")
        command.add_argument("--master-port", type=int,
                             help="torch.distributed port of the first instance (default: the profile's); "
                                  "instance k uses it + k")
        command.add_argument("--run-id", help="run name with one group (default tp<N>-<first>-<last>); the "
                             "prefix of every run name with several")
        command.add_argument("--capacity", type=int, help=f"SIRCL all-reduce capacity "
                             f"(default {plan_mod.DEFAULT_CAPACITY})")
        command.add_argument("--dispatch", type=int, help="SIRCL dispatch ceiling (default: the capacity)")
        command.add_argument("--gather", type=int, help=f"SIRCL all-gather capacity (default {plan_mod.DEFAULT_GATHER})")
        plan_mod.add_oneshot_argument(command)
        plan_mod.add_large_blocks_argument(command)
        plan_mod.add_nccl_free_arguments(command)
        plan_mod.add_tuning_argument(command)
        command.add_argument("--spin-limit", type=int,
                             help="SIRCL_SPIN_LIMIT: poll budget of waits without a time limit; the sessions' "
                                  "flag waits are time-limited and ignore it (default: unset)")
        command.add_argument("--startup-wait", type=float, default=plan_mod.DEFAULT_STARTUP_WAIT_S,
                             metavar="SECONDS", help="SIRCL_STARTUP_WAIT_S: flag-wait limit during setup, warm-up, "
                             "graph capture, profiling, sleep and wake-up (default %(default)g)")
        command.add_argument("--serving-wait", type=float, default=plan_mod.DEFAULT_SERVING_WAIT_S,
                             metavar="SECONDS", help="SIRCL_SERVING_WAIT_S: flag-wait limit from the first step "
                             "after warm-up (default %(default)g)")
        command.add_argument("--gid-index", type=int, help="RoCE GID index (default: site, else profile)")
        command.add_argument("--nccl", choices=plan_mod.NCCL_MODES, default=plan_mod.DEFAULT_NCCL_MODE,
                             type=plan_mod.nccl_mode_value,
                             help=plan_mod.NCCL_MODE_HELP)
        command.add_argument("--large-allreduce", choices=plan_mod.LARGE_MODES, default="auto",
                             help="SIRCL_LARGE_ALLREDUCE: all-reduces above the dispatch ceiling")
        command.add_argument("--mhc-prefill-shard", choices=plan_mod.MHC_MODES, default="profile",
                             help="GLM-5.3-Flash mHC prefill sharding: the profile's setting, or off")
        command.add_argument("--dcp-size", dest="dcp_size", type=int, default=1, metavar="N",
                             help="vLLM's decode-context parallelism: N divides the profile's tensor parallelism; "
                                  "above 1 the launcher sets --decode-context-parallel-size N and a SIRCL session per "
                                  "decode-context-parallel group (SIRCL_GROUPS tp,dcp); refused where the checkpoint "
                                  "or its attention backend does not run DCP (default 1)")
        plan_mod.add_schedule_arguments(command)
        plan_mod.add_reasoning_argument(command)
        plan_mod.add_minimum_arguments(command)
        plan_mod.add_link_arguments(command)
        command.add_argument("--env", action="append", metavar="KEY=VALUE",
                             help="add or replace one variable on every rank (repeatable), vLLM and B12X runtime "
                                  "switches among them; SIRCL_*, VLLM_PLUGINS, PYTHONPATH and the other variables "
                                  "the launcher sets are refused")
        plan_mod.add_vllm_arguments(command)
        plan_mod.add_overlay_argument(command)
        plan_mod.add_checkpoint_arguments(command)
        plan_mod.add_b12x_cache_argument(command)
        if name == "plan":
            command.add_argument("--json", action="store_true")
        if name in ("preflight", "start"):
            command.add_argument("--allow-running", action="append", default=[], metavar="NAME",
                                 help="a running container that may stay (repeatable)")
        if name == "preflight":
            command.add_argument("--no-fabric", action="store_true", help="skip the ring harness's fabric checks")
        if name == "start":
            command.add_argument("--print", action="store_true", help="print the plan and commands; contact nothing")
            command.add_argument("--wait", action="store_true", help="wait for the API and the receipts")
        if name in ("start", "wait"):
            command.add_argument("--timeout", type=float, default=3600)
            command.add_argument("--interval", type=float, default=10)
            command.add_argument("--stop-on-failure", action="store_true")
        if name == "check":
            command.add_argument("--long-prompt", type=int, default=0, metavar="TOKENS",
                                 help="also time one prompt of about TOKENS tokens (e.g. 16384)")
        if name == "logs":
            command.add_argument("--rank", type=int)
            command.add_argument("--tail", type=int, default=100)
        if name in ("collect", "stop"):
            command.add_argument("--output", type=Path, default=Path("sircl-serve-results"))
        if name == "stop":
            command.add_argument("--all-runs", action="store_true",
                                 help="remove every sircl-serve container on every Spark of the site")
            command.add_argument("--no-collect", action="store_true")
    bundle.add_arguments(sub)
    shims = sub.add_parser("shims", help="SIRCL's version-pinned vLLM shims and their status in a vLLM tree")
    source = shims.add_mutually_exclusive_group()
    source.add_argument("--vllm-tree", type=Path, metavar="PATH",
                        help="a local vllm package directory, or a checkout holding one: each shim's status there")
    source.add_argument("--probe", type=Path, metavar="FILE",
                        help="a file holding a stage probe record (SIRCL-SERVE-PROBE line): each shim's status in "
                             "the vLLM the probe saw in the serving image")
    shims.add_argument("--json", action="store_true")
    return top


def tolerant_streams() -> None:
    """Make standard output and standard error replace characters their encoding cannot represent."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except (ValueError, io.UnsupportedOperation):
                pass


def main(argv: Sequence[str] | None = None, *, run: Runner = default_runner) -> int:
    tolerant_streams()
    args = parser().parse_args(plan_mod.join_dash_values(sys.argv[1:] if argv is None else argv))
    if args.command in ("bundle", "bundle-check"):
        return bundle.main(args, run=run)
    if args.command == "shims":
        return shims_command(args)
    try:
        contexts = contexts_from_args(args, run=run)
    except (SiteError, ProfileError, ServePlanError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    plans = [ctx.plan for ctx in contexts]
    if args.command == "plan":
        if args.json:
            record = plans[0].to_json() if len(plans) == 1 else {
                "schema": "sircl-serve-plans/v1", "instances": [plan.to_json() for plan in plans]}
            print(json.dumps(record, indent=1))
        else:
            print("\n\n".join(plan_mod.render_text(plan) for plan in plans))
        return 0
    if args.command == "start" and args.print:
        print("\n\n".join(plan_mod.render_text(plan) for plan in plans))
        print("remote commands of stage and start (nothing was contacted):")
        for line in commands.preview(plans):
            print(f"  {line}")
        return 0
    if args.command == "preflight":
        return preflight(contexts, allow_running=args.allow_running, fabric=not args.no_fabric)
    if args.command == "stage":
        return stage(contexts)
    if args.command == "start":
        return start(contexts, allow_running=args.allow_running, wait_after=args.wait, timeout=args.timeout,
                     interval=args.interval, stop_on_failure=args.stop_on_failure)
    if args.command == "wait":
        return wait(contexts, timeout=args.timeout, interval=args.interval, stop_on_failure=args.stop_on_failure)
    if args.command == "status":
        return status(contexts)
    if args.command == "check":
        return check(contexts, long_prompt=args.long_prompt)
    if args.command == "logs":
        return logs(contexts, rank=args.rank, tail=args.tail)
    if args.command == "collect":
        return collect(contexts, args.output)
    return stop(contexts, all_runs=args.all_runs, collect_first=not args.no_collect, output=args.output)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
