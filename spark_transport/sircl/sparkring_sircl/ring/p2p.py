"""Point-to-point latency and bandwidth on a cabled ring of DGX Sparks (the ring harness's p2p cases).

``python -m sparkring_sircl.ring.p2p {plan,run} --site <site.json> [options]``
runs SIRCL's point-to-point channels (:mod:`sparkring_sircl.p2p`) between
chosen Sparks, one process per rank inside the serving image (GPU 0, host
networking), without vLLM. The ranks form one group: the whole ring
(``--config ring``, default) or consecutive Sparks along it (``--config
path4``, ``--config pair``, or ``--positions 0-3``). Every pair of the group has
a channel; a pair whose ranks share no cable is joined through the relays of
the Sparks between them, its relayed lanes paced by forward windows within the
relay rule (:mod:`sparkring_sircl.p2p.budget`), which the plan prints.

Cases, each per message size (default 4 KiB to 64 MiB in powers of four):

- ``pair A-B``: ranks A and B only, the others waiting at the barrier.
  ``checked``: messages A to B and B to A whose bytes come from a seed, compared
  bit for bit by the receiver. ``latency``: ping-pong, A sends and receives one
  message per iteration while B mirrors it; the one-way time is half of each
  round trip (CUDA events on A). ``bandwidth``: A sends a burst of messages,
  B receives them and answers with one 16-byte message; the burst's bytes over
  A's time from the first send to the answer;
- ``shift d``: every rank at once sends to the rank ``d`` ahead and receives
  from the rank ``d`` behind (``batch_isend_irecv``): checked exchanges, then
  bursts timed on every rank; the bandwidth is one rank's received bytes over its
  time, every relay queue of the cycle carrying the lanes of ``d`` pairs.

Default cases on the ring of eight: pairs 0-1, 0-2, 0-3 and 0-4 (zero to three
relays) and shifts 1, 2 and 4; on a path of four: pairs 0-1, 0-2, 0-3 and shift
1. ``preflight``, ``stage`` and ``cleanup`` are the ring harness's
(``python -m sparkring_sircl.ring``): the staged source digest is the same, and
every rank builds the point-to-point library into the staged build cache on its
first run (:mod:`sparkring_sircl.p2p.build`).

Safety: ``plan`` and ``run --print`` contact nothing (OFFLINE); ``run``
starts containers on the Sparks of the group and uses their RDMA devices
(MUTATES HOST: it refuses while other containers run unless ``--force``); it
does not stop a serving stack, so run it only on Sparks that serve nothing.

Status: implemented; the plan, the launch commands and the summary have CPU
tests (``tests/test_ring_p2p.py``); the worker (:mod:`.p2p_worker`) uses the
channels the GPU emulation checks and imports with torch 2.10 and CUDA Python,
but has not run (it needs the ring's RDMA devices).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .. import routes as routes_mod
from ..p2p import budget
from ..p2p.protocol import DEFAULT_CHUNK_BYTES, DEFAULT_WINDOW_BYTES
from . import plan as plan_mod
from . import remote
from .site import Site, SiteError

SCHEMA = "sircl-ring-p2p-plan/v1"
RESULT_SCHEMA = "sircl-ring-p2p-rank-result/v1"
DEFAULT_SIZES = tuple(4096 << (2 * k) for k in range(8))          # 4 KiB, 16 KiB, ... 64 MiB
CONFIGURATIONS = {"ring": None, "ring8": None, "path4": (0, 1, 2, 3), "pair": (0, 1)}


class P2PPlanError(ValueError):
    """A point-to-point plan that cannot run as asked."""


@dataclasses.dataclass(frozen=True)
class P2POptions:
    sizes: tuple[int, ...] = DEFAULT_SIZES
    checked_iterations: int = 3
    latency_iterations: int = 200           # round trips per size up to 64 KiB, fewer above (latency_rounds)
    bandwidth_bytes: int = 256 << 20        # bytes per timed burst (at least 4 and at most 256 messages)
    bandwidth_repeats: int = 5
    warmup_iterations: int = 5
    seed: int = 20261007
    cpu_policy: str = "performance"
    lane_check_ms: int = 2000
    worker_timeout_s: int = 1800
    startup_wait_s: float = 300.0
    serving_wait_s: float = 20.0
    window_bytes: int = DEFAULT_WINDOW_BYTES      # SIRCL_P2P_WINDOW_BYTES: largest window of a relayed lane
    chunk_bytes: int = DEFAULT_CHUNK_BYTES        # SIRCL_P2P_CHUNK_BYTES
    session_env: tuple[tuple[str, str], ...] = ()   # other SIRCL_P2P_* settings for every rank

    def __post_init__(self) -> None:
        if not self.sizes or any(size < 1 or size > 1 << 30 for size in self.sizes):
            raise P2PPlanError("point-to-point sizes are 1 byte to 1 GiB")
        for name, value in self.session_env:
            if not name.startswith("SIRCL_P2P_"):
                raise P2PPlanError(f"--session-env takes SIRCL_P2P_* settings, got {name}")


def latency_rounds(size: int, iterations: int) -> int:
    """Round trips timed at ``size``: ``iterations`` up to 64 KiB, then about 64 MiB of traffic, at least 10."""
    if size <= 65536:
        return iterations
    return max(10, min(iterations, (64 << 20) // size))


def burst_messages(size: int, burst_bytes: int) -> int:
    return max(4, min(256, burst_bytes // max(size, 1)))


def default_cases(world: int, closed: bool) -> list[dict]:
    """Pairs from rank 0 to every distance up to the farthest, and shifts 1, 2 and ``world / 2`` on a cycle."""
    farthest = world // 2 if closed else world - 1
    cases = [{"kind": "pair", "ranks": [0, d]} for d in range(1, farthest + 1)]
    shifts = sorted({1, 2, world // 2} if closed and world >= 4 else {1})
    cases += [{"kind": "shift", "distance": d} for d in shifts if 1 <= d < world]
    return cases


def parse_cases(text: str, world: int) -> list[dict]:
    """``0-1,0-4,shift1,shift4``: pairs and shifts."""
    cases = []
    for item in (part.strip() for part in text.split(",") if part.strip()):
        if item.startswith("shift"):
            distance = int(item[len("shift"):])
            if not 1 <= distance < world:
                raise P2PPlanError(f"shift {distance} needs a distance from 1 to {world - 1}")
            cases.append({"kind": "shift", "distance": distance})
            continue
        try:
            a, b = (int(v) for v in item.split("-"))
        except ValueError:
            raise P2PPlanError(f"case {item!r} is neither A-B nor shiftD") from None
        if a == b or not (0 <= a < world and 0 <= b < world):
            raise P2PPlanError(f"pair {item} does not join two ranks of the group of {world}")
        cases.append({"kind": "pair", "ranks": [a, b]})
    if not cases:
        raise P2PPlanError("no cases given")
    return cases


@dataclasses.dataclass(frozen=True)
class P2PPlan:
    configuration: plan_mod.ConfigurationPlan
    options: P2POptions
    cases: tuple[dict, ...]
    windows: tuple[tuple[tuple[int, ...], ...], ...]     # [rank][peer][lane], as every rank derives them
    unavailable: tuple[tuple[int, int, str], ...]
    hops: tuple[tuple[int, ...], ...]                    # [rank][peer] cables crossed (relays + 1)

    @property
    def world(self) -> int:
        return self.configuration.world

    @property
    def run_dir(self) -> str:
        return self.configuration.run_dir

    def to_json(self) -> dict:
        document = self.configuration.to_json()
        document["schema"] = SCHEMA
        document["p2p"] = {"options": dataclasses.asdict(self.options), "cases": list(self.cases),
                           "windows": [[list(lanes) for lanes in row] for row in self.windows],
                           "unavailable": [list(item) for item in self.unavailable],
                           "hops": [list(row) for row in self.hops]}
        return document


def build(site: Site, run_id: str, *, positions: Sequence[int] | None = None, name: str = "p2p",
          cases: str | None = None, options: P2POptions = P2POptions(), digest: str | None = None) -> P2PPlan:
    """The plan of one point-to-point run: the group of ``positions`` (default the whole ring)."""
    members = tuple(range(site.size)) if positions is None else tuple(int(p) for p in positions)
    configuration = plan_mod.build_plan(
        site, name, run_id, groups=(members,), digest=digest,
        options=plan_mod.Options(cpu_policy=options.cpu_policy, lane_check_ms=options.lane_check_ms,
                                 worker_timeout_s=options.worker_timeout_s, startup_wait_s=options.startup_wait_s,
                                 serving_wait_s=options.serving_wait_s))
    (group,) = configuration.groups
    layout = routes_mod.Layout.parse(group.layout)
    world = layout.world
    closed = len(members) == site.size and site.size > 2
    chosen = parse_cases(cases, world) if cases else default_cases(world, closed)
    lanes = budget.LaneSet.of("p2p", layout, 2)
    table, unavailable = budget.group_windows(lanes, 2, max_window=options.window_bytes, chunk=options.chunk_bytes)
    routes = routes_mod.derive_routes(layout, 2)
    hops = tuple(tuple(0 if peer == rank else max(len(lane.relays) for lane in routes.ranks[rank][peer]) + 1
                       for peer in range(world)) for rank in range(world))
    return P2PPlan(configuration, options, tuple(chosen),
                   tuple(tuple(tuple(int(w) for w in lanes_) for lanes_ in row) for row in table),
                   tuple(sorted((int(a), int(b), str(why)) for (a, b), why in unavailable.items())), hops)


def render_text(plan: P2PPlan) -> str:
    configuration = plan.configuration
    (group,) = configuration.groups
    options = plan.options
    lines = [f"point-to-point run {configuration.run_id}: {plan.world} ranks on Sparks {list(group.positions)} "
             f"({group.layout})",
             f"  image {configuration.image}; GPU 0; host networking; control exchange "
             f"tcp://{configuration.leader_address}:{configuration.control_port}",
             f"  sizes {', '.join(_size_text(s) for s in options.sizes)}; checked {options.checked_iterations} per "
             f"direction; latency up to {options.latency_iterations} round trips; bandwidth bursts of "
             f"{_size_text(options.bandwidth_bytes)} x {options.bandwidth_repeats}",
             f"  forward windows: at most {options.window_bytes} B per relayed lane in chunks of {options.chunk_bytes} "
             f"B, within {routes_mod.RELAY_QUEUE_SHARE:.0%} of every {routes_mod.DEFAULT_HAIRPIN_QUEUE >> 10} KiB "
             "relay hairpin queue"]
    for case in plan.cases:
        if case["kind"] == "pair":
            a, b = case["ranks"]
            relays = plan.hops[a][b] - 1
            lanes = ", ".join(f"{w} B" if w else "direct" for w in plan.windows[a][b])
            back = ", ".join(f"{w} B" if w else "direct" for w in plan.windows[b][a])
            lines.append(f"  pair {a}-{b} (Sparks {group.positions[a]}-{group.positions[b]}): {relays} relay(s); "
                         f"windows {a}->{b} [{lanes}], {b}->{a} [{back}]")
        else:
            d = case["distance"]
            lines.append(f"  shift {d}: every rank sends to rank r+{d} and receives from rank r-{d} at once")
    for a, b, why in plan.unavailable:
        lines.append(f"  NO CHANNEL {a}->{b}: {why}")
    for rank in configuration.ranks:
        lines.append(f"  rank {rank.global_rank}: {rank.host} ({rank.ssh}), container {rank.container}, routes "
                     f"{rank.peer_routes}")
    lines.append(f"  results: {configuration.run_dir}/p2p-rank-<N>.json on each Spark, collected into "
                 "--output/<run id>/p2p")
    return "\n".join(lines)


def _size_text(size: int) -> str:
    for unit, shift in (("MiB", 20), ("KiB", 10)):
        if size >= 1 << shift and size % (1 << shift) == 0:
            return f"{size >> shift} {unit}"
    return f"{size} B"


# -- launch ---------------------------------------------------------------------------------


def worker_command(plan: P2PPlan, rank: plan_mod.RankPlan) -> str:
    run = f"/sircl/runs/{plan.configuration.run_id}/{plan.configuration.name}"
    return (f"exec python3 -m sparkring_sircl.ring.p2p_worker --plan {run}/p2p-plan.json "
            f"--global-rank {rank.global_rank} > {run}/p2p-rank-{rank.global_rank}.log 2>&1")


def docker_run(plan: P2PPlan, rank: plan_mod.RankPlan) -> str:
    """One rank's container, with the ring harness's label, mounts and environment."""
    configuration = plan.configuration
    environment = {
        "PYTHONPATH": f"/sircl/src/{configuration.source_digest}",
        "SIRCL_BUILD_CACHE_DIR": "/sircl/build-cache",
        "CUTE_DSL_CACHE_DIR": "/sircl/cute-cache",
        "GLOO_SOCKET_IFNAME": configuration.lan_interface,
        "CUDA_VISIBLE_DEVICES": "0",
    }
    q = remote.q
    parts = [remote.docker_command(rank.docker), "run", "-d", "--name", q(rank.container),
             "--label", q(f"{remote.LABEL}={configuration.run_id}"),
             "--privileged", "--gpus", q("device=0"), "--network", "host", "--ipc", "host",
             "--ulimit", "memlock=-1", "--entrypoint", "bash", "-v", q(f"{configuration.remote_dir}:/sircl")]
    for key, value in environment.items():
        parts += ["-e", q(f"{key}={value}")]
    parts += [q(configuration.image), "-lc", q(worker_command(plan, rank))]
    return " ".join(parts)


def _states(site: Site, plan: P2PPlan) -> dict[int, tuple[str, int]]:
    states = {}
    for rank in plan.configuration.ranks:
        answer = remote.ssh(site.host(rank.position).ssh,
                            f"echo \"{rank.global_rank} $({remote.container_state(rank.container, docker=rank.docker)})\"",
                            timeout=30)
        for line in answer.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                code = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else -1
                states[int(parts[0])] = (parts[1], code)
    return states


def run(site: Site, plan: P2PPlan, output: Path, *, force: bool, timeout: float) -> bool:
    configuration = plan.configuration
    folder = output / configuration.name
    folder.mkdir(parents=True, exist_ok=True)
    document = plan.to_json()
    (folder / "p2p-plan.json").write_text(json.dumps(document, indent=1), encoding="utf-8")
    (folder / "p2p-plan.txt").write_text(render_text(plan) + "\n", encoding="utf-8")
    positions = sorted({rank.position for rank in configuration.ranks})
    for position in positions:
        host = site.host(position)
        running = remote.ssh(host.ssh, remote.running_containers(docker=host.docker), timeout=30)
        if not running.ok:
            print(f"{host.name}: cannot list containers: {running.stderr.strip()}")
            return False
        others = [line.split("\t")[0] for line in running.stdout.splitlines()
                  if line.strip() and not line.split("\t")[-1].strip()]
        if others and not force:
            print(f"{host.name}: refusing to run while containers run ({', '.join(others)}); pass --force")
            return False
        staged = remote.ssh(host.ssh, f"test -d {remote.q(site.remote_dir + '/src/' + configuration.source_digest)}",
                            timeout=30)
        if not staged.ok:
            print(f"{host.name}: sources {configuration.source_digest} are not staged; run "
                  "`python -m sparkring_sircl.ring stage --site ... --config ring` from this checkout")
            return False
    ok = False
    try:
        payload = json.dumps(document, indent=1).encode()
        for position in positions:
            written = remote.ssh(site.host(position).ssh, remote.write_file(f"{plan.run_dir}/p2p-plan.json"),
                                 input_bytes=payload, timeout=30)
            if not written.ok:
                print(f"{site.host(position).name}: writing the plan failed: {written.stderr.strip()}")
                return False
        for rank in configuration.ranks:
            launched = remote.ssh(rank.ssh, docker_run(plan, rank), timeout=60)
            if not launched.ok:
                print(f"rank {rank.global_rank} ({rank.host}): container did not start: {launched.stderr.strip()}")
                return False
        print(f"{configuration.name}: {plan.world} containers started; waiting up to {timeout:.0f} s")
        deadline = time.monotonic() + timeout
        failed_at = None
        while True:
            states = _states(site, plan)
            exited = {r: s for r, s in states.items() if s[0] in ("exited", "dead", "missing")}
            if any(code != 0 for _, code in exited.values()) and failed_at is None:
                failed_at = time.monotonic()
                print(f"a rank failed: {exited}; waiting 60 s for the others to stop")
            if len(exited) == plan.world:
                ok = all(code == 0 for _, code in exited.values())
                break
            if (failed_at is not None and time.monotonic() - failed_at > 60) or time.monotonic() > deadline:
                break
            time.sleep(5)
        return ok
    finally:
        for rank in configuration.ranks:
            for suffix in ("json", "log"):
                answer = remote.ssh(rank.ssh, remote.read_file(f"{plan.run_dir}/p2p-rank-{rank.global_rank}.{suffix}"),
                                    timeout=60)
                if answer.stdout:
                    (folder / f"p2p-rank-{rank.global_rank}.{suffix}").write_text(answer.stdout, encoding="utf-8")
        for position in positions:
            host = site.host(position)
            left = remote.ssh(host.ssh, remote.remove_harness_containers(configuration.run_id, docker=host.docker),
                              timeout=60)
            if not left.ok or left.stdout.strip() != "0":
                print(f"{host.name}: harness containers may remain; run `python -m sparkring_sircl.ring cleanup`")
        merged = merge(document, load_results(folder, plan.world))
        (folder / "p2p-result.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
        text = table(merged)
        (folder / "p2p-summary.txt").write_text(text + "\n", encoding="utf-8")
        print(text)


# -- results ------------------------------------------------------------------------------------


def load_results(folder: Path, world: int) -> list[dict | None]:
    results = []
    for rank in range(world):
        path = folder / f"p2p-rank-{rank}.json"
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            results.append(None)
    return results


def merge(plan: dict, results: Sequence[dict | None]) -> dict:
    """One record per case and size: rank A's timings of a pair, every rank's of a shift, every check."""
    problems = []
    rows: dict[tuple, dict] = {}
    for rank, result in enumerate(results):
        if result is None:
            problems.append(f"rank {rank} wrote no result")
            continue
        if result.get("error"):
            problems.append(f"rank {rank}: {result['error']}")
        for record in result.get("records", ()):
            key = (record["case"], record["bytes"])
            row = rows.setdefault(key, {"case": record["case"], "bytes": record["bytes"], "hops": record.get("hops"),
                                        "checked": 0, "mismatches": 0, "skipped": None, "ranks": {}})
            row["checked"] += record.get("checked", 0)
            row["mismatches"] += record.get("mismatches", 0)
            if record.get("skipped"):
                row["skipped"] = record["skipped"]
            if record.get("latency") or record.get("bandwidth_gbps") is not None:
                row["ranks"][str(rank)] = {k: record[k] for k in ("latency", "bandwidth_gbps") if k in record}
    ordered = sorted(rows.values(), key=lambda row: (row["case"], row["bytes"]))
    mismatched = sum(row["mismatches"] for row in ordered)
    status = "passed" if not problems and not mismatched and len(results) == plan.get("world") else "failed"
    return {"schema": "sircl-ring-p2p-result/v1", "run_id": plan.get("run_id"), "world": plan.get("world"),
            "status": status, "problems": problems, "rows": ordered,
            "channels": [result.get("channels") if result else None for result in results]}


def table(merged: dict) -> str:
    lines = [f"point-to-point run {merged['run_id']}: {merged['status']}"]
    lines += [f"  problem: {problem}" for problem in merged["problems"]]
    lines.append(f"  {'case':<12} {'bytes':>10} {'hops':>4} {'checked':>7}  {'one-way p50/p99 us':>20}  "
                 f"{'GB/s (min of ranks)':>19}")
    for row in merged["rows"]:
        if row["skipped"]:
            lines.append(f"  {row['case']:<12} {row['bytes']:>10} {'':>4} {'-':>7}  skipped: {row['skipped']}")
            continue
        latency = [r["latency"] for r in row["ranks"].values() if r.get("latency")]
        rates = [r["bandwidth_gbps"] for r in row["ranks"].values() if r.get("bandwidth_gbps") is not None]
        lat = f"{latency[0]['p50_us']:.1f}/{latency[0]['p99_us']:.1f}" if latency else "-"
        rate = f"{min(rates):.2f}" if rates else "-"
        checked = f"{row['checked']}" + (f" ({row['mismatches']} differ)" if row["mismatches"] else "")
        lines.append(f"  {row['case']:<12} {row['bytes']:>10} {row['hops'] or '':>4} {checked:>7}  {lat:>20}  "
                     f"{rate:>19}")
    return "\n".join(lines)


# -- entry point ------------------------------------------------------------------------------


def _sizes(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(v) for v in text.split(",") if v.strip())
    except ValueError:
        raise P2PPlanError(f"--sizes takes byte counts separated by commas, got {text!r}") from None


def _positions(text: str) -> tuple[int, ...]:
    if "-" in text and "," not in text:
        first, last = (int(v) for v in text.split("-"))
        return tuple(range(first, last + 1))
    return tuple(int(v) for v in text.split(","))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.ring.p2p", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run", "summarize"):
        sub = commands.add_parser(name)
        if name == "summarize":
            sub.add_argument("--results", required=True, type=Path, help="the run's p2p folder")
            continue
        sub.add_argument("--site", required=True, type=Path, help="site description (sircl-ring-site/v1)")
        sub.add_argument("--config", default="ring", choices=sorted(CONFIGURATIONS),
                         help="ring (every Spark), path4 (Sparks 0-3) or pair (Sparks 0-1)")
        sub.add_argument("--positions", help="consecutive Sparks instead of --config, e.g. 2-5")
        sub.add_argument("--cases", help="pairs and shifts in group ranks, e.g. 0-1,0-4,shift1,shift4 (default: "
                                         "rank 0 to every distance, and shifts)")
        sub.add_argument("--sizes", help="message sizes in bytes, comma-separated (default 4 KiB to 64 MiB in powers "
                                         "of four)")
        sub.add_argument("--latency-iterations", type=int, default=P2POptions.latency_iterations)
        sub.add_argument("--bandwidth-bytes", type=int, default=P2POptions.bandwidth_bytes)
        sub.add_argument("--window", type=int, default=DEFAULT_WINDOW_BYTES,
                         help="SIRCL_P2P_WINDOW_BYTES: largest forward window of a relayed lane")
        sub.add_argument("--session-env", action="append", metavar="NAME=VALUE",
                         help="a SIRCL_P2P_* setting for every rank (repeatable), e.g. SIRCL_P2P_SLOT_BYTES=1048576")
        sub.add_argument("--cpu-policy", choices=plan_mod.CPU_POLICIES, default="performance")
        sub.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
        if name == "plan":
            sub.add_argument("--json", action="store_true")
        else:
            sub.add_argument("--print", action="store_true", help="print the launch plan and contact nothing")
            sub.add_argument("--force", action="store_true", help="proceed although other containers run")
            sub.add_argument("--output", type=Path, default=Path("sircl-ring-results"))
            sub.add_argument("--timeout", type=float, default=2400)
    args = parser.parse_args(argv)
    if args.command == "summarize":
        document = json.loads((args.results / "p2p-plan.json").read_text(encoding="utf-8"))
        merged = merge(document, load_results(args.results, document["world"]))
        print(table(merged))
        return 0 if merged["status"] == "passed" else 1
    try:
        site = Site.load(args.site)
        positions = _positions(args.positions) if args.positions else CONFIGURATIONS[args.config]
        session_env = []
        for item in args.session_env or ():
            key, _, value = item.partition("=")
            session_env.append((key.strip(), value.strip()))
        options = P2POptions(sizes=_sizes(args.sizes) if args.sizes else DEFAULT_SIZES,
                             latency_iterations=args.latency_iterations, bandwidth_bytes=args.bandwidth_bytes,
                             window_bytes=args.window, cpu_policy=args.cpu_policy, session_env=tuple(session_env))
        plan = build(site, args.run_id, positions=positions, cases=args.cases, options=options)
    except (SiteError, plan_mod.PlanError, P2PPlanError, routes_mod.RouteError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if args.command == "plan":
        print(json.dumps(plan.to_json(), indent=1) if args.json else render_text(plan))
        return 0
    print(render_text(plan))
    if args.print:
        for rank in plan.configuration.ranks:
            print(f"  ssh {rank.ssh} {docker_run(plan, rank)}")
        return 0
    output = args.output / args.run_id
    ok = run(site, plan, output, force=args.force, timeout=args.timeout)
    print(f"results in {output / plan.configuration.name}")
    return 0 if ok else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
