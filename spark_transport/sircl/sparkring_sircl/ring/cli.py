"""Operator commands of the ring harness: ``python -m sparkring_sircl.ring <command>``.

Commands and their safety classes (RUNBOOK.md has the procedure):

- ``plan`` (OFFLINE): print the launch plan of the chosen configurations;
- ``preflight`` (READ-ONLY REMOTE): check every Spark the configurations use;
- ``stage`` (MUTATES HOST, files under the site's remote directory and one
  short-lived build container): copy the package sources and build the native
  library inside the serving image;
- ``run`` (MUTATES HOST, starts one container per rank on GPU 0; refuses while
  any other container runs unless ``--force``): run configurations, collect
  results, remove the containers;
- ``cleanup`` (MUTATES HOST, removes harness-labelled containers only);
- ``summarize`` (OFFLINE): merge collected results and print the table.

``run --print`` prints the plan and contacts nothing.

The site file may describe a ring, a path of Sparks or explicit cables such as
a pair cabled port 0 to port 0 (``cabling``, :mod:`.site`). Preflight requires
an active RDMA function only on the ports the site cables, so the end Sparks
of a path, each with one free port, pass with two functions down.
``--worker-timeout`` sets the rank watchdog (default 1,500 s) and, without
``--timeout``, the run waits for its containers at least 300 s longer than
the watchdog (default 1,800 s). ``--chain-slot-bytes`` sets every session's
chain slot (``SIRCL_CHAIN_SLOT_BYTES``), the largest chain chunk.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import io
import ipaddress
import json
import sys
import tarfile
import time
from pathlib import Path

from .. import routes as routes_mod
from . import plan as plan_mod
from . import remote, summary
from . import trace as trace_mod
from .site import CABLINGS, Site, SiteError


def _configurations(args, site: Site) -> list[tuple[str, tuple[tuple[int, ...], ...] | None]]:
    if args.groups:
        return [(args.name or "custom", plan_mod.parse_groups(args.groups))]
    names = [name.strip() for name in args.config.split(",") if name.strip()]
    if names == ["all"]:
        if site.fabric().kind == "cycle":
            names = ["pairs", "path4", "two-tp4", "ring"] if site.size >= 8 else ["pairs", "ring"]
        else:
            names = (["pairs", "path4", "two-tp4", "path"] if site.size >= 8 else ["pairs", "path"] if site.size > 2
                     else ["path"])
    return [(name, None) for name in names]


def _options(args) -> plan_mod.Options:
    base = plan_mod.Options()
    options = dataclasses.replace(_base_options(args, base), **_tune_options(args, base))
    if getattr(args, "worker_timeout", None) is not None and options.worker_timeout_s <= options.startup_wait_s:
        raise plan_mod.PlanError(f"--worker-timeout {options.worker_timeout_s} s must exceed the setup wait limit "
                                 f"({options.startup_wait_s:g} s, --startup-wait): a rank still waiting at setup "
                                 "would be ended before its own wait names the peer it waits for")
    return options


CHAIN_SLOT_ALIGNMENT = 4096
CHAIN_SLOT_LIMIT = 1 << 31


def _chain_slot_env(args) -> list[tuple[str, str]]:
    """``--chain-slot-bytes`` as the session variable SIRCL_CHAIN_SLOT_BYTES of every rank (none when unset)."""
    value = getattr(args, "chain_slot_bytes", None)
    if value is None:
        return []
    if value <= 0 or value % CHAIN_SLOT_ALIGNMENT or value > CHAIN_SLOT_LIMIT:
        raise plan_mod.PlanError(f"--chain-slot-bytes {value} must be a positive multiple of {CHAIN_SLOT_ALIGNMENT} "
                                 f"up to {CHAIN_SLOT_LIMIT}")
    if any(item.partition("=")[0].strip() == "SIRCL_CHAIN_SLOT_BYTES" for item in args.session_env or ()):
        raise plan_mod.PlanError("--chain-slot-bytes and --session-env SIRCL_CHAIN_SLOT_BYTES both set the chain "
                                 "slot; give one")
    return [("SIRCL_CHAIN_SLOT_BYTES", str(value))]


def _base_options(args, base: plan_mod.Options) -> plan_mod.Options:
    return plan_mod.Options(
        correctness_iterations=args.correctness_iterations or base.correctness_iterations,
        eager_iterations=args.eager_iterations or base.eager_iterations,
        graph_iterations=args.graph_iterations or base.graph_iterations,
        spin_limit=args.spin_limit or base.spin_limit,
        worker_timeout_s=(base.worker_timeout_s if getattr(args, "worker_timeout", None) is None
                          else args.worker_timeout),
        transport_only=bool(args.transport_only),
        cpu_policy=args.cpu_policy,
        large=bool(args.large),
        large_capacity=args.large_capacity or base.large_capacity,
        rotate_buffers=args.rotate_buffers or base.rotate_buffers,
        large_iterations=args.large_iterations or base.large_iterations,
        large_piece_bytes=args.large_piece,
        startup_wait_s=args.startup_wait or base.startup_wait_s,
        large_schedules=tuple(args.large_schedules.split(",")) if args.large_schedules else base.large_schedules,
        chain_chunks=tuple(int(value) for value in args.chain_chunks.split(",")) if args.chain_chunks else (),
        serving_wait_s=args.serving_wait or base.serving_wait_s,
        forward_window=args.forward_window,
        dcp_decode_rows=_row_counts(args.dcp_decode_rows),
        dcp_prefill_rows=_row_counts(args.dcp_prefill_rows),
        swing_sizes=_row_counts(args.swing_sizes) or (),
        latency_sizes=_row_counts(args.latency_sizes) or (),
        post_orders=tuple(item.strip() for item in args.post_orders.split(",") if item.strip())
        if args.post_orders else (),
        latency_relay_us=base.latency_relay_us if args.relay_us is None else args.relay_us,
        latency_post_us=base.latency_post_us if args.post_us is None else args.post_us,
        latency_write_us=base.latency_write_us if args.write_us is None else args.write_us,
        session_env=tuple(_session_env(args.session_env or ()) + _chain_slot_env(args)),
        tuning_tables=tuple(str(path) for path in (getattr(args, "tuning_table", None) or ())),
        eager_profile=bool(getattr(args, "eager_profile", False)),
        eager_path=getattr(args, "eager_path", None) or "session",
        large_blocks=_grid_caps(args.large_blocks),
        baseline=args.baseline or "",
        nccl_library=args.nccl_library or "",
        nccl_env=tuple(_session_env(args.nccl_env or ())),
        gather_schedules=(tuple(args.gather_schedules.split(",")) if args.gather_schedules
                          else base.gather_schedules),
        link_chunks=tuple(int(value) for value in args.link_chunks.split(",")) if args.link_chunks else (),
        gather_link_chunks=_pieces(args.gather_link_chunks, args.session_env, "SIRCL_GATHER_LINK_CHUNK_BYTES"),
        scatter_link_chunks=_pieces(args.scatter_link_chunks, args.session_env, "SIRCL_SCATTER_LINK_CHUNK_BYTES"),
        reduce_link_chunks=_pieces(args.reduce_link_chunks, args.session_env, "SIRCL_REDUCE_LINK_CHUNK_BYTES"),
        scatter_schedules=(tuple(args.scatter_schedules.split(",")) if args.scatter_schedules
                           else base.scatter_schedules),
        host_send_gbps=args.host_send_gbps or args.host_cap_gbps or base.host_send_gbps,
        host_recv_gbps=args.host_recv_gbps or args.host_cap_gbps or base.host_recv_gbps,
        cable_gbps=args.cable_gbps or base.cable_gbps,
    )


def _pieces(text, session_env, name) -> tuple[int, ...]:
    """A collective's swept link pieces: the option's list, else the piece its variable sets through
    ``--session-env``, else none (the common sweep)."""
    if text:
        return tuple(int(value) for value in text.split(","))
    for item in session_env or ():
        key, _, value = item.partition("=")
        if key == name and value.isdigit():
            return (int(value),)
    return ()


def _session_env(items) -> list[tuple[str, str]]:
    """``--session-env NAME=VALUE`` items as (name, value) pairs; the plan checks the names."""
    pairs = []
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name.strip():
            raise plan_mod.PlanError(f"--session-env takes NAME=VALUE, got {item!r}")
        pairs.append((name.strip(), value.strip()))
    return pairs


def _tune_options(args, base: plan_mod.Options) -> dict:
    """The tune fields of the options (the ``tune`` command, or ``plan --tune``), with the tune command's
    shorter default call counts where the line names none."""
    if not (getattr(args, "command", None) == "tune" or getattr(args, "tune", False)):
        return {}

    def numbers(text, default):
        return tuple(int(item) for item in text.split(",") if item.strip()) if text else default

    sizes = numbers(args.tune_sizes, plan_mod.TUNE_QUICK_SIZES if args.quick else plan_mod.TUNE_SIZES)
    return {
        "tune": True,
        "tune_collectives": tuple(args.tune_collectives.split(",")) if args.tune_collectives else base.tune_collectives,
        "tune_sizes": sizes,
        "tune_modes": tuple(args.tune_modes.split(",")) if args.tune_modes else base.tune_modes,
        "tune_grids": numbers(args.tune_grids, base.tune_grids),
        "tune_pieces": numbers(args.tune_pieces, base.tune_pieces),
        "tune_staggers": numbers(args.tune_staggers, base.tune_staggers),
        "tune_link_blocks": numbers(args.tune_link_blocks, base.tune_link_blocks),
        "tune_chain_blocks": numbers(args.tune_chain_blocks, base.tune_chain_blocks),
        "rotate_buffers": args.rotate_buffers or plan_mod.TUNE_ROTATE_BUFFERS,
        "tune_prune": args.tune_prune or base.tune_prune,
        "tune_prune_from": base.tune_prune_from if args.tune_prune_from is None else args.tune_prune_from,
        "correctness_iterations": args.correctness_iterations or 1,
        "eager_iterations": args.eager_iterations or 40,
        "graph_iterations": args.graph_iterations or 100,
        "large_iterations": args.large_iterations or 8,
        "warmup_iterations": 5,
    }


def _grid_caps(text: str | None) -> tuple[int, ...]:
    """Grid caps of ``--large-blocks``."""
    if not text:
        return ()
    try:
        return tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError:
        raise plan_mod.PlanError(f"grid caps {text!r} are not comma-separated integers") from None


def _sizes(text: str) -> tuple[int, ...]:
    """Byte counts of ``--large-sizes``."""
    try:
        return tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError:
        raise plan_mod.PlanError(f"sizes {text!r} are not comma-separated integers") from None


def _row_counts(text: str | None) -> tuple[int, ...] | None:
    """Row counts of ``--dcp-decode-rows`` / ``--dcp-prefill-rows``: unset keeps the configuration's, ``none``
    runs none."""
    if text is None:
        return None
    if text.strip().lower() in ("", "none"):
        return ()
    try:
        return tuple(int(item) for item in text.split(",") if item.strip())
    except ValueError:
        raise plan_mod.PlanError(f"row counts {text!r} are not comma-separated integers") from None


def _plans(args, site: Site) -> list[plan_mod.ConfigurationPlan]:
    digest = plan_mod.source_digest()
    options = _options(args)
    plans = []
    for name, groups in _configurations(args, site):
        configured = plan_mod.configuration_options(name, options)
        if args.large_schedules:
            # Schedules named on the command line replace a configuration's own sweep.
            configured = dataclasses.replace(configured, large_schedules=options.large_schedules)
        if args.gather_schedules:
            configured = dataclasses.replace(configured, gather_schedules=options.gather_schedules)
        if args.scatter_schedules:
            configured = dataclasses.replace(configured, scatter_schedules=options.scatter_schedules)
        if args.large_sizes:
            # Sizes named on the command line replace a configuration's large all-reduce sizes.
            configured = dataclasses.replace(configured, large_allreduce_sizes=_sizes(args.large_sizes))
        if args.large_gather_sizes:
            configured = dataclasses.replace(configured, large_allgather_sizes=_sizes(args.large_gather_sizes))
        for flag, field in (("chain_gather_sizes", "chain_gather_sizes"),
                            ("reduce_scatter_sizes", "large_reduce_scatter_sizes")):
            text = getattr(args, flag, None)
            if text:
                if not configured.large:
                    raise plan_mod.PlanError(f"configuration {name}: --{flag.replace('_', '-')} adds large-message "
                                             "cases; pass --large or name a -large configuration")
                configured = dataclasses.replace(configured, **{field: _sizes(text)})
        given = [option for option in plan_mod.CASE_OPTIONS
                 if getattr(args, option[2:].replace("-", "_"), None)]
        unused = plan_mod.unused_case_options(configured, given)
        if unused:
            raise plan_mod.PlanError(f"configuration {name}: " + "; ".join(unused))
        plans.append(plan_mod.build_plan(site, name, args.run_id, groups=groups, options=configured,
                                         digest=digest))
    if options.tuning_tables:
        matched = {digest for built in plans for digest in [group.tuning_table for group in built.groups]
                   + [built.world_tuning_table] if digest}
        unused = [source for digest, source in plans[0].tuning_tables if digest not in matched] if plans else []
        if unused:
            raise plan_mod.PlanError(f"no session of the planned configurations matches the tuning table(s) "
                                     f"{', '.join(unused)} (group shape, size, lanes, relays or build differ)")
    return plans


def _hosts(plans: list[plan_mod.ConfigurationPlan], site: Site) -> list[int]:
    positions = sorted({rank.position for plan in plans for rank in plan.ranks})
    return positions or list(range(site.size))


def _parse_lines(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        key, separator, value = line.partition("\t")
        if separator:
            values[key.strip()] = value.strip()
    return values


# -- preflight ----------------------------------------------------------------------


def fabric_netdevs(values: dict[str, str]) -> set[str]:
    """Network devices of the Spark's four RDMA functions.

    The canonical names of ``routes.ROLES`` plus the names the Spark reports in
    its ``device:<rdma device>`` lines (``<n>: <STATE> <netdev>``). Other
    interfaces, such as Wi-Fi or a USB Ethernet adapter on the same home LAN,
    are not fabric interfaces.
    """
    names = {role.netdev for role in routes_mod.ROLES}
    for role in routes_mod.ROLES:
        words = values.get(f"device:{role.device}", "").split()
        if len(words) >= 3:
            names.add(words[-1])
    return names


def preflight(site: Site, plans: list[plan_mod.ConfigurationPlan], *, force: bool) -> tuple[bool, list[str]]:
    """Read-only checks of every Spark; returns (ok, report lines)."""
    report, blockers = [], []
    facts: dict[int, dict[str, str]] = {}
    for position in _hosts(plans, site):
        host = site.host(position)
        result = remote.ssh(host.ssh, remote.preflight_script(site.image, site.lan_interface, docker=host.docker),
                            timeout=60)
        if not result.ok and not result.stdout:
            blockers.append(f"{host.name}: unreachable over ssh ({result.stderr.strip()})")
            continue
        values = _parse_lines(result.stdout)
        facts[position] = values
        running = remote.ssh(host.ssh, remote.running_containers(docker=host.docker), timeout=30)
        others = [line.split("\t")[0] for line in running.stdout.splitlines()
                  if line.strip() and not line.split("\t")[-1].strip()]
        docker_state = values.get("docker", "missing")
        docker_usable = docker_state not in ("", "missing") and not docker_state.startswith("unusable")
        report.append(f"{host.name}: docker command \"{host.docker}\" "
                      f"({'server ' + docker_state if docker_usable else 'unusable'}), "
                      f"image {values.get('image', 'missing')[:19]}, gpu {values.get('gpu', 'missing')}, "
                      f"lan {values.get('lan', 'missing')}, running containers {others or 'none'}")
        if not docker_usable:
            blockers.append(f"{host.name}: the docker command \"{host.docker}\" cannot reach the Docker daemon "
                            f"({docker_state}); give the SSH user Docker access or set this Spark's \"docker\" "
                            "in the site file, for example \"sudo -n docker\"")
        elif values.get("image", "missing") == "missing":
            blockers.append(f"{host.name}: image {site.image} is not present")
        if values.get("gpu", "missing") in ("missing", ""):
            blockers.append(f"{host.name}: no GPU reported by nvidia-smi")
        if others and not force:
            blockers.append(f"{host.name}: containers are running ({', '.join(others)}); stop them or pass --force")
        lan = values.get("lan", "")
        if lan.split("/")[0] != host.lan_address:
            blockers.append(f"{host.name}: {site.lan_interface} has {lan or 'no IPv4 address'}, the site lists "
                            f"{host.lan_address}")
        # Only the functions of the ports the site cables must be active: an end Spark of a path, or a Spark of
        # a pair cabled port 0 to port 0, has a port without a cable, whose two functions are down.
        cabled = site.cabled_ports(position)
        free = []
        for role in routes_mod.ROLES:
            state = values.get(f"device:{role.device}", "missing")
            if role.port not in cabled:
                free.append(f"{role.device} {state or 'missing'}")
                continue
            if "ACTIVE" not in state:
                blockers.append(f"{host.name}: RDMA device {role.device} is {state or 'missing'}")
        if free:
            ports = ", ".join(str(port) for port in (0, 1) if port not in cabled)
            report.append(f"{host.name}: port {ports} holds no cable of the site; its RDMA functions are not "
                          f"required ({'; '.join(free)})")
        fabric = fabric_netdevs(values)
        for key, value in values.items():
            interface = key[len("address:"):]
            if key.startswith("address:") and interface in fabric:
                try:
                    if ipaddress.IPv4Address(host.lan_address) in ipaddress.IPv4Interface(value).network:
                        blockers.append(f"{host.name}: the wired-LAN address {host.lan_address} lies in the "
                                        f"subnet of the fabric interface {interface} ({value}); the control "
                                        "exchange must not use fabric addresses")
                except ValueError:
                    pass
    # Every lane's destination must route over the lane's own device (relay plan origin routes), the
    # world session's lanes included.
    for plan in plans:
        sessions = [(group.layout, group.positions, group.lanes, "") for group in plan.groups]
        if plan.world_layout:
            world = routes_mod.Layout.parse(plan.world_layout)
            sessions.append((plan.world_layout, world.positions, 2, "world session "))
        for layout_text, positions, lane_count, label in sessions:
            routes = routes_mod.derive_routes(routes_mod.Layout.parse(layout_text), lane_count)
            for rank_index, position in enumerate(positions):
                if position not in facts:
                    continue
                checks: list[tuple[str, str, str]] = []
                for peer, lanes in sorted(routes.ranks[rank_index].items()):
                    peer_facts = facts.get(positions[peer], {})
                    for lane in lanes:
                        address = peer_facts.get(f"address:{lane.remote.netdev}", "")
                        if not address:
                            blockers.append(f"{site.host(positions[peer]).name}: {lane.remote.netdev} has no "
                                            "IPv4 address")
                            continue
                        checks.append((address.split("/")[0], lane.local.netdev,
                                       f"{label}rank {rank_index} lane {lane.lane} to rank {peer} "
                                       f"({lane.hops} hop(s))"))
                if not checks:
                    continue
                host = site.host(position)
                answer = remote.ssh(host.ssh, remote.route_get([c[0] for c in checks]), timeout=30)
                found = _parse_lines(answer.stdout)
                for destination, netdev, what in checks:
                    line = found.get(f"route:{destination}", "")
                    if f" dev {netdev} " not in f" {line} ":
                        blockers.append(f"{host.name} (configuration {plan.name}): {what} needs {destination} "
                                        f"routed over {netdev}; ip route get says {line or 'nothing'}")
    return not blockers, report + [f"BLOCKER: {b}" for b in blockers]


# -- stage --------------------------------------------------------------------------


def _source_tar() -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, path in plan_mod.staged_files():
            archive.add(path, arcname=name, recursive=False)
    return buffer.getvalue()


def stage(site: Site, plans: list[plan_mod.ConfigurationPlan], run_id: str) -> bool:
    digest = plan_mod.source_digest()
    tar = _source_tar()
    ok = True
    for position in _hosts(plans, site):
        host = site.host(position)
        unpacked = remote.ssh(host.ssh, remote.extract_tar(f"{site.remote_dir}/src/{digest}"), input_bytes=tar,
                              timeout=120)
        if not unpacked.ok:
            print(f"{host.name}: copying sources failed: {unpacked.stderr.strip()}")
            ok = False
            continue
        built = remote.ssh(host.ssh, remote.prepare_command(site.image, site.remote_dir, run_id, digest,
                                                            docker=host.docker), timeout=900)
        print(f"{host.name}: {'built' if built.ok else 'BUILD FAILED'}: "
              f"{(built.stdout + built.stderr).strip().splitlines()[-3:]}")
        ok &= built.ok
    return ok


# -- run -----------------------------------------------------------------------------


def _states(site: Site, plan: plan_mod.ConfigurationPlan) -> dict[int, tuple[str, int]]:
    states = {}
    by_host: dict[int, list[plan_mod.RankPlan]] = {}
    for rank in plan.ranks:
        by_host.setdefault(rank.position, []).append(rank)
    for position, ranks in by_host.items():
        command = "; ".join(f"echo \"{r.global_rank} $({remote.container_state(r.container, docker=r.docker)})\""
                            for r in ranks)
        answer = remote.ssh(site.host(position).ssh, command, timeout=30)
        for line in answer.stdout.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                status = parts[1]
                code = int(parts[2]) if len(parts) > 2 and parts[2].lstrip("-").isdigit() else -1
                states[int(parts[0])] = (status, code)
    return states


def run_configuration(site: Site, plan: plan_mod.ConfigurationPlan, output: Path, *, force: bool,
                      timeout: float) -> bool:
    folder = output / plan.name
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "plan.json").write_text(json.dumps(plan.to_json(), indent=1), encoding="utf-8")
    (folder / "plan.txt").write_text(plan_mod.render_text(plan) + "\n", encoding="utf-8")
    positions = sorted({rank.position for rank in plan.ranks})
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
        staged = remote.ssh(host.ssh, f"test -d {remote.q(site.remote_dir + '/src/' + plan.source_digest)} && "
                            f"ls {remote.q(site.remote_dir + '/build-cache')}/*.so >/dev/null", timeout=30)
        if not staged.ok:
            print(f"{host.name}: sources {plan.source_digest} or the native build are not staged; run `stage`")
            return False
    tables = {}
    if plan.tuning_tables:
        from .. import tuning as tuning_mod

        for digest, source in plan.tuning_tables:
            data = Path(source).read_bytes()
            if tuning_mod.document_hash(json.loads(data)) != digest:
                print(f"the tuning table {source} changed after it was planned (hash {digest}); plan again")
                return False
            tables[f"tuning-{digest}.json"] = data
    started: list[plan_mod.RankPlan] = []
    ok = False
    try:
        plan_bytes = json.dumps(plan.to_json(), indent=1).encode()
        for position in positions:
            for file_name, data in [("plan.json", plan_bytes), *tables.items()]:
                written = remote.ssh(site.host(position).ssh, remote.write_file(f"{plan.run_dir}/{file_name}"),
                                     input_bytes=data, timeout=30)
                if not written.ok:
                    print(f"{site.host(position).name}: writing {file_name} failed: {written.stderr.strip()}")
                    return False
        for rank in plan.ranks:
            launched = remote.ssh(rank.ssh, remote.docker_run(plan, rank), timeout=60)
            if not launched.ok:
                print(f"rank {rank.global_rank} ({rank.host}): container did not start: {launched.stderr.strip()}")
                return False
            started.append(rank)
        print(f"{plan.name}: {len(started)} containers started; waiting up to {timeout:.0f} s")
        deadline = time.monotonic() + timeout
        failed_at = None
        while True:
            states = _states(site, plan)
            exited = {r: s for r, s in states.items() if s[0] in ("exited", "dead", "missing")}
            if any(code != 0 for _, code in exited.values()) and failed_at is None:
                failed_at = time.monotonic()
                print(f"{plan.name}: a rank failed: {exited}; waiting 60 s for the others to stop")
            if len(exited) == plan.world:
                ok = all(code == 0 for _, code in exited.values())
                break
            if failed_at is not None and time.monotonic() - failed_at > 60:
                break
            if time.monotonic() > deadline:
                print(f"{plan.name}: timed out after {timeout:.0f} s")
                break
            time.sleep(5)
        return ok
    finally:
        for rank in plan.ranks:
            for suffix in ("json", "log"):
                answer = remote.ssh(rank.ssh, remote.read_file(f"{plan.run_dir}/rank-{rank.global_rank}.{suffix}"),
                                    timeout=60)
                if answer.stdout:
                    (folder / f"rank-{rank.global_rank}.{suffix}").write_text(answer.stdout, encoding="utf-8")
        for position in positions:
            host = site.host(position)
            left = remote.ssh(host.ssh, remote.remove_harness_containers(plan.run_id, docker=host.docker), timeout=60)
            if not left.ok or left.stdout.strip() != "0":
                detail = left.stdout.strip() if left.ok else (left.stderr.strip() or f"exit code {left.returncode}")
                print(f"{host.name}: harness containers may remain ({detail}); run cleanup")
        merged = summary.merge(plan.to_json(), summary.load_results(folder, plan.world))
        (folder / "result.json").write_text(json.dumps(merged, indent=1), encoding="utf-8")
        text = summary.table(merged)
        (folder / "summary.txt").write_text(text + "\n", encoding="utf-8")
        print(text)


# -- tuning tables ------------------------------------------------------------------------


def write_tuning_tables(folder: Path) -> int:
    """Build the tuning table of every group of a tune run's configuration folder from its rank results
    (``tuning-group<index>.json`` next to them) and print it; returns how many it wrote."""
    from .. import tuning as tuning_mod

    plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
    merged = summary.merge(plan, summary.load_results(folder, plan["world"]))
    tables = summary.tuning_tables(plan, merged, created=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    for index, document in tables.items():
        path = folder / f"tuning-group{index}.json"
        path.write_text(json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(tuning_mod.render(document))
        for note in summary.tune_notes(merged, document, index):
            print(note)
        print(f"tuning table of group {index}: {path} (hash {tuning_mod.document_hash(document)})")
    if not tables:
        print(f"{folder}: no exact tune measurements, no tuning table")
    return len(tables)


# -- entry point -------------------------------------------------------------------------


def run_timeout(options: plan_mod.Options) -> float:
    """The run's wait for a configuration's containers when --timeout is unset: the default, or the rank
    watchdog plus a margin when that is longer, so a hung rank's watchdog fires first."""
    return float(max(plan_mod.RUN_TIMEOUT_S, options.worker_timeout_s + plan_mod.RUN_TIMEOUT_MARGIN_S))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m sparkring_sircl.ring", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "preflight", "stage", "run", "cleanup", "tune"):
        sub = commands.add_parser(name)
        sub.add_argument("--site", required=True, type=Path, help="site description (sircl-ring-site/v1)")
        sub.add_argument("--config", default="all",
                         help="pairs, path4, two-tp4, ring (or ring8), path4-large, two-tp4-large, ring-large "
                              "(or ring8-large), dcp4, ring-swing (or ring8-swing), ring-latency (or "
                              "ring8-latency), path4-latency, path4-crossover, path (every Spark of a site cabled "
                              "as a path or a pair), or all")
        sub.add_argument("--groups", help="custom groups instead of --config, e.g. '0-3;4-7'")
        sub.add_argument("--name", help="name of the custom configuration")
        sub.add_argument("--run-id", default=f"{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}",
                         help="the run's name in the results and on the Sparks (default: the time and this "
                              "process's id, unique among concurrent runs)")
        sub.add_argument("--correctness-iterations", type=int)
        sub.add_argument("--eager-iterations", type=int)
        sub.add_argument("--graph-iterations", type=int)
        sub.add_argument("--spin-limit", type=int)
        sub.add_argument("--worker-timeout", type=int, metavar="SECONDS",
                         help=f"the rank watchdog: a rank that has not finished after this many seconds records its "
                              f"error and ends (default {plan_mod.DEFAULT_WORKER_TIMEOUT_S}; above --startup-wait)")
        sub.add_argument("--chain-slot-bytes", type=int, metavar="BYTES",
                         help="SIRCL_CHAIN_SLOT_BYTES of every session: the chain slot, the largest chain chunk "
                              "(multiple of 4096; default the session's, 1048576); chain chunks and tune pieces "
                              "above it run no chain case")
        sub.add_argument("--transport-only", action="store_true",
                         help="CPU-staged ops through the native layer only (no CUDA kernels)")
        sub.add_argument("--cpu-policy", choices=plan_mod.CPU_POLICIES, default="performance",
                         help="performance: pin every rank to performance cores and give the progress "
                              "thread its own (default); none: no pinning")
        sub.add_argument("--large", action="store_true",
                         help="add the large-message cases (two-shot, all_reduce_large, all_gather_large)")
        sub.add_argument("--rotate-buffers", type=int,
                         help="input and output windows every timed case cycles through call by call, so a call "
                              "reads and writes buffers the previous N-1 calls did not touch (default 1, one "
                              f"buffer; tune: {plan_mod.TUNE_ROTATE_BUFFERS})")
        sub.add_argument("--large-capacity", type=int,
                         help=f"session capacity with --large (default {plan_mod.LARGE_CAPACITY})")
        sub.add_argument("--large-iterations", type=int, help="timed calls per large-message case (default 20)")
        sub.add_argument("--forward-window", type=int,
                         help="SIRCL_FORWARD_WINDOW_BYTES of every session (0 turns forward windows off)")
        sub.add_argument("--large-schedules",
                         help="SIRCL_LARGE_SCHEDULE of the large all-reduce cases, comma-separated from auto, "
                              "pieces, chain and ring (default auto; pieces,chain,ring in the -large "
                              "configurations)")
        sub.add_argument("--chain-chunks",
                         help="chain chunk sizes in bytes to sweep, comma-separated (default the session's; "
                              "262144,524288,1048576 in the -large configurations)")
        sub.add_argument("--baseline", choices=("nccl",),
                         help="nccl: NCCL's all-reduce and all-gather of every case's size and mode beside SIRCL's "
                              "(pairs and whole cycles)")
        sub.add_argument("--nccl-library",
                         help="the NCCL library of the baseline inside the image (default "
                              "/opt/sparkring/toolchain/nccl/lib/libnccl.so.2)")
        sub.add_argument("--nccl-env", action="append", metavar="NAME=VALUE",
                         help="an NCCL_* variable (or LD_PRELOAD) of the NCCL baseline's environment, in place "
                              "of its default (repeatable)")
        sub.add_argument("--large-gather-sizes",
                         help="all_gather_large shard sizes in bytes, comma-separated, in place of the "
                              "configuration's")
        sub.add_argument("--large-blocks",
                         help="grid caps of two-shot and large-message launches, comma-separated powers of two "
                              "(e.g. 4,8,16,32): every case those launches serve runs once per cap")
        sub.add_argument("--large-sizes",
                         help="large all-reduce message sizes in bytes, comma-separated, in place of the "
                              "configuration's (e.g. 196608,393216,589824 with ring8-large)")
        sub.add_argument("--large-piece", type=int,
                         help="SIRCL_LARGE_PIECE_BYTES: op size of the two-shot pieces of all_reduce_large "
                              "(default the larger of 4 MiB and the capacity)")
        sub.add_argument("--gather-schedules",
                         help="SIRCL_GATHER_SCHEDULE values of the link all-gather cases (--chain-gather-sizes), "
                              "e.g. pieces,chain,ring")
        sub.add_argument("--scatter-schedules",
                         help="SIRCL_SCATTER_SCHEDULE values of the reduce-scatter cases (--reduce-scatter-sizes), "
                              "e.g. pieces,chain,ring")
        sub.add_argument("--eager-profile", action="store_true",
                         help="eager rows of all_reduce, all_gather, all_reduce_large and all_gather_large record "
                              "the sessions' call stage times (SIRCL_CALL_PROFILE); the summary prints them")
        sub.add_argument("--eager-path", choices=("session", "adapter"),
                         help="adapter: those collectives go through the vLLM adapter's planner and executor, "
                              "timed apart (default session: the session's methods)")
        sub.add_argument("--chain-gather-sizes",
                         help="link all-gather cases: shards of [rows, 4096] BF16 gathered along dimension 0, bytes "
                              "per rank (multiples of 8192), comma-separated, in place of the configuration's; "
                              "each runs under every --gather-schedules value and gather link piece")
        sub.add_argument("--reduce-scatter-sizes",
                         help="reduce-scatter cases: inputs of [rows, 4096] BF16, bytes per rank (multiples of "
                              "8192), comma-separated, in place of the configuration's; each runs under every "
                              "--scatter-schedules value and scatter link piece")
        sub.add_argument("--host-send-gbps", type=float,
                         help="GB/s a Spark's NIC sends from host memory, for the bounds (default 24)")
        sub.add_argument("--host-recv-gbps", type=float,
                         help="GB/s a Spark's NIC receives into host memory, for the bounds (default 26.8)")
        sub.add_argument("--host-cap-gbps", type=float,
                         help="one GB/s rate for both directions of a Spark's host interface, for the bounds "
                              "(the default of --host-send-gbps and --host-recv-gbps)")
        sub.add_argument("--cable-gbps", type=float,
                         help="GB/s per cable direction over both lanes, for the bounds (default 24)")
        sub.add_argument("--link-chunks",
                         help="link piece sizes of the chain and ring cases, e.g. 262144,524288")
        sub.add_argument("--gather-link-chunks",
                         help="link piece sizes of the chain and ring all-gather cases (default --link-chunks)")
        sub.add_argument("--scatter-link-chunks",
                         help="link piece sizes of the chain and ring reduce-scatter cases (default --link-chunks)")
        sub.add_argument("--reduce-link-chunks",
                         help="link piece sizes of the ring all-reduce cases (default --link-chunks)")
        sub.add_argument("--session-env", action="append", metavar="NAME=VALUE",
                         help="a documented SIRCL_* session variable for every rank (repeatable), for "
                              "example SIRCL_CHAIN_SLOTS=8")
        sub.add_argument("--startup-wait", type=float,
                         help="flag-wait limit in seconds during setup and warm-up (default 300)")
        sub.add_argument("--serving-wait", type=float,
                         help="flag-wait limit in seconds during the timed cases (default 20)")
        sub.add_argument("--dcp-decode-rows",
                         help="dcp4: decode row counts, comma-separated, or none (default 1,2,4,8,16,32,64,128)")
        sub.add_argument("--dcp-prefill-rows",
                         help="dcp4: prefill chunk row counts, comma-separated, or none (default 8192)")
        sub.add_argument("--swing-sizes",
                         help="ring-swing: Swing all-reduce sizes in bytes, comma-separated "
                              "(default 1048576,1048592,1572864,2097152)")
        sub.add_argument("--latency-sizes",
                         help="ring-latency, path4-latency: one-shot and two-shot all-reduce sizes in bytes, "
                              "comma-separated "
                              "(default 4 KiB to 64 KiB, 96 KiB and 128 KiB)")
        sub.add_argument("--post-orders",
                         help="ring-latency, path4-latency: posting orders compared, comma-separated from rank, "
                              "ring-farthest and farthest; the first builds the session (default rank,ring-farthest; "
                              "rank,farthest on path4-latency)")
        sub.add_argument("--relay-us", type=float,
                         help="latency model: added one-way latency per NIC relay in us (default 0.75)")
        sub.add_argument("--post-us", type=float,
                         help="latency model: posting time per lane in us where the session does not measure "
                              "it (default 0.3)")
        sub.add_argument("--write-us", type=float,
                         help="latency model: one-way latency of a direct RDMA write in us (default 0: excluded)")
        if name in ("preflight", "run", "tune"):
            sub.add_argument("--force", action="store_true", help="proceed although other containers run")
        if name in ("plan", "run"):
            sub.add_argument("--tuning-table", action="append", type=Path, metavar="PATH",
                             help="a tuning table (tune, tune-table) the sessions choose from, written beside "
                                  "plan.json on every Spark (repeatable: one per group shape)")
        if name in ("plan", "tune"):
            sub.add_argument("--tune", action="store_true", help=argparse.SUPPRESS)
            sub.add_argument("--quick", action="store_true", help="tune: every fourth size (4 KiB, 16 KiB, ... 64 MiB)")
            sub.add_argument("--tune-collectives", help="tune: collectives, comma-separated (default all four)")
            sub.add_argument("--tune-sizes", help="tune: per-rank sizes in bytes, comma-separated")
            sub.add_argument("--tune-modes", help="tune: graph,eager (default both)")
            sub.add_argument("--tune-grids", help="tune: launch grid caps, comma-separated (default 4,8,16,32)")
            sub.add_argument("--tune-pieces", help="tune: chain chunks and link pieces (default 262144,524288,1048576)")
            sub.add_argument("--tune-staggers", help="tune: ring staggers of the partials (link 2) and of the "
                                                     "forwarded pieces (link 3) (default 0,1)")
            sub.add_argument("--tune-link-blocks",
                             help="tune: blocks per role of the link kernels' candidates (ring schedules, chain "
                                  "all-gather and reduce-scatter), comma-separated; 0: the session's (default 1,2,4)")
            sub.add_argument("--tune-chain-blocks",
                             help="tune: blocks per role of the chain all-reduce's candidates; 0: the session's "
                                  "(default 1,2,4)")
            sub.add_argument("--tune-prune-from", type=int,
                             help="tune: the smallest size in bytes at which a candidate can be dropped (default "
                                  "4194304)")
            sub.add_argument("--tune-prune", type=float, help="tune: drop a candidate this many times slower than "
                                                              "the fastest at two sizes in a row (default 1.5)")
        if name in ("run", "tune"):
            sub.add_argument("--print", action="store_true", help="print the launch plan and contact nothing")
            sub.add_argument("--output", type=Path, default=Path("sircl-ring-results"))
            sub.add_argument("--timeout", type=float,
                             help=f"seconds the run waits for a configuration's containers (default "
                                  f"{plan_mod.RUN_TIMEOUT_S}, or the rank watchdog plus "
                                  f"{plan_mod.RUN_TIMEOUT_MARGIN_S} when that is longer)")
        if name == "plan":
            sub.add_argument("--json", action="store_true")
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--results", required=True, type=Path, help="a configuration's result folder")
    traced = commands.add_parser("trace", help="stage times of the traced chain and link cases of a run "
                                                "(--session-env SIRCL_EVENT_TRACE=<records>)")
    traced.add_argument("--results", required=True, type=Path, help="a configuration's result folder")
    rebuilt = commands.add_parser("tune-table", help="the tuning tables of a tune run's results, rebuilt")
    rebuilt.add_argument("--results", required=True, type=Path, help="a configuration's or a run's result folder")
    args = parser.parse_args(argv)

    if args.command == "tune-table":
        folders = ([args.results] if (args.results / "plan.json").is_file()
                   else sorted(path.parent for path in args.results.glob("*/plan.json")))
        if not folders:
            print(f"error: {args.results} holds no plan.json, directly or in a configuration folder", file=sys.stderr)
            return 2
        written = 0
        for folder in folders:
            written += write_tuning_tables(folder)
        return 0 if written else 1
    if args.command in ("summarize", "trace"):
        # A configuration's folder holds plan.json; a run's folder holds one folder per configuration.
        folders = ([args.results] if (args.results / "plan.json").is_file()
                   else sorted(path.parent for path in args.results.glob("*/plan.json")))
        if not folders:
            print(f"error: {args.results} holds no plan.json, directly or in a configuration folder", file=sys.stderr)
            return 2
        status = 0
        for folder in folders:
            plan = json.loads((folder / "plan.json").read_text(encoding="utf-8"))
            merged = summary.merge(plan, summary.load_results(folder, plan["world"]))
            if len(folders) > 1:
                print(f"== {folder.name}")
            print(summary.table(merged) if args.command == "summarize" else trace_mod.table(merged))
            status = status or (0 if merged["status"] == "passed" else 1)
        return status
    try:
        site = Site.load(args.site, cablings=CABLINGS)
        plans = _plans(args, site)
    except (SiteError, plan_mod.PlanError, routes_mod.RouteError, OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    for plan in plans:
        if args.command == "plan" and args.json:
            print(json.dumps(plan.to_json(), indent=1))
        else:
            print(plan_mod.render_text(plan))
    if args.command == "plan" or (args.command in ("run", "tune") and args.print):
        return 0
    if args.command == "preflight":
        ok, lines = preflight(site, plans, force=args.force)
        print("\n".join(lines))
        print("preflight passed" if ok else "preflight found blockers")
        return 0 if ok else 1
    if args.command == "stage":
        return 0 if stage(site, plans, args.run_id) else 1
    if args.command == "cleanup":
        clean = True
        for position in range(site.size):
            host = site.host(position)
            answer = remote.ssh(host.ssh, remote.remove_harness_containers(docker=host.docker), timeout=60)
            if answer.ok:
                print(f"{host.name}: {answer.stdout.strip()} harness containers left")
                clean &= answer.stdout.strip() == "0"
            else:
                detail = answer.stderr.strip() or answer.stdout.strip() or f"exit code {answer.returncode}"
                print(f"{host.name}: cleanup with docker command \"{host.docker}\" failed: {detail}")
                clean = False
        return 0 if clean else 1
    output = args.output / args.run_id
    results = []
    for plan in plans:
        timeout = args.timeout if args.timeout is not None else run_timeout(plan.options)
        results.append(run_configuration(site, plan, output, force=args.force, timeout=timeout))
        if args.command == "tune":
            write_tuning_tables(output / plan.name)
        if not results[-1]:
            print(f"configuration {plan.name} failed; later configurations are not run")
            break
    print(f"results in {output}")
    return 0 if all(results) and len(results) == len(plans) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
