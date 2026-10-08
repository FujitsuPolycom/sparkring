"""Merge the ranks' results of one configuration into one result and a table.

For every group and case (collective, mode, shape) the merged record holds:
whether every rank's output matched the reference bit for bit; the latency
percentiles of the slowest rank (the per-call maximum over the group's ranks,
then the percentile); the algorithm and bus bandwidth at the slowest median
(all-reduce: message bytes per second, bus factor ``2 (W - 1) / W``;
all-gather: gathered bytes per second, bus factor ``(W - 1) / W``); each
rank's own median and CPU placement; and every error-counter increase of any
Spark of the group. A configuration passes when every rank finished with exit
code 0, every output matched and no error counter moved.

Rows that name a posting order (the ``ring-latency`` cases) also carry the
latency model of :mod:`sparkring_sircl.latency_model` for their algorithm and
every rank's peers in that order, with each rank's measured posting time per
lane where its session counts it and the plan's assumption elsewhere. The
merged result then lists, per group, mode and posting order, the one-shot and
two-shot medians by size with the first size at which the two-shot all-reduce
is faster and the ``SIRCL_ONESHOT_MAX_BYTES`` below it (``crossover``), and
per size and algorithm the medians of every posting order (``post_orders``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from .. import bounds as bounds_mod
from .. import latency_model as latency_mod
from .. import routes as routes_mod
from . import nccl as nccl_mod
from .worker import bandwidths, percentile


def _bound(plan: Mapping, group: Mapping, run: Mapping) -> dict:
    """The case's bound (``sparkring_sircl.bounds``) on the group's lanes, from the run's traffic schedule."""
    traffic = run.get("traffic")
    collective = str(run.get("collective", ""))
    family = ("all_reduce" if collective.startswith("all_reduce") else
              "all_gather" if collective.startswith("all_gather") else collective)
    if not traffic or family not in bounds_mod.COLLECTIVES or run.get("world"):
        return {}
    options = plan.get("options", {})
    # Plans that name one host rate for both directions carry it as host_cap_gbps.
    cap = options.get("host_cap_gbps")
    try:
        layout = routes_mod.Layout.parse(group["layout"])
        nbytes = run["bytes"] * (layout.world if family == "all_gather" else 1)
        seconds = bounds_mod.bound_seconds(layout, int(group.get("lanes", 2)), family, traffic, nbytes,
                                           run.get("chain_order"),
                                           send_gbps=float(options.get("host_send_gbps", cap or bounds_mod.HOST_SEND_GBPS)),
                                           recv_gbps=float(options.get("host_recv_gbps", cap or bounds_mod.HOST_RECV_GBPS)),
                                           cable_gbps=float(options.get("cable_gbps", bounds_mod.CABLE_GBPS)))
    except (bounds_mod.BoundError, routes_mod.RouteError, KeyError, ValueError):
        return {}
    return {"bound_ms": round(seconds * 1e3, 4)}


def _latency(plan: Mapping, group: Mapping, runs: Sequence[Mapping]) -> dict:
    """The latency model of a one-shot or two-shot row that names its posting order: every rank's peers in
    that order (from its record, else from the order's name) and every rank's measured posting time per lane
    (else the plan's ``latency_post_us``)."""
    first = runs[0]
    algorithm = first.get("algorithm")
    order = first.get("post_order")
    if order is None or algorithm not in latency_mod.ALGORITHMS:
        return {}
    try:
        layout = routes_mod.Layout.parse(group["layout"])
        lanes = int(group.get("lanes", 2))
        if len(runs) != layout.world:
            return {}
        peers = [run.get("post_order_peers") for run in runs]
        orders = (tuple(tuple(int(peer) for peer in item) for item in peers) if all(peers)
                  else latency_mod.orders_for(layout, lanes, order))
        parameters = latency_mod.parameters_from(plan.get("options", {}))
        measured = [(run.get("posting") or {}).get("us_per_lane") for run in runs]
        post_us = [parameters.post_us if value is None else float(value) for value in measured]
        estimate = latency_mod.allreduce(layout, lanes, int(first["bytes"]), algorithm, orders, parameters,
                                         post_us)
    except (ValueError, routes_mod.RouteError, KeyError):
        return {}
    return {"latency_model_us": round(estimate.us, 3), "latency_model": estimate.to_json(),
            "posting_measured": all(value is not None for value in measured),
            "posting_us_per_lane": [round(value, 4) for value in post_us]}


def crossovers(cases: Sequence[Mapping]) -> list[dict]:
    """Per group, mode and posting order: the one-shot and two-shot medians by size, the first size at
    which the two-shot all-reduce is faster, the largest size below it (``SIRCL_ONESHOT_MAX_BYTES``; None
    when the two-shot all-reduce is faster from the smallest size), and the larger sizes at which the
    one-shot all-reduce is faster again."""
    table: dict[tuple, dict[int, dict[str, float]]] = {}
    for case in cases:
        name = str(case.get("collective"))
        if name not in ("all_reduce_oneshot", "all_reduce_twoshot") or not case.get("slowest_p50_us"):
            continue
        key = (case["group"], case["mode"], case.get("post_order"))
        caps = [case.get("large_blocks")]
        if name == "all_reduce_oneshot":
            # One-shot rows run once; they join the two-shot rows of every grid cap.
            caps = sorted({other.get("large_blocks") for other in cases
                           if other.get("collective") == "all_reduce_twoshot"}, key=lambda cap: cap or 0) or [None]
        for cap in caps:
            table.setdefault(key + (cap,), {}).setdefault(case["bytes"], {})[name[len("all_reduce_"):]] = \
                case["slowest_p50_us"]
    found = []
    for (group, mode, order, cap), sizes in sorted(table.items(), key=lambda item: (item[0][0], item[0][1],
                                                                                      str(item[0][2]),
                                                                                      item[0][3] or 0)):
        both = sorted(size for size, times in sizes.items() if {"oneshot", "twoshot"} <= set(times))
        if not both:
            continue
        first_twoshot = next((size for size in both if sizes[size]["twoshot"] < sizes[size]["oneshot"]), None)
        below = [size for size in both if first_twoshot is None or size < first_twoshot]
        found.append({
            "group": group, "mode": mode, "post_order": order, "large_blocks": cap, "sizes": both,
            "oneshot_us": [sizes[size]["oneshot"] for size in both],
            "twoshot_us": [sizes[size]["twoshot"] for size in both],
            "first_twoshot_bytes": first_twoshot,
            "oneshot_max_bytes": max(below) if below else None,
            "oneshot_faster_above": [size for size in both if first_twoshot is not None and size > first_twoshot
                                     and sizes[size]["oneshot"] <= sizes[size]["twoshot"]],
        })
    return found


def order_comparison(cases: Sequence[Mapping]) -> list[dict]:
    """Per group, mode, algorithm and size with more than one posting order: every order's median."""
    table: dict[tuple, dict[str, float]] = {}
    for case in cases:
        if case.get("post_order") is None or not case.get("slowest_p50_us"):
            continue
        key = (case["group"], case["mode"], str(case.get("algorithm")), case["bytes"], case.get("large_blocks") or 0)
        table.setdefault(key, {})[case["post_order"]] = case["slowest_p50_us"]
    return [{"group": group, "mode": mode, "algorithm": algorithm, "bytes": nbytes,
             "large_blocks": cap or None, "p50_us": times}
            for (group, mode, algorithm, nbytes, cap), times in sorted(table.items()) if len(times) > 1]


def merge(plan: Mapping, results: Sequence[Mapping | None]) -> dict:
    ranks = []
    problems = []
    for global_rank, rank_plan in enumerate(plan["ranks"]):
        result = results[global_rank] if global_rank < len(results) else None
        if result is None:
            problems.append(f"rank {global_rank} ({rank_plan['host']}) wrote no result")
            ranks.append({"global_rank": global_rank, "host": rank_plan["host"], "status": "missing"})
            continue
        if result.get("error"):
            problems.append(f"rank {global_rank} ({rank_plan['host']}): {result['error']}")
        ranks.append({"global_rank": global_rank, "host": rank_plan["host"], "group": result.get("group"),
                      "exit_code": result.get("exit_code"), "error": result.get("error"),
                      "setup_seconds": result.get("setup_seconds"), "session": result.get("session"),
                      "counters_total": result.get("counters_total", {}),
                      "counter_sources": result.get("counter_sources", {})})
    cases = []
    for group in plan["groups"]:
        members = [results[r] for r in group["global_ranks"] if r < len(results) and results[r] is not None]
        if not members:
            continue
        count = min(len(member.get("runs", [])) for member in members)
        for index in range(count):
            runs = [member["runs"][index] for member in members]
            first = runs[0]
            times = [run.get("times_us", []) for run in runs]
            length = min((len(t) for t in times), default=0)
            slowest = [max(t[i] for t in times) for i in range(length)]
            counters: dict[str, int] = {}
            for member, run in zip(members, runs):
                for key, value in run.get("counters", {}).items():
                    counters[f"{member['host']}:{key}"] = value
            record = {
                "group": group["index"], "collective": first["collective"], "mode": first["mode"],
                "shape": first["shape"], "dim": first["dim"], "bytes": first["bytes"],
                "correct": all(run.get("correct", False) for run in runs),
                "mismatched_calls": sum(run.get("mismatched_calls", 0) for run in runs),
                "checked_calls": sum(run.get("checked", 0) for run in runs),
                "slowest_p50_us": round(percentile(slowest, 0.5), 2) if slowest else None,
                "slowest_p90_us": round(percentile(slowest, 0.9), 2) if slowest else None,
                "slowest_p99_us": round(percentile(slowest, 0.99), 2) if slowest else None,
                "rank_p50_us": [run.get("p50_us") for run in runs],
                "rank_placement": [run.get("placement") for run in runs],
                "algorithm": first.get("algorithm"),
                "counters": counters,
                "world_ranks": len(group["global_ranks"]),
            }
            if first.get("large_blocks") is not None:
                record["large_blocks"] = first["large_blocks"]
            if first.get("tune"):
                record["tune"] = first["tune"]
            profiles = [run.get("call_profile") for run in runs]
            if any(profiles):
                record["call_profile"] = {str(rank): profile for rank, profile in zip(group["global_ranks"], profiles)
                                          if profile}
            adapter = [run.get("adapter_profile") for run in runs]
            if any(adapter):
                record["adapter_profile"] = {str(rank): entry for rank, entry in zip(group["global_ranks"], adapter)
                                             if entry}
            forward = [run.get("forward") for run in runs]
            if all(entry is not None for entry in forward):
                # The rank that waited longest for its forward windows, per call.
                worst = max(forward, key=lambda entry: entry["wait_us_per_call"])
                record["forward_wait_us_per_call"] = worst["wait_us_per_call"]
                record["forward_waits_per_call"] = worst["waits_per_call"]
                record["forward_proven_kib_per_call"] = max(entry["proven_kib_per_call"] for entry in forward)
            traces = {str(rank): run["event_trace"] for rank, run in zip(group["global_ranks"], runs)
                      if run.get("event_trace")}
            if traces:
                record["event_traces"] = traces
            if slowest:
                # A world-session case records its own world; group cases span the group.
                record.update(bandwidths(first["collective"], first["bytes"], first.get("world", len(members)),
                                         record["slowest_p50_us"]))
            record.update(_bound(plan, group, first))
            if record.get("bound_ms") and record["slowest_p50_us"]:
                record["of_bound"] = round(record["bound_ms"] * 1e3 / record["slowest_p50_us"], 4)
            if first.get("post_order") is not None:
                record["post_order"] = first["post_order"]
                record.update(_latency(plan, group, runs))
                if record.get("latency_model_us") and record["slowest_p50_us"]:
                    record["of_latency_model"] = round(record["latency_model_us"] / record["slowest_p50_us"], 4)
            if "target_ms" in first and record["slowest_p50_us"]:
                milliseconds = record["slowest_p50_us"] / 1000.0
                record.update(target_ms=first["target_ms"], stretch_ms=first["stretch_ms"],
                              target_met=milliseconds <= first["target_ms"],
                              stretch_met=milliseconds <= first["stretch_ms"])
            if not record["correct"]:
                problems.append(f"group {group['index']} {record['collective']} {record['mode']} "
                                f"{record['bytes']} bytes: {record['mismatched_calls']} calls differ")
            if counters:
                problems.append(f"group {group['index']} {record['collective']} {record['mode']} "
                                f"{record['bytes']} bytes: error counters moved {counters}")
            cases.append(record)
    _pair_with_nccl(cases)
    warnings = []
    for group_index, micros in sorted(nccl_mod.degraded_pairs(cases).items()):
        warnings.append(f"group {group_index}: NCCL degraded: its 8 KiB eager all-reduce took {micros:.1f} us on "
                        f"the pair (above {nccl_mod.DEGRADED_PAIR_8K_US:g} us); its NCCL rows are no valid baseline")
        for case in cases:
            if case["group"] == group_index and (case["collective"].startswith("nccl_") or "vs_nccl" in case):
                case["nccl_degraded"] = True
    tuning = []
    for group in plan["groups"]:
        planned = group.get("tuning_table") or ""
        members = [results[r] for r in group["global_ranks"] if r < len(results) and results[r] is not None]
        reported = [member for member in members if "tuning" in member]
        info = (reported[0].get("tuning") if reported else None) or {}
        if not planned and not info:
            continue
        tuning.append({"group": group["index"], "planned": planned, "table": info.get("table"),
                       "decisions": info.get("decisions", {}), "unusable": info.get("unusable", {})})
        used = {(member.get("tuning") or {}).get("table") or "" for member in reported}
        if reported and used != {planned}:
            problems.append(f"group {group['index']}: the plan names tuning table {planned or 'none'}, the "
                            f"sessions decided from {', '.join(sorted(value or 'none' for value in used))}")
    warnings += tune_coverage(plan, results, cases)
    exit_codes = [rank.get("exit_code") for rank in ranks]
    passed = not problems and all(code == 0 for code in exit_codes)
    return {"schema": "sircl-ring-configuration-result/v1", "configuration": plan["configuration"],
            "run_id": plan["run_id"], "status": "passed" if passed else "failed", "problems": problems,
            "groups": [{key: group[key] for key in ("index", "positions", "global_ranks", "layout", "max_relays",
                                                    "relay_load", "route_texts", "warnings")}
                       for group in plan["groups"]],
            "ranks": ranks, "cases": cases, "crossover": crossovers(cases), "post_orders": order_comparison(cases),
            "warnings": warnings, "tuning": tuning}


def profile_lines(result: Mapping) -> list[str]:
    """The eager call profiles of a result: per case, the stage medians of its first and slowest-total rank,
    and the adapter's planner and executor times."""
    from .. import callprofile

    lines = []
    for case in result["cases"]:
        profiles = case.get("call_profile") or {}
        adapter = case.get("adapter_profile") or {}
        if not profiles and not adapter:
            continue
        head = f"eager profile, group {case['group']} {case['collective']} {case['bytes']} bytes"
        if case.get("slowest_p50_us") is not None:
            head += f" (event p50 {case['slowest_p50_us']:g} us)"
        lines.append(head + ":")
        totals = {rank: max((row.get("total", 0) for row in rows), default=0) for rank, rows in profiles.items()}
        shown = list(dict.fromkeys([next(iter(profiles), None), max(totals, key=totals.get, default=None)]))
        for rank in shown:
            if rank is None:
                continue
            for line in callprofile.render({"rows": profiles[rank]}):
                lines.append(f"  rank {rank}: {line}")
        for rank, entry in adapter.items():
            lines.append(f"  rank {rank} adapter: plan {entry['plan_us']:g} us, execute {entry['execute_us']:g} us "
                         f"({', '.join(entry['methods'])})")
            break
    return lines


def tune_coverage(plan: Mapping, results: Sequence[Mapping | None], cases: Sequence[Mapping]) -> list[str]:
    """Warnings of a tune run: per group, collective and mode, every family the group's sessions could run
    at the largest size (``tune_families`` of its first rank) that has no exact measurement at the largest
    measured size, and the largest size it was measured at."""
    warnings = []
    for group in plan["groups"]:
        members = [results[r] for r in group["global_ranks"] if r < len(results) and results[r] is not None]
        families = next((member.get("tune_families") for member in members if member.get("tune_families")), None)
        if not families:
            continue
        measured: dict[tuple[str, str], dict[int, set[str]]] = {}
        for case in cases:
            tune = case.get("tune")
            if case["group"] != group["index"] or not tune or not case.get("correct"):
                continue
            choice = tune["choice"]
            family = ("nccl" if choice.get("backend") == "nccl"
                      else choice.get("schedule") or choice.get("algorithm") or "grid")
            measured.setdefault((tune["collective"], case["mode"]), {}).setdefault(case["bytes"], set()).add(family)
        for (collective, mode), by_size in sorted(measured.items()):
            largest = max(by_size)
            missing = [family for family in families.get(collective, ()) if family not in by_size[largest]]
            for family in missing:
                sizes = [size for size, found in by_size.items() if family in found]
                where = f"last measured at {max(sizes)} bytes (pruned)" if sizes else "never measured"
                warnings.append(f"tune coverage, group {group['index']} {collective} {mode}: no {family} "
                                f"measurement at {largest} bytes, {where}")
    return warnings


def tuning_rows(result: Mapping) -> dict[int, list[dict]]:
    """The tune command's measurements of a merged result, by group: every exact candidate's collective,
    mode, bytes, choice and slowest-rank median."""
    rows: dict[int, list[dict]] = {}
    for case in result["cases"]:
        tune = case.get("tune")
        if tune and case.get("correct") and case.get("slowest_p50_us"):
            rows.setdefault(int(case["group"]), []).append({
                "collective": tune["collective"], "mode": case["mode"], "bytes": case["bytes"],
                "choice": tune["choice"], "p50_us": case["slowest_p50_us"]})
    return rows


def tuning_tables(plan: Mapping, result: Mapping, *, created: str = "") -> dict[int, dict]:
    """A tuning table per group of a tune run (``tuning.build_document``), keyed by the group's shape,
    size, lanes and relays, the plan's image and this package's hashes and version."""
    from .. import tuning as tuning_mod

    tables = {}
    for index, rows in sorted(tuning_rows(result).items()):
        group = next(g for g in plan["groups"] if g["index"] == index)
        layout = routes_mod.Layout.parse(group["layout"])
        key = tuning_mod.facts(layout.identity(), layout.world, int(group.get("lanes", 2)), int(group["max_relays"]))
        key["image"] = plan.get("image", "")
        tables[index] = tuning_mod.build_document(key, rows, run_id=str(plan.get("run_id", "")), created=created)
    return tables


def _family(collective: str) -> str | None:
    if collective.startswith("nccl_"):
        return collective[len("nccl_"):]
    if collective.startswith("all_reduce"):
        return "all_reduce"
    if collective.startswith("all_gather"):
        return "all_gather"
    return None


def _pair_with_nccl(cases: list[dict]) -> None:
    """Every SIRCL all-reduce and all-gather row with an NCCL row of its group, family, mode and bytes gets
    ``nccl_p50_us`` and ``vs_nccl`` (NCCL's median over SIRCL's: above 1, SIRCL is faster)."""
    nccl = {(case["group"], _family(case["collective"]), case["mode"], case["bytes"]): case
            for case in cases if case["collective"].startswith("nccl_") and case.get("slowest_p50_us")}
    for case in cases:
        if case["collective"].startswith("nccl_") or not case.get("slowest_p50_us"):
            continue
        if case.get("dim", 0) != 0 and _family(case["collective"]) == "all_gather":
            continue
        match = nccl.get((case["group"], _family(case["collective"]), case["mode"], case["bytes"]))
        if match is not None:
            case["nccl_p50_us"] = match["slowest_p50_us"]
            case["vs_nccl"] = round(match["slowest_p50_us"] / case["slowest_p50_us"], 3)


def _target_note(case: Mapping) -> str:
    algorithm = case.get("algorithm")
    note = f"  [{algorithm}]" if algorithm and case.get("collective") != "all_reduce" else ""
    if case.get("post_order") is not None:
        note = f"  [{algorithm}, posting order {case['post_order']}]"
    if case.get("bound_ms"):
        note += f"  [bound {case['bound_ms']:.3g} ms, {100 * case.get('of_bound', 0):.0f} % of it]"
    if case.get("latency_model_us"):
        critical = case["latency_model"]["phases"][-1]
        note += (f"  [latency model {case['latency_model_us']:.3g} us"
                 f"{'' if case.get('posting_measured') else ' (posting assumed)'}, "
                 f"{100 * case.get('of_latency_model', 0):.0f} % of it; last lane {critical['sender']}->"
                 f"{critical['receiver']} handed at {critical['handed_us']:.3g} us, {critical['relays']} relay(s)]")
    if case.get("nccl_degraded"):
        note += "  [NCCL degraded]"
    if case.get("large_blocks") is not None:
        note += f"  [grid cap {case['large_blocks']}]"
    if case.get("forward_wait_us_per_call"):
        note += (f"  [forward-window waits {case['forward_waits_per_call']:g} per call on the slowest rank, "
                 f"{case['forward_wait_us_per_call']:.3g} us]")
    if "target_ms" not in case:
        return note
    verdict = ("stretch met" if case["stretch_met"] else "target met" if case["target_met"] else "target missed")
    return f"{note}  [target {case['target_ms']:g} ms, stretch {case['stretch_ms']:g} ms: {verdict}]"


def table(result: Mapping) -> str:
    lines = [f"configuration {result['configuration']} (run {result['run_id']}): {result['status'].upper()}"]
    # With NCCL rows, a column of NCCL's median over SIRCL's for every row that has an NCCL counterpart.
    versus = any(case.get("vs_nccl") for case in result["cases"])
    header = f"{'group':>5} {'collective':<10} {'mode':<5} {'bytes':>7} {'shape':<12} {'exact':<5} " \
             f"{'p50 us':>8} {'p90 us':>8} {'p99 us':>8} {'busbw GB/s':>10}" + \
             (f" {'vs NCCL':>8}" if versus else "") + "  counters"
    lines.append(header)
    for case in result["cases"]:
        shape = "x".join(str(v) for v in case["shape"])
        moved = ", ".join(f"{k}+{v}" for k, v in case["counters"].items()) or "-"
        ratio = ""
        if versus:
            ratio = f" {case['vs_nccl']:>7.2f}x" if case.get("vs_nccl") else f" {'-':>8}"
        lines.append(f"{case['group']:>5} {case['collective']:<10} {case['mode']:<5} {case['bytes']:>7} "
                     f"{shape:<12} {'yes' if case['correct'] else 'NO':<5} {case['slowest_p50_us'] or 0:>8.2f} "
                     f"{case['slowest_p90_us'] or 0:>8.2f} {case['slowest_p99_us'] or 0:>8.2f} "
                     f"{case.get('busbw_gbps') or 0:>10.2f}{ratio}  {moved}{_target_note(case)}")
    for found in result.get("crossover", ()):
        order = f", posting order {found['post_order']}" if found.get("post_order") is not None else ""
        if found.get("large_blocks") is not None:
            order += f", grid cap {found['large_blocks']}"
        if found["first_twoshot_bytes"] is None:
            verdict = f"one-shot faster at every size up to {found['sizes'][-1]} bytes"
        elif found["oneshot_max_bytes"] is None:
            verdict = f"two-shot faster from {found['first_twoshot_bytes']} bytes, the smallest size"
        else:
            verdict = (f"one-shot faster up to {found['oneshot_max_bytes']} bytes, two-shot from "
                       f"{found['first_twoshot_bytes']} bytes: SIRCL_ONESHOT_MAX_BYTES={found['oneshot_max_bytes']}")
        if found["oneshot_faster_above"]:
            verdict += f" (one-shot faster again at {found['oneshot_faster_above']})"
        lines.append(f"crossover, group {found['group']} {found['mode']}{order}: {verdict}")
    for compared in result.get("post_orders", ()):
        times = ", ".join(f"{order} {value:.2f}" for order, value in compared["p50_us"].items())
        cap = f", grid cap {compared['large_blocks']}" if compared.get("large_blocks") else ""
        lines.append(f"posting orders, group {compared['group']} {compared['mode']} {compared['algorithm']} "
                     f"{compared['bytes']} bytes{cap}: p50 us {times}")
    for entry in result.get("tuning", ()):
        counted = "; ".join(f"{label} x{count}" for label, count in entry["decisions"].items()) or "none"
        unusable = "; ".join(f"{label} x{count}" for label, count in entry["unusable"].items())
        lines.append(f"tuning, group {entry['group']}: table {entry['table'] or 'none'}; rank 0's decisions: "
                     f"{counted}" + (f"; not runnable here: {unusable}" if unusable else ""))
    lines.extend(profile_lines(result))
    for warning in result.get("warnings", ()):
        lines.append(f"warning: {warning}")
    for problem in result["problems"]:
        lines.append(f"problem: {problem}")
    return "\n".join(lines)


def load_results(directory: Path, world: int) -> list[dict | None]:
    results: list[dict | None] = []
    for rank in range(world):
        path = directory / f"rank-{rank}.json"
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            results.append(None)
    return results
