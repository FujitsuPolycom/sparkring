"""Run a bounded upgrade experiment without changing the controller checkout."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import time
import uuid

from . import agent as agents, sources
from .contracts import (
    Refused,
    Uncertain,
    beneath,
    check_policy,
    controller_inputs,
    encoded,
    load_policy,
    require,
    sha,
)
from .execution import Executor, acceptable
from .io import lock, storage_check, write_json


def controller_digest():
    return sha(encoded(controller_inputs()))


def report_markdown(report):
    lines = [
        "# SparkRing image upgrade run",
        "",
        f"Result: **{report['status']}**.",
        "",
        f"Run: `{report['run_id']}`. Policy: `{report['policy_sha256']}`.",
        "",
        "This is candidate evidence, not permission to replace a serving deployment.",
        "",
    ]
    if report.get("reason"):
        lines += ["Required attention: " + report["reason"], ""]
    lines += ["| Component | Target commit | Reconciliation |", "|---|---|---|"]
    for item in report.get("sources", []):
        result = report.get("reconciliation", {}).get(item["id"], {})
        lines.append(
            f"| {item['id']} | `{item['target']}` | {result.get('disposition', 'not evaluated')} |"
        )
    lines += [
        "",
        "No stable-image promotion or production deployment is implemented by this runner.",
        "",
    ]
    return "\n".join(lines)


def _save_report(directory, report):
    write_json(directory / "report.json", report)
    (directory / "REPORT.md").write_text(report_markdown(report), encoding="utf-8")
    (directory / "PR.md").write_text(report_markdown(report), encoding="utf-8")


def run(
    policy_path,
    state_root,
    *,
    execute=False,
    build=False,
    publish=False,
    hardware_lease=None,
    builder_lease=None,
    agent=None,
    executor=None,
    force=False,
):
    policy = load_policy(policy_path)
    require(not (build or publish) or execute, "Build/publication require execution")
    require(
        not publish or build, "Publication requires a verified build in the same run"
    )
    executor = executor or Executor(
        policy,
        execute=execute,
        build=build,
        hardware_lease=hardware_lease,
        builder_lease=builder_lease,
        publish=publish,
    )
    if agent is None and policy.get("agent") and execute:
        agent = agents.ChatAgent(policy["agent"])
    started = time.monotonic()
    deadline = started + policy["budgets"]["run_seconds"]
    with lock(state_root) as root:
        storage_check(root, policy["budgets"])
        state_path = root / "state.json"
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        require(
            not state.get("uncertain_run"),
            "An interrupted or uncertain execution needs operator resolution: "
            + str(state.get("uncertain_run")),
        )
        run_id = (
            datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            + "-"
            + uuid.uuid4().hex[:8]
        )
        directory = root / "runs" / run_id
        directory.mkdir(parents=True)
        report = {
            "schema": "sparkring-upgrade-run/v1",
            "run_id": run_id,
            "status": "running",
            "policy_sha256": policy["_digest"],
            "controller_sha256": controller_digest(),
            "execution_requested": execute,
            "build_requested": build,
            "publication_requested": publish,
            "sources": [],
            "reconciliation": {},
            "qualified_gates": [],
            "hardware_qualified": False,
        }
        report["simulation"] = bool(getattr(executor, "simulation", False))
        require(
            not (publish and report["simulation"]),
            "Simulated results cannot be published",
        )
        write_json(
            directory / "policy.json",
            {k: v for k, v in policy.items() if not k.startswith("_")},
        )
        write_json(directory / "policy-inputs.json", policy["_inputs"])
        # An unexpected controller exit must not let a second run adopt its external work.
        state["uncertain_run"] = run_id if execute else None
        write_json(state_path, state, replace=True)
        try:
            for source in policy["sources"]:
                require(time.monotonic() < deadline, "Discovery exhausted run budget")
                record = sources.discover(source, root / "mirrors", policy["budgets"])
                report["sources"].append(record)
                storage_check(root, policy["budgets"])
            fingerprint = sha(
                encoded(
                    {
                        "policy": policy["_digest"],
                        "controller": report["controller_sha256"],
                        "targets": {s["id"]: s["target"] for s in report["sources"]},
                    }
                )
            )
            report["input_sha256"] = fingerprint
            level = (
                "publish"
                if publish
                else "build"
                if build
                else "execute"
                if execute
                else "plan"
            )
            if (
                not force
                and state.get("completed_inputs", {}).get(level) == fingerprint
            ):
                report["status"] = "unchanged"
            else:
                write_json(directory / "discovery.json", report["sources"])
                context = {
                    "input_sha": fingerprint,
                    "run_id": run_id,
                    "run_root": str(directory),
                    "policy_root": policy["_root"],
                    "policy": policy["_path"],
                    "controller_root": str(Path(__file__).resolve().parents[3]),
                    "bundle": str(directory / "bundle.json"),
                }
                for source, record in zip(policy["sources"], report["sources"]):
                    check_policy(policy)
                    require(
                        time.monotonic() < deadline,
                        "Source reconciliation exhausted run budget",
                    )
                    _reconcile(
                        policy,
                        source,
                        record,
                        directory,
                        report,
                        context,
                        deadline,
                        executor,
                        agent,
                        execute,
                    )
                    storage_check(root, policy["budgets"])
                if not execute:
                    report["status"] = "planned"
                else:
                    for name, item in report["reconciliation"].items():
                        target = directory / "accepted" / name
                        sources.copy_snapshot(item["candidate_path"], target)
                        require(
                            sources.tree_digest(target)
                            == item["candidate_tree_sha256"],
                            "Accepted source copy differs",
                        )
                        item["candidate_path"] = str(target)
                    bundle = {
                        "schema": "sparkring-upgrade-bundle/v1",
                        "input_sha256": fingerprint,
                        "policy_sha256": policy["_digest"],
                        "platform": policy.get("platform"),
                        "required_features": policy.get("required_features", []),
                        "sources": report["reconciliation"],
                    }
                    write_json(directory / "bundle.json", bundle)
                    native = [
                        name
                        for name, item in bundle["sources"].items()
                        if item["native_changed"]
                    ]
                    if native and build:
                        require(
                            policy.get("build", {}).get("supports_native_rebuild")
                            is True,
                            "Native/build inputs changed; a reviewed native rebuild recipe is required: "
                            + ", ".join(native),
                        )
                    if not build:
                        report["status"] = "reconciled"
                    else:
                        result = executor.action(
                            "build", context, directory / "build", deadline
                        )
                        require(
                            result.get("source_trees")
                            == {
                                n: s["candidate_tree_sha256"]
                                for n, s in report["reconciliation"].items()
                            },
                            "Built image receipt does not identify every accepted source tree",
                        )
                        report["build"] = result
                        context["image_id"] = result["image_id"]
                        gates = [
                            g
                            for g in policy["gates"]
                            if g["stage"] in ("image", "hardware")
                        ]
                        require(
                            any(g["stage"] == "image" for g in gates),
                            "Candidate image requires an independent image gate",
                        )
                        for gate in gates:
                            value = executor.gate(
                                gate,
                                directory / "accepted",
                                directory / "gates" / gate["id"],
                                {
                                    **context,
                                    "variant": "image",
                                    "subject_sha256": context["image_id"],
                                },
                                deadline,
                            )
                            require(
                                acceptable(value, gate),
                                "Image/hardware acceptance failed: " + gate["id"],
                            )
                            report["qualified_gates"].append(gate["id"])
                        report["hardware_qualified"] = any(
                            g["stage"] == "hardware" for g in gates
                        )
                        report["status"] = "candidate"
                        if publish:
                            report["publication"] = executor.action(
                                "publish", context, directory / "publication", deadline
                            )
                            report["status"] = "published-candidate"
                        if report["simulation"]:
                            report["status"] = "candidate-simulation"
            if report["status"] in (
                "planned",
                "reconciled",
                "candidate",
                "published-candidate",
                "candidate-simulation",
            ):
                state.setdefault("completed_inputs", {})[level] = fingerprint
            state["uncertain_run"] = None
        except (Refused, OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            report["reason"] = str(error)[:6000]
            uncertain = isinstance(error, Uncertain)
            report["status"] = "uncertain" if uncertain else "blocked"
            state["uncertain_run"] = run_id if uncertain else None
        except BaseException:
            report["status"] = "uncertain" if execute else "interrupted"
            report["reason"] = (
                "Controller interrupted; external work must be inspected before another execution"
            )
            report["elapsed_seconds"] = time.monotonic() - started
            _save_report(directory, report)
            raise
        report["elapsed_seconds"] = time.monotonic() - started
        report["artifact_directory"] = str(directory)
        _save_report(directory, report)
        state.update(last_run=run_id, last_status=report["status"])
        write_json(state_path, state, replace=True)
        return report


def _reconcile(
    policy,
    source,
    record,
    directory,
    report,
    context,
    deadline,
    executor,
    agent,
    execute,
):
    name = source["id"]
    folder = directory / "sources" / name
    excluded = source.get("excluded_paths", [])
    baseline = sources.materialize(
        record,
        record["baseline"],
        folder / "baseline",
        policy["budgets"]["source_bytes"],
        excluded,
    )
    upstream = sources.materialize(
        record,
        record["target"],
        folder / "upstream",
        policy["budgets"]["source_bytes"],
        excluded,
    )
    patch = (
        beneath(policy["_root"], source["patch"]).read_bytes()
        if source.get("patch")
        else b""
    )
    if patch:
        applied, feedback = sources.apply_patch(baseline, patch)
        require(
            applied,
            "Approved baseline patch cannot reproduce its baseline: " + str(feedback),
        )
    upstream_tree = sources.git(upstream, "write-tree").decode().strip()
    candidate = sources.copy_snapshot(upstream, directory / "candidates" / name)
    applied, feedback = (
        sources.apply_fragments(candidate, patch) if patch else (True, [])
    )
    row = {
        "repository": source["repository"],
        "baseline_commit": record["baseline"],
        "target_commit": record["target"],
        "excluded_paths": excluded,
        "baseline_path": str(baseline),
        "upstream_path": str(upstream),
        "candidate_path": str(candidate),
        "disposition": "retain"
        if applied and patch
        else "upstream"
        if applied
        else "unresolved",
        "reason": "Mechanical application only; behavior gates remain mandatory.",
        "native_changed": record["native_changed"],
    }
    report["reconciliation"][name] = row
    if not execute:
        write_json(
            folder / "reconciliation-request.json",
            agents.request_for(
                source,
                record,
                upstream,
                baseline,
                patch,
                feedback,
                policy.get("agent", {}).get("context_bytes", 1000000),
                candidate=candidate,
                policy_sha256=policy["_digest"],
            ),
        )
        return
    ids = list(dict.fromkeys(c["oracle"] for c in source["contracts"]))
    context = {**context, "baseline_path": str(baseline)}
    gates = [next(g for g in policy["gates"] if g["id"] == key) for key in ids]
    baseline_results = {}
    upstream_results = {}
    for gate in gates:
        for variant, path, results in (
            ("baseline", baseline, baseline_results),
            ("upstream", upstream, upstream_results),
        ):
            before = sources.tree_digest(path)
            value = executor.gate(
                gate,
                path,
                folder / "gates" / (variant + "-" + gate["id"]),
                {
                    **context,
                    "component": name,
                    "variant": variant,
                    "subject_sha256": before,
                },
                deadline,
            )
            results[gate["id"]] = value
            require(sources.tree_digest(path) == before, "Oracle changed source bytes")
        require(
            acceptable(baseline_results[gate["id"]], gate),
            "Baseline oracle is not passing: " + gate["id"],
        )
    for attempt in range(policy["budgets"]["agent_attempts"] + 1):
        if applied:
            passed = True
            evidence = []
            for gate in gates:
                before = sources.tree_digest(candidate)
                value = executor.gate(
                    gate,
                    candidate,
                    folder / "gates" / f"candidate-{attempt}-{gate['id']}",
                    {
                        **context,
                        "component": name,
                        "variant": "candidate",
                        "subject_sha256": before,
                    },
                    deadline,
                )
                evidence.append(value)
                require(
                    sources.tree_digest(candidate) == before,
                    "Oracle changed candidate source bytes",
                )
                passed &= acceptable(value, gate, baseline_results[gate["id"]])
            if passed:
                row["oracles"] = evidence
                row["candidate_tree_sha256"] = sources.tree_digest(candidate)
                before, after = (
                    sources.inventory(baseline),
                    sources.inventory(candidate),
                )
                row["native_changed"] = [
                    p
                    for p in sorted(set(before) | set(after))
                    if before.get(p) != after.get(p)
                    and any(
                        p == n or p.startswith(n + "/") for n in source["native_paths"]
                    )
                ]
                patch = sources.complete_patch(candidate, upstream_tree)
                row["patch_sha256"] = sha(patch)
                (folder / "accepted.patch").write_bytes(patch)
                row["patch_path"] = str(folder / "accepted.patch")
                return
            feedback = {"candidate_gates": evidence, "baseline_gates": baseline_results}
        require(
            attempt < policy["budgets"]["agent_attempts"],
            "Reconciliation retry budget exhausted: " + name,
        )
        require(
            agent is not None,
            "Semantic reconciliation requires an agent endpoint or --proposal-dir",
        )
        require(time.monotonic() < deadline, "Agent time budget exhausted")
        request = agents.request_for(
            source,
            record,
            upstream,
            baseline,
            patch,
            feedback,
            policy.get("agent", {}).get("context_bytes", 1000000),
            candidate=candidate,
            policy_sha256=policy["_digest"],
        )
        write_json(folder / f"agent-request-{attempt}.json", request)
        proposal = agents.validate(
            agent.propose(request, timeout=max(1, deadline - time.monotonic()))
        )
        write_json(folder / f"agent-response-{attempt}.json", proposal)
        row.update(disposition=proposal["disposition"], reason=proposal["reason"])
        require(
            proposal["disposition"] not in ("unresolved", "incompatible"),
            "Agent could not establish compatibility: " + proposal["reason"],
        )
        if proposal["disposition"] == "retire":
            require(
                all(
                    c["kind"] != "optimization"
                    or next(g for g in gates if g["id"] == c["oracle"]).get("metrics")
                    for c in source["contracts"]
                ),
                "Optimization retirement requires a protected performance oracle",
            )
            require(
                all(
                    acceptable(upstream_results[g["id"]], g, baseline_results[g["id"]])
                    for g in gates
                ),
                "Patch retirement refused: upstream has not passed every protected oracle",
            )
            candidate = sources.copy_snapshot(upstream, folder / f"retired-{attempt}")
            patch, applied = b"", True
        else:
            candidate = sources.copy_snapshot(candidate, folder / f"adapted-{attempt}")
            applied, feedback = sources.apply_patch(
                candidate,
                proposal["patch"].encode(),
                source["editable_paths"],
                [*source["native_paths"], *source.get("protected_paths", [])],
            )
        row["candidate_path"] = str(candidate)


def resolve_uncertain(state_root, run_id):
    """Operator acknowledgement after independently confirming no owned work remains."""
    with lock(state_root) as root:
        path = root / "state.json"
        state = json.loads(path.read_text())
        require(state.get("uncertain_run") == run_id, "Uncertain run identity differs")
        state["uncertain_run"] = None
        state["operator_resolution"] = {
            "run_id": run_id,
            "external_work_confirmed_stopped": True,
        }
        write_json(path, state, replace=True)
