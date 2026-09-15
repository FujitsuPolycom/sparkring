"""Quiesce a leased serving pair before source/image work and restore exact IDs.

This entry point builds candidates only. Transfer and hardware qualification use
their separate adapters after the saved deployment has been restored. Uncertain
external work prohibits an automatic restart that could overlap a live compiler.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.images.upgrades import runner  # noqa: E402
from runtime.images.upgrades.agent import FileAgent  # noqa: E402
from runtime.images.upgrades.build_native import validate_recipe  # noqa: E402
from runtime.images.upgrades.contracts import (  # noqa: E402
    Uncertain,
    beneath,
    load_policy,
    read,
    require,
    sha,
)
from runtime.images.upgrades.execution import builder_lease_valid  # noqa: E402
from runtime.images.upgrades.hardware import Pair, wait_ready  # noqa: E402
from runtime.images.upgrades.io import write_json  # noqa: E402


def bound_document(policy, path):
    file = beneath(policy["_root"], path)
    require(
        policy["_inputs"].get(path) == sha(file.read_bytes()),
        "Maintenance input is not bound to the approved policy: " + path,
    )
    return read(file)


def configuration(policy, path):
    config = bound_document(policy, path)
    require(
        config.get("schema") == "sparkring-build-maintenance/v1",
        "Unknown build-maintenance configuration",
    )
    require(
        set(config)
        == {
            "schema",
            "hosts",
            "hostnames",
            "gate_id",
            "rollback_snapshots",
            "infrastructure_snapshots",
            "rollback_api",
            "rollback_model",
            "startup_seconds",
        },
        "Build-maintenance fields differ",
    )
    require(
        len(config["rollback_snapshots"]) == 2
        and len(config["infrastructure_snapshots"]) == 2,
        "Build maintenance requires two explicit container inventories",
    )
    require(
        type(config["startup_seconds"]) is int
        and 60 <= config["startup_seconds"] <= 3600,
        "Invalid rollback startup budget",
    )
    saved = [bound_document(policy, name) for name in config["rollback_snapshots"]]
    infrastructure = [
        [bound_document(policy, name) for name in names]
        for names in config["infrastructure_snapshots"]
    ]
    return config, saved, infrastructure


def inventory(pair, saved, infrastructure):
    """Require exact known container identities, without touching infrastructure."""
    for rank in (0, 1):
        approved = [saved[rank], *infrastructure[rank]]
        for snapshot in approved:
            actual = pair.inspect(rank, snapshot["Id"])
            require(
                all(actual[key] == snapshot[key] for key in ("Id", "Image", "Name")),
                "Maintenance container identity differs",
            )
        running = pair.call(rank, ["docker", "ps", "--no-trunc", "--quiet"])
        require(
            set(running.decode().split()) <= {item["Id"] for item in approved},
            "Unregistered containers are running; build maintenance deferred",
        )


def run(
    policy_path,
    state,
    config_path,
    builder_lease,
    hardware_lease,
    output,
    *,
    approved_policy,
    run_id,
    proposal_dir=None,
):
    policy = load_policy(policy_path)
    require(policy["_digest"] == approved_policy, "Approved build policy differs")
    if "native" in policy:
        validate_recipe(policy["native"])
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{1,35}", run_id),
        "Invalid build-maintenance run identifier",
    )
    require(
        not any(gate["stage"] == "hardware" for gate in policy["gates"]),
        "Build maintenance requires separate hardware qualification",
    )
    builder_lease_valid(builder_lease, policy)
    state_file = Path(state) / "state.json"
    require(
        not state_file.exists() or not read(state_file).get("uncertain_run"),
        "Resolve uncertain builder work before stopping the saved deployment",
    )
    config, saved, infrastructure = configuration(policy, config_path)
    pair = Pair(
        policy,
        hardware_lease,
        config["hosts"],
        run_id=run_id,
        gate_id=config["gate_id"],
        hostnames=config["hostnames"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    journal = {
        "schema": "sparkring-build-maintenance-result/v1",
        "run_id": run_id,
        "policy_sha256": policy["_digest"],
        "saved_container_ids": [item["Id"] for item in saved],
        "phase": "preflight",
        "baseline_restored": False,
        "build": None,
        "failure": None,
        "cleanup_errors": [],
    }

    def record(phase):
        journal["phase"] = phase
        write_json(output / "maintenance.json", journal, replace=True)

    stop_attempted = work_started = False
    report = None
    try:
        record("preflight")
        inventory(pair, saved, infrastructure)
        require(
            all(
                pair.inspect(rank, item["Id"])["State"]["Running"]
                for rank, item in enumerate(saved)
            ),
            "Saved deployment must be running before build maintenance",
        )
        pair.assert_idle(config["rollback_api"])
        record("stopping-serving")
        stop_attempted = True
        pair.stop_saved(saved)
        require(
            not any(
                pair.inspect(rank, item["Id"])["State"]["Running"]
                for rank, item in enumerate(saved)
            ),
            "Saved workers did not stop; compilation is prohibited",
        )
        inventory(pair, saved, infrastructure)
        record("building")
        work_started = True
        report = runner.run(
            policy_path,
            state,
            execute=True,
            build=True,
            publish=False,
            builder_lease=builder_lease,
            agent=FileAgent(proposal_dir) if proposal_dir is not None else None,
        )
        journal["build"] = report
    except BaseException as error:
        journal["failure"] = str(error)
        raise
    finally:
        if stop_attempted:
            try:
                require(
                    not work_started
                    or (
                        report is not None
                        and report.get("status")
                        in ("candidate", "unchanged", "blocked")
                        and report.get("simulation") is not True
                    ),
                    "Build outcome is uncertain; inspect owned work before rollback",
                )
                inventory(pair, saved, infrastructure)
                record("restoring-serving")
                pair.start_saved(saved)
                for rank in (1, 0):
                    print(
                        "ssh -t "
                        + pair.hosts[rank]
                        + " "
                        + json.dumps("docker logs -f --tail 80 " + saved[rank]["Id"]),
                        flush=True,
                    )
                wait_ready(
                    config["rollback_api"],
                    config["rollback_model"],
                    seconds=config["startup_seconds"],
                )
                require(
                    all(
                        pair.inspect(rank, item["Id"])["State"]["Running"]
                        for rank, item in enumerate(saved)
                    ),
                    "Saved workers are not both running after rollback",
                )
                journal["baseline_restored"] = True
            except BaseException as error:
                journal["cleanup_errors"].append(str(error))
        record(
            "attention-required"
            if journal["cleanup_errors"] or journal["failure"]
            else "finished"
        )
        if journal["cleanup_errors"]:
            raise Uncertain(
                "Build maintenance needs inspection: "
                + "; ".join(journal["cleanup_errors"])
            )
    return journal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "state", "builder-lease", "hardware-lease", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("config", "approved-policy", "run-id"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--proposal-dir", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "Build maintenance requires explicit execution")
    result = run(
        args.policy,
        args.state,
        args.config,
        args.builder_lease,
        args.hardware_lease,
        args.output,
        approved_policy=args.approved_policy,
        run_id=args.run_id,
        proposal_dir=args.proposal_dir,
    )
    print(json.dumps(result), flush=True)
    return 0 if result["build"]["status"] in ("candidate", "unchanged") else 2


if __name__ == "__main__":
    raise SystemExit(main())
