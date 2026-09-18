"""Coordinate ordered model trials and restore an exact saved deployment.

Nightly runs always restore the saved baseline. Leaving the tested Qwen image on
the serving port requires the separate, explicit supervised-promotion option.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from runtime.common.container_spec import Bind, ContainerSpec  # noqa: E402
from runtime.images.upgrades.contracts import (  # noqa: E402
    Uncertain,
    beneath,
    load_policy,
    read,
    require,
)
from runtime.images.upgrades.hardware import Pair, wait_ready  # noqa: E402
from runtime.images.upgrades.io import write_json  # noqa: E402
from runtime.images.upgrades import tp2_suite  # noqa: E402


def from_document(value):
    item = dict(value)
    for name in (
        "entrypoint",
        "command",
        "devices",
        "health_command",
        "cap_add",
        "security_opt",
    ):
        if name in item:
            item[name] = tuple(item[name])
    item["mounts"] = tuple(Bind(**mount) for mount in item["mounts"])
    return ContainerSpec(**item)


def trial_order(config):
    require(
        config.get("schema") == "sparkring-tp2-experiment/v1",
        "Unknown experiment configuration",
    )
    require(
        [item["model"] for item in config["sites"]] == ["glm", "qwen"],
        "Run GLM first and Qwen second",
    )
    require(
        len(config["rollback_snapshots"]) == 2,
        "Two exact rollback snapshots are required",
    )
    require(
        type(config.get("public_port")) is int
        and 1024 <= config["public_port"] <= 65535,
        "Invalid serving port",
    )
    return config["sites"]


def run(
    config_path,
    policy_path,
    lease_path,
    image_id,
    output,
    *,
    run_id,
    input_sha256,
    gate_id,
    leave_qualified=False,
):
    config_path = Path(config_path).resolve()
    config = read(config_path)
    order = trial_order(config)
    root = config_path.parent
    policy = load_policy(policy_path)
    rollback = [read(beneath(root, path)) for path in config["rollback_snapshots"]]
    pair = Pair(
        policy,
        lease_path,
        config["hosts"],
        run_id=run_id,
        gate_id=gate_id,
        hostnames=config["hostnames"],
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    initial = [pair.inspect(rank, item["Id"]) for rank, item in enumerate(rollback)]
    for original, actual in zip(rollback, initial):
        require(
            original["Image"] == actual["Image"] and original["Name"] == actual["Name"],
            "Saved rollback is no longer the selected deployment",
        )
    running = [item["State"]["Running"] for item in initial]
    require(
        all(running) or not any(running),
        "Partially running baseline needs operator inspection",
    )
    results = []
    owned = []
    promoted = False
    failure = None
    try:
        if any(running):
            pair.assert_idle(config["rollback_api"])
            pair.stop_saved(rollback)
        for item in order:
            site_path = beneath(root, item["path"])
            model_run = run_id + "-" + item["model"]
            require(len(model_run) <= 48, "Experiment identifier is too long")
            owned.append(model_run)
            result = tp2_suite.run(
                site_path,
                policy_path,
                lease_path,
                image_id,
                output / item["model"],
                run_id=model_run,
                input_sha256=input_sha256,
                gate_id=gate_id,
                leave_running=False,
            )
            results.append(result)
            require(
                result["outcome"] == "passed",
                "Model qualification failed: " + item["model"],
            )
        if leave_qualified:
            qwen_plan = read(output / "qwen/plan.json")
            serving_pair = Pair(
                policy,
                lease_path,
                config["hosts"],
                run_id=owned[-1],
                gate_id=gate_id,
                hostnames=config["hostnames"],
            )
            specs = []
            for rank, document in enumerate(qwen_plan["specs"]):
                spec = from_document(document)
                require(
                    spec.image_id == image_id,
                    "Tested image identity differs from promotion target",
                )
                command = tp2_suite.option(
                    spec.command, "--port", config["public_port"]
                )
                spec = replace(
                    spec,
                    name=spec.name + "-serve",
                    command=tuple(command),
                    labels={**spec.labels, "sparkring.upgrade.promotion": "supervised"},
                )
                serving_pair.create(rank, spec)
                specs.append(spec)
            for rank in (1, 0):
                print(serving_pair.start(rank, specs[rank].name), flush=True)
            base = f"http://{config['api_host']}:{config['public_port']}"
            wait_ready(
                base,
                config["rollback_model"],
                seconds=config.get("startup_seconds", 1200),
            )
            saved = [
                serving_pair.owned(rank, spec.name) for rank, spec in enumerate(specs)
            ]
            write_json(
                output / "serving-state.json",
                {
                    "image_id": image_id,
                    "api": base,
                    "model": config["rollback_model"],
                    "containers": saved,
                    "scope": "Explicit supervised promotion after ordered model trials; no automatic nightly promotion.",
                },
            )
            promoted = True
    except BaseException as error:
        failure = str(error)
        raise
    finally:
        cleanup = []
        if not promoted:
            for model_run in owned:
                temporary = Pair(
                    policy,
                    lease_path,
                    config["hosts"],
                    run_id=model_run,
                    gate_id=gate_id,
                    hostnames=config["hostnames"],
                )
                for rank in (0, 1):
                    try:
                        ids = (
                            temporary.call(
                                rank,
                                [
                                    "docker",
                                    "ps",
                                    "-q",
                                    "--filter",
                                    "label=sparkring.upgrade.run=" + model_run,
                                ],
                            )
                            .decode()
                            .split()
                        )
                        for identifier in ids:
                            temporary.stop(rank, identifier)
                    except Exception as error:
                        cleanup.append(str(error))
            if not cleanup:
                try:
                    pair.start_saved(rollback)
                    wait_ready(
                        config["rollback_api"],
                        config["rollback_model"],
                        seconds=config.get("startup_seconds", 1200),
                    )
                except Exception as error:
                    cleanup.append(str(error))
        summary = {
            "schema": "sparkring-tp2-experiment-result/v1",
            "image_id": image_id,
            "input_sha256": input_sha256,
            "run_id": run_id,
            "models": results,
            "promoted": promoted,
            "baseline_restored": not promoted and not cleanup,
            "failure": failure,
            "cleanup_errors": cleanup,
            "passed": len(results) == 2 and not failure and not cleanup,
        }
        write_json(output / "experiment.json", summary)
        if cleanup:
            raise Uncertain(
                "Experiment cleanup/rollback requires inspection: " + "; ".join(cleanup)
            )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "policy", "lease", "output"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("image-id", "run-id", "input-sha256", "gate-id"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--leave-qualified",
        action="store_true",
        help="Supervised only: leave the tested Qwen image on the declared serving port",
    )
    args = parser.parse_args()
    require(args.execute, "Explicit experiment execution is required")
    require(
        re.fullmatch(r"[a-z0-9][a-z0-9-]{1,35}", args.run_id),
        "Invalid experiment run identifier",
    )
    print(
        json.dumps(
            run(
                args.config,
                args.policy,
                args.lease,
                args.image_id,
                args.output,
                run_id=args.run_id,
                input_sha256=args.input_sha256,
                gate_id=args.gate_id,
                leave_qualified=args.leave_qualified,
            )
        ),
        flush=True,
    )
