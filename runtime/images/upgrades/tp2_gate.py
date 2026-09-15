"""Registered hardware gate for ordered model checks and exact baseline restoration.

This adapter never promotes a candidate. Per-model throughput comparisons retain
their distinct control-image identities; a composite result is not a fabricated
single-image performance control.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime.images.upgrades.contracts import beneath, read, require  # noqa: E402
from runtime.images.upgrades.io import write_json  # noqa: E402
from runtime.images.upgrades import tp2_experiment  # noqa: E402
from runtime.images.upgrades.tp2_suite import load_site  # noqa: E402


def qualification_inputs(config_path):
    path = Path(config_path).resolve()
    config = read(path)
    for item in tp2_experiment.trial_order(config):
        site, _ = load_site(beneath(path.parent, item["path"]))
        require(
            site.get("cache_checks") is True,
            "Nightly model gate requires persistent-cache checks",
        )
        require(
            site.get("media_checks") is True,
            "Nightly model gate requires its declared media workload",
        )
        require(
            site.get("performance"),
            "Nightly model gate requires a matched performance control",
        )
    return config


def receipt(report, *, gate_id, input_sha256, image_id):
    models = report.get("models", [])
    valid = (
        report.get("passed") is True
        and report.get("promoted") is False
        and report.get("baseline_restored") is True
        and not report.get("failure")
        and not report.get("cleanup_errors")
        and len(models) == 2
        and all(
            model.get("outcome") == "passed"
            and model.get("subject_sha256") == image_id
            and model.get("input_sha256") == input_sha256
            and model.get("skipped") == 0
            and type(model.get("assertions")) is int
            and model["assertions"] > 0
            and model.get("evidence", {}).get("cache")
            and model.get("evidence", {}).get("media")
            and model.get("evidence", {}).get("performance", {}).get("passed") is True
            for model in models
        )
    )
    return {
        "schema": "sparkring-upgrade-gate/v1",
        "gate": gate_id,
        "input_sha256": input_sha256,
        "subject_sha256": image_id,
        "variant": "image",
        "outcome": "passed" if valid else "failed",
        "assertions": sum(model.get("assertions", 0) for model in models) + 1,
        "skipped": 0,
        "evidence": report,
        "scope": "Ordered GLM and Qwen TP2 cache, selected media, matched throughput, and saved-deployment restoration; no candidate promotion.",
    }


def run(args):
    qualification_inputs(args.config)
    output = Path(args.result).resolve()
    report = None
    try:
        report = tp2_experiment.run(
            args.config,
            args.policy,
            args.lease,
            args.image_id,
            output.parent / "model-trials",
            run_id=args.run_id,
            input_sha256=args.input_sha256,
            gate_id=args.gate_id,
            leave_qualified=False,
        )
    finally:
        saved = output.parent / "model-trials/experiment.json"
        if report is None and saved.is_file():
            report = read(saved)
        if report is not None:
            write_json(
                output,
                receipt(
                    report,
                    gate_id=args.gate_id,
                    input_sha256=args.input_sha256,
                    image_id=args.image_id,
                ),
            )
    return read(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("config", "policy", "lease", "result"):
        parser.add_argument("--" + field, type=Path, required=True)
    for field in ("image-id", "run-id", "input-sha256", "gate-id"):
        parser.add_argument("--" + field, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    require(args.execute, "Hardware gate requires explicit execution authorization")
    result = run(args)
    print(json.dumps(result))
    return 0 if result["outcome"] == "passed" else 2


if __name__ == "__main__":
    sys.exit(main())
