"""Run baseline and candidate B12X source scopes with immutable input receipts."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "baseline", "peer", "baseline-peer", "controller", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.controller))
    from runtime.images.upgrades.sources import tree_digest

    upgrades = args.controller / "runtime/images/upgrades"
    suite = upgrades / "checks/b12x-sm121-suite-v1.json"
    inputs = {
        "baseline_tree": tree_digest(args.baseline),
        "candidate_tree": tree_digest(args.source),
        "suite_sha256": hashlib.sha256(suite.read_bytes()).hexdigest(),
        "coordinators": {
            name: hashlib.sha256(
                (root / "vllm/v1/worker/b12x_startup.py").read_bytes()
            ).hexdigest()
            for name, root in (
                ("baseline", args.baseline_peer),
                ("candidate", args.peer),
            )
        },
        "scope": "CPU source contracts; no GPU, native compilation or installed-image qualification",
    }
    identity = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "inputs.json").write_text(
        json.dumps({**inputs, "input_sha256": identity}, indent=2) + "\n"
    )
    outcomes = []
    for variant, source, peer in (
        ("baseline", args.baseline, args.baseline_peer),
        ("candidate", args.source, args.peer),
    ):
        env = dict(
            os.environ,
            SPARKRING_UPGRADE_VARIANT=variant,
            SPARKRING_UPGRADE_GATE="b12x-sm121-checkpoint",
            SPARKRING_UPGRADE_SUBJECT=inputs[variant + "_tree"],
            SPARKRING_UPGRADE_INPUT=identity,
            VLLM_TARGET_DEVICE="cpu",
            CUDA_VISIBLE_DEVICES="",
            NVIDIA_VISIBLE_DEVICES="void",
            SPARKRING_FEATURES="",
            SPARKRING_TRANSPORT_PROFILE="",
        )
        result_path = args.output / (variant + ".json")
        with (args.output / (variant + ".log")).open("w") as stream:
            process = subprocess.run(
                [
                    sys.executable,
                    str(upgrades / "kraken_gate.py"),
                    "--source",
                    str(source),
                    "--baseline",
                    str(args.baseline),
                    "--suite",
                    str(suite),
                    "--result",
                    str(result_path),
                    "--peer-vllm-root",
                    str(peer),
                    "--peer-vllm-sha256",
                    inputs["coordinators"][variant],
                ],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result = json.loads(result_path.read_text()) if result_path.exists() else {}
        summary = {
            "variant": variant,
            "returncode": process.returncode,
            "outcome": result.get("outcome"),
            "tests": result.get("assertions"),
            "skipped": result.get("skipped"),
            "scopes": result.get("scopes"),
        }
        outcomes.append(summary)
        print(json.dumps(summary), flush=True)
    assert all(
        result["outcome"] == "passed" and result["returncode"] == 0
        for result in outcomes
    ), outcomes


if __name__ == "__main__":
    main()
