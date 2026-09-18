"""Verify a candidate's full inventory and preserved feature-source preimages."""

import importlib.util
import json
import os
from pathlib import Path
import hashlib


def verify_bindings(root=Path("/")):
    assertions, failed = 0, []
    manifest = root / "opt/sparkring/features/qwen-prefill/manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text())
        if value.get("schema") != "sparkring-qwen-prefill/v1" or not value.get(
            "image_source_preimages"
        ):
            failed.append("Missing Qwen prefill source preimages")
        else:
            for name, expected in value["image_source_preimages"].items():
                path = root / name.lstrip("/")
                if (
                    not path.resolve().is_relative_to(root.resolve())
                    or not path.is_file()
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected
                ):
                    failed.append("Feature source preimage changed: " + name)
                assertions += 1
    installed_path = root / "opt/sparkring/receipts/native-installed.json"
    if not installed_path.exists():
        installed_path = root / "opt/sparkring/receipts/candidate-installed.json"
    active = None
    if installed_path.exists():
        installed = json.loads(installed_path.read_text())
        active = installed.get(
            "active_contracts", list(installed.get("integration_contracts", {}))
        )
    contracts = (
        [root / name.lstrip("/") for name in active if "vllm-connector-jobs-" in name]
        if active is not None
        else (root / "opt/sparkring/contracts").glob("vllm-connector-jobs-*.json")
    )
    for contract in contracts:
        value = json.loads(contract.read_text())
        if (
            value.get("schema") != "sparkring-vllm-kv-block-lease-contract/v1"
            or not isinstance(value.get("files"), list)
            or not value["files"]
        ):
            failed.append("Unknown or empty connector contract: " + contract.name)
            continue
        for item in value["files"]:
            name, expected = item["path"], item["sha256"]
            source = root / "opt/venv/lib/python3.12/site-packages" / name
            if (
                not source.resolve().is_relative_to(root.resolve())
                or not source.is_file()
                or hashlib.sha256(source.read_bytes()).hexdigest() != expected
            ):
                failed.append("Connector source binding changed: " + name)
            assertions += 1
    return assertions, failed


def main():
    native = Path("/opt/sparkring/receipts/native-installed.json").exists()
    path = Path(
        "/opt/sparkring/bin/native-image.py"
        if native
        else "/opt/sparkring/bin/candidate-image.py"
    )
    spec = importlib.util.spec_from_file_location("candidate_verifier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.verify()
    assertions, failed = verify_bindings()
    receipt = {
        "schema": "sparkring-upgrade-gate/v1",
        "gate": os.environ["SPARKRING_UPGRADE_GATE"],
        "subject_sha256": os.environ["SPARKRING_UPGRADE_SUBJECT"],
        "variant": os.environ["SPARKRING_UPGRADE_VARIANT"],
        "input_sha256": os.environ["SPARKRING_UPGRADE_INPUT"],
        "assertions": assertions + result["files_verified"],
        "skipped": 0,
        "outcome": "failed" if failed else "passed",
        "findings": failed,
    }
    Path("/out/result.json").write_text(json.dumps(receipt))


if __name__ == "__main__":
    main()
