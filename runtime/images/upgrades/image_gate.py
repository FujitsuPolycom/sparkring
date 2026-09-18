"""Verify a candidate's full inventory and preserved feature-source preimages."""

import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import hashlib


def checked_file(root, name, expected):
    path = PurePosixPath(name)
    if not path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Invalid installed binding path: " + name)
    local = root / name.lstrip("/")
    if (
        not local.resolve().is_relative_to(root.resolve())
        or local.is_symlink()
        or not local.is_file()
        or hashlib.sha256(local.read_bytes()).hexdigest() != expected
    ):
        raise ValueError("Installed binding differs: " + name)
    return local


def selected_manifests(root, installed):
    """Only a receipt-bound feature migration can retire inherited hook targets."""
    update = installed.get("feature_update")
    legacy = root / "opt/sparkring/features/qwen-prefill/manifest.json"
    if update is None:
        return [(legacy, "sparkring-qwen-prefill/v1")] if legacy.exists() else []
    parent = json.loads(
        checked_file(
            root, update["parent_catalog"], update["parent_capabilities_sha256"]
        ).read_text()
    )
    catalog = "/opt/sparkring/features/capabilities.json"
    if update["catalog"] != catalog or not update.get("assets"):
        raise ValueError("Feature migration lacks its owned catalog and assets")
    child = json.loads(
        checked_file(root, catalog, update["catalog_sha256"]).read_text()
    )
    for name, asset in update["assets"].items():
        checked_file(root, name, asset["sha256"])
    if (
        child.get("schema") != "sparkring-image-capabilities/v1"
        or parent.get("schema") != "sparkring-image-capabilities/v1"
        or not child.get("features")
        or update["assets"][catalog]["sha256"] != update["catalog_sha256"]
    ):
        raise ValueError("Unsupported feature migration catalog")
    for name in set(parent["features"]) - set(child["features"]):
        disposition = child.get("unsupported_features", {}).get(name, {})
        if (
            not disposition.get("reason")
            or disposition.get("replacement") not in child["features"]
        ):
            raise ValueError("Feature removal lacks a replacement: " + name)
    manifests = []
    for name, feature in child["features"].items():
        if name not in {"qwen-collectives", "qwen4-prefill", "qwen-prefill"}:
            raise ValueError("Unknown feature binding: " + name)
        directory = PurePosixPath(feature["directory"])
        if directory.is_absolute() or ".." in directory.parts or "\\" in str(directory):
            raise ValueError("Invalid feature directory")
        for relative, expected in feature["files"].items():
            checked_file(root, "/opt/sparkring/features/" + relative, expected)
        if name != "qwen-collectives":
            relative = str(directory / "manifest.json")
            digest = feature["manifest_sha256"]
            if feature["files"].get(relative) != digest:
                raise ValueError("Feature manifest is not inventoried: " + name)
            path = checked_file(root, "/opt/sparkring/features/" + relative, digest)
            manifests.append((path, "sparkring-" + name + "/v1"))
    return manifests


def verify_installed_lease(root, contract):
    """Run the consumer's installed schema/hash/AST checks without importing vLLM."""
    site = root / "opt/venv/lib/python3.12/site-packages"
    path = site / "sparkcache/runtime_patches/verify_lease_contract.py"
    if (
        not path.resolve().is_relative_to(site.resolve())
        or path.is_symlink()
        or not path.is_file()
    ):
        raise ValueError(
            "SparkCache lease verifier is missing or escapes its installed root"
        )
    spec = importlib.util.spec_from_file_location(
        "installed_sparkcache_lease_verifier", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.verify_contract(site, contract)


def verify_bindings(root=Path("/")):
    assertions, failed = 0, []
    installed_path = root / "opt/sparkring/receipts/native-installed.json"
    if not installed_path.exists():
        installed_path = root / "opt/sparkring/receipts/candidate-installed.json"
    installed = (
        json.loads(installed_path.read_text()) if installed_path.exists() else {}
    )
    try:
        manifests = selected_manifests(root, installed)
    except (ValueError, KeyError, TypeError, OSError) as error:
        failed.append("Feature migration binding failed: " + str(error))
        manifests = []
    for manifest, schema in manifests:
        value = json.loads(manifest.read_text())
        if value.get("schema") != schema or not value.get("image_source_preimages"):
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
    active = None
    if installed_path.exists():
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
        try:
            verify_installed_lease(root, contract)
        except Exception as error:
            failed.append(
                "Installed SparkCache lease verifier rejected "
                + contract.name
                + ": "
                + str(error)
            )
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
