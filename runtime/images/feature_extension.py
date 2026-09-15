"""Compose opt-in feature sources over an exactly verified SparkRing image."""

import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import shutil

RECEIPT = Path("/opt/sparkring/receipts/candidate-installed.json")
PARENT_RECEIPT = "/opt/sparkring/receipts/feature-parent-installed.json"
DESCRIPTOR = "/opt/sparkring/receipts/feature-extension.json"
INSTALLER = "/opt/sparkring/bin/feature-extension.py"
FEATURE_ROOT = "/opt/sparkring/features/"
SITE = "/opt/venv/lib/python3.12/site-packages/"
HOOKS = {SITE + "sparkring_features.pth", SITE + "sparkring_transport.pth"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def pinned_bytes(data, expected):
    canonical = data.replace(b"\r\n", b"\n")
    for value in (data, canonical, canonical.replace(b"\n", b"\r\n")):
        if digest(value) == expected:
            return value
    raise ValueError("Feature source content differs from its pinned bytes")


def descriptor(path):
    result = json.loads(Path(path).read_text())
    if result.get("schema") != "sparkring-feature-extension/v1":
        raise ValueError("Unsupported feature extension descriptor")
    for target, source in result["assets"].items():
        name = PurePosixPath(target)
        if (
            not name.is_absolute()
            or ".." in name.parts
            or "\\" in target
            or not (target.startswith(FEATURE_ROOT) or target in HOOKS)
        ):
            raise ValueError(
                "Feature installation target is outside its owner: " + target
            )
        if ("source" in source) == ("text" in source):
            raise ValueError("Each asset requires exactly one source or literal text")
        if "source" in source:
            relative = PurePosixPath(source["source"])
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in source["source"]
            ):
                raise ValueError("Feature input must be a contained repository path")
    return result


def prepare(descriptor_path, repository, output):
    record = descriptor(descriptor_path)
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Build context must not exist")
    payloads = {}
    for target, source in record["assets"].items():
        if "source" in source:
            path = repository / source["source"]
            if path.is_symlink() or not path.resolve().is_relative_to(repository):
                raise ValueError("Feature source escapes the repository")
            raw = path.read_bytes()
        else:
            raw = source["text"].encode()
        payloads[target] = pinned_bytes(raw, source["sha256"])
    installer = pinned_bytes(Path(__file__).read_bytes(), record["installer_sha256"])
    output.mkdir(parents=True)
    for target, raw in payloads.items():
        path = output / "payload" / target.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "feature_extension.py").write_bytes(installer)
    (output / "descriptor.json").write_bytes(Path(descriptor_path).read_bytes())
    shutil.copyfile(
        Path(descriptor_path).with_name("Dockerfile"), output / "Dockerfile"
    )
    return {
        "descriptor_sha256": digest((output / "descriptor.json").read_bytes()),
        "assets": len(payloads),
        "context": str(output),
    }


def expected_receipt(parent, record, raw_descriptor, parent_raw, payloads, installer):
    result = copy.deepcopy(parent)
    for path, raw in payloads.items():
        if path in parent["files"]:
            raise ValueError(
                "Feature additions must not replace inherited files: " + path
            )
        result["files"][path] = digest(raw)
    for path, raw in [
        (PARENT_RECEIPT, parent_raw),
        (DESCRIPTOR, raw_descriptor),
        (INSTALLER, installer),
    ]:
        if path in parent["files"]:
            raise ValueError("Feature receipt path already belongs to the parent")
        result["files"][path] = digest(raw)
    result["feature_extension"] = {
        "id": record["id"],
        "descriptor_sha256": digest(raw_descriptor),
        "parent_image_id": record["parent"]["image_id"],
        "parent_receipt_sha256": digest(parent_raw),
        "capabilities": sorted(record["capabilities"]),
    }
    return result


def load_parent_verifier():
    spec = importlib.util.spec_from_file_location(
        "candidate_image", "/opt/sparkring/bin/candidate-image.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def install(context):
    context = Path(context)
    raw_descriptor = (context / "descriptor.json").read_bytes()
    record = descriptor(context / "descriptor.json")
    parent_raw = RECEIPT.read_bytes()
    if digest(parent_raw) != record["parent"]["receipt_sha256"]:
        raise ValueError("Parent receipt differs from the selected image")
    parent = json.loads(parent_raw)
    verifier = load_parent_verifier()
    verifier.verify()
    payloads = {}
    for path, source in record["assets"].items():
        if Path(path).exists() or Path(path).is_symlink():
            raise ValueError(
                "Refusing to replace an existing installation path: " + path
            )
        raw = (context / "payload" / path.lstrip("/")).read_bytes()
        if digest(raw) != source["sha256"]:
            raise ValueError("Prepared feature asset changed: " + path)
        if path.endswith(".py"):
            compile(raw, path, "exec")
        payloads[path] = raw
    installer = (context / "feature_extension.py").read_bytes()
    if digest(installer) != record["installer_sha256"]:
        raise ValueError("Installer source changed")
    result = expected_receipt(
        parent, record, raw_descriptor, parent_raw, payloads, installer
    )
    for path, raw in {
        **payloads,
        PARENT_RECEIPT: parent_raw,
        DESCRIPTOR: raw_descriptor,
        INSTALLER: installer,
    }.items():
        target = Path(path)
        if target.exists() or target.is_symlink():
            raise ValueError("Installation target appeared during preparation: " + path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    RECEIPT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    verifier.verify()
    verify()


def verify():
    record = descriptor(DESCRIPTOR)
    raw_descriptor = Path(DESCRIPTOR).read_bytes()
    parent_raw = Path(PARENT_RECEIPT).read_bytes()
    if digest(parent_raw) != record["parent"]["receipt_sha256"]:
        raise ValueError("Recorded parent receipt changed")
    payloads = {path: Path(path).read_bytes() for path in record["assets"]}
    for path, raw in payloads.items():
        if digest(raw) != record["assets"][path]["sha256"]:
            raise ValueError("Installed feature asset changed: " + path)
    installer = Path(INSTALLER).read_bytes()
    if digest(installer) != record["installer_sha256"]:
        raise ValueError("Installed feature verifier changed")
    expected = expected_receipt(
        json.loads(parent_raw), record, raw_descriptor, parent_raw, payloads, installer
    )
    if json.loads(RECEIPT.read_bytes()) != expected:
        raise ValueError(
            "Complete installed inventory differs from the feature composition"
        )
    evidence = load_parent_verifier().verify()
    return {
        "schema": "sparkring-feature-verification/v1",
        "descriptor_sha256": digest(raw_descriptor),
        "capabilities": sorted(record["capabilities"]),
        "files_verified": evidence["files_verified"],
        "serving_qualified": False,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--descriptor", type=Path, required=True)
    p.add_argument("--repository", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("install")
    p.add_argument("--context", type=Path, required=True)
    sub.add_parser("verify")
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.descriptor, args.repository, args.output)))
    elif args.action == "install":
        install(args.context)
    else:
        print(json.dumps(verify()))
