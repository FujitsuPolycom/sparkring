"""Derive an installer image that adds or replaces a few Python files in one layer.

An installer image lock (``runtime/releases/<release>/installer-image.json``)
pins the image and the SHA-256 of two receipts inside it. The external-base
receipt, ``/opt/sparkring/receipts/external-base-installed.json``, maps every
installed file that the image's ``verify`` checks to its SHA-256. The toolchain
receipt, ``/opt/sparkring/toolchain/installed.json``, records the external-base
receipt's SHA-256. A derived layer copies its files over the parent, records
each file's SHA-256 in the external-base receipt, re-records the toolchain
receipt, and adds a provenance receipt listing every path with its inherited
and resulting SHA-256.

``prepare`` works offline from a descriptor and the parent's two receipts and
writes a Docker build context. ``record`` inspects the image built from that
context, runs the installer's admission for every profile of the lock
(``runtime/common/installer_image.py``, including the image's isolated
``verify``), and writes the derived image lock. Neither action pushes,
publishes or selects an image for a profile.

Derived files live in the serving interpreter's site-packages. The builder
refuses native libraries, the startup hooks that select the transport, feature
and status packages, and the B12X sources whose SHA-256 the prepared RoCE
transport verifies at startup: changing those needs a new transport manifest.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from integrations.vllm.rocenante_prepared.sparkring_transport_selector import HOST_SOURCE_FILES  # noqa: E402
from runtime.common import installer_image  # noqa: E402
from runtime.images.feature_extension import pinned_bytes  # noqa: E402

SCHEMA = "sparkring-derived-layer-descriptor/v1"
SITE = "/usr/local/lib/python3.12/dist-packages/"
BASE_RECEIPT = installer_image.PARENT_RECEIPT
TOOLCHAIN_RECEIPT = installer_image.TOOLCHAIN_RECEIPT
RECEIPTS = "/opt/sparkring/receipts/"
HOOKS = {SITE + name for name in (
    "sparkring_features.pth", "sparkring_transport.pth", "sparkring_runtime_status.pth")}
TRANSPORT_SOURCES = {SITE + "b12x/" + name for name in HOST_SOURCE_FILES}
NATIVE = (".so", ".dll", ".pyd", ".a", ".o", ".pyc")
DOCKERFILE = "ARG PARENT_IMAGE\nFROM ${PARENT_IMAGE}\nCOPY files/ /\n"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _image_path(name, *, root):
    path = PurePosixPath(name)
    if (not path.is_absolute() or str(path) != name or ".." in path.parts or "\\" in name
            or not name.startswith(root)):
        raise ValueError(f"Derived path must be a normalized absolute path under {root}: {name}")
    return name


def descriptor(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if record.get("schema") != SCHEMA:
        raise ValueError("Unsupported derived-layer descriptor")
    if not isinstance(record.get("id"), str) or not re.fullmatch(r"[a-z][a-z0-9-]{0,63}", record["id"]):
        raise ValueError("Derived-layer id must be a short lowercase identifier")
    if not isinstance(record.get("purpose"), str) or not record["purpose"].strip():
        raise ValueError("Derived-layer descriptor requires a purpose")
    _image_path(record.get("provenance", ""), root=RECEIPTS)
    files = record.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("Derived-layer descriptor lists no files")
    for target, row in files.items():
        _image_path(target, root=SITE)
        name = PurePosixPath(target).name
        if name.endswith(NATIVE) or ".so." in name:
            raise ValueError("A derived layer carries Python source, not native files: " + target)
        if target in HOOKS:
            raise ValueError("Startup hooks that select image components are not derived-layer files: " + target)
        if target in TRANSPORT_SOURCES:
            raise ValueError("The prepared transport verifies this B12X source; it needs a new transport manifest: "
                             + target)
        if set(row) != {"source", "sha256", "inherited_sha256"}:
            raise ValueError("Each derived file records source, sha256 and inherited_sha256: " + target)
        source = PurePosixPath(row["source"])
        if source.is_absolute() or ".." in source.parts or "\\" in row["source"]:
            raise ValueError("Derived file source must be a contained repository path: " + target)
        for key in ("sha256", "inherited_sha256"):
            value = row[key]
            if not (value is None and key == "inherited_sha256") and not (
                    isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)):
                raise ValueError(f"Derived file {key} must be a SHA-256: {target}")
    return record


def _parent_lock(record, repository):
    path = repository / record["parent_lock"]
    if not path.resolve().is_relative_to(repository):
        raise ValueError("Parent lock escapes the repository")
    lock = json.loads(path.read_text(encoding="utf-8"))
    for profile in installer_image.profiles_of(lock):
        installer_image.validate(lock, profile)
    return lock


def prepare(descriptor_path, repository, base_receipt, toolchain_receipt, output):
    """Write a build context whose receipts record the derived files."""
    record = descriptor(descriptor_path)
    repository, output = Path(repository).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Build context must not exist")
    lock = _parent_lock(record, repository)
    base_raw, toolchain_raw = Path(base_receipt).read_bytes(), Path(toolchain_receipt).read_bytes()
    if digest(base_raw) != lock["parent_receipt_sha256"]:
        raise ValueError("External-base receipt differs from the parent lock")
    if digest(toolchain_raw) != lock["toolchain_receipt_sha256"]:
        raise ValueError("Toolchain receipt differs from the parent lock")
    base, toolchain = json.loads(base_raw), json.loads(toolchain_raw)
    if toolchain.get("parent_receipt_sha256") != digest(base_raw):
        raise ValueError("Toolchain receipt does not record the external-base receipt")
    if record["provenance"] in base["files"]:
        raise ValueError("Provenance path already belongs to the parent")
    payloads, rows = {}, {}
    derived = copy.deepcopy(base)
    for target, row in record["files"].items():
        source = repository / row["source"]
        if source.is_symlink() or not source.resolve().is_relative_to(repository):
            raise ValueError("Derived file source escapes the repository: " + row["source"])
        payloads[target] = pinned_bytes(source.read_bytes(), row["sha256"])
        if base["files"].get(target) != row["inherited_sha256"]:
            raise ValueError("Parent receipt records a different inherited file: " + target)
        derived["files"][target] = row["sha256"]
        rows[target] = {"source": row["source"], "inherited_sha256": row["inherited_sha256"],
                        "sha256": row["sha256"]}
    base_out = (json.dumps(derived, indent=2, sort_keys=True) + "\n").encode()
    toolchain["parent_receipt_sha256"] = digest(base_out)
    toolchain_out = (json.dumps(toolchain, indent=2) + "\n").encode()
    provenance = {
        "schema": "sparkring-derived-layer/v1", "id": record["id"], "purpose": record["purpose"],
        "parent_release": lock["name"], "parent_image_id": lock["image_id"],
        "parent_receipt_sha256": digest(base_raw), "files": rows,
        "receipts": {BASE_RECEIPT: digest(base_out), TOOLCHAIN_RECEIPT: digest(toolchain_out)},
    }
    provenance_out = (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode()
    written = {**payloads, BASE_RECEIPT: base_out, TOOLCHAIN_RECEIPT: toolchain_out,
               record["provenance"]: provenance_out}
    output.mkdir(parents=True)
    for target, raw in written.items():
        path = output / "files" / target.lstrip("/")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    (output / "Dockerfile").write_text(DOCKERFILE)
    plan = {
        "schema": "sparkring-derived-layer-plan/v1", "id": record["id"],
        "descriptor_sha256": digest(Path(descriptor_path).read_bytes()), "parent_lock": lock,
        "added": sorted(target for target, row in rows.items() if row["inherited_sha256"] is None),
        "replaced": sorted(target for target, row in rows.items() if row["inherited_sha256"] is not None),
        "receipts": provenance["receipts"], "payload_bytes": sum(len(raw) for raw in written.values()),
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    return {"context": str(output), "files": len(payloads), "receipts": plan["receipts"],
            "build": ["docker", "build", "--build-arg", "PARENT_IMAGE=<local tag of "
                      + lock["image_id"] + ">", "-t", "<tag>", str(output)]}


def derived_lock(plan, image, name):
    """Bind the parent lock's contract to the built image.

    ``image_reference`` is the local configuration ID until a registry digest
    exists. ``download_bytes`` adds the layer's uncompressed payload to the
    parent's download, an upper bound until the registry reports the layer.
    """
    lock = dict(plan["parent_lock"])
    lock.update(name=name, image_id=image["Id"], image_reference=image["Id"], image_bytes=image["Size"],
                download_bytes=lock["download_bytes"] + plan["payload_bytes"],
                parent_receipt_sha256=plan["receipts"][BASE_RECEIPT],
                toolchain_receipt_sha256=plan["receipts"][TOOLCHAIN_RECEIPT])
    for profile in installer_image.profiles_of(lock):
        installer_image.validate(lock, profile)
    return lock


def _run(command, text=True):
    return subprocess.run(command, capture_output=True, check=True, text=text)


def record(context, image_id, name, output, run=_run):
    """Admit the built image for every profile of the lock and write that lock."""
    plan = json.loads((Path(context) / "plan.json").read_text(encoding="utf-8"))
    output = Path(output)
    if output.exists():
        raise ValueError("Output lock must not exist")
    parent = plan["parent_lock"]["image_id"]
    for target in plan["added"]:
        try:
            run(["docker", "run", "--rm", "--pull", "never", "--network", "none", "--entrypoint", "/bin/sh",
                 parent, "-c", 'test ! -e "$1" && test ! -L "$1"', "sh", target])
        except subprocess.CalledProcessError as error:
            raise ValueError("The parent image already has an added path: " + target) from error
    image = json.loads(run(["docker", "image", "inspect", image_id]).stdout)[0]
    lock = derived_lock(plan, image, name)
    for profile in installer_image.profiles_of(lock):
        installer_image.admit(lock, run=run, profile=profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    return {"lock": str(output), "image_id": lock["image_id"], "profiles": lock["profiles"],
            "serving_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="action", required=True)
    prepared = actions.add_parser("prepare", help="write a build context offline")
    prepared.add_argument("--descriptor", required=True, type=Path)
    prepared.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[2])
    prepared.add_argument("--base-receipt", required=True, type=Path,
                          help="the parent's " + BASE_RECEIPT)
    prepared.add_argument("--toolchain-receipt", required=True, type=Path,
                          help="the parent's " + TOOLCHAIN_RECEIPT)
    prepared.add_argument("--output", required=True, type=Path)
    recorded = actions.add_parser("record", help="admit a built image and write its installer lock")
    recorded.add_argument("--context", required=True, type=Path)
    recorded.add_argument("--image", required=True, help="built image configuration ID")
    recorded.add_argument("--name", required=True, help="release name for the derived lock")
    recorded.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.action == "prepare":
            result = prepare(args.descriptor, args.repository, args.base_receipt, args.toolchain_receipt,
                             args.output)
        else:
            result = record(args.context, args.image, args.name, args.output)
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
