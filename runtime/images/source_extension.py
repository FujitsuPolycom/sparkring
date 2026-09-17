"""Install bounded Python source changes over a receipt-pinned SparkRing image.

The descriptor names every replacement, its inherited bytes, and its resulting
bytes. Installation checks all inputs before writing. Native libraries, feature
hooks, distribution versions, and unrelated receipt metadata remain inherited.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tempfile

SITE = "/opt/venv/lib/python3.12/site-packages/"
RECEIPT = "/opt/sparkring/receipts/candidate-installed.json"
PARENT_RECEIPT = "/opt/sparkring/receipts/source-parent-installed.json"
DESCRIPTOR = "/opt/sparkring/receipts/source-extension.json"
INSTALLER = "/opt/sparkring/bin/source-extension.py"
PATCH = "/opt/sparkring/receipts/source-extension.patch"
RESERVED = (PARENT_RECEIPT, DESCRIPTOR, INSTALLER, PATCH)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_object(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(data, object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("JSON document must be an object")
    return value


def sha256(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("A lowercase SHA256 digest is required")
    return value


def relative_path(value):
    if (not isinstance(value, str) or not value or "\\" in value or ":" in value
            or any(part in {"", ".", ".."} for part in value.split("/"))
            or PurePosixPath(value).is_absolute()):
        raise ValueError("Input must be a contained relative path")
    return PurePosixPath(value)


def absolute_path(value):
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError("Installation path must be absolute")
    relative_path(value[1:])
    return PurePosixPath(value)


def contained(root, relative, *, writable=False):
    """Bound reads to root; reject symlink and non-directory write ancestors."""
    root = Path(root).resolve()
    relative = relative_path(relative)
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root):
        raise ValueError("Path escapes its filesystem root: " + str(relative))
    if writable:
        cursor = root
        for index, part in enumerate(relative.parts):
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("Symlink is not admitted: " + str(relative))
            if index < len(relative.parts) - 1 and cursor.exists() and not cursor.is_dir():
                raise ValueError("Path ancestor is not a directory: " + str(relative))
    return target


def located(root, absolute, *, writable=False):
    absolute_path(absolute)
    return contained(root, absolute[1:], writable=writable)


def pinned_bytes(raw, expected):
    """Recover exactly pinned line endings across Git checkout platforms."""
    canonical = raw.replace(b"\r\n", b"\n")
    for candidate in (raw, canonical, canonical.replace(b"\n", b"\r\n")):
        if digest(candidate) == expected:
            return candidate
    raise ValueError("Source differs from its pinned bytes")


def descriptor(data):
    record = json_object(data)
    required = {"schema", "id", "parent", "installer_sha256", "provenance",
                "sources", "integration_contracts", "patch"}
    if set(record) != required or record.get("schema") != "sparkring-source-extension/v1":
        raise ValueError("Unsupported source extension descriptor fields or schema")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{0,100}", record.get("id", "")):
        raise ValueError("Invalid source extension identity")
    parent = record["parent"]
    if (not isinstance(parent, dict) or set(parent) not in (
            {"image_id", "receipt_sha256"}, {"image_id", "receipt_sha256", "reference"})
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", parent.get("image_id", ""))):
        raise ValueError("Exact parent image ID and receipt digest are required")
    sha256(parent["receipt_sha256"])
    if "reference" in parent and not re.fullmatch(
            r"[a-zA-Z0-9][a-zA-Z0-9._:/-]*@sha256:[0-9a-f]{64}", parent["reference"]):
        raise ValueError("Parent reference must use an immutable registry digest")
    sha256(record["installer_sha256"])
    if not isinstance(record["provenance"], dict) or not record["provenance"]:
        raise ValueError("Explicit source provenance is required")
    patch = record["patch"]
    if not isinstance(patch, dict) or set(patch) != {"source", "sha256"}:
        raise ValueError("Patch requires source and digest")
    relative_path(patch["source"])
    sha256(patch["sha256"])
    if not isinstance(record["sources"], dict) or not record["sources"]:
        raise ValueError("Source inventory must not be empty")
    for name, item in record["sources"].items():
        path = relative_path(name)
        if (not re.fullmatch(r"(?:b12x|vllm)/[A-Za-z0-9_./-]+\.py", name)
                or "__pycache__" in path.parts):
            raise ValueError("Only b12x and vllm Python package sources are admitted")
        if not isinstance(item, dict) or set(item) != {"sha256", "parent_sha256"}:
            raise ValueError("Source requires resulting digest and parent digest or null")
        sha256(item["sha256"])
        if item["parent_sha256"] is not None:
            sha256(item["parent_sha256"])
    if not isinstance(record["integration_contracts"], dict):
        raise ValueError("Integration contracts must be a mapping")
    for target, item in record["integration_contracts"].items():
        if not re.fullmatch(r"/opt/sparkring/contracts/[A-Za-z0-9][A-Za-z0-9_.-]*\.json", target):
            raise ValueError("Integration contract requires a JSON basename under contracts")
        if not isinstance(item, dict) or set(item) != {"source", "sha256"}:
            raise ValueError("Integration contract requires source and digest")
        relative_path(item["source"])
        sha256(item["sha256"])
    return record


def assets(record):
    return {**{SITE + name: item for name, item in record["sources"].items()},
            **record["integration_contracts"]}


def validate_payload(target, raw, expected):
    if digest(raw) != expected:
        raise ValueError("Payload digest mismatch: " + target)
    if target.endswith(".py"):
        compile(raw, target, "exec")
    else:
        json_object(raw)


def dockerfile(record):
    return (f'ARG PARENT_IMAGE={record["parent"]["image_id"]}\n'
            "FROM ${PARENT_IMAGE}\n"
            "COPY . /tmp/sparkring-source-context/\n"
            "RUN /opt/venv/bin/python /tmp/sparkring-source-context/source_extension.py install "
            "--context /tmp/sparkring-source-context && rm -rf /tmp/sparkring-source-context\n"
            f'LABEL org.sparkring.source-extension="{record["id"]}"\n'
            f'ENTRYPOINT ["/opt/venv/bin/python", "{INSTALLER}"]\n'
            'CMD ["--help"]\n').encode()


def prepare(descriptor_path, repository, output):
    repository = Path(repository).resolve()
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError("Build context must not exist")
    raw_descriptor = Path(descriptor_path).read_bytes()
    record = descriptor(raw_descriptor)
    patch = pinned_bytes(contained(repository, record["patch"]["source"], writable=True).read_bytes(),
                         record["patch"]["sha256"])
    validate_patch(patch, record)
    payloads = {}
    for target, item in record["integration_contracts"].items():
        source = contained(repository, item["source"], writable=True)
        raw = pinned_bytes(source.read_bytes(), item["sha256"])
        validate_payload(target, raw, item["sha256"])
        payloads[target] = raw
    installer = pinned_bytes(Path(__file__).read_bytes(), record["installer_sha256"])
    # No output is created until every source and the installer have been checked.
    output.mkdir(parents=True)
    for target, raw in payloads.items():
        destination = output / "payload" / target.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(raw)
    (output / "descriptor.json").write_bytes(raw_descriptor)
    (output / "source_extension.py").write_bytes(installer)
    (output / "source.patch").write_bytes(patch)
    (output / "Dockerfile").write_bytes(dockerfile(record))
    return {"descriptor_sha256": digest(raw_descriptor), "assets": len(assets(record)),
            "context": str(output.resolve())}


def validate_patch(raw, record):
    """Admit ordinary unified text edits with exactly the declared Python paths."""
    if digest(raw) != record["patch"]["sha256"]:
        raise ValueError("Patch differs from its pinned bytes")
    lines = raw.decode("utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts or starts[0] != 0:
        raise ValueError("Patch must contain ordinary Git text diffs")
    found = set()
    for start, end in zip(starts, [*starts[1:], len(lines)]):
        block = lines[start:end]
        match = re.fullmatch(r"diff --git a/([^ ]+) b/([^ ]+)", block[0])
        if not match or match[1] != match[2] or match[1] not in record["sources"]:
            raise ValueError("Patch path differs from the declared source inventory")
        name = match[1]
        if name in found:
            raise ValueError("Patch contains a duplicate source path")
        found.add(name)
        new = record["sources"][name]["parent_sha256"] is None
        index = 1
        if new:
            if block[index:index + 1] != ["new file mode 100644"]:
                raise ValueError("Source addition must be a regular non-executable file")
            index += 1
        if index < len(block) and block[index].startswith("index "):
            if not re.fullmatch(r"index [0-9a-f]+\.\.[0-9a-f]+(?: 100(?:644|755))?", block[index]):
                raise ValueError("Patch index or file mode is unsupported")
            index += 1
        before = "/dev/null" if new else "a/" + name
        if block[index:index + 2] != ["--- " + before, "+++ b/" + name]:
            raise ValueError("Patch must not delete, rename, or change source file modes")
        index += 2
        if index >= len(block) or not block[index].startswith("@@ "):
            raise ValueError("Patch requires a text hunk for each source")
        while index < len(block):
            hunk = re.fullmatch(r"@@ -[0-9]+(?:,([0-9]+))? \+[0-9]+(?:,([0-9]+))? @@.*", block[index])
            if not hunk:
                raise ValueError("Patch contains an undeclared header or malformed hunk")
            remaining = [int(hunk[1] or 1), int(hunk[2] or 1)]
            index += 1
            while any(remaining):
                if index >= len(block) or not block[index]:
                    raise ValueError("Patch hunk has fewer lines than declared")
                line = block[index]
                if line == "\\ No newline at end of file":
                    index += 1
                    continue
                if line[0] not in {" ", "+", "-"}:
                    raise ValueError("Patch contains an unsupported hunk line")
                remaining[0] -= line[0] in {" ", "-"}
                remaining[1] -= line[0] in {" ", "+"}
                if min(remaining) < 0:
                    raise ValueError("Patch hunk exceeds its declared line counts")
                index += 1
            if index < len(block) and block[index] == "\\ No newline at end of file":
                index += 1
    if found != set(record["sources"]):
        raise ValueError("Patch does not cover the complete source inventory")


def apply_sources(record, patch, root):
    """Apply the admitted patch in a private Git tree, never in the installation."""
    validate_patch(patch, record)
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
    with tempfile.TemporaryDirectory(prefix="sparkring-source-") as directory:
        temporary = Path(directory)
        tree = temporary / "tree"
        tree.mkdir()
        for name, item in record["sources"].items():
            if item["parent_sha256"] is None:
                continue
            raw = located(root, SITE + name, writable=True).read_bytes()
            if digest(raw) != item["parent_sha256"]:
                raise ValueError("Replacement preimage changed: " + name)
            target = tree.joinpath(*PurePosixPath(name).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        patch_path = temporary / "source.patch"
        patch_path.write_bytes(patch)
        for arguments in (["init", "--quiet"], ["apply", "--check", str(patch_path)],
                          ["apply", "--whitespace=nowarn", str(patch_path)]):
            result = subprocess.run(["git", *arguments], cwd=tree, env=environment,
                                    capture_output=True, text=True, encoding="utf-8")
            if result.returncode:
                raise ValueError("Source patch cannot be applied: " + result.stderr.strip())
        actual = {path.relative_to(tree).as_posix() for path in tree.rglob("*")
                  if path.is_file() and ".git" not in path.relative_to(tree).parts}
        if actual != set(record["sources"]):
            raise ValueError("Patched tree differs from the complete source inventory")
        payloads = {}
        for name, item in record["sources"].items():
            raw = contained(tree, name, writable=True).read_bytes()
            validate_payload(SITE + name, raw, item["sha256"])
            payloads[SITE + name] = raw
        return payloads


def validate_parent(parent):
    if (parent.get("schema") != "sparkring-candidate-installed/v1"
            or not isinstance(parent.get("files"), dict) or not parent["files"]
            or not isinstance(parent.get("versions"), dict) or not parent["versions"]):
        raise ValueError("Unsupported or empty parent installed receipt")
    if "source_extension" in parent:
        raise ValueError("Parent already contains a source extension; select its recorded base")
    for name, expected in parent["files"].items():
        absolute_path(name)
        sha256(expected)
    if RECEIPT in parent["files"] or any(name in parent["files"] for name in RESERVED):
        raise ValueError("Source receipt paths collide with inherited inventory")


def verify_inventory(receipt, root, version_reader):
    for name, expected in receipt["files"].items():
        target = located(root, name)
        if not target.is_file() or file_digest(target) != expected:
            raise ValueError("Installed inventory mismatch: " + name)
    for name in receipt.get("removed_authored_files", []):
        target = located(root, name)
        if target.exists() or target.is_symlink():
            raise ValueError("Removed authored file reappeared: " + name)
    for name, expected in receipt["versions"].items():
        if version_reader(name) != expected:
            raise ValueError("Installed distribution version mismatch: " + name)


def expected_receipt(parent, record, raw_descriptor, parent_raw, payloads, installer, patch):
    validate_parent(parent)
    result = copy.deepcopy(parent)
    for name, item in record["sources"].items():
        target = SITE + name
        if target in parent.get("removed_authored_files", []):
            raise ValueError("Source addition conflicts with inherited removal: " + target)
        if item["parent_sha256"] is None:
            if target in parent["files"]:
                raise ValueError("Source addition already belongs to the parent: " + target)
        elif parent["files"].get(target) != item["parent_sha256"]:
            raise ValueError("Replacement lacks matching inherited ownership: " + target)
    for target in record["integration_contracts"]:
        if target in parent["files"]:
            raise ValueError("Integration contract must use a new destination")
    additions = {**payloads, PARENT_RECEIPT: parent_raw,
                 DESCRIPTOR: raw_descriptor, INSTALLER: installer, PATCH: patch}
    result["files"].update({name: digest(raw) for name, raw in additions.items()})
    result["source_extension"] = {
        "id": record["id"], "descriptor_sha256": digest(raw_descriptor),
        "parent_image_id": record["parent"]["image_id"],
        "parent_receipt_sha256": digest(parent_raw),
        "provenance": copy.deepcopy(record["provenance"]),
        "qualification": "Source inventory verified; serving requires profile validation.",
    }
    return result


def install(context, filesystem_root=Path("/"), version_reader=importlib.metadata.version):
    context, root = Path(context).resolve(), Path(filesystem_root).resolve()
    raw_descriptor = contained(context, "descriptor.json", writable=True).read_bytes()
    record = descriptor(raw_descriptor)
    receipt_path = located(root, RECEIPT, writable=True)
    parent_raw = receipt_path.read_bytes()
    if digest(parent_raw) != record["parent"]["receipt_sha256"]:
        raise ValueError("Parent receipt differs from the selected image")
    parent = json_object(parent_raw)
    validate_parent(parent)
    verify_inventory(parent, root, version_reader)
    patch = contained(context, "source.patch", writable=True).read_bytes()
    payloads = apply_sources(record, patch, root)
    for target, item in record["integration_contracts"].items():
        raw = contained(context, "payload/" + target[1:], writable=True).read_bytes()
        validate_payload(target, raw, item["sha256"])
        payloads[target] = raw
    installer = contained(context, "source_extension.py", writable=True).read_bytes()
    if digest(installer) != record["installer_sha256"]:
        raise ValueError("Installer source differs from its pinned bytes")
    result = expected_receipt(parent, record, raw_descriptor, parent_raw, payloads, installer, patch)
    writes = {**payloads, PARENT_RECEIPT: parent_raw, DESCRIPTOR: raw_descriptor,
              INSTALLER: installer, PATCH: patch}
    targets = {}
    for target in writes:
        path = located(root, target, writable=True)
        if target not in parent["files"] and path.exists():
            raise ValueError("Unlisted installation path already exists: " + target)
        if target in parent["files"] and not path.is_file():
            raise ValueError("Replacement must be a regular inherited file: " + target)
        targets[target] = path
    # An isolated image build owns this filesystem. All admission failures above
    # occur before writes; an interrupted filesystem write fails the build layer.
    for target, raw in writes.items():
        path = targets[target]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
    receipt_path.write_bytes((json.dumps(result, indent=2, sort_keys=True) + "\n").encode())
    return verify(root, version_reader)


def verify(filesystem_root=Path("/"), version_reader=importlib.metadata.version):
    root = Path(filesystem_root).resolve()
    raw_descriptor = located(root, DESCRIPTOR, writable=True).read_bytes()
    record = descriptor(raw_descriptor)
    parent_raw = located(root, PARENT_RECEIPT, writable=True).read_bytes()
    if digest(parent_raw) != record["parent"]["receipt_sha256"]:
        raise ValueError("Recorded parent receipt changed")
    parent = json_object(parent_raw)
    patch = located(root, PATCH, writable=True).read_bytes()
    validate_patch(patch, record)
    payloads = {}
    for target, item in assets(record).items():
        raw = located(root, target, writable=True).read_bytes()
        validate_payload(target, raw, item["sha256"])
        payloads[target] = raw
    installer = located(root, INSTALLER, writable=True).read_bytes()
    if digest(installer) != record["installer_sha256"]:
        raise ValueError("Installed source verifier changed")
    expected = expected_receipt(parent, record, raw_descriptor, parent_raw, payloads, installer, patch)
    receipt_raw = located(root, RECEIPT, writable=True).read_bytes()
    receipt = json_object(receipt_raw)
    if receipt != expected:
        raise ValueError("Complete installed inventory differs from the source composition")
    verify_inventory(receipt, root, version_reader)
    return {"schema": "sparkring-source-verification/v1",
            "descriptor_sha256": digest(raw_descriptor), "receipt_sha256": digest(receipt_raw),
            "files_verified": len(receipt["files"]), "serving_qualified": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    prepare_cmd = commands.add_parser("prepare")
    prepare_cmd.add_argument("--descriptor", type=Path, required=True)
    prepare_cmd.add_argument("--repository", type=Path, required=True)
    prepare_cmd.add_argument("--output", type=Path, required=True)
    install_cmd = commands.add_parser("install")
    install_cmd.add_argument("--context", type=Path, required=True)
    commands.add_parser("verify")
    serve_cmd = commands.add_parser("serve")
    serve_cmd.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.action == "prepare":
        result = prepare(args.descriptor, args.repository, args.output)
    elif args.action == "install":
        result = install(args.context)
    else:
        result = verify()
        if args.action == "serve":
            command = ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main",
                       "serve", *args.arguments]
            os.execve(command[0], command, os.environ)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
