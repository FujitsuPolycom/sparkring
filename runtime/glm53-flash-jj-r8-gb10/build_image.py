#!/usr/bin/env python3
"""Build the source-pinned Linux/ARM64 GLM-5.3 and SparkCache image."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = HERE / "pins.json"


class BuildError(RuntimeError):
    """A build input differs from the immutable image contract."""


def run(
    argv: Iterable[str], *, cwd: Path | None = None, binary: bool = False
) -> str | bytes:
    arguments = tuple(str(argument) for argument in argv)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        text=not binary,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr or completed.stdout
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise BuildError(f"command failed ({' '.join(arguments)}): {detail.strip()}")
    return completed.stdout


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pins() -> dict[str, Any]:
    value = json.loads(PINS.read_text(encoding="utf-8"))
    if value.get("schema") != "sparkring-glm53-jj-r8-gb10-image/v1":
        raise BuildError("pins.json uses an unsupported schema")
    return value


def verify_git_source(source: Path, record: dict[str, Any], subtree: str) -> None:
    observed_commit = str(run(("git", "-C", str(source), "rev-parse", "HEAD"))).strip()
    observed_tree = str(
        run(("git", "-C", str(source), "rev-parse", "HEAD^{tree}"))
    ).strip()
    if observed_commit != record["commit"]:
        raise BuildError(
            f"{subtree} commit mismatch: {observed_commit} != {record['commit']}"
        )
    if observed_tree != record["tree"]:
        raise BuildError(
            f"{subtree} tree mismatch: {observed_tree} != {record['tree']}"
        )
    package_tree = str(
        run(
            (
                "git",
                "-C",
                str(source),
                "rev-parse",
                f"{record['commit']}:{subtree}",
            )
        )
    ).strip()
    if package_tree != record["package_tree"]:
        raise BuildError(
            f"{subtree} package tree mismatch: {package_tree} != "
            f"{record['package_tree']}"
        )
    dirty = str(
        run(("git", "-C", str(source), "status", "--porcelain", "--", subtree))
    ).strip()
    if dirty:
        raise BuildError(f"{subtree} source differs from its checked-out revision")
    for relative, expected in record.get("runtime_file_sha256", {}).items():
        payload = run(
            ("git", "-C", str(source), "show", f"{record['commit']}:{relative}"),
            binary=True,
        )
        assert isinstance(payload, bytes)
        observed = hashlib.sha256(payload).hexdigest()
        if observed != expected:
            raise BuildError(
                f"{relative} identity mismatch: {observed} != {expected}"
            )
    if "delta_patch_id" in record:
        delta = subprocess.run(
            (
                "git",
                "-C",
                str(source),
                "diff",
                f"{record['proven_base_commit']}..{record['commit']}",
            ),
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        patch = subprocess.run(
            ("git", "patch-id", "--stable"),
            input=delta,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout.decode("ascii").split()[0]
        if patch != record["delta_patch_id"]:
            raise BuildError(
                f"{subtree} delta patch ID mismatch: {patch} != "
                f"{record['delta_patch_id']}"
            )


def extract_git_subtree(source: Path, commit: str, subtree: str, output: Path) -> None:
    payload = run(
        ("git", "-C", str(source), "archive", "--format=tar", commit, subtree),
        binary=True,
    )
    assert isinstance(payload, bytes)
    prefix = PurePosixPath(subtree)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name)
            if relative == prefix:
                continue
            if not relative.is_relative_to(prefix) or ".." in relative.parts:
                raise BuildError(f"Git archive escaped {subtree}: {member.name}")
            destination = output.joinpath(*relative.relative_to(prefix).parts)
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise BuildError(f"Git archive contains a non-file: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise BuildError(f"cannot read Git archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(stream.read())


def source_manifest(root: Path, package: str, commit: str) -> dict[str, Any]:
    files = {
        f"{package}/{path.relative_to(root).as_posix()}": file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    if not files:
        raise BuildError(f"source package is empty: {package}")
    return {"commit": commit, "files": files}


def sparkcache_source_sha256(root: Path) -> str:
    digest = hashlib.sha256(b"sparkcache-source-tree/v1\x00")
    count = 0
    for path in sorted(
        (candidate for candidate in root.rglob("*") if candidate.is_file()),
        key=lambda candidate: candidate.relative_to(root).as_posix(),
    ):
        relative = path.relative_to(root)
        if (
            any(part in {"__pycache__", ".pytest_cache", "build"} for part in relative.parts)
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(hashlib.sha256(normalized).digest())
        count += 1
    if not count:
        raise BuildError("SparkCache source tree is empty")
    return digest.hexdigest()


def inspect_image(engine: str, image: str) -> dict[str, Any]:
    documents = json.loads(str(run((engine, "image", "inspect", image))))
    if not isinstance(documents, list) or len(documents) != 1:
        raise BuildError("container engine returned an unexpected inspection")
    return documents[0]


def validate_parent(document: dict[str, Any], pins: dict[str, Any]) -> None:
    if document.get("Architecture") != "arm64" or document.get("Os") != "linux":
        raise BuildError("parent image must use linux/arm64")
    if document.get("Id") != pins["parent"]["image_id"]:
        raise BuildError("parent image ID differs from pins.json")
    labels = document.get("Config", {}).get("Labels") or {}
    for name, expected in pins["parent"]["required_labels"].items():
        if labels.get(name) != expected:
            raise BuildError(
                f"parent label {name} drift: {labels.get(name)!r} != {expected!r}"
            )


def record_native_extensions(engine: str, image: str) -> dict[str, Any]:
    code = """
import hashlib, json, pathlib
root = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm')
files = {}
for path in sorted(root.rglob('*.so')):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files[path.relative_to(root).as_posix()] = digest
print(json.dumps({'files': files}, sort_keys=True))
"""
    result = json.loads(
        str(
            run(
                (
                    engine,
                    "run",
                    "--rm",
                    "--entrypoint",
                    "python3",
                    image,
                    "-c",
                    code,
                )
            )
        )
    )
    if not result.get("files"):
        raise BuildError("parent image contains no vLLM native extensions")
    return result


def prepare_context(
    context: Path,
    *,
    vllm_source: Path,
    sparkcache_source: Path,
    native_extensions: dict[str, Any],
    pins: dict[str, Any],
) -> dict[str, Any]:
    if context.exists():
        raise BuildError(f"build context already exists: {context}")
    (context / "bundle/sources").mkdir(parents=True)
    verify_git_source(vllm_source, pins["vllm"], "vllm")
    verify_git_source(sparkcache_source, pins["sparkcache"], "sparkcache")
    vllm_output = context / "bundle/sources/vllm"
    sparkcache_output = context / "bundle/sources/sparkcache"
    extract_git_subtree(vllm_source, pins["vllm"]["commit"], "vllm", vllm_output)
    extract_git_subtree(
        sparkcache_source,
        pins["sparkcache"]["commit"],
        "sparkcache",
        sparkcache_output,
    )
    observed_source = sparkcache_source_sha256(sparkcache_output)
    if observed_source != pins["sparkcache"]["source_tree_sha256"]:
        raise BuildError(
            "SparkCache source digest mismatch: "
            f"{observed_source} != {pins['sparkcache']['source_tree_sha256']}"
        )
    receipts = context / "bundle/receipts"
    receipts.mkdir(parents=True)
    manifests = {
        "vllm-source-manifest.json": source_manifest(
            vllm_output, "vllm", pins["vllm"]["commit"]
        ),
        "sparkcache-source-manifest.json": source_manifest(
            sparkcache_output, "sparkcache", pins["sparkcache"]["commit"]
        ),
        "native-extension-manifest.json": native_extensions,
        "pins.json": pins,
    }
    for name, value in manifests.items():
        (receipts / name).write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    for name in ("Dockerfile", "install_overlay.py", "verify_image.py"):
        shutil.copy2(HERE / name, context / name)
    receipt = {
        "schema": "sparkring-glm53-jj-r8-gb10-build-context/v1",
        "status": "implemented",
        "sources": {
            "vllm": {
                "commit": pins["vllm"]["commit"],
                "tree": pins["vllm"]["tree"],
                "files": len(manifests["vllm-source-manifest.json"]["files"]),
            },
            "sparkcache": {
                "commit": pins["sparkcache"]["commit"],
                "tree": pins["sparkcache"]["tree"],
                "source_tree_sha256": observed_source,
                "files": len(manifests["sparkcache-source-manifest.json"]["files"]),
            },
            "native_extensions": len(native_extensions["files"]),
        },
        "inputs": {
            relative: file_sha256(context / relative)
            for relative in (
                "Dockerfile",
                "install_overlay.py",
                "verify_image.py",
                "bundle/receipts/pins.json",
                "bundle/receipts/vllm-source-manifest.json",
                "bundle/receipts/sparkcache-source-manifest.json",
                "bundle/receipts/native-extension-manifest.json",
            )
        },
        "limitation": "This receipt verifies source preparation, not a built image.",
    }
    receipt_path = context / "bundle/receipts/source-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-source", type=Path, required=True)
    parser.add_argument("--sparkcache-source", type=Path, required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--engine", default="docker")
    parser.add_argument("--context-root", type=Path)
    args = parser.parse_args()
    pins = load_pins()
    engine = args.engine
    parent_ref = pins["parent"]["reference"]
    run((engine, "pull", "--platform", "linux/arm64", parent_ref))
    parent = inspect_image(engine, parent_ref)
    validate_parent(parent, pins)
    native_extensions = record_native_extensions(engine, parent_ref)
    sparkring_revision = str(run(("git", "-C", str(ROOT), "rev-parse", "HEAD"))).strip()
    dirty = str(
        run(
            (
                "git",
                "-C",
                str(ROOT),
                "status",
                "--porcelain",
                "--",
                str(HERE.relative_to(ROOT)),
            )
        )
    ).strip()
    if dirty:
        raise BuildError("image builder inputs differ from the SparkRing revision")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.context_root:
        context = args.context_root.resolve()
    else:
        temporary = tempfile.TemporaryDirectory(prefix="sparkring-jj-r8-image-")
        context = Path(temporary.name) / "context"
    try:
        prepare_context(
            context,
            vllm_source=args.vllm_source.resolve(),
            sparkcache_source=args.sparkcache_source.resolve(),
            native_extensions=native_extensions,
            pins=pins,
        )
        source_receipt = context / "bundle/receipts/source-receipt.json"
        run(
            (
                engine,
                "build",
                "--platform",
                "linux/arm64",
                "--file",
                str(context / "Dockerfile"),
                "--build-arg",
                f"PARENT_IMAGE={parent_ref}",
                "--build-arg",
                f"PARENT_IMAGE_ID={pins['parent']['image_id']}",
                "--build-arg",
                f"SPARKRING_REVISION={sparkring_revision}",
                "--build-arg",
                f"SOURCE_RECEIPT_SHA256={file_sha256(source_receipt)}",
                "--tag",
                args.output_image,
                str(context),
            )
        )
        verify = HERE / "verify_image.py"
        run(
            (
                "python3",
                str(verify),
                "--engine",
                engine,
                "--image",
                args.output_image,
                "--output",
                str(args.receipt.resolve()),
            )
        )
    finally:
        if temporary is not None:
            temporary.cleanup()
    print(f"image={args.output_image}")
    print(f"receipt={args.receipt.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
