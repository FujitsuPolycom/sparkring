#!/usr/bin/env python3
"""Prepare exact public sources for the GLM-5.3 Python-overlay image."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PINS = HERE / "pins.json"
MANIFEST = HERE / "vllm-python-overlay.json"
PINS_SCHEMA = "sparkring-glm53-public-python-overlay/v1"
OVERLAY_SCHEMA = "sparkring-vllm-python-overlay/v1"
RECEIPT_SCHEMA = "sparkring-glm53-public-python-overlay-context/v1"


class PrepareError(RuntimeError):
    """A source input differs from the public Python-overlay contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(argv: Iterable[str], *, cwd: Path | None = None) -> str:
    arguments = list(argv)
    environment = os.environ.copy()
    environment["GIT_TERMINAL_PROMPT"] = "0"
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PrepareError(
            f"command did not complete ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout.strip()


def load_json(path: Path, schema: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise PrepareError(f"{path} does not use schema {schema}")
    return value


def clone_detached(
    destination: Path,
    *,
    repository: str,
    commit: str,
    tree: str,
    fetch_depth: int,
) -> None:
    destination.mkdir(parents=True)
    run(("git", "init", "--quiet", str(destination)))
    run(("git", "-C", str(destination), "config", "core.autocrlf", "false"))
    run(("git", "-C", str(destination), "config", "core.longpaths", "true"))
    run(("git", "-C", str(destination), "remote", "add", "origin", repository))
    run(
        (
            "git",
            "-C",
            str(destination),
            "fetch",
            "--quiet",
            "--depth",
            str(fetch_depth),
            "origin",
            commit,
        )
    )
    run(("git", "-C", str(destination), "checkout", "--quiet", "--detach", "FETCH_HEAD"))
    verify_git_source(destination, commit=commit, tree=tree)


def verify_git_source(source: Path, *, commit: str, tree: str) -> None:
    observed_commit = run(("git", "-C", str(source), "rev-parse", "HEAD"))
    observed_tree = run(("git", "-C", str(source), "rev-parse", "HEAD^{tree}"))
    if observed_commit != commit:
        raise PrepareError(
            f"{source.name} commit mismatch: expected {commit}, got {observed_commit}"
        )
    if observed_tree != tree:
        raise PrepareError(
            f"{source.name} tree mismatch: expected {tree}, got {observed_tree}"
        )
    if run(("git", "-C", str(source), "status", "--porcelain")):
        raise PrepareError(f"{source.name} source tree is not clean")


def _load_overlay_contract():
    path = HERE / "overlay_contract.py"
    spec = importlib.util.spec_from_file_location("glm53_overlay_contract", path)
    if spec is None or spec.loader is None:
        raise PrepareError(f"cannot load overlay contract from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_vllm_lineage(
    source: Path, pins: dict[str, Any], manifest: dict[str, Any]
) -> None:
    overlay = _load_overlay_contract()
    records = overlay.validate_overlay_manifest(manifest)
    vllm = pins["vllm"]
    base = vllm["native_commit"]
    target = vllm["python_commit"]
    if run(("git", "-C", str(source), "merge-base", base, target)) != base:
        raise PrepareError("the retained-native vLLM commit is not the overlay base")
    if run(("git", "-C", str(source), "rev-parse", f"{base}^{{tree}}")) != vllm[
        "native_tree"
    ]:
        raise PrepareError("the retained-native vLLM Git tree differs from its pin")

    observed: dict[str, str] = {}
    diff = run(("git", "-C", str(source), "diff", "--name-status", f"{base}..{target}", "--", "vllm"))
    for line in diff.splitlines():
        status, path = line.split("\t", 1)
        observed[path] = "add" if status == "A" else "replace" if status == "M" else status
    expected = {record["path"]: record["operation"] for record in records}
    if observed != expected:
        raise PrepareError("the vLLM Python delta differs from the 31-file allowlist")

    for path, expected_object in vllm["native_source_objects"].items():
        base_object = run(("git", "-C", str(source), "rev-parse", f"{base}:{path}"))
        target_object = run(("git", "-C", str(source), "rev-parse", f"{target}:{path}"))
        if base_object != expected_object or target_object != expected_object:
            raise PrepareError(f"vLLM native build input changed: {path}")

    for record in records:
        path = record["path"]
        target_bytes = subprocess.run(
            ("git", "-C", str(source), "show", f"{target}:{path}"),
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if hashlib.sha256(target_bytes).hexdigest() != record["target_sha256"]:
            raise PrepareError(f"vLLM target blob mismatch: {path}")
        if len(target_bytes) != record["target_bytes"]:
            raise PrepareError(f"vLLM target byte count mismatch: {path}")
        base_result = subprocess.run(
            ("git", "-C", str(source), "show", f"{base}:{path}"),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        if record["operation"] == "add":
            if base_result.returncode == 0:
                raise PrepareError(f"vLLM added path already exists in base: {path}")
        elif (
            base_result.returncode != 0
            or hashlib.sha256(base_result.stdout).hexdigest() != record["base_sha256"]
        ):
            raise PrepareError(f"vLLM base blob mismatch: {path}")


def verify_vllm_runtime_patches(
    source: Path,
    pins: dict[str, Any],
    patch_root: Path,
) -> None:
    """Verify each exact-input runtime patch and restore the clean source."""

    for record in pins["vllm"].get("runtime_patches", ()):
        patch = patch_root / Path(record["path"]).name
        target = source / record["target"]
        if sha256_file(patch) != record["sha256"]:
            raise PrepareError(f"vLLM runtime patch mismatch: {record['path']}")
        if sha256_file(target) != record["preimage_sha256"]:
            raise PrepareError(
                f"vLLM runtime patch preimage mismatch: {record['target']}"
            )
        run(("git", "-C", str(source), "apply", "--check", str(patch)))
        run(("git", "-C", str(source), "apply", str(patch)))
        try:
            observed = sha256_file(target)
            if observed != record["postimage_sha256"]:
                raise PrepareError(
                    f"vLLM runtime patch postimage mismatch: {record['target']}"
                )
        finally:
            run(("git", "-C", str(source), "apply", "--reverse", str(patch)))
    verify_git_source(
        source,
        commit=pins["vllm"]["python_commit"],
        tree=pins["vllm"]["python_tree"],
    )


def verify_composed_runtime_patches(
    source: Path,
    pins: dict[str, Any],
    patch_root: Path,
    sparkcache: Path,
) -> None:
    """Verify patches whose preimages include the SparkCache vLLM chain."""

    contract = sparkcache / pins["sparkcache"]["contract"]["path"]
    if sha256_file(contract) != pins["sparkcache"]["contract"]["sha256"]:
        raise PrepareError("SparkCache vLLM contract differs from its pin")
    applied_sparkcache: list[Path] = []
    applied_composed: list[Path] = []
    try:
        for record in pins["sparkcache"]["patches"]:
            patch = sparkcache / record["path"]
            run(("git", "-C", str(source), "apply", str(patch)))
            applied_sparkcache.append(patch)

        for record in pins["vllm"].get("composed_runtime_patches", ()):
            patch = patch_root / Path(record["path"]).name
            if sha256_file(patch) != record["sha256"]:
                raise PrepareError(f"vLLM composed patch mismatch: {record['path']}")
            for target_record in record["targets"]:
                target = source / target_record["path"]
                if sha256_file(target) != target_record["preimage_sha256"]:
                    raise PrepareError(
                        "vLLM composed patch preimage mismatch: "
                        f"{target_record['path']}"
                    )
            run(("git", "-C", str(source), "apply", "--check", str(patch)))
            run(("git", "-C", str(source), "apply", str(patch)))
            applied_composed.append(patch)
            for target_record in record["targets"]:
                target = source / target_record["path"]
                if sha256_file(target) != target_record["postimage_sha256"]:
                    raise PrepareError(
                        "vLLM composed patch postimage mismatch: "
                        f"{target_record['path']}"
                    )

            test_record = record.get("test_patch")
            if test_record is None:
                continue
            test_patch = patch_root / Path(test_record["path"]).name
            test_target = source / test_record["target"]
            if sha256_file(test_patch) != test_record["sha256"]:
                raise PrepareError("vLLM recurrent-boundary test patch mismatch")
            if sha256_file(test_target) != test_record["preimage_sha256"]:
                raise PrepareError("vLLM recurrent-boundary test preimage mismatch")
            run(("git", "-C", str(source), "apply", "--check", str(test_patch)))
            run(("git", "-C", str(source), "apply", str(test_patch)))
            try:
                if sha256_file(test_target) != test_record["postimage_sha256"]:
                    raise PrepareError("vLLM recurrent-boundary test postimage mismatch")
            finally:
                run(("git", "-C", str(source), "apply", "--reverse", str(test_patch)))
        verifier = sparkcache / "sparkcache/runtime_patches/verify_lease_contract.py"
        run(
            (
                sys.executable,
                str(verifier),
                "--vllm-root",
                str(source),
                "--contract",
                str(contract),
            )
        )
    finally:
        for patch in reversed(applied_composed):
            run(("git", "-C", str(source), "apply", "--reverse", str(patch)))
        for patch in reversed(applied_sparkcache):
            run(("git", "-C", str(source), "apply", "--reverse", str(patch)))
    verify_git_source(
        source,
        commit=pins["vllm"]["python_commit"],
        tree=pins["vllm"]["python_tree"],
    )


def copy_overlay(source: Path, destination: Path, manifest: dict[str, Any]) -> None:
    target = manifest["target"]["commit"]
    for record in manifest["files"]:
        relative = Path(record["path"])
        output = destination / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = subprocess.run(
            ("git", "-C", str(source), "show", f"{target}:{relative.as_posix()}"),
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        output.write_bytes(payload)


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
        if path.is_symlink():
            raise PrepareError(f"SparkCache source contains a symlink: {relative}")
        encoded = relative.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
        normalized = path.read_bytes().replace(b"\r\n", b"\n")
        digest.update(hashlib.sha256(normalized).digest())
        count += 1
    if not count:
        raise PrepareError("SparkCache source tree is empty")
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def prepare(output: Path, *, repository_root: Path = ROOT) -> dict[str, Any]:
    if output.exists():
        raise PrepareError(f"output already exists: {output}")
    pins = load_json(PINS, PINS_SCHEMA)
    manifest = load_json(MANIFEST, OVERLAY_SCHEMA)
    if sha256_file(MANIFEST) != pins["vllm"]["overlay_manifest_sha256"]:
        raise PrepareError("vLLM overlay manifest SHA-256 differs from pins.json")

    sources = output / "bundle" / "sources"
    runtime = output / "bundle" / "runtime"
    overlay = output / "bundle" / "vllm-overlay"
    sources.mkdir(parents=True)
    runtime.mkdir(parents=True)

    vllm = sources / "vllm"
    clone_detached(
        vllm,
        repository=pins["vllm"]["repository"],
        commit=pins["vllm"]["python_commit"],
        tree=pins["vllm"]["python_tree"],
        fetch_depth=13,
    )
    verify_vllm_lineage(vllm, pins, manifest)
    copy_overlay(vllm, overlay, manifest)
    patch_root = HERE / "patches"
    verify_vllm_runtime_patches(vllm, pins, patch_root)

    b12x = sources / "b12x"
    clone_detached(
        b12x,
        repository=pins["b12x"]["repository"],
        commit=pins["b12x"]["commit"],
        tree=pins["b12x"]["tree"],
        fetch_depth=1,
    )

    sparkcache = sources / "sparkcache"
    clone_detached(
        sparkcache,
        repository=pins["sparkcache"]["repository"],
        commit=pins["sparkcache"]["commit"],
        tree=pins["sparkcache"]["tree"],
        fetch_depth=1,
    )
    observed_sparkcache = sparkcache_source_sha256(sparkcache / "sparkcache")
    if observed_sparkcache != pins["sparkcache"]["source_tree_sha256"]:
        raise PrepareError(
            "SparkCache deployable source mismatch: expected "
            f"{pins['sparkcache']['source_tree_sha256']}, got {observed_sparkcache}"
        )
    for patch in pins["sparkcache"]["patches"]:
        path = sparkcache / patch["path"]
        if sha256_file(path) != patch["sha256"]:
            raise PrepareError(f"SparkCache patch mismatch: {patch['path']}")
    verify_composed_runtime_patches(vllm, pins, patch_root, sparkcache)
    contract = sparkcache / pins["sparkcache"]["contract"]["path"]
    if sha256_file(contract) != pins["sparkcache"]["contract"]["sha256"]:
        raise PrepareError("SparkCache vLLM contract differs from its pin")

    for filename in (
        "pins.json",
        "vllm-python-overlay.json",
        "overlay_contract.py",
        "verify_image.py",
        "Containerfile",
        "build-image.sh",
        "README.md",
    ):
        copy_file(HERE / filename, runtime / filename)
    runtime_patches = []
    patch_records = list(pins["vllm"].get("runtime_patches", ()))
    for patch in pins["vllm"].get("composed_runtime_patches", ()):
        patch_records.append(patch)
        if "test_patch" in patch:
            patch_records.append(patch["test_patch"])
    for patch in patch_records:
        name = Path(patch["path"]).name
        copy_file(patch_root / name, runtime / "patches" / name)
        runtime_patches.append(f"bundle/runtime/patches/{name}")
    copy_file(repository_root / "LICENSE", runtime / "SparkRing-LICENSE")

    receipt_inputs = tuple(
        f"bundle/runtime/{name}"
        for name in (
            "pins.json",
            "vllm-python-overlay.json",
            "overlay_contract.py",
            "verify_image.py",
            "Containerfile",
            "build-image.sh",
            "README.md",
            "SparkRing-LICENSE",
        )
    ) + tuple(runtime_patches)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "sources": {
            "vllm_python": {
                "commit": pins["vllm"]["python_commit"],
                "tree": pins["vllm"]["python_tree"],
                "files": len(manifest["files"]),
            },
            "vllm_native": {
                "commit": pins["vllm"]["native_commit"],
                "tree": pins["vllm"]["native_tree"],
            },
            "b12x": {"commit": pins["b12x"]["commit"], "tree": pins["b12x"]["tree"]},
            "sparkcache": {
                "commit": pins["sparkcache"]["commit"],
                "tree": pins["sparkcache"]["tree"],
                "source_tree_sha256": observed_sparkcache,
            },
        },
        "vllm_runtime_patches": pins["vllm"].get("runtime_patches", []),
        "vllm_composed_runtime_patches": pins["vllm"].get(
            "composed_runtime_patches", []
        ),
        "files": {relative: sha256_file(output / relative) for relative in receipt_inputs},
    }
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_context(output)
    return receipt


def verify_context(context: Path) -> dict[str, Any]:
    pins = load_json(context / "bundle/runtime/pins.json", PINS_SCHEMA)
    manifest = load_json(
        context / "bundle/runtime/vllm-python-overlay.json", OVERLAY_SCHEMA
    )
    receipt = load_json(context / "receipt.json", RECEIPT_SCHEMA)
    for relative, expected in receipt["files"].items():
        if sha256_file(context / relative) != expected:
            raise PrepareError(f"prepared context file mismatch: {relative}")
    verify_git_source(
        context / "bundle/sources/vllm",
        commit=pins["vllm"]["python_commit"],
        tree=pins["vllm"]["python_tree"],
    )
    verify_vllm_lineage(context / "bundle/sources/vllm", pins, manifest)
    if receipt.get("vllm_runtime_patches") != pins["vllm"].get(
        "runtime_patches", []
    ):
        raise PrepareError("prepared context vLLM runtime patch receipt differs")
    if receipt.get("vllm_composed_runtime_patches") != pins["vllm"].get(
        "composed_runtime_patches", []
    ):
        raise PrepareError("prepared context composed vLLM patch receipt differs")
    verify_vllm_runtime_patches(
        context / "bundle/sources/vllm",
        pins,
        context / "bundle/runtime/patches",
    )
    verify_composed_runtime_patches(
        context / "bundle/sources/vllm",
        pins,
        context / "bundle/runtime/patches",
        context / "bundle/sources/sparkcache",
    )
    verify_git_source(
        context / "bundle/sources/b12x",
        commit=pins["b12x"]["commit"],
        tree=pins["b12x"]["tree"],
    )
    verify_git_source(
        context / "bundle/sources/sparkcache",
        commit=pins["sparkcache"]["commit"],
        tree=pins["sparkcache"]["tree"],
    )
    overlay = _load_overlay_contract()
    overlay.verify_overlay_files(
        context / "bundle/vllm-overlay", manifest, stage="target"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = (
            verify_context(args.output.resolve())
            if args.verify
            else prepare(args.output.resolve(), repository_root=args.repo_root.resolve())
        )
    except (OSError, KeyError, json.JSONDecodeError, PrepareError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
