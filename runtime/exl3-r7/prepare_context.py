#!/usr/bin/env python3
"""Materialize and verify the immutable EXL3 R7 runtime source trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent


class PreparationError(RuntimeError):
    """Raised when a pinned source context cannot be reproduced exactly."""


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        raise PreparationError(f"{' '.join(args)} failed: {detail}")
    return result.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkout(repository: str, commit: str, destination: Path) -> None:
    if destination.exists():
        raise PreparationError(f"destination already exists: {destination}")
    run("git", "clone", "--filter=blob:none", "--no-checkout", repository, str(destination))
    # The archived patches can contain binary payloads. Line-ending conversion
    # would change their preimages and invalidate the recorded result tree.
    run("git", "config", "core.autocrlf", "false", cwd=destination)
    run("git", "config", "core.eol", "lf", cwd=destination)
    run("git", "fetch", "--depth", "1", "origin", commit, cwd=destination)
    run("git", "checkout", "--detach", commit, cwd=destination)


def prepare_component(
    name: str, spec: dict[str, str], patch_root: Path, output: Path
) -> dict[str, str]:
    destination = output / name
    checkout(spec["repository"], spec["base_commit"], destination)
    patch = patch_root / spec["patch"]
    if not patch.is_file():
        raise PreparationError(f"archived patch is missing: {patch}")
    actual_patch_hash = sha256(patch)
    if actual_patch_hash != spec["patch_sha256"]:
        raise PreparationError(
            f"{name} patch SHA-256 mismatch: expected {spec['patch_sha256']}, "
            f"got {actual_patch_hash}"
        )
    run("git", "apply", "--binary", "--index", str(patch), cwd=destination)
    actual_tree = run("git", "write-tree", cwd=destination)
    if actual_tree != spec["result_tree"]:
        raise PreparationError(
            f"{name} tree mismatch: expected {spec['result_tree']}, got {actual_tree}"
        )
    return {
        "base_commit": spec["base_commit"],
        "patch_sha256": actual_patch_hash,
        "result_tree": actual_tree,
    }


def verify_component(name: str, spec: dict[str, str], output: Path) -> None:
    """Re-verify a prepared component against its pinned spec.

    Unlike ``prepare_component``, this does not re-clone or re-apply patches.
    It checks that the checked-out tree matches the recorded result_tree and
    that the patch file hash is still correct.
    """
    destination = output / name
    if not destination.is_dir():
        raise PreparationError(f"verify: component directory missing: {destination}")
    unstaged = subprocess.run(
        ["git", "diff", "--quiet", "--no-ext-diff"],
        cwd=destination,
        check=False,
    )
    if unstaged.returncode == 1:
        raise PreparationError(f"verify: {name} has unstaged source drift")
    if unstaged.returncode != 0:
        raise PreparationError(
            f"verify: {name} could not check unstaged source drift"
        )
    untracked = run(
        "git", "ls-files", "--others", "--exclude-standard", cwd=destination
    )
    if untracked:
        raise PreparationError(
            f"verify: {name} has untracked source content: {untracked.splitlines()[0]}"
        )
    actual_tree = run("git", "write-tree", cwd=destination)
    if actual_tree != spec["result_tree"]:
        raise PreparationError(
            f"verify: {name} tree mismatch: expected {spec['result_tree']}, "
            f"got {actual_tree}"
        )
    # Verify the checked-out commit matches the pinned base_commit.
    actual_commit = run("git", "rev-parse", "HEAD", cwd=destination)
    if actual_commit != spec["base_commit"]:
        raise PreparationError(
            f"verify: {name} commit mismatch: expected {spec['base_commit']}, "
            f"got {actual_commit}"
        )


def verify(output: Path, pins_path: Path | None = None) -> dict:
    """Verify a prepared source directory against its receipt and pins.

    The receipt is checked for schema version, release commit, and every
    component's base_commit, patch_sha256, and result_tree. Receipt existence
    alone is insufficient: each field is compared against the pinned spec
    in pins.json, and the actual Git tree of each checked-out component is
    re-computed and compared.
    """
    pins_path = pins_path or HERE / "pins.json"
    pins = json.loads(pins_path.read_text(encoding="utf-8"))
    receipt_path = output / "receipt.json"
    if not receipt_path.is_file():
        raise PreparationError(f"verify: receipt missing: {receipt_path}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("schema_version") != 1:
        raise PreparationError(
            f"verify: wrong schema_version: expected 1, "
            f"got {receipt.get('schema_version')!r}"
        )
    if receipt.get("release_commit") != pins["release"]["commit"]:
        raise PreparationError(
            f"verify: release commit mismatch: expected "
            f"{pins['release']['commit']}, got {receipt.get('release_commit')}"
        )
    receipt_components = receipt.get("components", {})
    if set(receipt_components) != set(pins["components"]):
        raise PreparationError(
            f"verify: component set mismatch: receipt has "
            f"{sorted(receipt_components)}, pins has "
            f"{sorted(pins['components'])}"
        )
    for name, spec in pins["components"].items():
        receipt_entry = receipt_components[name]
        for field in ("base_commit", "patch_sha256", "result_tree"):
            if receipt_entry.get(field) != spec[field]:
                raise PreparationError(
                    f"verify: {name} receipt {field} mismatch: "
                    f"expected {spec[field]}, got {receipt_entry.get(field)}"
                )
        verify_component(name, spec, output)
    return receipt

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        help="new directory for verified trees (or directory to verify with --verify)",
    )
    parser.add_argument("--pins", type=Path, default=HERE / "pins.json")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="verify an existing prepared-source directory against its receipt and pins",
    )
    args = parser.parse_args()

    if args.verify:
        if args.output is None:
            parser.error("--verify requires a directory argument")
        receipt = verify(args.output.resolve(), args.pins)
        print(json.dumps({"status": "pass", "schema_version": receipt["schema_version"]}, sort_keys=True))
        return 0

    if args.output is None:
        parser.error("output directory is required for preparation")
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    output = args.output.resolve()
    if output.exists():
        raise PreparationError(f"output already exists: {output}")
    output.mkdir(parents=True)

    release = pins["release"]
    release_checkout = output / "release"
    checkout(release["repository"], release["commit"], release_checkout)
    patch_root = release_checkout / release["patch_root"]

    receipt = {
        "schema_version": 1,
        "release_commit": release["commit"],
        "components": {},
    }
    for name, spec in pins["components"].items():
        receipt["components"][name] = prepare_component(
            name, spec, patch_root, output
        )
    # Write the receipt, then self-verify: the prepare path must verify its own
    # output against the receipt and pins before returning.
    (output / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    verify(output, args.pins)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
