#!/usr/bin/env python3
"""Pull the container images a lock pins, by digest, and verify what arrived.

The locks name their images by SHA-256 manifest digest rather than by tag, so
a pull resolves to one immutable set of bytes and any substitution is visible.
This retrieves those images from wherever their publisher serves them; it
copies nothing into this repository and republishes nothing, so it carries no
redistribution obligation for images this project does not own.

Safety class: MUTATES HOST. Pulling writes into the local container store.
`--plan` prints what would be pulled and contacts nothing.

An image absent from its registry is the failure this exists to surface early.
A digest states which bytes are correct; it does not oblige a publisher to
keep serving them. `docs/RUNTIME_INPUT_DURABILITY.md` states what follows from
that.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")

# Every lock that names a container image, and the paths within it that do.
LOCK_IMAGE_PATHS = {
    "runtime/runtime-lock.json": (
        ("base_image", "builder"),
        ("base_image", "runtime"),
    ),
    "runtime/faststart-lock.json": (("base_image",), ("serving_image",)),
}


@dataclass(frozen=True)
class PinnedImage:
    """One image named by repository and manifest digest."""

    source: str
    subject: str
    repository: str
    digest: str

    @property
    def reference(self) -> str:
        return f"{self.repository}@{self.digest}"


def _node(document: object, keys: tuple[str, ...]) -> object:
    node = document
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def collect_pinned_images(root: Path = ROOT) -> list[PinnedImage]:
    """Return every digest-pinned image the tracked locks name."""

    images: list[PinnedImage] = []
    for relative, paths in LOCK_IMAGE_PATHS.items():
        lock_path = root / relative
        if not lock_path.is_file():
            continue
        document = json.loads(lock_path.read_text(encoding="utf-8"))
        for keys in paths:
            node = _node(document, keys)
            if not isinstance(node, dict):
                continue
            repository = str(node.get("repository", "")).strip()
            digest = str(
                node.get("manifest_digest") or node.get("digest") or ""
            ).strip()
            if not repository or not DIGEST.match(digest):
                continue
            images.append(
                PinnedImage(
                    source=relative,
                    subject=".".join(keys) or "base_image",
                    repository=repository,
                    digest=digest,
                )
            )
    return images


def _engine() -> str | None:
    for candidate in ("docker", "podman"):
        if shutil.which(candidate):
            return candidate
    return None


def pull(image: PinnedImage, engine: str, timeout: int = 3600) -> tuple[bool, str]:
    """Pull one image by digest and report what the engine said."""

    result = subprocess.run(
        [engine, "pull", image.reference],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip().splitlines()
        return False, detail[-1] if detail else "pull failed"
    return True, "pulled"


def verify_local(image: PinnedImage, engine: str) -> tuple[bool, str]:
    """Confirm the local store holds the digest the lock names."""

    result = subprocess.run(
        [engine, "image", "inspect", image.reference, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        return False, "absent from the local image store after pull"
    return True, result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pull_pinned_images",
        description=(
            "Pull every container image the tracked locks pin by digest and "
            "verify the local store holds that digest."
        ),
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print what would be pulled and contact nothing",
    )
    arguments = parser.parse_args(argv)

    images = collect_pinned_images()
    if not images:
        print("no digest-pinned images found in the tracked locks")
        return 1

    if arguments.plan:
        print(f"plan: {len(images)} digest-pinned image(s)")
        for image in images:
            print(f"  {image.source} {image.subject}: {image.reference}")
        print("no registry was contacted")
        return 0

    engine = _engine()
    if engine is None:
        print("FAIL no container engine found; install docker or podman")
        return 1

    failures = 0
    for image in images:
        ok, detail = pull(image, engine)
        if ok:
            ok, detail = verify_local(image, engine)
        status = "OK  " if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"{status} {image.subject} {image.reference}: {detail}")

    if failures:
        print(f"{failures} pinned image(s) could not be retrieved")
        return 1
    print(f"all {len(images)} pinned image(s) retrieved and verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
