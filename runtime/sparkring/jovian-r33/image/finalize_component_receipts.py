#!/usr/bin/env python3
"""Generate or reproduce every terminal source-identity receipt required by image assembly."""
from __future__ import annotations

import argparse
import filecmp
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


RECEIPTS = {
    "post-build-source-identities.json": "artifacts/post-build-source-identities.json",
    "flashinfer-resume-source-receipt.json": "artifacts/flashinfer/resume-source-receipt.json",
    "flashkda-source-identity.json": "artifacts/vllm-native/flashkda-source-identity.json",
}


def producer_commands(build_root: Path, script_root: Path, staging: Path) -> list[list[str]]:
    return [
        [sys.executable, str(script_root / "capture_post_build_sources.py"),
         "--build-root", str(build_root), "--output", str(staging / "post-build-source-identities.json")],
        [sys.executable, str(script_root / "validate_flashinfer_resume.py"),
         "--source", str(build_root / "sources/flashinfer"),
         "--artifacts", str(build_root / "artifacts/flashinfer"),
         "--output", str(staging / "flashinfer-resume-source-receipt.json")],
        [sys.executable, str(script_root / "capture_flashkda_identity.py"),
         "--source", str(build_root / "build/vllm-native/_deps/flashkda-src"),
         "--output", str(staging / "flashkda-source-identity.json")],
    ]


def finalize(build_root: Path, script_root: Path, verify_existing: bool, run=subprocess.run) -> None:
    destinations = {name: build_root / relative for name, relative in RECEIPTS.items()}
    if not verify_existing:
        existing = [str(path) for path in destinations.values() if path.exists()]
        if existing:
            raise RuntimeError(f"receipt targets already exist; use --verify-existing: {existing}")
    with tempfile.TemporaryDirectory(prefix="sparkring-r33-receipts-") as directory:
        staging = Path(directory)
        for command in producer_commands(build_root, script_root, staging):
            run(command, check=True)
        for name, destination in destinations.items():
            produced = staging / name
            if not produced.is_file():
                raise RuntimeError(f"receipt producer omitted {name}")
            if verify_existing:
                if not destination.is_file() or not filecmp.cmp(produced, destination, shallow=False):
                    raise RuntimeError(f"existing receipt is not reproducible: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(produced, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-root", type=Path, default=Path("/var/tmp/sparkring-r33-20260910"))
    parser.add_argument("--script-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--verify-existing", action="store_true")
    args = parser.parse_args()
    finalize(args.build_root.resolve(), args.script_root.resolve(), args.verify_existing)
    print("R33 terminal identity receipts reproduced" if args.verify_existing else "R33 terminal identity receipts created")


if __name__ == "__main__":
    main()
