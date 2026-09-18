"""Stage the explicitly versioned prepared adaptive RoCE transport bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

ROOT = Path(__file__).resolve().parent
PROFILE = "tp2-rocenante-adaptive-prepared"


def stage(destination: Path) -> dict:
    """Copy only the manifest-bound runtime payload into an empty destination."""
    manifest_bytes = (ROOT / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["name"] != PROFILE:
        raise ValueError("prepared transport manifest has a different identity")
    for name, expected in manifest["files"].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
            raise ValueError("invalid prepared transport source path")
        source = ROOT.joinpath(*relative.parts)
        if source.is_symlink() or any(parent.is_symlink() for parent in source.parents[:len(relative.parts)]):
            raise ValueError("prepared transport sources must not be symbolic links")
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError("prepared transport source differs: " + name)
    destination.mkdir(parents=True, exist_ok=False)
    for name in manifest["files"]:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    shutil.copyfile(ROOT / "manifest.json", destination / "manifest.json")
    return {
        "profile": PROFILE,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "file_count": len(manifest["files"]),
        "destination": str(destination.resolve()),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(stage(parser.parse_args().destination), indent=2))
