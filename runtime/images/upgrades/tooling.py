"""Prepare checksum-pinned CPU-test and native-build tools outside serving images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.request

WHEELS = (
    "tblib==3.1.0",
    "pytest-mock==3.15.1",
    "setuptools-rust==1.12.0",
    "semantic_version==2.10.0",
    "build==1.3.0",
    "pyproject-hooks==1.2.0",
)
RUST = "rust-1.95.0-aarch64-unknown-linux-gnu.tar.xz"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def prepare(output, *, bytes_per_second=64 * 1024**2):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs = output / "inputs"
    wheels = inputs / "wheels"
    wheels.mkdir(parents=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "--only-binary=:all:",
            "--no-deps",
            "--dest",
            str(wheels),
            *WHEELS,
        ],
        check=True,
    )
    url = "https://static.rust-lang.org/dist/" + RUST
    with urllib.request.urlopen(url + ".sha256", timeout=30) as stream:
        digest = stream.read(1024).decode().split()[0]
    if not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Rust publisher checksum is malformed")
    path = inputs / RUST
    count = 0
    started = time.monotonic()
    with urllib.request.urlopen(url, timeout=60) as response, path.open("xb") as stream:
        while block := response.read(1024**2):
            count += len(block)
            if count > 1024**3:
                raise ValueError("Rust tooling archive exceeds 1 GiB")
            stream.write(block)
            delay = count / bytes_per_second - (time.monotonic() - started)
            if delay > 0:
                time.sleep(delay)
    if sha(path) != digest:
        raise ValueError("Rust archive differs from its publisher checksum")
    manifest = {
        "schema": "sparkring-build-tools/v1",
        "python_requirements": list(WHEELS),
        "rust": {"url": url, "sha256": digest},
        "files": {
            p.relative_to(inputs).as_posix(): sha(p)
            for p in sorted(inputs.rglob("*"))
            if p.is_file()
        },
    }
    (inputs / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    shutil.copyfile(__file__, output / "tooling.py")
    shutil.copyfile(
        Path(__file__).with_name("Dockerfile.tooling"), output / "Dockerfile"
    )
    verify(inputs)
    return {
        "context": str(output),
        "manifest_sha256": sha(inputs / "manifest.json"),
        "bytes": sum(p.stat().st_size for p in inputs.rglob("*") if p.is_file()),
    }


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema") != "sparkring-build-tools/v1" or not manifest.get(
        "files"
    ):
        raise ValueError("Missing build-tools manifest")
    for name, expected in manifest["files"].items():
        path = root / name
        if (
            not path.resolve().is_relative_to(root.resolve())
            or path.is_symlink()
            or sha(path) != expected
        ):
            raise ValueError("Build-tool input differs: " + name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.root)), flush=True)
    else:
        verify(args.root)
