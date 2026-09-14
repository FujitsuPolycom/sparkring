"""Package the source-pinned Qwen collective selector for read-only mounting."""

import argparse
import hashlib
import json
from pathlib import Path


def package(destination):
    destination.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).with_name("qwen38_collective_policy.py")
    files = {
        source.name: source.read_bytes(),
        "qwen38_collective_policy.pth": (
            "/opt/sparkring/qwen38-collectives\n" "import qwen38_collective_policy\n"
        ).encode(),
    }
    for name, data in files.items():
        (destination / name).write_bytes(data)
    manifest = {
        "schema": "sparkring-qwen38-collective-policy/v1",
        "status": "research-only",
        "files": {
            name: hashlib.sha256(data).hexdigest() for name, data in files.items()
        },
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    print(json.dumps(package(parser.parse_args().destination), indent=2))
