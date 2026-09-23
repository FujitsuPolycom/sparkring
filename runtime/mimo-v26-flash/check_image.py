"""Reject images without the profile's pinned MiMo attention sources."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


def validate(metadata: dict, contract: dict) -> None:
    if metadata.get("Architecture") != "arm64":
        raise ValueError("MiMo requires a linux/arm64 image")
    labels = metadata.get("Config", {}).get("Labels") or {}
    for package, source in contract["sources"].items():
        key = f"org.sparkring.candidate.{package}-commit"
        if labels.get(key) != source["revision"]:
            raise ValueError(
                f"Image {package} source does not match {source['revision']}; "
                "build runtime/mimo-v26-flash/build-image.sh first"
            )


def main() -> int:
    contract = json.loads(Path(__file__).with_name("b12x-image.json").read_text())
    try:
        metadata = json.loads(subprocess.check_output(
            ["docker", "image", "inspect", sys.argv[1]], text=True
        ))[0]
        validate(metadata, contract)
    except (ValueError, subprocess.CalledProcessError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
