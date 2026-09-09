"""Prepare verified transport assets for a shared image without building it."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from sparkring_transport_selector import PROFILE, ROOT, startup_hook, verify_bundle


def prepare(destination: Path) -> dict:
    digest = hashlib.sha256((ROOT / PROFILE / "manifest.json").read_bytes()).hexdigest()
    verify_bundle(PROFILE, digest)
    destination.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / PROFILE, destination / PROFILE,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for name in ("sparkring_transport_selector.py", "entrypoint.py"):
        shutil.copyfile(ROOT / name, destination / name)
    (destination / "sparkring_transport.pth").write_text(
        startup_hook(Path("/opt/sparkring/transports")), encoding="utf-8", newline="\n"
    )
    receipt = {
        "schema": "sparkring-transport-install-plan/v1",
        "status": "implemented",
        "image_destination": "/opt/sparkring/transports",
        "startup_hook_destination": "serving Python site-packages/sparkring_transport.pth",
        "profile": PROFILE,
        "manifest_sha256": digest,
        "builds_image": False,
        "starts_serving": False,
        "hardware_qualified_shared_image": False,
    }
    (destination / "install-plan.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination), indent=2))


if __name__ == "__main__":
    main()
