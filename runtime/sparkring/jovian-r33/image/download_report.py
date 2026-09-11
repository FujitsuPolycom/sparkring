#!/usr/bin/env python3
"""Download exact remote wheels from a pip installation report."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    downloads = {}
    for item in report["install"]:
        info = item["download_info"]
        url = info["url"]
        parsed = urlparse(url)
        expected = info.get("archive_info", {}).get("hashes", {}).get("sha256")
        if not expected:
            raise RuntimeError(f"pip report omitted SHA-256: {url}")
        filename = Path(unquote(parsed.path)).name
        if not filename.endswith(".whl"):
            raise RuntimeError(f"closure contains a non-wheel artifact: {url}")
        destination = args.wheelhouse / filename
        if parsed.scheme == "file":
            if not destination.is_file():
                raise RuntimeError(f"local report wheel is absent: {destination}")
            data_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
        else:
            with urlopen(url) as response:
                data = response.read()
            data_hash = hashlib.sha256(data).hexdigest()
            if data_hash == expected:
                destination.write_bytes(data)
        if data_hash != expected:
            raise RuntimeError(f"download hash mismatch: {filename}: {data_hash} != {expected}")
        downloads[item["metadata"]["name"]] = {
            "version": item["metadata"]["version"],
            "filename": filename,
            "sha256": expected,
            "url": url,
        }
    args.output.write_text(json.dumps({"schema": "sparkring-r33-python-closure/v1", "downloads": downloads}, indent=2, sort_keys=True) + "\n")
    inherited: set[str] = set()
    install_wheels = sorted(
        item["filename"]
        for name, item in downloads.items()
        if name.lower().replace("_", "-") not in inherited
    )
    (args.output.parent / "closure-install-wheels.txt").write_text("\n".join(install_wheels) + "\n")
    print(json.dumps({"downloaded_or_verified": len(downloads), "closure_install_wheels": len(install_wheels)}, indent=2))


if __name__ == "__main__":
    main()
