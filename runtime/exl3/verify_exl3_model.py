#!/usr/bin/env python3
"""Verify every pinned EXL3 model-manifest entry inside the derived image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from model_manifest import stderr_progress, verify_model


DEFAULT_PINS = Path("/opt/sparkring-exl3/pins.json")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--pins", type=Path, default=DEFAULT_PINS)
    args = parser.parse_args()
    pins = json.loads(args.pins.read_text(encoding="utf-8"))
    if pins.get("schema") != "sparkring-public-exl3-pins/v1":
        raise RuntimeError("wrong public EXL3 pins schema")
    report = verify_model(
        args.model_path.resolve(),
        pins["model"],
        progress=stderr_progress,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
