#!/usr/bin/env python3
"""Verify a GLM-5.3 DFlash7 Python-overlay image and its exact labels."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "glm53-flash-adaptive-mtp-python-overlay"


def _module():
    path = COMMON / "verify_image.py"
    spec = importlib.util.spec_from_file_location("glm53_shared_overlay_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared image verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _module()
shared.PINS = HERE / "pins.json"
shared.RECEIPT_SCHEMA = "sparkring-glm53-dflash7-python-overlay-image/v1"
_shared_expected_output_labels = shared.expected_output_labels


def expected_output_labels(pins: dict[str, Any]) -> dict[str, str]:
    labels = _shared_expected_output_labels(pins)
    labels["org.sparkcache.deployment-profile"] = (
        "glm53-flash-dflash7-python-overlay"
    )
    return labels


shared.expected_output_labels = expected_output_labels


def main() -> int:
    return shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
