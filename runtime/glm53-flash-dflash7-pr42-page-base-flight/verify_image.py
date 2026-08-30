#!/usr/bin/env python3
"""Verify the isolated DFlash7 SparkCache PR42 image and receipt."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
COMMON = HERE.parent / "glm53-flash-adaptive-mtp-python-overlay"
PINS = HERE / "pins.json"
DEPLOYMENT = "glm53-flash-dflash7-python-overlay"


def _module():
    path = COMMON / "verify_image.py"
    spec = importlib.util.spec_from_file_location("glm53_pr42_shared_verify", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shared image verifier: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


shared = _module()
shared.PINS = PINS
shared.RECEIPT_SCHEMA = "sparkring-glm53-dflash7-pr42-page-base-flight-image/v1"
_shared_expected_output_labels = shared.expected_output_labels
_shared_verify_image = shared.verify_image


def expected_output_labels(pins: dict[str, Any]) -> dict[str, str]:
    labels = _shared_expected_output_labels(pins)
    labels.update(
        {
            "org.sparkcache.deployment-profile": DEPLOYMENT,
            "org.sparkcache.cuda-config-schema": "canonical-v1",
            "org.sparkcache.feature.page-base-read-flight": (
                "implemented-gpu-free-tested"
            ),
            "org.sparkcache.feature.page-base-read-flight-pr": "42",
            "org.sparkcache.parent-image-id": (
                "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8"
            ),
            "org.sparkcache.cache-namespace-impact": "none",
        }
    )
    return labels


def verify_image(engine: str, image: str, pins_path: Path = PINS) -> dict[str, Any]:
    pins = shared.load_pins(pins_path)
    result = _shared_verify_image(engine, image, pins_path)
    expected_cuda = pins["sparkcache"]["cuda_placement_library_sha256"]
    observed_cuda = result["artifacts"]["sparkcache_cuda_placement_sha256"]
    if observed_cuda != expected_cuda:
        raise shared.VerifyError(
            "SparkCache CUDA placement library differs from the PR42 pin"
        )
    result["page_base_restore_flight_contract"] = pins[
        "page_base_restore_flight"
    ]
    return result


shared.expected_output_labels = expected_output_labels
shared.verify_image = verify_image


def main() -> int:
    return shared.main()


if __name__ == "__main__":
    raise SystemExit(main())
