#!/usr/bin/env python3
"""Prepare the isolated DFlash7 SparkCache PR42 image context."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "runtime" / "glm53-flash-dflash7-python-overlay"
PINS = HERE / "pins.json"
DEPLOYMENT = "glm53-flash-dflash7-python-overlay"


def _module():
    path = BASE / "prepare_context.py"
    spec = importlib.util.spec_from_file_location("glm53_dflash7_base_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load DFlash7 context preparer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _module()
base.HERE = HERE
base.ROOT = ROOT
base.PINS = PINS
_base_render_containerfile = base._render_containerfile
_base_render_verify_image = base._render_verify_image


def _feature() -> dict[str, object]:
    return json.loads(PINS.read_text(encoding="utf-8"))["page_base_restore_flight"]


def _render_containerfile() -> str:
    text = _base_render_containerfile()
    replacements = {
        "/cache/jit/vllm/dflash7-py-0b67266-native-da4d7be": (
            "/cache/jit/vllm/dflash7-pr42-page-base-flight"
        ),
        "/cache/jit/b12x/b1d541f9/dflash7-cute": (
            "/cache/jit/b12x/b1d541f9/dflash7-pr42-page-base-flight"
        ),
        "/cache/jit/triton/dflash7-vllm-py-0b67266-native-da4d7be": (
            "/cache/jit/triton/dflash7-pr42-page-base-flight"
        ),
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"base DFlash7 Containerfile omits required text: {old}")
        text = text.replace(old, new)
    marker = f'      org.sparkcache.deployment-profile="{DEPLOYMENT}" \\\n'
    if text.count(marker) != 1:
        raise RuntimeError("PR42 Containerfile omits the deployment label")
    labels = (
        marker
        + '      org.sparkcache.feature.page-base-read-flight="implemented-gpu-free-tested" \\\n'
        + '      org.sparkcache.feature.page-base-read-flight-pr="42" \\\n'
        + '      org.sparkcache.parent-image-id="sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8" \\\n'
        + '      org.sparkcache.cache-namespace-impact="none" \\\n'
    )
    return text.replace(marker, labels)


def _render_verify_image() -> str:
    text = _base_render_verify_image()
    text = text.replace(
        "sparkring-glm53-dflash7-python-overlay-image/v1",
        "sparkring-glm53-dflash7-pr42-page-base-flight-image/v1",
    )
    text = text.replace("glm53-flash-dflash7-python-overlay", DEPLOYMENT)
    marker = f'        "org.sparkcache.deployment-profile": "{DEPLOYMENT}",\n'
    if text.count(marker) != 1:
        raise RuntimeError("PR42 verifier omits the deployment label")
    expected = (
        marker
        + '        "org.sparkcache.feature.page-base-read-flight": '
        + '"implemented-gpu-free-tested",\n'
        + '        "org.sparkcache.feature.page-base-read-flight-pr": "42",\n'
        + '        "org.sparkcache.parent-image-id": '
        + '"sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8",\n'
        + '        "org.sparkcache.cache-namespace-impact": "none",\n'
    )
    text = text.replace(marker, expected)
    receipt_marker = '        "artifacts": artifacts,\n'
    if text.count(receipt_marker) != 1:
        raise RuntimeError("PR42 verifier omits the receipt artifact marker")
    return text.replace(
        receipt_marker,
        receipt_marker
        + '        "page_base_restore_flight_contract": '
        + 'pins["page_base_restore_flight"],\n',
    )


base._render_containerfile = _render_containerfile
base._render_verify_image = _render_verify_image


def main() -> int:
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
