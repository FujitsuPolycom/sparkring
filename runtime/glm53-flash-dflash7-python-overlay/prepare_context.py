#!/usr/bin/env python3
"""Prepare the exact shared Python overlay with DFlash7 image metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COMMON = ROOT / "runtime" / "glm53-flash-adaptive-mtp-python-overlay"
PINS = HERE / "pins.json"
RECEIPT_SCHEMA = "sparkring-glm53-public-python-overlay-context/v1"
DEEP_EP_RECEIPT = "deep-ep-removal-receipt.json"


class PrepareError(RuntimeError):
    """The DFlash7 prepared context differs from its exact source contract."""


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise PrepareError(f"cannot load shared overlay module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = _module("glm53_shared_python_overlay_prepare", COMMON / "prepare_context.py")


def _longpath_run(argv, *, cwd=None):
    """Enable long paths in prepared Git repositories on Windows hosts."""

    arguments = tuple(argv)
    result = _original_run(arguments, cwd=cwd)
    if len(arguments) >= 4 and arguments[:3] == ("git", "init", "--quiet"):
        _original_run(
            ("git", "-C", arguments[3], "config", "core.longpaths", "true")
        )
    return result


_original_run = common.run


def _render_containerfile() -> str:
    text = (COMMON / "Containerfile").read_text(encoding="utf-8")
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    cleanup = pins["runtime_cleanup"]["deep_ep"]
    replacements = {
        "/cache/jit/vllm/py-0b67266-native-da4d7be": (
            "/cache/jit/vllm/dflash7-py-0b67266-native-da4d7be"
        ),
        "/cache/jit/b12x/b1d541f9/cute": (
            "/cache/jit/b12x/b1d541f9/dflash7-cute"
        ),
        "/cache/jit/triton/vllm-py-0b67266-native-da4d7be": (
            "/cache/jit/triton/dflash7-vllm-py-0b67266-native-da4d7be"
        ),
        "SparkRing GLM-5.3 adaptive-MTP Python overlay": (
            "SparkRing GLM-5.3 DFlash7 Python overlay"
        ),
        (
            "vLLM 0b67266 Python over da4d7be native extensions with B12X "
            "b1d541f and SparkCache"
        ): (
            "GLM-5.3 DFlash7 with vLLM 0b67266 Python over da4d7be native "
            "extensions, B12X b1d541f, and SparkCache"
        ),
        "glm53-flash-adaptive-mtp-python-overlay": (
            "glm53-flash-dflash7-python-overlay"
        ),
    }
    for old, new in replacements.items():
        if old not in text:
            raise PrepareError(f"shared Containerfile omits required text: {old}")
        text = text.replace(old, new)
    argument_marker = "ARG SPARKCACHE_CUDA_PLACEMENT_SHA256\n"
    if text.count(argument_marker) != 1:
        raise PrepareError("shared Containerfile omits the CUDA placement argument")
    text = text.replace(
        argument_marker,
        argument_marker + "ARG DEEP_EP_REMOVAL_RECEIPT_SHA256\n",
    )
    receipt_marker = (
        "COPY receipt.json ${PYTHON_OVERLAY_ROOT}/source-receipt.json\n"
    )
    if text.count(receipt_marker) != 1:
        raise PrepareError("shared Containerfile omits the source receipt copy")
    cleanup_block = f"""{receipt_marker}COPY bundle/runtime/remove_distribution.py ${{PYTHON_OVERLAY_ROOT}}/remove_distribution.py
COPY bundle/runtime/{DEEP_EP_RECEIPT} ${{PYTHON_OVERLAY_ROOT}}/{DEEP_EP_RECEIPT}
RUN test \"$(sha256sum \"${{PYTHON_OVERLAY_ROOT}}/{DEEP_EP_RECEIPT}\" | cut -d' ' -f1)\" = \"${{DEEP_EP_REMOVAL_RECEIPT_SHA256}}\" \\
 && python3 \"${{PYTHON_OVERLAY_ROOT}}/remove_distribution.py\" \\
      --receipt \"${{PYTHON_OVERLAY_ROOT}}/{DEEP_EP_RECEIPT}\" \\
 && python3 -c 'import importlib.util; assert importlib.util.find_spec("deep_ep") is None'
"""
    text = text.replace(receipt_marker, cleanup_block)
    label_marker = (
        "      org.sparkcache.deployment-profile="
        '"glm53-flash-dflash7-python-overlay" \\\n'
    )
    if text.count(label_marker) != 1:
        raise PrepareError("rendered Containerfile omits the DFlash7 profile label")
    cleanup_labels = (
        label_marker
        + "      org.sparkring.runtime.removed-deep-ep-distribution="
        + f'"{cleanup["distribution"]}=={cleanup["version"]}" \\\n'
        + "      org.sparkring.runtime.deep-ep-removal-receipt-sha256="
        + '"${DEEP_EP_REMOVAL_RECEIPT_SHA256}" \\\n'
    )
    text = text.replace(label_marker, cleanup_labels)
    return text


def _render_verify_image() -> str:
    text = (COMMON / "verify_image.py").read_text(encoding="utf-8")
    replacements = {
        "sparkring-glm53-public-python-overlay-image/v1": (
            "sparkring-glm53-dflash7-python-overlay-image/v1"
        ),
        "glm53-flash-adaptive-mtp-python-overlay": (
            "glm53-flash-dflash7-python-overlay"
        ),
    }
    for old, new in replacements.items():
        if old not in text:
            raise PrepareError(f"shared image verifier omits required text: {old}")
        text = text.replace(old, new)
    return text


def _deep_ep_removal_receipt(pins: dict[str, Any]) -> dict[str, str]:
    cleanup = pins["runtime_cleanup"]["deep_ep"]
    return {
        "schema": "sparkring-python-distribution-removal/v1",
        "status": "implemented",
        "module": cleanup["module"],
        "distribution": cleanup["distribution"],
        "version": cleanup["version"],
        "postcondition": "module-absent",
        "reason": cleanup["reason"],
    }


def _replace_runtime_files(context: Path) -> None:
    runtime = context / "bundle" / "runtime"
    pins = json.loads(PINS.read_text(encoding="utf-8"))
    shutil.copy2(PINS, runtime / "pins.json")
    shutil.copy2(HERE / "build-image.sh", runtime / "build-image.sh")
    shutil.copy2(HERE / "README.md", runtime / "README.md")
    shutil.copy2(HERE / "remove_distribution.py", runtime / "remove_distribution.py")
    removal_receipt = runtime / DEEP_EP_RECEIPT
    removal_receipt.write_text(
        json.dumps(_deep_ep_removal_receipt(pins), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    if common.sha256_file(removal_receipt) != pins["runtime_cleanup"]["deep_ep"][
        "receipt_sha256"
    ]:
        raise PrepareError("DeepEP removal receipt differs from its pin")
    (runtime / "Containerfile").write_text(
        _render_containerfile(), encoding="utf-8", newline="\n"
    )
    (runtime / "verify_image.py").write_text(
        _render_verify_image(), encoding="utf-8", newline="\n"
    )
    receipt_path = context / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["workload"] = json.loads(PINS.read_text(encoding="utf-8"))["workload"]
    for name in (
        "pins.json",
        "verify_image.py",
        "Containerfile",
        "build-image.sh",
        "README.md",
        "remove_distribution.py",
        DEEP_EP_RECEIPT,
    ):
        relative = f"bundle/runtime/{name}"
        receipt["files"][relative] = common.sha256_file(context / relative)
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def prepare(output: Path, *, repository_root: Path = ROOT) -> dict[str, Any]:
    original_pins = common.PINS
    original_run = common.run
    try:
        common.PINS = PINS
        common.run = _longpath_run
        common.prepare(output, repository_root=repository_root)
        _replace_runtime_files(output)
        common.verify_context(output)
        return json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    finally:
        common.PINS = original_pins
        common.run = original_run


def verify_context(context: Path) -> dict[str, Any]:
    original_pins = common.PINS
    try:
        common.PINS = PINS
        return common.verify_context(context)
    finally:
        common.PINS = original_pins


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    try:
        result = (
            verify_context(args.output.resolve())
            if args.verify
            else prepare(args.output.resolve(), repository_root=args.repo_root.resolve())
        )
    except (OSError, KeyError, json.JSONDecodeError, PrepareError, common.PrepareError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
