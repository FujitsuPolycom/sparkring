#!/usr/bin/env python3
"""Fail-closed verifier for the public SparkRing EXL3 derived image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path


PREFIX = Path("/opt/sparkring-exl3")
SITE_PACKAGES = Path("/opt/venv/lib/python3.12/site-packages")
SPARK_RUNTIME_ROOT = Path("/opt/spark-vllm")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def verify_sources(pins: dict) -> None:
    for relative, expected in pins["overlay_files"].items():
        path = SITE_PACKAGES / relative
        require(path.is_file(), f"missing composed vLLM file: {path}")
        observed = sha256(path)
        require(
            observed == expected,
            f"composed vLLM hash mismatch for {relative}: "
            f"expected {expected}, got {observed}",
        )
    for relative, record in pins["profile_patches"].items():
        path = SITE_PACKAGES / relative
        require(path.is_file(), f"missing profile-patched vLLM file: {path}")
        observed = sha256(path)
        expected = record["postimage"]
        require(
            observed == expected,
            f"profile-patched vLLM hash mismatch for {relative}: "
            f"expected {expected}, got {observed}",
        )
    for relative, record in pins["spark_runtime_overlay_files"].items():
        path = SPARK_RUNTIME_ROOT / relative
        require(path.is_file(), f"missing Spark runtime overlay file: {path}")
        observed = sha256(path)
        expected = record["postimage"]
        require(
            observed == expected,
            f"Spark runtime overlay hash mismatch for {relative}: "
            f"expected {expected}, got {observed}",
        )


def verify_imports() -> None:
    require(
        importlib.metadata.version("nvidia-cutlass-dsl") == "4.6.0",
        "CUTLASS DSL is not exactly 4.6.0",
    )
    importlib.import_module("sparkinfer.moe.fused_moe")
    mixed = importlib.import_module(
        "sparkinfer.moe._shared.kernels.w4a16.mixed_trellis"
    )
    require(
        hasattr(mixed, "compile_mixed_trellis"),
        "SparkInfer mixed Trellis compiler is unavailable",
    )
    extension = importlib.import_module("exllamav3_ext")
    require(
        hasattr(extension, "exl3_gemm"),
        "exllamav3_ext does not expose exl3_gemm",
    )
    exl3 = importlib.import_module(
        "vllm.model_executor.layers.quantization.exl3"
    )
    require(hasattr(exl3, "Exl3Config"), "vLLM EXL3 config is unavailable")


def verify_gpu() -> None:
    torch = importlib.import_module("torch")
    require(torch.cuda.is_available(), "CUDA is unavailable")
    capability = tuple(torch.cuda.get_device_capability())
    require(capability == (12, 1), f"expected GB10 SM121, got {capability}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("build", "runtime", "gpu"), required=True)
    args = parser.parse_args()
    pins = json.loads((PREFIX / "pins.json").read_text(encoding="utf-8"))
    require(
        pins["schema"] == "sparkring-public-exl3-pins/v1",
        "wrong EXL3 pin schema",
    )
    verify_sources(pins)
    verify_imports()
    if args.phase == "gpu":
        verify_gpu()
    print(
        json.dumps(
            {
                "schema": "sparkring-public-exl3-verification/v1",
                "phase": args.phase,
                "profile_id": pins["profile_id"],
                "status": "pass",
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
