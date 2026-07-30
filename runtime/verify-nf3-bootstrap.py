#!/usr/bin/env python3
"""Fail-closed import/ABI check for the thin NF3 bootstrap image."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument(
        "--expected-receipt-sha256",
        default=os.environ.get("SPARKRING_NF3_INPUT_RECEIPT_SHA256"),
        required=os.environ.get("SPARKRING_NF3_INPUT_RECEIPT_SHA256") is None,
    )
    parser.add_argument(
        "--expected-b12x-commit",
        default=os.environ.get("SPARKRING_NF3_B12X_COMMIT"),
        required=os.environ.get("SPARKRING_NF3_B12X_COMMIT") is None,
    )
    parser.add_argument(
        "--expected-spark-port-commit",
        default=os.environ.get("SPARKRING_NF3_SPARK_PORT_COMMIT"),
        required=os.environ.get("SPARKRING_NF3_SPARK_PORT_COMMIT") is None,
    )
    parser.add_argument(
        "--expected-sparkring-commit",
        default=os.environ.get("SPARKRING_NF3_SOURCE_COMMIT"),
        required=os.environ.get("SPARKRING_NF3_SOURCE_COMMIT") is None,
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    actual_receipt_sha256 = _sha256(args.receipt)
    if actual_receipt_sha256 != args.expected_receipt_sha256:
        raise RuntimeError(
            "NF3 input receipt hash mismatch: "
            f"{actual_receipt_sha256} != {args.expected_receipt_sha256}"
        )
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    if receipt.get("schema") != "sparkring-nf3-bootstrap-input/v1":
        raise RuntimeError("unknown or missing NF3 bootstrap receipt schema")
    expected_pins = {
        "b12x": args.expected_b12x_commit,
        "spark_port": args.expected_spark_port_commit,
    }
    for field, expected in expected_pins.items():
        observed = (receipt.get(field) or {}).get("commit")
        if observed != expected:
            raise RuntimeError(
                f"NF3 receipt {field} commit mismatch: {observed} != {expected}"
            )
    if receipt.get("sparkring_source_commit") != args.expected_sparkring_commit:
        raise RuntimeError("NF3 receipt SparkRing source commit mismatch")

    site_packages = Path("/opt/venv/lib/python3.12/site-packages")
    installed_roots = {
        "b12x/": site_packages / "b12x",
        "overlay/": site_packages,
        "sparkring/": Path("/opt/spark-vllm"),
    }
    file_records = receipt.get("files")
    if not isinstance(file_records, dict) or not file_records:
        raise RuntimeError("NF3 receipt has no file inventory")
    for relative, expected in file_records.items():
        prefix = next(
            (candidate for candidate in installed_roots if relative.startswith(candidate)),
            None,
        )
        if prefix is None:
            raise RuntimeError(f"NF3 receipt has unsupported path: {relative}")
        suffix = relative.removeprefix(prefix)
        installed = installed_roots[prefix] / suffix
        if not installed.is_file():
            raise RuntimeError(f"NF3 installed file is missing: {installed}")
        actual = _sha256(installed)
        if actual != expected:
            raise RuntimeError(
                f"NF3 installed-file hash mismatch for {relative}: "
                f"{actual} != {expected}"
            )

    import hybrid_loader

    modelopt = importlib.import_module(
        "vllm.model_executor.layers.quantization.modelopt"
    )
    method = modelopt.ModelOptNvFp4Config.FusedMoEMethodCls
    if (method.__module__, method.__name__) != (
        "hybrid_loader",
        "HybridNvFp4MoE",
    ):
        raise RuntimeError(
            "hybrid loader did not own ModelOpt FusedMoEMethodCls: "
            f"{method.__module__}.{method.__name__}"
        )

    base = method.__mro__[1]
    for name in ("create_weights", "apply", "maybe_make_prepare_finalize"):
        candidate = inspect.signature(getattr(method, name))
        base_signature = inspect.signature(getattr(base, name))
        required = {
            parameter
            for parameter, value in base_signature.parameters.items()
            if parameter != "self"
            and value.default is inspect.Parameter.empty
            and value.kind
            not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            )
        }
        has_kwargs = any(
            value.kind is inspect.Parameter.VAR_KEYWORD
            for value in candidate.parameters.values()
        )
        missing = required - set(candidate.parameters)
        if missing and not has_kwargs:
            raise RuntimeError(
                f"hybrid {name} cannot accept base parameters: {sorted(missing)}"
            )

    modules = {
        "kernel": importlib.import_module("b12x.moe.fused.w4a16.kernel"),
        "prepare": importlib.import_module("b12x.moe.fused.w4a16.prepare"),
        "host": importlib.import_module("b12x.moe.fused.w4a16.host"),
        "fp4": importlib.import_module("b12x.cute.fp4"),
    }
    required_abi = {
        "kernel": ("compile_w4a16_fused_moe", "run_w4a16_moe"),
        "prepare": ("prepare_nf3_moe_weights",),
        "host": ("validate_nf3_moe_inputs",),
        "fp4": ("packed_dequant_nf3x8_to_bfloat2x4", "nf3_codebook_pools"),
    }
    for label, names in required_abi.items():
        missing = [name for name in names if not hasattr(modules[label], name)]
        if missing:
            raise RuntimeError(f"{label} missing NF3 ABI: {missing}")

    for module_name in (
        "spark_nf3_startup_profile_cap",
        "spark_nf3_workspace_reserve",
    ):
        module = importlib.import_module(module_name)
        if not callable(getattr(module, "install", None)):
            raise RuntimeError(f"{module_name} has no install() hook")

    report = {
        "schema": "sparkring-nf3-bootstrap-verification/v1",
        "hybrid_method": f"{method.__module__}.{method.__name__}",
        "hybrid_loader_sha256": _sha256(hybrid_loader.__file__),
        "b12x": {
            label: {
                "path": module.__file__,
                "sha256": _sha256(module.__file__),
            }
            for label, module in modules.items()
        },
        "input_receipt_sha256": actual_receipt_sha256,
        "passed": True,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
