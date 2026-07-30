#!/usr/bin/env python3
"""Fail-closed ABI probe for the NF3 plus packed-MLA compatibility layer."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
from pathlib import Path


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

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
    method_signatures = {}
    for name in ("create_weights", "apply", "maybe_make_prepare_finalize"):
        candidate = inspect.signature(getattr(method, name))
        base_signature = inspect.signature(getattr(base, name))
        method_signatures[name] = {
            "hybrid": str(candidate),
            "base": str(base_signature),
        }
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
                f"hybrid {name} cannot accept base parameters: "
                f"{sorted(missing)}"
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
        "fp4": (
            "packed_dequant_nf3x8_to_bfloat2x4",
            "nf3_codebook_pools",
        ),
    }
    for label, names in required_abi.items():
        missing = [name for name in names if not hasattr(modules[label], name)]
        if missing:
            raise RuntimeError(f"{label} missing NF3 ABI: {missing}")

    mla = importlib.import_module("b12x.integration.mla")
    mla_signatures = {}
    for name in ("sparse_mla_decode_forward", "sparse_mla_extend_forward"):
        signature = inspect.signature(getattr(mla, name))
        mla_signatures[name] = str(signature)
        if "scale_format" not in signature.parameters:
            raise RuntimeError(
                f"{name} missing packed-MLA scale_format ABI required by "
                "nvfp4_ds_mla"
            )

    model_contract = None
    if args.model_config:
        config = json.loads(args.model_config.read_text(encoding="utf-8"))
        quant = config.get("quantization_config") or {}
        bit_map = quant.get("hybrid_bit_map")
        if not isinstance(bit_map, dict):
            raise RuntimeError("model config has no hybrid_bit_map")
        expected_layers = {str(layer) for layer in range(3, 78)}
        if set(bit_map) != expected_layers:
            raise RuntimeError("hybrid layer keys do not cover layers 3..77")
        for layer, bits in bit_map.items():
            counts = {
                "experts": len(bits),
                "nvfp4": bits.count(4),
                "nf3": bits.count(3),
            }
            if counts != {"experts": 256, "nvfp4": 64, "nf3": 192}:
                raise RuntimeError(f"layer {layer} tier mismatch: {counts}")
        model_contract = {
            "hybrid_layers": len(bit_map),
            "first_layer": 3,
            "last_layer": 77,
            "per_layer": {"experts": 256, "nvfp4": 64, "nf3": 192},
            "hybrid_scheme": quant.get("hybrid_scheme"),
        }

    report = {
        "schema": "sparkring-nf3-nvfp4-rope8-verification/v1",
        "hybrid_method": f"{method.__module__}.{method.__name__}",
        "method_signatures": method_signatures,
        "mla_signatures": mla_signatures,
        "hybrid_loader_sha256": _sha256(hybrid_loader.__file__),
        "b12x": {
            label: {
                "path": module.__file__,
                "sha256": _sha256(module.__file__),
            }
            for label, module in modules.items()
        },
        "model_contract": model_contract,
        "passed": True,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
