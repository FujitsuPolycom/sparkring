#!/usr/bin/env python3
"""Derive the public stock-DCP4 R7 baseline from the candidate generator output.

The stock-DCP4 baseline is the speculation-off control: the zero-MTP, DCP4,
KV9.0 GB, fp8_ds_mla, hybrid-transport, target-only-K6 baseline from which
all fixed-MTP derivative profiles were derived. This generator reconstructs
that baseline from the tracked candidate generator plus the recipe's serving
contract, so no untracked profile file is required.

The stock-DCP4 baseline is the MTP-off, Q24-graph, 9-GB-KV/rank profile
that ``prepare_exl3_r7_mtp2.py`` consumes as its ``--stock-profile`` input.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import generate_exl3_r7_candidate as gen

ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "recipes/glm52-exl3-r7-3.5bpw.json"

STOCK_KV_BYTES_PER_RANK = 9_000_000_000
STOCK_MAX_QUERY_ROWS = 24
DCP_COMM_BACKEND = "ag_rs"
DCP_KV_CACHE_INTERLEAVE_SIZE = "1"


class StockProfileError(ValueError):
    """The derived stock-DCP4 profile does not match the qualified contract."""


def _load_recipe() -> dict:
    return json.loads(RECIPE_PATH.read_text(encoding="utf-8"))


def derive_stock_profile(candidate_template: dict, pins: dict, recipe: dict) -> dict:
    """Derive the stock-DCP4 baseline from the candidate generator output.

    The stock baseline is the MTP-off, Q24-graph, 9-GB-KV profile. It
    inherits the transport, online-K6, and attestation contract from the
    candidate generator, then applies the DCP4 serving args and removes
    MTP speculation.
    """

    # The stock-DCP4 control uses the hybrid SIRCL + patched NCCL-IB transport.
    stock_template = copy.deepcopy(candidate_template)
    stock_template["transport"] = "sircl-nccl-ib"
    base = gen.generate(stock_template, pins, recipe)
    profile = copy.deepcopy(base)

    profile["profile_id"] = recipe["recipe_id"]

    # Stock DCP4: MTP off, Q8 query ceiling, Q24 graph capture, 9.0 GB KV/rank.
    # The generator already sets MAX_QUERY_ROWS=8 and captures Q1-Q32; the
    # stock control must capture Q24 (the MTP2 ceiling) but the runtime
    # query-row env stays at 8 (MTP-off default: 8*(0+1)=8).
    profile["environment"]["VLLM_SPARK_MTP_TOKENS"] = "0"
    profile["environment"]["VLLM_SPARK_TP4_INDEXER_GRAPH_CUSTOM"] = "0"

    args = profile["extra_vllm_args"]
    # Preserve the qualified argument order: DCP configuration follows the
    # selected MoE backend and precedes the attention backend.
    insert_at = args.index("--moe-backend") + 2
    args[insert_at:insert_at] = [
        "--dcp-comm-backend", DCP_COMM_BACKEND,
        "--dcp-kv-cache-interleave-size", DCP_KV_CACHE_INTERLEAVE_SIZE,
    ]

    # Set KV cache dtype to fp8_ds_mla (the qualified stock value)
    kv_index = args.index("--kv-cache-dtype") + 1
    args[kv_index] = "fp8_ds_mla"

    # The generator already captures Q1-Q32 (which includes Q24, the MTP2
    # ceiling) and sets max-cudagraph-capture-size=32. The stock control
    # requires Q24 in the capture list; Q1-Q32 satisfies that.

    # Remove speculative config if present
    if "--speculative-config" in args:
        spec_idx = args.index("--speculative-config")
        del args[spec_idx:spec_idx + 2]

    return profile


def validate_stock_profile(profile: dict) -> None:
    """Fail closed unless the profile matches the stock-DCP4 contract."""

    if profile.get("schema") != gen.OUTPUT_SCHEMA:
        raise StockProfileError("stock profile has the wrong schema")
    if profile.get("model_family") != "exl3-r7":
        raise StockProfileError("stock profile is not an EXL3 R7 profile")
    if profile.get("model_container_path") != "/models/glm52-exl3-r7-3.5bpw":
        raise StockProfileError("stock profile has the wrong checkpoint mount")
    env = profile.get("environment")
    if not isinstance(env, dict):
        raise StockProfileError("stock profile has malformed environment")
    if env.get("VLLM_SPARK_MTP_TOKENS") != "0":
        raise StockProfileError("stock DCP4 must be MTP-off")
    if env.get("ONLINE_QUANT") != "exl3-b6":
        raise StockProfileError("stock DCP4 must preserve target online K6")
    if env.get("VLLM_EXL3_ONLINE_TRELLIS_BITS") != "6":
        raise StockProfileError("stock DCP4 must preserve K6 trellis bits")
    if env.get("VLLM_SPARK_SHARED_CAPTURE_STREAM") != "1":
        raise StockProfileError("stock DCP4 must preserve shared capture stream")
    args = profile.get("extra_vllm_args")
    if not isinstance(args, list):
        raise StockProfileError("stock profile has malformed arguments")
    if "--speculative-config" in args:
        raise StockProfileError("stock DCP4 must not have speculative config")
    if "--enforce-eager" in args:
        raise StockProfileError("stock DCP4 must use CUDA graphs")
    idx = args.index("--dcp-comm-backend") + 1 if "--dcp-comm-backend" in args else -1
    if idx < 0 or args[idx] != DCP_COMM_BACKEND:
        raise StockProfileError("stock DCP4 must use ag_rs DCP backend")
    idx = args.index("--dcp-kv-cache-interleave-size") + 1 if "--dcp-kv-cache-interleave-size" in args else -1
    if idx < 0 or args[idx] != DCP_KV_CACHE_INTERLEAVE_SIZE:
        raise StockProfileError("stock DCP4 must use interleave size 1")
    idx = args.index("--kv-cache-dtype") + 1 if "--kv-cache-dtype" in args else -1
    if idx < 0 or args[idx] != "fp8_ds_mla":
        raise StockProfileError("stock DCP4 must use fp8_ds_mla KV cache")
    comp_idx = args.index("--compilation-config") + 1 if "--compilation-config" in args else -1
    if comp_idx < 0:
        raise StockProfileError("stock DCP4 must have compilation config")
    compilation = json.loads(args[comp_idx])
    if compilation.get("cudagraph_mode") != "FULL_AND_PIECEWISE":
        raise StockProfileError("stock DCP4 must use FULL_AND_PIECEWISE graphs")
    sizes = compilation.get("cudagraph_capture_sizes")
    if not isinstance(sizes, list) or STOCK_MAX_QUERY_ROWS not in sizes:
        raise StockProfileError("stock DCP4 must capture Q24")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=gen.TEMPLATE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        template = json.loads(args.template.read_text(encoding="utf-8"))
        pins = json.loads(gen.PINS_PATH.read_text(encoding="utf-8"))
        recipe = _load_recipe()
        profile = derive_stock_profile(template, pins, recipe)
        validate_stock_profile(profile)
    except (OSError, json.JSONDecodeError, KeyError, gen.CandidateError, StockProfileError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    print(f"WROTE: {args.output}")
    print("OFFLINE: stock-DCP4 baseline derived from tracked inputs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
