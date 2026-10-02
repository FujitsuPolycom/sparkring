#!/usr/bin/env python3
"""Build a Qwen3.8-Flash-Next checkpoint whose text attention projections are MXFP8.

QAD step-5500 ships its 240 text attention projections (GDN ``linear_attn.in_proj_{qkv,z,a,b}``
/ ``out_proj`` and full-attention ``self_attn.{q,k,v,o}_proj`` / ``indexer.index_qk_proj``)
as source BF16. QAD step-4000 ships the same source weights post-training-quantized to
MXFP8 block32 (README: "MXFP8, Frozen"). This script copies the donor's MXFP8
``weight`` / ``weight_scale`` pairs into a new directory built from the source checkpoint:

- shards without a target tensor are hardlinked (same filesystem, no extra disk);
- shards with one are rewritten: each target BF16 ``weight`` is replaced by the donor's
  F8_E4M3 ``weight`` plus U8 (E8M0) ``weight_scale``, after asserting that vLLM's own
  ``_mxfp8_e4m3_quantize_torch`` on the source BF16 reproduces the donor bytes exactly;
- ``model.safetensors.index.json``, ``config.json`` and ``hf_quant_config.json`` gain one
  ``{"quant_algo": "MXFP8", "group_size": 32}`` entry per target module;
- ``derivation.json`` records the inputs; the source's HF revision marker is not copied.

Run inside a vLLM image (torch, safetensors, vLLM's MXFP8 utils), CPU-only, as the host
user so outputs are not root-owned, with one mount covering source, donor and output so
hardlinks work::

    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp \\
      -e CUDA_VISIBLE_DEVICES= -v ~/models:/models \\
      -v "$PWD/scripts/transplant_mxfp8_attention.py:/t.py:ro" \\
      --entrypoint python3 local/vllm-kk-mxfp8:20260930-b8e242af-9b4f167a /t.py \\
      --src /models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4@qad-step5500-ple1000 \\
      --donor /models/local-inference-lab/Qwen3.8-Flash-Next-NVFP4 \\
      --out /models/local/Qwen3.8-Flash-Next-NVFP4-QAD5500-mxfp8attn
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TARGET = re.compile(
    r"^model\.language_model\.layers\.\d+\."
    r"(linear_attn\.(in_proj_(qkv|z|a|b)|out_proj)"
    r"|self_attn\.(q_proj|k_proj|v_proj|o_proj|indexer\.index_qk_proj))\.weight$"
)
ENTRY = {"quant_algo": "MXFP8", "group_size": 32}
SKIP_FILES = {".vllm_console_revision.json", "model.safetensors.index.json"}


def load_index(d: Path) -> dict:
    return json.loads((d / "model.safetensors.index.json").read_text())


def read(d: Path, wm: dict, key: str) -> torch.Tensor:
    with safe_open(str(d / wm[key]), "pt") as f:
        return f.get_tensor(key)


def as_bytes(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def quant_config(cfg: dict) -> dict:
    if "quantization_config" in cfg:
        return cfg["quantization_config"]
    if "quantization" in cfg:
        return cfg["quantization"]
    return cfg  # hf_quant_config.json in flat form


def main() -> int:
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        _mxfp8_e4m3_quantize_torch,
    )

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--donor", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        sys.exit(f"refusing to overwrite existing {a.out}")

    src_idx, donor_idx = load_index(a.src), load_index(a.donor)
    swm, dwm = src_idx["weight_map"], donor_idx["weight_map"]
    targets = sorted(k for k in swm if TARGET.match(k))
    modules = [k[: -len(".weight")] for k in targets]
    if len(targets) != 240:
        sys.exit(f"expected 240 target projections, found {len(targets)}")
    for m in modules:
        for suffix in (".weight", ".weight_scale"):
            if m + suffix not in dwm:
                sys.exit(f"donor lacks {m}{suffix}")
        if m + ".weight_scale" in swm:
            sys.exit(f"source already has {m}.weight_scale")

    by_shard: dict[str, list[str]] = defaultdict(list)
    for k in targets:
        by_shard[swm[k]].append(k)
    shards = sorted(set(swm.values()))

    a.out.mkdir(parents=True)
    new_wm = dict(swm)
    for shard in shards:
        if shard not in by_shard:
            os.link(a.src / shard, a.out / shard)
            continue
        tensors, meta = {}, None
        with safe_open(str(a.src / shard), "pt") as f:
            meta = f.metadata()
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
        for k in by_shard[shard]:
            m = k[: -len(".weight")]
            src_bf16 = tensors[k]
            if src_bf16.dtype != torch.bfloat16:
                sys.exit(f"{k} is {src_bf16.dtype}, expected bfloat16")
            dw, ds = read(a.donor, dwm, m + ".weight"), read(a.donor, dwm, m + ".weight_scale")
            if dw.dtype != torch.float8_e4m3fn or ds.dtype != torch.uint8:
                sys.exit(f"donor {m} is {dw.dtype}/{ds.dtype}")
            q, s = _mxfp8_e4m3_quantize_torch(src_bf16, False)
            s = as_bytes(s).reshape(ds.shape)
            if not (torch.equal(as_bytes(q), as_bytes(dw)) and torch.equal(s, ds)):
                sys.exit(f"requantized {m} differs from donor; donor is not PTQ of this source")
            tensors[k] = dw
            tensors[m + ".weight_scale"] = ds
            new_wm[m + ".weight_scale"] = shard
        save_file(tensors, str(a.out / shard), metadata=meta)
        print(f"rewrote {shard}: {len(by_shard[shard])} projections", flush=True)

    for p in a.src.iterdir():
        if p.is_file() and p.suffix != ".safetensors" and p.name not in SKIP_FILES:
            shutil.copy2(p, a.out / p.name)

    total = sum((a.out / s).stat().st_size for s in shards)
    idx = {"metadata": {**src_idx.get("metadata", {}), "total_size": total},
           "weight_map": dict(sorted(new_wm.items()))}
    (a.out / "model.safetensors.index.json").write_text(json.dumps(idx, indent=2) + "\n")

    for name in ("config.json", "hf_quant_config.json"):
        path = a.out / name
        cfg = json.loads(path.read_text())
        ql = quant_config(cfg).setdefault("quantized_layers", {})
        clash = [m for m in modules if m in ql]
        if clash:
            sys.exit(f"{name} already lists {clash[:3]}")
        ql.update({m: dict(ENTRY) for m in modules})
        path.write_text(json.dumps(cfg, indent=2) + "\n")

    digest = hashlib.sha256("\n".join(modules).encode()).hexdigest()
    (a.out / "derivation.json").write_text(json.dumps({
        "kind": "mxfp8_attention_transplant",
        "source": str(a.src),
        "source_revision_marker": json.loads((a.src / ".vllm_console_revision.json").read_text())
        if (a.src / ".vllm_console_revision.json").exists() else None,
        "donor": str(a.donor),
        "modules": len(modules),
        "modules_sha256": digest,
        "format": ENTRY,
        "requantization_check": "vLLM _mxfp8_e4m3_quantize_torch(source BF16) == donor bytes, every module",
        "rewritten_shards": sorted(by_shard),
        "hardlinked_shards": len(shards) - len(by_shard),
    }, indent=2) + "\n")
    print(f"done: {len(modules)} modules, {len(by_shard)} shards rewritten, "
          f"{len(shards) - len(by_shard)} hardlinked, total {total / 2**30:.2f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
