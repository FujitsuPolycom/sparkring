# GLM-5.2 EXL3 3.25-bpw recipe

Status: **live-validated candidate; public bootstrap pending**

This recipe records the exact long-context EXL3 serving configuration currently
used on the maintainer's four directly cabled DGX Sparks:

[`willfalco/GLM-5.2-EXL3-TR3-3.25bpw`](https://huggingface.co/willfalco/GLM-5.2-EXL3-TR3-3.25bpw)
at revision `d7d79c2d14599dfce7a5d12b85f7ad73f40e623d`.

The machine-readable source of truth is
[`recipes/glm52-exl3-tr3-3.25bpw.json`](../recipes/glm52-exl3-tr3-3.25bpw.json).
It contains the model hashes, EXL3 source commits, complete static environment,
and vLLM argument vector. Site-specific addresses, SSH users, NIC names, GID
indices, and paths remain in the ignored site configuration.

## Current contract

| Setting | Value |
|---|---|
| Hardware | 4x DGX Spark, direct 200-Gb/s cycle |
| Parallelism | TP4 / DCP4, `ag_rs` |
| Quantization | EXL3/Trellis 3.25 bpw; K3x192 and K4x64 expert tiers |
| MTP | fixed MTP3 |
| Batching | 8 sequences, 4,096 batched tokens, Q32 graph ceiling |
| Maximum model length | 1,048,576 tokens |
| KV allocation | 9,000,000,000 bytes per rank |
| Reported KV capacity | 1,125,632 tokens |
| KV representation | NVFP4 latent plus FP8 RoPE, per-token scale |
| Attention | `B12X_MLA_SPARSE`, CKV gather, exact global top-k |
| CUDA graphs | full and piecewise through Q32 |
| Prefix cache | enabled |
| SparkCache | disabled |
| Served name | `glm-5.2-exl3-tr3-3.25bpw` |

The 1M/9-GB settings are the current capacity profile. The earlier conservative
live gate used the same image and execution contract with a 262,144-token model
limit and 7,000,000,000 KV bytes per rank.

## Inspect the recipe offline

```bash
python scripts/sparkring_recipe.py plan \
  --recipe glm52-exl3-tr3-3.25bpw
```

This verifies the immutable model metadata, source pins, topology assumptions,
Q32/C8 relationship, fixed-MTP3 policy, packed-KV settings, and required explicit
unsets. It does not contact or alter any Spark.

## Evidence boundary

The configuration and its derived image completed four-rank startup, CUDA-graph
capture, API correctness, deterministic output, and sustained C1/C2/C8 serving
on the maintainer cluster. The current recipe therefore has `live-validated`
maturity.

It is **not yet public-build-ready**. The ARM64 ExLlamaV3 composition, composed
vLLM overlay, build script, and receipt verifier still need to move from the
maintainer lane into the public tree. After that, a clean checkout must build
the derived image and repeat the four-rank gate. Until those steps pass:

- NF3 remains the default public-functional recipe and quickstart.
- The EXL3 manifest is an exact, reviewable deployment contract—not a claim of
  one-command reproduction.
- Do not substitute mutable model tags, source heads, image tags, or arbitrary
  EXL3 wheels and call the result this recipe.

## Next publication gate

1. Publish the pinned ARM64 ExLlamaV3 adapter and composed vLLM EXL3 overlay.
2. Add a receipt-gated derived-image builder rooted in the NF3 public image.
3. Add model download/adoption checks for all 81 shards and `tier_bitmap.json`.
4. Generate site-local launch files from this recipe rather than copying flags.
5. Run a clean-checkout four-Spark startup, correctness, C1/C2/C8, and post-run
   health gate.
