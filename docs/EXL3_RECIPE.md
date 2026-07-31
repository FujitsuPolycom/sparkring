# GLM-5.2 EXL3 3.25-bpw recipe

Status: **live-validated configuration; public source bootstrap
offline-validated; clean-checkout four-Spark gate pending**

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

## Build and launch from public sources

Complete [PREREQUISITES.md](PREREQUISITES.md) and the site/SSH/fabric sections
of [QUICKSTART.md](QUICKSTART.md) first. Use the same ignored `site.yaml` as
the NF3 bootstrap. Plan mode is entirely offline:

```bash
python scripts/bootstrap_exl3.py plan \
  --site scripts/config/site.yaml
```

The first execution is intentionally explicit and should normally use
`--no-launch` so the generated files can be reviewed before serving:

```bash
python scripts/bootstrap_exl3.py execute \
  --site scripts/config/site.yaml \
  --no-launch \
  --confirmation BOOTSTRAP-EXL3-ALL-FOUR
```

It performs these receipt-gated operations:

1. verifies management SSH and the exact three-hop direct-ring fanout tree;
2. verifies the 200 GbE/RDMA fabric before downloading or building;
3. adopts or resumes the pinned 81-shard model on rank 0, with a capacity gate
   and full per-shard verification against the pinned Hugging Face manifest;
4. copies model bytes to both neighbors in parallel, then relays to the opposite
   rank over direct 200 GbE addresses using resumable `rsync`;
5. builds or reuses the exact NF3 NVFP4/FP8-RoPE base layer;
6. reconstructs the exact public ExLlamaV3 and SparkInfer Git trees from pinned
   base commits plus hash-checked patches;
7. builds and verifies one derived ARM64 image on rank 0, then fans that exact
   image ID over the same direct-ring tree; and
8. writes ignored resolved files under `.sparkring/bootstrap-exl3/` and runs
   the full public preflight.

Review the generated contract:

```bash
cat .sparkring/bootstrap-exl3/site.yaml
cat .sparkring/bootstrap-exl3/launch.json

python scripts/sparkring_exl3_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  plan
```

Then start exactly that profile:

```bash
python scripts/sparkring_exl3_launcher.py \
  --site .sparkring/bootstrap-exl3/site.yaml \
  --profile .sparkring/bootstrap-exl3/launch.json \
  --execute start
```

Both the model download and derived-image build are resumable: exact existing
payloads are verified and skipped. Partial Hugging Face and `rsync` transfers
resume in place. First-time adoption hashes all 81 weight shards, so it can
take several minutes without consuming GPU time. Mutable source heads,
incomplete or same-size-corrupted shard sets, wrong metadata, unattested source
trees, mismatched base-image IDs, and dirty build contexts fail closed.

## Evidence boundary

The configuration and its derived image completed four-rank startup, CUDA-graph
capture, API correctness, deterministic output, and sustained C1/C2/C8 serving
on the maintainer cluster. The current recipe therefore has `live-validated`
maturity.

The public source composition, full vLLM EXL3 overlay, derived-image builder,
model verifier, 200 GbE fanout, and dry-run-first launcher are now published and
offline-validated. The remaining acceptance step is a clean checkout building
the derived image and repeating the four-rank gate. Until that passes:

- NF3 remains the default public-functional recipe and quickstart.
- The EXL3 bootstrap is a candidate, not a claim of independent one-command
  reproduction.
- Do not substitute mutable model tags, source heads, image tags, or arbitrary
  EXL3 wheels and call the result this recipe.

## Next publication gate

1. Run the public bootstrap from a clean checkout on rank 0.
2. Confirm the 81-shard adoption/download and direct-ring fanout receipts.
3. Confirm the derived image ID on all four ranks.
4. Complete four-rank startup, correctness, C1/C2/C8, and post-run health
   gates.
5. Record the public-path receipt and promote the bootstrap from candidate.
