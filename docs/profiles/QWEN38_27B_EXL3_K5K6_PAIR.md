# Qwen3.8-27B EXL3 K5/K6 two-Spark profile

## Tested setup

The profile ran on two directly cabled DGX Sparks and produced the results
linked below.

Use the
[two-Spark quickstart](../QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) to build and
distribute the public image, verify the pinned checkpoint, and
launch one rank per Spark.

## Serving contract

| Setting | Value |
|---|---|
| Model | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Parallelism | TP2/DCP1 across two DGX Sparks |
| Transport | patched NCCL over one direct RoCEv2 link |
| Advertised request limit | 1,048,576 tokens through static YaRN factor 4 |
| Maximum sequences | 32 |
| Scheduler budget | 8,192 tokens |
| Key-value cache | explicit FP8, 0.70 unified-memory utilization |
| Prefix caching | native cache with mamba alignment |
| Speculation | Qwen MTP depth 3, probabilistic drafts, standard rejection sampling |
| Decode | full-decode CUDA graphs |
| External key-value cache | disabled |
| SIRCL | unsupported for width 5,120 |

The pair profile does not copy DeepSeek's model-specific MLA layout, DSpark
proposer, explicit KV byte reservation, or block geometry. Its 8,192-token
scheduler budget matches the operator's selected comparison envelope. The
Qwen LMCache path cannot compose with that budget, so LMCache and SparkCache
remain outside this profile.

## Benchmark results

See the [benchmark table, screenshot, and machine-readable data](../../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md).

## Limitations

- Static YaRN can shift short-context output distributions relative to native
  262,144-token serving.
- Generic FP8 KV is explicit. The checkpoint has no KV-format metadata that
  makes raw vLLM `auto` equivalent to this profile.
- Patched NCCL is active; SparkRing custom collectives are not.

The sanitized evidence is in
[`performance/receipts/qwen38-27b/temp1/20260823-tp2/`](../../performance/receipts/qwen38-27b/temp1/20260823-tp2/).
