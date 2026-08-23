# Qwen3.8-27B EXL3 K5/K6 two-Spark profile

## Tested setup

The profile ran on two directly cabled DGX Sparks and produced the
temperature-one results linked below.

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
| Temperature-one benchmark sampling | request sets temperature 1.0; pinned model config supplies effective top-p 0.95/top-k 20 |
| Decode | full-decode CUDA graphs |
| External key-value cache | disabled |
| SIRCL | unsupported for width 5,120 |

The pair profile does not copy DeepSeek's model-specific MLA layout, DSpark
proposer, explicit KV byte reservation, or block geometry. Its 8,192-token
scheduler budget matches the operator's selected comparison envelope. The
Qwen LMCache path cannot compose with that budget, so LMCache and SparkCache
remain outside this profile.

## Earlier startup check

Conditions: two directly cabled NVIDIA DGX Sparks, the pinned checkpoint,
identical runtime inputs, vLLM
`229effc810ee6b8112f661472f6aace4eb8c787d`, ExLlamaV3
`5f3c537ca9d89893d771256f5c43c93656553fbb`, patched NCCL SHA-256
`e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54`,
TP2/DCP1, a 1,000,000-token static-YaRN limit, FP8 KV, native prefix caching,
and probabilistic MTP3.
LMCache, SparkCache and SIRCL were disabled.

Measurement: engine logs supplied startup, model memory, graph and KV-capacity
values. Bounded requests checked health, arithmetic, tools, vision, repeated
prefixes and distinct shared-prefix suffixes. A separate temperature-1 request
measured speculative-counter deltas.

Result: both ranks rendezvoused and served. `/v1/models` advertised 1,000,000
tokens. The engine reported 11.22 GiB of model memory per rank, 67.96 GiB of
available KV memory per rank, 4,093,750 logical KV tokens, and 4.09x maximum
concurrency at the advertised limit. The temperature-1 probe accepted 85 of
126 draft tokens over 42 speculative steps: 67.5% draft acceptance and 3.02
mean acceptance length. Arithmetic returned `391`, the tool parser emitted
`multiply(a=6,b=7)`, vision returned `VISION_OK`, and shared suffixes returned
`13` and `17`.

Conclusion: the earlier startup checks passed.

## Benchmark results

See the [temperature-one table, screenshot, and machine-readable data](../../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md).

## Limitations

- Static YaRN can shift short-context output distributions relative to native
  262,144-token serving.
- Generic FP8 KV is explicit. The checkpoint has no KV-format metadata that
  makes raw vLLM `auto` equivalent to this profile.
- Patched NCCL is active; SparkRing custom collectives are not.

The sanitized evidence is in
[`performance/receipts/qwen38-27b/temp1/20260823-tp2/`](../../performance/receipts/qwen38-27b/temp1/20260823-tp2/).
