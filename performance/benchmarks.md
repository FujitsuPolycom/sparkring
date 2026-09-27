# Benchmark results

Each row links the exact measured configuration, including its image identity.
The first table includes retired profiles; its results do not establish
performance for other images or configurations.

Throughput values are tokens per second; decode is aggregate output throughput.
Each record specifies its sampling, workload and measurement conditions;
results from different conditions are not matched comparisons.

| Profile | Decode context | Prefill | C1 decode | C8 decode | Highest C at this context | Coding peak |
|---|---:|---:|---:|---:|---:|---:|
| [Qwen3.8-Flash-Next NVFP4 QAD step 5500 · `sparkring install` · 2 Sparks](records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md) | 16K | 3,927 (16K scout) | 46.8 | 159.2 | C16: 240.1 | — |
| [Qwen3.8-Flash-Next NVFP4 QAD step 5500 · `sparkring install` · 4 Sparks](records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md) | 16K | 4,855 (16K scout) | 80.2 | 273.6 | C16: 413.9 | — |
| [GLM-5.3-Flash NVFP4-Spark · `sparkring install` · 2 Sparks](records/images/dev-20260925-qwendecode-glm-prefill-20260926.md) | 8K | 2,320 (16K scout) | 38.3 | 113.8 | C8: 113.8 | — |
| [GLM-5.3-Flash NVFP4-Spark · `sparkring install` · 4 Sparks](records/images/dev-20260925-qwendecode-glm-prefill-20260926.md) | 8K | 3,758 (16K scout) | 61.6 | 197.5 | C16: 265.1 | — |
| [GLM-5.3-Flash NVFP4-Spark · MTP3 + mesh · 4 Sparks](records/glm53-flash/spark-mtp3-mesh-20260905.md) | 8K | 2,703 (8K scout) | 48.2 | 168.8 | C16: 231.3 | — |
| [GLM-5.3-Flash NVFP4-Spark · DFlash2 exact request-batch graphs · 4 Sparks](records/glm53-flash/dflash2-exact-concurrency-graphs-20260904.md) | 16K | 2,717 (16K scout) | 43.05 | 134.3 | C16: 187.0 | — |
| [GLM-5.3-Flash NVFP4 · DFlash2/B12X-KDA DCP4 · 4 Sparks](records/glm53-flash/b12x-kda-dcp4-20260903.md) | 16K | 2,649 (16K scout) | 37.97 | — | C4: 90.36 | — |
| [GLM-5.2 EXL3 3.5-bpw · 4 Sparks](records/glm-3.5bpw/normalized-base-20260822.md) | 16K | 671 (16K) | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| [DeepSeek-V4-Flash DSpark · 2 Sparks](records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | 16K | 1,926 (16K) | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| [DeepSeek-V4-Flash-0731 · 4 Sparks](records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | 16K | 2,488 (16K) | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| [DeepSeek-V4.1-Flash · Engram on NVMe · DSpark k=5 · 4 Sparks](records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md) | short prompts, temp 0 | 1,432 (11,592 tokens) | 49.8 | — | C6: 159.9 | 77.3 |
| [Qwen3.8-27B EXL3 K5/K6 · 2 Sparks](records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | 16K | 1,367 (16K) | 29.50 | 142.20 | C16: 184.39 | 39.95 |
| [Qwen3.8-27B EXL3 K5/K6 · 4 Sparks](records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | 16K | 1,964 (16K) | 35.07 | 191.02 | C8: 191.02 | 48.46 |

The DFlash exact-graph row preserves the [published summary](https://github.com/FujitsuPolycom/sparkring/blob/c65a9981e2a69f821ac716f6f13484d99d23f4ea/README.md);
its linked record contains conditions and selected concurrency comparisons.

The DeepSeek-V4.1 row uses the record's initial eight-category prompt run;
its aggregate decode includes time to first token. A separate balanced,
packed-Engram configuration recorded **1,873 tok/s at 16K** and **2,058 tok/s
at 64K** prefill. Those values do not share the row's decode configuration.

See [full results](../docs/RESULTS.md) and the
[mesh validation report](records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
for repeat counts, accuracy checks, settings, and limitations.

### Qwen3.8-Flash-Next with `sparkring install`

Status: **implemented**; one run per cluster, not serving-qualified.

`sudo sparkring install` deployed the `qwen38-flash-next-tp2` (one directly
cabled pair) and `qwen38-flash-next-qad-tp4` (one four-Spark ring) profiles
with checkpoint `qad-step5500-ple1000` of
[Local Inference Lab's Qwen3.8-Flash-Next NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4),
MTP3 and image `dev-20260925-qwendecode-cuda1342-nccl2323-status031`.
[llm-inference-bench](https://github.com/local-inference-lab/llm-inference-bench)
0.6.2 measured each at temperature 1.0 with 20 s per cell after a 5 s warm-up
and up to 2,048 output tokens. Prefill is one cold prompt's length divided by
its time to first token.

| Prompt | Two Sparks (tok/s) | Time to first token | Four Sparks (tok/s) | Time to first token |
|---|---:|---:|---:|---:|
| 8K | 3,744 | 2.19 s | 4,722 | 1.74 s |
| 16K | 3,927 | 4.17 s | 4,855 | 3.38 s |
| 32K | 3,841 | 8.53 s | 4,756 | 6.89 s |
| 64K | 3,689 | 17.77 s | 4,559 | 14.38 s |
| 128K | 3,366 | 38.94 s | 4,098 | 31.98 s |

Decode: aggregate output tokens per second, then (verification steps per
second × tokens per step).

| Two Sparks | 1 stream | 8 streams | 16 streams |
|---|---:|---:|---:|
| 0 context | 52.6 (23.4 × 2.25) | 198.0 (83.3 × 2.38) | 284.2 (122.0 × 2.33) |
| 8K | 44.8 (23.4 × 1.91) | 171.9 (85.8 × 2.00) | 241.8 (120.9 × 2.00) |
| 16K | 46.8 (23.2 × 2.02) | 159.2 (83.7 × 1.90) | 240.1 (122.5 × 1.96) |
| 32K | 44.9 (23.1 × 1.95) | 160.2 (82.7 × 1.94) | 241.6 (120.3 × 2.01) |
| 64K | 48.0 (22.7 × 2.11) | 164.6 (81.1 × 2.03) | 237.6 (118.8 × 2.00) |

| Four Sparks | 1 stream | 8 streams | 16 streams |
|---|---:|---:|---:|
| 0 context | 73.6 (33.8 × 2.17) | 276.0 (119.4 × 2.31) | 407.5 (172.7 × 2.36) |
| 8K | 81.2 (33.6 × 2.42) | 267.6 (119.5 × 2.24) | 408.3 (170.3 × 2.40) |
| 16K | 80.2 (33.8 × 2.37) | 273.6 (119.1 × 2.30) | 413.9 (172.8 × 2.39) |
| 32K | 72.3 (33.1 × 2.18) | 268.3 (115.5 × 2.32) | 397.4 (168.3 × 2.36) |
| 64K | 67.2 (32.3 × 2.08) | 270.1 (112.0 × 2.41) | 393.7 (160.1 × 2.46) |

At temperature 1.0 the tokens per step follow the generated text, so decode
rates vary between runs; steps per second vary less. The
[record](records/images/dev-20260925-qwendecode-qwen-step5500-20260926.md)
has the conditions, single-stream probes by prompt type and the raw matrices.

### GLM-5.3-Flash with MTP3 and SparkCache

These observations use the [SparkRing R33 image](../runtime/releases/sparkring-r33/release.json).

Status: **research-only**. These reported values lack a complete public
harness/method record and must not be treated as matched comparisons with the
benchmarks above. Their linked records separately qualify bounded functional
checks and describe the missing measurement details.

| NVFP4-Spark MTP3 + SparkCache profile | Context | Prefill tok/s, median of 3 | C1 output tok/s | C4 aggregate output tok/s |
|---|---:|---:|---:|---:|
| [R33 four-Spark ring](records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md) | 8K | 3,274 | 51.4 | 132.5 |
| [R33 two-Spark pair](records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md) | 8K | 2,340 | 33.1 | 66.1 |

The [TP4/DCP4 record](records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md)
contains the DCP4 configuration's 8.36M KV pool and functional checks.
For its throughput observations, apply the [published correction](https://github.com/FujitsuPolycom/sparkring/pull/271):
the first measurement windows overlap host-level distribution traffic and lack
replacement measurements. The C4 difference has no profiling-based attribution;
the prefill-only full-KV gather does not explain a decode-window difference.
