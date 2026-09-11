# Benchmark results

Each row links the exact measured configuration. These records include retired
profiles and are not benchmark results for the shared-image build.

Decode is aggregate output throughput. Each record specifies its sampling, workload and measurement conditions; results from different conditions are not matched comparisons.

| Profile | Decode context | Prefill | C1 decode | C8 decode | Highest C at this context | Coding peak |
|---|---:|---:|---:|---:|---:|---:|
| [GLM-5.3 NVFP4-Spark · native MTP3 + mesh · 4 Sparks](records/glm53-flash/spark-mtp3-mesh-20260905.md) | 8K | 2,703 (8K scout) | 48.2 | 168.8 | C16: 231.3 | — |
| [GLM-5.3 NVFP4-Spark · DFlash2 exact request-batch graphs · 4 Sparks](records/glm53-flash/dflash2-exact-concurrency-graphs-20260904.md) | 16K | 2,717 (16K scout) | 43.05 | 134.3 | C16: 187.0 | — |
| [GLM-5.3 NVFP4 · DFlash2/B12X-KDA DCP4 · 4 Sparks](records/glm53-flash/b12x-kda-dcp4-20260903.md) | 16K | 2,649 (16K scout) | 37.97 | — | C4: 90.36 | — |
| [GLM-5.2 EXL3 3.5-bpw · 4 Sparks](records/glm-3.5bpw/normalized-base-20260822.md) | 16K | 671 (16K) | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| [DeepSeek-V4-Flash DSpark · 2 Sparks](records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | 16K | 1,926 (16K) | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| [DeepSeek-V4-Flash-0731 · 4 Sparks](records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | 16K | 2,488 (16K) | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| [DeepSeek-V4.1-Flash · Engram on NVMe · DSpark k=5 · 4 Sparks](records/deepseek-v41-flash/cycle-tp4-dspark5-graphs-20260910.md) | short prompts, temp 0 | 1,873 (16K) / 2,058 (64K) | 56.2 | — | C6: 159.9 | 77.3 |
| [Qwen3.8-27B EXL3 K5/K6 · 2 Sparks](records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | 16K | 1,367 (16K) | 29.50 | 142.20 | C16: 184.39 | 39.95 |
| [Qwen3.8-27B EXL3 K5/K6 · 4 Sparks](records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | 16K | 1,964 (16K) | 35.07 | 191.02 | C8: 191.02 | 48.46 |

See [full results](../docs/RESULTS.md) and the
[mesh validation report](records/glm53-flash/spark-mtp3-validation-summary-20260905.md)
for repeat counts, accuracy checks, settings, and limitations.

### R33 throughput observations

Status: **research-only**. These reported values lack a complete public
harness/method record and must not be treated as matched comparisons with the
benchmarks above. Their linked records separately qualify bounded functional
checks and describe the missing measurement details.

| NVFP4-Spark MTP3 + SparkCache profile | Context | Prefill tok/s, median of 3 | C1 output tok/s | C4 aggregate output tok/s |
|---|---:|---:|---:|---:|
| [R33 four-Spark ring](records/glm53-flash/r33-image020-tp4-sparkcache-20260911.md) | 8K | 3,274 | 51.4 | 132.5 |
| [R33 two-Spark pair](records/glm53-flash/r33-image020-tp2-sparkcache-20260911.md) | 8K | 2,340 | 33.1 | 66.1 |
