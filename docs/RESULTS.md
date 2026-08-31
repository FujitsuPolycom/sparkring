# SparkRing benchmark results

All values are tokens per second. Decode values are aggregate throughput across
all active streams.

## At a glance — 16K context

| Profile | Prefill | C1 decode | C8 decode | Highest tested concurrency | Coding Peak |
|---|---:|---:|---:|---:|---:|
| GLM-5.2 EXL3 3.5-bpw, four Sparks | 671 | 20.15 | 64.13 | C8: 64.13 | 25.39 |
| DeepSeek-V4-Flash DSpark, two Sparks | 1,926 | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| DeepSeek-V4-Flash-0731, four Sparks | 2,488 | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| Qwen3.8-27B EXL3 K5/K6, two Sparks | 1,367 | 29.50 | 142.20 | C8: 142.20 | 39.95 |
| Qwen3.8-27B EXL3 K5/K6, four Sparks | 1,964 | 35.07 | 191.02 | C8: 191.02 | 48.46 |

## Full matrices

| Profile | Results | Green matrix | Receipts |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw, four Sparks | [Full record](../performance/records/glm-3.5bpw/normalized-base-20260822.md) | [Matrix image](../performance/records/glm-3.5bpw/normalized-base-20260822.png) | [Receipts](../performance/receipts/glm-3.5bpw/temp1/) |
| DeepSeek-V4-Flash DSpark, two Sparks | [Full record](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | [Matrix image](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.png) | [Receipts](../performance/receipts/deepseek-v4-flash/temp1/) |
| DeepSeek-V4-Flash-0731, four Sparks | [Full record](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | [Matrix image](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.png) | [Receipts](../performance/receipts/deepseek-v4-flash/temp1/20260823-tp4/) |
| Qwen3.8-27B EXL3 K5/K6, two Sparks | [Full record](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | [Matrix image](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.png) | [Receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp2/) |
| Qwen3.8-27B EXL3 K5/K6, four Sparks | [Full record](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | [Matrix image](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.png) | [Receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp4/) |

The normalized DeepSeek two-Spark profile also completed a
[three-hour llama-benchy prefix-cache benchmark](../performance/records/deepseek-v4-flash/llama-benchy-normalized-tp2-20260822.md).

## Published JJ r7-compatible GLM-5.3 functional smoke evidence

The published base and SparkCache images are **implemented and TP4
smoke-verified**, not generally qualified. The base passed one exact C4
strong-oracle cohort. The SparkCache image passed fresh C4 publication and,
after full process replacement plus one readiness inference, four exact
external restores with a 100% measured hit ratio. This is functional evidence,
not a tokens-per-second benchmark. See the
[published-image quickstart](GLM53_JJ_R7_GB10_TP4_QUICKSTART.md),
[record](../performance/records/glm53-flash/jj-r7-gb10-tp4-smoke-20260830.md)
and
[machine receipt](../performance/receipts/glm53-flash/jj-r7-gb10-tp4-smoke-20260830/validation.json).

## GLM-5.2 EXL3 3.5-bpw — four Sparks

| Context | Prefill | C1 decode | C2 decode | C4 decode | C8 decode |
|---|---:|---:|---:|---:|---:|
| 2K | 694 | 22.00 | 28.28 | 46.98 | 65.35 |
| 8K | 675 | 19.15 | 30.21 | 47.70 | 64.46 |
| 16K | 671 | 20.15 | 32.38 | 45.38 | 64.13 |
| 32K | 661 | 21.61 | 30.52 | 46.08 | 65.79 |
| 64K | 649 | 20.17 | 30.12 | 45.52 | 63.58 |
| 128K | 635 | 19.67 | 30.64 | 45.73 | 62.63 |

The completed C8 and long-context cells are N=3 means. Other displayed cells
are single accepted observations. See the [full record](../performance/records/glm-3.5bpw/normalized-base-20260822.md).

## Notes

- Dashes mean the cell was not measured or did not fit the tested KV pool.

- Qwen TP2 C16 was measured through 64K; its 128K cell has N=2. Qwen TP2 C32 and Qwen TP4 C16/C32 were not measured.

- Use each full record for exact settings, sample counts, and limitations.
