# SparkRing benchmark results

All values are tokens per second from temperature 1.0 runs. Decode values are
aggregate throughput across all active streams.

## At a glance — 16K context

| Profile | Prefill | C1 decode | C8 decode | Highest tested concurrency | Coding Peak |
|---|---:|---:|---:|---:|---:|
| GLM-5.2 EXL3 3.5-bpw, four Sparks | 671 | 20.15 | 62.71 | C8: 62.71 | 25.39 |
| DeepSeek-V4-Flash DSpark, two Sparks | 1,926 | 58.36 | 162.69 | C32: 307.13 | 59.31 |
| DeepSeek-V4-Flash-0731, four Sparks | 2,488 | 68.84 | 265.16 | C32: 508.11 | 95.77 |
| Qwen3.8-27B EXL3 K5/K6, two Sparks | 1,367 | 29.50 | 142.20 | C8: 142.20 | 39.95 |
| Qwen3.8-27B EXL3 K5/K6, four Sparks | 1,964 | 35.07 | 191.02 | C8: 191.02 | 48.46 |

## Full matrices

| Profile | Results | Green matrix | Receipts |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw, four Sparks | [Full record](../performance/records/glm-3.5bpw/normalized-base-20260822.md) | [Coding Peak image](../performance/records/glm-3.5bpw/coding-peak-temperature1-20260822.png) | [Receipts](../performance/receipts/glm-3.5bpw/temp1/) |
| DeepSeek-V4-Flash DSpark, two Sparks | [Full record](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md) | [Matrix image](../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.png) | [Receipts](../performance/receipts/deepseek-v4-flash/temp1/) |
| DeepSeek-V4-Flash-0731, four Sparks | [Full record](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md) | [Matrix image](../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.png) | [Receipts](../performance/receipts/deepseek-v4-flash/temp1/20260823-tp4/) |
| Qwen3.8-27B EXL3 K5/K6, two Sparks | [Full record](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.md) | [Matrix image](../performance/records/qwen38-27b/normalized-tp2-1m-probmtp-temp1-20260823.png) | [Receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp2/) |
| Qwen3.8-27B EXL3 K5/K6, four Sparks | [Full record](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md) | [Matrix image](../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.png) | [Receipts](../performance/receipts/qwen38-27b/temp1/20260823-tp4/) |

## GLM 262,144-token setup

| Context | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | — | 679 | 673 | 666 | 657 | 645 |
| Decode, one stream | 22.6 | 22.0 | 21.3 | 20.4 | 21.4 | — |
| Decode, four streams | 50.3 | 51.9 | 49.2 | 45.6 | 47.2 | — |
| Decode, eight streams | 78.4 | 71.3 | 70.0 | 65.5 | 67.8 | — |

`71.3` is the 8K/C8 result. The 16K/C8 result is `70.0`. This captured
matrix does not record per-cell repeat counts, so these are displayed values,
not documented multi-run averages.

## Notes

- Dashes mean the cell was not measured or did not fit the tested KV pool.

- Qwen C16 and C32 were not measured.

- Use each full record for exact settings, sample counts, and limitations.
