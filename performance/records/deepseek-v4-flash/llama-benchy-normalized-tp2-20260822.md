# DeepSeek-V4-Flash — two-Spark llama-benchy results

## Setup

| Field | Value |
|---|---|
| Profile | normalized two-Spark TP2/DCP1 base profile |
| Harness | `llama-benchy` 0.4.1.dev1 at `e9be344578cec17745066b220798b80a0d2686d3` |
| Workload | 2,048 prompt tokens; 128 generated tokens; prefix caching enabled |
| Context depths | 0, 4,096, 8,192, 16,384, 32,768, 65,535, 100,000 |
| Concurrency | 1, 2, 5, 10 |
| Repetitions | one warm-up plus three measured runs per test |
| Duration | 3:05:59 |
| API latency | 1.85 ms mean |
| Coherence test | passed |

The endpoint exposed `deepseek-ai/DeepSeek-V4-Flash-0731` as its served-model
name. The benchmark targeted the normalized TP2/DCP1 server used by the main
two-Spark campaign.

The CSV uses llama-benchy's standard shareable columns: aggregate and
per-request mean ± standard deviation, peak generation rate, time to first
response, estimated prompt-processing time, and end-to-end time to first token.
The matrices below are a compact view of its aggregate means.

## Cross-check against sustained decode bench

| Context | llama-benchy context load | llm-decode-bench prefill | Difference |
|---:|---:|---:|---:|
| 8K | 1,855 | 1,800 | +3.1% |
| 16K | 1,954 | 1,926 | +1.4% |
| 32K | 1,943 | 1,922 | +1.1% |
| 64K | 1,893 | 1,856 | +2.0% |

The prefill measurements closely agree. Decode measures a different workload:
llama-benchy sends finite 128-token follow-up requests over a loaded prefix,
while llm-decode-bench holds streams open for a sustained measurement window.
Do not average or directly substitute the two decode tables.

The run did not request exact output length, so 128 tokens was a maximum rather
than a guaranteed completion length. Its concurrency results describe short
request batches, not continuous C1-C32 saturation.

## Follow-up prompt processing

Aggregate `pp2048` tokens/s after loading the stated context depth:

| Depth | C1 | C2 | C5 | C10 |
|---:|---:|---:|---:|---:|
| 0 | 1,801.61 | 1,772.33 | 1,939.80 | 2,006.80 |
| 4,096 | 635.98 | 654.41 | 666.57 | 668.12 |
| 8,192 | 636.42 | 643.77 | 654.64 | 652.72 |
| 16,384 | 625.39 | 631.47 | 636.87 | 638.58 |
| 32,768 | 598.34 | 601.79 | 619.57 | 620.09 |
| 65,535 | 1,288.81 | 1,312.73 | 1,446.31 | 1,496.27 |
| 100,000 | 1,201.42 | 1,276.56 | 1,257.44 | 1,447.51 |

## Follow-up generation

Aggregate `tg128` tokens/s after loading the stated context depth:

| Depth | C1 | C2 | C5 | C10 |
|---:|---:|---:|---:|---:|
| 0 | 35.11 | 58.27 | 59.84 | 66.78 |
| 4,096 | 31.77 | 27.70 | 30.43 | 30.97 |
| 8,192 | 40.52 | 31.30 | 29.58 | 31.70 |
| 16,384 | 36.45 | 31.85 | 30.06 | 28.29 |
| 32,768 | 34.75 | 27.14 | 28.97 | 26.71 |
| 65,535 | 42.59 | 35.82 | 44.78 | 49.86 |
| 100,000 | 40.50 | 37.37 | 32.63 | 48.25 |

## Context-load prompt processing

Aggregate `ctx_pp` tokens/s for the prefix-cache setup request:

| Depth | C1 | C2 | C5 | C10 |
|---:|---:|---:|---:|---:|
| 4,096 | 1,775.73 | 1,853.93 | 1,983.66 | 2,001.36 |
| 8,192 | 1,855.41 | 1,962.72 | 2,001.13 | 2,002.14 |
| 16,384 | 1,953.53 | 1,963.58 | 1,990.09 | 1,974.14 |
| 32,768 | 1,942.52 | 1,958.87 | 1,958.38 | 1,947.33 |
| 65,535 | 1,892.99 | 1,891.98 | 1,894.20 | 1,886.28 |
| 100,000 | 1,820.94 | 1,824.80 | 1,804.87 | 1,819.87 |

## Context-load generation probe

Aggregate `ctx_tg` tokens/s for the prefix-cache setup request:

| Depth | C1 | C2 | C5 | C10 |
|---:|---:|---:|---:|---:|
| 4,096 | 36.67 | 38.53 | 41.75 | 40.64 |
| 8,192 | 29.41 | 34.26 | 25.16 | 26.96 |
| 16,384 | 41.66 | 20.82 | 15.29 | 13.83 |
| 32,768 | 44.63 | 11.51 | 8.84 | 6.70 |
| 65,535 | 33.09 | 6.55 | 3.82 | 3.32 |
| 100,000 | 38.87 | 3.79 | 2.73 | 2.44 |

## Source data

The [CSV](llama-benchy-normalized-tp2-20260822.csv) contains all 104 summary
rows, including means, standard deviations, per-request rates, peak generation
rates, and latency fields.

- CSV SHA-256: `8b85f95b839ed88196df12456f527b2d5465c7f8b65f3b6812185d875df1212a`
- Console log SHA-256: `cd7e91b328c74e440433b9ff61ed196be19d3a1e59d5b193b40ae6095b9ad069`
- Server log SHA-256: `6b83428d7cd75f21b7b508d621b2d9b1bec4c6e8ca7ad38b7ccdbf2965706915`

The console and server logs are not published because they contain local paths
and private network addresses.
