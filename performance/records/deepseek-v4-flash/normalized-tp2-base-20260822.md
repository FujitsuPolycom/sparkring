# DeepSeek-V4-Flash normalized TP2 base benchmark

| Field | Value |
|---|---|
| Lane | public-functional |
| Status | **live-validated candidate** — tested on hardware, still being tested, not qualified |
| Hardware | two directly cabled NVIDIA DGX Sparks, TP2/DCP1 |

These results apply only to the base recipe with a 1,048,576-token request limit,
32 sequences, 4,096 batched tokens, 16 GiB KV per rank, block size 256,
`fp8_ds_mla`, DSpark K5, async scheduling, full-input-length reservation, and
no SparkCache. The checkpoint revision was not pinned.

The benchmark used isolated-server sustained decode. Aggregate values are
generated tokens per second. Temperature 0 and 1.0 changed only temperature;
`top_p` was unset in both datasets.

## Prefill

| Context | Prompt tokens | TTFT | Prompt tok/s |
|---:|---:|---:|---:|
| 2K | 2,050 | 1.13 s | 1,822 |
| 8K | 8,194 | 4.27 s | 1,921 |
| 16K | 16,386 | 8.17 s | 2,005 |
| 32K | 32,770 | 16.39 s | 1,999 |
| 64K | 65,538 | 33.81 s | 1,938 |
| 128K | 131,074 | 72.52 s | 1,808 |

The first 128K pass compiled during measurement and was discarded; the table
uses the clean repeat.

## Sustained decode, temperature 0

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 52.07 | 109.22 | 152.54 | 179.96 | — | — |
| 8K | 67.12 | 107.87 | 133.52 | 204.88 | — | — |
| 16K | 52.69 | 128.28 | 152.31 | 223.53 | 308.54 | 444.89 |
| 32K | 73.57 | 115.12 | 146.70 | 159.73 | — | — |
| 64K | 76.55 | 121.83 | 142.97 | — | — | — |
| 128K | 54.02 | 85.34 | — | — | — | — |

Repetition summaries: 16K/C1 N=5 mean 62.97 tok/s (CV 15.26%); 16K/C8
N=3 mean 228.24 tok/s (CV 7.80%). The 16K/C32 entry is one accepted 240-second
measurement.

## Sustained decode, temperature 1.0

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 67.62 | 82.76 | 107.43 | 142.53 | — | — |
| 8K | 34.97 | 111.43 | 103.63 | 151.06 | — | — |
| 16K | 75.13 | 61.12 | 114.37 | 134.32 | 202.71 | 349.00 |
| 32K | 51.59* | 78.18 | 104.70 | 160.54 | — | — |
| 64K | 32.59 | 90.27 | 97.66 | — | — | — |
| 128K | 59.10 | 77.59 | — | — | — | — |

`*` 32K/C1 is the N=5 mean. Other repetition summaries: 16K/C1 N=5 mean
58.36 tok/s (CV 29.58%); 16K/C8 N=3 mean 164.93 tok/s (CV 17.23%).
Synthetic sustained-text throughput varies with DSpark acceptance, so these
single-cell differences are not transport or thermal verdicts.

## Temperature probes and coding workload

| Probe | C1 | C8 | C32 |
|---|---:|---:|---:|
| 16K, temperature 0.3, `top_p` unset | 51.10 | 176.71 | 406.76 |

| Coding Peak temperature | N | Mean tok/s | Median | Range |
|---:|---:|---:|---:|---:|
| 0 | 5 | 60.90 | 60.33 | 60.17–62.30 |
| 0.3 | 5 | 61.03 | 60.82 | 60.29–62.57 |
| 1.0 | 5 | 59.31 | 60.13 | 56.31–61.35 |

## Limits

- Rows named `DISCARD`, `ABORTED`, JIT-affected, or accounting-invalid in the
  operator archive are excluded.
- Saved speculative acceptance fields from the harness were last-scrape
  diagnostics, not run averages, and are not used here.
- Hardware summaries created before the measurement-boundary fix include a
  cancellation tail and are invalid for exact thermal interpretation. Decode
  token arithmetic remains usable.
- JSON and bounded rank logs remain in the maintainer-held operator archive;
  [the machine-readable summary](normalized-tp2-base-20260822.json) contains the
  accepted table values and exclusion policy.
