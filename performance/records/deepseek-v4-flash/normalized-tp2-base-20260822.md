# DeepSeek-V4-Flash normalized TP2 base benchmark

| Field | Value |
|---|---|
| Lane | public-functional |
| Status | **live-validated candidate** — tested on hardware, not qualified |
| Hardware | two directly cabled NVIDIA DGX Sparks, TP2/DCP1 |
| Sampling | temperature 1.0, top-p 1.0 |

These results apply only to the cache-disabled base recipe with a
1,048,576-token request limit, 32 sequences, 4,096 batched tokens, 16 GiB KV
per rank, block size 256, `fp8_ds_mla`, DSpark K5, asynchronous scheduling,
and full-input-length reservation. The checkpoint revision was not pinned.

## Prefill

| Context | Prompt tokens | TTFT | Prompt tok/s |
|---:|---:|---:|---:|
| 2K | 2,050 | 1.13 s | 1,822 |
| 8K | 8,194 | 4.27 s | 1,921 |
| 16K | 16,386 | 8.17 s | 2,005 |
| 32K | 32,770 | 16.39 s | 1,999 |
| 64K | 65,538 | 33.81 s | 1,938 |
| 128K | 131,074 | 72.52 s | 1,808 |

The first 128K pass compiled during measurement and was discarded. The table
uses the clean repeat.

## Sustained decode

Aggregate generated tokens per second:

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 67.62 | 82.76 | 107.43 | 142.53 | — | — |
| 8K | 34.97 | 111.43 | 103.63 | 151.06 | — | — |
| 16K | 58.36* | 61.12 | 114.37 | 164.93† | 202.71 | 295.34 |
| 32K | 51.59‡ | 78.18 | 104.70 | 160.54 | — | — |
| 64K | 32.59 | 90.27 | 97.66 | — | — | — |
| 128K | 59.10 | 77.59 | — | — | — | — |

`*` N=5 mean, CV 29.58%. `†` N=3 mean, CV 17.23%. `‡` N=5 mean,
CV 29.14%. Other cells are one accepted observation. The 16K/C32 result uses
the isolated vLLM generation-token counter over a 120-second window; the
companion 32K/C32 attempt returned one request error and is excluded.

DSpark acceptance varies with the generated token path, so single-cell
differences are not transport or thermal verdicts.

## Coding workload

Coding Peak at temperature 1.0 completed five requests: mean 59.31 tok/s,
median 60.13, range 56.31–61.35 tok/s.

## Limits

- Rows named `DISCARD`, `ABORTED`, JIT-affected, accounting-invalid, or with
  request errors are excluded.
- Saved last-scrape acceptance fields are diagnostics, not run averages.
- Hardware summaries created before the measurement-boundary fix include a
  cancellation tail and are not used for precise thermal or power claims.
- Sanitized decode and Coding Peak receipts are published in the
  [receipt bundle](../../receipts/deepseek-v4-flash/temp1/). Endpoint bindings,
  SSH targets, local paths, event logs, and unreliable hardware summaries are
  removed; each file retains a public replay command and the private source
  receipt's SHA-256 digest.
- Raw rank logs and unsanitized JSON remain in the maintainer-held operator archive.
- Prefill TTFT values are retained because they end before sampled decode.
- [The machine-readable summary](normalized-tp2-base-20260822.json) contains
  the displayed values and exclusion policy.
