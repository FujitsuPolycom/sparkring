# DeepSeek-V4-Flash DSpark package — two Sparks, TP2/DCP1

| Field | Value |
|---|---|
| Lane | public-functional |
| Maturity | live-validated candidate; performance measured |
| Hardware | two directly cabled NVIDIA DGX Sparks, TP2/DCP1 |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark@913f0657a874f76844e2e91cbe706dbcaceeb6d7` |
| Serving contract | 1,048,576-token request limit; 32 sequences; 4,096 batched tokens; 16 GiB KV/rank; block 256; `fp8_ds_mla`; DSpark K5 |
| Sampling | temperature 1.0, effective top-p 1.0 |

The tables report means from accepted, fully aligned measurements. C1/C2 use
at least five observations per context; other applicable decode cells use at
least three. This DSpark package has the same model configuration, tokenizer
configuration, and weight index as the plain 0731 package used by TP4, but all
48 safetensors payload identifiers differ. Treat TP2-versus-TP4 comparisons as
practical topology evidence, not an exact same-weight scaling experiment.

Temperature 1.0; two directly cabled NVIDIA DGX Sparks, TP2/DCP1; at least 3 accepted repetitions per applicable cell, with at least 5 for C1/C2.

## Prefill

| Context | Mean tok/s | SD | N | Mean TTFT s |
|---:|---:|---:|---:|---:|
| 2K | 1792.75 | 64.49 | 4 | 1.145 |
| 8K | 1800.00 | 108.40 | 4 | 4.566 |
| 16K | 1926.25 | 55.58 | 4 | 8.514 |
| 32K | 1922.25 | 66.22 | 4 | 17.062 |
| 64K | 1855.50 | 63.85 | 4 | 35.352 |
| 128K | 1691.25 | 83.30 | 4 | 77.644 |

## Sustained decode

Aggregate generated tokens per second, shown as mean ± SD (N):

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 62.53 ± 13.02 (5) | 75.79 ± 11.36 (5) | 106.07 ± 7.99 (3) | 144.05 ± 5.50 (4) | 201.26 ± 11.60 (3) | 275.26 ± 16.88 (3) |
| 8K | 48.42 ± 12.84 (5) | 89.67 ± 13.55 (5) | 110.90 ± 7.38 (3) | 156.68 ± 19.40 (4) | 217.37 ± 21.64 (3) | 299.66 ± 27.79 (4) |
| 16K | 58.36 ± 17.26 (5) | 77.65 ± 16.54 (5) | 104.16 ± 12.33 (3) | 162.69 ± 18.29 (6) | 202.74 ± 18.05 (4) | 307.13 ± 20.92 (5) |
| 32K | 51.59 ± 15.03 (5) | 85.05 ± 10.88 (5) | 107.13 ± 3.77 (3) | 147.40 ± 19.72 (4) | 223.25 ± 14.45 (3) | 301.00 ± 27.60 (3) |
| 64K | 50.06 ± 17.52 (5) | 76.57 ± 11.89 (5) | 108.41 ± 17.44 (3) | 154.57 ± 20.55 (3) | 205.27 ± 10.13 (3) | — |
| 128K | 53.05 ± 11.34 (5) | 73.82 ± 5.69 (5) | 86.43 ± 7.86 (3) | — | — | — |

## Coding Peak

Mean 59.31 tok/s; median 60.13; range 56.31–61.35; N=5.

## Receipt scope

Combined 59 machine-readable temperature-1 receipts. JIT/server-log-rejected invocations are excluded. Request-error, timed-out, underfilled, capacity-limited, invalid, and non-positive rows are excluded; valid rows from otherwise mixed receipts are retained.
