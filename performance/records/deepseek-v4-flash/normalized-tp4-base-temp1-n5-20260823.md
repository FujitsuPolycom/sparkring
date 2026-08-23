# DeepSeek-V4-Flash-0731 — four Sparks, TP4/DCP1

| Field | Value |
|---|---|
| Lane | public-functional |
| Maturity | live-validated candidate; performance measured |
| Hardware | four directly cabled NVIDIA DGX Sparks, TP4/DCP1 |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Serving contract | 1,048,576-token request limit; 32 sequences; 4,096 batched tokens; 16 GiB KV/rank; block 256; `fp8_ds_mla`; DSpark K5 |
| Sampling | temperature 1.0, effective top-p 1.0 |
| Transport | patched NCCL over the four-Spark cycle; SIRCL disabled |

The tables report means from accepted, fully aligned measurements. C1/C2 use
five observations per context; other applicable decode cells use at least
three. TP2 used the separately packaged DSpark revision `913f0657…`; the
configuration and tensor index match, but the weight payloads do not. The two
tables are useful operational evidence but not an exact same-weight scaling
experiment.

Temperature 1.0; four directly cabled NVIDIA DGX Sparks, TP4/DCP1; at least 3 accepted repetitions per applicable cell, with at least 5 for C1/C2.

## Prefill

| Context | Mean tok/s | SD | N | Mean TTFT s |
|---:|---:|---:|---:|---:|
| 2K | 2343.00 | 0.00 | 3 | 0.875 |
| 8K | 2409.00 | 12.12 | 3 | 3.401 |
| 16K | 2488.33 | 18.18 | 3 | 6.586 |
| 32K | 2464.33 | 17.67 | 3 | 13.297 |
| 64K | 2389.33 | 6.35 | 3 | 27.432 |
| 128K | 2223.33 | 3.79 | 3 | 58.958 |

## Sustained decode

Aggregate generated tokens per second, shown as mean ± SD (N):

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 105.88 ± 25.67 (5) | 131.89 ± 24.57 (5) | 187.88 ± 23.46 (3) | 245.96 ± 37.65 (3) | 373.16 ± 25.12 (3) | 463.06 ± 20.00 (3) |
| 8K | 103.41 ± 20.28 (5) | 122.96 ± 37.19 (5) | 184.75 ± 42.58 (3) | 253.33 ± 26.89 (3) | 367.58 ± 24.83 (3) | 463.98 ± 25.52 (3) |
| 16K | 68.84 ± 20.05 (5) | 139.01 ± 19.49 (5) | 210.48 ± 29.31 (3) | 265.16 ± 20.24 (3) | 428.48 ± 24.26 (3) | 508.11 ± 17.35 (3) |
| 32K | 92.48 ± 31.07 (5) | 118.21 ± 9.14 (5) | 176.80 ± 51.90 (3) | 233.96 ± 27.22 (3) | 399.50 ± 19.70 (3) | 476.95 ± 12.40 (3) |
| 64K | 92.91 ± 21.04 (5) | 141.49 ± 28.91 (5) | 186.10 ± 30.43 (3) | 277.81 ± 10.75 (3) | 364.56 ± 15.50 (3) | — |
| 128K | 89.98 ± 25.96 (5) | 136.07 ± 35.57 (5) | 184.66 ± 27.47 (3) | 251.52 ± 10.86 (3) | — | — |

## Coding Peak

Mean 95.77 tok/s; median 95.15; range 89.47–100.44; N=15.

## Receipt scope

Combined 31 machine-readable temperature-1 receipts. JIT/server-log-rejected invocations are excluded. Request-error, timed-out, underfilled, capacity-limited, invalid, and non-positive rows are excluded; valid rows from otherwise mixed receipts are retained.
