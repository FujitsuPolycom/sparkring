# DeepSeek-V4-Flash-0731 — four Sparks, TP4/DCP1

## Conditions

| Field | Value |
|---|---|
| Lane | public-functional |
| Status | implemented; live-benchmarked |
| Hardware | four directly cabled NVIDIA DGX Sparks, TP4/DCP1 |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Runtime image | `ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Harness | `llm_decode_bench.py` 0.4.31; SHA-256 `07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851` |
| Serving contract | 1,048,576-token request limit; 32 sequences; 4,096 batched tokens; 16 GiB KV/rank; block 256; `fp8_ds_mla`; DSpark K5 |
| Sampling | temperature 1.0, effective top-p 1.0 |
| Transport | patched NCCL over the four-Spark cycle; SIRCL disabled |
| Inputs | 2K–128K; C1/C2/C4/C8/C16/C32 where the KV pool fit; 100% unique prompts |

## Measurement

Prefill is prompt tokens divided by client TTFT. Decode is the isolated vLLM `generation_tokens_total` delta over a monotonic-clock window.

Each decode cell waited for full concurrency, zero queue, and three seconds of stable state, followed by a 10-second decode warm-up. Cell durations and capacity timeouts vary by concurrency and are preserved in the receipts.

The tables report means and sample standard deviations from accepted cells. C1/C2 use five observations per context; other applicable decode cells use at least three. Included cells passed alignment, request-error, timeout, and capacity gates.

Raw records: [sanitized command receipts](../../receipts/deepseek-v4-flash/temp1/20260823-tp4/).

## Result

### Prefill

| Context | Mean tok/s | SD | N | Mean TTFT s |
|---:|---:|---:|---:|---:|
| 2K | 2343.00 | 0.00 | 3 | 0.875 |
| 8K | 2409.00 | 12.12 | 3 | 3.401 |
| 16K | 2488.33 | 18.18 | 3 | 6.586 |
| 32K | 2464.33 | 17.67 | 3 | 13.297 |
| 64K | 2389.33 | 6.35 | 3 | 27.432 |
| 128K | 2223.33 | 3.79 | 3 | 58.958 |

### Sustained decode

Aggregate generated tokens per second, shown as mean ± SD (N):

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 105.88 ± 25.67 (5) | 131.89 ± 24.57 (5) | 187.88 ± 23.46 (3) | 245.96 ± 37.65 (3) | 373.16 ± 25.12 (3) | 463.06 ± 20.00 (3) |
| 8K | 103.41 ± 20.28 (5) | 122.96 ± 37.19 (5) | 184.75 ± 42.58 (3) | 253.33 ± 26.89 (3) | 367.58 ± 24.83 (3) | 463.98 ± 25.52 (3) |
| 16K | 68.84 ± 20.05 (5) | 139.01 ± 19.49 (5) | 210.48 ± 29.31 (3) | 265.16 ± 20.24 (3) | 428.48 ± 24.26 (3) | 508.11 ± 17.35 (3) |
| 32K | 92.48 ± 31.07 (5) | 118.21 ± 9.14 (5) | 176.80 ± 51.90 (3) | 233.96 ± 27.22 (3) | 399.50 ± 19.70 (3) | 476.95 ± 12.40 (3) |
| 64K | 92.91 ± 21.04 (5) | 141.49 ± 28.91 (5) | 186.10 ± 30.43 (3) | 277.81 ± 10.75 (3) | 364.56 ± 15.50 (3) | — |
| 128K | 89.98 ± 25.96 (5) | 136.07 ± 35.57 (5) | 184.66 ± 27.47 (3) | 251.52 ± 10.86 (3) | — | — |

### Coding Peak

Mean 95.77 tok/s; median 95.15; range 89.47–100.44; N=15.

## Conclusion

The four-Spark profile served through 128K prefill and every decode cell that fit its KV pool.

## Limitations

Combined 31 machine-readable temperature-1 receipts. JIT/server-log-rejected invocations are excluded. Request-error, timed-out, underfilled, capacity-limited, invalid, and non-positive rows are excluded; valid rows from otherwise mixed receipts are retained.

TP2 used the DSpark package at `913f0657…`. Its configuration and tensor index match TP4, but its weight payloads differ. TP2/TP4 comparisons therefore include both topology and package differences.
