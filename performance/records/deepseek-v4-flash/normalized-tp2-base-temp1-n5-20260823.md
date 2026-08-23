# DeepSeek-V4-Flash DSpark package — two Sparks, TP2/DCP1

## Conditions

| Field | Value |
|---|---|
| Lane | public-functional |
| Status | implemented; live-benchmarked |
| Hardware | two directly cabled NVIDIA DGX Sparks, TP2/DCP1 |
| Checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark@913f0657a874f76844e2e91cbe706dbcaceeb6d7` |
| Runtime image | `ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Harness | `llm_decode_bench.py` 0.4.31; SHA-256 `07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851` |
| Serving contract | 1,048,576-token request limit; 32 sequences; 4,096 batched tokens; 16 GiB KV/rank; block 256; `fp8_ds_mla`; DSpark K5 |
| Sampling | temperature 1.0, effective top-p 1.0 |
| Inputs | 2K–128K; C1/C2/C4/C8/C16/C32 where the KV pool fit; 100% unique prompts |

## Measurement

Prefill is prompt tokens divided by client TTFT. Decode divides an accepted token delta by a monotonic-clock window. The receipts identify the authority: OpenAI stream usage, Prometheus fallback, or the isolated vLLM generation counter.

Each decode cell waited for full concurrency, zero queue, and three seconds of stable state, followed by a 10-second decode warm-up. Cell durations and capacity timeouts vary by concurrency and are preserved in the receipts.

The tables report means and sample standard deviations from accepted cells. C1/C2 use at least five observations per context; other applicable decode cells use at least three. Included cells passed alignment, request-error, timeout, and capacity gates.

Raw records: [all sanitized TP2 command receipts](../../receipts/deepseek-v4-flash/temp1/).

## Result

### Prefill

| Context | Mean tok/s | SD | N | Mean TTFT s |
|---:|---:|---:|---:|---:|
| 2K | 1792.75 | 64.49 | 4 | 1.145 |
| 8K | 1800.00 | 108.40 | 4 | 4.566 |
| 16K | 1926.25 | 55.58 | 4 | 8.514 |
| 32K | 1922.25 | 66.22 | 4 | 17.062 |
| 64K | 1855.50 | 63.85 | 4 | 35.352 |
| 128K | 1691.25 | 83.30 | 4 | 77.644 |

### Sustained decode

Aggregate generated tokens per second, shown as mean ± SD (N):

| Context | C1 | C2 | C4 | C8 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|
| 2K | 62.53 ± 13.02 (5) | 75.79 ± 11.36 (5) | 106.07 ± 7.99 (3) | 144.05 ± 5.50 (4) | 201.26 ± 11.60 (3) | 275.26 ± 16.88 (3) |
| 8K | 48.42 ± 12.84 (5) | 89.67 ± 13.55 (5) | 110.90 ± 7.38 (3) | 156.68 ± 19.40 (4) | 217.37 ± 21.64 (3) | 299.66 ± 27.79 (4) |
| 16K | 58.36 ± 17.26 (5) | 77.65 ± 16.54 (5) | 104.16 ± 12.33 (3) | 162.69 ± 18.29 (6) | 202.74 ± 18.05 (4) | 307.13 ± 20.92 (5) |
| 32K | 51.59 ± 15.03 (5) | 85.05 ± 10.88 (5) | 107.13 ± 3.77 (3) | 147.40 ± 19.72 (4) | 223.25 ± 14.45 (3) | 301.00 ± 27.60 (3) |
| 64K | 50.06 ± 17.52 (5) | 76.57 ± 11.89 (5) | 108.41 ± 17.44 (3) | 154.57 ± 20.55 (3) | 205.27 ± 10.13 (3) | — |
| 128K | 53.05 ± 11.34 (5) | 73.82 ± 5.69 (5) | 86.43 ± 7.86 (3) | — | — | — |

### Coding Peak

Mean 59.31 tok/s; median 60.13; range 56.31–61.35; N=5.

## Conclusion

The two-Spark profile served through 128K prefill and every decode cell that fit its KV pool.

## Limitations

Combined 59 machine-readable receipts. JIT/server-log-rejected invocations are excluded. Request-error, timed-out, underfilled, capacity-limited, invalid, and non-positive rows are excluded; valid rows from otherwise mixed receipts are retained.

The DSpark and plain 0731 packages share model and tokenizer configuration and a tensor index, but their 48 weight payloads differ. TP2/TP4 comparisons therefore include both topology and package differences.
