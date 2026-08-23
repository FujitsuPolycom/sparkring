# Qwen3.8-27B EXL3 K5/K6 — four-Spark temperature-one results

![Green-text benchmark result](normalized-tp4-1m-probmtp-temp1-20260823.png)

## Conditions

| Field | Value |
|---|---|
| Lane | public-functional |
| Status | implemented; live-benchmarked |
| Hardware | four directly cabled NVIDIA DGX Sparks, TP4/DCP1 |
| Checkpoint | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Runtime image | `sha256:d1f3dc428c537e98f04bf6957f6bbccf77a4a51cab06b6a1cba36e543b285679` |
| Harness | `llm_decode_bench.py` 0.4.31; SHA-256 `07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851` |
| Serving limits | 1,048,576 tokens through static YaRN; 64 sequences; 8,192 batched tokens |
| KV and scheduling | FP8 KV; 1,600-token hybrid alignment; asynchronous scheduling; full-input-length reservation |
| Decode | native prefix caching; full-decode CUDA graphs; probabilistic Qwen MTP3 with standard rejection |
| Sampling | temperature 1.0; effective top-p 0.95 and top-k 20 from `generation_config.json` |
| Inputs | 2K–128K; C1/C2/C4/C8; 100% unique prompts |

## Measurement

Prefill is prompt tokens divided by client TTFT. Decode is the isolated vLLM `generation_tokens_total` delta over a monotonic-clock window.

Each decode cell waited for full concurrency, zero queue, and three seconds of stable state, followed by a 10-second decode warm-up. Durations and capacity timeouts are preserved in the receipts.

Repeated cells are arithmetic means; N appears in the table. Single-observation cells have no uncertainty estimate. Included cells passed alignment, request-error, timeout, capacity, and client/server token-agreement gates.

Raw records: [sanitized command receipts](../../receipts/qwen38-27b/temp1/20260823-tp4/).

## Result

Aggregate generated tokens per second:

| Context | Prefill N=3 | C1 N=5 | C2 N=3 | C4 N=1 | C8 N=1 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1,874.67 | 35.70 | 65.84 | 121.49 | 202.34 | — | — |
| 8K | 2,001.33 | 34.47 | 65.02 | 116.04 | 200.12 | — | — |
| 16K | 1,964.00 | 35.07 | 66.08 | 115.60 | 191.02 | — | — |
| 32K | 1,855.00 | 33.07 | 62.96 | 112.70 | 184.26 | — | — |
| 64K | 1,615.67 | 32.64 | 59.15 | 102.92 | 166.36 | — | — |
| 128K | 1,278.67 | 29.95 | 54.94 | 86.98 | 137.80 | — | — |

Coding Peak completed 15/15 requests: mean 48.46 tok/s, median 47.89, range 44.57–52.13 tok/s, with zero CJK-contaminated runs.

## Conclusion

The tested four-Spark profile is healthy through 128K prefill and C1/C2/C4/C8 sustained decode. Throughput scales from roughly 30–36 tok/s at C1 to 138–202 aggregate tok/s at C8, depending on context.

## Limitations

- N is shown in the result table. C16 and C32 were not run and are shown as dashes, not zero throughput.
- Rejected, timed-out, underfilled, or request-error attempts are excluded.

The [machine-readable results](normalized-tp4-1m-probmtp-temp1-20260823.json) and [HTML render](normalized-tp4-1m-probmtp-temp1-20260823.html) accompany this record.
