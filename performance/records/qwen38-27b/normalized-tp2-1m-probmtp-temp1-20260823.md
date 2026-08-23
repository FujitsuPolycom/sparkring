# Qwen3.8-27B EXL3 K5/K6 — two-Spark results

![Green-text benchmark result](normalized-tp2-1m-probmtp-temp1-20260823.png)

## Conditions

| Field | Value |
|---|---|
| Lane | public-functional |
| Status | implemented; live-benchmarked |
| Hardware | two directly cabled NVIDIA DGX Sparks, TP2/DCP1 |
| Checkpoint | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Runtime image | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Harness | `llm_decode_bench.py` 0.4.31; SHA-256 `07aad353cd9c894e14e9d1392c8509d3af8999c4022d3d22b29423a4572f5851` |
| Serving limits | 1,048,576 tokens through static YaRN; 32 sequences; 8,192 batched tokens |
| KV and scheduling | FP8 KV; 1,600-token hybrid alignment; asynchronous scheduling; full-input-length reservation |
| Decode | native prefix caching; full-decode CUDA graphs; probabilistic Qwen MTP3 with standard rejection |
| Sampling | temperature 1.0; effective top-p 0.95 and top-k 20 from `generation_config.json` |
| Inputs | 2K–128K; C1/C2/C4/C8; 100% unique prompts |

## Measurement

Prefill is prompt tokens divided by client TTFT. Decode is the isolated vLLM `generation_tokens_total` delta over a monotonic-clock window.

Each decode cell waited for full concurrency, zero queue, and three seconds of stable state, followed by a 10-second decode warm-up. Durations and capacity timeouts are preserved in the receipts.

Repeated cells are arithmetic means; N appears in the table. Single-observation cells have no uncertainty estimate. Included cells passed alignment, request-error, timeout, capacity, and client/server token-agreement gates.

Raw records: [sanitized command receipts](../../receipts/qwen38-27b/temp1/20260823-tp2/).

## Result

Aggregate generated tokens per second:

| Context | Prefill N=3 | C1 N=4 | C2 N=1 | C4 N=1 | C8 N=1 | C16 | C32 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2K | 1,273.67 | 27.93 | 53.85 | 100.10 | 154.06 | — | — |
| 8K | 1,401.33 | 27.60 | 52.88 | 100.70 | 150.10 | — | — |
| 16K | 1,366.67 | 29.50 | 49.67 | 96.20 | 142.20 | — | — |
| 32K | 1,254.00 | 27.24 | 49.60 | 89.36 | 133.50 | — | — |
| 64K | 1,050.33 | 25.99 | 47.23 | 80.54 | 112.48 | — | — |
| 128K | 785.00 | 24.85 | 41.32 | 72.00 | 90.36 | — | — |

Coding Peak completed 15/15 requests: mean 39.95 tok/s, median 39.70, range 37.22–42.63 tok/s, with zero CJK-contaminated runs.

## Conclusion

The tested two-Spark profile is healthy through 128K prefill and C1/C2/C4/C8 sustained decode. Throughput scales from roughly 25–30 tok/s at C1 to 90–154 aggregate tok/s at C8, depending on context.

## Limitations

- N is shown in the result table. C16 and C32 were not run and are shown as dashes, not zero throughput.
- Rejected, timed-out, underfilled, or request-error attempts are excluded.

The [machine-readable results](normalized-tp2-1m-probmtp-temp1-20260823.json) and [HTML render](normalized-tp2-1m-probmtp-temp1-20260823.html) accompany this record.
