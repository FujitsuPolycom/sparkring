# Qwen3.8-27B EXL3 K5/K6 — two-Spark temperature-one results

![Green-text benchmark result](normalized-tp2-1m-probmtp-temp1-20260823.png)

## Conditions

Two directly cabled NVIDIA DGX Sparks ran the pinned Qwen3.8-27B EXL3 K5/K6 checkpoint at TP2/DCP1. The server used a 1,048,576-token static-YaRN limit, 32 maximum sequences, an 8,192-token scheduler budget, FP8 KV, requested block size 16 with effective 1,600-token hybrid alignment, asynchronous scheduling with full-input-length reservation, native prefix caching, full-decode CUDA graphs, and probabilistic Qwen MTP3 with standard rejection.

Benchmark requests set temperature 1.0 and left top-p/top-k unset. vLLM confirmed that the pinned `generation_config.json` supplied effective top-p 0.95 and top-k 20. Decode prompts were 100% unique between streams.

## Measurement

`llm-decode-bench` 0.4.31 measured client-observed prefill and sustained aggregate decode. Every included decode cell reached the requested concurrency with zero queued requests before timing. Client and isolated-server generated-token counters agreed. The first C32 campaign attempt was rejected after the old 600-second client read timeout dropped late streams; none of its values appear here.

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

The [machine-readable results](normalized-tp2-1m-probmtp-temp1-20260823.json) and [standalone HTML render](normalized-tp2-1m-probmtp-temp1-20260823.html) accompany this record.
