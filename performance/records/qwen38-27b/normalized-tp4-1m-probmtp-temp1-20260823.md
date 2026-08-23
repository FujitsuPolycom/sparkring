# Qwen3.8-27B EXL3 K5/K6 — four-Spark temperature-one results

![Green-text benchmark result](normalized-tp4-1m-probmtp-temp1-20260823.png)

## Conditions

Four directly cabled NVIDIA DGX Sparks ran the pinned Qwen3.8-27B EXL3 K5/K6 checkpoint at TP4/DCP1. The server used a 1,048,576-token static-YaRN limit, 64 maximum sequences, an 8,192-token scheduler budget, FP8 KV, requested block size 16 with effective 1,600-token hybrid alignment, asynchronous scheduling with full-input-length reservation, native prefix caching, full-decode CUDA graphs, and probabilistic Qwen MTP3 with standard rejection.

Benchmark requests set temperature 1.0 and left top-p/top-k unset. vLLM confirmed that the pinned `generation_config.json` supplied effective top-p 0.95 and top-k 20. Decode prompts were 100% unique between streams.

## Measurement

`llm-decode-bench` 0.4.31 measured client-observed prefill and sustained aggregate decode. Every included decode cell reached the requested concurrency with zero queued requests before timing. Client and isolated-server generated-token counters agreed. A separate older C32 receipt came from a 262,144-token server and is intentionally excluded from this normalized 1,048,576-token record.

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

The [machine-readable results](normalized-tp4-1m-probmtp-temp1-20260823.json) and [standalone HTML render](normalized-tp4-1m-probmtp-temp1-20260823.html) accompany this record.
