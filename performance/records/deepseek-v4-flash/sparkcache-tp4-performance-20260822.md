# DeepSeek-V4 SparkCache TP4 performance

Lane: **public-functional**. Status: **research-only**.
Hardware: four directly cabled NVIDIA DGX Spark systems in a cycle,
TP4/DCP1. Evidence scope: the exact public-artifact SparkCache TP4 composition
recorded in
[`sparkcache-tp4-public-reproduction-20260822.md`](sparkcache-tp4-public-reproduction-20260822.md).
The measurements do not transfer to another image, checkpoint, cache state,
topology, or harness.

## Conditions

- Serving image ID:
  `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7`.
- Checkpoint complete-manifest SHA-256:
  `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f`.
- SparkCache `0.1.0a1` wheel SHA-256:
  `87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc`.
- Serving: 524,288-token model limit, 32 sequences, 4,096-token scheduler
  budget, 32 GiB KV per rank, `fp8_ds_mla`, DSpark K5, DCP1, persistent
  rank-local SparkCache enabled.
- A known 73,728-token external restore passed before measurement.
- Endpoint diagnostics used `runtime/exl3-r7/endpoint_benchmark.py`, unique
  prompt nonces, exact-shape warmups, 128 generated tokens, semantic checks,
  and finite-logprob checks.
- Concurrency used a clean checkout of
  `local-inference-lab/llm-inference-bench` commit
  `0b4185b5b435e948b199c9077a00b084864aa963`, 16 measured plus one warmup
  request per C1/C2/C8 cell, 16,384 prompt tokens, 128 generated tokens,
  temperature zero, DCP1, and the observed engine pool
  `--kv-budget 4382668`.

## Measurement

Raw-record location: unavailable in a public artifact; endpoint, concurrency,
inspect, metric, and log artifacts remain in the private validation workspace.
Each endpoint shape ran once, and the concurrency campaign ran one finite cell
per concurrency, so no between-run variability or uncertainty interval is
available.

### Unique-prompt diagnostics

| Prompt tokens | TTFT | Prompt tokens / TTFT second | Inter-token decode |
|---:|---:|---:|---:|
| 1,024 | 0.531 s | 1,928.4 tok/s | 119.47 tok/s |
| 8,192 | 3.500 s | 2,340.6 tok/s | 121.30 tok/s |
| 32,768 | 14.156 s | 2,314.8 tok/s | 114.41 tok/s |
| 131,072 | 61.453 s | 2,132.88 tok/s | 100.40 tok/s |

The 131K exact-shape warmup took 62.719 seconds. Every cell completed and left
all four ranks running without OOM.

### Warm repeated-prefix 16K concurrency matrix

| Requested concurrency | Completed | Aggregate decode | Average TTFT | Average ITL |
|---:|---:|---:|---:|---:|
| 1 | 16/16 | 59.222 tok/s | 0.339 s | 14.32 ms |
| 2 | 16/16 | 83.180 tok/s | 0.499 s | 19.92 ms |
| 8 | 16/16 | 134.571 tok/s | 2.340 s | 39.56 ms |

The saved speculative-acceptance field was a last-scrape diagnostic, not a
whole-window statistic, so it is not published as a run average.

All cells completed with zero errors, `capacity_limited=false`, no warmup
timeout, no underfilled cell, and requested maximum running counts 1, 2, and 8.

The repeated-context cells were dominated by vLLM's in-process native prefix
cache after the first request. End-of-run metrics contained 891,648 native-hit
tokens and 81,920 external-hit tokens; the latter includes the separate known
73,728-token SparkCache semantic restore. Therefore the table is warm
repeated-prefix decode evidence, not a measurement of repeated external-cache
restore performance. The unique-nonce endpoint cells above were cold for their
measured prompt content.

## Result

All four unique-prompt cells and all 48 measured concurrency requests passed
their bounded correctness and health gates. The warm repeated-prefix matrix had
no harness capacity flag, but native prefix reuse dominated its cache state.

## Conclusion

The exact SparkCache TP4 composition sustained correct prompt/decode service
through 131K unique prompts and C8 finite concurrency. The record does not
isolate external-cache restore throughput from native prefix reuse.

## Limitations

- Any saved hardware summary predates exact measurement-boundary closure and
  is invalid for precise thermal or power attribution.
- Cache state evolved across the ordered matrix.
- The matching cache-disabled run is recorded in
  [`base-tp4-performance-20260822.md`](base-tp4-performance-20260822.md), but
  the ordered arms were not randomized or bracketed, their recipe settings
  differ, and native-prefix/cache history prevents a connector-only A/B claim.
- Raw endpoint, concurrency, inspect, metric, and log artifacts remain in the
  private validation workspace. Raw-record location is unavailable in a public
  artifact. Each endpoint shape ran once, and the concurrency campaign ran one
  finite cell per concurrency, so no between-run variability or uncertainty
  interval is available.
