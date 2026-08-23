# DeepSeek-V4 base TP4 research performance

Lane: **public-functional**. Status: **research-only**.
Hardware: four directly cabled NVIDIA DGX Spark systems in a cycle, TP4/DCP1.
Evidence scope: one cache-disabled base recipe stack using image ID
`sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7`,
measured after the exact SparkCache TP4 stack on the same appliance. The order
was not randomized or bracketed, so reported differences remain research
signals rather than causal qualification evidence.

## Conditions

- Model checkpoint complete-manifest SHA-256:
  `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f`.
- Serving: 1,048,576-token model limit, 32 sequences, 8,192-token scheduler
  budget, 16 GiB KV per rank, `fp8_ds_mla`, DSpark K5, DCP1, no external KV
  connector.
- Endpoint diagnostics used `runtime/exl3-r7/endpoint_benchmark.py`, unique
  prompt nonces, exact-shape warmups, 128 generated tokens, semantic checks,
  and finite-logprob checks.
- Concurrency used a clean checkout of
  `local-inference-lab/llm-inference-bench` commit
  `0b4185b5b435e948b199c9077a00b084864aa963`, 16 measured plus one warmup
  request per C1/C2/C8 cell, 16,384 prompt tokens, 128 generated tokens,
  temperature zero, and observed engine pool `--kv-budget 1519925`.

## Measurement

Raw-record location: unavailable in a public artifact; endpoint, concurrency,
inspect, metric, and log artifacts remain in the private validation workspace.
Each endpoint shape ran once, and the concurrency campaign ran one finite cell
per concurrency, so no between-run variability or uncertainty interval is
available.

### Unique-prompt diagnostics

| Prompt tokens | TTFT | Prompt tokens / TTFT second | Inter-token decode |
|---:|---:|---:|---:|
| 1,024 | 0.469 s | 2,183.4 tok/s | 121.30 tok/s |
| 8,192 | 3.187 s | 2,570.4 tok/s | 123.18 tok/s |
| 32,768 | 12.437 s | 2,634.7 tok/s | 121.30 tok/s |
| 131,072 | 55.719 s | 2,352.38 tok/s | 121.30 tok/s |

The 131K exact-shape warmup took 56.156 seconds. Every cell completed and left
all four ranks running without OOM.

### Warm repeated-prefix 16K concurrency matrix

| Requested concurrency | Completed | Aggregate decode | Average TTFT | Average ITL |
|---:|---:|---:|---:|---:|
| 1 | 16/16 | 56.850 tok/s | 0.337 s | 15.03 ms |
| 2 | 16/16 | 82.380 tok/s | 0.452 s | 20.68 ms |
| 8 | 16/16 | 164.478 tok/s | 0.931 s | 38.54 ms |

All cells completed with zero errors, no capacity limitation, no warmup
timeout, no underfilled cell, and requested maximum running counts 1, 2, and 8.
As with the SparkCache matrix, repeated contexts warm vLLM's native in-process
prefix cache.

### Matched research signals

The SparkCache cells used the same model checkpoint, hardware, endpoint
harness shapes, and pinned concurrency harness inputs. They differed in model
limit, scheduler budget, KV reservation, connector/overlay state, and cache
history as defined by their recipes; this is a recipe-level comparison, not a
connector-only A/B.

| Cell | SparkCache TP4 | Base TP4 | Observed SparkCache change |
|---|---:|---:|---:|
| P1K TTFT | 0.531 s | 0.469 s | 13.2% slower |
| P8K TTFT | 3.500 s | 3.187 s | 9.8% slower |
| P32K TTFT | 14.156 s | 12.437 s | 13.8% slower |
| P131K TTFT | 61.453 s | 55.719 s | 10.3% slower |
| C1 aggregate decode | 59.222 tok/s | 56.850 tok/s | 4.2% faster |
| C2 aggregate decode | 83.180 tok/s | 82.380 tok/s | 1.0% faster |
| C8 aggregate decode | 134.571 tok/s | 164.478 tok/s | 18.2% slower |
| C8 average TTFT | 2.340 s | 0.931 s | 151.3% slower |

The warm C1/C2 changes are small, while the SparkCache C8 cell has lower
throughput and longer observed TTFT. Native prefix reuse dominates both
repeated-context matrices, and the visible records do not attribute the C8
difference to a specific queue or restore mechanism.

## Result

All four unique-prompt cells and all 48 measured concurrency requests passed
their bounded correctness and health gates. The matched tables show consistent
unique-prompt TTFT overhead for the SparkCache recipe and a lower C8 aggregate
rate under the ordered warm-prefix campaign.

## Conclusion

The cache-disabled TP4 recipe sustained correct prompt/decode service through
131K unique prompts and C8 finite concurrency. The ordered matched campaign is
useful as a recipe-level research signal but does not isolate connector cost.

## Limitations

- Any saved hardware summary predates exact measurement-boundary closure and
  is invalid for precise thermal or power attribution.
- The recipe settings differ beyond connector enablement; the comparison is
  not a single-variable A/B.
- Cache state evolved and the arms were not randomized or bracketed.
- Raw endpoint, concurrency, inspect, metric, and log artifacts remain in the
  private validation workspace. Raw-record location is unavailable in a public
  artifact. Each endpoint shape ran once, and the concurrency campaign ran one
  finite cell per concurrency, so no between-run variability or uncertainty
  interval is available.
