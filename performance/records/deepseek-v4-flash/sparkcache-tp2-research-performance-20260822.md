# DeepSeek-V4 SparkCache TP2 research performance

Lane: **public-functional**. Status: **research-only**. Hardware:
two directly cabled NVIDIA DGX Spark systems,
TP2/DCP1. Evidence scope: one SparkCache-enabled stack using the separately
identified checkpoint manifest
`905dca196ad0495e86557df98fb55a35ab8ae3b4a7fbe84a2696382a070e8a78`.
These measurements do not reproduce or qualify the recipe's `bd6b0117...`
checkpoint composition and are not comparable to another model, harness
version, topology, or cache state.

## Conditions

- Serving image ID:
  `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7`.
- SparkCache `0.1.0a1` wheel SHA-256:
  `87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc`.
- SparkRing TP2 launcher SHA-256:
  `7253d6adfcd3b2310f3975fa2f8bd89ac01870562cf39c6516a2d7a73b3b5afa`.
- Serving contract: 131,072-token model limit, six sequences, 4,096-token
  scheduler budget, 16 GiB KV per rank, DSpark K5, DCP1, persistent
  rank-local SparkCache enabled.
- Endpoint diagnostics used `runtime/exl3-r7/endpoint_benchmark.py`, a unique
  prompt nonce, 128 generated tokens, semantic warmup, exact-shape warmup, and
  finite-logprob checks. Unique prompts were external-cache misses.
- The concurrency matrix used a clean checkout of
  `local-inference-lab/llm-inference-bench` commit
  `0b4185b5b435e948b199c9077a00b084864aa963`, script version `0.4.29`, one
  warmup plus six measured requests per cell, 16,251 actual prompt tokens,
  128 generated tokens, and temperature zero.

The concurrency harness incorrectly derived a 63,796-token capacity from HMA
block counts. No cell ran under that value. The executed matrix passed the
observed engine pool explicitly as `--kv-budget 529634`.

## Measurement

Raw-record location: unavailable in a public artifact; endpoint and concurrency
JSON remain in the private validation workspace. Each endpoint shape ran once,
and the concurrency campaign ran one finite cell per concurrency, so no
between-run variability or uncertainty interval is available.

### Prompt and decode diagnostics

| Prompt tokens | TTFT | Prompt tokens / TTFT second | Inter-token decode |
|---:|---:|---:|---:|
| 1,024 | 0.797 s | 1,284.8 tok/s | 75.96 tok/s |
| 8,192 | 4.547 s | 1,801.6 tok/s | 77.39 tok/s |
| 32,768 | 17.000 s | 1,927.5 tok/s | 76.69 tok/s |
| 65,536 | 33.672 s | 1,946.3 tok/s | 71.91 tok/s |

Every cell generated 128 tokens, passed its semantic and finite-logprob checks,
and left rank-0 health at HTTP 200.

### Finite 16K concurrency matrix

Integrated 16K prefill measured 2,018 prompt tokens per second. The displayed
8.05-second TTFT and 16,251-token count are rounded summaries; the rate comes
from the unrounded harness values.

| Requested concurrency | Completed | Aggregate decode | TTFT p50 | Average ITL | Effective average concurrency |
|---:|---:|---:|---:|---:|---:|
| 1 | 6/6 | 36.39 tok/s | 0.397 s | 24.55 ms | 1.0 |
| 2 | 6/6 | 53.93 tok/s | 0.478 s | 31.63 ms | 1.7 |
| 4 | 6/6 | 77.46 tok/s | 0.721 s | 35.15 ms | 2.7 |
| 6 | 6/6 | 83.97 tok/s | 0.970 s | 59.02 ms | 4.9 |

The saved speculative-acceptance field was a last-scrape diagnostic, not a
whole-window statistic, so it is not published as a run average.

All cells completed without request errors, capacity skips, or warmup timeouts.
The finite request-count `0.4.29` record is not input-compatible with the
repository's sustained-duration `0.4.31` comparator.

## Result

All four unique-prompt cells and all 24 measured concurrency requests passed
their bounded correctness and health gates. Aggregate decode rose through C6,
while average ITL and TTFT also increased with concurrency.

## Conclusion

The TP2 SparkCache research stack sustained correct prompt/decode service
through 65K unique prompts and C6 finite concurrency on the stated checkpoint.
The record does not establish a connector-only performance benefit.

## Limitations

Any saved hardware summary predates exact measurement-boundary closure and is
invalid for precise thermal or power attribution.

The matrix ended with 16 manifests and approximately 1.63 GB of files in each
rank-local cache root. The roots were preserved. In-process native prefix state
and persistent SparkCache state evolved across the ordered cells; this is not a
cold-versus-warm causal comparison. Store/restart/restore evidence is recorded
separately in
[`sparkcache-tp2-launcher-research-validation-20260822.md`](sparkcache-tp2-launcher-research-validation-20260822.md).

Raw-record location: unavailable in a public artifact; endpoint and concurrency
JSON remain in the private validation workspace. Each endpoint shape ran once,
and the concurrency campaign ran one finite cell per concurrency, so no
between-run variability or uncertainty interval is available.
