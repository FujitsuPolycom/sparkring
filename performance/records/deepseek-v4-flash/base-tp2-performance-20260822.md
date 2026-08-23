# DeepSeek-V4 base TP2 research performance

Lane: **public-functional**. Status: **research-only**.
Hardware: two directly cabled NVIDIA DGX Spark systems, TP2/DCP1. Evidence
scope: one cache-disabled base recipe deployment using image ID
`sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7`.
The base recipe does not pin a checkpoint revision or complete manifest, so
these measurements do not constitute a reproducible qualification.

## Conditions

- Model served name: `deepseek-v4-flash-0731`.
- Serving contract: 1,048,576-token request limit, 32 sequences, 8,192-token
  scheduler budget, 12 GiB KV per rank, `fp8_ds_mla`, DSpark K5.
- Endpoint diagnostics used `runtime/exl3-r7/endpoint_benchmark.py`, unique
  prompt nonces, 128 generated tokens, exact-shape warmups, semantic checks,
  and finite-logprob checks.
- The concurrency matrix used a fresh clean checkout of
  `local-inference-lab/llm-inference-bench` commit
  `0b4185b5b435e948b199c9077a00b084864aa963`, script version `0.4.29`, one
  warmup plus eight measured requests per cell, approximately 16K prompt
  tokens, 128 generated tokens, and temperature zero.
- The matrix used the observed engine pool explicitly as
  `--kv-budget 1139967`.

## Measurement

Raw-record location: unavailable in a public artifact; endpoint, concurrency,
inspect, and log artifacts remain in the private validation workspace. Each
endpoint shape ran once, and the concurrency campaign ran one finite cell per
concurrency, so no between-run variability or uncertainty interval is
available.

### Unique-prompt diagnostics

| Prompt tokens | TTFT | Prompt tokens / TTFT second | Inter-token decode |
|---:|---:|---:|---:|
| 1,024 | 0.781 s | 1,311.1 tok/s | 77.39 tok/s |
| 8,192 | 4.141 s | 1,978.3 tok/s | 77.44 tok/s |
| 32,768 | 15.907 s | 2,060.0 tok/s | 78.15 tok/s |
| 65,536 | 32.844 s | 1,995.4 tok/s | 78.15 tok/s |

Every cell generated 128 tokens and passed semantic, 96-value finite-logprob,
and post-run health checks.

### Finite 16K concurrency matrix

The exact prompt contained 16,250 tokens. Integrated prefill measured 8.13
seconds to first token and 1,999 prompt tokens per second.

| Requested concurrency | Completed | Aggregate decode | TTFT p50 | Average ITL | Request latency p50 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8/8 | 36.294 tok/s | 0.399 s | 24.58 ms | 3.563 s |
| 2 | 8/8 | 50.974 tok/s | 0.502 s | 34.00 ms | 4.719 s |
| 4 | 8/8 | 80.572 tok/s | 0.769 s | 42.65 ms | 6.230 s |
| 8 | 8/8 | 118.491 tok/s | 1.101 s | 53.07 ms | 7.653 s |

The saved speculative-acceptance field was a last-scrape diagnostic, not a
whole-window statistic, so it is not published as a run average.

All cells completed without request errors, capacity limits, or warmup
timeouts; requested maximum running counts 1, 2, 4, and 8 were reached.

## Result

All four unique-prompt cells and all 32 measured concurrency requests passed
their bounded correctness and health gates. Aggregate decode increased through
C8 without a harness capacity flag.

## Conclusion

The cache-disabled TP2 recipe sustained correct prompt/decode service through
65K unique prompts and C8 finite concurrency under the stated conditions. The
base recipe remains implemented, not qualified, because checkpoint identity is
not pinned by that recipe.

## Limitations

- Any saved hardware summary predates exact measurement-boundary closure and
  is invalid for precise thermal or power attribution.
- This finite request-count `0.4.29` record is not input-compatible with the
  repository's sustained-duration `0.4.31` comparator.
- The SparkCache TP2 matrix used six requests, C1/C2/C4/C6, and a 16,251-token
  prompt. This base matrix uses eight requests, C1/C2/C4/C8, and 16,250 tokens;
  the two records are not a formally paired A/B comparison.
- Raw-record location: unavailable in a public artifact; endpoint,
  concurrency, inspect, and log artifacts remain in the private validation
  workspace. Each endpoint shape ran once, and the concurrency campaign ran
  one finite cell per concurrency, so no between-run variability or
  uncertainty interval is available.
