# GLM-5.2 base operator-image research performance

Lane: **public-functional**. Status: **research-only**.
Artifact status: **qualified operator image**. Hardware: four directly cabled
NVIDIA DGX Spark systems in a cycle, TP4/DCP4. Evidence scope: the
cache-disabled fixed-MTP4 operator recipe using image ID
`sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513`,
measured after its SparkCache composition on the same appliance. The arms were
not randomized or bracketed, so differences remain research signals.

## Conditions

- Model:
  `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`.
- Serving: TP4/DCP4 `ag_rs`, fixed MTP4, dynamic `nvfp4_ds_mla`, 9.25 GB KV
  per rank, 262,144-token model limit, eight sequences, 4,096-token scheduler
  budget, exact-Q40 runtime, no external KV connector.
- Endpoint diagnostics used `runtime/exl3-r7/endpoint_benchmark.py`, unique
  prompt nonces, exact-shape warmups, 128 generated tokens, semantic checks,
  and finite-logprob checks.
- Concurrency used a clean checkout of
  `local-inference-lab/llm-inference-bench` commit
  `0b4185b5b435e948b199c9077a00b084864aa963`, 16 measured plus one warmup
  request per C1/C2/C8 cell, 16,384 prompt tokens, 128 generated tokens,
  temperature zero, DCP4, and observed pool `--kv-budget 1156864`.

Every start preserved the exact-Q40 create-once receipt by exact path and SHA,
proved the source path absent, and required a fresh per-rank attestation before
health was accepted.

## Measurement

Raw-record location: unavailable in a public artifact; endpoint, concurrency,
transport, inspect, and log artifacts remain in the private validation
workspace. Each endpoint shape ran once, and the concurrency campaign ran one
finite cell per concurrency, so no between-run variability or uncertainty
interval is available.

### Unique-prompt diagnostics

| Prompt tokens | TTFT | Prompt tokens / TTFT second | Inter-token decode |
|---:|---:|---:|---:|
| 1,024 | 1.844 s | 555.31 tok/s | 33.31 tok/s |
| 8,192 | 13.328 s | 614.65 tok/s | 32.00 tok/s |
| 32,768 | 52.172 s | 628.08 tok/s | 32.00 tok/s |
| 131,072 | 212.313 s | 617.35 tok/s | 31.88 tok/s |

The 131K exact-shape warmup took 212.984 seconds. Every cell completed and
left all four ranks running without OOM.

### Warm repeated-prefix 16K concurrency matrix

| Requested concurrency | Completed | Aggregate decode | Average TTFT | Average ITL |
|---:|---:|---:|---:|---:|
| 1 | 16/16 | 21.343 tok/s | 1.212 s | 37.58 ms |
| 2 | 16/16 | 29.644 tok/s | 1.573 s | 55.33 ms |
| 8 | 16/16 | 49.403 tok/s | 5.796 s | 113.81 ms |

All cells completed with zero errors, no capacity limitation, no warmup
timeout, no underfilled cell, and requested maximum running counts 1, 2, and 8.
Post-run health, model identity, PIDs, and OOM state remained unchanged. The
all-rank transport audit passed.

### Matched research signals

The SparkCache endpoint cells used the same operator image, model, hardware,
serving geometry, and endpoint harness shapes. They differed in connector,
wheel, vLLM overlays, mounted runtime contract, and accumulated cache state.
The sequence was not randomized or bracketed.

| Prompt | SparkCache prefill | Base prefill | Observed SparkCache change | SparkCache decode | Base decode | Observed SparkCache change |
|---:|---:|---:|---:|---:|---:|---:|
| 1K | 590.5 tok/s | 555.31 tok/s | 6.3% faster | 34.01 tok/s | 33.31 tok/s | 2.1% faster |
| 8K | 636.3 tok/s | 614.65 tok/s | 3.5% faster | 31.39 tok/s | 32.00 tok/s | 1.9% slower |
| 32K | 630.53 tok/s | 628.08 tok/s | 0.4% faster | 28.13 tok/s | 32.00 tok/s | 12.1% slower |
| 131K | 641.43 tok/s | 617.35 tok/s | 3.9% faster | 29.45 tok/s | 31.88 tok/s | 7.6% slower |

The GLM SparkCache concurrency cells are not a valid A/B. External-cache
restore deferral produced positive queue samples, and the pinned harness marks
any positive queue sample as `capacity_limited=true`. The base matrix passed
that gate; the SparkCache matrix is retained only as research diagnostic
evidence.

## Result

All four unique-prompt cells and all 48 measured base concurrency requests
passed their bounded correctness, health, and transport gates. The matched
unique-prompt table shows mixed, shape-dependent differences for the
SparkCache recipe.

## Conclusion

The cache-disabled GLM operator recipe sustained correct prompt/decode service
through 131K unique prompts and C8 finite concurrency. The ordered matched
campaign is a recipe-level research signal, not a connector-only A/B.

## Limitations

- Any saved hardware summary predates exact measurement-boundary closure and
  is invalid for precise thermal or power attribution.
- The qualified operator image has no pullable immutable image reference.
- The matched sequence was not randomized or bracketed, and cache state
  evolved.
- Raw-record location: unavailable in a public artifact; endpoint,
  concurrency, transport, inspect, and log artifacts remain in the private
  validation workspace. Each endpoint shape ran once, and the concurrency
  campaign ran one finite cell per concurrency, so no between-run variability
  or uncertainty interval is available.
