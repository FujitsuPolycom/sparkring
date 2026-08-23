# GLM-5.2 SparkCache operator-image performance

Lane: **public-functional**. Status: **research-only**. Artifact
status: **qualified operator image**. Hardware: four directly cabled NVIDIA
DGX Spark systems in a cycle, TP4/DCP4. Evidence scope: one SparkCache
`0.1.0a2` stack using the qualified maintainer/operator image
`sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513`.
The prompt/decode cells are diagnostic measurements. The concurrency matrix is
not promotion evidence because the pinned harness marked every cell capacity
limited.

## Conditions

- Model:
  `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`.
- Serving: TP4/DCP4 `ag_rs`, fixed MTP4, dynamic `nvfp4_ds_mla`, 9.25 GB KV
  per rank, 262,144-token model limit, eight sequences, 4,096-token scheduler
  budget, exact-Q40 runtime.
- SparkCache wheel SHA-256:
  `3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b`.
- A receipt-preserving restart restored 225,536 external-cache tokens with zero
  native-prefix hits before the matrix. The corrected final-content semantic
  gate passed with exact content, finish reason `stop`, arithmetic
  inventory-publication response `2`, and `SPARKCACHE_CANARY_OK`. Separate
  all-rank metrics passed. Reasoning-body equality was false and
  remained diagnostic-only.
- Exact semantic-gate artifact scope and results are recorded in
  [`sparkcache-semantic-gate-validation-20260822.md`](sparkcache-semantic-gate-validation-20260822.md).
- Endpoint cells used `runtime/exl3-r7/endpoint_benchmark.py`, unique prompt
  nonces, exact-shape warmups, 128 generated tokens, and finite-logprob checks.
- Concurrency cells used a fresh clean checkout of
  `local-inference-lab/llm-inference-bench` commit
  `0b4185b5b435e948b199c9077a00b084864aa963`, 16 measured plus one warmup
  request per cell, 16,384 prompt tokens, 128 generated tokens, and temperature
  zero.

## Measurement

Raw-record location: unavailable in a public artifact; endpoint, concurrency,
transport, and host artifacts remain in the private validation workspace. Each
endpoint shape ran once, and the concurrency campaign ran one finite cell per
concurrency, so no between-run variability or uncertainty interval is
available.

### Unique-prompt diagnostics

| Prompt tokens | TTFT | Prompt tokens / TTFT second | Inter-token decode | Measured wall |
|---:|---:|---:|---:|---:|
| 1,024 | 1.734 s | 590.5 tok/s | 34.01 tok/s | — |
| 8,192 | 12.875 s | 636.3 tok/s | 31.39 tok/s | — |
| 32,768 | 51.969 s | 630.53 tok/s | 28.13 tok/s | 56.484 s |
| 131,072 | 204.343 s | 641.43 tok/s | 29.45 tok/s | 208.656 s |

The 131K exact-shape warmup took 202.375 seconds. It committed a 130,816-token
entry before the separate nonce-bearing measured request. Every cell completed,
and health/OOM checks remained clean.

### Warm external-cache 16K concurrency matrix

After the first identical 16K context request, SparkCache restored 16,128
tokens in approximately 133–137 ms on repeated requests. The table therefore
describes warm external-cache decode, not cold prefill.

| Requested concurrency | Completed | Aggregate decode | Average TTFT | Average ITL | Harness capacity flag |
|---:|---:|---:|---:|---:|---|
| 1 | 16/16 | 21.261 tok/s | 1.082 s | 38.83 ms | limited |
| 2 | 16/16 | 30.246 tok/s | 1.558 s | 53.79 ms | limited |
| 8 | 16/16 | 58.896 tok/s | 4.399 s | 97.99 ms | limited |

All 48 measured requests returned HTTP 200 and exactly 128 output tokens, with
zero request errors, no warmup timeout, no underfilled cell, and requested
maximum running counts 1, 2, and 8. The pinned harness sets
`capacity_limited=true` whenever any queue sample is positive. Average queued
requests were 0.2, 0.3, and 1.6 respectively. External-cache restore defers a
hit while the connector installs state, so this flag does not prove KV-capacity
exhaustion. It nevertheless fails the published promotion condition and hides
the harness headline result. The table is therefore research-only diagnostic
output.

### Transport and health

Post-run health and model identity were unchanged. The all-rank transport audit
passed: each rank recorded the same all-reduce delta of 179,439 and vocabulary
delta of 4,388, with no fatal or overflow condition reported.

## Result

All four unique-prompt cells passed. All 48 concurrency requests completed
correctly, but the pinned harness marked every concurrency cell capacity
limited because external-cache deferral produced positive queue samples. The
transport and post-health gates passed.

## Conclusion

The GLM SparkCache operator stack sustained correct prompt/decode service
through 131K unique prompts. The concurrency campaign is retained as
research-only diagnostic output and does not satisfy promotion conditions.

## Limitations

- Any saved hardware summary predates exact measurement-boundary closure and
  is invalid for precise thermal or power attribution.
- The qualified operator image has no pullable immutable image reference, so
  this result cannot be reproduced from the SparkRing composition alone.
- Cache state evolved across ordered cells. The endpoint cells use unique
  prompts; the concurrency cells intentionally reuse one exact context.
- The positive-queue heuristic makes the concurrency record invalid for
  promotion even though every request completed.
- Raw-record location: unavailable in a public artifact; endpoint,
  concurrency, transport, and host artifacts remain in the private validation
  workspace. Each endpoint shape ran once, and the concurrency campaign ran
  one finite cell per concurrency, so no between-run variability or
  uncertainty interval is available.
