# SparkCache composition recipes

These recipes add durable, rank-local prefix-state storage to the supported
SparkRing serving profiles. They are compositions: each file names a base
SparkRing recipe and records only the serving values, immutable artifacts,
cache policy, restart contract, evidence, and limitations qualified with
SparkCache.

| Composition | Parallelism | Qualified context | Qualified scheduler budget |
|---|---:|---:|---:|
| [`deepseek-v4-flash-0731-tp2-dcp1.json`](deepseek-v4-flash-0731-tp2-dcp1.json) | TP2/DCP1 | 131,072 | 4,096 |
| [`deepseek-v4-flash-0731-tp4-dcp1.json`](deepseek-v4-flash-0731-tp4-dcp1.json) | TP4/DCP1 | 524,288 | 4,096 |
| [`glm52-exl3-r7-3.5bpw-tp4-dcp4.json`](glm52-exl3-r7-3.5bpw-tp4-dcp4.json) | TP4/DCP4 | 262,144 | 4,096 |

Every composition was qualified with SparkCache `0.1.0a1` wheel SHA-256
`87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc`.
Install that exact wheel into the runtime Python environment on every rank and
verify its hash before launch. The connector module is
`sparkcache.spark_context_cache_connector`; the load-failure policy is
`recompute`, so an unprovable restore becomes a cache miss and serving does not
consume unverified bytes.

Use a different rank-local host directory for every model stack and physical
rank. Mount those directories at the same container path across a stack. Do
not share one cache root between DeepSeek and GLM, between different
checkpoints, or between incompatible connector settings.

## Scheduler budget

`--max-num-batched-tokens 4096` is the conservative qualified default in all
three compositions. A value of `8192` is unsupported and is not an operator
option in these recipes. It can be added to a composition only after the exact
profile passes a separate cold-store, coordinated-restart, external-hit, and
post-restore canary smoke at that value.

## Qualification evidence

Conditions: two or four directly cabled NVIDIA DGX Sparks, the immutable
runtime image and checkpoint identity in each composition, TP and DCP degrees
recorded in each file, dedicated rank-local cache roots, and the exact
SparkCache wheel identified above.

Measurement: each gate sent a deterministic semantic prompt on a cold cache,
required every rank to commit the same reusable-token digest, restarted all
ranks without removing the cache roots, required manifest discovery on every
rank, repeated the identical prompt, read vLLM cache metrics, and sent a fresh
post-restore canary.

Result:

| Composition | Prompt / reusable tokens | Restore time by rank | External hits | Semantic gates |
|---|---:|---:|---:|---|
| DeepSeek TP2/DCP1 | 73,774 / 73,728 | 459.8, 517.0 ms | 73,728 | `SPARKCACHE_OK:9540`; canary passed |
| DeepSeek TP4/DCP1 | 73,774 / 73,728 | 483.9, 413.9, 443.0, 494.6 ms | 73,728 | `SPARKCACHE_OK:9540`; canary passed |
| GLM TP4/DCP4 | 225,555 / 225,536 | 3,954.2, 3,385.0, 3,649.6, 3,722.4 ms | 225,536 | quorum prime `2`; `SPARKCACHE_OK:9540`; canary passed |

Conclusion: the three exact compositions restore durable prefix state after
a coordinated engine restart and continue generating correct output. The
result qualifies the recorded artifacts and settings; it does not establish
general vLLM, checkpoint, topology, or production reliability.

Limitations: DeepSeek DCP2 and DCP4, streaming snapshots, native restore,
other scheduler budgets, other images, other checkpoints, and other cache
geometries are unsupported. The GLM composition also requires the Q40 receipt
preservation and regeneration procedure recorded in its recipe before every
coordinated restart.
