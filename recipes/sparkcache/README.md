# SparkCache composition recipes

These recipes add durable, rank-local prefix-state storage to the supported
SparkRing serving profiles. They are compositions: each file names a base
SparkRing recipe and records only the serving values, immutable artifacts,
cache policy, restart contract, evidence, and limitations qualified with
SparkCache.

| Composition | Parallelism | Candidate context / sequences | Historical qualified context / sequences | Scheduler budget |
|---|---:|---:|---:|---:|
| [`deepseek-v4-flash-0731-tp2-dcp1.json`](deepseek-v4-flash-0731-tp2-dcp1.json) | TP2/DCP1 | 1,048,576 / 32 | 131,072 / 6 | 4,096 |
| [`deepseek-v4-flash-0731-tp4-dcp1.json`](deepseek-v4-flash-0731-tp4-dcp1.json) | TP4/DCP1 | 1,048,576 / 32 | 524,288 / 32 | 4,096 |
| [`glm52-exl3-r7-3.5bpw-tp4-dcp4.json`](glm52-exl3-r7-3.5bpw-tp4-dcp4.json) | TP4/DCP4 | 1,048,576 / 16 | 262,144 / 8 | 4,096 |

The recipe objects are normalized candidates. The durable-state receipts below
remain qualified only for the historical context and sequence limits shown in
the table; they do not promote the candidate limits.

The DeepSeek compositions were qualified with SparkCache `0.1.0a1` wheel
SHA-256
`87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc`.
The GLM composition was qualified with SparkCache `0.1.0a2` wheel SHA-256
`3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b`.
Use the artifact named by the selected recipe and verify its hash on every rank
when reproducing a published qualification gate. Operators may use another
SparkCache build; the receipts on this page do not describe that artifact. The
connector module is `sparkcache.spark_context_cache_connector`; the
load-failure policy is `recompute`, so an unprovable restore becomes a cache
miss and serving does not consume unverified bytes.

Use a different rank-local host directory for every model stack and physical
rank. Mount those directories at the same container path across a stack. Do
not share one cache root between DeepSeek and GLM, between different
checkpoints, or between incompatible connector settings.

## Scheduler budget

`--max-num-batched-tokens 4096` is the scheduler budget used by the published
qualification gates for all three compositions; it is not a configuration
ceiling. Operators may change the value, and `8192` is known to work. The
receipts below cover `4096`; performance and capacity behavior at other values
is outside their evidence scope.

## Qualification evidence

Conditions: two or four directly cabled NVIDIA DGX Sparks, the immutable
runtime image and checkpoint identity in each composition, TP and DCP degrees
recorded in each file, dedicated rank-local cache roots, and the exact
SparkCache wheel identified by that composition.

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
| GLM TP4/DCP4 | 225,555 / 225,536 | 4,171.2, 3,588.4, 3,697.5, 3,172.2 ms | 225,536 | all-rank inventory prime returned `2`; final content `SPARKCACHE_OK:9540`; canary passed; reasoning-trace equality inconclusive |

Conclusion: the three exact compositions restore durable prefix state after
a coordinated engine restart and continue generating correct output. The
result qualifies the recorded artifacts and settings; it does not establish
general vLLM, checkpoint, topology, or production reliability.

Limitations: The published receipts use a 4,096-token scheduler budget; other
budgets are operator choices whose performance and capacity behavior is not
recorded here. The DeepSeek receipts cover DCP1, not DCP2 or DCP4. These
recipes disable streaming snapshots and native restore, so the receipts do not
cover either mode. Other images, checkpoints, and cache geometries are also
outside the recorded evidence. The GLM composition requires the 40-query-row
exact-state receipt preservation and regeneration procedure recorded in its
recipe before every coordinated restart. Full GLM reasoning-trace equality is
not a qualification condition because repeated requests varied while final
content and finish reason remained exact.
