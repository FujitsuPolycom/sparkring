# SparkCache composition recipes

These recipes add durable, rank-local prefix-state storage to the supported
SparkRing serving profiles. They are compositions: each file names a base
SparkRing recipe and records only the serving values, immutable artifacts,
cache policy, restart contract, evidence, and limitations qualified with
SparkCache.

The GLM-5.3 JSON in this directory is a historical exact-artifact recipe. The
published JJ r7-compatible GLM-5.3 image identities and operator environment are in
[`runtime/glm53-flash-jj-r7-gb10/`](../../runtime/glm53-flash-jj-r7-gb10/README.md).
The historical recipe remains here because its qualification evidence is bound
to its own digest and settings.

| Composition | Status | Parallelism | Published profile context / sequences | Qualified receipt context / sequences | Scheduler budget |
|---|---|---:|---:|---:|---:|
| [`deepseek-v4-flash-0731-tp2-dcp1.json`](deepseek-v4-flash-0731-tp2-dcp1.json) | implemented | TP2/DCP1 | 1,048,576 / 32 | 131,072 / 6 | 4,096 |
| [`deepseek-v4-flash-0731-tp4-dcp1.json`](deepseek-v4-flash-0731-tp4-dcp1.json) | implemented | TP4/DCP1 | 1,048,576 / 32 | 524,288 / 32 | 4,096 |
| [`glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json`](glm53-flash-nvfp4-dflash2-bf16-tp4-dcp1.json) | historical qualified artifact | TP4/DCP1 | 524,288 / 32 | 524,288 / 32 | 8,192 |
| [`glm52-exl3-r7-3.5bpw-tp4-dcp4.json`](glm52-exl3-r7-3.5bpw-tp4-dcp4.json) | implemented | TP4/DCP4 | 1,048,576 / 16 | 262,144 / 8 | 4,096 |

## Unsupported integrations

**Qwen3.8-27B EXL3 K5/K6: unsupported.** No SparkCache composition recipe or live
cache evidence is published for Qwen. The four-Spark base profile disables
external key-value caching, and the two-Spark base profile explicitly
omits LMCache to keep its 8,192-token scheduler budget.

The GLM-5.3 Flash recipe records the same 524,288-token and 32-sequence
geometry used by its qualification. The other three recipes are implemented
at the published values in the table. Their durable-state receipts qualify
only the narrower context and sequence limits shown in the receipt column.

The DeepSeek compositions were qualified with SparkCache `0.1.0a1` wheel
SHA-256
`87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc`.
The GLM composition was qualified with SparkCache `0.1.0a2` wheel SHA-256
`3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b`.
The GLM-5.3 Flash composition was qualified from SparkCache commit
`3860a2250193a6679ac6bac857af53e0757841f8` and source-tree SHA-256
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`
using cache profile `glm53-flash-hybrid` and the vLLM lease contract named by
its recipe. It did not use a published wheel.
Use the artifact named by the selected recipe and verify its hash on every rank
when reproducing a published qualification. Operators may use another
SparkCache build; the receipts on this page do not describe that artifact. The
connector module is `sparkcache.spark_context_cache_connector`; the
load-failure policy is `recompute`, so an unprovable restore becomes a cache
miss and serving does not consume unverified bytes.

Use a different rank-local host directory for every model stack and physical
rank. Mount those directories at the same container path across a stack. Do
not share one cache root between DeepSeek and GLM, between different
checkpoints, or between incompatible connector settings.

The GLM-5.3 Flash composition uses the public BF16
`incoai/GLM-5.3-Flash-DFlash2` checkpoint under CC BY-NC-ND 4.0. Its model
card limits use to research and evaluation and directs commercial licensing
inquiries to Inco AI.

## Scheduler budget

`--max-num-batched-tokens 4096` is the scheduler budget used by the DeepSeek
and GLM-5.2 qualifications; it is not a configuration ceiling. Their
receipts cover `4096`. The GLM-5.3 Flash qualification covers `8192`. Performance and
capacity behavior at another value is outside each receipt's evidence scope.

## Qualification evidence

Conditions: two or four directly cabled NVIDIA DGX Sparks, the immutable
runtime image and checkpoint identity in each composition, TP and DCP degrees
recorded in each file, dedicated rank-local cache roots, and the exact
SparkCache wheel or source-tree artifact identified by that composition.

Measurement: each validation sent a deterministic semantic prompt on a cold cache,
required every rank to commit the same reusable-token digest, restarted all
ranks without removing the cache roots, required manifest discovery on every
rank, repeated the identical prompt, read vLLM cache metrics, and sent a fresh
post-restore canary.

Result:

| Composition | Prompt / reusable tokens | Restore time by rank | External hits | Semantic checks |
|---|---:|---:|---:|---|
| DeepSeek TP2/DCP1 | 73,774 / 73,728 | 459.8, 517.0 ms | 73,728 | `SPARKCACHE_OK:9540`; canary passed |
| DeepSeek TP4/DCP1 | 73,774 / 73,728 | 483.9, 413.9, 443.0, 494.6 ms | 73,728 | `SPARKCACHE_OK:9540`; canary passed |
| GLM-5.3 Flash DFlash2 TP4/DCP1 | 8,215 / 8,192 | 155.6, 147.2, 194.0, 151.8 ms | 8,192 | `SPARKCACHE_GLM53_OK`; canary passed; 301 draft tokens from 43 drafts |
| GLM TP4/DCP4 | 225,555 / 225,536 | 4,171.2, 3,588.4, 3,697.5, 3,172.2 ms | 225,536 | all-rank inventory prime returned `2`; final content `SPARKCACHE_OK:9540`; canary passed; reasoning-trace equality inconclusive |

Conclusion: the four exact compositions restore durable prefix state after
a coordinated engine restart and continue generating correct output. The
result qualifies the recorded artifacts and settings; it does not establish
general vLLM, checkpoint, topology, or production reliability.

Limitations: The published receipts use a 4,096-token scheduler budget; other
budgets are operator choices whose performance and capacity behavior is not
recorded here, except that the GLM-5.3 Flash receipt uses 8,192. The DeepSeek
receipts cover DCP1, not DCP2 or DCP4. These
recipes disable streaming snapshots and SparkCache CUDA restore, so the receipts do not
cover either mode. Other images, checkpoints, and cache geometries are also
outside the recorded evidence. The GLM-5.3 Flash receipt covers an 8,192-token
restored span and does not establish throughput neutrality or larger-span
restore performance. The GLM-5.2 composition requires the 40-query-row
exact-state receipt preservation and regeneration procedure recorded in its
recipe before every coordinated restart. Full GLM reasoning-trace equality is
not a qualification condition because repeated requests varied while final
content and finish reason remained exact.
