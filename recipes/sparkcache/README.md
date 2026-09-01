# SparkCache composition recipes

These recipes add durable, rank-local prefix-state storage to the supported
SparkRing serving profiles. They are compositions: each file names a base
SparkRing recipe and records only the serving values, immutable artifacts,
cache policy, restart contract, evidence, and limitations qualified with
SparkCache.

| Composition | Status | Parallelism | Published profile context / sequences | Qualified receipt context / sequences | Scheduler budget |
|---|---|---:|---:|---:|---:|
| [`deepseek-v4-flash-0731-tp2-dcp1.json`](deepseek-v4-flash-0731-tp2-dcp1.json) | implemented | TP2/DCP1 | 1,048,576 / 32 | 131,072 / 6 | 4,096 |
| [`deepseek-v4-flash-0731-tp4-dcp1.json`](deepseek-v4-flash-0731-tp4-dcp1.json) | implemented | TP4/DCP1 | 1,048,576 / 32 | 524,288 / 32 | 4,096 |
| [`glm53-flash-nvfp4-dflash2-bf16-tp4.json`](glm53-flash-nvfp4-dflash2-bf16-tp4.json) | DCP1/DCP2 implemented; DCP4 qualified and preferred | TP4 with DCP1/DCP2/DCP4 | 1,048,576 / 16 | DCP4 publication through 124,928 stored tokens; restores through 999,424 tokens | 8,192 |
| [`glm52-exl3-r7-3.5bpw-tp4-dcp4.json`](glm52-exl3-r7-3.5bpw-tp4-dcp4.json) | implemented | TP4/DCP4 | 1,048,576 / 16 | 262,144 / 8 | 4,096 |

## Unsupported integrations

**Qwen3.8-27B EXL3 K5/K6: unsupported.** No SparkCache composition recipe or live
cache evidence is published for Qwen. The four-Spark base profile disables
external key-value caching, and the two-Spark base profile explicitly
omits LMCache to keep its 8,192-token scheduler budget.

The GLM-5.3 Flash composition uses one operator image for DCP1, DCP2, and
DCP4. Asynchronous publication is disabled in the DCP1 and DCP2 recipe
profiles because their capture-ring sizes are not live-qualified. DCP4 is the
preferred profile and enables two 3 GiB capture slots per rank.

The other three recipes are implemented at the published values in the table.
Their durable-state receipts qualify only the narrower context and sequence
limits shown in the receipt column.

The DeepSeek compositions were qualified with SparkCache `0.1.0a1` wheel
SHA-256
`87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc`.
The GLM composition was qualified with SparkCache `0.1.0a2` wheel SHA-256
`3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b`.
The GLM-5.3 Flash composition uses SparkCache commit
`c5dda75ec46bf235f6ece6e0d0174c1e41bd805a` and deployable-source SHA-256
`dffc2bead0a7c1cebb7a52757d38bd89146305b3ff351353ece9ac464c4c421d`
inside the immutable image named by its recipe. It uses cache profile
`glm53-flash-hybrid` and the vLLM ownership contract recorded by the operator
image.

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
| GLM-5.3 Flash DFlash2 TP4/DCP4 | 125,999 / 124,928 published; 1,000,000 / 999,424 restored | 2,276.2–2,395.2 ms for the 1M restore | 999,424 | exact 126K publication, 900K restore, and 1M restore needles passed |
| GLM TP4/DCP4 | 225,555 / 225,536 | 4,171.2, 3,588.4, 3,697.5, 3,172.2 ms | 225,536 | all-rank inventory prime returned `2`; final content `SPARKCACHE_OK:9540`; canary passed; reasoning-trace equality inconclusive |

Conclusion: the four exact compositions restore durable prefix state after
a coordinated engine restart and continue generating correct output. The
result qualifies the recorded artifacts and settings; it does not establish
general vLLM, checkpoint, topology, or production reliability.

Limitations: The published receipts use a 4,096-token scheduler budget; other
budgets are operator choices whose performance and capacity behavior is not
recorded here, except that the GLM-5.3 Flash receipt uses 8,192. The DeepSeek
receipts cover DCP1, not DCP2 or DCP4. The DeepSeek and GLM-5.2 recipes
disable streaming snapshots and CUDA restore. The GLM-5.3 recipe enables CUDA
restore but disables row-streaming snapshots. Other images, checkpoints, and
cache geometries are outside the recorded evidence.

The GLM-5.3 Flash receipt qualifies asynchronous publication only through
124,928 stored tokens and 231.8 MiB per rank. DCP1/DCP2 asynchronous capture,
larger asynchronous publication, page-tail publication, and concurrent
deep-context publication remain outside that evidence. The GLM-5.2
composition requires the 40-query-row
exact-state receipt preservation and regeneration procedure recorded in its
recipe before every coordinated restart. Full GLM reasoning-trace equality is
not a qualification condition because repeated requests varied while final
content and finish reason remained exact.
