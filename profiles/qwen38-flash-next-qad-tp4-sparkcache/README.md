# Qwen3.8-Flash-Next with SparkCache on four Sparks

Status: **Development**. This selection enables the published shared image's
Qwen hybrid connector. [Bounded text checks](../../performance/records/qwen38-flash-next/sparkcache-tp4.json)
passed cold/warm reuse, disk restore on all four ranks, and corrupted-object
rejection followed by correct recomputation. The cache-disabled TP4 default is unchanged.

Complete the [QAD TP4 prerequisites](../qwen38-flash-next-qad-tp4/README.md#prepare-image-model-and-fabric).
Use the same pinned shared image, QAD checkpoint and managed fabric. Render
profile `qwen38-flash-next-qad-tp4-sparkcache` through the
[Compose procedure](../../docs/operations/compose.md), with a distinct deployment
name and a dedicated cache directory outside all model and source directories.
Never stop or replace another workload without its owner's approval.

After preparing that private site:

```bash
python3 scripts/sparkring.py compose render qwen38-flash-next-qad-tp4-sparkcache \
  --site .sparkring/qwen-qad-cache.site.yaml --output .sparkring/deployments/qwen-qad-cache
python3 scripts/sparkring.py compose check --deployment .sparkring/deployments/qwen-qad-cache
```

Continue with the shared guide's host checks and reviewed start/stop procedure.

The [configuration](../qwen38-flash-next-qad-tp4/sparkcache.json) retains
TP4/DCP1, MTP3, 262K context, 16 sequences, an 8192-token scheduler budget,
24 GiB FP8 KV per rank, both Qwen feature hooks, and three-image/one-video limits.
It adds aligned checkpoint persistence with 4 GiB disk capacity per rank,
two 512 MiB capture slots and a 256 MiB restore budget per rank. Media limits are
configuration, not media qualification.

The cache selection requests 32-token attention blocks so the runtime rounds
physical hybrid pages to the persistent connector's 32-token chunk alignment.
The cache-disabled profile requests 16 tokens; its TP4 runtime produces
1424-token pages, which the connector correctly rejects as incompatible.

The test restored 7,200 cached tokens from an identical 7,860-token request.
One corrupted rank-zero object caused full recomputation with zero cached credit.
These are correctness checks, not performance, media or full-context qualification.

Cache entries are bound to the QAD checkpoint and TP degree. TP2 and TP4 use
separate cache roots and incompatible rank layouts even when their checkpoint
revision matches. Invalid restores recompute.

The API has no configured authentication; restrict it to trusted clients or
an authenticated gateway. Use the coordinator's inspected-ID stop procedure;
retain saved containers and caches for rollback. Do not treat cache-disabled
performance or the TP2 restore record as evidence for this TP4 selection.
