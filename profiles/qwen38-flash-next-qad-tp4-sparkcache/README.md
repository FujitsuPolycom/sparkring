# Qwen QAD TP4 with aligned SparkCache persistence

Status: **Experimental**. This selection enables the shared image's existing
Qwen hybrid connector; hardware restore and failed-restore recomputation are
not yet qualified. It does not change the cache-disabled TP4 default.

Complete the [QAD TP4 prerequisites](../qwen38-flash-next-qad-tp4/README.md).
Use the same pinned shared image, QAD checkpoint and managed fabric. Render
profile `qwen38-flash-next-qad-tp4-sparkcache` through the
[Compose procedure](../../docs/operations/compose.md), with a distinct deployment
name and a dedicated cache directory outside all model and source directories.
Never stop or replace another workload without its owner's approval.

The [configuration](../qwen38-flash-next-qad-tp4/sparkcache.json) retains
TP4/DCP1, MTP3, 262144 context, 16 sequences, an 8192-token scheduler budget,
24 GiB FP8 KV per rank, both Qwen feature hooks, and three-image/one-video limits.
It adds aligned checkpoint persistence with 4 GiB disk capacity per rank,
two 512 MiB capture slots and a 256 MiB restore budget per rank. Media limits are
configuration, not media qualification.

The cache selection requests 32-token attention blocks so the runtime rounds
physical hybrid pages to the persistent connector's 32-token chunk alignment.
The cache-disabled profile requests 16 tokens; its TP4 runtime produces
1424-token pages, which the connector correctly rejects as incompatible.

Checkpoint identity uses SHA-256 of the literal Hugging Face repository,
`@`, and pinned QAD revision. Its value and TP degree separate these entries
from the original TP2/PTQ composition. Existing cache identity wire values
and the TP2 fingerprint remain unchanged. Invalid restores recompute.

The API has no configured authentication; restrict it to trusted clients or
an authenticated gateway. Use the coordinator's inspected-ID stop procedure;
retain saved containers and caches for rollback. Do not treat cache-disabled
performance or the TP2 restore record as evidence for this TP4 selection.
