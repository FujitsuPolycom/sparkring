# Qwen3.8-Flash-Next with SparkCache on two Sparks

Use the [single Qwen TP2 quickstart](../qwen38-flash-next-tp2/README.md).
Its default selects the published persistent-cache configuration; it also
contains the cache-disabled alternative and the restart procedure.

[Generated Compose deployments](compose/README.md) are also available.
[Bounded TP2 checks](../../performance/records/qwen38-flash-next/compose-tp2.json)
cover startup, shutdown and persistent-cache restore through fresh containers
with the original non-QAD checkpoint, not the QAD selection in the quickstart.

The configuration owner is
[sparkcache.json](../qwen38-flash-next-tp2/sparkcache.json).
[Validation scope](../../performance/records/qwen38-flash-next/r37-sparkcache.json)
remains specific to its recorded image and checkpoint.
