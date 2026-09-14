# Qwen3.8-Flash-Next with SparkCache on two Sparks

[Generated Compose deployments](compose/README.md) are also available. Compose serving remains unqualified.

Status: **Experimental**. This configuration uses the published R37 cache
extension with managed B12X loading, MTP3, TP2/DCP1, native 262K context,
16 sequences, 8192 batched tokens and 24 GiB KV per rank. It allows three
images and one video per request, with 16-frame loader sampling.

Follow the [SparkCache quickstart](../qwen38-flash-next-tp2/SPARKCACHE.md) for
the immutable image digest, checkpoint verification and guarded launch commands.
The shared configuration is [sparkcache.json](../qwen38-flash-next-tp2/sparkcache.json).

[Validation evidence](../../performance/records/qwen38-flash-next/r37-sparkcache.json)
covers bounded text/media disk restoration, changed-input misses and corruption
recomputation. General media quality and prolonged store-pressure stability are
not established. The [cache-disabled profile](../qwen38-flash-next-tp2/README.md)
remains available as a separate choice.
