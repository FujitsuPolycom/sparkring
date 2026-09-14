# Qwen3.8-Flash-Next with SparkCache on two Sparks

Use the [single Qwen TP2 quickstart](../qwen38-flash-next-tp2/README.md).
Its default selects the published persistent-cache configuration; it also
contains the cache-disabled alternative and the restart procedure.

[Generated Compose deployments](compose/README.md) are also available.
Serving through Compose still requires hardware validation.

The configuration owner is
[sparkcache.json](../qwen38-flash-next-tp2/sparkcache.json).
[Validation scope](../../performance/records/qwen38-flash-next/r37-sparkcache.json)
remains Experimental and specific to its pinned image.
