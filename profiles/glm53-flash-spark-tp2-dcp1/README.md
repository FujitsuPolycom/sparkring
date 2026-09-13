# GLM-5.3-Flash TP2 without SparkCache

Use the [two-Spark quickstart](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md)
and its [SparkCache-off selection](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md#sparkcache-off).
Keep the published R33 image receipt; omitting it selects a different source-image configuration.

This selection uses DCP1, InstantTensor loading and 8.75 GiB KV per rank.
See [profile.json](profile.json) for its exact release and evidence scope.

For **R35** without SparkCache, use the
[R35 TP2 instructions](../../docs/operations/r35-local-launch.md#tp2) with
`CACHE_ARGS=()`. Keep the R35 image receipt. R35 is Experimental.
