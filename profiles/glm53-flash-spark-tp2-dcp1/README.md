# GLM-5.3-Flash TP2 without SparkCache

Follow the [two-Spark quickstart](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md)
using its published R35 image and locally recorded image receipt. Select
`CACHE_ARGS=()` before planning or creating containers, as described in
[SparkCache off](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md#sparkcache-off).

R35 is **Experimental**. This configuration uses MTP3, DCP1, 1M configured
context, InstantTensor loading and 8.75 GiB KV per rank. Coalescing is disabled;
mHC remains enabled. Cache-on test results do not qualify this cache-off mode.

The [catalog profile](profile.json) retains its published R33 identity and
defaults. For that release, use the quickstart's
[R33 fallback](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md#r33-fallback)
with `CACHE_ARGS=()` and its R33 receipt. Omitting the receipt selects a
different source-image configuration.
