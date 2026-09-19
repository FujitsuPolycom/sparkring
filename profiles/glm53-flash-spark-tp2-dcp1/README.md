# GLM-5.3-Flash TP2 without SparkCache

Status: **Experimental**. Use the [retained R37 TP2 procedure](https://github.com/FujitsuPolycom/sparkring/blob/5b28d768b37b21f5c97d910887e07144fcf251ef/profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md#sparkcache-off).
Use that document's repository revision, image receipt and checkpoint selection;
set `CACHE_ARGS=()` before planning or creating containers. Do not mix its
commands with the 2026.09.3 native-image quickstart.

R37 TP2 is **Experimental** and has not been hardware-qualified by the TP4 test.
This configuration uses MTP3, DCP1, 1M configured
context, InstantTensor loading and 8.75 GiB KV per rank. Coalescing is disabled;
mHC remains enabled. Cache-on test results do not qualify this cache-off mode.

For the bounded cache-enabled deployment, use the
[2026.09.3 quickstart](../glm53-flash-spark-tp2-dcp1-sparkcache/README.md).

The [catalog profile](profile.json) retains its published R33 identity and
defaults. For that release, use the retained quickstart's
[R33 fallback](https://github.com/FujitsuPolycom/sparkring/blob/5b28d768b37b21f5c97d910887e07144fcf251ef/profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md#r33-fallback)
with `CACHE_ARGS=()` and its R33 receipt. Omitting the receipt selects a
different source-image configuration.
