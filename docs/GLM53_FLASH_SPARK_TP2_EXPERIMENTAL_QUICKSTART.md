# GLM-5.3 Flash NVFP4-Spark TP2 quickstart

Use the [published-image TP2 guide](../profiles/glm53-flash-spark-tp2-dcp1/README.md)
for a two-Spark deployment with a 1M-token per-request context limit, or its
[SparkCache variant](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md).

The [retained source-image TP2 guide](../runtime/profiles/glm53-flash-spark-tp2/README.md)
defines a separate contract
for 8.75 GiB FP8 KV per node, managed B12X loading, static MTP3, and single-DAC
transport across both PCI domains. The reference allocator estimated 1,050,118
KV tokens; the configured per-request context limit is 262,144.
The guide defines the source-image receipt, active 2 GiB host-memory guard,
and manual lifecycle commands.

The retired 5 GiB adaptive-MTP profile and its image are preserved in
[Git history](https://github.com/FujitsuPolycom/sparkring/tree/2f01b6ee8f6173745c4b6b165498bbef82fc03f1/runtime/profiles/glm53-flash-spark-tp2).
Their bounded [installation and video observations](../performance/records/glm53-flash/tp2-public-image-install-20260906.md)
remain historical evidence; they do not qualify either deployment linked above.
