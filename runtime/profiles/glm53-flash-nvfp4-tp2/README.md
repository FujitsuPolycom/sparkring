# GLM-5.3 Flash NVFP4-Spark TP2

The canonical two-Spark profile is [NVFP4-Spark with 8.75 GiB KV per node](../glm53-flash-spark-tp2/README.md).
It contains the exact checkpoint revision, approximately 1,050,118-token reference KV-pool estimate,
262,144-token request limit, guarded manual launcher, and shared-image receipt instructions.
This directory's launcher forwards to that canonical implementation.

The retired original-NVFP4 6.75 GiB configuration remains available in
[Git history](https://github.com/FujitsuPolycom/sparkring/tree/2f01b6ee8f6173745c4b6b165498bbef82fc03f1/runtime/profiles/glm53-flash-nvfp4-tp2).
Its source/image evidence does not qualify the Spark configuration.
