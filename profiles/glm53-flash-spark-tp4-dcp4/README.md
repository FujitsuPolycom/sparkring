# GLM-5.3-Flash DCP4 without SparkCache

Checkpoint: [`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark), Local Inference Lab's NVFP4/MXFP8 quantization of Z.ai's [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16).

Status: **retired**. SparkRing does not offer or support decode context
parallelism 4 (DCP4). This catalog entry retains the R33 image selection
`tp4-dcp4` and its DCP4 evidence for reproducing that image and configuration;
it is not a deployment guide.

For GLM-5.3-Flash on four Sparks, use a supported DCP1 profile: the installer
profile `glm53-flash-nvfp4-spark-tp4` ([Install SparkRing](../../docs/operations/install.md))
or [`glm53-flash-spark-tp4-dcp1-sparkcache`](../glm53-flash-spark-tp4-dcp1-sparkcache/README.md).

The [R33 TP4/DCP4 record](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md#cache-disabled-tp4-dcp4-observation)
describes the cache-disabled observation and the contract/entrypoint overlay
that reproduction requires. The deployment-suite planner accepts DCP1
selections only, and no other image inherits these DCP4 results.

See [profile.json](profile.json) for the exact release and evidence scope.
