# GLM-5.3-Flash DCP4 without SparkCache

Checkpoint: [`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark), Local Inference Lab's NVFP4 Spark checkpoint of Z.ai's GLM-5.3-Flash.

Use the [four-Spark quickstart](../glm53-flash-spark-tp4-dcp1-sparkcache/README.md#dcp4-alternative)
and select `tp4-dcp4`. DCP1 is the default; DCP4 is an alternative.

This catalog entry retains the R33 image and its DCP4 evidence. Reproducing
that release requires the documented contract/entrypoint overlay and prepared
managed fabric. The deployment-suite planner accepts DCP1 selections only.
R37 does not inherit these DCP4 results; use the image-specific instructions
in the linked guide.

See [profile.json](profile.json) for the exact release and evidence scope.
