# GLM-5.3 Flash on TP4, without SparkCache

Checkpoint: [`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark), Local Inference Lab's NVFP4 Spark checkpoint of Z.ai's GLM-5.3-Flash.

Development installer selection using the pinned shared-2026.09.3 ARM64 image. SparkCache is disabled; native in-memory prefix caching remains profile-owned.

```bash
sudo sparkring install --profile glm53-flash-spark-tp4-dcp1-nocache --plan
sudo sparkring install --profile glm53-flash-spark-tp4-dcp1-nocache --yes
```

See [installation guide](../../docs/operations/install.md) for setup and logs. Hardware acceptance remains pending. Existing managed-mesh ownership must be resolved before TP4 replacement.
