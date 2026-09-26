# GLM-5.3-Flash-NVFP4-Spark

Checkpoint: [`local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark), Local Inference Lab's NVFP4 Spark checkpoint of Z.ai's GLM-5.3-Flash.

This deployment uses the pinned [recipe](recipe.json). Status: **Experimental**.
It is retained for reproducing that image and configuration; select the
[profile catalog](../README.md) for maintained deployments.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve glm53-mtp3-cache-checkpoints-tp4
```

Follow the [deployment instructions](../../docs/GLM53_MTP3_CACHE_CHECKPOINTS_QUICKSTART.md). Keep private site inputs outside Git. Follow the guide only with its named
image, model revisions and topology.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
