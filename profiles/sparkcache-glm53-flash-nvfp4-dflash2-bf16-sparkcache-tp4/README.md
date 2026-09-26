# GLM-5.3-Flash-NVFP4 + SparkCache

Checkpoint: [`local-inference-lab/GLM-5.3-Flash-NVFP4`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4), Local Inference Lab's NVFP4 quantization-aware distillation of Z.ai's [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16).

This deployment uses the pinned [recipe](recipe.json). Status: **Validated**.
It is retained for reproducing that image and configuration; select the
[profile catalog](../README.md) for maintained deployments.

The plain Hugging Face repository also provides a QAD checkpoint at revision
`175ae8ce3b5af842b0d0140dbeb43e9cfc557c49`. See
[checkpoint selection](../glm53-checkpoints.md) for its download pin. This
reproduction recipe retains `520de24`; its image and cache evidence do not
qualify QAD or permit reuse of its snapshots under another checkpoint identity.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4
```

Follow the [deployment instructions](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md). Keep private site inputs outside Git. Follow the guide only with its named
image, model revisions and topology.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
