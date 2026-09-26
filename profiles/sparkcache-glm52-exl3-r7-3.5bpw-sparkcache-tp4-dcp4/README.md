# GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 + SparkCache

Checkpoint: [`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78), brandonmusic's EXL3 3.5 bpw quantization of Z.ai's [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2).

This deployment uses the pinned [recipe](recipe.json). Status: **Development**.
This is a separate GLM-5.2 cache composition, not the GLM-5.3 shared image.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4
```

Follow the [deployment instructions](../../recipes/sparkcache/README.md). Keep private site inputs outside Git. Follow the guide only with its named
image, model revisions and topology.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
