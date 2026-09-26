# DeepSeek-V4-Flash-0731 + SparkCache

Checkpoint: [`deepseek-ai/DeepSeek-V4-Flash-0731`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731), DeepSeek's published weights.

This deployment uses the authoritative [recipe](recipe.json). Status: **Development**. This is an alternative composition, not a standalone launcher.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1
```

Read the [composition requirements](../../recipes/sparkcache/README.md) and the exact artifact identities in the recipe. The catalog entry is guide-only: `resolve` inspects configuration and does not create a cache-enabled deployment. Do not add a GLM R35/R37 image or its connector contract to this DeepSeek composition. Keep private site inputs outside Git.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.

The [retained deployment record](../../performance/records/deepseek-v4-flash/sparkcache-tp4-public-reproduction-20260822.md) identifies the tested image, cache artifact and bounded workload. Use it as reproduction evidence, not as qualification of another image.
