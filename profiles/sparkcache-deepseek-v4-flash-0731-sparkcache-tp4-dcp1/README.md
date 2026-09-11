# DeepSeek-V4-Flash-0731 + SparkCache

This deployment uses the authoritative [recipe](recipe.json). Status: **implemented**; navigation: **alternative**.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1
```

Follow the [deployment instructions](../../recipes/sparkcache/README.md). Keep private site inputs outside Git. The guide defines the relevant topology, prerequisites and startup gates.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
