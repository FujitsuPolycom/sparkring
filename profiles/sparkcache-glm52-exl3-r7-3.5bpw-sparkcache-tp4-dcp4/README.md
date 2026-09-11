# GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 + SparkCache

This deployment uses the authoritative [recipe](recipe.json). Status: **implemented**; navigation: **alternative**.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profile.py resolve sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4
```

Follow the [deployment instructions](../../recipes/sparkcache/README.md). Keep private site inputs outside Git. The guide defines the relevant topology, prerequisites and startup gates.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
