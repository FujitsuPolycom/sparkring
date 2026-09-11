# GLM-5.3-Flash-NVFP4-Spark

This deployment uses the authoritative [recipe](recipe.json). Status: **research-only**; navigation: **retired**.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve glm53-mtp3-cache-checkpoints-tp4
```

Follow the [deployment instructions](../../docs/GLM53_MTP3_CACHE_CHECKPOINTS_QUICKSTART.md). Keep private site inputs outside Git. The guide defines the relevant topology, prerequisites and startup gates.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
