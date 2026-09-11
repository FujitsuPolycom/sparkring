# GLM-5.3-Flash-NVFP4 + SparkCache

This deployment uses the authoritative [recipe](recipe.json). Status: **qualified**; navigation: **retired**.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4
```

Follow the [deployment instructions](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md). Keep private site inputs outside Git. The guide defines the relevant topology, prerequisites and startup gates.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
