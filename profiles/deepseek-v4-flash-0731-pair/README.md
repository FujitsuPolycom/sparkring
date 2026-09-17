# DeepSeek-V4-Flash-0731

Serve on a two-Spark pair using the [recipe](recipe.json). Status: **Development**; the cached published image has [bounded TP2 serving checks](../../performance/records/deepseek-v4-flash/image827a8e8c-tp2.json), with full weight verification and long-context qualification outstanding.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve deepseek-v4-flash-0731-pair
```

Follow the [deployment instructions](../../docs/operations/deepseek-0731.md). Keep private site inputs outside Git. The guide defines the relevant topology, prerequisites and startup gates.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
