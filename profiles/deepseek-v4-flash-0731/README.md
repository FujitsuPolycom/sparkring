# DeepSeek-V4-Flash-0731

Serve on a four-Spark cycle using the [recipe](recipe.json). Status: **Development**; the selected published image still needs exact replay validation.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve deepseek-v4-flash-0731
```

Follow the [deployment instructions](../../docs/operations/deepseek-0731.md). Keep private site inputs outside Git. The guide defines the relevant topology, prerequisites and startup gates.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
