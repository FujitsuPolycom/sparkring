# GLM-5.3 Flash on TP2, without SparkCache

Development installer selection using the pinned shared-2026.09.3 ARM64 image. SparkCache is disabled; native in-memory prefix caching remains profile-owned.

```bash
sudo sparkring install --profile glm53-flash-spark-tp2-dcp1-nocache --plan
sudo sparkring install --profile glm53-flash-spark-tp2-dcp1-nocache --yes
```

See [installation guide](../../docs/operations/install.md) for setup and logs. Hardware acceptance remains pending. Existing managed-mesh ownership must be resolved before TP4 replacement.
