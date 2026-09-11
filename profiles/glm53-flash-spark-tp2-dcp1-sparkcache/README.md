# GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache

Status: **qualified**. Bounded functional checks for the exact published image, model and topology only. Throughput observations lack a complete public methodology; full-context and arbitrary concurrency are not qualified.

```bash
python scripts/profile.py resolve glm53-flash-spark-tp2-dcp1-sparkcache
```

Use the [primary quickstart](../../runtime/profiles/glm53-flash-spark-tp2/README.md) with the exact release selected by [profile.json](profile.json). The resolver reports the image and immutable contract before any operation.

An active memory guard is required before creating or starting either rank.

Configured context is not a completed long-context test. See the [publication record](../../runtime/sparkring/jovian-r33/publication.json) for the exact measured conditions.
