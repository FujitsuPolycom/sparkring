# GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache

Status: **qualified**. Bounded functional checks for the exact published image, model and topology only. Throughput observations lack a complete public methodology; full-context and arbitrary concurrency are not qualified.

```bash
python scripts/profile.py resolve glm53-flash-spark-tp4-dcp1-sparkcache
```

Use the [primary quickstart](../../docs/GLM53_TP4_PREFILL_QUICKSTART.md) with the exact release selected by [profile.json](profile.json). The resolver reports the image and immutable contract before any operation.

Complete managed mesh host setup before serving; do not replace live queue pairs.

Configured context is not a completed long-context test. See the [publication record](../../runtime/sparkring/jovian-r33/publication.json) for the exact measured conditions.
