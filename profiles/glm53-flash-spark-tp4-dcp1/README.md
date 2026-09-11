# GLM-5.3 Flash NVFP4-Spark · MTP3

Status: **research-only**. Source contract only; packaged alternatives do not inherit another profile’s functional qualification.

```bash
python scripts/profiles.py resolve glm53-flash-spark-tp4-dcp1
```

Use the [primary quickstart](../../docs/GLM53_TP4_PREFILL_QUICKSTART.md) with the exact release selected by [profile.json](profile.json). The resolver reports the image and immutable contract before any operation.

Complete managed mesh host setup before serving; do not replace live queue pairs.

Configured context is not a completed long-context test. See the [publication record](../../runtime/sparkring/jovian-r33/publication.json) for the exact measured conditions.

## Network setup

For a four-Spark ring, use the managed mesh setup in the quickstart above.
For a switched fabric, follow the [switched NCCL setup](../glm53-flash-spark-tp4-switched/README.md),
which selects its own pinned image and explicit HCA/GID settings. Topology is
not detected automatically. That configuration does not enable SparkCache.

## DCP4 alternative

TP4/DCP1 is the default. [TP4/DCP4](../glm53-flash-spark-tp4-dcp4-sparkcache/README.md)
is a validated alternative with an 8.4M-token recorded KV pool; it requires
the contract/entrypoint overlay described in its guide.
