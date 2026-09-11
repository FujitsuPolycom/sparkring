# GLM-5.3-Flash TP4/DCP4

Quant: [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark).
SparkCache is disabled in this configuration.

A separate cache-disabled TP4/DCP4 startup passed semantic generation and 1M admission. This is an observation, not the cache-enabled recovery qualification.

```bash
python scripts/profiles.py resolve glm53-flash-spark-tp4-dcp4
```

Follow the [four-Spark quickstart](../../docs/GLM53_TP4_PREFILL_QUICKSTART.md)
and its [DCP4 overlay procedure](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md#reproduction-overlay-and-quickstart).
Use the published image plus the supplied contract and entrypoint overlay;
DCP4 requires the managed fabric installation on every rank.

The recorded KV pool is **8,364,901 tokens** at 24 GiB per rank. The configured
context limit is 1,048,576 tokens; a completed request of that size is not claimed.

[DCP1](../glm53-flash-spark-tp4-dcp1-sparkcache/README.md) is the default configuration. DCP4 is an alternative
with a larger KV pool. The [record](../../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md)
compares its bounded prefill/decode observations. For a switched fabric, use
[switched setup](../glm53-flash-spark-tp4-switched/README.md); it has a separate configuration.
