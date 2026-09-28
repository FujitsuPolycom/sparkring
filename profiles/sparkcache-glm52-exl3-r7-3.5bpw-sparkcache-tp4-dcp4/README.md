# GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 + SparkCache

Checkpoint: [`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78`](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78), brandonmusic's EXL3 3.5 bpw quantization of Z.ai's [GLM-5.2](https://huggingface.co/zai-org/GLM-5.2).

Status: **retired**. This SparkCache composition of the
[GLM-5.2 EXL3 profile](../glm52-exl3-r7-3.5bpw/README.md) is defined only for
TP4 with decode context parallelism 4 (DCP4), which SparkRing does not offer
or support. No supported profile serves GLM-5.2. The pinned
[recipe](recipe.json), with evidence status **Development**, is retained for
reproducing that image and configuration; select the
[profile catalog](../README.md) for maintained deployments. This is a separate
GLM-5.2 cache composition, not the GLM-5.3 shared image.

Inspect the resolved configuration without contacting a host:

```bash
python scripts/profiles.py resolve sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4
```

The [SparkCache composition notes](../../recipes/sparkcache/README.md) record
its artifacts and qualification limits. Keep private site inputs outside Git,
and reproduce it only with its named image, model revisions and topology.

The recipe records configuration and evidence boundaries. Its implementation status does not qualify a rebuilt image. Configured context, allocated KV capacity and completed request tests are separate facts.
