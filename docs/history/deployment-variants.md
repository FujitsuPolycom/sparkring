# Retained deployment variants

These records identify configurations retired before repository restructuring.
They retain their original artifact identities and evidence limitations.

### Retired GLM-5.3 profiles

These guides retain their pinned configurations and evidence for reproduction.
For deployment with the shared image, use the matching two- or four-Spark entry above.

| Profile | Layout | Retained guide | Replacement |
|---|---|---|---|
| NVFP4-Spark MTP3 cache/checkpoint mesh | TP4/DCP4 | [Pinned cache/checkpoint setup](../../docs/GLM53_MTP3_CACHE_CHECKPOINTS_QUICKSTART.md) | [Shared-image MTP3 profiles](../../docs/GLM53_TP4_PREFILL_QUICKSTART.md) |
| NVFP4 with BF16 DFlash2 | TP4/DCP1, DCP2 or DCP4 | [Pinned DFlash2 setup](../../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) | [Shared-image MTP3](../../docs/GLM53_TP4_PREFILL_QUICKSTART.md) |
| NVFP4-Spark MTP3 with 5 GiB KV per rank | TP2/DCP1 | [Pinned TP2 setup](https://github.com/FujitsuPolycom/sparkring/blob/2f01b6ee8f6173745c4b6b165498bbef82fc03f1/docs/GLM53_FLASH_SPARK_TP2_EXPERIMENTAL_QUICKSTART.md) | [Shared-image NVFP4-Spark TP2](../../runtime/profiles/glm53-flash-spark-tp2/README.md) |
| Original NVFP4 MTP3 with 6.75 GiB KV per rank | TP2/DCP1 | [Pinned original-NVFP4 setup](https://github.com/FujitsuPolycom/sparkring/blob/2f01b6ee8f6173745c4b6b165498bbef82fc03f1/runtime/profiles/glm53-flash-nvfp4-tp2/README.md) | [Shared-image NVFP4-Spark TP2](../../runtime/profiles/glm53-flash-spark-tp2/README.md) |

The retired 5 GiB TP2 configuration has a recorded
[blue-video recognition issue](https://github.com/FujitsuPolycom/sparkring/issues/229).

DFlash2 uses a separate draft checkpoint with
[CC BY-NC-ND 4.0 terms](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2#license).
