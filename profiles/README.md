# Deployment profiles

Choose a quickstart below. Each profile defines its configuration, release and
evidence scope. See the [configuration guide](../docs/development/configuration.md)
for inspecting defaults or preparing private site inputs, and
[benchmarks](../performance/benchmarks.md) for measured results.

Model names come from `model-names.json`. The quant column links the
checkpoint repository selected by the profile; exact revisions remain pinned
in its configuration. “Stock” identifies the publisher’s original checkpoint.

<!-- BEGIN GENERATED PROFILES -->

Configured context is a per-request limit, not measured KV capacity or a completed long-context test.
Development profiles are under active development; validated profiles have documented checks for the selected configuration. See each guide for the exact testing scope.

### Four Sparks

| Model | Quant | Layout | Configured context (tokens) | KV* (tokens) | Status | Navigation | Quickstart |
|---|---|---|---:|---:|---|---|---|
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP4 | 1,048,576 | [8,364,901](../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md) | Validated | recommended | [Guide](glm53-flash-spark-tp4-dcp4-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | TP4/DCP1 | 1,048,576 | [~1,000,000](../performance/capacity-references.md) | Development | alternative | [Guide](deepseek-v4-flash-0731/README.md) |
| DeepSeek-V4-Flash-Vision-Exp | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | TP4/DCP1 | 1,048,576 | — | Experimental | alternative | [Guide](deepseek-v4-flash-vision-exp-tp4/README.md) |
| DeepSeek-V4.1-Flash | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | TP4/DCP1 | 1,048,576 | [~2,182,642](../profiles/deepseek-v41-flash-cycle/recipe.json) | Development | alternative | [Guide](deepseek-v41-flash-cycle/README.md) |
| GLM-5.2 | [EXL3 3.5bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78) | TP4/DCP4 | 1,048,576 | [1,156,864](../profiles/glm52-exl3-r7-3.5bpw/recipe.json) | Development | alternative | [Guide](glm52-exl3-r7-3.5bpw/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP1 | 1,048,576 | [~2,300,000](../performance/capacity-references.md) | Experimental | alternative | [Guide](glm53-flash-spark-tp4-dcp1/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP1 | 1,048,576 | [~2,280,000](../performance/capacity-references.md) | Validated | alternative | [Guide](glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP4 | 1,048,576 | — | Development | alternative | [Guide](glm53-flash-spark-tp4-dcp4/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP1 · switched | 1,048,576 | [~2,300,000](../performance/capacity-references.md) | Experimental | alternative | [Guide](glm53-flash-spark-tp4-switched/README.md) |
| Qwen3.8-27B | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | TP4/DCP1 | 1,048,576 | [8,743,342](../profiles/qwen38-27b-exl3-k5k6/recipe.json) | Development | alternative | [Guide](qwen38-27b-exl3-k5k6/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | TP4/DCP1 | 1,048,576 | — | Development | alternative | [Guide](sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) |
| GLM-5.2 | [EXL3 3.5bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78) | TP4/DCP4 | 1,048,576 | — | Development | alternative | [Guide](sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) |

### Two Sparks

| Model | Quant | Layout | Configured context (tokens) | KV* (tokens) | Status | Navigation | Quickstart |
|---|---|---|---:|---:|---|---|---|
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP2/DCP1 | 1,048,576 | [1,081,922](../runtime/profiles/glm53-flash-spark-tp2/README.md) | Validated | recommended | [Guide](glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | TP2/DCP1 | 1,048,576 | [~1,000,000](../performance/capacity-references.md) | Development | alternative | [Guide](deepseek-v4-flash-0731-pair/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP2/DCP1 | 1,048,576 | — | Experimental | alternative | [Guide](glm53-flash-spark-tp2-dcp1/README.md) |
| Qwen3.8-27B | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | TP2/DCP1 | 1,048,576 | [4,130,233](../profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | Development | alternative | [Guide](qwen38-27b-exl3-k5k6-pair/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | TP2/DCP1 | 1,048,576 | — | Development | alternative | [Guide](sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) |

### Retired profiles

| Model | Quant | Layout | Configured context (tokens) | KV* (tokens) | Status | Navigation | Quickstart |
|---|---|---|---:|---:|---|---|---|
| GLM-5.3-Flash | [NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4) | TP4/DCP4 | 1,048,576 | [4,321,618](../profiles/glm53-flash-nvfp4-dflash2-bf16-tp4/recipe.json) | Development | retired | [Guide](glm53-flash-nvfp4-dflash2-bf16-tp4/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP4 | 1,048,576 | — | Experimental | retired | [Guide](glm53-mtp3-cache-checkpoints-tp4/README.md) |
| GLM-5.3-Flash | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | TP4/DCP4 | 1,048,576 | — | Experimental | retired | [Guide](glm53-spark-mtp3-managed-mesh-tp4/README.md) |
| GLM-5.3-Flash | [NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4) | TP4/DCP4 | 1,048,576 | [4,321,618](../profiles/sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4/recipe.json) | Validated | retired | [Guide](sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4/README.md) |

Additional pinned historical variants, including original NVFP4 TP2, are in the [retained deployment index](../docs/history/deployment-variants.md).

Qualification applies only to the exact image, checkpoint, topology and workload in the selected guide.
Switched deployments have no switched-hardware qualification. Qwen with SparkCache is unsupported;
six-node work remains research-only and is outside this deployment catalog.

<!-- END GENERATED PROFILES -->

\* KV capacity depends on configuration and enabled features, including SparkCache.
Linked figures retain their measurement conditions.
