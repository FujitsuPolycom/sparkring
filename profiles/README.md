# Deployment profiles

Choose a quickstart below. Each profile defines its configuration, release and
evidence scope. See the [configuration guide](../docs/development/configuration.md)
for inspecting defaults or preparing private site inputs, and
[benchmarks](../performance/benchmarks.md) for measured results.

<!-- BEGIN GENERATED PROFILES -->

Configured context is a per-request limit, not measured KV capacity or a completed long-context test.
Available profiles have an implementation; validated profiles have documented checks for the selected configuration. Both may have run on Spark hardware. See each guide for the exact testing scope.

### Four Sparks

| Model / features | Layout | Configured context (tokens) | KV (tokens) | Status | Navigation | Quickstart |
|---|---|---:|---:|---|---|---|
| GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache | TP4/DCP1 | 1,048,576 | — | Validated | recommended | [Guide](glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | TP4/DCP1 | 1,048,576 | — | Available | alternative | [Guide](deepseek-v4-flash-0731/README.md) |
| DeepSeek-V4-Flash-Vision-Exp | TP4/DCP1 | 1,048,576 | — | Experimental | alternative | [Guide](deepseek-v4-flash-vision-exp-tp4/README.md) |
| DeepSeek-V4.1-Flash | TP4/DCP1 | 430,080 | [2,182,642](../profiles/deepseek-v41-flash-cycle/recipe.json) | Available | alternative | [Guide](deepseek-v41-flash-cycle/README.md) |
| GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 | TP4/DCP4 | 1,048,576 | [1,156,864](../profiles/glm52-exl3-r7-3.5bpw/recipe.json) | Available | alternative | [Guide](glm52-exl3-r7-3.5bpw/README.md) |
| GLM-5.3 Flash NVFP4-Spark · native MTP3 | TP4/DCP1 | 1,048,576 | — | Experimental | alternative | [Guide](glm53-flash-spark-tp4-dcp1/README.md) |
| GLM-5.3 Flash NVFP4-Spark · switched | TP4/DCP1 | 1,048,576 | — | Experimental | alternative | [Guide](glm53-flash-spark-tp4-switched/README.md) |
| Qwen3.8-27B-EXL3-K5K6-hydrated | TP4/DCP1 | 1,048,576 | [8,743,342](../profiles/qwen38-27b-exl3-k5k6/recipe.json) | Available | alternative | [Guide](qwen38-27b-exl3-k5k6/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | TP4/DCP1 | 1,048,576 | — | Available | alternative | [Guide](sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) |
| GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78 + SparkCache | TP4/DCP4 | 1,048,576 | — | Available | alternative | [Guide](sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) |

### Two Sparks

| Model / features | Layout | Configured context (tokens) | KV (tokens) | Status | Navigation | Quickstart |
|---|---|---:|---:|---|---|---|
| GLM-5.3 Flash NVFP4-Spark · native MTP3 + SparkCache | TP2/DCP1 | 1,048,576 | [1,081,922](../runtime/profiles/glm53-flash-spark-tp2/README.md) | Validated | recommended | [Guide](glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | TP2/DCP1 | 1,048,576 | — | Available | alternative | [Guide](deepseek-v4-flash-0731-pair/README.md) |
| GLM-5.3 Flash NVFP4-Spark · native MTP3 | TP2/DCP1 | 1,048,576 | — | Experimental | alternative | [Guide](glm53-flash-spark-tp2-dcp1/README.md) |
| Qwen3.8-27B-EXL3-K5K6-hydrated | TP2/DCP1 | 1,048,576 | [4,130,233](../profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | Available | alternative | [Guide](qwen38-27b-exl3-k5k6-pair/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | TP2/DCP1 | 1,048,576 | — | Available | alternative | [Guide](sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) |

### Retired profiles

| Model / features | Layout | Configured context (tokens) | KV (tokens) | Status | Navigation | Quickstart |
|---|---|---:|---:|---|---|---|
| GLM-5.3-Flash-NVFP4 | TP4/DCP4 | 1,048,576 | — | Available | retired | [Guide](glm53-flash-nvfp4-dflash2-bf16-tp4/README.md) |
| GLM-5.3-Flash-NVFP4-Spark | TP4/DCP4 | 1,048,576 | — | Experimental | retired | [Guide](glm53-mtp3-cache-checkpoints-tp4/README.md) |
| GLM-5.3-Flash-NVFP4-Spark | TP4/DCP4 | 1,048,576 | — | Experimental | retired | [Guide](glm53-spark-mtp3-managed-mesh-tp4/README.md) |
| GLM-5.3-Flash-NVFP4 + SparkCache | TP4/DCP4 | 1,048,576 | — | Validated | retired | [Guide](sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4/README.md) |

Additional pinned historical variants, including original NVFP4 TP2, are in the [retained deployment index](../docs/history/deployment-variants.md).

Qualification applies only to the exact image, checkpoint, topology and workload in the selected guide.
Switched deployments have no switched-hardware qualification. Qwen with SparkCache is unsupported;
six-node work remains research-only and is outside this deployment catalog.

<!-- END GENERATED PROFILES -->
