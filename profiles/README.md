# Deployment profiles

Choose a quickstart below. Each profile defines its configuration, release and
evidence scope. See the [configuration guide](../docs/development/configuration.md)
for inspecting defaults or preparing private site inputs, and
[benchmarks](../performance/benchmarks.md) for measured results.

Model names come from `model-names.json`. The quant column links the
checkpoint repository selected by the profile; exact revisions remain pinned
in its configuration. “Stock” identifies the publisher’s original checkpoint.

The [NVIDIA GLM NVFP4 target](glm53-nvidia-nvfp4.md) is an optional Development
variant of the GLM TP4 settings below; NVFP4-Spark remains their default.

<!-- BEGIN GENERATED PROFILES -->

### Four Sparks

| Model | Quant | Runtime | DCP | Context / KV* | SparkCache | Status | Quickstart |
|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | vLLM | DCP1/DCP4 | 1M / ([2.3M](../performance/capacity-references.md)/[8.4M](../performance/records/glm53-flash/r33-image020-tp4-dcp4-sparkcache-20260911.md)) | [Optional](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Validated | [Guide](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | vLLM | DCP1 | 1M / [1M](../performance/capacity-references.md) | [Optional](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) | Development | [Guide](../profiles/deepseek-v4-flash-0731/README.md) |
| DeepSeek-V4-Flash-Vision-Exp | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp) | vLLM | DCP1 | 1M / — | No | Experimental | [Guide](../profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |
| DeepSeek-V4.1-Flash | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | vLLM | DCP1 | 1M / [2.2M](../profiles/deepseek-v41-flash-cycle/recipe.json) | No | Development | [Guide](../profiles/deepseek-v41-flash-cycle/README.md) |
| DeepSeek-V4.1-Flash | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) | SGLang | — | 262K / [1.5M](../performance/records/deepseek-v41-flash/sglang-soak-20260912.md) | No | Development | [Guide](../profiles/deepseek-v41-flash-sglang-cycle/README.md) |
| GLM-5.2 | [EXL3 3.5bpw](https://huggingface.co/brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78) | vLLM | DCP4 | 1M / [1.2M](../profiles/glm52-exl3-r7-3.5bpw/recipe.json) | [Optional](../profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) | Development | [Guide](../profiles/glm52-exl3-r7-3.5bpw/README.md) |
| Qwen3.8-27B | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | vLLM | DCP1 | 1M / [8.7M](../profiles/qwen38-27b-exl3-k5k6/recipe.json) | No | Development | [Guide](../profiles/qwen38-27b-exl3-k5k6/README.md) |
| Qwen3.8-Flash-Next | [NVFP4 QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/629bc3218833a38b475b719f34aa571666f4a03e) | vLLM | DCP1 | 262K / [3.2M](../performance/records/qwen38-flash-next/r37-shared-tp4.json) | [Optional](../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) | Development | [Guide](../profiles/qwen38-flash-next-qad-tp4/README.md) |

### Two Sparks

| Model | Quant | Runtime | DCP | Context / KV* | SparkCache | Status | Quickstart |
|---|---|---|---|---|---|---|---|
| **GLM-5.3-Flash** | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark) | vLLM | DCP1 | 1M / [1.1M](../performance/records/glm53-flash/r35-tp2-sparkcache.json) | [Optional](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Validated | [Guide](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) | vLLM | DCP1 | 1M / [1M](../performance/capacity-references.md) | [Optional](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) | Development | [Guide](../profiles/deepseek-v4-flash-0731-pair/README.md) |
| Qwen3.8-27B | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) | vLLM | DCP1 | 1M / [4.1M](../profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | No | Development | [Guide](../profiles/qwen38-27b-exl3-k5k6-pair/README.md) |
| Qwen3.8-Flash-Next | [NVFP4](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4) | vLLM | DCP1 | 262K / [3M](../performance/records/qwen38-flash-next/r37-tp2.json) | [Optional](../profiles/qwen38-flash-next-tp2/README.md) | Experimental | [Guide](../profiles/qwen38-flash-next-tp2/README.md) |


Status and context describe the linked default. DCP and KV figures follow the same order;
capacity depends on enabled features. Expand a deployment below for each option’s own status and guide.
Switched support is a separate network configuration and has no switched-hardware qualification.

## Configuration variants

These are saved configurations, not separate models. Profile IDs remain stable for scripts.

<details>
<summary>GLM-5.2 · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP4 | direct-cycle-4 | Off | Development | [glm52-exl3-r7-3.5bpw](../profiles/glm52-exl3-r7-3.5bpw/README.md) |
| DCP4 | direct-cycle-4 | On | Development | [sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4](../profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-0731 · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [deepseek-v4-flash-0731-pair](../profiles/deepseek-v4-flash-0731-pair/README.md) |
| DCP1 | direct-pair-2 | On | Development | [sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-0731 · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [deepseek-v4-flash-0731](../profiles/deepseek-v4-flash-0731/README.md) |
| DCP1 | direct-cycle-4 | On | Development | [sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-Vision-Exp · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Experimental | [deepseek-v4-flash-vision-exp-tp4](../profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 4 Sparks · SGLang</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| EP4 | direct-cycle-4 | Off | Development | [deepseek-v41-flash-sglang-cycle](../profiles/deepseek-v41-flash-sglang-cycle/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [deepseek-v41-flash-cycle](../profiles/deepseek-v41-flash-cycle/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | tp2-rocenante-adaptive | On | Validated | [glm53-flash-spark-tp2-dcp1-sparkcache (default)](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) |
| DCP1 | tp2-rocenante-adaptive | Off | Experimental | [glm53-flash-spark-tp2-dcp1](../profiles/glm53-flash-spark-tp2-dcp1/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | sparkring-rocenante-mesh | On | Validated | [glm53-flash-spark-tp4-dcp1-sparkcache (default)](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) |
| DCP1 | sparkring-rocenante-mesh | Off | Experimental | [glm53-flash-spark-tp4-dcp1](../profiles/glm53-flash-spark-tp4-dcp1/README.md) |
| DCP4 | sparkring-rocenante-mesh | Off | Development | [glm53-flash-spark-tp4-dcp4](../profiles/glm53-flash-spark-tp4-dcp4/README.md) |
| DCP4 | sparkring-rocenante-mesh | On | Validated | [glm53-flash-spark-tp4-dcp4-sparkcache](../profiles/glm53-flash-spark-tp4-dcp4-sparkcache/README.md) |
| DCP1 | switched | Off | Experimental | [glm53-flash-spark-tp4-switched](../profiles/glm53-flash-spark-tp4-switched/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Experimental | [qwen38-flash-next-tp2](../profiles/qwen38-flash-next-tp2/README.md) |
| DCP1 | direct-pair-2 | On | Experimental | [qwen38-flash-next-tp2-sparkcache](../profiles/qwen38-flash-next-tp2/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [qwen38-flash-next-qad-tp4](../profiles/qwen38-flash-next-qad-tp4/README.md) |
| DCP1 | direct-cycle-4 | On | Development | [qwen38-flash-next-qad-tp4-sparkcache](../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |

</details>

<details>
<summary>Qwen3.8-27B · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [qwen38-27b-exl3-k5k6-pair](../profiles/qwen38-27b-exl3-k5k6-pair/README.md) |

</details>

<details>
<summary>Qwen3.8-27B · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [qwen38-27b-exl3-k5k6](../profiles/qwen38-27b-exl3-k5k6/README.md) |

</details>

### Retired profiles

<details>
<summary>Retired configurations</summary>

Retained for compatibility and historical evidence; use an active deployment above for setup.

- [glm53-flash-nvfp4-dflash2-bf16-tp4](../profiles/glm53-flash-nvfp4-dflash2-bf16-tp4/README.md) — GLM-5.3-Flash; Development
- [glm53-mtp3-cache-checkpoints-tp4](../profiles/glm53-mtp3-cache-checkpoints-tp4/README.md) — GLM-5.3-Flash; Experimental
- [glm53-spark-mtp3-managed-mesh-tp4](../profiles/glm53-spark-mtp3-managed-mesh-tp4/README.md) — GLM-5.3-Flash; Experimental
- [sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4](../profiles/sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4/README.md) — GLM-5.3-Flash; Validated

[Additional historical variants](../docs/history/deployment-variants.md)

</details>

Qwen Flash Next TP2 offers optional SparkCache with bounded text/media persistence validation. Six-node deployments remain experimental and are outside this catalog.

<!-- END GENERATED PROFILES -->

\* KV capacity depends on configuration and enabled features, including SparkCache.
Linked figures retain their measurement conditions.
