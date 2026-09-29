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

The Qwen3.8-Flash-Next rows link the installer profiles, which `sparkring install`
runs on image `dev-20260928-plainstatus-cuda1342-nccl2323-status033`
([Install SparkRing](../docs/operations/install.md)). Their KV links point to
the shared-2026.09.3 [correctness summary](../runtime/releases/shared-2026.09.3/correctness.json),
whose figures were measured on the shared-2026.09.3 image with checkpoint step
4000 (revision `629bc3218833`), not on the installer image with step 5500. On the installer image, the TP4 profile's KV cache
held 3,131,214 tokens ([installer tuning record](../performance/records/qwen38-flash-next/installer-tuning-20260925.md)).

To contribute a profile that `sparkring install` sets up, see
[Contributing an installer profile](../docs/development/installer-profiles.md).

<!-- BEGIN GENERATED PROFILES -->

### Four Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status |
|---|---|---|---|---|---|
| **[DeepSeek-V4.1-Flash](../docs/operations/install.md)**<br>vLLM | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)<br>by DeepSeek | 1 | 1M / — | No | Development |
| **[GLM-5.3-Flash](../docs/operations/install.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)<br>by Local Inference Lab | 1 | 1M / [6.1M](../performance/records/images/dev-20260928-plainstatus-glm53-flash-tp4-20260929.md) | No | Development |
| **[GLM-5.3-Flash](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)<br>by Local Inference Lab | 1 | 1M / [2.3M](../runtime/releases/shared-2026.09.3/correctness.json) | [Optional](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Validated |
| **[MiMo-V2.6-Flash-MOPD](../docs/operations/install.md)**<br>vLLM | [MXFP8/BF16](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)<br>by Xiaomi MiMo | 1 | 262K / — | No | Development |
| **[Qwen3.8-Flash-Next](../profiles/qwen38-flash-next-qad-tp4/README.md)**<br>vLLM | [NVFP4 QAD step 5500 PLE](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)<br>by Local Inference Lab | 1 | 262K / [3.1M](../runtime/releases/shared-2026.09.3/correctness.json) | No | Development |
| **[Swift-1.5-Qwen3.8-Flash-Next](../profiles/swift15-qwen38-flash-next-tp4/README.md)**<br>vLLM | [NVFP4 experts/BF16](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4)<br>by UkisAI | 1 | 262K / — | No | Experimental |
| [DeepSeek-V4-Flash-0731](../profiles/deepseek-v4-flash-0731/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)<br>by DeepSeek | 1 | 1M / [1M](../performance/capacity-references.md) | [Optional](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) | Development |
| [DeepSeek-V4-Flash-Vision-Exp](../profiles/deepseek-v4-flash-vision-exp-tp4/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)<br>by DeepSeek | 1 | 1M / — | No | Experimental |
| [DeepSeek-V4.1-Flash](../profiles/deepseek-v41-flash-sglang-cycle/README.md)<br>SGLang | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)<br>by DeepSeek | — | 262K / [1.5M](../performance/records/deepseek-v41-flash/sglang-soak-20260912.md) | No | Development |
| [Qwen3.8-27B](../profiles/qwen38-27b-exl3-k5k6/README.md)<br>vLLM | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)<br>by malaiwah | 1 | 1M / [8.7M](../profiles/qwen38-27b-exl3-k5k6/recipe.json) | No | Development |

### Two Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status |
|---|---|---|---|---|---|
| **[GLM-5.3-Flash](../docs/operations/install.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)<br>by Local Inference Lab | 1 | 1M / [1.5M](../performance/records/images/dev-20260928-plainstatus-glm53-flash-tp2-20260929.md) | No | Development |
| **[GLM-5.3-Flash](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)<br>by Local Inference Lab | 1 | 1M / [1.1M](../runtime/releases/shared-2026.09.3/correctness.json) | [Optional](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Validated |
| **[MiMo-V2.6-Flash-MOPD](../docs/operations/install.md)**<br>vLLM | [MXFP8/BF16](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)<br>by Xiaomi MiMo | 1 | 262K / — | No | Development |
| **[Qwen3.8-Flash-Next](../profiles/qwen38-flash-next-tp2/README.md)**<br>vLLM | [NVFP4 QAD step 5500 PLE](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)<br>by Local Inference Lab | 1 | 262K / [2.9M](../runtime/releases/shared-2026.09.3/correctness.json) | No | Development |
| **[Swift-1.5-Qwen3.8-Flash-Next](../profiles/swift15-qwen38-flash-next-tp2/README.md)**<br>vLLM | [NVFP4 experts/BF16](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4)<br>by UkisAI | 1 | 262K / — | No | Development |
| [DeepSeek-V4-Flash-0731](../profiles/deepseek-v4-flash-0731-pair/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)<br>by DeepSeek | 1 | 1M / [2.2M](../performance/records/deepseek-v4-flash/image827a8e8c-tp2.json) | [Optional](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) | Development |
| [Qwen3.8-27B](../profiles/qwen38-27b-exl3-k5k6-pair/README.md)<br>vLLM | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)<br>by malaiwah | 1 | 1M / [4.1M](../profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | No | Development |


Status and context describe the linked default. DCP and KV figures follow the same order;
capacity depends on enabled features. Expand a deployment below for each option’s own status and guide.
Switched support is a separate network configuration and has no switched-hardware qualification.

## Configuration variants

Profile IDs identify saved configurations. Guide status describes the primary quickstart;
record links preserve configuration evidence when the guide selects a different release.

<details>
<summary>MiMo-V2.6-Flash-MOPD · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [mimo-v26-flash-mopd-tp2 (default)](../docs/operations/install.md) |

</details>

<details>
<summary>MiMo-V2.6-Flash-MOPD · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [mimo-v26-flash-mopd-tp4 (default)](../docs/operations/install.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-0731 · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [deepseek-v4-flash-0731-pair](../profiles/deepseek-v4-flash-0731-pair/README.md) |
| DCP1 | direct-pair-2 | On | Development | [sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-0731 · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [deepseek-v4-flash-0731](../profiles/deepseek-v4-flash-0731/README.md) |
| DCP1 | direct-cycle-4 | On | Development | [sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-Vision-Exp · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Experimental | [deepseek-v4-flash-vision-exp-tp4](../profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 4 Sparks · SGLang</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| EP4 | direct-cycle-4 | Off | Development | [deepseek-v41-flash-sglang-cycle](../profiles/deepseek-v41-flash-sglang-cycle/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [deepseek-v41-flash-tp4 (default)](../docs/operations/install.md) |
| DCP1 | direct-cycle-4 | Off | Development | [deepseek-v41-flash-cycle](../profiles/deepseek-v41-flash-cycle/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [glm53-flash-nvfp4-spark-tp2 (default)](../docs/operations/install.md) |
| DCP1 | tp2-rocenante-adaptive | On | Validated | [glm53-flash-spark-tp2-dcp1-sparkcache (default)](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) · [record](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/profile.json) |
| DCP1 | tp2-rocenante-adaptive | Off | Experimental | [glm53-flash-spark-tp2-dcp1](../profiles/glm53-flash-spark-tp2-dcp1/README.md) · [record](../profiles/glm53-flash-spark-tp2-dcp1/profile.json) |
| DCP1 | tp2-rocenante-adaptive | Off | Experimental | [glm53-flash-spark-tp2-dcp1-nocache](../profiles/glm53-flash-spark-tp2-dcp1-nocache/README.md) · [record](../profiles/glm53-flash-spark-tp2-dcp1-nocache/profile.json) |

</details>

<details>
<summary>GLM-5.3-Flash · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [glm53-flash-nvfp4-spark-tp4 (default)](../docs/operations/install.md) |
| DCP1 | sparkring-rocenante-mesh | On | Validated | [glm53-flash-spark-tp4-dcp1-sparkcache (default)](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) · [record](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/profile.json) |
| DCP1 | sparkring-rocenante-mesh | Off | Experimental | [glm53-flash-spark-tp4-dcp1](../profiles/glm53-flash-spark-tp4-dcp1/README.md) · [record](../profiles/glm53-flash-spark-tp4-dcp1/profile.json) |
| DCP1 | sparkring-rocenante-mesh | Off | Experimental | [glm53-flash-spark-tp4-dcp1-nocache](../profiles/glm53-flash-spark-tp4-dcp1-nocache/README.md) · [record](../profiles/glm53-flash-spark-tp4-dcp1-nocache/profile.json) |
| DCP1 | switched | Off | Experimental | [glm53-flash-spark-tp4-switched](../profiles/glm53-flash-spark-tp4-switched/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [qwen38-flash-next-tp2 (default)](../profiles/qwen38-flash-next-tp2/README.md) |
| DCP1 | direct-pair-2 | On | Validated | [qwen38-flash-next-tp2-sparkcache](../profiles/qwen38-flash-next-tp2/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [qwen38-flash-next-qad-tp4 (default)](../profiles/qwen38-flash-next-qad-tp4/README.md) |
| DCP1 | direct-cycle-4 | On | Validated | [qwen38-flash-next-qad-tp4-sparkcache](../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |

</details>

<details>
<summary>Qwen3.8-27B · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [qwen38-27b-exl3-k5k6-pair](../profiles/qwen38-27b-exl3-k5k6-pair/README.md) |

</details>

<details>
<summary>Qwen3.8-27B · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Development | [qwen38-27b-exl3-k5k6](../profiles/qwen38-27b-exl3-k5k6/README.md) |

</details>

<details>
<summary>Swift-1.5-Qwen3.8-Flash-Next · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-pair-2 | Off | Development | [swift15-qwen38-flash-next-tp2 (default)](../profiles/swift15-qwen38-flash-next-tp2/README.md) |

</details>

<details>
<summary>Swift-1.5-Qwen3.8-Flash-Next · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | direct-cycle-4 | Off | Experimental | [swift15-qwen38-flash-next-tp4 (default)](../profiles/swift15-qwen38-flash-next-tp4/README.md) |

</details>

### Retired profiles

<details>
<summary>Retired configurations</summary>

Retained for compatibility and historical evidence; use an active deployment above for setup.

- [glm52-exl3-r7-3.5bpw](../profiles/glm52-exl3-r7-3.5bpw/README.md) — GLM-5.2; Development
- [glm53-flash-nvfp4-dflash2-bf16-tp4](../profiles/glm53-flash-nvfp4-dflash2-bf16-tp4/README.md) — GLM-5.3-Flash; Development
- [glm53-flash-spark-tp4-dcp4](../profiles/glm53-flash-spark-tp4-dcp4/README.md) — GLM-5.3-Flash; Development
- [glm53-flash-spark-tp4-dcp4-sparkcache](../profiles/glm53-flash-spark-tp4-dcp4-sparkcache/README.md) — GLM-5.3-Flash; Validated
- [glm53-mtp3-cache-checkpoints-tp4](../profiles/glm53-mtp3-cache-checkpoints-tp4/README.md) — GLM-5.3-Flash; Experimental
- [glm53-spark-mtp3-managed-mesh-tp4](../profiles/glm53-spark-mtp3-managed-mesh-tp4/README.md) — GLM-5.3-Flash; Experimental
- [sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4](../profiles/sparkcache-glm52-exl3-r7-3.5bpw-sparkcache-tp4-dcp4/README.md) — GLM-5.2; Development
- [sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4](../profiles/sparkcache-glm53-flash-nvfp4-dflash2-bf16-sparkcache-tp4/README.md) — GLM-5.3-Flash; Validated

[Additional historical variants](../docs/history/deployment-variants.md)

</details>

Qwen Flash Next TP2 offers optional SparkCache with bounded text/media persistence validation. Six-node deployments remain experimental and are outside this catalog.

<!-- END GENERATED PROFILES -->

\* KV capacity depends on configuration and enabled features, including SparkCache.
Linked figures retain their measurement conditions.
