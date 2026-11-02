# Deployment profiles

SparkRing's profiles run three models created by Local Inference Lab, Inc. and
licensed under the Local Inference Lab License, Version 1.0:
[GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)
(`glm53-flash-nvfp4-spark-tp2`, `glm53-flash-nvfp4-spark-tp4`,
`glm53-flash-nvfp4-spark-tp2-dcp2`, `glm53-flash-csf-tp8`,
`glm53-flash-nvfp4-sparkcache-tp2`, `glm53-flash-nvfp4-sparkcache-tp4`),
[GLM-5.3-NVFP4-CSF](https://huggingface.co/local-inference-lab/GLM-5.3-NVFP4-CSF)
(`glm53-nvfp4-csf-tp6`, `glm53-nvfp4-csf-tp8`) and
[Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)
(`qwen38-flash-next-tp2`, `qwen38-flash-next-csf-sparkcache-tp2`).
The Experimental Qwen SparkCache installer variants and their build and
qualification requirements are documented in the [hardware plan](../docs/research/qwen38-sparkcache.md).

A profile is a saved configuration of one model on a fixed number of DGX
Sparks: checkpoint, serving image, parallelism, cabling and serving settings.
Its ID names it in commands and records. Installer profiles are the ones that
`sparkring install --profile ID` sets up on its default image
([Install SparkRing](../docs/operations/install.md)); every other profile has
its own guide. [`catalog.json`](catalog.json) and
[`research-catalog.json`](research-catalog.json) list every profile, and
[`scripts/generate_profiles.py`](../scripts/generate_profiles.py) generates
the tables below from them; edit the profiles, not the tables.

## Reading the tables

- **Status.** Validated is `qualified`: documented checks passed for the exact
  image, checkpoint, topology and workload in the linked guide. Development is
  `implemented`; Experimental is `research-only`.
- **Bold rows** are profiles marked recommended, which the variant tables
  label `(default)`; other rows are alternatives.
- **Quant** links the checkpoint repository; Stock is the publisher's original
  checkpoint. Profiles pin exact revisions.
- **Installer-profile rows** show what `sparkring install` installs on its
  default image, release 2026.10.5's
  `dev-20261016-sparkcache-4a17f1f5198e-status038`: the checkpoint, its
  configured context and its recorded KV pool. Other rows show the profile's
  own default.
- **DCP** is the decode-context-parallel size, the number of Sparks that split
  each request's KV cache. A row with several values lists its default first.
- **Context / KV.** Context is the configured per-request limit, not a
  completed long-context test. KV is the engine-reported KV-cache pool, in
  tokens, from the linked record.

The GLM-5.3-Flash installer profiles of two and four Sparks and
`qwen38-flash-next-tp2` install Local Inference Lab's CSF checkpoint on an
image whose vLLM reads it, the default image among them, and their own default
checkpoint (NVFP4-Spark, or Qwen QAD step 5500) on every other image.
`--checkpoint` selects another listed checkpoint, such as
[NVIDIA's NVFP4 GLM-5.3-Flash](glm53-nvidia-nvfp4.md) on four Sparks
([GLM-5.3-Flash checkpoints](glm53-checkpoints.md),
[names and settings](../docs/operations/install-reference.md#another-checkpoint-of-a-profile)).

Shorter aliases select some profiles by model and Spark count
([`profile_aliases.py`](../runtime/common/profile_aliases.py)):

| Alias | Profile ID |
|---|---|
| `glm53-flash-tp2` | `glm53-flash-nvfp4-spark-tp2` |
| `glm53-flash-tp2-dcp2` | `glm53-flash-nvfp4-spark-tp2-dcp2` |
| `glm53-flash-tp4` | `glm53-flash-nvfp4-spark-tp4` |
| `qwen38-flash-next-tp4` | `qwen38-flash-next-qad-tp4` |
| `qwen38-flash-next-tp8` | `qwen38-flash-next-qad-tp8` |

[Configuration guide](../docs/development/configuration.md): inspect defaults
and prepare private site inputs. [Benchmarks](../performance/benchmarks.md):
measured results. [Contributing an installer profile](../docs/development/installer-profiles.md):
add a profile that `sparkring install` sets up.

<!-- BEGIN GENERATED PROFILES -->

### Four Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status |
|---|---|---|---|---|---|
| **[DeepSeek-V4.1-Flash](../docs/operations/install.md)**<br>vLLM | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)<br>by DeepSeek | 1 | 1M / [12.8M](../performance/records/images/dev-20261013-dcp2prefill-status038-sircl-tp6-deepseek-v41-flash-tp4-kv-ring8-20261014.md) | No | Development |
| **[GLM-5.3-Flash](../docs/operations/install.md)**<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1 | 1M / [5.7M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-tp4-20261013.md) | No | Development |
| **[GLM-5.3-Flash](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)<br>by Local Inference Lab | 1 | 1M / [2.3M](../runtime/releases/shared-2026.09.3/correctness.json) | [Optional](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) | Validated |
| **[MiMo-V2.6-Flash-MOPD](../docs/operations/install.md)**<br>vLLM | [MXFP8/BF16](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)<br>by Xiaomi MiMo | 1 | 262K / [2.2M](../performance/records/images/dev-20261004-kraken-matrix-20261004.md) | No | Development |
| **[Qwen3.8-Flash-Next](../profiles/qwen38-flash-next-qad-tp4/README.md)**<br>vLLM | [NVFP4 QAD step 5500 PLE](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4/tree/60215d26cf5e42c2db6128774032d57fc62678da)<br>by Local Inference Lab | 1 | 262K / [2.8M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-release-2026.10.3-plain-tp4-20261013.md) | No | Development |
| **[Swift-1.5-Qwen3.8-Flash-Next](../profiles/swift15-qwen38-flash-next-tp4/README.md)**<br>vLLM | [NVFP4 experts/BF16](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4)<br>by UkisAI | 1 | 262K / [3.1M](../performance/records/images/dev-20261004-kraken-matrix-20261004.md) | No | Experimental |
| [DeepSeek-V4-Flash-0731](../profiles/deepseek-v4-flash-0731/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)<br>by DeepSeek | 1 | 1M / [1M](../performance/capacity-references.md) | [Optional](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) | Development |
| [DeepSeek-V4-Flash-Vision-Exp](../profiles/deepseek-v4-flash-vision-exp-tp4/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp)<br>by DeepSeek | 1 | 1M / — | No | Experimental |
| [DeepSeek-V4.1-Flash](../profiles/deepseek-v41-flash-sglang-cycle/README.md)<br>SGLang | [FP8/MXFP4](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)<br>by DeepSeek | — | 262K / [1.5M](../performance/records/deepseek-v41-flash/sglang-soak-20260912.md) | No | Development |
| [GLM-5.3-Flash](../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp4/README.md)<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1 | 1M / — | Included | Experimental |
| [GLM-5.3-Flash](../profiles/glm53-flash-nvfp4-sparkcache-tp4/README.md)<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1 | 1M / — | Included | Experimental |
| [Qwen3.8-27B](../profiles/qwen38-27b-exl3-k5k6/README.md)<br>vLLM | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)<br>by malaiwah | 1 | 1M / [8.7M](../profiles/qwen38-27b-exl3-k5k6/recipe.json) | No | Development |

### Two Sparks

| Model | Quant | DCP | Context / KV* | SparkCache | Status |
|---|---|---|---|---|---|
| **[GLM-5.3-Flash](../docs/operations/install.md)**<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1 | 1M / [1.5M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md) | No | Development |
| **[GLM-5.3-Flash](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md)**<br>vLLM | [NVFP4-Spark](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-Spark)<br>by Local Inference Lab | 1 | 1M / [1.1M](../runtime/releases/shared-2026.09.3/correctness.json) | [Optional](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) | Validated |
| **[MiMo-V2.6-Flash-MOPD](../docs/operations/install.md)**<br>vLLM | [MXFP8/BF16](https://huggingface.co/XiaomiMiMo/MiMo-V2.6-Flash-MOPD)<br>by Xiaomi MiMo | 1 | 262K / [656K](../performance/records/images/dev-20261004-kraken-matrix-20261004.md) | No | Development |
| **[Qwen3.8-Flash-Next](../profiles/qwen38-flash-next-tp2/README.md)**<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1 | 262K / [2.5M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-kv-pools-20261012.md) | No | Development |
| **[Swift-1.5-Qwen3.8-Flash-Next](../profiles/swift15-qwen38-flash-next-tp2/README.md)**<br>vLLM | [NVFP4 experts/BF16](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-NVFP4)<br>by UkisAI | 1 | 262K / [2.9M](../performance/records/images/dev-20261004-kraken-matrix-20261004.md) | No | Development |
| [DeepSeek-V4-Flash-0731](../profiles/deepseek-v4-flash-0731-pair/README.md)<br>vLLM | [Stock](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)<br>by DeepSeek | 1 | 1M / [2.2M](../performance/records/deepseek-v4-flash/image827a8e8c-tp2.json) | [Optional](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) | Development |
| [GLM-5.3-Flash](../profiles/glm53-flash-nvfp4-spark-tp2-dcp2/README.md)<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 2/1 | 1M / ([4.1M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md)/—) | No | Experimental |
| [GLM-5.3-Flash](../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp2/README.md)<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1/2 | 1M / (—/[4.1M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md)) | Included | Experimental |
| [GLM-5.3-Flash](../profiles/glm53-flash-nvfp4-sparkcache-tp2/README.md)<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1/2 | 1M / (—/[4.1M](../performance/records/images/dev-20261010-kraken-csf-sircl032-libsircl060cd-plugins-status036-glm53-flash-nvfp4-spark-tp2-dcp2-pair-20261013.md)) | Included | Experimental |
| [Qwen3.8-27B](../profiles/qwen38-27b-exl3-k5k6-pair/README.md)<br>vLLM | [EXL3 K5/K6](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated)<br>by malaiwah | 1 | 1M / [4.1M](../profiles/qwen38-27b-exl3-k5k6-pair/recipe.json) | No | Development |
| [Qwen3.8-Flash-Next](../profiles/qwen38-flash-next-csf-sparkcache-tp2/README.md)<br>vLLM | [NVFP4/MXFP8 CSF](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-MXFP8-CSF-QAD)<br>by Local Inference Lab | 1 | 262K / — | Included | Experimental |


Status and context describe the linked default. DCP and KV figures follow the same order;
capacity depends on enabled features. Expand a deployment below for each option’s own status and guide.
Switched support is a separate network configuration and has no switched-hardware qualification.

## Configuration variants

Network names the cabling: a pair is two Sparks cabled to each other, a ring connects each Spark
to the next, and a path is a ring without its last cable ([requirements](../docs/operations/install.md#requirements)).

Profile IDs identify saved configurations. Guide status describes the primary quickstart;
record links preserve configuration evidence when the guide selects a different release.

<details>
<summary>MiMo-V2.6-Flash-MOPD · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | Off | Development | [mimo-v26-flash-mopd-tp2 (default)](../docs/operations/install.md) |

</details>

<details>
<summary>MiMo-V2.6-Flash-MOPD · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Development | [mimo-v26-flash-mopd-tp4 (default)](../docs/operations/install.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-0731 · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | Off | Development | [deepseek-v4-flash-0731-pair](../profiles/deepseek-v4-flash-0731-pair/README.md) |
| DCP1 | pair | On | Development | [sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp2-dcp1/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-0731 · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Development | [deepseek-v4-flash-0731](../profiles/deepseek-v4-flash-0731/README.md) |
| DCP1 | ring of 4 | On | Development | [sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1](../profiles/sparkcache-deepseek-v4-flash-0731-sparkcache-tp4-dcp1/README.md) |

</details>

<details>
<summary>DeepSeek-V4-Flash-Vision-Exp · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Experimental | [deepseek-v4-flash-vision-exp-tp4](../profiles/deepseek-v4-flash-vision-exp-tp4/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 4 Sparks · SGLang</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| EP4 | ring of 4 | Off | Development | [deepseek-v41-flash-sglang-cycle](../profiles/deepseek-v41-flash-sglang-cycle/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Development | [deepseek-v41-flash-tp4 (default)](../docs/operations/install.md) |
| DCP1 | ring of 4 | Off | Development | [deepseek-v41-flash-cycle](../profiles/deepseek-v41-flash-cycle/README.md) |
| DCP1 | ring of 4 | On | Experimental | [deepseek-v41-flash-sparkcache-tp4](../profiles/deepseek-v41-flash-sparkcache-tp4/README.md) |

</details>

<details>
<summary>DeepSeek-V4.1-Flash · 8 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 8 | Off | Experimental | [deepseek-v41-flash-tp8 (default)](../profiles/deepseek-v41-flash-tp8/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash (NVFP4/MXFP8 CSF) · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP2 | pair | Off | Experimental | [glm53-flash-nvfp4-spark-tp2-dcp2](../profiles/glm53-flash-nvfp4-spark-tp2-dcp2/README.md) |
| DCP1 | pair | On | Experimental | [glm53-flash-nvfp4-sparkcache-crc32-tp2](../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp2/README.md) |
| DCP1 | pair | On | Experimental | [glm53-flash-nvfp4-sparkcache-tp2](../profiles/glm53-flash-nvfp4-sparkcache-tp2/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash (NVFP4/MXFP8 CSF) · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | On | Experimental | [glm53-flash-nvfp4-sparkcache-crc32-tp4](../profiles/glm53-flash-nvfp4-sparkcache-crc32-tp4/README.md) |
| DCP1 | ring of 4 | On | Experimental | [glm53-flash-nvfp4-sparkcache-tp4](../profiles/glm53-flash-nvfp4-sparkcache-tp4/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash · 8 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 8 | Off | Experimental | [glm53-flash-csf-tp8 (default)](../profiles/glm53-flash-csf-tp8/README.md) |

</details>

<details>
<summary>GLM-5.3-Flash (NVFP4-Spark) · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | Off | Development | [glm53-flash-nvfp4-spark-tp2 (default)](../docs/operations/install.md) |
| DCP1 | pair (RoCEnante adaptive bundle) | On | Validated | [glm53-flash-spark-tp2-dcp1-sparkcache (default)](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/README.md) · [record](../profiles/glm53-flash-spark-tp2-dcp1-sparkcache/profile.json) |
| DCP1 | pair (RoCEnante adaptive bundle) | Off | Experimental | [glm53-flash-spark-tp2-dcp1](../profiles/glm53-flash-spark-tp2-dcp1/README.md) · [record](../profiles/glm53-flash-spark-tp2-dcp1/profile.json) |
| DCP1 | pair (RoCEnante adaptive bundle) | Off | Experimental | [glm53-flash-spark-tp2-dcp1-nocache](../profiles/glm53-flash-spark-tp2-dcp1-nocache/README.md) · [record](../profiles/glm53-flash-spark-tp2-dcp1-nocache/profile.json) |

</details>

<details>
<summary>GLM-5.3-Flash (NVFP4-Spark) · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Development | [glm53-flash-nvfp4-spark-tp4 (default)](../docs/operations/install.md) |
| DCP1 | RoCEnante mesh (SIRCL carries most collectives) | On | Validated | [glm53-flash-spark-tp4-dcp1-sparkcache (default)](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/README.md) · [record](../profiles/glm53-flash-spark-tp4-dcp1-sparkcache/profile.json) |
| DCP1 | RoCEnante mesh (SIRCL carries most collectives) | Off | Experimental | [glm53-flash-spark-tp4-dcp1](../profiles/glm53-flash-spark-tp4-dcp1/README.md) · [record](../profiles/glm53-flash-spark-tp4-dcp1/profile.json) |
| DCP1 | RoCEnante mesh (SIRCL carries most collectives) | Off | Experimental | [glm53-flash-spark-tp4-dcp1-nocache](../profiles/glm53-flash-spark-tp4-dcp1-nocache/README.md) · [record](../profiles/glm53-flash-spark-tp4-dcp1-nocache/profile.json) |
| DCP1 | switched | Off | Experimental | [glm53-flash-spark-tp4-switched](../profiles/glm53-flash-spark-tp4-switched/README.md) |

</details>

<details>
<summary>GLM-5.3 (NVFP4 experts/BF16) · 8 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP4 | ring of 8 | Off | Experimental | [glm53-nvfp4-tp8 (default)](../profiles/glm53-nvfp4-tp8/README.md) |
| DCP1 | ring of 8 | Off | Experimental | [glm53-nvfp4-tp8-dcp1](../profiles/glm53-nvfp4-tp8-dcp1/README.md) |

</details>

<details>
<summary>GLM-5.3 · 6 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP2 | path of 6 | Off | Experimental | [glm53-nvfp4-csf-tp6](../profiles/glm53-nvfp4-csf-tp6/README.md) |

</details>

<details>
<summary>GLM-5.3 (NVFP4 CSF experts/BF16) · 8 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP4 | ring of 8 | Off | Experimental | [glm53-nvfp4-csf-tp8](../profiles/glm53-nvfp4-csf-tp8/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next (NVFP4) · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | Off | Development | [qwen38-flash-next-tp2 (default)](../profiles/qwen38-flash-next-tp2/README.md) |
| DCP1 | pair | On | Validated | [qwen38-flash-next-tp2-sparkcache](../profiles/qwen38-flash-next-tp2/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Development | [qwen38-flash-next-qad-tp4 (default)](../profiles/qwen38-flash-next-qad-tp4/README.md) |
| DCP1 | ring of 4 | On | Validated | [qwen38-flash-next-qad-tp4-sparkcache](../profiles/qwen38-flash-next-qad-tp4-sparkcache/README.md) |
| DCP1 | ring of 4 | On | Experimental | [qwen38-flash-next-step5500-sparkcache-tp4](../profiles/qwen38-flash-next-step5500-sparkcache-tp4/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next · 8 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 8 | Off | Experimental | [qwen38-flash-next-qad-tp8 (default)](../profiles/qwen38-flash-next-qad-tp8/README.md) |

</details>

<details>
<summary>Qwen3.8-Flash-Next (NVFP4/MXFP8 CSF) · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | On | Experimental | [qwen38-flash-next-csf-sparkcache-tp2](../profiles/qwen38-flash-next-csf-sparkcache-tp2/README.md) |

</details>

<details>
<summary>Qwen3.8-27B · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | Off | Development | [qwen38-27b-exl3-k5k6-pair](../profiles/qwen38-27b-exl3-k5k6-pair/README.md) |

</details>

<details>
<summary>Qwen3.8-27B · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Development | [qwen38-27b-exl3-k5k6](../profiles/qwen38-27b-exl3-k5k6/README.md) |

</details>

<details>
<summary>Swift-1.5-Qwen3.8-Flash-Next · 2 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | pair | Off | Development | [swift15-qwen38-flash-next-tp2 (default)](../profiles/swift15-qwen38-flash-next-tp2/README.md) |

</details>

<details>
<summary>Swift-1.5-Qwen3.8-Flash-Next · 4 Sparks · vLLM</summary>

| Parallelism | Network | SparkCache | Guide status | Configuration and guide |
|---|---|---|---|---|
| DCP1 | ring of 4 | Off | Experimental | [swift15-qwen38-flash-next-tp4 (default)](../profiles/swift15-qwen38-flash-next-tp4/README.md) |

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

The Qwen3.8-Flash-Next SparkCache variants of two and four Sparks are qualified only for bounded correctness, restart and cache-restore checks on release shared-2026.09.3.
Profiles of six and eight Sparks are research-only and appear only under Configuration variants.

<!-- END GENERATED PROFILES -->

The Experimental GLM SparkCache CRC32 profiles for
[two Sparks](glm53-flash-nvfp4-sparkcache-crc32-tp2/README.md) and
[four Sparks](glm53-flash-nvfp4-sparkcache-crc32-tp4/README.md) run on the
default image, release 2026.10.5's, whose lock admits them. Their separate IDs
preserve the earlier variants' image-specific evidence. The two-Spark profile
installed and passed `sparkring check` on that image composition's recheck
build ([record](../performance/records/images/dev-20261016-sparkcache-4a17f1f5198e-status038-sparkcache-recovery-ring8-20261016.md)); the four-Spark profile has no hardware result.

\* KV capacity depends on configuration and enabled features, including
SparkCache, and does not show that concurrent full-context requests fit. Each
linked record states its measurement conditions.
