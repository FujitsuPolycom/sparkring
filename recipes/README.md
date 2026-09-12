# Serving recipes

The JSON files in this directory are machine-readable deployment contracts.
They record immutable artifacts, model identities, topology, serving values,
evidence, and limitations for supported SparkRing profiles.

Operators should begin with the linked quickstart. Recipes are useful for
automation, inspection, and reproducibility; they are not command-by-command
installation guides.

| Model profile | Status | Topology | Recipe | Operator guide |
|---|---|---|---|---|
| DeepSeek-V4-Flash-Vision-Exp with DSpark | research-only | four Sparks, TP4 cycle | [`deepseek-v4-flash-vision-exp-tp4.json`](deepseek-v4-flash-vision-exp-tp4.json) | [Vision-Exp quickstart](../docs/DEEPSEEK_V4_FLASH_VISION_EXP_TP4_QUICKSTART.md) |
| GLM-5.3 native MTP3 with recurrent checkpoints and optimized SparkCache | research-only | four Sparks, TP4/DCP4 mesh | [Cache/checkpoint recipe](glm53-mtp3-cache-checkpoints-tp4.json) | [Cache/checkpoint quickstart](../docs/GLM53_MTP3_CACHE_CHECKPOINTS_QUICKSTART.md) |
| GLM-5.3 Flash NVFP4-Spark + native MTP3 + managed mesh + SparkCache | research-only | four Sparks, TP4/DCP4, hardware-forwarded opposite peers | [`glm53-spark-mtp3-managed-mesh-tp4.json`](glm53-spark-mtp3-managed-mesh-tp4.json) | [Managed-mesh quickstart](../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md) |
| GLM-5.3 Flash NVFP4 + BF16 DFlash2 | implemented; DCP4 preferred | four Sparks, TP4 with DCP1/DCP2/DCP4 | [`glm53-flash-nvfp4-dflash2-bf16-tp4.json`](glm53-flash-nvfp4-dflash2-bf16-tp4.json) | [GLM-5.3 quickstart](../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw | implemented | four Sparks, TP4/DCP4 | [`glm52-exl3-r7-3.5bpw.json`](glm52-exl3-r7-3.5bpw.json) | [GLM-5.2 quickstart](../docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | implemented | two Sparks, TP2/DCP1 | [`deepseek-v4-flash-0731-pair.json`](deepseek-v4-flash-0731-pair.json) | [DeepSeek quickstart](../docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | implemented | four Sparks, TP4/DCP1 | [`deepseek-v4-flash-0731.json`](deepseek-v4-flash-0731.json) | [DeepSeek quickstart](../docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4.1-Flash, SGLang decoder-tail replay, DSpark block five | implemented (self-built SGLang image; deployment qualification ongoing) | four Sparks, TP4/EP4 cycle | [`deepseek-v41-flash-sglang-cycle.json`](deepseek-v41-flash-sglang-cycle.json) | [SGLang operator guide](../runtime/deepseek-v41-sglang/README.md) |
| DeepSeek-V4.1-Flash vLLM fallback, Engram on NVMe, DSpark k=5 | implemented (self-built stock vLLM image) | four Sparks, TP4/DCP1 | [`deepseek-v41-flash-cycle.json`](deepseek-v41-flash-cycle.json) | [DeepSeek-V4.1 quickstart](../docs/DEEPSEEK_V41_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | implemented | two Sparks, TP2/DCP1 | [`qwen38-27b-exl3-k5k6-pair.json`](qwen38-27b-exl3-k5k6-pair.json) | [Qwen pair quickstart](../docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | implemented | four Sparks, TP4/DCP1 | [`qwen38-27b-exl3-k5k6.json`](qwen38-27b-exl3-k5k6.json) | [Qwen quickstart](../docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |

The Vision-Exp [artifact contract](../runtime/deepseek-vision-exp/profile.json)
records the Anemll serving image, MiaAI-Lab recipe, and SparkRing patched NCCL.
Contributor-reported observations are retained; independent reproduction of
the selected composition is not claimed.

[`sparkcache/`](sparkcache/) contains compositions that add persistent,
rank-local prefix storage to a base recipe. A composition may have a narrower
evidence scope than its base serving profile.

## Native MTP3 mesh profile

The GLM-5.3 Flash NVFP4-Spark native-MTP3 mesh deployment is **research-only**.
The [serving recipe](glm53-spark-mtp3-managed-mesh-tp4.json) indexes its settings and evidence.
Its executable machine-readable inputs are the [profile pins](../runtime/glm53-spark-mtp3-mesh/pins.json)
and [site template](../runtime/glm53-spark-mtp3-mesh/site.example.json), consumed
by the [profile renderer](../runtime/glm53-spark-mtp3-mesh/README.md).
The renderer consumes the dedicated mesh site contract rather than interpreting recipe JSON directly.
The [operator quickstart](../docs/GLM53_SPARK_MTP3_MESH_QUICKSTART.md) covers
TP4/DCP4 serving, native MTP depth three, source-bound transport composition,
and managed host-fabric installation, readiness, shutdown and recovery.
Healthy managed forwarding has no scheduled expiry. Hot replacement under
live RDMA sessions and unattended high availability are unsupported.

## Status definitions

Status applies to the exact profile and evidence named by each file:

- `implemented`: the repository provides a complete launch contract and
  GPU-free validation.
- `qualified`: the named immutable artifact also has recorded live hardware
  evidence under the stated conditions.
- `research-only`: the file records exploratory behavior that is not an
  operator default.
- `unsupported`: no working integration is published.

[Profile validation: performance, accuracy, and restart checks](../docs/PROFILE_VALIDATION.md).
