# Serving recipes

The JSON files in this directory are machine-readable deployment contracts.
They record immutable artifacts, model identities, topology, serving values,
evidence, and limitations for supported SparkRing profiles.

Operators should begin with the linked quickstart. Recipes are useful for
automation, inspection, and reproducibility; they are not command-by-command
installation guides.

| Model profile | Status | Topology | Recipe | Operator guide |
|---|---|---|---|---|
| GLM-5.3 Flash NVFP4 + BF16 DFlash2 | implemented; DCP4 preferred | four Sparks, TP4 with DCP1/DCP2/DCP4 | [`glm53-flash-nvfp4-dflash2-bf16-tp4.json`](glm53-flash-nvfp4-dflash2-bf16-tp4.json) | [GLM-5.3 quickstart](../docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) |
| GLM-5.2 EXL3 3.5-bpw | implemented | four Sparks, TP4/DCP4 | [`glm52-exl3-r7-3.5bpw.json`](glm52-exl3-r7-3.5bpw.json) | [GLM-5.2 quickstart](../docs/GLM52_35BPW_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | implemented | two Sparks, TP2/DCP1 | [`deepseek-v4-flash-0731-pair.json`](deepseek-v4-flash-0731-pair.json) | [DeepSeek quickstart](../docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | implemented | four Sparks, TP4/DCP1 | [`deepseek-v4-flash-0731.json`](deepseek-v4-flash-0731.json) | [DeepSeek quickstart](../docs/DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | implemented | two Sparks, TP2/DCP1 | [`qwen38-27b-exl3-k5k6-pair.json`](qwen38-27b-exl3-k5k6-pair.json) | [Qwen pair quickstart](../docs/QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md) |
| Qwen3.8-27B EXL3 K5/K6 | implemented | four Sparks, TP4/DCP1 | [`qwen38-27b-exl3-k5k6.json`](qwen38-27b-exl3-k5k6.json) | [Qwen quickstart](../docs/QWEN38_27B_EXL3_K5K6_QUICKSTART.md) |

[`sparkcache/`](sparkcache/) contains compositions that add persistent,
rank-local prefix storage to a base recipe. A composition may have a narrower
evidence scope than its base serving profile.

Status applies to the exact profile and evidence named by each file:

- `implemented`: the repository provides a complete launch contract and
  GPU-free validation.
- `qualified`: the named immutable artifact also has recorded live hardware
  evidence under the stated conditions.
- `research-only`: the file records exploratory behavior that is not an
  operator default.
- `unsupported`: no working integration is published.
