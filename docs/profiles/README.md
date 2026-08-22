# Deployment profiles

A profile names one model identity, hardware topology, serving configuration,
and evidence scope. SparkRing supports exactly the following six profiles.

| Profile | Topology | Status and evidence scope | Documentation |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | four-Spark cycle, TP4/DCP4 | qualified on one four-DGX-Spark appliance; a rebuilt image retains implemented status until promotion | [Recipe](../../recipes/glm52-exl3-r7-3.5bpw.json), [quickstart](../GLM52_35BPW_QUICKSTART.md), [serving contract](../GLM52_35BPW_FIXED_MTP4_PROFILE.md) |
| DeepSeek-V4-Flash-0731 | two-Spark pair, TP2/DCP1 | implemented launch; the checkpoint revision is not pinned and SIRCL is unsupported | [Recipe](../../recipes/deepseek-v4-flash-0731-pair.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | four-Spark cycle, TP4/DCP1 | implemented launch; the checkpoint revision is not pinned and SIRCL width 4096 is research-only | [Recipe](../../recipes/deepseek-v4-flash-0731.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md), [profile record](DEEPSEEK_V4_FLASH_0731.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | four-Spark cycle, TP4/DCP4 | qualified durable prefix-state composition for the exact artifacts and settings in the recipe | [Recipe](../../recipes/sparkcache/glm52-exl3-r7-3.5bpw-tp4-dcp4.json), [composition evidence](../../recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | two-Spark pair, TP2/DCP1 | qualified durable prefix-state composition for the exact artifacts and settings in the recipe | [Recipe](../../recipes/sparkcache/deepseek-v4-flash-0731-tp2-dcp1.json), [composition evidence](../../recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | four-Spark cycle, TP4/DCP1 | qualified durable prefix-state composition for the exact artifacts and settings in the recipe | [Recipe](../../recipes/sparkcache/deepseek-v4-flash-0731-tp4-dcp1.json), [composition evidence](../../recipes/sparkcache/README.md) |

The GLM profiles pin model revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. The DeepSeek base recipes do
not pin a checkpoint revision; each SparkCache composition records the exact
checkpoint hash used by its qualification gate.
