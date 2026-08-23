# Deployment profiles

A profile names one model identity, hardware topology, serving configuration,
and evidence scope. SparkRing supports exactly the following six profiles officially. 

| Profile | Topology | Status and evidence scope | Documentation |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | four-Spark cycle, TP4/DCP4 | candidate at 16 sequences; historical eight-sequence qualification retained | [Recipe](../../recipes/glm52-exl3-r7-3.5bpw.json), [quickstart](../GLM52_35BPW_QUICKSTART.md), [serving contract](../GLM52_35BPW_FIXED_MTP4_PROFILE.md) |
| DeepSeek-V4-Flash-0731 | two-Spark pair, TP2/DCP1 | candidate normalized settings; checkpoint revision unpinned; SIRCL unsupported | [Recipe](../../recipes/deepseek-v4-flash-0731-pair.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | four-Spark cycle, TP4/DCP1 | candidate normalized settings; checkpoint revision unpinned; SIRCL width 4096 research-only | [Recipe](../../recipes/deepseek-v4-flash-0731.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md), [profile record](DEEPSEEK_V4_FLASH_0731.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | four-Spark cycle, TP4/DCP4 | candidate at 1M context/16 sequences; historical 262K/eight-sequence durable-state receipt retained | [Recipe](../../recipes/sparkcache/glm52-exl3-r7-3.5bpw-tp4-dcp4.json), [composition evidence](../../recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | two-Spark pair, TP2/DCP1 | candidate at 1M context/32 sequences; historical 131K/six-sequence receipt retained | [Recipe](../../recipes/sparkcache/deepseek-v4-flash-0731-tp2-dcp1.json), [composition evidence](../../recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | four-Spark cycle, TP4/DCP1 | candidate at 1M context; historical 524K receipt retained | [Recipe](../../recipes/sparkcache/deepseek-v4-flash-0731-tp4-dcp1.json), [composition evidence](../../recipes/sparkcache/README.md) |

The GLM profiles pin model revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. The DeepSeek base recipes do
not pin a checkpoint revision; each SparkCache composition records the exact
checkpoint hash used by its qualification gate.
