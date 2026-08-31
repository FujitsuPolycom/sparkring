# Deployment profiles

A profile names one model identity, hardware topology, serving configuration,
and evidence scope. SparkRing publishes the following ten base and
composition profiles.

| Profile | Topology | Status and evidence scope | Documentation |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | four-Spark cycle, TP4/DCP4 | 1,048,576-token/16-sequence profile; benchmark results through C8 | [Recipe](../../recipes/glm52-exl3-r7-3.5bpw.json), [quickstart](../GLM52_35BPW_QUICKSTART.md), [serving contract](../GLM52_35BPW_FIXED_MTP4_PROFILE.md) |
| DeepSeek-V4-Flash DSpark | two-Spark pair, TP2/DCP1 | `913f0657…`; live-benchmarked; SIRCL unsupported | [Recipe](../../recipes/deepseek-v4-flash-0731-pair.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md) |
| DeepSeek-V4-Flash-0731 | four-Spark cycle, TP4/DCP1 | `7872f01…`; live-benchmarked; SIRCL width 4096 research-only | [Recipe](../../recipes/deepseek-v4-flash-0731.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md), [profile record](DEEPSEEK_V4_FLASH_0731.md) |
| Qwen3.8-27B EXL3 K5/K6 | two-Spark pair, TP2/DCP1 | 1,048,576-token profile; benchmarked through C8 | [Recipe](../../recipes/qwen38-27b-exl3-k5k6-pair.json), [builder](../../runtime/qwen38/README.md), [quickstart](../QWEN38_27B_EXL3_K5K6_PAIR_QUICKSTART.md), [profile record](QWEN38_27B_EXL3_K5K6_PAIR.md) |
| Qwen3.8-27B EXL3 K5/K6 | four-Spark cycle, TP4/DCP1 | 1,048,576-token profile; benchmarked through C8; SIRCL unsupported | [Recipe](../../recipes/qwen38-27b-exl3-k5k6.json), [builder](../../runtime/qwen38/README.md), [quickstart](../QWEN38_27B_EXL3_K5K6_QUICKSTART.md), [profile record](QWEN38_27B_EXL3_K5K6.md) |
| Published JJ r7-compatible GLM-5.3 Flash base | four-Spark cycle, TP4/DCP1 | launcher defaults: 524,288-token limit, 16 sequences, 8,192 batched tokens; implemented and unqualified at those limits. Bounded C4 smoke used 262,144 and 4,096 | [Artifacts](../../runtime/glm53-flash-jj-r7-gb10/artifacts.json), [quickstart](../GLM53_JJ_R7_GB10_TP4_QUICKSTART.md), [smoke record](../../performance/records/glm53-flash/jj-r7-gb10-tp4-smoke-20260830.md) |
| GLM-5.2 EXL3 3.5-bpw + SparkCache | four-Spark cycle, TP4/DCP4 | implemented at 1M context/16 sequences; qualified at 262K/eight sequences | [Recipe](../../recipes/sparkcache/glm52-exl3-r7-3.5bpw-tp4-dcp4.json), [composition evidence](../../recipes/sparkcache/README.md) |
| Published JJ r7-compatible GLM-5.3 Flash + SparkCache | four-Spark cycle, TP4/DCP1 | launcher defaults: 524,288-token limit, 16 sequences, 8,192 batched tokens, and 524,288-token cache span; implemented and unqualified at those limits. Bounded C4 smoke used 262,144 and 4,096 | [Artifacts](../../runtime/glm53-flash-jj-r7-gb10/artifacts.json), [quickstart](../GLM53_JJ_R7_GB10_TP4_QUICKSTART.md), [smoke record](../../performance/records/glm53-flash/jj-r7-gb10-tp4-smoke-20260830.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | two-Spark pair, TP2/DCP1 | implemented at 1M context/32 sequences; qualified at 131K/six sequences | [Recipe](../../recipes/sparkcache/deepseek-v4-flash-0731-tp2-dcp1.json), [composition evidence](../../recipes/sparkcache/README.md) |
| DeepSeek-V4-Flash-0731 + SparkCache | four-Spark cycle, TP4/DCP1 | implemented at 1M context/32 sequences; qualified at 524K/32 sequences | [Recipe](../../recipes/sparkcache/deepseek-v4-flash-0731-tp4-dcp1.json), [composition evidence](../../recipes/sparkcache/README.md) |

The GLM-5.2 profiles pin model revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f`. The GLM-5.3 profiles pin target
revision `520de24eabf507659eaef7c70f14fd584527facc` and public BF16 DFlash2
revision `dc77ff1c99eeb2df044ee3d4f0094eb033fee410`. The DeepSeek TP2 recipe records
the DSpark package revision `913f0657…`; the TP4 recipe records the plain 0731
revision `7872f01…`. Each SparkCache composition separately records its exact
checkpoint hash. Both Qwen base profiles pin
revision `ab3a91a13813df8096cb4c1d560ed3669035d0cf` and the checkpoint's
published configuration hash.

## Unsupported integrations

| Integration | Topology | Status |
|---|---|---|
| Qwen3.8-27B EXL3 K5/K6 + SparkCache | four-Spark cycle, TP4/DCP1 | **unsupported.** No composition recipe or live cache evidence is published. |
| Qwen3.8-27B EXL3 K5/K6 + SparkCache | two-Spark pair, TP2/DCP1 | **unsupported.** No composition recipe or live cache evidence is published. |
