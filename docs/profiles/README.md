# Four-Spark profiles

A profile names one model identity, its four-rank serving configuration, and the
evidence scope for that configuration. SparkRing supports exactly the following
two profiles.

| Profile | Model identity | Status and evidence scope | Documentation |
|---|---|---|---|
| GLM-5.2 EXL3 3.5-bpw | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | qualified on one four-DGX-Spark appliance; a rebuilt image retains implemented status until promotion | [Quickstart](../GLM52_35BPW_QUICKSTART.md), [serving contract](../GLM52_35BPW_FIXED_MTP4_PROFILE.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` (no revision pinned) | implemented four-Spark launch; SIRCL width 4096 is research-only | [Recipe](../../recipes/deepseek-v4-flash-0731.json), [quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md), [profile record](DEEPSEEK_V4_FLASH_0731.md) |
