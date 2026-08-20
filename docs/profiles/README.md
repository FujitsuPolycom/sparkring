# Validated-profiles registry

SparkRing is the four-Spark transport and runtime; a **profile** is one model
identity plus the serving configuration under which it has been exercised
against that runtime, together with the evidence scope that exercise
established. This registry gives each profile one present-state row and, where
no dedicated document already exists, one page.

This registry is a navigation index, not an independent source of
configuration, maturity, or measurement truth. The canonical claim owners
remain the linked documents; when this index and a linked document disagree,
the linked document wins and the drift should be reported. The three GLM-5.2
lanes keep their existing quickstart and profile documents as their profile
pages; only the entries without an existing owner document have pages under
this directory.

## Registry

| Profile | Model identity | Hidden width | Maturity / evidence scope | Profile page(s) |
|---|---|---:|---|---|
| GLM-5.2 EXL3 3.25-bpw plus LMCache with 512-token cache chunks (CS512) (public default) | `willfalco/GLM-5.2-EXL3-TR3-3.25bpw@d7d79c2d14599dfce7a5d12b85f7ad73f40e623d` | 6144 | Default advertised public-functional configuration; bounded clean-checkout four-Spark live-validated; not fully accepted | [Public-default quickstart](../QUICKSTART.md), [EXL3 3.25-bpw recipe](../EXL3_RECIPE.md) |
| GLM-5.2 EXL3 3.5-bpw R7 fixed-MTP4 (operator profile) | `brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f` | 6144 | Accepted operator default; live-validated, acceptance scoped to the operator's four-Spark appliance; not the repository-wide public default | [Fixed-MTP4 profile](../EXL3_R7_FIXED_MTP4_PROFILE.md), [EXL3 3.5-bpw quickstart](../EXL3_R7_QUICKSTART.md) |
| GLM-5.2 NF3 hybrid (deterministic alternative) | `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid@66f3623dd8fefb5ca8046706912d5d31c8d196af` | 6144 | Accepted deterministic alternative; clean-checkout public bootstrap live-validated (`nvfp4-rope8` KV profile) | [NF3 quickstart](../NF3_QUICKSTART.md), [NF3 public validation](../NF3_NVFP4_PUBLIC_VALIDATION.md) |
| openai/gpt-oss-120b | `openai/gpt-oss-120b` (no revision pinned in this repository) | 2880 | Functional launch attested (operator record, 2026-08-17): loads, decodes, and prefills on the ring; not shadow-qualified, no numerical or performance claim | [GPT_OSS_120B.md](GPT_OSS_120B.md) |
| Shadow validation set (four small models) | `EleutherAI/pythia-70m`, `facebook/opt-125m`, `Qwen/Qwen3-0.6B`, `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 512 / 768 / 1024 / 2048 | Validation instruments, not serving recommendations; four-Spark shadow-mode windows executed 2026-08-17 with per-width gate outcomes | [SHADOW_VALIDATION_SET.md](SHADOW_VALIDATION_SET.md) |
| DeepSeek-V4-Flash-0731 | `deepseek-ai/DeepSeek-V4-Flash-0731` (no revision pinned in this repository) | 4096 | Functional launch attested (operator record, 2026-08-18): serves on the ring with an fp8 key-value cache and native DSpark speculation; a research-only configuration routes decode collectives through the width-4096 graph session with gate evidence on the profile page; not shadow-qualified | [DEEPSEEK_V4_FLASH_0731.md](DEEPSEEK_V4_FLASH_0731.md) |

The GLM-5.2 hidden width of 6144 is the historical default eager all-reduce
admission (`[Q, 6144]` BF16) documented in the
[vLLM integration README](../../spark_transport/integrations/vllm/README.md).
Non-6144 widths are admitted only through the opt-in
`VLLM_SPARK_TP4_EAGER_WIDTHS` mechanism described there and validated by the
[eager width admission validation runbook](../EAGER_WIDTH_VALIDATION_RUNBOOK.md).

## Evidence-scope vocabulary

Maturity labels in this registry reuse the repository's existing vocabulary:
**live-validated** (evidence gathered on the four directly cabled Sparks under
a stated gate), **offline-validated** (evidence from GPU-free suites or
offline comparison, no live cluster), **validation-grade** instruments
(profiles that exist to produce evidence, not to be served), and the status
labels **implemented / qualified / research-only / unsupported** from
[Write Without Hidden Context](../history/WRITING_STANDARD.md). A label here restates
the linked document's scope; it never widens it.
