# Profile: deepseek-ai/DeepSeek-V4-Flash-0731

Status: **research-only / unsupported.** A candidate profile with **no
validation record of any kind in this repository** — no serving run, no
shadow window, no probe record, no entry in
[measured results](../RESULTS.md) or
[testing history](../TESTING_HISTORY.md). It remains unsupported until a
validation record exists.

## Model identity and public-config facts

`deepseek-ai/DeepSeek-V4-Flash-0731`
([huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731));
no revision is pinned in this repository. The following are checkable from
the model's public `config.json` and file listing, not from any SparkRing
evidence:

- Architecture `DeepseekV4ForCausalLM`; hidden size 4096; 64 attention heads.
- FP8 `e4m3` block quantization (128x128 weight blocks) with `fp4` experts.
- 48 safetensors shards totalling approximately 166.9 GB.

Repository-side, the pinned runtime's vendored vLLM sources dispatch the
`DeepseekV4ForCausalLM` architecture
([`runtime/exl3/overlay/vllm/config/model.py`](../../runtime/exl3/overlay/vllm/config/model.py)),
and [runtime gaps](../RUNTIME_GAPS.md) lists DeepSeek-V4-family serving among
the areas still carrying qualification gaps. Neither establishes that this
model loads or serves on the four-Spark stack.

## What admission would require

Per the width-generic admission in the
[vLLM integration README](../../spark_transport/integrations/vllm/README.md)
and the method in the
[eager width admission validation runbook](../EAGER_WIDTH_VALIDATION_RUNBOOK.md):

- `VLLM_SPARK_TP4_EAGER_WIDTHS=4096`, identical on all four ranks (a mixed
  environment fails closed at session connect). The 64 attention heads and
  the 4096 width both divide evenly by the four-rank tensor parallelism.
- Eager only, contiguous CUDA BF16 `[Q, 4096]`; CUDA-graph capture remains
  6144-only.
- Shadow qualification per the runbook's leg-3 method — 10,000-collective
  comparison windows per observed signature in
  `VLLM_SPARK_TP4_MODE=shadow` — before any promotion decision.

Width 4096 currently has only offline admission-surface coverage: the
provider-equivalence permutations include `VLLM_SPARK_TP4_EAGER_WIDTHS=2880,4096`
([test_provider_rows_equivalence.py](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py)),
which compares parsed reservation surfaces with no GPU, link, or live vLLM
involved.
