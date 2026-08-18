# Profile: deepseek-ai/DeepSeek-V4-Flash-0731

Status: **functional launch attested (operator record, 2026-08-18); not
shadow-qualified.** The model loads and serves on the four-Spark ring; no
shadow-comparison window has been closed, so no correctness or performance
claim attaches.

## Operator launch record, 2026-08-18

The operator launched the model on the four directly cabled DGX Sparks the
day it was staged. Attested facts:

- Four-rank tensor-parallel serving from the deployed GLM runtime image:
  weights load (about 42 GB per rank), the engine reaches API health in
  about 175 seconds, and interactive chat completions serve with
  deterministic bounded probes returning exact requested outputs.
- Width 4096 admitted through `VLLM_SPARK_TP4_EAGER_WIDTHS=4096` under
  `VLLM_SPARK_TP4_MODE=shadow`, `--enforce-eager`, BF16 activations.
- The architecture requires `--kv-cache-dtype fp8`: its `fp8_ds_mla`
  key-value layout refuses any other cache dtype at load, and the first
  launch attempt failed on exactly that check before the flag was added.
- Tool calling works end-to-end with `--enable-auto-tool-choice
  --tool-call-parser deepseek_v4` (a request with `tool_choice: "auto"`
  serves correctly).
- Configured limits for the launch: 131,072-token request limit, eight
  sequences, 32 GiB of key-value cache per rank.

What this record does not establish: no shadow window closed for any
width-4096 signature, so numerical agreement with the stock path is
unmeasured; no throughput, latency, or quality number exists; no model
revision was pinned; and the model's native DSpark speculative mechanism
is not part of the attested configuration.

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
