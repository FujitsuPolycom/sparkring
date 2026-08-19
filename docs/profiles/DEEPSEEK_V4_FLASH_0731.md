# Profile: deepseek-ai/DeepSeek-V4-Flash-0731

Status: **functional launch attested (operator record, 2026-08-18); not
shadow-qualified.** The model loads and serves on the four-Spark ring; no
shadow-comparison window has been closed, so no correctness claim beyond
the bounded probes below attaches. A research-only serving configuration
routing its decode collectives through the transport's width-4096 graph
session exists and is evidenced below; it is not a qualified profile.

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
unmeasured; no model revision was pinned.

## Serving configuration with speculative decoding, 2026-08-18

A later same-day configuration serves with the model's native DSpark
speculative mechanism and CUDA graphs:

- `--speculative-config '{"method": "dspark",
  "num_speculative_tokens": 7, "moe_backend": "b12x"}'`. The
  `moe_backend` selection is load-bearing for draft quality: measured
  draft acceptance on code prompts was 42% without it and 86% with it.
  The checkpoint carries DSpark draft heads, not a classic MTP block;
  a `deepseek_mtp` speculative method fails at weight load.
- `VLLM_SPARK_TP4_MODE=custom`, CUDA graphs enabled, 32 sequences,
  32 GiB key-value cache per rank. The request limit is a launch
  choice against the checkpoint's native 1,048,576 maximum position
  (YaRN, factor 16): first served at 131,072 (engine-reported pool
  1,859,904 tokens), relaunched the same day at 524,288
  (engine-reported concurrency 8.36 full contexts; the token-count
  accounting differs between the two limits and is not reconciled).
- Operator-observed serving behavior on 2026-08-18: single-stream
  decode near 119-132 tokens per second on code prompts at about 86%
  draft acceptance (stock NCCL collectives inside graphs), and
  339.9 tokens per second aggregate at concurrency 32 with 16K-token
  resident contexts (research transport configuration below). These
  are operator observations, not qualified measurements.

## Research transport configuration: width-4096 graph collectives

With `VLLM_SPARK_TP4_GRAPH_WIDTH4096_RESEARCH=1` (research-only,
custom mode), the serving profile's decode all-reduces are captured
into CUDA graphs through one maximum-capacity BF16 [Q <= 512, 4096]
native session (sequential tiered_64k kernel, two-slot deferred ACK)
instead of the stock path. Evidence, 2026-08-18:

- Model-free four-rank gate: fixed-Q legs at Q8/Q48/Q256/Q288/Q320/
  Q512 and a 134-node mixed-Q leg on one session, all reporting zero
  mismatches and zero overflow on all four ranks
  ([probe record](../DUAL_PORT_STRIPING_PROBE_20260818.md) documents
  the instrument and the matched NCCL control).
- Serving launch: 7,445 width-4096 all-reduce graph nodes captured
  per rank, zero capture-phase stock fallback for admitted
  signatures, 9,501 replay commands completed with zero overflow
  after a 32-way request burst; bounded greedy probes returned
  correct results.
- Out of the session's scope by design: the drafter's width-256
  collectives and the [2048, 4096] chunked-prefill all-reduces, which
  remain on the stock path.

This configuration is research-only: no shadow-comparison window has
closed for it, and it carries no qualification.

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
- Eager admission is contiguous CUDA BF16 `[Q, 4096]`. CUDA-graph
  capture at width 4096 exists only behind the research-only input
  described above; the qualified graph paths remain 6144-only.
- Shadow qualification per the runbook's leg-3 method — 10,000-collective
  comparison windows per observed signature in
  `VLLM_SPARK_TP4_MODE=shadow` — before any promotion decision.

Width 4096 currently has only offline admission-surface coverage: the
provider-equivalence permutations include `VLLM_SPARK_TP4_EAGER_WIDTHS=2880,4096`
([test_provider_rows_equivalence.py](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py)),
which compares parsed reservation surfaces with no GPU, link, or live vLLM
involved.
