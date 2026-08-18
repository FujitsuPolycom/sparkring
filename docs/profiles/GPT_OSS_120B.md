# Profile: openai/gpt-oss-120b

Status: **functional launch attested (operator record, 2026-08-17); not
shadow-qualified.** The model loads and serves on the four-Spark ring; no
numerical comparison window was collected, so no correctness or performance
claim attaches.

## What it is

`openai/gpt-oss-120b` is a mixture-of-experts causal language model published
on Hugging Face
([huggingface.co/openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)).
Its public `config.json` records hidden size 2880, 64 attention heads, and
8 key-value heads — head counts that divide evenly by the four-rank tensor
parallelism SparkRing uses. No revision of this model is pinned anywhere in
this repository.

## Operator launch record, 2026-08-17

The operator launched `openai/gpt-oss-120b` on the four directly cabled DGX
Sparks and confirmed it working interactively. This is an attested record —
the serving session predates this page, and its logs were not preserved — so
its claims are limited to what launching and using the model establishes:

- Four-rank tensor-parallel serving from the deployed GLM runtime image, with
  the MXFP4 mixture-of-experts weights executing on the GB10 devices.
- Width 2880 admitted through `VLLM_SPARK_TP4_EAGER_WIDTHS=2880` under
  `VLLM_SPARK_TP4_MODE=shadow`, `--enforce-eager`, BF16 activations.
- Decode confirmed: interactive chat completions served to a human operator
  using the model's harmony chat template (the template's tokenizer requires
  the `o200k_base` encoding staged locally and named by
  `TIKTOKEN_ENCODINGS_BASE`).
- Prefill confirmed to at least an 11,681-token prompt, under a configured
  131,072-token request limit.

What this record does **not** establish: no shadow-comparison window was
closed for any width-2880 signature, so numerical agreement with the stock
path is unmeasured; no throughput, latency, or quality number exists; and no
model revision was pinned. The runbook-method qualification below remains
open.

What the repository does contain for the model's hidden width, 2880, is
**offline-validated admission-surface coverage only**:

- The provider-equivalence oracle exercises width 2880 as an environment
  permutation (`sparse_with_width_2880`, and `contiguous_with_widths` with
  `VLLM_SPARK_TP4_EAGER_WIDTHS=2880,4096`) and its harness emits admission
  bitmaps at widths 2880, 4096, and 6144
  ([test_provider_rows_equivalence.py](../../spark_transport/integrations/vllm/test_provider_rows_equivalence.py),
  [_provider_equivalence_harness.py](../../spark_transport/integrations/vllm/_provider_equivalence_harness.py)).
- The scope and limits of that evidence — parsed admission surfaces only; no
  GPU, no RDMA link, no native library, no live vLLM, no numerical or
  performance claim — are stated in the
  [eager width admission review handoff](../REVIEW_HANDOFF_EAGER_WIDTH_ADMISSION.md).

## What admission would require

Per the width-generic admission documented in the
[vLLM integration README](../../spark_transport/integrations/vllm/README.md):

- `VLLM_SPARK_TP4_EAGER_WIDTHS=2880`, identical on all four ranks; setting the
  variable remaps eager all-reduce control ports, so a mixed environment fails
  closed at session connect.
- Admission remains eager-only, contiguous CUDA BF16 `[Q, 2880]` under the
  existing query-row bounds. CUDA-graph capture stays 6144-only; a non-6144
  tensor observed during capture routes to the stock path.
- Shadow-mode qualification before any promotion, following the small-model
  shadow-matrix method in the
  [eager width admission validation runbook](../EAGER_WIDTH_VALIDATION_RUNBOOK.md)
  (leg 3): serve `--enforce-eager --dtype bfloat16 -tp 4` in
  `VLLM_SPARK_TP4_MODE=shadow` and close a 10,000-collective comparison window
  per observed signature.

## Known limitations

- The width-generic admission feature itself carries no performance or
  maturity claim; the integration README records that prior admission
  widenings regressed serving until qualified.
- The launch record above is an operator attestation without preserved
  artifacts; a reproducing run should capture its startup log, environment,
  and shadow-window statistics so this page can graduate to recorded
  evidence. The small-model bring-up accommodations recorded in the runbook
  (launch shape, tokenizer staging) applied to this launch as well.
