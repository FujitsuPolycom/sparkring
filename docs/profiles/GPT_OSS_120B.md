# Profile: openai/gpt-oss-120b

Status: **unsupported — no validation record exists in this repository.**

## What it is

`openai/gpt-oss-120b` is a mixture-of-experts causal language model published
on Hugging Face
([huggingface.co/openai/gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b)).
Its public `config.json` records hidden size 2880, 64 attention heads, and
8 key-value heads — head counts that divide evenly by the four-rank tensor
parallelism SparkRing uses. No revision of this model is pinned anywhere in
this repository.

## Evidence boundary

This repository contains **no serving run, no shadow-comparison window, and no
runbook record for this model**. It does not appear in the
[eager width admission validation runbook](../EAGER_WIDTH_VALIDATION_RUNBOOK.md),
in [measured results](../RESULTS.md), or in
[testing history](../TESTING_HISTORY.md). Neither functional serving nor
shadow qualification is established. Anything asserting otherwise is not
backed by this repository.

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
- Whether the pinned runtime images serve this architecture at all is
  unestablished here: the only GLM-external models this stack has ever served
  are the four shadow-matrix instruments recorded in the runbook, and their
  bring-up required launch-shape accommodations recorded there.
