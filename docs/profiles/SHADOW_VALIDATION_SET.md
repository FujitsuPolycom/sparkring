# Profile set: shadow width-validation instruments

Status: **validation-grade instruments — not serving recommendations.**

These four small models exist in this repository for exactly one purpose:
exercising the opt-in width-generic eager TP4 all-reduce admission
(`VLLM_SPARK_TP4_EAGER_WIDTHS`, documented in the
[vLLM integration README](../../spark_transport/integrations/vllm/README.md))
end-to-end through vLLM dispatch at hidden widths other than the historical
6144. Shadow mode executes the custom transport and the stock path for every
eligible collective and compares elementwise, so model quality is irrelevant;
the models were chosen for width coverage, TP4 head divisibility, and download
size. Nothing on this page is a recommendation to serve any of them, and no
performance claim attaches to any number below.

The canonical record is the leg-3 section of the
[eager width admission validation runbook](../EAGER_WIDTH_VALIDATION_RUNBOOK.md);
this page restates its outcomes and does not extend them.

## The set

| Model | Width | Heads Q/KV | Architecture |
|---|---:|---|---|
| `EleutherAI/pythia-70m` | 512 | 8 MHA | GPTNeoX |
| `facebook/opt-125m` | 768 | 12 MHA | OPT |
| `Qwen/Qwen3-0.6B` | 1024 | 16/8 | Qwen3 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2048 | 32/4 | Llama |

No revision is pinned for any of the four; the runbook stages them by
repository name into each serving rank's Hugging Face cache.

## Serving configuration used

All four ran identically on all four ranks, from the deployed image via the
v23-cloned launch harness, executed 2026-08-17:

- `VLLM_SPARK_TP4_MODE=shadow`
- `VLLM_SPARK_TP4_EAGER_WIDTHS=<model width>` (identical on every rank; a
  mixed environment fails closed at session connect)
- `--enforce-eager --dtype bfloat16 -tp 4`
- Deliberately unset: `SPARK_TP4_SHADOW_PROMOTE`, `SPARK_TP4_SHADOW_STRICT`,
  `VLLM_SPARK_TP4_GRAPH_Q1`, `VLLM_SPARK_TP4_GRAPH_DUAL_PORT_Q40`,
  `VLLM_SPARK_TP4_PREFILL_Q512`.

Traffic was batch-1 decode until each observed signature closed its
10,000-collective shadow window; every window is the `(1, W)` BF16 decode
shape, and all four ranks reported bit-identical statistics in every run.

## Per-width outcomes (executed 2026-08-17)

| Width | Model | outside_tolerance | rate | max_abs | Configured-gate verdict |
|---:|---|---:|---:|---:|---|
| 512 | pythia-70m | 3 / 5.12M | 5.9e-7 | 0.5 | FAIL — operator-dispositioned, unpromoted |
| 768 | opt-125m | 0 / 7.68M | 0 | 0.0156 | **PASS** |
| 1024 | Qwen3-0.6B | 1153 / 10.24M | 1.1e-4 | ~1 | FAIL — **open** |
| 2048 | TinyLlama-1.1B | 2 / 20.48M | 1.0e-7 | 0.0625 | FAIL — operator-dispositioned, unpromoted |

- **Width 768** passed its gate outright: zero elements outside tolerance.
- **Widths 512 and 2048** failed the configured gate (which requires
  `outside_tolerance == 0`) and carry a recorded operator disposition:
  accepted as explained, not promoted. The runbook attributes the divergence
  to BF16 reduction-order noise at near-cancellation (SIRCL pairwise tree
  versus stock ring order), with zero non-finite values and bit-identical
  statistics across all four ranks — jointly inconsistent with transport
  corruption.
- **Width 1024** remains an open disagreement. An FP32-oracle arbitration is
  recorded in the runbook as research-only: on the cancellation pattern the
  tree order lands two orders of magnitude closer to correctly rounded FP32
  truth than a naive sequential sum, but the comparator does not model the
  stock path's actual ring order and the inputs are synthetic. The signature
  is unpromoted pending oracle reruns on captured real inputs and a
  disagreement-policy decision for outlier-heavy models.

No signature from this matrix is promoted. The runbook also records the
launch-shape findings required to serve non-GLM models from the deployed image
and the candidate models excluded for TP4 head-divisibility reasons.
