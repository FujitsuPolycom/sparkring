# GLM-5.3-Flash two-Spark draft tokens by local argmax, 2026-09-30

Status: **research-only**. One Spark pair, three decode benchmark runs per
deployment, sequential correctness passes; not serving-qualified.

## Question

The two-Spark GLM-5.3-Flash profile `glm53-flash-nvfp4-spark-tp2` drafts three
tokens per step with GLM's multi-token-prediction (MTP) head, choosing each
draft token greedily. By default every draft step projects the hidden state
through the vocabulary head on both tensor-parallel ranks, all-gathers the full
154,880-entry logits, and takes their argmax. With
`"use_local_argmax_reduction": true` in `--speculative-config`, each rank takes
the maximum of its own vocabulary shard and the ranks all-gather only
(value, global index) pairs. Does the option select the same draft tokens, keep
CUDA graphs, and change decode speed at 1 and 8 concurrent requests?

## Conditions

- **Cluster:** two directly cabled DGX Sparks (spark-e rank 0,
  spark-d rank 1).
- **Model and image:** `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`
  revision `a608241037e4` on installer image
  `dev-20260928-plainstatus-cuda1342-nccl2323-status033` (image
  `sha256:4b7049d1e00f`), vLLM `0.1.dev21510+g1794dcf18` with the V2 model
  runner (`VLLM_USE_V2_MODEL_RUNNER=1`).
- **Deployments:** each installed with `install.sh` from a Git bundle.
  - *Stock:* the profile at SparkRing commit `8c32d80a`, speculative
    configuration
    `{"method":"mtp","num_speculative_tokens":3,"moe_backend":"humming","attention_backend":"B12X"}`.
    Measured once after installation and again after reinstallation
    following the flag deployment ("stock, reinstalled").
  - *Flag:* the same profile with `"use_local_argmax_reduction":true` added
    to that configuration and no other change to a file the installer reads.
- **Code path in the image:** with `method: mtp` and one MTP layer in the
  checkpoint (`num_nextn_predict_layers: 1`), vLLM builds `MTPSpeculator`
  (`vllm/v1/worker/gpu/spec_decode/mtp/speculator.py`). Its draft prefill and
  each draft decode step call `DraftModelSpeculator.sample_draft`
  (`vllm/v1/worker/gpu/spec_decode/speculator.py`), which with the option set
  and greedy drafting returns `Glm5NextMTP.get_top_tokens`. That calls
  `LogitsProcessor.get_top_tokens`
  (`vllm/model_executor/layers/logits_processor.py`) with the same vocabulary
  head and the same projection (`_apply_head`) as the full-logits path. At
  load, `_validate_local_argmax_reduction` rejects the option for
  probabilistic drafting or a draft model without `get_top_tokens`; adaptive
  verification is rejected in `SpeculativeConfig`. The profile uses greedy
  drafting and no adaptive verification.
- **Data exchanged per draft token and request:** full logits, 154,880 bytes
  sent per rank (77,440 bf16 values); local argmax, 8 bytes per rank.
- **Correctness passes** ([`decode_passes.py`](tp2-local-argmax-20260930/programs/decode_passes.py)):
  24 prompts (7 code, 6 mathematics, 7 prose, 2 structured output, 2 tool
  calls), one request at a time, 512 output tokens with at least 256, token
  IDs returned by the server. Greedy passes use temperature 0; sampled passes
  use temperature 1.0 with seed 1000 + prompt index. Speculative acceptance is
  the change in vLLM's `/metrics` counters over the pass.
- **Decode benchmark:** llm-inference-bench 0.6.2 `llm_decode_bench.py`,
  `--concurrency 1,8 --contexts 0 --duration 20 --decode-warmup-seconds 5
  --max-tokens 2048 --temperature 1.0 --token-targeting exact --no-hw-monitor
  --no-resume`, from a separate machine on Node A's network, three runs per
  deployment. Each run also sends the tool's 8K, 64K and 128K prefill scout
  requests before the decode cells. Steps per second is the tool's aggregate
  tokens per second divided by its mean accepted length, so it compares engine
  speed across runs whose acceptance differs. Every cell ran with all
  requests in flight and none queued.
- **Other traffic:** a host outside this measurement sent 46 requests naming a
  model these deployments do not serve; the API server rejected each with
  HTTP 404 before scheduling. 42 of them arrived during the flag deployment's
  second greedy pass, its first sampled pass, the single-request cell of its
  first benchmark run (four requests) and the first second of that run's
  eight-request cell (one request). Fourteen three-token temperature-0 probe
  requests overlap the stock deployment's first sampled pass (at most 28 of
  its 4,181 drafts).

## Results

### Draft selection

A CPU check ([`tiebreak_check.py`](tp2-local-argmax-20260930/programs/tiebreak_check.py))
runs the image's `LogitsProcessor.get_top_tokens` for both ranks on bf16
logits rounded to coarse grids, with the head projection and all-gather
replaced by the two vocabulary shards. Over 12,800 rows, 6,952 with tied
maxima within or across shards, its result matched a full-vocabulary
`argmax` on every row and was identical on both ranks.

### Greedy output equality

Neither deployment reproduces its own temperature-0 output. Five identical
single requests to the stock deployment returned a first-token log-probability
between −0.29 and −0.79 for the top candidate, and one of the five chose a
different first token. Output comparison therefore bounds draft-token
equality only to this noise:

| Greedy pass pairs | Pairs | Identical outputs | First divergence, median per pair (tokens) |
|---|---|---|---|
| Same variant: stock ×2, stock vs reinstalled ×2, flag ×1 | 4 | 0 of 96 | 22–43.5 |
| Stock vs flag | 6 | 0 of 144 | 22.5–49.5 |

Both groups include prompts that diverge at the first generated token.

### Speculative acceptance

| Pass | Stock | Stock, reinstalled | Flag |
|---|---|---|---|
| Greedy, acceptance rate (mean accepted length) | 0.646 (2.94), 0.657 (2.97) | 0.663 (2.99) | 0.658 (2.97), 0.653 (2.96) |
| Greedy, per-position acceptance | 0.845/0.634/0.459, 0.846/0.652/0.473 | 0.852/0.657/0.481 | 0.853/0.655/0.466, 0.841/0.645/0.473 |
| Sampled, acceptance rate (mean accepted length) | 0.610 (2.83), 0.587 (2.76) | — | 0.610 (2.83), 0.595 (2.79) |

Pooled over sampled passes and benchmark cells at temperature 1.0, the stock
deployments accepted 26,304 of 45,633 draft tokens (0.576) and the flag
deployment 20,494 of 35,208 (0.582).

### CUDA graphs

Every deployment logged the same captures on rank 0: 10 piecewise and 8 full
target graphs, and for the speculator 10 piecewise and 4 full draft-prefill
graphs and 4 full draft-decode graphs. Both ranks of every deployment logged
`Graph capturing finished` (6–7 s) and, with the flag, `Using local argmax
reduction for draft token generation`. Through every pass and benchmark,
neither rank of either variant logged an error, traceback, eager fallback or
recompilation. Both variants compiled the same six Triton kernels on first
use, at the first request and at the benchmark's long-prefill shapes, on each
rank. Single-request step rates match the stock deployment's, consistent with
graph replay in both.

### Decode speed

Mean and range of three runs per deployment, temperature 1.0:

| Requests | Measure | Stock | Stock, reinstalled | Flag |
|---|---|---|---|---|
| 1 | Steps/s | 14.82 (14.61–14.94) | 14.80 (14.72–14.91) | 14.75 (14.61–14.94) |
| 1 | Tokens/s | 38.7 (35.3–43.2) | 37.4 (34.1–40.0) | 36.0 (32.6–38.9) |
| 1 | Accepted length | 2.61 (2.41–2.89) | 2.53 (2.32–2.68) | 2.44 (2.23–2.61) |
| 1 | Time to first token, p50 (ms) | 323 (312–328) | 317 (312–328) | 334 (328–344) |
| 1 | Inter-token latency, p50 (ms) | 25.7 (23.4–27.1) | 26.7 (25.0–28.5) | 27.2 (25.9–28.8) |
| 8 | Steps/s | 42.03 (41.78–42.43) | 41.05 (39.73–41.79) | 40.86 (39.54–42.72) |
| 8 | Tokens/s | 111.9 (110.4–113.6) | 110.5 (106.1–116.1) | 108.1 (101.7–113.9) |
| 8 | Accepted length | 2.66 (2.63–2.71) | 2.69 (2.63–2.78) | 2.65 (2.57–2.70) |
| 8 | Time to first token, p50 (ms) | 1,062 (1,000–1,125) | 1,224 (1,109–1,360) | 922 (844–985) |
| 8 | Inter-token latency, p50 (ms) | 69.4 (68.2–70.4) | 70.5 (67.5–72.6) | 72.0 (68.3–77.0) |

Per-run values, acceptance counts and pairwise divergence statistics are in
[`results/summary.json`](tp2-local-argmax-20260930/results/summary.json),
built by [`summarize.py`](tp2-local-argmax-20260930/programs/summarize.py).

## Conclusion

The option is reached in the serving image and selects draft tokens by the
same projection and tie-breaking as the full-logits argmax; the CPU check
confirms the reduction on heavily tied inputs. On hardware, bitwise output
equality cannot be tested because this deployment is not deterministic at
temperature 0 even at the first token; within that limit, greedy outputs
diverge from the stock deployment no earlier than the stock deployment
diverges from itself, and greedy and sampled acceptance match. CUDA graphs
capture and serve unchanged on both ranks.

Decode speed did not change measurably. Single-request steps per second
differ by under 0.5%, inside one deployment's run-to-run range. At eight
requests the flag's mean is 1.6% below the mean of the six stock runs and
inside their range (39.73–42.43). The data the option removes, about 0.46 MB per rank per
verification step at one request and 3.7 MB at eight, is below what this
benchmark resolves. The profile is correct with the option; its speed effect
on this pair is inconclusive and smaller than about 2% at 1 and 8 requests.
Sixteen requests, four-Spark rings and contexts above the benchmark's
decode prompt were not measured.

The `glm53-flash-nvfp4-spark-tp2` profile does not set the option: it is
correct, but this comparison found no decode speed difference that would
justify restarting installed pairs to adopt it.
