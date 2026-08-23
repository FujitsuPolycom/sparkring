# Qwen3.8-27B two-Spark 1M probabilistic-MTP bring-up

Lane: **public-functional**. Status: **research-only**. Hardware: two NVIDIA
DGX Sparks at TP2/DCP1 over one direct RoCEv2 link. Evidence scope: one
startup and bounded functional-probe set on a maintainer-built source runtime.
No performance, 1M-input correctness, restart or public-image result is
included.

## Conditions

- Pinned checkpoint
  `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf`;
  all 16 `SHA256SUMS` entries passed on both ranks.
- vLLM `229effc810ee6b8112f661472f6aace4eb8c787d`, ExLlamaV3
  `5f3c537ca9d89893d771256f5c43c93656553fbb`, and patched NCCL SHA-256
  `e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54`.
- A 1,000,000-token advertised limit through Qwen's static-YaRN factor-4
  override; 32 sequences; 8,192 scheduled tokens; explicit FP8 KV; native
  prefix caching with mamba alignment; full-decode CUDA graphs; and 0.70
  unified-memory utilization.
- Qwen MTP depth 3 with probabilistic draft sampling and standard rejection
  sampling. Checkpoint sampler defaults were temperature 1.0, top-p 0.95 and
  top-k 20.
- LMCache, SparkCache and SIRCL were disabled.

## Measurement

Startup, model-memory, graph and KV-capacity values came from the two vLLM
rank logs from one successful launch. Functional checks used bounded
OpenAI-compatible API requests. Exact-marker requests set temperature 0 and
disabled thinking so the finite output budget tested the marker rather than
reasoning length.

The probabilistic-MTP probe was one 128-token request that explicitly set
temperature 1.0, top-p 0.95 and top-k 20. Draft steps, proposed tokens,
accepted tokens, and accepted tokens by position are after-minus-before
Prometheus counter deltas. Draft acceptance is accepted/proposed draft tokens;
mean acceptance length is `1 + accepted draft tokens / speculative steps`.

Rank logs and raw API responses remain maintainer-held. The JSON beside this
document is a sanitized summary, not the raw launch evidence.

## Result

Both ranks rendezvoused and served. The engine advertised one million tokens,
loaded 11.22 GiB of model memory per rank, exposed 67.96 GiB of available KV
memory per rank, and allocated 4,093,750 logical KV tokens: 4.09x capacity at
the advertised request limit. Decode graph capture completed in 17 seconds
and used 1.46 GiB.

Bounded checks passed API health, repeated arithmetic (`391`), tool parsing
(`multiply(a=6,b=7)`), data-URL vision (`VISION_OK`), repeated-prefix equality,
and shared-prefix suffix divergence (`13`, `17`). Exact-marker requests
disabled thinking so a bounded output budget could test the marker rather
than reasoning length.

A fresh temperature-1 request produced 128 completion tokens over 42
speculative steps. The server proposed 126 draft tokens and accepted 85:
67.5% draft acceptance and 3.02 mean acceptance length. Accepted tokens by
draft position were 37, 28 and 20.

## Conclusion

The maintainer-built TP2 serving object is implemented as a research profile.
The startup proves that the selected static-YaRN, scheduler, FP8-KV and
probabilistic-MTP settings compose on this pair. It does not prove that a 1M
prompt is correct, that the public image builder reproduces the launch, or
that the profile meets any performance or reliability threshold.

## Limitations

- Every observation is from one launch; no repetition variance or confidence
  interval is available.
- No one-million-token request or performance benchmark ran.
- Static YaRN can alter short-context distributions.
- The public clean-checkout image builder did not produce the live runtime.
- Restart, reboot persistence, and long-duration reliability are untested.

The machine-readable values are in
[`dgx2-1m-probmtp-20260823.json`](dgx2-1m-probmtp-20260823.json).
