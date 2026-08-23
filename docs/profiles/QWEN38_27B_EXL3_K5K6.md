# Qwen3.8-27B EXL3 K5/K6 four-Spark profile

## Status

**Status: implemented on a maintainer-built runtime; not qualified.** The
candidate serving object started and served on four directly cabled DGX
Sparks. A bounded live run passed API, deterministic arithmetic, tool-use,
vision, and hybrid-prefix checks, then measured prefill and decode at limited
contexts and concurrency. The public clean-checkout image builder is
offline-validated; ARM64 build execution and live four-rank evidence remain
pending.

Use the
[Qwen3.8-27B four-Spark quickstart](../QWEN38_27B_EXL3_K5K6_QUICKSTART.md)
to build one local runtime image, distribute it, and launch one rank per
Spark.

## Serving contract

| Setting | Value |
|---|---|
| Model | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Public runtime builder | `runtime/qwen38/build-image.sh`, using pins in `runtime/qwen38/pins.json`; build once and distribute one image ID |
| Historical measured runtime | `FujitsuPolycom/qwen38-spark-pair@b9e1031b80b6f3f64bfc75ae3922322f56954fd6`, maintainer-built and replicated identically to every rank |
| Parallelism | TP4/DCP1 across four DGX Sparks |
| Collective transport | patched NCCL on the direct cycle |
| Request limit | 262,144 tokens |
| Maximum sequences | 64 |
| Scheduler budget | 8,192 tokens |
| Scheduling | chunked prefill, asynchronous, full-input-length reservation |
| Hybrid block geometry | request 16 attention tokens; runtime aligns effective attention and mamba blocks to 1,600 tokens |
| Key-value dtype | FP8 |
| EXL3 prefill | FP8, reconstruction tile 256 |
| Prefix caching | enabled with mamba alignment |
| Speculation | Qwen MTP depth 3 |
| Decode execution | full-decode CUDA graphs |
| External key-value cache | disabled |
| SIRCL | unsupported for the width-5,120 path |

The candidate deliberately keeps Qwen's model-specific values instead of
copying DeepSeek's 1M context, 4,096-token scheduler budget, DSpark proposer,
or MLA cache format. It reuses DeepSeek's four-Spark physical topology,
non-adjacent-rank routing, multi-node process shape, and patched-NCCL cycle
settings.

## Public reproduction path

The [four-Spark quickstart](../QWEN38_27B_EXL3_K5K6_QUICKSTART.md) is
self-contained within public repositories. It builds a local ARM64 image from
immutable CUDA, vLLM, ExLlamaV3, companion-recipe, Torch, B12X, and NCCL
inputs; distributes one image ID and the separately verified checkpoint; runs
the rank launcher in `--check` mode; starts one container per rank; and runs
the public `scripts/qwen38_smoke.py` gate.

No published Qwen image is required. A published image would reduce build
time, but it would not replace site-specific topology checks, model
verification, or live functional evidence for the selected image ID.

## Evidence

Conditions: four NVIDIA DGX Sparks in the direct cycle, the pinned checkpoint,
identical copies of a maintainer-held source-built runtime, TP4/DCP1 with
vLLM's multi-node `mp` executor, patched NCCL on two RoCE devices per rank, and
the serving contract above. External key-value caching and SIRCL were
disabled. All 16 checkpoint hashes, both source commits, the clean vLLM
worktree, and the ExLlamaV3 ARM patch digest passed on every rank.

Measurement: the startup/capacity values came from engine logs. Prefill used
one client `prompt_tokens / TTFT` scout per shape. Decode used OpenAI
continuous-usage output tokens over one approximately 15-second sustained
window per cell after a same-message one-token scout and readiness warmup. The
decode values are warm repeated-prefix throughput, not cold end-to-end
throughput. Raw artifacts remain maintainer-held; the public summary records
exact output-token counts, timing windows, cache observations, and gate
outcomes.

Result: all ranks rendezvoused and stayed alive. The API, deterministic
arithmetic, tool parser, data-URL vision, and three hybrid-prefix gates passed.
The engine reported 74.74 GiB of key-value memory per rank, 8,382,750 logical
key-value tokens, and 31.98x maximum concurrency at the 262,144-token request
limit. A bounded N=1 sweep measured 38.4-38.6 aggregate decode tokens/s at C1,
128.4-129.8 at C4, and 1,288-1,986 prefill tokens/s from 4K through 128K.

Conclusion: the four-rank serving object is implemented and
live-benchmarked. The profile remains a candidate because the observations do
not establish sustained performance, restart behavior, or complete
correctness.

The full conditions and machine-readable values are in the
[live bring-up record](../../performance/records/qwen38-27b/dgx4-live-20260823.md)
and [quick benchmark data](../../performance/records/qwen38-27b/dgx4-quick-20260823.json).

Limitations:

- Each performance coordinate is one bounded observation. The sustained
  decode matrix covers only warm repeated-prefix C1 and C4 at 4K and 16K
  contexts; it is not cold end-to-end throughput.
- The 64K and 128K cells are prefill scouts, not long-context generation
  results.
- Native prefix caching was enabled, and no scout has a reliable cached-token
  delta; the prefill values are not cold or unique-prompt measurements.
- Restart, reboot persistence, eager-versus-graph equivalence, and
  long-duration serving have not been tested.
- The runtime has no published Qwen image. Every rank must use the same
  prepared source tree, environment, model revision, patched NCCL library, and
  chat template.
- The exact measured source archive and raw harness output are maintainer-held;
  public reproduction remains pending.
- The clean-checkout image builder and preflight pass offline tests only. No
  image produced by that builder has completed a four-rank startup or the
  public smoke gate, so historical capacity and performance do not transfer.
- The tracked launcher adds fail-closed checks and makes effective scheduler
  defaults and the requested block input explicit; those wrapper changes have
  offline validation only.
- The pinned CUDA base does not supply `libibverbs.so.1`. Install
  `libibverbs1`, `ibverbs-providers`, and `ibverbs-utils` before starting NCCL.

## Pending integration

**SparkCache: Pending.** No Qwen3.8-27B SparkCache composition recipe or live
cache evidence is published. The base profile keeps external key-value caching
disabled while the four-Spark serving path is tested.
