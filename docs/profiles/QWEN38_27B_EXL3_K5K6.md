# Qwen3.8-27B EXL3 K5/K6 four-Spark profile

## Tested setup

The profile ran on four directly cabled DGX Sparks and produced the results
linked below.

Use the
[Qwen3.8-27B four-Spark quickstart](../QWEN38_27B_EXL3_K5K6_QUICKSTART.md)
to build one local runtime image, distribute it, and launch one rank per
Spark.

## Serving contract

| Setting | Value |
|---|---|
| Model | `malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated@ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| Public runtime builder | `runtime/qwen38/build-image.sh`, using pins in `runtime/qwen38/pins.json`; build once and distribute one image ID |
| Tested runtime source | `FujitsuPolycom/qwen38-spark-pair@b9e1031b80b6f3f64bfc75ae3922322f56954fd6` |
| Parallelism | TP4/DCP1 across four DGX Sparks |
| Collective transport | patched NCCL on the direct cycle |
| Request limit | 1,048,576 tokens through static YaRN factor 4 over native 262,144 |
| Maximum sequences | 64 |
| Scheduler budget | 8,192 tokens |
| Scheduling | chunked prefill, asynchronous, full-input-length reservation |
| Hybrid block geometry | request 16 attention tokens; runtime aligns effective attention and mamba blocks to 1,600 tokens |
| Key-value dtype | FP8 |
| EXL3 prefill | FP8, reconstruction tile 256 |
| Prefix caching | enabled with mamba alignment |
| Speculation | Qwen MTP depth 3, probabilistic drafts, standard rejection sampling |
| Decode execution | full-decode CUDA graphs |
| External key-value cache | disabled |
| SIRCL | unsupported for the width-5,120 path |

The four-Spark and [two-Spark](QWEN38_27B_EXL3_K5K6_PAIR.md) normalized
profiles use the same static-YaRN model-length and MTP sampling contract. The
four-Spark profile does not copy DeepSeek's DSpark proposer or MLA cache
format. It reuses DeepSeek's four-Spark physical topology,
non-adjacent-rank routing, multi-node process shape, and patched-NCCL cycle
settings.

## Build and launch

The [four-Spark quickstart](../QWEN38_27B_EXL3_K5K6_QUICKSTART.md) is
self-contained within public repositories. It builds a local ARM64 image from
immutable CUDA, vLLM, ExLlamaV3, companion-recipe, Torch, B12X, and NCCL
inputs; distributes one image ID and the separately verified checkpoint; runs
the rank launcher in `--check` mode; starts one container per rank; and runs
the public `scripts/qwen38_smoke.py` gate.

No published Qwen image is required. A published image would reduce build
time, but it would not replace site-specific topology checks, model
verification, or live functional evidence for the selected image ID.

## Benchmark results

See the [benchmark table, screenshot, and machine-readable data](../../performance/records/qwen38-27b/normalized-tp4-1m-probmtp-temp1-20260823.md).

Limitations:

- The pinned CUDA base does not supply `libibverbs.so.1`. Install
  `libibverbs1`, `ibverbs-providers`, and `ibverbs-utils` before starting NCCL.

## SparkCache

SparkCache is not included. External key-value caching is disabled.
