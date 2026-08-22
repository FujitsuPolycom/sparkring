# SparkRing results

This page lists results for the supported DGX Spark profiles. Each number only
applies to the model, settings, hardware, and topology stated with it.

## GLM-5.2 EXL3 3.5-bpw

**Status: candidate at 1,048,576 tokens and 16 sequences. The older
262,144-token, eight-sequence setup remains qualified on one appliance.**
The serving contract is [the fixed-MTP4 profile](GLM52_35BPW_FIXED_MTP4_PROFILE.md): TP4/DCP4,
fixed MTP4, `nvfp4_ds_mla` key-value cache with 9.25 GB per rank, bounded
full-CKV gather, and SIRCL TP collectives with patched NCCL fallback.

This first table shows the older 262,144-token, eight-sequence setup. It does
not describe the new settings:

| Context | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, tokens/s | — | 679 | 673 | 666 | 657 | 645 |
| Decode, one stream | 22.6 | 22.0 | 21.3 | 20.4 | 21.4 | — |
| Decode, four streams | 50.3 | 51.9 | 49.2 | 45.6 | 47.2 | — |
| Decode, eight streams | 78.4 | 71.3 | 70.0 | 65.5 | 67.8 | — |

Coding prompts reached 27.3 tokens/s single-stream with 96.64% measured draft
acceptance. The matched C8 cell regressed 11.63% under MTP4. A rebuilt image
has no acceptance status until it completes
[the full acceptance checklist](GLM52_35BPW_PROMOTION_CHECKLIST.md). The figures
above belong to qualified operator image
`sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513`;
they do not transfer to rebuilt image
`sha256:5569c4778c9561a8595ac283c7adf31e22be1d35517aa208569af5224244a2da`,
whose separate scope is recorded in
[`rebuilt-image-20260821.md`](../performance/records/glm-3.5bpw/rebuilt-image-20260821.md).

One rebuilt image has been carried from a clean checkout to a serving
endpoint on four ranks, with its checkpoint, generated profiles, and in-image
runtime bytes verified against their pinned identities. That record reports
no throughput figure and does not promote the profile. See
[the rebuilt-image bring-up record](../performance/records/glm-3.5bpw/rebuilt-image-20260821.md).

The new 1,048,576-token, 16-sequence setup started successfully. At temperature
1.0 and top-p 0.95, prefill measured 694 tok/s at 2K down to 635 tok/s at 128K.
Decode is complete through C8 from 2K to 32K, plus 64K/C1. The remaining
long-context coordinates are pending. See
the [full benchmark record](../performance/records/glm-3.5bpw/normalized-base-20260822.md).
Its fresh-namespace launch also exposed a create-once exact-Q40 receipt
restartability limitation; no receipt was deleted or overwritten.

## DeepSeek-V4-Flash-0731

**Status: candidate settings; not qualified.** Older
two-Spark-pair and four-Spark-cycle launches exercised API health, chat
completions, tool calling, and DSpark
speculative decoding using the immutable image pinned by
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json). Both use
`fp8_ds_mla` for the MLA key-value cache and a 1,048,576-token request limit.
The base settings reserve 17,179,869,184 bytes per rank, use a
4,096-token scheduler budget, and set block size 256 in both topologies.

The new two-Spark base setup was benchmarked through 128K prefill and sustained
decode at temperature 1.0 and top-p 1.0. At 16K it measured 295.34 aggregate
tok/s at C32 using the isolated vLLM generation counter. Repeated C1/C8 cells
show substantial DSpark-acceptance variance. See the
[full TP2 benchmark record](../performance/records/deepseek-v4-flash/normalized-tp2-base-20260822.md).

The public tree retains functional SparkCache launcher and restore validation,
but does not publish performance tables from other sampling temperatures. The
normalized SparkCache profiles still require temperature-1 remeasurement.

## Qwen3.8-27B EXL3 K5/K6

**Status: experimental four-Spark candidate.** A maintainer-held image and
model started and served on four directly cabled DGX Sparks. The public builder
has not reproduced that exact runtime from a clean checkout. The clean-checkout
builder in `runtime/qwen38/` is offline-validated and has no live result yet.

### Conditions

The candidate fixes TP4/DCP1, a 262,144-token request limit, 64 sequences, an
8,192-token cache-free scheduler budget, FP8 key-value cache, Qwen MTP depth 3,
native prefix caching with recurrent-state alignment, full-decode CUDA graphs,
and patched NCCL. SIRCL and external key-value caching are disabled.

### Functional result

All ranks rendezvoused and remained alive. API health, deterministic arithmetic,
tool calling, data-URL vision, and a maintainer-held hybrid-prefix probe passed.
The engine reported 74.74 GiB of key-value memory per rank, 8,382,750 logical
tokens, and 31.98x maximum concurrency at 262,144 tokens.

Temperature-1 performance qualification is pending. No Qwen throughput table
is published until the benchmark is rerun under the documented sampling and
accounting contract. See the
[profile record](profiles/QWEN38_27B_EXL3_K5K6.md) and
[experimental quickstart](QWEN38_27B_EXL3_K5K6_QUICKSTART.md).

A separate research-only width-4096 SIRCL candidate completed four-rank CUDA
graph capture, deterministic API smoke, and sustained C1/C8 decode while every
rank's native published, consumed, and completed counters remained equal and
overflow remained zero. The candidate used a 4,096-token batch budget and
six hash-bound runtime mounts. This evidence validates the bounded execution
path; it does not qualify output agreement with the patched-NCCL control or a
performance advantage. See the
[SIRCL live-validation record](../performance/records/deepseek-v4-flash/sircl-width4096-live-validation-20260822.md).

## Interpretation

Do not compare these values across model identities or use them as a generic
hardware benchmark. The GLM result is acceptance evidence for one appliance;
the DeepSeek observations establish functional serving only.
