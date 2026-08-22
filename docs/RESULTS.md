# SparkRing results

This page records evidence for the two supported four-DGX-Spark profiles. A
result applies only to its stated model identity, runtime configuration,
hardware topology, and evidence scope.

## GLM-5.2 EXL3 3.5-bpw

**Status: qualified on one four-Spark appliance.** The qualified serving contract
is [the fixed-MTP4 profile](GLM52_35BPW_FIXED_MTP4_PROFILE.md): TP4/DCP4,
fixed MTP4, `nvfp4_ds_mla` key-value cache with 9.25 GB per rank, bounded
full-CKV gather, and SIRCL TP collectives with patched NCCL fallback.

| Context | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, tokens/s | — | 679 | 673 | 666 | 657 | 645 |
| Decode, one stream | 22.6 | 22.0 | 21.3 | 20.4 | 21.4 | — |
| Decode, four streams | 50.3 | 51.9 | 49.2 | 45.6 | 47.2 | — |
| Decode, eight streams | 78.4 | 71.3 | 70.0 | 65.5 | 67.8 | — |

Coding prompts reached 27.3 tokens/s single-stream with 96.64% measured draft
acceptance. The matched C8 cell regressed 11.63% under MTP4. A rebuilt image
has no acceptance status until it completes
[the promotion checklist](GLM52_35BPW_PROMOTION_CHECKLIST.md); the figures
above belong to the image ID that produced them and do not transfer to a
rebuild.

One rebuilt image has been carried from a clean checkout to a serving
endpoint on four ranks, with its checkpoint, generated profiles, and in-image
runtime bytes verified against their pinned identities. That record reports
no throughput figure and does not promote the profile. See
[the rebuilt-image bring-up record](../performance/records/glm-3.5bpw/rebuilt-image-20260821.md).

## DeepSeek-V4-Flash-0731

**Status: implemented; not qualified.** One four-Spark launch
exercised API health, chat completions, tool calling, and DSpark speculative
decoding using the immutable image pinned by
[`runtime/faststart-lock.json`](../runtime/faststart-lock.json). The launch used
`fp8_ds_mla` for the MLA key-value cache, a 32 GiB per-rank reservation, and a
524,288-token request limit.

Operator-observed performance is not a qualified measurement: random-text
decode was roughly 55–65 tokens/s single-stream and prefill was roughly
2,400–2,700 tokens/s from 8K through 128K context. The launch record does not
qualify numerical agreement with a stock collective path or establish a
four-rank run in which every rank pulled the published image digest. See
[the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md).

## Interpretation

Do not compare these values across model identities or use them as a generic
hardware benchmark. The GLM result is acceptance evidence for one appliance;
the DeepSeek observations establish functional serving only.
