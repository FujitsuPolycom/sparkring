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

The new 1,048,576-token, 16-sequence setup started successfully and
produced prefill measurements from 694 tok/s at 2K to 635 tok/s at 128K.
Temperature-1 decode is complete at 2K/8K through C8 and at 16K/32K for C1/C2;
the remaining long-context and higher-concurrency coordinates are pending. See
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

The new two-Spark base setup was benchmarked through 128K
prefill and sustained decode at temperatures 0 and 1.0 with `top_p` unset.
At 16K it measured 444.89 aggregate tok/s at C32 for temperature 0 and 349.00
tok/s for temperature 1.0. These are one accepted observation each; repeated
C1/C8 cells show substantial DSpark-acceptance variance. See the
[full TP2 benchmark record](../performance/records/deepseek-v4-flash/normalized-tp2-base-20260822.md).

Historical operator-observed aggregate decode throughput for 256-token prompts
and 512-token generations used an 8,192-token scheduler budget, runtime-selected
block geometry, and a 12 GiB pair reservation. It was 36.7 tokens/s on the pair and 56.5 tokens/s on the
cycle at one concurrent request, rising to 237.7 and 347.5 tokens/s at 32
concurrent requests. These are not qualified measurements. The launch record
does not qualify numerical agreement with a stock collective path or establish
a four-rank run in which every rank pulled the published image digest. See
[the profile record](profiles/DEEPSEEK_V4_FLASH_0731.md).

The additional reports cover the two-Spark base recipe, the TP2 SparkCache
launcher on a separate research checkpoint, and matched four-Spark
base/SparkCache tests. They do not change the status of the base profiles or
apply an older SparkCache receipt to new settings:

- [two-Spark base performance](../performance/records/deepseek-v4-flash/base-tp2-performance-20260822.md);
- [TP2 launcher persistence](../performance/records/deepseek-v4-flash/sparkcache-tp2-launcher-research-validation-20260822.md)
  and [research performance](../performance/records/deepseek-v4-flash/sparkcache-tp2-research-performance-20260822.md);
- [four-Spark base performance](../performance/records/deepseek-v4-flash/base-tp4-performance-20260822.md);
- [TP4 public-artifact reproduction](../performance/records/deepseek-v4-flash/sparkcache-tp4-public-reproduction-20260822.md)
  and [research performance](../performance/records/deepseek-v4-flash/sparkcache-tp4-performance-20260822.md).

## Baseline versus SparkCache comparison classes

The table below records the 2026-08-22 benchmark settings, not the normalized
candidate recipes. The repository accepts measured performance data as
`research-only` even when
the arms are not a controlled mechanism comparison. A SparkCache-only
attribution requires equality of the material performance settings defined in
[`performance/README.md`](../performance/README.md).

| Pair | Material settings held equal | Unequal or unproven material settings | Accepted use |
|---|---|---|---|
| DeepSeek TP2/DCP1 | serving image, hardware, topology, TP2/DCP1, key-value dtype, DSpark K5, prompt shapes | Base checkpoint identity was not recorded; model limit 1,048,576 vs 131,072; maximum sequences 32 vs 6; scheduler budget 8,192 vs 4,096; key-value reservation 12 vs 16 GiB per rank; base effective block geometry unrecorded vs 256 explicit. GPU utilization 0.70 vs 0.78 is non-operative because both arms set explicit key-value bytes. | Recipe-level research data. It demonstrates each configuration's observed behavior but cannot attribute a difference only to SparkCache. |
| DeepSeek TP4/DCP1 | checkpoint digest `bd6b01…`, serving image, hardware, topology, TP4/DCP1, maximum sequences 32, key-value dtype, DSpark K5, prompt shapes | Model limit 1,048,576 vs 524,288; scheduler budget 8,192 vs 4,096; key-value reservation 16 vs 32 GiB per rank; base effective block geometry unrecorded vs 256 explicit. | Recipe-level research data. |
| GLM TP4/DCP4 | checkpoint, operator image, hardware, topology, TP4/DCP4, model limit 262,144, maximum sequences 8, scheduler budget 4,096, 9.25 GB key-value reservation, block geometry 64, fixed MTP4, exact-Q40/SIRCL contract, prompt shapes | No unrelated performance knob changed. SparkCache wheel, connector, required overlays, mounts, and selected cache state comprise the treatment. | Mechanism-level research comparison for the exact operator artifact. The one ordered sample per shape does not establish a precise regression percentage or statistical confidence. |

The GLM image's maintainer-local availability limits public reproduction and
generalization; it does not invalidate measurements made with the same exact
image in both arms. The DeepSeek TP2 persistence cycle validates the launcher
and cache path on a layout-compatible research checkpoint, but it does not
replace the qualified checkpoint receipt.

## Interpretation

Do not compare these values across model identities or use them as a generic
hardware benchmark. The GLM result is acceptance evidence for one appliance;
the DeepSeek observations establish functional serving only.
