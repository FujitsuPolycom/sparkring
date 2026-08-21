# GLM-5.2 EXL3 3.5-bpw fixed-MTP4 profile

## Status and scope

This profile is **qualified** on one four-DGX-Spark appliance. It is not a
claim about any rebuilt image. A build from the tracked inputs is
**implemented** with offline test evidence until the image identity that it produces passes
[the promotion checklist](GLM52_35BPW_PROMOTION_CHECKLIST.md).

The profile is defined by
[`recipes/glm52-exl3-r7-3.5bpw.json`](../recipes/glm52-exl3-r7-3.5bpw.json).
The checkpoint is
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`.

## Serving contract

| Setting | Qualified value |
|---|---|
| Parallelism | TP4 plus DCP4 `ag_rs` |
| Speculation | fixed MTP4, greedy draft sampling |
| Request limit | 262,144 tokens |
| Maximum sequences | 8 |
| Key-value cache | `nvfp4_ds_mla`, dynamic per-token scale, FP8 RoPE |
| Key-value allocation | 9,250,000,000 bytes per rank |
| Graph capture | `FULL_AND_PIECEWISE`, query rows Q1 through Q40 |
| TP transport | SIRCL with patched NCCL fallback |
| DCP and indexer | stock collectives |
| Online quantization | EXL3 K6, target-only scope |
| Prefix caching | native prefix caching enabled |

The target-only exact-Q40 policy applies only to exactly 40 query rows. It uses
capacity 40 and route block 8; other row counts retain the profile's normal
routing behavior.

## Measured results

The qualified appliance measured the following unique-context results. Conditions
and evidence scope are recorded in [results](RESULTS.md).

| Context | 4K | 8K | 16K | 32K | 64K | 128K |
|---|---:|---:|---:|---:|---:|---:|
| Prefill, tokens/s | — | 679 | 673 | 666 | 657 | 645 |
| Decode, one stream | 22.6 | 22.0 | 21.3 | 20.4 | 21.4 | — |
| Decode, four streams | 50.3 | 51.9 | 49.2 | 45.6 | 47.2 | — |
| Decode, eight streams | 78.4 | 71.3 | 70.0 | 65.5 | 67.8 | — |

Coding prompts reached 27.3 tokens/s on one stream with measured draft
acceptance of 96.64%. These measurements apply only to the qualified appliance
and serving contract above.

## Limits

The profile's fixed-MTP4 configuration improved measured C1-C4 cells but
regressed the matched C8 cell by 11.63%. The TP native path covers qualified TP
all-reduce and vocabulary families; DCP and indexer collectives remain stock.
Use [the quickstart](GLM52_35BPW_QUICKSTART.md) for deployment and
[the reproduction procedure](GLM52_35BPW_REPRODUCTION.md) for the generated
layers.
