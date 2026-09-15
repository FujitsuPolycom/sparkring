# GLM-5.2 EXL3 3.5-bpw fixed-MTP4 profile

## Status and scope

The tested profile uses a 1,048,576-token request limit and 16 sequences on
four directly cabled DGX Sparks.

The profile is defined by
[`recipes/glm52-exl3-r7-3.5bpw.json`](../recipes/glm52-exl3-r7-3.5bpw.json).
The checkpoint is
`brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78@9ab9579774cc432df91567a36f6e9e863e0d4c9f`.

## Serving contract

| Setting | Value |
|---|---|
| Parallelism | TP4 plus DCP4 `ag_rs` |
| Speculation | fixed MTP4, greedy draft sampling |
| Request limit | 1,048,576 tokens |
| Maximum sequences | 16 |
| Key-value cache | `nvfp4_ds_mla`, dynamic per-token scale, FP8 RoPE |
| Key-value allocation | 9,250,000,000 bytes per rank |
| Key-value block size | 64 tokens |
| Graph capture | `FULL_AND_PIECEWISE`, query rows Q1 through Q40 |
| TP transport | SIRCL with patched NCCL fallback |
| DCP and indexer | stock collectives |
| Online quantization | EXL3 K6, target-only scope |
| Prefix caching | native prefix caching enabled |

The target-only exact-Q40 policy applies only to exactly 40 query rows. It uses
capacity 40 and route block 8; other row counts retain the profile's normal
routing behavior.

## Measured results

The [benchmark record](../performance/records/glm-3.5bpw/normalized-base-20260822.md)
contains prefill and C1/C2/C4/C8 decode through 128K, with the exact settings,
sample counts, variability limits, and receipts. Coding Peak averaged 25.39
tokens/s over five requests. These results do not qualify an untested image
or establish quality at the configured 1,048,576-token limit.

## Limits

The TP native path covers tested TP all-reduce and vocabulary families; DCP and
indexer collectives remain stock.
Use [the quickstart](../profiles/glm52-exl3-r7-3.5bpw/README.md) for deployment and
[the reproduction procedure](GLM52_35BPW_REPRODUCTION.md) for the generated
layers.
