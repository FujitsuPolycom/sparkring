# DeepSeek-V4-Flash-0731 profile

## Status

**Status: implemented and live-benchmarked; not qualified.**
DeepSeek-V4-Flash loads and serves as either two or four tensor-parallel ranks
on directly cabled DGX Sparks. The TP2 benchmark used the DSpark package at
`913f0657…`; TP4 used the plain 0731 package at `7872f01…`. Use the
[DeepSeek quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md) to deploy it.

## Serving contract

| Setting | Value |
|---|---|
| Runtime image | `ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Parallelism | TP2 across a directly cabled pair or TP4 across a four-Spark cycle |
| TP2 checkpoint | `deepseek-ai/DeepSeek-V4-Flash-DSpark@913f0657a874f76844e2e91cbe706dbcaceeb6d7` |
| TP4 checkpoint | `deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Activations | `bfloat16` |
| Request limit | 1,048,576 tokens |
| Maximum sequences | 32 |
| Key-value reservation | 17,179,869,184 bytes per rank in both base targets; SparkCache TP4 retains 34,359,738,368 bytes per rank |
| Scheduler budget | 4,096 tokens |
| Block size | 256 tokens |
| Key-value dtype | `fp8_ds_mla` |
| Speculation | DSpark, five speculative tokens, B12X MoE backend |
| API model name | `deepseek-v4-flash-0731` |

`fp8_ds_mla` is required. It declares the model's MLA key-value geometry and
matches the launch command; do not replace it with generic `fp8`.

The model uses FP8 e4m3 weights with 128-by-128 blocks, FP4 experts, hidden
width 4096, and 48 safetensors shards totaling about 167 GB. The two packages
have identical configuration and index files but different tensor payloads.
Every rank within one deployment must use the same package and revision.

## Evidence boundary

The implemented pair and cycle launches exercised API health, chat
completions, tool calling, and DSpark speculative decoding. Both topologies
completed prefill and sustained-decode matrices through 128K,
with C1/C2 measured at least five times and every other applicable cell at
least three times. The [TP2 record](../../performance/records/deepseek-v4-flash/normalized-tp2-base-temp1-n5-20260823.md)
and [TP4 record](../../performance/records/deepseek-v4-flash/normalized-tp4-base-temp1-n5-20260823.md)
retain the conditions, variability, and source-receipt hashes. These results
remain evidence for the recorded image/checkpoint objects; they do not qualify
an unrecorded model revision or locally built image.

The native transport's width-4096 graph mode is research-only and is not part
of this profile's qualification. One hash-bound candidate using the pinned
serving image completed target and DSpark CUDA-graph capture and API smoke,
with native replay advancing equally on every rank and zero overflow. The
matched comparison is retained in
[`sircl-width4096-nccl-ab-20260822.md`](../../performance/records/deepseek-v4-flash/sircl-width4096-nccl-ab-20260822.md).
The normal launch continues to use the NCCL configuration in
`scripts/config/deepseek-v4-flash-0731.env.example`.
