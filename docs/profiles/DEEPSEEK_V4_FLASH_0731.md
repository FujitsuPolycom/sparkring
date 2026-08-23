# DeepSeek-V4-Flash-0731 profile

## Status

**Status: implemented; not qualified.** `deepseek-ai/DeepSeek-V4-Flash-0731`
loads and serves as four tensor-parallel ranks on directly cabled DGX Sparks.
No model revision is pinned in this repository and no shadow comparison has
qualified its collective results. Use the
[DeepSeek quickstart](../DEEPSEEK_V4_FLASH_QUICKSTART.md) to deploy it.

## Serving contract

| Setting | Value |
|---|---|
| Runtime image | `ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Parallelism | TP4 across four DGX Sparks |
| Activations | `bfloat16` |
| Request limit | 1,048,576 tokens |
| Key-value reservation | 17,179,869,184 bytes per rank in both base targets; SparkCache TP4 retains 34,359,738,368 bytes per rank |
| Scheduler budget | 4,096 tokens |
| Block size | 256 tokens |
| Key-value dtype | `fp8_ds_mla` |
| Speculation | DSpark, five speculative tokens, B12X MoE backend |
| API model name | `deepseek-v4-flash-0731` |

`fp8_ds_mla` is required. It declares the model's MLA key-value geometry and
matches the launch command; do not replace it with generic `fp8`.

The model uses FP8 e4m3 weights with 128-by-128 blocks, FP4 experts, hidden
width 4096, and 48 safetensors shards totaling about 167 GB. Download identical
model bytes to every rank and record any revision and hashes used for an
operator deployment.

## Evidence boundary

The implemented launch exercised API health, chat completions, tool calling, and
DSpark speculative decoding on the four-Spark cycle. The published image pin
is immutable, but a four-rank deployment with all ranks pulled from that digest
has not been recorded. Any deployment using a locally built image is evidence
for that image identity, not the published digest.

The native transport's width-4096 graph mode is research-only and is not part
of this profile's qualification. The normal launch uses the NCCL configuration
in `scripts/config/deepseek-v4-flash-0731.env.example`.
