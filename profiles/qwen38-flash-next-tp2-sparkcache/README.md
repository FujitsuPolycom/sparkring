# Qwen3.8-Flash-Next with SparkCache on two Sparks

Checkpoint: [`local-inference-lab/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4), Local Inference Lab's NVFP4 quantization-aware distillation of Qwen's [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next).

Status: **qualified for bounded correctness and restart checks**.
Use the [Qwen TP2 quickstart](../qwen38-flash-next-tp2/README.md).
Set `PROFILE_ID=qwen38-flash-next-tp2-sparkcache` before its image/checkpoint
block. The shared guide includes image pull, model verification, startup and
restart instructions for both cache choices.

Both variants use [shared-2026.09.3](../../runtime/releases/shared-2026.09.3/README.md).
The [configuration](../qwen38-flash-next-tp2/sparkcache.json) owns the flags;
the [qualification record](../../runtime/releases/shared-2026.09.3/qualification.json)
records bounded text/media correctness, concurrent-request
checks, and physical cache restore on both ranks after retained restarts.

Persistent caching uses aligned checkpoints, a fresh release-specific namespace,
4 GiB disk capacity per rank, two 512 MiB capture slots and a 256 MiB restore
budget per rank. These buffers are separate from the 24 GiB KV pin.

Use [Compose](../../docs/operations/compose.md) profile
`qwen38-flash-next-tp2-sparkcache` for coordinated deployments.
The API has no configured key. Restrict access to trusted clients or an
authenticated gateway. Request `cache_salt` does not isolate this image's disk
cache; use separate deployments/cache directories for tenant isolation.
