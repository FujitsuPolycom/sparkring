# Qwen3.8-Flash-Next with SparkCache on two Sparks

Status: **qualified for bounded correctness and restart checks**.
Use the [Qwen TP2 quickstart](../qwen38-flash-next-tp2/README.md).
It selects SparkCache by default and includes the cache-disabled alternative,
image pull, model verification, startup and restart instructions.

Both variants use [shared-2026.09.2](../../runtime/releases/shared-2026.09.2/README.md).
The [configuration](../qwen38-flash-next-tp2/sparkcache.json) owns the flags;
the [qualification record](../../runtime/releases/shared-2026.09.2/qualification.json)
records bounded text/media correctness, request-order and concurrent-request
checks, and physical cache restore on both ranks after retained restarts.

Persistent caching uses aligned checkpoints, a fresh release-specific namespace,
4 GiB disk capacity per rank, two 512 MiB capture slots and a 256 MiB restore
budget per rank. These buffers are separate from the 24 GiB KV pin.

Use [Compose](../../docs/operations/compose.md) profile
`qwen38-flash-next-tp2-sparkcache` for coordinated deployments.
The API has no configured key. Restrict access to trusted clients or an
authenticated gateway. Request `cache_salt` does not isolate this image's disk
cache; use separate deployments/cache directories for tenant isolation.
