# DeepSeek-0731 TP2 bounded serving check

Status: **qualified for startup and one short text request only**. The ARM64
shared image with Docker configuration ID
`sha256:22da81cae0572ae2985a5c34a125db4f3cc58e871fa7e6256dffe59828f1ae5d`
started DeepSeek-V4-Flash-0731 on two NVIDIA GB10 nodes and returned the correct
answer `42` to a multiplication prompt. The response ended normally after two
completion tokens. [Machine-readable evidence](shared-22da81ca-tp2-smoke-20260918.json)
records checkpoint metadata and receipt hashes.

| Setting | Tested value |
|---|---|
| Runtime / parallelism | vLLM; TP2/DCP1 |
| Context limit / sequences / batch | 131,072 / 8 / 4,096 |
| KV allocation | 8 GiB per node; reported 290,555-token pool |
| Checkpoint loader | B12X |
| Attention / communication | FlashInfer SM120 sparse MLA / NCCL 2.31.2 |
| Speculation | DSpark, five tokens; B12X draft MoE backend |
| Compiler workers | Two per preparation stage |
| SparkCache | Disabled |
| Test request | C1; 18 prompt tokens; temperature 0; thinking disabled |

The test reused existing checkpoint mounts. Configuration and index hashes,
48 shards and their total size matched on both ranks; it did not recompute
checksums over every weight tensor. No weights were downloaded or converted.
Only the test's containers were stopped afterward; saved deployments remained
available.

This is not qualification of the [public pair profile](../../../profiles/deepseek-v4-flash-0731-pair/README.md)
with 1M context, 32 sequences and 16 GiB KV per node. No performance,
full-context, concurrent-load, media, restart, SparkCache or soak claim follows
from the short request. Existing quickstart image and serving defaults are
unchanged.
