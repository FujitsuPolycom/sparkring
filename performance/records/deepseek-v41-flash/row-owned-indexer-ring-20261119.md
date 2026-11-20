# Distributed indexer scoring for DeepSeek V4.1 prefill

**Research-only.** Splitting the encoder's indexer scoring across four DGX Sparks
raised median cold prefill from 4,525 to 5,234 tokens/s at 128K. Increasing
the chunk size to 16,384 tokens raised the median to 5,463 tokens/s.

## Conditions

The four Sparks used a directly cabled cycle and SparkRing's patched NCCL 2.32.3
over RoCE. The B12X installer image `sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`
served DeepSeek-V4.1-Flash-1017 at revision `1bae0e1088993b9b82de08f59de5be5a34e42170`.
Each launch used TP4, DCP1, eager execution, 16 sequences,
a 262,144-token context limit, 0.83 GPU memory utilization, and DSpark5 with
adaptive verification off. Both retained owned residual rows and the replicated
Engram projection from the [original experiment](row-owned-prefill-ring-20261119.md).
The baseline and first scorer variant used 8,192-token batches. The second
scorer variant used 16,384-token batches. The scorer variants exposed the
PyTorch profiler while their timed requests ran with profiling idle.

## Measurement

A local loopback client sent exact token IDs for synthetic prompts to the
streaming completion endpoint. Each variant ran three requests at each length,
at concurrency one and temperature zero, with one output token. The engine
reported zero cached tokens for every timed request. Rates divide the server's
prompt token count by client time to the first nonempty text delta.
See the [raw observations](row-owned-indexer-ring-20261119.jsonl) for the
sample times, token counts, and plugin digests.

A PyTorch trace on each rank captured one late 8,192-token chunk during a
separate baseline request at 128K. Name-based kernel attribution placed
344–352 ms in indexer kernels per rank. The scorer assigns whole native
scoring tiles to the owning ranks, then gathers the selected top-k rows.
All four ranks logged three indexer gathers for each observed owned chunk.

## Result

| Prompt | Row-owned baseline, 8K chunks | Distributed scoring, 8K chunks | Distributed scoring, 16K chunks |
|---|---:|---:|---:|
| 16K | 5,359 (5,308–5,376) | 5,445 (5,350–5,469) | 5,651 (5,645–5,687) |
| 64K | 5,059 (4,854–5,109) | 5,361 (5,334–5,378) | 5,576 (5,569–5,590) |
| 128K | 4,525 (4,465–4,677) | 5,234 (5,162–5,246) | 5,463 (5,448–5,481) |

Values are median prompt tokens/s with the observed range in parentheses.

The fixed 16K and 128K prompts produced the same first text token in the
profile-enabled baseline and both scorer launches. Offline checks reproduced the
full top-k table on all four simulated ranks and validated the edited method
against the pinned image source.

A single 16K-chunk screen with eight NCCL channels returned 5,622, 5,522,
and 5,350 tokens/s at the three prompt lengths. The three-repeat comparisons
use four channels.

An experimental variant projected attention inputs before gathering on
17 encoder layers, reducing the gathered width from 5,120 to 1,792 values
per row. Its 16K-chunk medians were 5,585, 5,485, and 5,388 tokens/s.
Both fixed-prompt first tokens matched the baseline.

With distributed scoring and 32K chunks, the medians were 5,692, 5,201, and
5,098 tokens/s. The engine reported a KV pool of 967,186 tokens, compared
with 3,544,968 for 16K chunks at the same 262,144-token context limit.

With 16K chunks and an 8 MiB NCCL buffer, the medians were 5,497, 5,413,
and 5,289 tokens/s. The faster 16K configuration uses the default NCCL
buffer size. [NCCL buffer setting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-buffsize)

After restarting the 16K-chunk scorer, an 8,200-token continuation reused
7,936 cached tokens. An unaligned 16,387-token prefill and a 64-token decode
both completed while submitted concurrently.

A separate PyTorch capture recorded one 16K forward with distributed
scoring. Per-rank GPU kernel time attributed by name ranged from 784 to
795 ms for MoE, 599 to 612 ms for other GEMM kernels, 508 to 589 ms for
NCCL, and 543 to 553 ms for attention. These are kernel-time sums for
one captured forward. The pinned attention path still prepares the
query and output projection plans for tensor-parallel heads and groups.
Attention on each rank's own query rows requires a separate full-head
prefill plan and complete local KV writes.

## Limits

This experiment covers one cycle and one serving image. The 16K ranges
overlap. The 6,000 tokens/s goal, sustained decode, and end-to-end installer
validation remain open.
