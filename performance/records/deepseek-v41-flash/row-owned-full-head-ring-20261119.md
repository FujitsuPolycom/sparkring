# DeepSeek V4.1 prefill on a four-Spark cycle with 8K chunks

**Research-only.** DeepSeek-V4.1-Flash-1017 exceeded 6,000 prompt tokens/s
at 16K, 64K, and 128K with full-head local-query attention, compact expert
routes, and narrow compressor and KV gathers. The engine used 8,192-token
prefill chunks.

## Conditions

Four DGX Sparks on one directly cabled NCCL cycle used TP4, DCP1, the
B12X installer image
`sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`,
and checkpoint `1bae0e1088993b9b82de08f59de5be5a34e42170`. Serving
used eager execution, disk Engram with a replicated projection, DSpark5
with adaptive verification disabled, a 262,144-token context cap, a
16-sequence scheduler limit, and patched NCCL 2.32.3 over RoCE.

The existing [indexer-scoring experiment](row-owned-indexer-ring-20261119.md)
supplies the 8K-chunk comparison. Each later row represents three
concurrency-one cold requests per length on the same cycle. The
loopback client sent exact token IDs, used temperature zero and one
completion token, and timed from request submission to first text delta.
Every measured request reported zero cached tokens.

## Results

| Configuration | 16K median tok/s | 64K median tok/s | 128K median tok/s |
|---|---:|---:|---:|
| Distributed indexer scoring | 5,445 | 5,361 | 5,234 |
| Add compact expert routing | 5,550 | 5,466 | 5,360 |
| Add full-head attention and narrow KV gathers | 6,226 | 6,121 | 5,990 |
| Build metadata for owned query rows | 6,267 | 6,179 | 6,011 |
| Pack the compressor and indexer projections | 6,334 | 6,250 | 6,124 |
| Add Engram feature bands | 6,332 | 6,197 | 6,095 |

All nine compressor-variant samples cleared 6,000 tok/s. The 128K
samples ranged from 6,113 to 6,145 tok/s. [Samples](row-owned-full-head-ring-20261119.jsonl)

The Engram banding experiment split each replicated-projection lookup
reduction into feature bands of at most 80 MiB. The counters recorded four
Engram reductions per chunk. The unbanded compressor build had the highest
median at each length and is the selected installer configuration.

## Serving checks

At 8,192 rows, the compressor variant logged 20 local-attention layers,
17 narrow KV gathers and three packed compressor gathers. It gathered
compact expert routes at 20 layers and reduced 22 tensors, including
two Engram lookups. The fixed 16K and 128K prompts returned the
baseline's first token. An 8,200-token continuation reused 7,936
cached tokens. An unaligned 16,387-token prefill and a concurrent
64-token decode both completed.

The full-head trial held 76.96 GiB of model weights on the API rank.
The compact-routing-only launch held 74.63 GiB. These measurements
cover one direct NCCL cycle. The installer was tested through plan
generation, and the measured deployment used manual containers.
Sustained decode, multimodal behavior, open-chain throughput and
full numerical parity across arbitrary prompts remain unqualified.
