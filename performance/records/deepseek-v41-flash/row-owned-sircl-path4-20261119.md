# DeepSeek V4.1 row-owned prefill on a four-Spark SIRCL path

**Research-only.** Ring 47 served DeepSeek-V4.1-Flash-1017 with native
SIRCL, 8,192-token row-owned prefill, adaptive DSpark5, and decode-only
CUDA graphs. All four ranks reported `nccl=none`, `pynccl=skipped`, and
no NCCL decision rows for TP and EP. The launch logs showed no NCCL
communicator initialization.

## Conditions

Four DGX Sparks used TP4, DCP1, the existing B12X installer image
`sha256:3cc9a81fd344fe52fe9cc261b46dc051e3c3f494e51b8142846a5f854835c502`,
and checkpoint `1bae0e1088993b9b82de08f59de5be5a34e42170`. The trial
used SIRCL `path:4` with `SIRCL_NCCL=never`, two lanes per peer and the
pinned `path-4` settings for large all-reduce, all-gather and
reduce-scatter. TP and EP shared the native session. A private fabric
document described the three links in use. The shared fabric records
and host routes stayed in place.

The launch kept the full-head query weights, compact expert routes,
narrow KV gathers, packed compressor projections, and replicated disk
Engram projection. `FULL_DECODE_ONLY` used compilation mode `NONE` and
capture sizes through 96. The context limit was 262,144 tokens.

## Throughput

| Measurement | Samples in run order (tok/s) | Median tok/s |
|---|---|---:|
| 16,384-token cold prefill | 6,016.09, 6,019.82, 6,017.88 | 6,017.88 |
| 65,536-token cold prefill | 5,916.06, 5,916.96, 5,920.16 | 5,916.96 |
| 131,072-token cold prefill | 5,796.17, 5,786.05, 5,781.23 | 5,786.05 |
| Short greedy decode | 54.05, 84.91, 77.60 | 77.60 |

The concurrency-one loopback client sent exact token IDs for prefill and
measured from request submission to the first text delta. Every request
reported zero cached prompt tokens and returned one output token. Decode
used three 384-token runs of the same short prompt at temperature zero,
measured after the first token. DSpark accepted 775 of 1,603 drafted
tokens over those runs. [Sample data](row-owned-sircl-path4-20261119.jsonl)

The 16K samples cleared 6,000 tok/s. The 64K and 128K samples remained
below that mark. The earlier [cycle trial](row-owned-full-head-ring-20261119.md#decode-only-graph-trial)
used a different transport and topology.

## Decode workload probe

A separate probe ran each workload twice with 384 output tokens. The
median rates distinguish decode after the first token from request
throughput including prefill. The sampled runs used temperature one and
fixed seeds. [Sample data](row-owned-sircl-path4-20261119.jsonl)

| Prompt and temperature | Decode tok/s | End-to-end tok/s |
|---|---:|---:|
| Short, 0 | 99.37 | 94.47 |
| Short, 1 | 75.50 | 72.73 |
| 8K, 0 | 87.38 | 66.39 |
| 8K, 1 | 68.03 | 54.86 |
| 64K, 1 | 72.95 | 23.42 |

A single cold 8K request at temperature one also completed 2,048 output
tokens at 82.67 decode tok/s and 78.17 end-to-end tok/s.

## Serving and installer checks

All four ranks finished graph capture. An 8,192-token cold prefill
completed. Its 8,200-token continuation reused 7,936 cached tokens and
generated eight tokens. A 16,387-token unaligned prefill completed during
a concurrent 64-token decode. The row-owned counters showed 20 local
attention layers, 17 narrow KV gathers and three packed compressor
gathers at 8,192 rows. Two fixed prompts at 16K and 128K returned the
same first tokens as the earlier cycle run.

Hardware validation used manually staged containers. The installer
renders the same serving options with `--transport sircl --nccl never`
in its offline path-4 tests. A live installer rollout and a SIRCL
cycle-4 run remain unverified.
